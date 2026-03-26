"""XGBoost classifier for rocket trajectory classification.

Metric:
    min over classes j of (recall_j)
    = min_j [ #{correctly predicted as j AND true label j} / #{true label j} ]

    This is the minimum per-class recall. The system must not be bad at any
    single rocket class, so worst-case recall across all classes is the
    binding constraint.

Strategy:
    - GroupKFold on traj_ind prevents data leakage between train and validation.
    - Inverse-frequency sample weights boost recall on minority classes
      (label 2 is the rarest at ~7% of trajectories).
    - Per-fold logging reports both the aggregate score and per-class breakdown.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the minimum per-class recall across all classes.

    This is the official evaluation metric for the assignment. It returns
    the recall of the worst-performing class, ensuring the model is not
    evaluated favourably when it fails on a minority class.

    Args:
        y_true: Ground-truth integer labels array of shape (N,).
        y_pred: Predicted integer labels array of shape (N,).

    Returns:
        A scalar in [0.0, 1.0] representing the minimum recall over all
        classes present in ``y_true``.
    """
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recall = float(np.sum((y_pred == cls) & mask) / mask.sum())
        recalls.append(recall)
    return float(np.min(recalls))


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights to balance minority classes.

    Each sample receives a weight inversely proportional to its class
    frequency: ``w_i = N / (K * N_j)`` where N is the total number of
    samples, K is the number of classes, and N_j is the count of class j.
    This is equivalent to upweighting rare classes without synthetic
    oversampling (SMOTE), preserving the original feature distribution.

    Args:
        y: Integer class label array of shape (N,).

    Returns:
        Float32 weight array of shape (N,) where samples from minority
        classes receive higher weights.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts, strict=True))
    n = len(y)
    n_classes = len(classes)
    weights = np.array([n / (n_classes * freq[c]) for c in y], dtype=np.float32)
    return weights


def train_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> tuple[XGBClassifier, list[float]]:
    """Train XGBoost with GroupKFold cross-validation, logging per-fold results.

    Data leakage prevention: ``GroupKFold`` on ``traj_ind`` ensures that all
    radar points belonging to a single trajectory appear exclusively in either
    the training set or the validation set within each fold — never both.
    After cross-validation, the model is retrained on the full dataset.

    Args:
        X: Per-trajectory feature DataFrame. Rows correspond to trajectories;
            columns are physics features. Must not contain a ``"label"``
            column (it is filtered out internally if present).
        y: Integer class label array of shape (n_trajectories,). Valid values
            are 0, 1, and 2.
        groups: Group identifier array of shape (n_trajectories,) containing
            ``traj_ind`` values. Passed directly to ``GroupKFold.split()``.
        n_splits: Number of cross-validation folds. Defaults to 5.
        xgb_params: Optional dictionary of XGBoost hyperparameters that
            overrides the defaults. Partial overrides are supported.

    Returns:
        A tuple of:
            - model: ``XGBClassifier`` retrained on the full dataset.
            - fold_scores: List of per-fold min-recall scores of length
              ``n_splits``.
    """
    feature_cols = [c for c in X.columns if c != "label"]
    X_vals = X[feature_cols].to_numpy(dtype=np.float32)

    default_params = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 42,
        "n_jobs": -1,
    }
    if xgb_params:
        default_params.update(xgb_params)

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_vals, y, groups), 1):
        X_tr, X_val = X_vals[train_idx], X_vals[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        sample_weights = _compute_sample_weights(y_tr)

        model = XGBClassifier(**default_params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        score = min_class_recall(y_val, y_pred)

        # Per-class recall breakdown
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
        per_class = []
        for i in range(3):
            denom = cm[i].sum()
            per_class.append(f"class{i}={cm[i, i] / denom:.3f}" if denom > 0 else f"class{i}=N/A")

        logger.info(
            "Fold %d/%d | min-recall=%.4f | %s",
            fold,
            n_splits,
            score,
            "  ".join(per_class),
        )
        fold_scores.append(score)

    logger.info(
        "CV min-recall: %.4f ± %.4f (mean ± std)",
        np.mean(fold_scores),
        np.std(fold_scores),
    )

    # Retrain on full data
    logger.info("Retraining on full dataset...")
    sample_weights_full = _compute_sample_weights(y)
    final_model = XGBClassifier(**default_params)
    final_model.fit(X_vals, y, sample_weight=sample_weights_full, verbose=False)

    return final_model, fold_scores


def predict(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Generate class predictions for the given feature matrix.

    Args:
        model: Fitted ``XGBClassifier`` instance returned by
            ``train_with_cv``.
        X: Per-trajectory feature DataFrame with the same columns used
            during training. A ``"label"`` column is ignored if present.

    Returns:
        Integer array of shape (n_trajectories,) with predicted class labels
        in {0, 1, 2}.
    """
    feature_cols = [c for c in X.columns if c != "label"]
    X_vals = X[feature_cols].to_numpy(dtype=np.float32)
    return model.predict(X_vals).astype(int)
