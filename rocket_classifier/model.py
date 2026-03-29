"""XGBoost classifier for rocket trajectory classification.

Metric:
    min over classes j of (recall_j)
    = min_j [ #{correctly predicted as j AND true label j} / #{true label j} ]

    This is the minimum per-class recall. The system must not be bad at any
    single rocket class, so worst-case recall across all classes is the
    binding constraint.

Strategy:
    - GroupKFold on traj_ind prevents data leakage between train and validation.
    - NaN imputation is performed per-fold using only training-fold medians,
      preventing information leakage through imputation statistics.
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

# XGBoost defaults tuned for this dataset.
# Kept as a module-level constant so interpret.py and main.py share the same config.
DEFAULT_XGB_PARAMS: dict = {
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
    # use_label_encoder removed — deprecated in XGBoost 2.0
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": -1,
}


def _get_feature_cols(X: pd.DataFrame) -> list[str]:
    """Return feature column names, filtering out 'label' if present."""
    return [c for c in X.columns if c != "label"]


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


def _impute_nan(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Replace NaN values with the given per-column medians (in-place)."""
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_indices = np.where(nan_mask.any(axis=0))[0]
        for col in col_indices:
            X[nan_mask[:, col], col] = medians[col]
    return X


def train_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> tuple[XGBClassifier, list[float], np.ndarray]:
    """Train XGBoost with GroupKFold cross-validation, logging per-fold results.

    Data leakage prevention:
        1. ``GroupKFold`` on ``traj_ind`` ensures that all radar points
           belonging to a single trajectory appear exclusively in either
           the training set or the validation set within each fold.
        2. NaN imputation uses only training-fold medians per fold,
           preventing information from the validation fold from leaking
           into the imputed training features.

    After cross-validation, the model is retrained on the full dataset.

    Args:
        X: Per-trajectory feature DataFrame (may contain NaN). Rows
            correspond to trajectories; columns are physics features.
            A ``"label"`` column is filtered out if present.
        y: Integer class label array of shape (n_trajectories,).
        groups: Group identifier array of shape (n_trajectories,)
            containing ``traj_ind`` values.
        n_splits: Number of cross-validation folds. Defaults to 5.
        xgb_params: Optional dictionary of XGBoost hyperparameters that
            overrides the defaults. Partial overrides are supported.

    Returns:
        A tuple of:
            - model: ``XGBClassifier`` retrained on the full dataset.
            - fold_scores: List of per-fold min-recall scores.
            - train_medians: Per-column median array used for NaN
              imputation on the full training set (needed at inference).
    """
    feature_cols = _get_feature_cols(X)
    X_raw = X[feature_cols].to_numpy(dtype=np.float32)

    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups), 1):
        X_tr, X_val = X_raw[train_idx].copy(), X_raw[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]

        # Per-fold imputation: medians from training fold only (no leakage)
        fold_medians = np.nanmedian(X_tr, axis=0)
        _impute_nan(X_tr, fold_medians)
        _impute_nan(X_val, fold_medians)

        sample_weights = _compute_sample_weights(y_tr)

        model = XGBClassifier(**params)
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

    # Retrain on full data with full-data medians
    logger.info("Retraining on full dataset...")
    train_medians = np.nanmedian(X_raw, axis=0)
    X_full = X_raw.copy()
    _impute_nan(X_full, train_medians)

    sample_weights_full = _compute_sample_weights(y)
    final_model = XGBClassifier(**params)
    final_model.fit(X_full, y, sample_weight=sample_weights_full, verbose=False)

    return final_model, fold_scores, train_medians


def predict(model: XGBClassifier, X: pd.DataFrame, train_medians: np.ndarray) -> np.ndarray:
    """Generate class predictions for the given feature matrix.

    Args:
        model: Fitted ``XGBClassifier`` instance returned by ``train_with_cv``.
        X: Per-trajectory feature DataFrame with the same columns used
            during training. A ``"label"`` column is ignored if present.
        train_medians: Per-column median array from training data, used
            to fill NaN values consistently with training.

    Returns:
        Integer array of shape (n_trajectories,) with predicted class labels
        in {0, 1, 2}.
    """
    feature_cols = _get_feature_cols(X)
    X_vals = X[feature_cols].to_numpy(dtype=np.float32)
    _impute_nan(X_vals, train_medians)
    return model.predict(X_vals).astype(int)
