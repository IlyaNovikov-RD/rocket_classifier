"""
XGBoost classifier for rocket trajectory classification.

Metric: min over classes j of (recall_j)
  = min_j [ #{correctly predicted as j AND true label j} / #{true label j} ]
This is the minimum per-class recall — we must not be bad at any single class.

Strategy:
- Use GroupKFold on traj_ind to prevent data leakage.
- Tune class_weight via scale_pos_weight / sample_weight to boost recall on
  minority classes (label 2 is rarest).
- Report per-class recall and the min-recall metric after CV.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute minimum per-class recall across all classes.

    This is the official evaluation metric from the assignment.

    Args:
        y_true: Ground-truth labels array.
        y_pred: Predicted labels array.

    Returns:
        Scalar in [0, 1]: minimum recall over all classes present in y_true.
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
    """Inverse-frequency sample weights to balance minority classes."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n = len(y)
    n_classes = len(classes)
    weights = np.array([n / (n_classes * freq[c]) for c in y], dtype=np.float32)
    return weights


def train_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    xgb_params: Optional[dict] = None,
) -> tuple[XGBClassifier, list[float]]:
    """Train XGBoost with GroupKFold CV, logging per-fold min-recall.

    Data leakage prevention: GroupKFold ensures all points of a trajectory
    are in either train or validation, never both.

    Args:
        X:          Feature matrix (per-trajectory).
        y:          Integer labels array.
        groups:     Group array (traj_ind) matching X rows.
        n_splits:   Number of CV folds.
        xgb_params: Optional XGBoost hyperparameters override.

    Returns:
        Tuple of (model trained on full data, list of per-fold min-recall scores).
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
            per_class.append(f"class{i}={cm[i, i]/denom:.3f}" if denom > 0 else f"class{i}=N/A")

        logger.info(
            "Fold %d/%d | min-recall=%.4f | %s",
            fold, n_splits, score, "  ".join(per_class),
        )
        fold_scores.append(score)

    logger.info(
        "CV min-recall: %.4f ± %.4f (mean ± std)",
        np.mean(fold_scores), np.std(fold_scores),
    )

    # Retrain on full data
    logger.info("Retraining on full dataset...")
    sample_weights_full = _compute_sample_weights(y)
    final_model = XGBClassifier(**default_params)
    final_model.fit(X_vals, y, sample_weight=sample_weights_full, verbose=False)

    return final_model, fold_scores


def predict(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions for the given feature matrix.

    Args:
        model: Trained XGBClassifier.
        X:     Feature DataFrame (same columns as training, without 'label').

    Returns:
        Integer array of predicted labels.
    """
    feature_cols = [c for c in X.columns if c != "label"]
    X_vals = X[feature_cols].to_numpy(dtype=np.float32)
    return model.predict(X_vals).astype(int)
