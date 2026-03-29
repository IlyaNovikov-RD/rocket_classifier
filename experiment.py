"""Experiment v4: softprob+thresholds AND two-stage classifier, single CV run.

Evaluates 3 approaches in one pass:
  A. Baseline (current softmax)
  B. softprob + global OOB threshold tuning
  C. Two-stage: {0,1} vs {2}, then specialized 0 vs 1 classifier
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from rocket_classifier.main import load_data, get_features, FEATURE_CACHE_TRAIN
from rocket_classifier.model import (
    DEFAULT_XGB_PARAMS,
    _compute_sample_weights,
    _impute_nan,
    min_class_recall,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOFTPROB = {**DEFAULT_XGB_PARAMS, "objective": "multi:softprob"}

STAGE1_PARAMS = {
    "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "device": "cpu", "random_state": 42, "n_jobs": -1,
}

STAGE2_PARAMS = {
    "n_estimators": 800, "max_depth": 8, "learning_rate": 0.03,
    "subsample": 0.85, "colsample_bytree": 0.7, "min_child_weight": 3,
    "gamma": 0.05, "reg_alpha": 0.05, "reg_lambda": 1.0,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "device": "cpu", "random_state": 42, "n_jobs": -1,
}


def optimize_biases(y_true, proba):
    lp = np.log(proba + 1e-12)
    best_s, best_b = -1.0, np.zeros(3)
    for b1 in np.linspace(-4, 4, 100):
        for b2 in np.linspace(-4, 4, 100):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s: best_s, best_b = s, b.copy()
    for b1 in np.linspace(best_b[1]-0.1, best_b[1]+0.1, 60):
        for b2 in np.linspace(best_b[2]-0.1, best_b[2]+0.1, 60):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s: best_s, best_b = s, b.copy()
    return best_b, best_s


def two_stage_predict(X_tr, y_tr, X_val, y_val):
    """Two-stage classifier: {0,1} vs {2}, then 0 vs 1."""
    # Stage 1: is it class 2?
    y1_tr = (y_tr == 2).astype(int)
    y1_val = (y_val == 2).astype(int)
    sw1 = _compute_sample_weights(y1_tr)
    m1 = XGBClassifier(**STAGE1_PARAMS)
    m1.fit(X_tr, y1_tr, sample_weight=sw1, verbose=False)
    p1_val = m1.predict_proba(X_val)[:, 1]  # prob of being class 2

    # Stage 2: among non-class-2, is it 0 or 1?
    mask_01_tr = y_tr != 2
    X2_tr = X_tr[mask_01_tr]
    y2_tr = (y_tr[mask_01_tr] == 1).astype(int)  # 0->0, 1->1
    sw2 = _compute_sample_weights(y2_tr)
    m2 = XGBClassifier(**STAGE2_PARAMS)
    m2.fit(X2_tr, y2_tr, sample_weight=sw2, verbose=False)
    p2_val = m2.predict_proba(X_val)[:, 1]  # prob of being class 1 (vs 0)

    # Combine: build 3-class probability
    # P(class2) = p1, P(class1) = (1-p1)*p2, P(class0) = (1-p1)*(1-p2)
    proba = np.zeros((len(X_val), 3))
    proba[:, 2] = p1_val
    proba[:, 1] = (1 - p1_val) * p2_val
    proba[:, 0] = (1 - p1_val) * (1 - p2_val)

    return proba, m1, m2


def main():
    t0 = time.time()
    train_raw, _, _ = load_data()
    train_feats = get_features(train_raw, FEATURE_CACHE_TRAIN, "train")
    feat_cols = [c for c in train_feats.columns if c != "label"]
    X = train_feats[feat_cols].to_numpy(dtype=np.float32)
    y = train_feats["label"].to_numpy(dtype=int)
    groups = np.array(train_feats.index.tolist())
    logger.info("Data: %d x %d  Labels: 0=%d 1=%d 2=%d",
                *X.shape, (y==0).sum(), (y==1).sum(), (y==2).sum())

    gkf = GroupKFold(n_splits=5)

    # Collect OOB probabilities for all 3 approaches
    oob_baseline = np.zeros(len(y), dtype=int)
    oob_softprob = np.zeros((len(y), 3))
    oob_twostage = np.zeros((len(y), 3))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X[tr_idx].copy(), X[val_idx].copy()
        y_tr, y_val = y[tr_idx], y[val_idx]
        med = np.nanmedian(X_tr, axis=0)
        _impute_nan(X_tr, med); _impute_nan(X_val, med)
        sw = _compute_sample_weights(y_tr)

        # A: Baseline (softmax)
        m_base = XGBClassifier(**DEFAULT_XGB_PARAMS)
        m_base.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        pred_base = m_base.predict(X_val)
        oob_baseline[val_idx] = pred_base
        score_a = min_class_recall(y_val, pred_base)

        # B: Softprob
        m_soft = XGBClassifier(**SOFTPROB)
        m_soft.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        prob_soft = m_soft.predict_proba(X_val)
        oob_softprob[val_idx] = prob_soft
        score_b_raw = min_class_recall(y_val, np.argmax(prob_soft, axis=1))
        _, score_b_oracle = optimize_biases(y_val, prob_soft)

        # C: Two-stage
        prob_ts, _, _ = two_stage_predict(X_tr, y_tr, X_val, y_val)
        oob_twostage[val_idx] = prob_ts
        pred_ts = np.argmax(prob_ts, axis=1)
        score_c_raw = min_class_recall(y_val, pred_ts)
        _, score_c_oracle = optimize_biases(y_val, prob_ts)

        n_err_a = (pred_base != y_val).sum()
        n_err_b = (np.argmax(prob_soft, axis=1) != y_val).sum()
        n_err_c = (pred_ts != y_val).sum()

        logger.info("Fold %d | A(baseline)=%.4f (%d err) | B(softprob)=%.4f raw, %.4f oracle (%d err) | C(2stage)=%.4f raw, %.4f oracle (%d err)",
                    fold, score_a, n_err_a, score_b_raw, score_b_oracle, n_err_b,
                    score_c_raw, score_c_oracle, n_err_c)

    # Global OOB threshold optimization
    logger.info("\n--- Global OOB threshold optimization ---")

    # Baseline
    score_a_global = min_class_recall(y, oob_baseline)
    logger.info("A (baseline):  global min-recall = %.4f", score_a_global)

    # Softprob + biases
    biases_b, _ = optimize_biases(y, oob_softprob)
    # Per-fold with global biases
    b_fold = []
    for _, (_, val_idx) in enumerate(gkf.split(X, y, groups)):
        p = np.argmax(np.log(oob_softprob[val_idx] + 1e-12) + biases_b, axis=1)
        b_fold.append(min_class_recall(y[val_idx], p))
    logger.info("B (softprob+bias): mean=%.4f  min=%.4f  per-fold=%s  biases=%s",
                np.mean(b_fold), np.min(b_fold),
                [f"{s:.4f}" for s in b_fold],
                [f"{b:.3f}" for b in biases_b])

    # Two-stage + biases
    biases_c, _ = optimize_biases(y, oob_twostage)
    c_fold = []
    for _, (_, val_idx) in enumerate(gkf.split(X, y, groups)):
        p = np.argmax(np.log(oob_twostage[val_idx] + 1e-12) + biases_c, axis=1)
        c_fold.append(min_class_recall(y[val_idx], p))
    logger.info("C (2stage+bias):   mean=%.4f  min=%.4f  per-fold=%s  biases=%s",
                np.mean(c_fold), np.min(c_fold),
                [f"{s:.4f}" for s in c_fold],
                [f"{b:.3f}" for b in biases_c])

    # Combined: average softprob and two-stage probabilities
    oob_combined = (oob_softprob + oob_twostage) / 2
    biases_d, _ = optimize_biases(y, oob_combined)
    d_fold = []
    for _, (_, val_idx) in enumerate(gkf.split(X, y, groups)):
        p = np.argmax(np.log(oob_combined[val_idx] + 1e-12) + biases_d, axis=1)
        d_fold.append(min_class_recall(y[val_idx], p))
    logger.info("D (combined+bias): mean=%.4f  min=%.4f  per-fold=%s  biases=%s",
                np.mean(d_fold), np.min(d_fold),
                [f"{s:.4f}" for s in d_fold],
                [f"{b:.3f}" for b in biases_d])

    logger.info("\n--- SUMMARY ---")
    logger.info("A (baseline):      %.4f mean", score_a_global)
    logger.info("B (softprob+bias): %.4f mean, %.4f min fold", np.mean(b_fold), np.min(b_fold))
    logger.info("C (2stage+bias):   %.4f mean, %.4f min fold", np.mean(c_fold), np.min(c_fold))
    logger.info("D (combined+bias): %.4f mean, %.4f min fold", np.mean(d_fold), np.min(d_fold))
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
