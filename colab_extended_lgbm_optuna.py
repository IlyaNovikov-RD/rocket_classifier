"""
Extended Features + LightGBM Optuna — Google Colab (H100 GPU)
==============================================================

Best attempt at pushing past 0.9988 min-recall.

Strategy:
  - 76 base features (from parquet cache) + 66 extended features computed
    from raw train.csv = 142 total features
  - Extended features target the 0↔1 boundary: phase segmentation
    (early/mid/late trajectory means), higher-order stats (skew, kurtosis),
    and curvature — all capture asymmetries in the thrust curve that global
    statistics average away
  - LightGBM with 100 Optuna trials on H100 GPU
  - Same 4-stage pipeline: feature selection → Optuna → threshold tuning
    → final model

Why this might work:
  The C3 experiment (extended features + XGBoost depth 8) hit oracle 0.9998
  on fold 3. LightGBM already outperformed XGBoost on the 61-feature set
  (0.9988 vs 0.9984). Combining both should push the practical ceiling.

Install dependencies (run first):
    !pip install -q optuna catboost lightgbm xgboost scikit-learn pandas numpy pyarrow

Upload to /content/ before running:
    - cache_train_features.parquet   (76 features + label, indexed by traj_ind)
    - train.csv                      (raw point-level data for extended features)

Usage:
    %run colab_extended_lgbm_optuna.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

_fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(_handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger("rocket-extended")

# ---------------------------------------------------------------------------
# Configuration — tuned for H100
# ---------------------------------------------------------------------------

PARQUET_PATH = Path("/content/cache_train_features.parquet")
RAW_CSV_PATH = Path("/content/train.csv")
MODEL_OUTPUT = Path("/content/extended_lgbm_model")
RESULTS_OUTPUT = Path("/content/extended_lgbm_results.json")

N_SPLITS = 10
N_TRIALS = 100          # double vs previous run
RANDOM_STATE = 42
N_CLASSES = 3

THRESHOLD_GRID_COARSE = 80
THRESHOLD_GRID_FINE = 50
THRESHOLD_GRID_ULTRA = 30


# ---------------------------------------------------------------------------
# Metric & Utilities
# ---------------------------------------------------------------------------

def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(np.sum((y_pred == cls) & mask) / mask.sum()))
    return float(np.min(recalls))


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts, strict=True))
    n, k = len(y), len(classes)
    return np.array([n / (k * freq[c]) for c in y], dtype=np.float32)


def impute_nan(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for col in np.where(nan_mask.any(axis=0))[0]:
            X[nan_mask[:, col], col] = medians[col]
    return X


def optimize_thresholds(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    lp = np.log(proba + 1e-12)
    best_s, best_b = -1.0, np.zeros(N_CLASSES)
    for b1 in np.linspace(-4, 4, THRESHOLD_GRID_COARSE):
        for b2 in np.linspace(-4, 4, THRESHOLD_GRID_COARSE):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    for b1 in np.linspace(best_b[1] - 0.12, best_b[1] + 0.12, THRESHOLD_GRID_FINE):
        for b2 in np.linspace(best_b[2] - 0.12, best_b[2] + 0.12, THRESHOLD_GRID_FINE):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    for b1 in np.linspace(best_b[1] - 0.02, best_b[1] + 0.02, THRESHOLD_GRID_ULTRA):
        for b2 in np.linspace(best_b[2] - 0.02, best_b[2] + 0.02, THRESHOLD_GRID_ULTRA):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    return best_b, best_s


# ---------------------------------------------------------------------------
# Extended Feature Engineering
# ---------------------------------------------------------------------------

def _higher_order_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    if arr.size < 3:
        return {f"{prefix}_skew": np.nan, f"{prefix}_kurt": np.nan,
                f"{prefix}_p10": np.nan, f"{prefix}_p90": np.nan,
                f"{prefix}_iqr": np.nan}
    m, s = np.mean(arr), np.std(arr)
    if s == 0:
        sk, ku = 0.0, 0.0
    else:
        c = (arr - m) / s
        sk, ku = float(np.mean(c**3)), float(np.mean(c**4) - 3.0)
    return {f"{prefix}_skew": sk, f"{prefix}_kurt": ku,
            f"{prefix}_p10": float(np.percentile(arr, 10)),
            f"{prefix}_p90": float(np.percentile(arr, 90)),
            f"{prefix}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))}


def _phase_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    if arr.size < 3:
        return {f"{prefix}_early": np.nan,
                f"{prefix}_mid": np.nan,
                f"{prefix}_late": np.nan}
    sp = np.array_split(arr, 3)
    return {f"{prefix}_early": float(np.mean(sp[0])),
            f"{prefix}_mid": float(np.mean(sp[1])),
            f"{prefix}_late": float(np.mean(sp[2]))}


def _compute_derivatives(pos: np.ndarray, dt: np.ndarray) -> tuple:
    vel = np.diff(pos, axis=0) / dt[:, np.newaxis]
    if vel.shape[0] < 2:
        return vel, np.empty((0, 3)), np.empty((0, 3))
    dt_acc = (dt[:-1] + dt[1:]) / 2.0
    acc = np.diff(vel, axis=0) / dt_acc[:, np.newaxis]
    if acc.shape[0] < 2:
        return vel, acc, np.empty((0, 3))
    dt_jerk = (dt_acc[:-1] + dt_acc[1:]) / 2.0
    jerk = np.diff(acc, axis=0) / dt_jerk[:, np.newaxis]
    return vel, acc, jerk


def extract_extended_features(group: pd.DataFrame) -> dict[str, float]:
    """Compute 66 extended features for one trajectory."""
    f: dict[str, float] = {}
    pos = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    times = group["time_stamp"].to_numpy(dtype="datetime64[ns]")
    n = len(pos)
    dt_ns = np.diff(times).astype(np.float64)
    dt_s = np.where(dt_ns / 1e9 <= 0, np.nan, dt_ns / 1e9)
    vdt = dt_s[~np.isnan(dt_s)]

    _nan_sfx = ["_skew", "_kurt", "_p10", "_p90", "_iqr", "_early", "_mid", "_late"]

    if n >= 2 and vdt.size > 0:
        dtf = np.where(np.isnan(dt_s), np.nanmedian(dt_s), dt_s)
        vel, acc, jerk = _compute_derivatives(pos, dtf)
        sp = np.linalg.norm(vel, axis=1)
        vz = vel[:, 2]
        vh = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)

        for arr, pfx in [(sp, "speed"), (vz, "vz"), (vh, "v_horiz")]:
            f.update(_higher_order_stats(arr, pfx))
            f.update(_phase_stats(arr, pfx))

        if acc.shape[0] > 0:
            am = np.linalg.norm(acc, axis=1)
            az = acc[:, 2]
            ah = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2)
            for arr, pfx in [(am, "acc_mag"), (az, "az"), (ah, "acc_horiz")]:
                f.update(_higher_order_stats(arr, pfx))
                f.update(_phase_stats(arr, pfx))
        else:
            for pfx in ["acc_mag", "az", "acc_horiz"]:
                for sfx in _nan_sfx:
                    f[pfx + sfx] = np.nan
            am = np.array([])

        if jerk.shape[0] > 0:
            jm = np.linalg.norm(jerk, axis=1)
            f.update(_higher_order_stats(jm, "jerk_mag"))
            f.update(_phase_stats(jm, "jerk_mag"))
        else:
            for sfx in _nan_sfx:
                f["jerk_mag" + sfx] = np.nan
            jm = np.array([])

        f["speed_ratio"] = float(sp[-1] / sp[0]) if sp[0] > 0 else np.nan
        f["speed_change"] = float(sp[-1] - sp[0])
        f["vz_ratio"] = float(vz[-1] / vz[0]) if abs(vz[0]) > 1e-6 else np.nan
        f["acc_over_speed"] = (float(np.mean(am) / (np.mean(sp) + 1e-9))
                               if am.size > 0 else np.nan)
        f["jerk_over_acc"] = (float(np.mean(jm) / (np.mean(am) + 1e-9))
                              if jm.size > 0 and am.size > 0 else np.nan)

        if vel.shape[0] >= 2:
            vn = np.clip(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9, None)
            vu = vel / vn
            ca = np.clip(np.sum(vu[:-1] * vu[1:], axis=1), -1, 1)
            ang = np.arccos(ca)
            f["total_curv"] = float(np.sum(ang))
            f["mean_curv"] = float(np.mean(ang))
            f["max_curv"] = float(np.max(ang))
        else:
            f["total_curv"] = f["mean_curv"] = f["max_curv"] = np.nan
    else:
        for pfx in ["speed", "vz", "v_horiz", "acc_mag", "az", "acc_horiz", "jerk_mag"]:
            for sfx in _nan_sfx:
                f[pfx + sfx] = np.nan
        for k in ["speed_ratio", "speed_change", "vz_ratio", "acc_over_speed",
                   "jerk_over_acc", "total_curv", "mean_curv", "max_curv"]:
            f[k] = np.nan

    z = pos[:, 2]
    ai = int(np.argmax(z))
    rise = float(z[ai] - z[0])
    dur = float(np.nansum(dt_s)) if len(dt_s) > 0 else 0.0
    f["apogee_over_dur"] = rise / dur if dur > 0 else np.nan
    hd = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
    f["apogee_over_horiz"] = rise / hd if hd > 1e-6 else np.nan
    return f


def build_extended_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended features for all trajectories."""
    df = raw_df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")
    df = df.sort_values(["traj_ind", "time_stamp"])
    records = []
    for tid, g in df.groupby("traj_ind", sort=False):
        feat = extract_extended_features(g.reset_index(drop=True))
        feat["traj_ind"] = tid
        records.append(feat)
    return pd.DataFrame(records).set_index("traj_ind")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and join base (76) + extended (66) = 142 features."""
    logger.info("Loading base features from %s ...", PARQUET_PATH)
    base_df = pd.read_parquet(PARQUET_PATH)
    base_feat_cols = [c for c in base_df.columns if c != "label"]
    y = base_df["label"].to_numpy(dtype=int)
    groups = base_df.index.to_numpy()

    ext_cache = Path("/content/cache_extended_features.parquet")
    if ext_cache.exists():
        logger.info("Loading extended features from cache...")
        ext_df = pd.read_parquet(ext_cache)
    elif RAW_CSV_PATH.exists():
        logger.info("Computing extended features from %s (~3 min)...", RAW_CSV_PATH)
        raw = pd.read_csv(RAW_CSV_PATH)
        ext_df = build_extended_features(raw)
        ext_df.to_parquet(ext_cache)
        logger.info("Cached extended features to %s", ext_cache)
    else:
        raise FileNotFoundError(
            f"{RAW_CSV_PATH} not found. Upload train.csv to /content/"
        )

    ext_df = ext_df.reindex(base_df.index)
    ext_feat_cols = list(ext_df.columns)

    all_feat_cols = base_feat_cols + ext_feat_cols
    X = np.hstack([
        base_df[base_feat_cols].to_numpy(dtype=np.float32),
        ext_df[ext_feat_cols].to_numpy(dtype=np.float32),
    ])

    logger.info(
        "Loaded %d rows x %d features (%d base + %d extended) | "
        "Labels: 0=%d  1=%d  2=%d",
        X.shape[0], X.shape[1], len(base_feat_cols), len(ext_feat_cols),
        (y == 0).sum(), (y == 1).sum(), (y == 2).sum(),
    )
    return X, y, groups, all_feat_cols


# ---------------------------------------------------------------------------
# Stage 1: Feature Selection
# ---------------------------------------------------------------------------

def _cv_score(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              mask: np.ndarray) -> float:
    X_sub = X[:, mask]
    gkf = GroupKFold(n_splits=5)
    oob = np.zeros((len(y), N_CLASSES))
    for tr, val in gkf.split(X_sub, y, groups):
        Xtr, Xv = X_sub[tr].copy(), X_sub[val].copy()
        ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        impute_nan(Xtr, med)
        impute_nan(Xv, med)
        sw = compute_sample_weights(ytr)
        m = LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="multiclass", num_class=N_CLASSES,
            device="gpu", gpu_use_dp=False,
            random_state=RANDOM_STATE, verbose=-1,
        )
        m.fit(Xtr, ytr, sample_weight=sw)
        oob[val] = m.predict_proba(Xv)
    _, score = optimize_thresholds(y, oob)
    return score


def run_feature_selection(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    logger.info("=" * 70)
    logger.info("STAGE 1: FEATURE SELECTION (%d features)", X.shape[1])
    logger.info("=" * 70)

    # Importance ranking
    X_imp = X.copy()
    med = np.nanmedian(X_imp, axis=0)
    impute_nan(X_imp, med)
    sw = compute_sample_weights(y)
    m = LGBMClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=N_CLASSES,
        device="gpu", gpu_use_dp=False,
        random_state=RANDOM_STATE, verbose=-1,
    )
    m.fit(X_imp, y, sample_weight=sw)
    importances = m.feature_importances_
    rank_order = np.argsort(importances)

    all_mask = np.ones(X.shape[1], dtype=bool)
    baseline = _cv_score(X, y, groups, all_mask)
    logger.info("Baseline score (all %d features): %.4f", X.shape[1], baseline)

    current_mask = all_mask.copy()
    current_score = baseline
    dropped = []

    for idx in rank_order:
        if current_mask.sum() <= 10:
            break
        trial = current_mask.copy()
        trial[idx] = False
        trial_score = _cv_score(X, y, groups, trial)
        fname, imp = feature_names[idx], importances[idx]
        if trial_score >= current_score - 0.0001:
            current_mask = trial
            current_score = trial_score
            dropped.append(fname)
            logger.info("  DROP %-35s (imp=%6.1f) -> %d features, score=%.4f",
                        fname, imp, current_mask.sum(), current_score)
        else:
            logger.info("  KEEP %-35s (imp=%6.1f) -> %.4f",
                        fname, imp, trial_score)

    selected = [feature_names[i] for i in range(len(feature_names)) if current_mask[i]]
    logger.info("Selected %d / %d features (dropped %d)", len(selected),
                len(feature_names), len(dropped))
    logger.info("Final score: %.4f (baseline %.4f)", current_score, baseline)
    return current_mask, selected


# ---------------------------------------------------------------------------
# Stage 2: Optuna (100 trials, LightGBM-focused)
# ---------------------------------------------------------------------------

def build_model(algo: str, params: dict) -> object:
    p = {k: v for k, v in params.items() if k != "algorithm"}
    if algo == "xgboost":
        return XGBClassifier(
            **p, objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", tree_method="hist", device="cuda",
            random_state=RANDOM_STATE, verbosity=0,
        )
    elif algo == "lightgbm":
        return LGBMClassifier(
            **p, objective="multiclass", num_class=N_CLASSES,
            device="gpu", gpu_use_dp=False,
            random_state=RANDOM_STATE, verbose=-1,
        )
    else:  # catboost
        depth = min(p.pop("max_depth", 6), 10)
        n_est = p.pop("n_estimators", 600)
        lr = p.pop("learning_rate", 0.05)
        sub = p.pop("subsample", 0.8)
        rl = p.pop("reg_lambda", 1.0)
        ra = p.pop("reg_alpha", 0.1)
        p.pop("colsample_bytree", None)
        p.pop("min_child_weight", None)
        return CatBoostClassifier(
            iterations=n_est, depth=depth, learning_rate=lr,
            subsample=sub, l2_leaf_reg=rl, random_strength=ra,
            bootstrap_type="Bernoulli",
            loss_function="MultiClass", classes_count=N_CLASSES,
            task_type="GPU", random_seed=RANDOM_STATE, verbose=0,
        )


def create_objective(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> callable:
    def objective(trial: optuna.Trial) -> float:
        # Bias Optuna toward LightGBM (known winner) — 70% LightGBM,
        # 15% XGBoost, 15% CatBoost
        algo = trial.suggest_categorical(
            "algorithm", ["lightgbm"] * 7 + ["xgboost", "xgboost",
                          "catboost", "catboost"] + ["lightgbm"] * 3,
        )
        hp = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        gkf = GroupKFold(n_splits=N_SPLITS)
        fold_raw, fold_tuned = [], []

        for fold_idx, (tr, val) in enumerate(gkf.split(X, y, groups)):
            Xtr, Xv = X[tr].copy(), X[val].copy()
            ytr, yv = y[tr], y[val]
            med = np.nanmedian(Xtr, axis=0)
            impute_nan(Xtr, med)
            impute_nan(Xv, med)
            sw = compute_sample_weights(ytr)

            model = build_model(algo, hp)
            model.fit(Xtr, ytr, sample_weight=sw)
            proba = model.predict_proba(Xv)

            fold_raw.append(min_class_recall(yv, np.argmax(proba, axis=1)))
            _, ts = optimize_thresholds(yv, proba)
            fold_tuned.append(ts)

            if fold_idx >= 2:
                trial.report(np.mean(fold_tuned), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_raw = float(np.mean(fold_raw))
        mean_tuned = float(np.mean(fold_tuned))
        min_tuned = float(np.min(fold_tuned))

        trial.set_user_attr("mean_raw", mean_raw)
        trial.set_user_attr("mean_tuned", mean_tuned)
        trial.set_user_attr("min_fold_tuned", min_tuned)
        trial.set_user_attr("fold_scores_tuned", fold_tuned)

        logger.info(
            "Trial %3d | %-9s | raw=%.4f | tuned=%.4f | min_fold=%.4f | n=%d d=%d lr=%.4f",
            trial.number, algo, mean_raw, mean_tuned, min_tuned,
            hp["n_estimators"], hp["max_depth"], hp["learning_rate"],
        )
        return mean_tuned

    return objective


# ---------------------------------------------------------------------------
# Stage 3 & 4: Final Model
# ---------------------------------------------------------------------------

def train_final_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      best_params: dict) -> tuple[object, np.ndarray, list[float]]:
    algo = best_params["algorithm"]
    logger.info("Training final %s model on 100%% of data...", algo)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oob_proba = np.zeros((len(y), N_CLASSES))

    for tr, val in gkf.split(X, y, groups):
        Xtr, Xv = X[tr].copy(), X[val].copy()
        ytr = y[tr]
        med = np.nanmedian(Xtr, axis=0)
        impute_nan(Xtr, med)
        impute_nan(Xv, med)
        sw = compute_sample_weights(ytr)
        model = build_model(algo, best_params)
        model.fit(Xtr, ytr, sample_weight=sw)
        oob_proba[val] = model.predict_proba(Xv)

    biases, oob_score = optimize_thresholds(y, oob_proba)
    logger.info("OOB global tuned score: %.6f", oob_score)

    oob_fold_scores = []
    for _, val in gkf.split(X, y, groups):
        preds = np.argmax(np.log(oob_proba[val] + 1e-12) + biases, axis=1)
        oob_fold_scores.append(min_class_recall(y[val], preds))

    # Full retrain
    X_full = X.copy()
    med_full = np.nanmedian(X_full, axis=0)
    impute_nan(X_full, med_full)
    sw_full = compute_sample_weights(y)
    final_model = build_model(algo, best_params)
    final_model.fit(X_full, y, sample_weight=sw_full)

    # Save
    import joblib
    joblib.dump(final_model, str(MODEL_OUTPUT.with_suffix(".pkl")))
    np.save("/content/extended_train_medians.npy", med_full)
    np.save("/content/extended_threshold_biases.npy", biases)
    logger.info("Artifacts saved: model, medians, biases")

    return final_model, biases, oob_fold_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    X_all, y, groups, all_feat = load_data()

    # Stage 1: feature selection
    feature_mask, selected_features = run_feature_selection(X_all, y, groups, all_feat)
    X = X_all[:, feature_mask]
    logger.info("Proceeding with %d / %d features", X.shape[1], X_all.shape[1])

    # Stage 2: Optuna
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 2: OPTUNA (%d trials, %d-fold, LightGBM-biased)", N_TRIALS, N_SPLITS)
    logger.info("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        study_name="rocket-extended-lgbm",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(create_objective(X, y, groups), n_trials=N_TRIALS,
                   show_progress_bar=True)

    best = study.best_trial
    logger.info("")
    logger.info("=" * 70)
    logger.info("OPTUNA COMPLETE")
    logger.info("=" * 70)
    logger.info("Best: trial #%d  algorithm=%s  tuned=%.6f  min_fold=%.6f",
                best.number, best.params["algorithm"], best.value,
                best.user_attrs.get("min_fold_tuned", -1))
    logger.info("Hyperparameters:")
    for k, v in sorted(best.params.items()):
        logger.info("  %-25s = %s", k, v)

    fold_scores = best.user_attrs.get("fold_scores_tuned", [])
    if fold_scores:
        logger.info("Per-fold (best trial): %s", [f"{s:.4f}" for s in fold_scores])
        logger.info("Mean: %.4f  Min: %.4f", np.mean(fold_scores), np.min(fold_scores))

    logger.info("\nTop 5 trials:")
    for t in sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]:
        if t.value is not None:
            logger.info("  #%3d  %-9s  tuned=%.4f  raw=%.4f  min_fold=%.4f",
                        t.number, t.params.get("algorithm", "?"),
                        t.value, t.user_attrs.get("mean_raw", -1),
                        t.user_attrs.get("min_fold_tuned", -1))

    # Stages 3 & 4 — single best model
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 3-4: FINAL MODEL + THRESHOLDS")
    logger.info("=" * 70)

    final_model, biases, oob_fold_scores = train_final_model(
        X, y, groups, best.params,
    )

    # Stage 5 — top-K ensemble across the best Optuna configs.
    # Each of the top-K configs trains its own OOB predictions; averaging
    # the K probability arrays reduces per-model variance and can push
    # the hard borderline cases across the decision boundary.
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 5: TOP-K ENSEMBLE (average OOB probabilities)")
    logger.info("=" * 70)

    TOP_K = 5  # number of best Optuna configs to ensemble
    top_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True,
    )[:TOP_K]

    logger.info("Ensembling top-%d configs:", TOP_K)
    for i, t in enumerate(top_trials, 1):
        logger.info("  %d. trial #%d  %s  tuned=%.4f",
                    i, t.number, t.params.get("algorithm"), t.value)

    gkf_ens = GroupKFold(n_splits=N_SPLITS)
    ensemble_oob = np.zeros((len(y), N_CLASSES))

    for trial in top_trials:
        trial_oob = np.zeros((len(y), N_CLASSES))
        for tr, val in gkf_ens.split(X, y, groups):
            Xtr, Xv = X[tr].copy(), X[val].copy()
            ytr = y[tr]
            med = np.nanmedian(Xtr, axis=0)
            impute_nan(Xtr, med)
            impute_nan(Xv, med)
            sw = compute_sample_weights(ytr)
            m = build_model(trial.params["algorithm"], trial.params)
            m.fit(Xtr, ytr, sample_weight=sw)
            trial_oob[val] = m.predict_proba(Xv)
        ensemble_oob += trial_oob
        logger.info("  Collected OOB for trial #%d", trial.number)

    ensemble_oob /= TOP_K  # average probabilities

    ens_biases, ens_score = optimize_thresholds(y, ensemble_oob)
    ens_fold_scores = []
    for _, val in gkf_ens.split(X, y, groups):
        preds = np.argmax(np.log(ensemble_oob[val] + 1e-12) + ens_biases, axis=1)
        ens_fold_scores.append(min_class_recall(y[val], preds))

    logger.info("Ensemble OOB global: %.6f", ens_score)
    logger.info("Ensemble per-fold:   %s", [f"{s:.4f}" for s in ens_fold_scores])
    logger.info("Ensemble mean=%.4f  min=%.4f",
                np.mean(ens_fold_scores), np.min(ens_fold_scores))
    logger.info("Ensemble biases: [%.6f, %.6f, %.6f]", *ens_biases)

    # Save ensemble artifacts if better than single model
    use_ensemble = np.mean(ens_fold_scores) > np.mean(oob_fold_scores)
    best_biases = ens_biases if use_ensemble else biases
    best_fold_scores = ens_fold_scores if use_ensemble else oob_fold_scores
    logger.info("Using %s (%.4f vs single %.4f)",
                "ENSEMBLE" if use_ensemble else "SINGLE MODEL",
                np.mean(ens_fold_scores), np.mean(oob_fold_scores))

    if use_ensemble:
        np.save("/content/extended_threshold_biases.npy", ens_biases)
        logger.info("Saved ensemble biases to /content/extended_threshold_biases.npy")

    # Train sanity (on single final model)
    X_full = X.copy()
    impute_nan(X_full, np.nanmedian(X_full, axis=0))
    train_proba = final_model.predict_proba(X_full)
    train_raw = min_class_recall(y, np.argmax(train_proba, axis=1))
    train_tuned = min_class_recall(
        y, np.argmax(np.log(train_proba + 1e-12) + best_biases, axis=1),
    )

    # Final report
    prev_mean, prev_min = 0.9988, 0.9959
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL REPORT")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  SELECTED FEATURES (%d / %d):", len(selected_features), len(all_feat))
    for i, f in enumerate(selected_features, 1):
        logger.info("    %3d. %s", i, f)
    logger.info("")
    logger.info("  PERFORMANCE:")
    logger.info("    Train Min-Recall (raw):           %.6f", train_raw)
    logger.info("    Train Min-Recall (tuned):         %.6f", train_tuned)
    logger.info("    Val single model (OOB mean):      %.6f", np.mean(oob_fold_scores))
    logger.info("    Val ensemble top-%d (OOB mean):   %.6f", TOP_K, np.mean(ens_fold_scores))
    logger.info("    BEST Val (OOB mean):              %.6f", np.mean(best_fold_scores))
    logger.info("    BEST Val (OOB min):               %.6f", np.min(best_fold_scores))
    logger.info("")
    logger.info("  OOB per-fold: %s", [f"{s:.4f}" for s in best_fold_scores])
    logger.info("")
    logger.info("  OPTIMAL THRESHOLDS (best config):")
    logger.info("    biases = np.array([%.6f, %.6f, %.6f])", *best_biases)
    logger.info("")
    logger.info("  PRODUCTION CODE:")
    logger.info("    selected_features = %s", selected_features)
    logger.info("    biases = np.array([%.6f, %.6f, %.6f])", *best_biases)
    logger.info("    X = df[selected_features].to_numpy()")
    logger.info("    proba = model.predict_proba(X)")
    logger.info("    preds = np.argmax(np.log(proba + 1e-12) + biases, axis=1)")
    logger.info("")

    delta = np.mean(best_fold_scores) - prev_mean
    logger.info("-" * 70)
    logger.info("COMPARISON vs PREVIOUS BEST (LightGBM, 61 features)")
    logger.info("-" * 70)
    logger.info("  Previous: mean=%.4f  min=%.4f", prev_mean, prev_min)
    logger.info("  This run: mean=%.4f  min=%.4f",
                np.mean(best_fold_scores), np.min(best_fold_scores))
    logger.info("  Delta:    %+.4f", delta)
    logger.info("  Gap to 1.0: %.4f", 1.0 - np.mean(best_fold_scores))
    logger.info("")
    logger.info("Total runtime: %.1f minutes", (time.time() - t0) / 60)

    results = {
        "selected_features": selected_features,
        "n_features_original": len(all_feat),
        "n_features_selected": len(selected_features),
        "best_algorithm": best.params["algorithm"],
        "best_params": best.params,
        "best_optuna_mean_tuned": best.value,
        "single_model_oob_mean": float(np.mean(oob_fold_scores)),
        "single_model_oob_min": float(np.min(oob_fold_scores)),
        "ensemble_oob_mean": float(np.mean(ens_fold_scores)),
        "ensemble_oob_min": float(np.min(ens_fold_scores)),
        "best_oob_mean": float(np.mean(best_fold_scores)),
        "best_oob_min": float(np.min(best_fold_scores)),
        "best_oob_fold_scores": best_fold_scores,
        "used_ensemble": use_ensemble,
        "biases": best_biases.tolist(),
        "train_raw": float(train_raw),
        "train_tuned": float(train_tuned),
        "delta_vs_previous": float(delta),
        "previous_oob_mean": prev_mean,
    }
    with open(RESULTS_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_OUTPUT)


if __name__ == "__main__":
    main()
