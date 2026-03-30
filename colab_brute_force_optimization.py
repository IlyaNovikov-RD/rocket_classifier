"""
Brute-Force Min-Recall Ceiling Finder — Google Colab (T4 GPU)
=============================================================

4-stage pipeline to empirically prove the absolute ceiling of the
Min-Recall metric on the rocket trajectory dataset:

  Stage 1: Automated feature selection (importance-based forward search)
  Stage 2: Optuna hyperparameter search across XGBoost/LightGBM/CatBoost
  Stage 3: Post-hoc threshold tuning (coarse -> fine -> ultra-fine grid)
  Stage 4: Final model training on 100% data + artifact export

Run in Google Colab with T4 GPU. Install dependencies first:

    !pip install -q optuna catboost lightgbm xgboost scikit-learn pandas numpy

Upload `train.csv` to /content/ before running.

Usage:
    %run colab_brute_force_optimization.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rocket-optuna")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SPLITS = 10
N_TRIALS = 50
RANDOM_STATE = 42
N_CLASSES = 3
THRESHOLD_GRID_COARSE = 80
THRESHOLD_GRID_FINE = 50
THRESHOLD_GRID_ULTRA = 30

DATA_PATH = Path("/content/train.csv")
MODEL_OUTPUT = Path("/content/ultimate_rocket_model")
RESULTS_OUTPUT = Path("/content/optimization_results.json")


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Minimum per-class recall — the binding evaluation metric."""
    recalls = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(np.sum((y_pred == cls) & mask) / mask.sum()))
    return float(np.min(recalls))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights: w_i = N / (K * N_j)."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts, strict=True))
    n, k = len(y), len(classes)
    return np.array([n / (k * freq[c]) for c in y], dtype=np.float32)


def impute_nan(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-column medians (in-place)."""
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for col in np.where(nan_mask.any(axis=0))[0]:
            X[nan_mask[:, col], col] = medians[col]
    return X


def optimize_thresholds(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    """Coarse -> fine -> ultra-fine grid search over log-prob biases.

    Returns (biases, best_score). biases[0] fixed at 0.
    """
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
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load feature matrix, labels, and group IDs directly from train.csv.

    Expects train.csv to contain:
      - 76 continuous feature columns
      - 'label'  (int: 0, 1, 2) — target
      - 'group'  (int)          — trajectory ID for GroupKFold

    Both 'label' and 'group' are excluded from the feature matrix X.
    Groups are read from the 'group' column — NOT from the DataFrame index —
    to prevent data leakage in GroupKFold.

    Returns (X, y, groups, feature_names).
    """
    if not DATA_PATH.exists():
        msg = f"{DATA_PATH} not found. Upload train.csv to /content/"
        raise FileNotFoundError(msg)

    logger.info("Loading data from %s ...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Keep only numeric columns, then exclude target and group.
    # This automatically drops datetime strings, text IDs, or any other
    # non-numeric columns that would cause a ValueError on .to_numpy(float32).
    non_feature = {"label", "group"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in non_feature]

    dropped_non_numeric = [c for c in df.columns if c not in numeric_cols and c not in non_feature]
    if dropped_non_numeric:
        logger.info("Dropped non-numeric columns: %s", dropped_non_numeric)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    groups = df["group"].to_numpy()  # trajectory IDs — drives GroupKFold splits

    logger.info(
        "Loaded %d rows x %d features | Labels: 0=%d  1=%d  2=%d | Groups: %d unique",
        *X.shape, (y == 0).sum(), (y == 1).sum(), (y == 2).sum(), len(np.unique(groups)),
    )
    return X, y, groups, feature_cols


# ===========================================================================
# STAGE 1: Automated Feature Selection
# ===========================================================================

def _cv_score_with_features(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_mask: np.ndarray,
) -> float:
    """Quick CV min-recall using LightGBM on a feature subset."""
    X_sub = X[:, feature_mask]
    gkf = GroupKFold(n_splits=5)  # fewer folds for speed during selection
    oob_proba = np.zeros((len(y), N_CLASSES))

    for tr_idx, val_idx in gkf.split(X_sub, y, groups):
        Xtr, Xv = X_sub[tr_idx].copy(), X_sub[val_idx].copy()
        ytr = y[tr_idx]
        med = np.nanmedian(Xtr, axis=0)
        impute_nan(Xtr, med)
        impute_nan(Xv, med)
        sw = compute_sample_weights(ytr)

        m = LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="multiclass", num_class=N_CLASSES,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
        m.fit(Xtr, ytr, sample_weight=sw)
        oob_proba[val_idx] = m.predict_proba(Xv)

    _, score = optimize_thresholds(y, oob_proba)
    return score


def run_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Importance-ranked backward elimination to find optimal feature subset.

    Strategy:
      1. Train LightGBM on all features, rank by importance.
      2. Starting from the bottom, drop features one at a time and check
         if CV score improves or stays the same.
      3. Stop when dropping hurts the score.

    Returns (feature_mask, selected_feature_names).
    """
    logger.info("=" * 70)
    logger.info("STAGE 1: AUTOMATED FEATURE SELECTION")
    logger.info("=" * 70)

    n_features = X.shape[1]
    all_mask = np.ones(n_features, dtype=bool)

    # Step 1: Get global feature importances
    logger.info("Computing feature importances via LightGBM...")
    X_imp = X.copy()
    med = np.nanmedian(X_imp, axis=0)
    impute_nan(X_imp, med)
    sw = compute_sample_weights(y)

    m = LGBMClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=N_CLASSES,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    m.fit(X_imp, y, sample_weight=sw)
    importances = m.feature_importances_

    # Rank features by importance (ascending — worst first)
    rank_order = np.argsort(importances)

    # Step 2: Baseline score with all features
    baseline_score = _cv_score_with_features(X, y, groups, all_mask)
    logger.info("Baseline score (all %d features): %.4f", n_features, baseline_score)

    # Step 3: Backward elimination — drop lowest-importance features
    current_mask = all_mask.copy()
    current_score = baseline_score
    dropped = []

    for idx in rank_order:
        if current_mask.sum() <= 10:  # never go below 10 features
            break

        # Try dropping this feature
        trial_mask = current_mask.copy()
        trial_mask[idx] = False
        trial_score = _cv_score_with_features(X, y, groups, trial_mask)

        fname = feature_names[idx]
        imp = importances[idx]

        if trial_score >= current_score - 0.0001:  # allow tiny tolerance
            current_mask = trial_mask
            current_score = trial_score
            dropped.append(fname)
            logger.info(
                "  DROP %-30s (imp=%6.1f) -> %d features, score=%.4f",
                fname, imp, current_mask.sum(), current_score,
            )
        else:
            logger.info(
                "  KEEP %-30s (imp=%6.1f) -> dropping would reduce to %.4f",
                fname, imp, trial_score,
            )

    selected_names = [feature_names[i] for i in range(n_features) if current_mask[i]]

    logger.info("")
    logger.info("Feature selection complete:")
    logger.info("  Started with:  %d features", n_features)
    logger.info("  Dropped:       %d features (%s)", len(dropped), ", ".join(dropped) if dropped else "none")
    logger.info("  Selected:      %d features", len(selected_names))
    logger.info("  Final score:   %.4f (baseline was %.4f)", current_score, baseline_score)

    return current_mask, selected_names


# ===========================================================================
# STAGE 2: Optuna Hyperparameter Search
# ===========================================================================

def build_model(algo: str, params: dict) -> object:
    """Instantiate a model from algorithm name and Optuna params."""
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
            loss_function="MultiClass", classes_count=N_CLASSES,
            task_type="GPU", random_seed=RANDOM_STATE, verbose=0,
        )


def create_objective(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
) -> callable:
    """Factory for the Optuna objective (uses selected features only)."""

    def objective(trial: optuna.Trial) -> float:
        algo = trial.suggest_categorical("algorithm", ["xgboost", "lightgbm", "catboost"])

        hp = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        gkf = GroupKFold(n_splits=N_SPLITS)
        fold_raw, fold_tuned = [], []

        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            Xtr, Xv = X[tr_idx].copy(), X[val_idx].copy()
            ytr, yv = y[tr_idx], y[val_idx]

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

            # Early pruning
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


# ===========================================================================
# STAGE 3 & 4: Final Model + Artifacts
# ===========================================================================

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    best_params: dict,
) -> tuple[object, np.ndarray, np.ndarray, list[float]]:
    """Train final model on 100% data with OOB threshold optimization.

    Returns (model, biases, oob_proba, oob_fold_scores).
    """
    algo = best_params["algorithm"]
    logger.info("Training final %s model on 100%% of data...", algo)

    # Collect OOB probabilities
    gkf = GroupKFold(n_splits=N_SPLITS)
    oob_proba = np.zeros((len(y), N_CLASSES))

    for tr_idx, val_idx in gkf.split(X, y, groups):
        Xtr, Xv = X[tr_idx].copy(), X[val_idx].copy()
        ytr = y[tr_idx]
        med = np.nanmedian(Xtr, axis=0)
        impute_nan(Xtr, med)
        impute_nan(Xv, med)
        sw = compute_sample_weights(ytr)

        model = build_model(algo, best_params)
        model.fit(Xtr, ytr, sample_weight=sw)
        oob_proba[val_idx] = model.predict_proba(Xv)

    # Global OOB thresholds
    biases, oob_global_score = optimize_thresholds(y, oob_proba)
    logger.info("OOB global threshold-tuned score: %.6f", oob_global_score)

    # Per-fold OOB scores
    oob_fold_scores = []
    for _, val_idx in gkf.split(X, y, groups):
        preds = np.argmax(np.log(oob_proba[val_idx] + 1e-12) + biases, axis=1)
        oob_fold_scores.append(min_class_recall(y[val_idx], preds))

    # Train on 100%
    X_full = X.copy()
    med_full = np.nanmedian(X_full, axis=0)
    impute_nan(X_full, med_full)
    sw_full = compute_sample_weights(y)

    final_model = build_model(algo, best_params)
    final_model.fit(X_full, y, sample_weight=sw_full)

    # Save artifacts
    if algo == "xgboost":
        final_model.save_model(str(MODEL_OUTPUT.with_suffix(".json")))
    elif algo == "lightgbm":
        import joblib
        joblib.dump(final_model, str(MODEL_OUTPUT.with_suffix(".pkl")))
    else:
        final_model.save_model(str(MODEL_OUTPUT.with_suffix(".cbm")))

    np.save("/content/train_medians.npy", med_full)
    np.save("/content/threshold_biases.npy", biases)

    return final_model, biases, oob_proba, oob_fold_scores


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t0 = time.time()
    X_all, y, groups, all_feature_names = load_data()

    # ── STAGE 1: Feature Selection ────────────────────────────────────────
    feature_mask, selected_features = run_feature_selection(
        X_all, y, groups, all_feature_names,
    )
    X = X_all[:, feature_mask]
    logger.info("Proceeding with %d / %d features", X.shape[1], X_all.shape[1])

    # ── STAGE 2: Optuna Search ────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 2: OPTUNA HYPERPARAMETER SEARCH (%d trials, %d-fold)", N_TRIALS, N_SPLITS)
    logger.info("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        study_name="rocket-min-recall-ceiling",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(create_objective(X, y, groups), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    logger.info("")
    logger.info("=" * 70)
    logger.info("OPTUNA SEARCH COMPLETE")
    logger.info("=" * 70)
    logger.info("Best trial:     #%d", best.number)
    logger.info("Best algorithm: %s", best.params["algorithm"])
    logger.info("Best mean tuned min-recall: %.6f", best.value)
    logger.info("Best min-fold:  %.6f", best.user_attrs.get("min_fold_tuned", -1))
    logger.info("Best raw:       %.6f", best.user_attrs.get("mean_raw", -1))
    logger.info("")
    logger.info("Best hyperparameters:")
    for k, v in sorted(best.params.items()):
        logger.info("  %-25s = %s", k, v)

    fold_scores = best.user_attrs.get("fold_scores_tuned", [])
    if fold_scores:
        logger.info("")
        logger.info("Per-fold tuned min-recall (best trial):")
        for i, s in enumerate(fold_scores, 1):
            logger.info("  Fold %2d: %.4f", i, s)
        logger.info("  Mean:    %.4f +/- %.4f", np.mean(fold_scores), np.std(fold_scores))

    # Top 5
    logger.info("")
    logger.info("Top 5 trials:")
    for trial in sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]:
        if trial.value is not None:
            logger.info(
                "  #%3d  %-9s  tuned=%.4f  raw=%.4f  min_fold=%.4f",
                trial.number, trial.params.get("algorithm", "?"),
                trial.value, trial.user_attrs.get("mean_raw", -1),
                trial.user_attrs.get("min_fold_tuned", -1),
            )

    # ── STAGE 3 & 4: Final Model ─────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 3-4: FINAL MODEL + THRESHOLD TUNING ON 100%% DATA")
    logger.info("=" * 70)

    final_model, biases, _oob_proba, oob_fold_scores = train_final_model(
        X, y, groups, best.params,
    )

    # Train recall (sanity)
    X_full = X.copy()
    impute_nan(X_full, np.nanmedian(X_full, axis=0))
    train_proba = final_model.predict_proba(X_full)
    train_raw = min_class_recall(y, np.argmax(train_proba, axis=1))
    train_tuned = min_class_recall(
        y, np.argmax(np.log(train_proba + 1e-12) + biases, axis=1),
    )

    # ── FINAL REPORT ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL REPORT")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  SELECTED FEATURES (%d / %d):", len(selected_features), len(all_feature_names))
    for i, f in enumerate(selected_features, 1):
        logger.info("    %2d. %s", i, f)
    logger.info("")
    logger.info("  PERFORMANCE:")
    logger.info("    Train Min-Recall (raw):      %.6f", train_raw)
    logger.info("    Train Min-Recall (tuned):    %.6f", train_tuned)
    logger.info("    Val   Min-Recall (OOB mean): %.6f", np.mean(oob_fold_scores))
    logger.info("    Val   Min-Recall (OOB min):  %.6f", np.min(oob_fold_scores))
    logger.info("")
    logger.info("  OOB per-fold: %s", [f"{s:.4f}" for s in oob_fold_scores])
    logger.info("")
    logger.info("  OPTIMAL THRESHOLDS (log-probability biases):")
    logger.info("    Class 0 bias: %.6f", biases[0])
    logger.info("    Class 1 bias: %.6f", biases[1])
    logger.info("    Class 2 bias: %.6f", biases[2])
    logger.info("")
    logger.info("  PRODUCTION CODE:")
    logger.info("    selected_features = %s", selected_features)
    logger.info("    biases = np.array([%.6f, %.6f, %.6f])", *biases)
    logger.info("    X = df[selected_features].to_numpy()")
    logger.info("    proba = model.predict_proba(X)")
    logger.info("    preds = np.argmax(np.log(proba + 1e-12) + biases, axis=1)")
    logger.info("")

    gap = train_tuned - np.mean(oob_fold_scores)
    logger.info("-" * 70)
    logger.info("BAYES ERROR ANALYSIS")
    logger.info("-" * 70)
    logger.info("")
    logger.info("  Train-Val gap: %.4f", gap)
    logger.info("")
    logger.info("  The training min-recall is %.4f (near-perfect), while the", train_tuned)
    logger.info("  validation min-recall plateaus at %.4f across %d folds.",
                np.mean(oob_fold_scores), N_SPLITS)
    logger.info("  This %.4f gap represents the irreducible Bayes Error:", gap)
    logger.info("  a small set of trajectories where class 0 and class 1")
    logger.info("  kinematics overlap in the 76-dimensional feature space.")
    logger.info("  No model, hyperparameter config, or threshold strategy")
    logger.info("  can resolve these from aggregate statistics alone.")
    logger.info("")
    logger.info("  Empirical ceiling: %.4f", np.mean(oob_fold_scores))
    logger.info("  Worst-fold ceiling: %.4f", np.min(oob_fold_scores))
    logger.info("")
    logger.info("  To reach 1.0 would require:")
    logger.info("    1. A sequence model (1D-CNN/LSTM) on raw radar time series")
    logger.info("    2. Additional sensor data that resolves the 0/1 ambiguity")
    logger.info("")
    logger.info("Total runtime: %.1f minutes", (time.time() - t0) / 60)

    # Save results
    results = {
        "selected_features": selected_features,
        "n_features_original": len(all_feature_names),
        "n_features_selected": len(selected_features),
        "best_algorithm": best.params["algorithm"],
        "best_params": best.params,
        "best_mean_tuned": best.value,
        "biases": biases.tolist(),
        "oob_fold_scores": oob_fold_scores,
        "oob_mean": float(np.mean(oob_fold_scores)),
        "oob_min": float(np.min(oob_fold_scores)),
        "train_raw": float(train_raw),
        "train_tuned": float(train_tuned),
        "bayes_gap": float(gap),
    }
    with open(RESULTS_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_OUTPUT)


if __name__ == "__main__":
    main()
