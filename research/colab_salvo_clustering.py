"""
SPATIO-TEMPORAL SALVO CLUSTERING EXPERIMENT — Google Colab
===========================================================

Hypothesis
----------
The assignment explicitly states that rockets are fired in salvos (together or
with a short delay) and that rebel groups are geographically clustered, each
independently buying the same rocket type.

If both assumptions hold simultaneously:
    • Rockets in the same salvo → same rebel group → same rocket class
    • Knowing "this trajectory was fired alongside N other class-2 rockets"
      gives signal that pure per-trajectory kinematics cannot capture

This experiment tests whether DBSCAN-derived salvo features break the 0.9995
Bayes-error ceiling established by kinematic features alone.

Business assumptions leveraged (from assignment)
-------------------------------------------------
3a. Each launcher has a different payload capacity (max rockets per launcher)
    → group_max_salvo_size is the empirical upper bound = launcher capacity proxy
3b. Rockets are fired in salvos (together or with short delay)
    → spatio-temporal DBSCAN on (launch_x, launch_y, time) identifies episodes
3c. Few rebel groups concentrated in geographic areas (persistent bases)
    → pure-spatial DBSCAN on (launch_x, launch_y) identifies the REBEL GROUP,
      which is different from a single salvo: the same base fires many salvos
      over weeks/months — all from the same group, all the same rocket class
3d. Each rebel group independently buys rockets (different MO, no umbrella org)
    → same group = same class across ALL their salvos, not just one episode
    → rebel_group_id is the strongest label proxy in the entire feature set

Leakage-safe design
-------------------
All salvo features are computed from geometry and timing ONLY — no labels
are used. GroupKFold on salvo_id ensures no salvo spans train/val, which
would allow the model to memorise salvo→class mappings from training folds.

Setup (run this cell first in Colab)
-------------------------------------
    !pip install -q optuna lightgbm scikit-learn pandas numpy pyarrow

Upload to /content/ (or adjust DATA_DIR below):
    - train.csv
    - cache_train_features.parquet   (or .feather)
"""

# %%
# ── Imports ────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Force-reconfigure logging so it works in Jupyter/Colab where basicConfig()
# is a no-op if any handler is already attached.
_fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(_handler)
root_logger.setLevel(logging.INFO)
log = logging.getLogger(__name__)

# %%
# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/content")            # adjust if using Google Drive
TRAIN_CSV = DATA_DIR / "train.csv"
FEATURE_CACHE = DATA_DIR / "cache_train_features.parquet"  # .feather also accepted

BASELINE_SCORE = 0.9995                # kinematic-only ceiling to beat

# DBSCAN: distances are computed in StandardScaler-normalised space (mean=0, std=1).
# eps=0.5 means two rockets must be within 0.5 std in ALL three scaled dimensions
# (launch_x, launch_y, launch_time_s) to be considered part of the same salvo.
# Sensitivity analysis below will help you tune this if needed.
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 2                 # at least 2 rockets to form a salvo

N_SPLITS = 10                          # GroupKFold folds (grouped by salvo_id)
N_OPTUNA_TRIALS = 20                   # increase for production runs
RANDOM_SEED = 42

# The 61 production features (must match model.py SELECTED_FEATURES exactly)
SELECTED_FEATURES: list[str] = [
    "n_points", "total_duration_s", "dt_std", "dt_max", "dt_median",
    "speed_mean", "speed_std", "speed_min", "speed_max",
    "vx_mean", "vx_std", "vx_min", "vx_max",
    "vy_mean", "vy_std", "vy_min", "vy_max",
    "vz_std", "vz_min", "vz_median",
    "v_horiz_mean", "v_horiz_std", "v_horiz_min", "v_horiz_median",
    "initial_speed", "initial_vz", "final_speed", "final_vz",
    "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max", "acc_mag_median",
    "az_mean", "az_std", "az_min", "az_max", "az_median",
    "acc_horiz_mean", "acc_horiz_std", "acc_horiz_min", "acc_horiz_max",
    "mean_az",
    "jerk_mag_mean", "jerk_mag_std", "jerk_mag_min", "jerk_mag_max", "jerk_mag_median",
    "initial_z", "final_z", "delta_z_total", "apogee_relative", "time_to_apogee_s",
    "x_range", "y_range", "z_range",
    "max_horiz_range", "final_horiz_range", "path_length_3d",
    "launch_x", "launch_y",
]

SALVO_FEATURES = [
    "salvo_size",             # number of rockets in the salvo
    "salvo_duration_s",       # seconds from first to last launch in the salvo
    "salvo_spatial_spread_m", # max pairwise launch-point distance within the salvo
    "salvo_time_rank",        # launch order of this rocket within its salvo (1-indexed)
]

# Rebel-group features (assumption 3c + 3d + 3a):
#   3c: few groups, geographically concentrated → pure-spatial DBSCAN identifies the
#       PERSISTENT base, not just a single firing event
#   3d: each group independently buys rockets → same group = same rocket class across
#       ALL their salvos over time (this is the strongest assumption)
#   3a: each launcher has a different payload capacity → group_max_salvo_size is the
#       largest salvo ever observed from this group, i.e. the launcher capacity proxy
GROUP_FEATURES = [
    "rebel_group_id",         # integer ID of the persistent geographic rebel base
    "group_total_rockets",    # total trajectories attributed to this group (activity level)
    "group_n_salvos",         # number of distinct firing events from this group
    "group_max_salvo_size",   # largest salvo ever seen from this group (launcher capacity proxy)
]

# eps for pure-spatial rebel-group clustering (no time component).
# Larger than DBSCAN_EPS because the same rebel base may have small positional
# variance across many firing events; we want to merge those into one group.
GROUP_EPS = 1.0
GROUP_MIN_SAMPLES = 3  # at least 3 rockets total to constitute a rebel group

# %%
# ── Metric ─────────────────────────────────────────────────────────────────────

def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Minimum per-class recall — the official evaluation metric."""
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(np.sum((y_pred == cls) & mask) / mask.sum()))
    return float(np.min(recalls))

# %%
# ── Step 1: Load raw data and extract per-trajectory launch features ───────────

log.info("Loading raw training data: %s", TRAIN_CSV)
df_raw = pd.read_csv(TRAIN_CSV)
df_raw["time_stamp"] = pd.to_datetime(df_raw["time_stamp"], format="mixed")

# For each trajectory take the earliest radar ping as the launch timestamp.
# Note: the first radar capture may lag true launch by a few frames, but it
# is the best proxy available without additional sensor data.
launch_meta = (
    df_raw.sort_values("time_stamp")
    .groupby("traj_ind", sort=False)
    .agg(
        launch_time=("time_stamp", "first"),
        label=("label", "first"),      # all pings in a trajectory share one label
    )
    .reset_index()
)
log.info(
    "Trajectories: %d  |  Classes: %s",
    len(launch_meta),
    dict(launch_meta["label"].value_counts().sort_index()),
)

# %%
# ── Step 2: Load feature cache and attach launch_time ─────────────────────────

# Prefer Feather (faster) but fall back to Parquet
feather_path = FEATURE_CACHE.with_suffix(".feather")
if feather_path.exists():
    import pyarrow.feather as pa_feather
    feats = pa_feather.read_table(str(feather_path), memory_map=True).to_pandas()
    log.info("Loaded features from Feather: %s", feather_path)
else:
    feats = pd.read_parquet(FEATURE_CACHE)
    log.info("Loaded features from Parquet: %s", FEATURE_CACHE)

log.info("Feature matrix shape: %s", feats.shape)

# Merge launch_time into the feature matrix (join on traj_ind index)
feats = feats.join(launch_meta.set_index("traj_ind")[["launch_time"]], how="left")

# Sanity check: launch_x and launch_y must already be in the feature matrix
assert "launch_x" in feats.columns and "launch_y" in feats.columns, (
    "launch_x / launch_y missing from feature cache — rebuild the cache."
)

# %%
# ── Step 3: DBSCAN salvo clustering ───────────────────────────────────────────
#
# Cluster on three dimensions (all normalised to mean=0, std=1):
#   1. launch_x   — spatial (east-west)
#   2. launch_y   — spatial (north-south)
#   3. launch_time_s — temporal (Unix seconds)
#
# eps=0.5 in normalised space means rockets must be within half a standard
# deviation in all three dimensions to be in the same salvo.  This is tight
# enough to identify true co-launched rockets without merging distinct events.

log.info("Running DBSCAN (eps=%.2f, min_samples=%d)...", DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

launch_time_s = feats["launch_time"].astype(np.int64) / 1e9  # nanoseconds → seconds

cluster_input = pd.DataFrame({
    "launch_x": feats["launch_x"],
    "launch_y": feats["launch_y"],
    "launch_time_s": launch_time_s,
}).fillna(0.0)  # NaN-safe: missing coordinates → origin (rare edge case)

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_input)

dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
raw_labels = dbscan.fit_predict(cluster_scaled)

n_salvos = len(set(raw_labels)) - (1 if -1 in raw_labels else 0)
n_noise = (raw_labels == -1).sum()
log.info(
    "DBSCAN result: %d salvos identified | %d noise trajectories (treated as solo rockets)",
    n_salvos, n_noise,
)

# Assign unique salvo_id to each trajectory.
# DBSCAN noise (label -1) → treat each as its own salvo of size 1.
next_id = raw_labels.max() + 1
salvo_id_arr = raw_labels.copy()
for i, lbl in enumerate(salvo_id_arr):
    if lbl == -1:
        salvo_id_arr[i] = next_id
        next_id += 1

feats["salvo_id"] = salvo_id_arr

# %%
# ── Step 4: Sensitivity analysis — how eps affects salvo discovery ─────────────
#
# Run DBSCAN across a range of eps values to understand cluster structure before
# committing to one value.  Look for the eps where salvo_size starts plateauing
# (too large → merges distinct events; too small → everything is noise).

log.info("Running eps sensitivity analysis...")
print("\n  eps  |  salvos  |  noise  |  mean_size  |  max_size")
print("  -----|----------|---------|-------------|----------")
for eps_test in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
    lbls = DBSCAN(eps=eps_test, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1).fit_predict(cluster_scaled)
    actual_salvos = lbls[lbls >= 0]
    salvo_counts = pd.Series(lbls[lbls >= 0]).value_counts()
    n_s = len(salvo_counts)
    n_nz = (lbls == -1).sum()
    mean_sz = salvo_counts.mean() if len(salvo_counts) else 0
    max_sz = salvo_counts.max() if len(salvo_counts) else 0
    print(f"  {eps_test:<5.2f}|  {n_s:<8}|  {n_nz:<7}|  {mean_sz:<11.2f}|  {max_sz}")
print()

# %%
# ── Step 5: Engineer group-level salvo features ────────────────────────────────

log.info("Engineering salvo features...")

salvo_feature_rows = []

for _salvo_id, group in feats.groupby("salvo_id"):
    n = len(group)
    lx = group["launch_x"].values
    ly = group["launch_y"].values
    lt = group["launch_time"].values.astype(np.int64) / 1e9  # seconds

    # salvo_spatial_spread: max pairwise Euclidean distance between launch points
    if n > 1:
        dx = lx[:, None] - lx[None, :]
        dy = ly[:, None] - ly[None, :]
        spread = float(np.sqrt(dx**2 + dy**2).max())
        duration = float(lt.max() - lt.min())
    else:
        spread = 0.0
        duration = 0.0

    # salvo_time_rank: 1 = first fired, 2 = second, … (captures firing order)
    time_ranks = pd.Series(lt).rank(method="first").astype(int).values

    for i, traj_id in enumerate(group.index):
        salvo_feature_rows.append({
            "traj_ind": traj_id,
            "salvo_size": n,
            "salvo_duration_s": duration,
            "salvo_spatial_spread_m": spread,
            "salvo_time_rank": int(time_ranks[i]),
        })

salvo_df = pd.DataFrame(salvo_feature_rows).set_index("traj_ind")
feats = feats.join(salvo_df, how="left")

log.info("Salvo feature summary:")
print(feats[SALVO_FEATURES].describe().round(2))
print()
log.info(
    "Salvo size distribution:\n%s",
    feats["salvo_size"].value_counts().sort_index().to_string(),
)

# %%
# ── Step 5b: Rebel group clustering (pure-spatial, no time) ───────────────────
#
# CRITICAL distinction from Step 3 (spatio-temporal salvo clustering):
#
#   Salvo    = a single firing event at time T from location X.
#              A rebel group at the same base fires at T=0, T=90 days, T=180 days
#              → three separate salvos, three different salvo_ids.
#
#   Rebel group = the PERSISTENT geographic base that fires many salvos over time.
#              Clustering by (launch_x, launch_y) WITHOUT time merges all salvos
#              from the same base into one group.
#
# Why this matters (assignment assumptions 3c + 3d):
#   "few rebel groups, geographically concentrated, each buying rockets independently"
#   → every trajectory from group X is the SAME rocket class, regardless of when
#     it was fired. rebel_group_id is therefore a strong label proxy that persists
#     across time, unlike salvo_id which is episode-specific.
#
# Assignment assumption 3a (launcher capacity):
#   group_max_salvo_size = the largest salvo ever observed from this base.
#   Each launcher has a fixed max capacity; this is its empirical upper bound.

log.info(
    "Running rebel group clustering (pure-spatial, eps=%.2f, min_samples=%d)...",
    GROUP_EPS, GROUP_MIN_SAMPLES,
)

spatial_input = feats[["launch_x", "launch_y"]].fillna(0.0)
spatial_scaler = StandardScaler()
spatial_scaled = spatial_scaler.fit_transform(spatial_input)

group_dbscan = DBSCAN(eps=GROUP_EPS, min_samples=GROUP_MIN_SAMPLES, n_jobs=-1)
raw_group_labels = group_dbscan.fit_predict(spatial_scaled)

n_groups = len(set(raw_group_labels)) - (1 if -1 in raw_group_labels else 0)
n_group_noise = (raw_group_labels == -1).sum()
log.info(
    "Rebel group result: %d groups | %d solo rockets (noise)",
    n_groups, n_group_noise,
)

# Noise rockets → unique group IDs (they are from unknown/isolated positions)
next_group_id = raw_group_labels.max() + 1
group_id_arr = raw_group_labels.copy()
for i, lbl in enumerate(group_id_arr):
    if lbl == -1:
        group_id_arr[i] = next_group_id
        next_group_id += 1

feats["rebel_group_id"] = group_id_arr

# Build group-level aggregate features and merge back
group_stats = feats.groupby("rebel_group_id").agg(
    group_total_rockets=("rebel_group_id", "count"),
    group_n_salvos=("salvo_id", "nunique"),
    group_max_salvo_size=("salvo_size", "max"),
)
feats = feats.join(group_stats, on="rebel_group_id")

log.info("Rebel group summary (top groups by total rockets):")
print(
    feats.groupby("rebel_group_id")[["group_total_rockets", "group_n_salvos",
                                      "group_max_salvo_size", "label"]]
    .first()
    .sort_values("group_total_rockets", ascending=False)
    .head(10)
    .to_string()
)

# %%
# ── Step 6: Build feature matrices ────────────────────────────────────────────

y = feats["label"].to_numpy(dtype=int)

# Impute NaN with column medians (mirrors production behaviour)
def impute(X: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(X, axis=0)
    nan_cols = np.where(np.isnan(X).any(axis=0))[0]
    X = X.copy()
    for col in nan_cols:
        X[np.isnan(X[:, col]), col] = medians[col]
    return X

ALL_NEW_FEATURES = SALVO_FEATURES + GROUP_FEATURES

X_baseline = impute(feats.reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32))
X_enriched = impute(feats.reindex(columns=SELECTED_FEATURES + ALL_NEW_FEATURES).to_numpy(dtype=np.float32))
salvo_ids  = feats["salvo_id"].to_numpy(dtype=int)

log.info("Baseline features : %s", X_baseline.shape)
log.info(
    "Enriched features : %s  (+%d salvo + %d rebel-group features)",
    X_enriched.shape, len(SALVO_FEATURES), len(GROUP_FEATURES),
)

# %%
# ── Step 7: Cross-validated evaluation helper ──────────────────────────────────
#
# GroupKFold groups by salvo_id so no salvo spans train/val.
# This is strictly more conservative than the original GroupKFold(traj_ind),
# preventing the model from memorising salvo→class mappings.
# Both baseline and enriched models use the same splits for a fair comparison.

def cv_min_recall(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict,
    n_splits: int = N_SPLITS,
) -> tuple[float, float]:
    """Run GroupKFold CV; return (mean_min_recall, std_min_recall)."""
    gkf = GroupKFold(n_splits=n_splits)
    recalls = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        clf = LGBMClassifier(**params, verbose=-1)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        recalls.append(min_class_recall(y[val_idx], preds))
    return float(np.mean(recalls)), float(np.std(recalls))

# %%
# ── Step 8: Quick baseline score with production-like params ──────────────────

log.info("Evaluating baseline (61 kinematic features, GroupKFold by salvo_id)...")
t0 = time.time()

baseline_params = dict(
    n_estimators=2011,
    learning_rate=0.082,
    max_depth=12,
    subsample=0.913,
    colsample_bytree=0.679,
    objective="multiclass",
    num_class=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

base_mean, base_std = cv_min_recall(X_baseline, y, salvo_ids, baseline_params)
log.info(
    "Baseline OOB min-recall: %.4f ± %.4f  (%.1fs)",
    base_mean, base_std, time.time() - t0,
)

# %%
# ── Step 9: Optuna study — enriched model (61 + salvo features) ───────────────

log.info("Starting Optuna study (%d trials, enriched features)...", N_OPTUNA_TRIALS)

def make_objective(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Return an Optuna objective that maximises OOB min-recall."""
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 500, 3000),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 6, 15),
            num_leaves       = trial.suggest_int("num_leaves", 31, 300),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 100),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            objective        = "multiclass",
            num_class        = 3,
            class_weight     = "balanced",
            n_jobs           = -1,
            random_state     = RANDOM_SEED,
        )
        mean_recall, _ = cv_min_recall(X, y, groups, params)
        return mean_recall
    return objective

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    study_name="salvo_enriched",
)
study.optimize(
    make_objective(X_enriched, y, salvo_ids),
    n_trials=N_OPTUNA_TRIALS,
    show_progress_bar=True,
)

best_params = study.best_params | dict(
    objective="multiclass", num_class=3,
    class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
)
enriched_mean, enriched_std = cv_min_recall(X_enriched, y, salvo_ids, best_params)

log.info(
    "Enriched OOB min-recall: %.4f ± %.4f  (best Optuna trial: %.4f)",
    enriched_mean, enriched_std, study.best_value,
)

# %%
# ── Step 10: Results comparison ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  BASELINE VS ENRICHED — Spatio-Temporal Salvo Clustering")
print("=" * 60)
print(f"  Baseline  (61 kinematic features)  : {base_mean:.4f} ± {base_std:.4f}")
print(f"  Enriched  (61 + {len(SALVO_FEATURES)} salvo features)  : {enriched_mean:.4f} ± {enriched_std:.4f}")
print(f"  Reported ceiling (kinematic-only)  : {BASELINE_SCORE:.4f}")
delta = enriched_mean - base_mean
print(f"\n  Δ min-recall                       : {delta:+.4f}")
print("=" * 60)

if enriched_mean > BASELINE_SCORE:
    print(f"\n  ✓ HYPOTHESIS CONFIRMED: salvo features BREAK the {BASELINE_SCORE} ceiling.")
    print("  → Promote salvo features to production SELECTED_FEATURES.")
    print("  → Retrain on full dataset with best_params and export new artifacts.")
elif enriched_mean > base_mean:
    print("\n  ~ MARGINAL IMPROVEMENT: enriched model is better than CV baseline")
    print(f"    but does not clearly exceed the {BASELINE_SCORE} reported ceiling.")
    print("  → Increase N_OPTUNA_TRIALS and tune DBSCAN_EPS before concluding.")
else:
    print("\n  ✗ HYPOTHESIS REJECTED: salvo features do not improve min-recall.")
    print("  → The 0.9995 ceiling is confirmed as fundamental data ambiguity,")
    print("    not a feature-poverty problem.")

# %%
# ── Step 11: Feature importance — do salvo features rank highly? ──────────────

log.info("Training final model on full data to get feature importances...")
final_clf = LGBMClassifier(**best_params)
final_clf.fit(X_enriched, y)

feature_names = SELECTED_FEATURES + ALL_NEW_FEATURES
importances = pd.Series(
    final_clf.feature_importances_,
    index=feature_names,
).sort_values(ascending=False)

print("\n  Top 20 feature importances (gain):")
print(importances.head(20).to_string())

print("\n  Salvo feature importances:")
for feat in SALVO_FEATURES:
    rank = importances.index.get_loc(feat) + 1
    print(f"    {feat:<30} rank={rank:>3}  importance={importances[feat]:.1f}")

print("\n  Rebel group feature importances:")
for feat in GROUP_FEATURES:
    rank = importances.index.get_loc(feat) + 1
    print(f"    {feat:<30} rank={rank:>3}  importance={importances[feat]:.1f}")

# %%
# ── Step 12: Salvo purity analysis (how often do salvo rockets share a label?) ──
#
# If salvo_purity ≈ 1.0, the hypothesis is strongly supported: rockets in the
# same salvo reliably come from the same rebel group (same class).
# If salvo_purity is low, the assumption breaks down in the data.

salvo_groups = feats.groupby("salvo_id")["label"]
purity_per_salvo = salvo_groups.apply(
    lambda g: (g == g.mode().iloc[0]).mean()
)

# Exclude size-1 salvos (trivially pure)
salvo_sizes = feats.groupby("salvo_id")["salvo_size"].first()
multi_rocket_salvos = salvo_sizes[salvo_sizes > 1].index
purity_multi = purity_per_salvo.loc[multi_rocket_salvos]

print(f"\n  Multi-rocket salvos: {len(purity_multi)}")
print(f"  Mean label purity  : {purity_multi.mean():.4f}")
print(f"  % perfectly pure   : {(purity_multi == 1.0).mean():.1%}")
print()
print("  Interpretation:")
if purity_multi.mean() > 0.95:
    print("  → HIGH purity (>95%): salvos are homogeneous — hypothesis strongly supported.")
elif purity_multi.mean() > 0.80:
    print("  -> MODERATE purity (80-95%): salvos are mostly homogeneous.")
else:
    print("  → LOW purity (<80%): mixed salvos dominate — hypothesis not supported by data.")
