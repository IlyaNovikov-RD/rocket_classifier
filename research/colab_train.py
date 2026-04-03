"""
Rocket Trajectory Classifier — Training Pipeline
=================================================
Google Colab · H100 GPU · LightGBM + Optuna

Problem
-------
Given radar-tracked 3D flight trajectories (x, y, z, time_stamp) from
classify each trajectory into one of
three rocket types. The evaluation metric is minimum per-class recall —
the system must perform well for every class, not just on average.

    score = min_{j ∈ {0,1,2}}  Σ 1[ŷᵢ = yᵢ = j] / Σ 1[yᵢ = j]

Approach
--------
LightGBM on 32 physics-informed + salvo-context features, with per-class
log-probability threshold tuning on out-of-bag predictions.

Assumptions
-----------
1. Flat terrain — z is absolute altitude, no correction needed.
2. Rockets follow frictionless ballistic physics — kinematic features are
   physically meaningful (velocity, acceleration, jerk via finite differences).
3. Business knowledge from domain expert:
   3a. Different launchers have different payload capacities.
   3b. Rockets are typically fired in salvos (together or with short delay).
   3c. Few rebel groups concentrated geographically; each buys independently
       → each group uses one rocket type (97.4% empirical salvo class purity).

Result
------
OOB min-recall: 0.999911  |  Biases: [0, 0.759, 0.658]
Oracle analysis (colab_analysis.py) confirms this is the practical ceiling.

Files to upload to /content/:
    REQUIRED: train.csv
    OPTIONAL: cache_train_features.parquet
        Skips the ~2 min kinematic feature computation step.
        Download from the GitHub Release: make download-all

Setup (first Colab cell):
    !pip install -q lightgbm optuna scikit-learn pandas numpy pyarrow

Expected runtime on H100: ~45 min (100 Optuna trials + 10-fold final CV)
"""

# %%
from __future__ import annotations

import json
import logging
import os
import subprocess as _sp
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
warnings.filterwarnings("ignore")

_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
_h = logging.StreamHandler()
_h.setFormatter(_fmt)
logging.getLogger().handlers = [_h]
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger("train")

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path("/content")
TRAIN_CSV = DATA_DIR / "train.csv"
FEATURE_CACHE = DATA_DIR / "cache_train_features.parquet"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

N_CLASSES = 3
N_SPLITS = 10  # CV folds for Optuna and final evaluation
N_TRIALS = 100  # Optuna trials (early stopping at 30 without improvement)
RANDOM_SEED = 42

# DBSCAN parameters validated across 3 independent feature-selection runs
SALVO_EPS, SALVO_MIN_SAMPLES = 0.5, 2  # spatiotemporal (assumption 3b)
GROUP_EPS, GROUP_MIN_SAMPLES = 0.25, 3  # pure-spatial   (assumption 3c)

# 32 features selected by backward elimination on the full training set.
# 25 kinematic features (assumptions 1 & 2) + 7 salvo/group features (assumption 3).
# Validated across 3 independent elimination runs — the same set survives every time.
SELECTED_FEATURES: list[str] = [
    # ── Kinematic (25) ─────────────────────────────────────────────────────────
    "n_points",
    "vy_mean",
    "vy_max",
    "vz_median",
    "v_horiz_std",
    "v_horiz_median",
    "initial_speed",
    "initial_vz",
    "final_vz",
    "acc_mag_mean",
    "acc_mag_min",
    "acc_mag_max",
    "az_std",
    "acc_horiz_std",
    "acc_horiz_min",
    "acc_horiz_max",
    "mean_az",
    "initial_z",
    "final_z",
    "delta_z_total",
    "apogee_relative",
    "x_range",
    "y_range",
    "launch_x",
    "launch_y",
    # ── Salvo (4) — assumption 3b ──────────────────────────────────────────────
    "salvo_size",
    "salvo_duration_s",
    "salvo_spatial_spread_m",
    "salvo_time_rank",
    # ── Rebel-group (3) — assumption 3a + 3c ──────────────────────────────────
    "group_total_rockets",
    "group_n_salvos",
    "group_max_salvo_size",
]

try:
    _sp.run(["nvidia-smi"], check=True, capture_output=True)
    DEVICE = "gpu"
except Exception:
    DEVICE = "cpu"
N_JOBS = int(os.cpu_count() or 4)
log.info("Device: %s | CPU cores: %d", DEVICE, N_JOBS)

# ── Utilities ──────────────────────────────────────────────────────────────────


def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official evaluation metric: minimum per-class recall."""
    return float(
        min(
            np.sum((y_pred == c) & (y_true == c)) / max(np.sum(y_true == c), 1)
            for c in range(N_CLASSES)
        )
    )


def optimize_thresholds(
    y: np.ndarray,
    proba: np.ndarray,
    g1: int = 80,
    g2: int = 50,
    g3: int = 30,
) -> tuple[np.ndarray, float]:
    """Coarse → fine → ultra-fine grid search over log-probability biases.

    Returns (biases, score) where biases[0] is fixed at 0.
    """
    lp = np.log(proba + 1e-12)
    best_s, best_b = -1.0, np.zeros(N_CLASSES)
    for b1 in np.linspace(-4, 4, g1):
        for b2 in np.linspace(-4, 4, g1):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    for b1 in np.linspace(best_b[1] - 0.12, best_b[1] + 0.12, g2):
        for b2 in np.linspace(best_b[2] - 0.12, best_b[2] + 0.12, g2):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    for b1 in np.linspace(best_b[1] - 0.02, best_b[1] + 0.02, g3):
        for b2 in np.linspace(best_b[2] - 0.02, best_b[2] + 0.02, g3):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    return best_b, best_s


def impute_nan(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    mask = np.isnan(X)
    if mask.any():
        for col in np.where(mask.any(axis=0))[0]:
            X[mask[:, col], col] = medians[col]
    return X


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights: w_i = N / (K · N_j)."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts, strict=True))
    return np.array([len(y) / (len(classes) * freq[c]) for c in y], dtype=np.float32)


def compute_group_priors(
    group_ids: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """P(class=k | rebel_group=g) from training data.

    Computed per CV fold (training partition only) so no label leaks to
    the validation fold.  Groups with < 3 members fall back to global prior.
    """
    gp = np.array([(y == c).mean() for c in range(N_CLASSES)], dtype=np.float32)
    priors: dict[int, np.ndarray] = {}
    for g in np.unique(group_ids):
        mask = group_ids == g
        priors[int(g)] = (
            np.array([(y[mask] == c).mean() for c in range(N_CLASSES)], dtype=np.float32)
            if mask.sum() >= 3
            else gp.copy()
        )
    return priors, gp


def append_group_priors(
    X: np.ndarray,
    group_ids: np.ndarray,
    priors: dict[int, np.ndarray],
    gp: np.ndarray,
) -> np.ndarray:
    """Append 3 group-class-prior columns to X (assumption 3c — independent procurement)."""
    cols = np.array([priors.get(int(g), gp) for g in group_ids], dtype=np.float32)
    return np.concatenate([X, cols], axis=1)


def make_lgbm(params: dict) -> LGBMClassifier:
    return LGBMClassifier(
        **params,
        objective="multiclass",
        num_class=N_CLASSES,
        device=DEVICE,
        gpu_use_dp=False,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


# ── Stage 0: Feature Engineering ──────────────────────────────────────────────

log.info("=" * 70)
log.info("STAGE 0: FEATURE ENGINEERING")
log.info("=" * 70)

log.info("Loading %s ...", TRAIN_CSV)
df_raw = pd.read_csv(TRAIN_CSV)
df_raw["time_stamp"] = pd.to_datetime(df_raw["time_stamp"], format="mixed")

launch_meta = (
    df_raw.sort_values("time_stamp")
    .groupby("traj_ind", sort=False)
    .agg(launch_time=("time_stamp", "first"), label=("label", "first"))
    .reset_index()
)
log.info(
    "Trajectories: %d | Labels: 0=%d  1=%d  2=%d",
    len(launch_meta),
    (launch_meta["label"] == 0).sum(),
    (launch_meta["label"] == 1).sum(),
    (launch_meta["label"] == 2).sum(),
)

# ── Inline kinematic feature engineering ─────────────────────────────────────
# Self-contained implementation — no external package needed.
# Mirrors rocket_classifier/features.py exactly.


def _compute_derivatives(
    pos: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _safe_stats(arr: np.ndarray, prefix: str) -> dict:
    if arr.size == 0:
        return {f"{prefix}_{s}": np.nan for s in ["mean", "std", "min", "max", "median"]}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }


def _extract_trajectory_features(group: pd.DataFrame) -> dict:
    feats: dict = {}
    pos = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    times = group["time_stamp"].to_numpy(dtype="datetime64[ns]")
    n = len(pos)
    feats["n_points"] = float(n)

    dt_ns = np.diff(times).astype(np.float64)
    dt_sec = dt_ns / 1e9
    dt_sec = np.where(dt_sec <= 0, np.nan, dt_sec)
    valid_dt = dt_sec[~np.isnan(dt_sec)]

    feats["total_duration_s"] = float(np.nansum(dt_sec)) if len(dt_sec) > 0 else 0.0
    feats.update(_safe_stats(valid_dt, "dt"))

    if n >= 2 and valid_dt.size > 0:
        dt_filled = np.where(np.isnan(dt_sec), np.nanmedian(dt_sec), dt_sec)
        vel, acc, jerk = _compute_derivatives(pos, dt_filled)
        speed = np.linalg.norm(vel, axis=1)
        vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
        v_horiz = np.sqrt(vx**2 + vy**2)
        feats.update(_safe_stats(speed, "speed"))
        feats.update(_safe_stats(vx, "vx"))
        feats.update(_safe_stats(vy, "vy"))
        feats.update(_safe_stats(vz, "vz"))
        feats.update(_safe_stats(v_horiz, "v_horiz"))
        feats["launch_angle_elev"] = (
            float(np.arctan2(vz[0], np.sqrt(vx[0] ** 2 + vy[0] ** 2))) if speed[0] > 0 else np.nan
        )
        feats["launch_angle_azimuth"] = float(np.arctan2(vy[0], vx[0])) if speed[0] > 0 else np.nan
        feats["initial_speed"] = float(speed[0])
        feats["initial_vz"] = float(vz[0])
        feats["initial_v_horiz"] = float(v_horiz[0])
        feats["final_speed"] = float(speed[-1])
        feats["final_vz"] = float(vz[-1])
        if acc.shape[0] > 0:
            acc_mag = np.linalg.norm(acc, axis=1)
            ax, ay, az_arr = acc[:, 0], acc[:, 1], acc[:, 2]
            feats.update(_safe_stats(acc_mag, "acc_mag"))
            feats.update(_safe_stats(az_arr, "az"))
            feats.update(_safe_stats(np.sqrt(ax**2 + ay**2), "acc_horiz"))
            feats["mean_az"] = float(np.mean(az_arr))
        else:
            for k in [
                "acc_mag_mean",
                "acc_mag_std",
                "acc_mag_min",
                "acc_mag_max",
                "acc_mag_median",
                "az_mean",
                "az_std",
                "az_min",
                "az_max",
                "az_median",
                "acc_horiz_mean",
                "acc_horiz_std",
                "acc_horiz_min",
                "acc_horiz_max",
                "acc_horiz_median",
                "mean_az",
            ]:
                feats[k] = np.nan
        if jerk.shape[0] > 0:
            feats.update(_safe_stats(np.linalg.norm(jerk, axis=1), "jerk_mag"))
        else:
            for k in [
                "jerk_mag_mean",
                "jerk_mag_std",
                "jerk_mag_min",
                "jerk_mag_max",
                "jerk_mag_median",
            ]:
                feats[k] = np.nan
    else:
        for pfx in ["speed", "vx", "vy", "vz", "v_horiz", "acc_mag", "az", "acc_horiz", "jerk_mag"]:
            for s in ["mean", "std", "min", "max", "median"]:
                feats[f"{pfx}_{s}"] = np.nan
        for k in [
            "launch_angle_elev",
            "launch_angle_azimuth",
            "initial_speed",
            "initial_vz",
            "initial_v_horiz",
            "final_speed",
            "final_vz",
            "mean_az",
        ]:
            feats[k] = np.nan

    z_vals = pos[:, 2]
    ap_idx = int(np.argmax(z_vals))
    feats["apogee_z"] = float(z_vals[ap_idx])
    feats["initial_z"] = float(z_vals[0])
    feats["final_z"] = float(z_vals[-1])
    feats["delta_z_total"] = float(z_vals[-1] - z_vals[0])
    feats["apogee_relative"] = float(z_vals[ap_idx] - z_vals[0])
    feats["apogee_time_frac"] = float(ap_idx / max(n - 1, 1))
    feats["time_to_apogee_s"] = (
        float((times[ap_idx] - times[0]).astype(np.float64) / 1e9) if ap_idx > 0 else 0.0
    )
    feats["x_range"] = float(np.ptp(pos[:, 0]))
    feats["y_range"] = float(np.ptp(pos[:, 1]))
    feats["z_range"] = float(np.ptp(pos[:, 2]))
    xy_disp = np.sqrt((pos[:, 0] - pos[0, 0]) ** 2 + (pos[:, 1] - pos[0, 1]) ** 2)
    feats["max_horiz_range"] = float(np.max(xy_disp))
    feats["final_horiz_range"] = float(xy_disp[-1])
    feats["path_length_3d"] = (
        float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if n >= 2 else 0.0
    )
    feats["launch_x"] = float(pos[0, 0])
    feats["launch_y"] = float(pos[0, 1])
    feats["launch_z"] = float(pos[0, 2])
    return feats


def build_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 76-feature kinematic matrix — one row per trajectory."""
    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")
    df = df.sort_values(["traj_ind", "time_stamp"])
    has_label = "label" in df.columns
    records = []
    for traj_id, group in df.groupby("traj_ind", sort=False):
        f = _extract_trajectory_features(group.reset_index(drop=True))
        f["traj_ind"] = traj_id
        if has_label:
            f["label"] = int(group["label"].iloc[0])
        records.append(f)
    return pd.DataFrame(records).set_index("traj_ind")


# Load kinematic features from cache or compute inline
if FEATURE_CACHE.exists():
    feats = pd.read_parquet(FEATURE_CACHE)
    log.info("Loaded kinematic features from cache: %s", feats.shape)
else:
    log.info("Computing kinematic features from train.csv (~2 min on H100)...")
    feats = build_kinematic_features(df_raw)
    feats.to_parquet(FEATURE_CACHE)
    log.info("Kinematic features computed and cached: %s", feats.shape)

feats = feats.join(launch_meta.set_index("traj_ind")[["launch_time"]], how="left")

# ── Salvo clustering — assumption 3b ─────────────────────────────────────────
ci = pd.DataFrame(
    {
        "launch_x": feats["launch_x"],
        "launch_y": feats["launch_y"],
        "launch_time_s": feats["launch_time"].astype(np.int64) / 1e9,
    }
).fillna(0.0)
raw_salvo = DBSCAN(eps=SALVO_EPS, min_samples=SALVO_MIN_SAMPLES, n_jobs=-1).fit_predict(
    StandardScaler().fit_transform(ci)
)
next_id = int(raw_salvo.max()) + 1
salvo_ids = raw_salvo.copy()
for i in range(len(salvo_ids)):
    if salvo_ids[i] == -1:
        salvo_ids[i] = next_id
        next_id += 1
feats["salvo_id"] = salvo_ids
n_salvos = len(set(raw_salvo[raw_salvo >= 0]))
log.info("Salvos: %d identified | noise: %d", n_salvos, (raw_salvo == -1).sum())

salvo_rows = []
for _sid, grp in feats.groupby("salvo_id"):
    n = len(grp)
    lx, ly = grp["launch_x"].values, grp["launch_y"].values
    lt = grp["launch_time"].values.astype(np.int64) / 1e9
    spread = dur = 0.0
    if n > 1:
        dur = float(lt.max() - lt.min())
        if n <= 1000:
            dx = lx[:, None] - lx[None, :]
            dy = ly[:, None] - ly[None, :]
            spread = float(np.sqrt(dx**2 + dy**2).max())
        else:
            spread = float(np.sqrt((lx.max() - lx.min()) ** 2 + (ly.max() - ly.min()) ** 2))
    ranks = pd.Series(lt).rank(method="first").astype(int).values
    for i, tid in enumerate(grp.index):
        salvo_rows.append(
            {
                "traj_ind": tid,
                "salvo_size": n,
                "salvo_duration_s": dur,
                "salvo_spatial_spread_m": spread,
                "salvo_time_rank": int(ranks[i]),
            }
        )
feats = feats.join(pd.DataFrame(salvo_rows).set_index("traj_ind"), how="left")

# ── Rebel-group clustering — assumption 3c ───────────────────────────────────
sp_in = feats[["launch_x", "launch_y"]].fillna(0.0)
sp_sc = StandardScaler().fit_transform(sp_in)
raw_group = DBSCAN(eps=GROUP_EPS, min_samples=GROUP_MIN_SAMPLES, n_jobs=-1).fit_predict(sp_sc)

# Auto-tune GROUP_EPS if fewer than 2 groups found
n_groups = len(set(raw_group[raw_group >= 0]))
if n_groups < 2:
    log.warning("Only %d group(s) found — auto-tuning GROUP_EPS...", n_groups)
    for eps_try in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]:
        lbls = DBSCAN(eps=eps_try, min_samples=GROUP_MIN_SAMPLES, n_jobs=-1).fit_predict(sp_sc)
        ng = len(set(lbls[lbls >= 0]))
        if 2 <= ng <= 20:
            raw_group = lbls
            n_groups = ng
            log.info("Auto-selected GROUP_EPS=%.2f → %d groups", eps_try, n_groups)
            break

next_gid = int(raw_group.max()) + 1
group_ids_arr = raw_group.copy()
for i in range(len(group_ids_arr)):
    if group_ids_arr[i] == -1:
        group_ids_arr[i] = next_gid
        next_gid += 1
feats["rebel_group_id"] = group_ids_arr
log.info("Rebel groups: %d identified | noise: %d", n_groups, (raw_group == -1).sum())

group_stats = feats.groupby("rebel_group_id").agg(
    group_total_rockets=("rebel_group_id", "count"),
    group_n_salvos=("salvo_id", "nunique"),
    group_max_salvo_size=("salvo_size", "max"),
)
feats = feats.join(group_stats, on="rebel_group_id")

# Salvo purity — empirical validation of assumption 3c
multi_mask = feats["salvo_size"] > 1
if multi_mask.sum() > 0:
    purity = (
        feats[multi_mask]
        .groupby("salvo_id")["label"]
        .apply(lambda g: (g == g.mode().iloc[0]).mean())
    )
    log.info(
        "Salvo class purity: mean=%.3f  pure100%%=%.1f%%",
        purity.mean(),
        (purity == 1.0).mean() * 100,
    )

# ── Prepare arrays ────────────────────────────────────────────────────────────
y = feats["label"].to_numpy(dtype=int)
cv_groups = feats.index.to_numpy(dtype=int)
rebel_group_ids = feats["rebel_group_id"].to_numpy(dtype=int)
X = feats.reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32)
log.info(
    "Feature matrix: %s | Classes: 0=%d  1=%d  2=%d",
    X.shape,
    (y == 0).sum(),
    (y == 1).sum(),
    (y == 2).sum(),
)

# ── Stage 1: Optuna Hyperparameter Search ─────────────────────────────────────

log.info("")
log.info("=" * 70)
log.info("STAGE 1: OPTUNA (%d trials, %d-fold GroupKFold)", N_TRIALS, N_SPLITS)
log.info("=" * 70)

gkf = GroupKFold(n_splits=N_SPLITS)
EARLY_STOP_PATIENCE = 30


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2500),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
    }
    oob_p = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    fold_tuned: list[float] = []

    for fold_idx, (tr, val) in enumerate(gkf.split(X, y, cv_groups)):
        Xtr, Xv = X[tr].copy(), X[val].copy()
        med = np.nanmedian(Xtr, axis=0)
        impute_nan(Xtr, med)
        impute_nan(Xv, med)
        sw = sample_weights(y[tr])
        priors, gp = compute_group_priors(rebel_group_ids[tr], y[tr])
        Xtr = append_group_priors(Xtr, rebel_group_ids[tr], priors, gp)
        Xv = append_group_priors(Xv, rebel_group_ids[val], priors, gp)
        clf = make_lgbm(params)
        clf.fit(Xtr, y[tr], sample_weight=sw)
        oob_p[val] = clf.predict_proba(Xv)

        # Prune after fold 2 if clearly below median
        if fold_idx == 1:
            _, fold_s = optimize_thresholds(y[val], oob_p[val], g1=50, g2=30, g3=20)
            fold_tuned.append(fold_s)
            completed = [t.value for t in trial.study.trials if t.value is not None]
            if len(completed) > 5:
                import statistics

                if fold_s < statistics.median(completed) - 0.002:
                    raise optuna.TrialPruned()

    raw = min_class_recall(y, np.argmax(np.log(oob_p + 1e-12), axis=1))
    _, tuned = optimize_thresholds(y, oob_p)
    mean_fold_tuned = float(np.mean(fold_tuned)) if fold_tuned else tuned

    score = raw + 0.001 * tuned  # raw is primary; tuned breaks ties
    log.info(
        "Trial %3d | raw=%.4f | tuned=%.4f | fold_mean=%.4f",
        trial.number,
        raw,
        tuned,
        mean_fold_tuned,
    )
    return score


def early_stop(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < EARLY_STOP_PATIENCE:
        return
    scores = [t.value for t in completed]
    if max(scores[-EARLY_STOP_PATIENCE:]) < max(scores):
        log.info(
            "Early stopping: best=%.6f unchanged for %d trials.",
            max(scores),
            EARLY_STOP_PATIENCE,
        )
        study.stop()


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
)
study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
    catch=(Exception,),
    callbacks=[early_stop],
)

best = study.best_trial
best_params = dict(best.params)
log.info("")
log.info("Optuna complete. Best trial #%d: score=%.6f", best.number, best.value)
log.info("Best params: %s", best_params)

# Top 5 trials
log.info("Top 5 trials:")
for t in sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]:
    if t.value is not None:
        log.info("  #%3d  score=%.6f", t.number, t.value)

# ── Stage 2: Final Model + OOB Threshold Tuning + 100%% Retrain ───────────────

log.info("")
log.info("=" * 70)
log.info("STAGE 2: FINAL MODEL — OOB THRESHOLD TUNING + 100%% RETRAIN")
log.info("=" * 70)

# OOB evaluation with best params
oob_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)
for tr, val in gkf.split(X, y, cv_groups):
    Xtr, Xv = X[tr].copy(), X[val].copy()
    med = np.nanmedian(Xtr, axis=0)
    impute_nan(Xtr, med)
    impute_nan(Xv, med)
    sw = sample_weights(y[tr])
    priors, gp = compute_group_priors(rebel_group_ids[tr], y[tr])
    Xtr = append_group_priors(Xtr, rebel_group_ids[tr], priors, gp)
    Xv = append_group_priors(Xv, rebel_group_ids[val], priors, gp)
    clf = make_lgbm(best_params)
    clf.fit(Xtr, y[tr], sample_weight=sw)
    oob_proba[val] = clf.predict_proba(Xv)

biases, oob_score = optimize_thresholds(y, oob_proba)
log.info("OOB global threshold-tuned score: %.6f", oob_score)
log.info("Optimal biases: [%.6f, %.6f, %.6f]", *biases)

# Per-fold breakdown
for fold_idx, (_, val) in enumerate(gkf.split(X, y, cv_groups)):
    preds = np.argmax(np.log(oob_proba[val] + 1e-12) + biases, axis=1)
    fold_score = min_class_recall(y[val], preds)
    log.info("  Fold %2d: %.4f", fold_idx + 1, fold_score)

# 100% retrain
log.info("Retraining on 100%% of data...")
X_full = X.copy()
med_full = np.nanmedian(X_full, axis=0)
impute_nan(X_full, med_full)
sw_full = sample_weights(y)
priors_full, gp_full = compute_group_priors(rebel_group_ids, y)
X_full_aug = append_group_priors(X_full, rebel_group_ids, priors_full, gp_full)

final_clf = make_lgbm(best_params)
final_clf.fit(X_full_aug, y, sample_weight=sw_full)

# Train recall sanity check
train_preds = np.argmax(np.log(final_clf.predict_proba(X_full_aug) + 1e-12) + biases, axis=1)
log.info("Train recall (sanity): %.6f", min_class_recall(y, train_preds))

# Save artifacts
ARTIFACTS_DIR.mkdir(exist_ok=True)
final_clf.booster_.save_model(str(ARTIFACTS_DIR / "model.lgb"))
np.save(str(ARTIFACTS_DIR / "train_medians.npy"), med_full)
np.save(str(ARTIFACTS_DIR / "threshold_biases.npy"), biases)
with open(ARTIFACTS_DIR / "results.json", "w") as f:
    json.dump(
        {
            "oob_score": float(oob_score),
            "biases": biases.tolist(),
            "best_params": best_params,
            "selected_features": SELECTED_FEATURES,
            "n_features": len(SELECTED_FEATURES),
        },
        f,
        indent=2,
    )

log.info("")
log.info("Artifacts saved to %s:", ARTIFACTS_DIR)
log.info("  model.lgb  |  train_medians.npy  |  threshold_biases.npy  |  results.json")
log.info("")
log.info("=" * 70)
log.info("  OOB min-recall: %.6f", oob_score)
log.info("  Biases:         %s", biases.round(6).tolist())
log.info("=" * 70)
