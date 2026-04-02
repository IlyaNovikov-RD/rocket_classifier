"""
FINAL CEILING-BREAKER EXPERIMENT — Google Colab (H100 GPU, 8+ hours)
=====================================================================

Goal: Break the 0.9995 global OOB min-recall ceiling.

Strategy (4 stages):

  Stage 1 (~45 min): Extended feature engineering
    - 76 base + 66 extended (previous) + ~150 NEW features = ~290 total
    - NEW: quintile segments, pre/post apogee asymmetry, energy conservation,
      FFT spectral, cross-axis correlations, torsion, trajectory complexity,
      heading dynamics, rolling variance, elevation fit residuals
    - Backward elimination with zero-tolerance threshold

  Stage 2 (~5.5 hours): Massive Optuna search — 500+ trials
    - LightGBM (gbdt + dart + goss), XGBoost, CatBoost
    - Global OOB min-recall as objective (not per-fold mean)
    - Full OOB probability arrays stored for every completed trial
    - Time-budget enforcement: stops gracefully if approaching 6h

  Stage 3 (~1 hour): Diverse ensemble from stored OOB arrays
    - Greedy forward selection maximising ensemble OOB min-recall
    - Weighted averaging with scipy.optimize
    - Stacking meta-learner (logistic regression + small LightGBM)
    - Cross-algorithm ensembles

  Stage 4 (~45 min): Ultra-fine threshold optimisation
    - 4-pass grid search (coarse->fine->ultra->hyper: up to 500x500)
    - Isotonic calibration before thresholding
    - Temperature scaling
    - scipy differential_evolution on 3-bias space
    - Per-sample margin analysis on hard cases

Install (run first):
    !pip install -q optuna catboost lightgbm xgboost scikit-learn pandas numpy pyarrow scipy

Upload to /content/ before running:
    - cache_train_features.parquet  (76 features + label, indexed by traj_ind)
    - train.csv                     (raw point-level data for extended features)

Usage:
    %run colab_final_ceiling_breaker.py
"""

from __future__ import annotations

import gc
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
from scipy.optimize import differential_evolution, minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(_handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger("ceiling-breaker")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PARQUET_PATH = Path("/content/cache_train_features.parquet")
RAW_CSV_PATH = Path("/content/train.csv")
CHECKPOINT_DIR = Path("/content/ceiling_breaker_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

N_SPLITS = 10
N_OPTUNA_TRIALS = 500
RANDOM_STATE = 42
N_CLASSES = 3
TOTAL_TIME_BUDGET_S = 9 * 3600  # 9 hours hard cap
STAGE2_TIME_BUDGET_S = 5.5 * 3600  # 5.5 hours for Optuna

# Threshold grid sizes (progressive refinement)
GRID_COARSE = 100
GRID_FINE = 100
GRID_ULTRA = 200
GRID_HYPER = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

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


def optimize_thresholds_4pass(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    """4-pass progressive threshold optimisation (coarse → fine → ultra → hyper)."""
    lp = np.log(proba + 1e-12)
    best_s, best_b = -1.0, np.zeros(N_CLASSES)

    # Pass 1: Coarse (100x100 on [-5, 5])
    for b1 in np.linspace(-5, 5, GRID_COARSE):
        for b2 in np.linspace(-5, 5, GRID_COARSE):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    # Pass 2: Fine (100x100 on [best +/- 0.15])
    for b1 in np.linspace(best_b[1] - 0.15, best_b[1] + 0.15, GRID_FINE):
        for b2 in np.linspace(best_b[2] - 0.15, best_b[2] + 0.15, GRID_FINE):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    # Pass 3: Ultra (200x200 on [best +/- 0.02])
    for b1 in np.linspace(best_b[1] - 0.02, best_b[1] + 0.02, GRID_ULTRA):
        for b2 in np.linspace(best_b[2] - 0.02, best_b[2] + 0.02, GRID_ULTRA):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    # Pass 4: Hyper (500x500 on [best +/- 0.005])
    for b1 in np.linspace(best_b[1] - 0.005, best_b[1] + 0.005, GRID_HYPER):
        for b2 in np.linspace(best_b[2] - 0.005, best_b[2] + 0.005, GRID_HYPER):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    return best_b, best_s


def optimize_thresholds_quick(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    """Quick 2-pass threshold optimisation for inner loops."""
    lp = np.log(proba + 1e-12)
    best_s, best_b = -1.0, np.zeros(N_CLASSES)
    for b1 in np.linspace(-4, 4, 80):
        for b2 in np.linspace(-4, 4, 80):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    for b1 in np.linspace(best_b[1] - 0.12, best_b[1] + 0.12, 50):
        for b2 in np.linspace(best_b[2] - 0.12, best_b[2] + 0.12, 50):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()
    return best_b, best_s


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Extended Feature Engineering (~150 new features)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    if arr.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_min": np.nan, f"{prefix}_max": np.nan,
                f"{prefix}_median": np.nan}
    return {f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_min": float(np.min(arr)), f"{prefix}_max": float(np.max(arr)),
            f"{prefix}_median": float(np.median(arr))}


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
        return {f"{prefix}_early": np.nan, f"{prefix}_mid": np.nan,
                f"{prefix}_late": np.nan}
    sp = np.array_split(arr, 3)
    return {f"{prefix}_early": float(np.mean(sp[0])),
            f"{prefix}_mid": float(np.mean(sp[1])),
            f"{prefix}_late": float(np.mean(sp[2]))}


def _quintile_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    """5-segment decomposition — higher resolution than 3-phase."""
    if arr.size < 5:
        return {f"{prefix}_q{i}": np.nan for i in range(1, 6)}
    parts = np.array_split(arr, 5)
    return {f"{prefix}_q{i}": float(np.mean(p)) for i, p in enumerate(parts, 1)}


def _pre_post_apogee(arr: np.ndarray, apogee_idx: int, prefix: str) -> dict[str, float]:
    """Split array at apogee; compare ascending vs descending phase."""
    pre = arr[:apogee_idx + 1] if apogee_idx > 0 else np.array([])
    post = arr[apogee_idx:] if apogee_idx < len(arr) - 1 else np.array([])
    f = {}
    f[f"{prefix}_pre_mean"] = float(np.mean(pre)) if pre.size > 0 else np.nan
    f[f"{prefix}_post_mean"] = float(np.mean(post)) if post.size > 0 else np.nan
    if pre.size > 0 and post.size > 0 and np.mean(post) != 0:
        f[f"{prefix}_pre_post_ratio"] = float(np.mean(pre) / (np.mean(post) + 1e-12))
    else:
        f[f"{prefix}_pre_post_ratio"] = np.nan
    return f


def _fft_features(arr: np.ndarray, prefix: str, n_bins: int = 3) -> dict[str, float]:
    """Frequency-domain features from FFT of a time series."""
    f = {}
    if arr.size < 8:
        for key in [f"{prefix}_fft_dominant_freq", f"{prefix}_fft_spectral_entropy"] + \
                   [f"{prefix}_fft_bin{i}" for i in range(n_bins)]:
            f[key] = np.nan
        return f

    # Zero-pad to next power of 2 for efficiency
    n = len(arr)
    fft_vals = np.abs(np.fft.rfft(arr - np.mean(arr)))
    freqs = np.fft.rfftfreq(n)

    # Dominant frequency (excluding DC)
    if len(fft_vals) > 1:
        f[f"{prefix}_fft_dominant_freq"] = float(freqs[1 + np.argmax(fft_vals[1:])])
    else:
        f[f"{prefix}_fft_dominant_freq"] = np.nan

    # Energy in first n_bins frequency bins
    for i in range(n_bins):
        if i + 1 < len(fft_vals):
            f[f"{prefix}_fft_bin{i}"] = float(fft_vals[i + 1] ** 2)
        else:
            f[f"{prefix}_fft_bin{i}"] = np.nan

    # Spectral entropy
    power = fft_vals[1:] ** 2
    total = power.sum()
    if total > 0:
        p = power / total
        p = p[p > 0]
        f[f"{prefix}_fft_spectral_entropy"] = float(-np.sum(p * np.log(p)))
    else:
        f[f"{prefix}_fft_spectral_entropy"] = np.nan

    return f


def _rolling_variance(arr: np.ndarray, prefix: str, window: int = 5) -> dict[str, float]:
    """Rolling variance statistics — captures local smoothness."""
    if arr.size < window + 1:
        return {f"{prefix}_rollvar_mean": np.nan, f"{prefix}_rollvar_max": np.nan}
    rv = np.array([np.var(arr[i:i + window]) for i in range(len(arr) - window + 1)])
    return {f"{prefix}_rollvar_mean": float(np.mean(rv)),
            f"{prefix}_rollvar_max": float(np.max(rv))}


def extract_new_features(group: pd.DataFrame) -> dict[str, float]:
    """Compute ~150 NEW features beyond the existing 76+66."""
    f: dict[str, float] = {}

    pos = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    times = group["time_stamp"].to_numpy(dtype="datetime64[ns]")
    n = len(pos)
    z = pos[:, 2]
    apogee_idx = int(np.argmax(z))

    dt_ns = np.diff(times).astype(np.float64)
    dt_s = np.where(dt_ns / 1e9 <= 0, np.nan, dt_ns / 1e9)
    vdt = dt_s[~np.isnan(dt_s)]

    # Initialise all features to NaN for the not-enough-data path
    _nan = np.nan

    if n >= 2 and vdt.size > 0:
        dtf = np.where(np.isnan(dt_s), np.nanmedian(dt_s), dt_s)
        vel, acc, jerk = _compute_derivatives(pos, dtf)
        speed = np.linalg.norm(vel, axis=1)
        vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
        v_horiz = np.sqrt(vx**2 + vy**2)

        # ── Quintile segment features ──────────────────────────────────────
        for arr, pfx in [(speed, "nspeed"), (vz, "nvz"), (v_horiz, "nvh")]:
            f.update(_quintile_stats(arr, pfx))

        # ── Pre/post apogee asymmetry ──────────────────────────────────────
        # Map apogee_idx from position space to velocity space (1 shorter)
        vel_apogee = min(apogee_idx, len(speed) - 1)
        for arr, pfx in [(speed, "nspeed"), (vz, "nvz")]:
            f.update(_pre_post_apogee(arr, vel_apogee, pfx))

        # ── First-10% / Last-10% extremes ─────────────────────────────────
        n_vel = len(speed)
        n10 = max(1, n_vel // 10)
        f["nspeed_first10_mean"] = float(np.mean(speed[:n10]))
        f["nspeed_first10_std"] = float(np.std(speed[:n10])) if n10 > 1 else _nan
        f["nspeed_last10_mean"] = float(np.mean(speed[-n10:]))
        f["nspeed_last10_std"] = float(np.std(speed[-n10:])) if n10 > 1 else _nan
        f["nvz_first10_mean"] = float(np.mean(vz[:n10]))
        f["nvz_last10_mean"] = float(np.mean(vz[-n10:]))

        # ── Rolling variance ───────────────────────────────────────────────
        f.update(_rolling_variance(speed, "nspeed"))
        f.update(_rolling_variance(vz, "nvz"))

        # ── FFT / spectral features ───────────────────────────────────────
        f.update(_fft_features(speed, "nspeed"))
        f.update(_fft_features(vz, "nvz"))

        # ── Cross-axis correlations ────────────────────────────────────────
        if n_vel >= 3:
            f["ncorr_vx_vy"] = float(np.corrcoef(vx, vy)[0, 1]) if np.std(vx) > 0 and np.std(vy) > 0 else _nan
            f["ncorr_vx_vz"] = float(np.corrcoef(vx, vz)[0, 1]) if np.std(vx) > 0 and np.std(vz) > 0 else _nan
            f["ncorr_vy_vz"] = float(np.corrcoef(vy, vz)[0, 1]) if np.std(vy) > 0 and np.std(vz) > 0 else _nan
            f["ncorr_speed_z"] = float(np.corrcoef(speed, z[:n_vel])[0, 1]) if np.std(speed) > 0 else _nan
        else:
            f["ncorr_vx_vy"] = f["ncorr_vx_vz"] = f["ncorr_vy_vz"] = f["ncorr_speed_z"] = _nan

        # ── Velocity autocorrelation ───────────────────────────────────────
        if n_vel >= 6:
            speed_c = speed - np.mean(speed)
            var = np.var(speed)
            for lag in [1, 2, 5]:
                if lag < n_vel and var > 0:
                    f[f"nspeed_autocorr_lag{lag}"] = float(
                        np.mean(speed_c[:-lag] * speed_c[lag:]) / var
                    )
                else:
                    f[f"nspeed_autocorr_lag{lag}"] = _nan
        else:
            for lag in [1, 2, 5]:
                f[f"nspeed_autocorr_lag{lag}"] = _nan

        # ── Heading dynamics ───────────────────────────────────────────────
        if n_vel >= 2:
            headings = np.arctan2(vy, vx)
            dheading = np.diff(headings)
            # Wrap to [-pi, pi]
            dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
            f["nheading_total_change"] = float(np.sum(np.abs(dheading)))
            f["nheading_mean_rate"] = float(np.mean(np.abs(dheading)))
            f["nheading_std_rate"] = float(np.std(dheading))
        else:
            f["nheading_total_change"] = f["nheading_mean_rate"] = f["nheading_std_rate"] = _nan

        # ── Acceleration features (if available) ──────────────────────────
        if acc.shape[0] > 0:
            acc_mag = np.linalg.norm(acc, axis=1)
            az_arr = acc[:, 2]

            f.update(_quintile_stats(acc_mag, "nacc"))
            f.update(_rolling_variance(acc_mag, "nacc"))
            f.update(_fft_features(acc_mag, "nacc"))

            # Pre/post apogee for acceleration
            acc_apogee = min(apogee_idx, len(acc_mag) - 1)
            f.update(_pre_post_apogee(acc_mag, acc_apogee, "nacc"))

            # Cross-axis acceleration correlations
            ax, ay = acc[:, 0], acc[:, 1]
            if len(ax) >= 3:
                f["ncorr_ax_ay"] = float(np.corrcoef(ax, ay)[0, 1]) if np.std(ax) > 0 and np.std(ay) > 0 else _nan
                f["ncorr_ax_az"] = float(np.corrcoef(ax, az_arr)[0, 1]) if np.std(ax) > 0 and np.std(az_arr) > 0 else _nan
            else:
                f["ncorr_ax_ay"] = f["ncorr_ax_az"] = _nan

            # ── Velocity-acceleration alignment ────────────────────────────
            # Dot product of velocity and acceleration unit vectors
            min_len = min(len(vel) - 1, len(acc))
            if min_len > 0:
                v_seg = vel[:min_len]
                a_seg = acc[:min_len]
                v_norm = np.linalg.norm(v_seg, axis=1, keepdims=True)
                a_norm = np.linalg.norm(a_seg, axis=1, keepdims=True)
                v_norm = np.clip(v_norm, 1e-12, None)
                a_norm = np.clip(a_norm, 1e-12, None)
                alignment = np.sum((v_seg / v_norm) * (a_seg / a_norm), axis=1)
                f["nvel_acc_align_mean"] = float(np.mean(alignment))
                f["nvel_acc_align_std"] = float(np.std(alignment))
            else:
                f["nvel_acc_align_mean"] = f["nvel_acc_align_std"] = _nan

            # ── Acceleration symmetry ──────────────────────────────────────
            half = len(acc_mag) // 2
            if half > 0:
                f["nacc_symmetry"] = float(np.mean(acc_mag[:half]) / (np.mean(acc_mag[half:]) + 1e-12))
            else:
                f["nacc_symmetry"] = _nan

        else:
            # NaN for all acc-dependent new features
            for key in ["nacc_q1", "nacc_q2", "nacc_q3", "nacc_q4", "nacc_q5",
                        "nacc_rollvar_mean", "nacc_rollvar_max",
                        "nacc_fft_dominant_freq", "nacc_fft_spectral_entropy",
                        "nacc_fft_bin0", "nacc_fft_bin1", "nacc_fft_bin2",
                        "nacc_pre_mean", "nacc_post_mean", "nacc_pre_post_ratio",
                        "ncorr_ax_ay", "ncorr_ax_az",
                        "nvel_acc_align_mean", "nvel_acc_align_std", "nacc_symmetry"]:
                f[key] = _nan

        # ── Jerk features (if available) ──────────────────────────────────
        if jerk.shape[0] > 0:
            jerk_mag = np.linalg.norm(jerk, axis=1)
            f.update(_quintile_stats(jerk_mag, "njerk"))
            f.update(_fft_features(jerk_mag, "njerk"))
        else:
            for key in ["njerk_q1", "njerk_q2", "njerk_q3", "njerk_q4", "njerk_q5",
                        "njerk_fft_dominant_freq", "njerk_fft_spectral_entropy",
                        "njerk_fft_bin0", "njerk_fft_bin1", "njerk_fft_bin2"]:
                f[key] = _nan

        # ── Curvature: torsion ─────────────────────────────────────────────
        if vel.shape[0] >= 2 and acc.shape[0] >= 1 and jerk.shape[0] >= 1:
            min_len3 = min(vel.shape[0] - 2, acc.shape[0] - 1, jerk.shape[0])
            if min_len3 > 0:
                v3 = vel[:min_len3]
                a3 = acc[:min_len3]
                j3 = jerk[:min_len3]
                # Torsion = det([v, a, j]) / |v x a|^2
                cross_va = np.cross(v3, a3)
                cross_norm_sq = np.sum(cross_va**2, axis=1)
                det_vaj = np.sum(cross_va * j3, axis=1)
                valid = cross_norm_sq > 1e-12
                torsion = np.where(valid, det_vaj / cross_norm_sq, 0.0)
                f["ntorsion_mean"] = float(np.mean(torsion))
                f["ntorsion_std"] = float(np.std(torsion))
                f["ntorsion_max"] = float(np.max(np.abs(torsion)))
            else:
                f["ntorsion_mean"] = f["ntorsion_std"] = f["ntorsion_max"] = _nan
        else:
            f["ntorsion_mean"] = f["ntorsion_std"] = f["ntorsion_max"] = _nan

        # ── Straightness indices ───────────────────────────────────────────
        disp_3d = np.linalg.norm(pos[-1] - pos[0])
        segs = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        path_len = float(np.sum(segs))
        f["nstraightness_3d"] = disp_3d / path_len if path_len > 1e-12 else _nan

        disp_xy = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
        segs_xy = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
        path_xy = float(np.sum(segs_xy))
        f["nsinuosity_xy"] = disp_xy / path_xy if path_xy > 1e-12 else _nan

        # ── Energy features ────────────────────────────────────────────────
        ke = 0.5 * speed**2
        pe = 9.81 * z[:n_vel]  # proportional to mgh
        total_energy = ke + pe

        f["nke_mean"] = float(np.mean(ke))
        f["nke_std"] = float(np.std(ke))
        f["nke_delta"] = float(ke[-1] - ke[0])
        f["npe_mean"] = float(np.mean(pe))
        f["ntotal_energy_std"] = float(np.std(total_energy))
        f["ntotal_energy_range"] = float(np.ptp(total_energy))
        f["nke_pe_ratio"] = float(np.mean(ke) / (np.mean(pe) + 1e-12))

        # ── Elevation profile fit ──────────────────────────────────────────
        if n >= 4:
            t_norm = np.linspace(0, 1, n)
            coeffs = np.polyfit(t_norm, z, 2)
            z_fit = np.polyval(coeffs, t_norm)
            residuals = z - z_fit
            f["nzfit_a2"] = float(coeffs[0])  # quadratic coeff ≈ -g
            f["nzfit_a1"] = float(coeffs[1])  # linear coeff ≈ v0z
            f["nzfit_residual_mae"] = float(np.mean(np.abs(residuals)))
            f["nzfit_residual_max"] = float(np.max(np.abs(residuals)))
        else:
            f["nzfit_a2"] = f["nzfit_a1"] = f["nzfit_residual_mae"] = f["nzfit_residual_max"] = _nan

        # ── Trajectory complexity ──────────────────────────────────────────
        # Number of vz sign changes (ballistic = exactly 1 at apogee)
        if len(vz) >= 2:
            vz_signs = np.sign(vz)
            vz_sign_changes = np.sum(vz_signs[:-1] != vz_signs[1:])
            f["nvz_sign_changes"] = float(vz_sign_changes)
        else:
            f["nvz_sign_changes"] = _nan

        # Consecutive positive acceleration (speed increasing)
        if len(speed) >= 2:
            dspeed = np.diff(speed)
            max_consec = 0
            current = 0
            for ds in dspeed:
                if ds > 0:
                    current += 1
                    max_consec = max(max_consec, current)
                else:
                    current = 0
            f["nmax_consec_accel"] = float(max_consec)
        else:
            f["nmax_consec_accel"] = _nan

        # Time above half-max altitude
        half_max_z = z.max() / 2.0
        f["ntime_above_half_z"] = float(np.mean(z >= half_max_z))

        # Descent rate in final 10%
        n10_pos = max(1, n // 10)
        if n10_pos >= 2:
            z_final = z[-n10_pos:]
            dz_final = np.diff(z_final)
            f["ndescent_rate_final10"] = float(np.mean(dz_final))
        else:
            f["ndescent_rate_final10"] = _nan

    else:
        # Not enough points — NaN for everything
        # Count of new feature keys (we'll let the first valid trajectory define the set)
        pass  # Will be filled with NaN at DataFrame join time

    return f


def build_new_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~150 new features for all trajectories."""
    df = raw_df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")
    df = df.sort_values(["traj_ind", "time_stamp"])
    records = []
    for tid, g in df.groupby("traj_ind", sort=False):
        feat = extract_new_features(g.reset_index(drop=True))
        feat["traj_ind"] = tid
        records.append(feat)
    return pd.DataFrame(records).set_index("traj_ind")


# ── Previous extended features (66) — keep for completeness ────────────────

def _prev_phase_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    if arr.size < 3:
        return {f"{prefix}_early": np.nan, f"{prefix}_mid": np.nan, f"{prefix}_late": np.nan}
    sp = np.array_split(arr, 3)
    return {f"{prefix}_early": float(np.mean(sp[0])),
            f"{prefix}_mid": float(np.mean(sp[1])),
            f"{prefix}_late": float(np.mean(sp[2]))}


def extract_prev_extended_features(group: pd.DataFrame) -> dict[str, float]:
    """66 extended features from previous experiment (skew, kurt, phase, curvature, ratios)."""
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
            f.update(_prev_phase_stats(arr, pfx))

        if acc.shape[0] > 0:
            am = np.linalg.norm(acc, axis=1)
            az = acc[:, 2]
            ah = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2)
            for arr, pfx in [(am, "acc_mag"), (az, "az"), (ah, "acc_horiz")]:
                f.update(_higher_order_stats(arr, pfx))
                f.update(_prev_phase_stats(arr, pfx))
        else:
            for pfx in ["acc_mag", "az", "acc_horiz"]:
                for sfx in _nan_sfx:
                    f[pfx + sfx] = np.nan
            am = np.array([])

        if jerk.shape[0] > 0:
            jm = np.linalg.norm(jerk, axis=1)
            f.update(_higher_order_stats(jm, "jerk_mag"))
            f.update(_prev_phase_stats(jm, "jerk_mag"))
        else:
            for sfx in _nan_sfx:
                f["jerk_mag" + sfx] = np.nan
            jm = np.array([])

        f["speed_ratio"] = float(sp[-1] / sp[0]) if sp[0] > 0 else np.nan
        f["speed_change"] = float(sp[-1] - sp[0])
        f["vz_ratio"] = float(vz[-1] / vz[0]) if abs(vz[0]) > 1e-6 else np.nan
        f["acc_over_speed"] = float(np.mean(am) / (np.mean(sp) + 1e-9)) if am.size > 0 else np.nan
        f["jerk_over_acc"] = float(np.mean(jm) / (np.mean(am) + 1e-9)) if jm.size > 0 and am.size > 0 else np.nan

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


def build_prev_extended_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")
    df = df.sort_values(["traj_ind", "time_stamp"])
    records = []
    for tid, g in df.groupby("traj_ind", sort=False):
        feat = extract_prev_extended_features(g.reset_index(drop=True))
        feat["traj_ind"] = tid
        records.append(feat)
    return pd.DataFrame(records).set_index("traj_ind")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load 76 base + 66 prev extended + ~150 new = ~290 features."""
    logger.info("Loading base features from %s ...", PARQUET_PATH)
    base_df = pd.read_parquet(PARQUET_PATH)
    base_cols = [c for c in base_df.columns if c != "label"]
    y = base_df["label"].to_numpy(dtype=int)
    groups = base_df.index.to_numpy()

    # Previous extended features (66)
    prev_cache = CHECKPOINT_DIR / "cache_prev_extended.parquet"
    if prev_cache.exists():
        logger.info("Loading prev extended features from cache...")
        prev_df = pd.read_parquet(prev_cache)
    else:
        logger.info("Computing prev extended features (~3 min)...")
        raw = pd.read_csv(RAW_CSV_PATH)
        prev_df = build_prev_extended_features(raw)
        prev_df.to_parquet(prev_cache)

    # NEW features (~150)
    new_cache = CHECKPOINT_DIR / "cache_new_features.parquet"
    if new_cache.exists():
        logger.info("Loading new features from cache...")
        new_df = pd.read_parquet(new_cache)
    else:
        logger.info("Computing ~150 new features (~10 min)...")
        raw = pd.read_csv(RAW_CSV_PATH)
        new_df = build_new_features(raw)
        new_df.to_parquet(new_cache)
        logger.info("Cached new features to %s", new_cache)

    # Align indices
    prev_df = prev_df.reindex(base_df.index)
    new_df = new_df.reindex(base_df.index)

    prev_cols = list(prev_df.columns)
    new_cols = list(new_df.columns)
    all_cols = base_cols + prev_cols + new_cols

    X = np.hstack([
        base_df[base_cols].to_numpy(dtype=np.float32),
        prev_df[prev_cols].to_numpy(dtype=np.float32),
        new_df[new_cols].to_numpy(dtype=np.float32),
    ])

    logger.info(
        "Total: %d rows x %d features (%d base + %d prev + %d new) | "
        "Labels: 0=%d  1=%d  2=%d",
        X.shape[0], X.shape[1], len(base_cols), len(prev_cols), len(new_cols),
        (y == 0).sum(), (y == 1).sum(), (y == 2).sum(),
    )
    return X, y, groups, all_cols


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Feature Selection (backward elimination, zero tolerance)
# ═══════════════════════════════════════════════════════════════════════════════

def _cv_score_quick(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    mask: np.ndarray) -> float:
    """Quick 5-fold CV score for feature selection."""
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
    _, score = optimize_thresholds_quick(y, oob)
    return score


def run_feature_selection(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    logger.info("=" * 72)
    logger.info("STAGE 1: FEATURE SELECTION (%d features)", X.shape[1])
    logger.info("=" * 72)

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
    baseline = _cv_score_quick(X, y, groups, all_mask)
    logger.info("Baseline score (all %d features): %.6f", X.shape[1], baseline)

    current_mask = all_mask.copy()
    current_score = baseline
    dropped = []

    for idx in rank_order:
        if current_mask.sum() <= 15:
            break
        trial_mask = current_mask.copy()
        trial_mask[idx] = False
        trial_score = _cv_score_quick(X, y, groups, trial_mask)
        fname, imp = feature_names[idx], importances[idx]
        # ZERO tolerance — only drop if score does NOT decrease at all
        if trial_score >= current_score:
            current_mask = trial_mask
            current_score = trial_score
            dropped.append(fname)
            logger.info("  DROP %-40s (imp=%6.1f) -> %d feats, score=%.6f",
                        fname, imp, current_mask.sum(), current_score)
        else:
            logger.info("  KEEP %-40s (imp=%6.1f) -> %.6f",
                        fname, imp, trial_score)

    selected = [feature_names[i] for i in range(len(feature_names)) if current_mask[i]]
    logger.info("Selected %d / %d features (dropped %d)",
                len(selected), len(feature_names), len(dropped))
    logger.info("Final selection score: %.6f (baseline %.6f)", current_score, baseline)

    # Checkpoint
    np.save(CHECKPOINT_DIR / "feature_mask.npy", current_mask)
    with open(CHECKPOINT_DIR / "selected_features.json", "w") as fp:
        json.dump(selected, fp)

    return current_mask, selected


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Massive Optuna Search (500 trials, 3 algorithms, global OOB)
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(algo: str, params: dict) -> object:
    p = {k: v for k, v in params.items() if k not in ("algorithm", "boosting_type")}
    if algo == "xgboost":
        return XGBClassifier(
            **p, objective="multi:softprob", num_class=N_CLASSES,
            eval_metric="mlogloss", tree_method="hist", device="cuda",
            random_state=RANDOM_STATE, verbosity=0,
        )
    elif algo == "lightgbm":
        boosting = params.get("boosting_type", "gbdt")
        return LGBMClassifier(
            **p, boosting_type=boosting,
            objective="multiclass", num_class=N_CLASSES,
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


# Store all completed OOB arrays for ensemble building in Stage 3
OOB_STORE: list[tuple[int, dict, np.ndarray]] = []


def create_objective(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     start_time: float) -> callable:
    def objective(trial: optuna.Trial) -> float:
        # Time budget check
        elapsed = time.time() - start_time
        if elapsed > STAGE2_TIME_BUDGET_S:
            raise optuna.exceptions.OptunaError("Stage 2 time budget exceeded")

        # Algorithm selection: 60% LightGBM, 20% XGBoost, 20% CatBoost
        algo = trial.suggest_categorical(
            "algorithm",
            ["lightgbm"] * 6 + ["xgboost"] * 2 + ["catboost"] * 2,
        )

        hp: dict = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        # LightGBM-specific: boosting type (gbdt, dart, goss)
        if algo == "lightgbm":
            hp["boosting_type"] = trial.suggest_categorical(
                "boosting_type", ["gbdt", "gbdt", "gbdt", "dart", "goss"],
            )
            if hp["boosting_type"] == "dart":
                hp["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.5)
                hp["max_drop"] = trial.suggest_int("max_drop", 10, 100)
                hp["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.5)

        # 10-fold CV, collect global OOB
        gkf = GroupKFold(n_splits=N_SPLITS)
        oob_proba = np.zeros((len(y), N_CLASSES))
        fold_raw = []

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
            oob_proba[val] = proba
            fold_raw.append(min_class_recall(yv, np.argmax(proba, axis=1)))

            # Early pruning after fold 3
            if fold_idx >= 2:
                partial_filled = oob_proba[oob_proba.sum(axis=1) > 0]
                partial_y = y[oob_proba.sum(axis=1) > 0]
                _, partial_score = optimize_thresholds_quick(partial_y, partial_filled)
                trial.report(partial_score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # GLOBAL OOB min-recall (the true metric)
        biases, global_oob_score = optimize_thresholds_quick(y, oob_proba)

        # Per-fold scores for reporting
        fold_tuned = []
        for _, val in gkf.split(X, y, groups):
            preds = np.argmax(np.log(oob_proba[val] + 1e-12) + biases, axis=1)
            fold_tuned.append(min_class_recall(y[val], preds))

        trial.set_user_attr("global_oob", global_oob_score)
        trial.set_user_attr("mean_raw", float(np.mean(fold_raw)))
        trial.set_user_attr("mean_tuned", float(np.mean(fold_tuned)))
        trial.set_user_attr("min_fold_tuned", float(np.min(fold_tuned)))
        trial.set_user_attr("fold_scores", fold_tuned)
        trial.set_user_attr("biases", biases.tolist())

        # Store OOB array for ensemble building
        OOB_STORE.append((trial.number, hp.copy(), oob_proba.copy()))

        logger.info(
            "Trial %3d | %-9s %-5s | global=%.6f | mean=%.4f | min_fold=%.4f | n=%d d=%d lr=%.4f",
            trial.number, algo, hp.get("boosting_type", ""),
            global_oob_score, np.mean(fold_tuned), np.min(fold_tuned),
            hp["n_estimators"], hp["max_depth"], hp["learning_rate"],
        )
        return global_oob_score

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Diverse Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Build the best ensemble from stored OOB arrays."""
    logger.info("=" * 72)
    logger.info("STAGE 3: DIVERSE ENSEMBLE (%d stored OOB arrays)", len(OOB_STORE))
    logger.info("=" * 72)

    if len(OOB_STORE) < 2:
        logger.warning("Not enough completed trials for ensemble.")
        return np.zeros((len(y), N_CLASSES)), np.zeros(3), 0.0, "none"

    # Sort by global OOB score
    scored = []
    for trial_num, params, oob in OOB_STORE:
        _, score = optimize_thresholds_quick(y, oob)
        scored.append((score, trial_num, params, oob))
    scored.sort(reverse=True, key=lambda x: x[0])

    top_n = min(30, len(scored))
    candidates = scored[:top_n]
    logger.info("Top-%d candidates (best global OOB = %.6f)", top_n, candidates[0][0])

    # ── Method 1: Greedy forward selection ──────────────────────────────────
    ensemble_oobs = [candidates[0][3]]
    ensemble_ids = [candidates[0][1]]
    avg_oob = candidates[0][3].copy()
    _, best_ens_score = optimize_thresholds_quick(y, avg_oob)

    for _score, trial_num, _params, oob in candidates[1:]:
        # Try adding this model
        trial_avg = (avg_oob * len(ensemble_oobs) + oob) / (len(ensemble_oobs) + 1)
        _, trial_score = optimize_thresholds_quick(y, trial_avg)
        if trial_score > best_ens_score:
            ensemble_oobs.append(oob)
            ensemble_ids.append(trial_num)
            avg_oob = trial_avg
            best_ens_score = trial_score
            logger.info("  + Added trial #%d -> ensemble size %d, score=%.6f",
                        trial_num, len(ensemble_oobs), best_ens_score)

    greedy_oob = avg_oob
    greedy_biases, greedy_score = optimize_thresholds_4pass(y, greedy_oob)
    logger.info("Greedy ensemble: %d models, global OOB=%.6f", len(ensemble_oobs), greedy_score)

    # ── Method 2: Weighted averaging (top-10, scipy optimise) ──────────────
    top10 = [c[3] for c in candidates[:min(10, len(candidates))]]

    def neg_score(weights):
        w = np.abs(weights)
        w /= w.sum() + 1e-12
        combined = sum(wi * oi for wi, oi in zip(w, top10, strict=True))
        _, s = optimize_thresholds_quick(y, combined)
        return -s

    try:
        result = minimize(neg_score, x0=np.ones(len(top10)) / len(top10),
                          method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-6})
        w_opt = np.abs(result.x)
        w_opt /= w_opt.sum()
        weighted_oob = sum(wi * oi for wi, oi in zip(w_opt, top10, strict=True))
        weighted_biases, weighted_score = optimize_thresholds_4pass(y, weighted_oob)
        logger.info("Weighted ensemble (top-%d): global OOB=%.6f, weights=%s",
                    len(top10), weighted_score,
                    [f"{w:.3f}" for w in w_opt])
    except Exception as e:
        logger.warning("Weighted optimisation failed: %s", e)
        weighted_oob = greedy_oob
        weighted_biases = greedy_biases
        weighted_score = greedy_score

    # ── Method 3: Stacking meta-learner ────────────────────────────────────
    top_k_stack = min(8, len(candidates))
    stack_oobs = [c[3] for c in candidates[:top_k_stack]]
    stacked_X = np.hstack(stack_oobs)  # (N, K*3)
    groups_stack = np.arange(len(y))  # dummy for now

    gkf = GroupKFold(n_splits=N_SPLITS)
    meta_oob = np.zeros((len(y), N_CLASSES))

    # Logistic regression meta-learner
    for tr, val in gkf.split(stacked_X, y, groups_stack):
        lr_meta = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
        lr_meta.fit(stacked_X[tr], y[tr], sample_weight=compute_sample_weights(y[tr]))
        meta_oob[val] = lr_meta.predict_proba(stacked_X[val])

    stack_biases, stack_score = optimize_thresholds_4pass(y, meta_oob)
    logger.info("Stacking (LR) meta-learner: global OOB=%.6f", stack_score)

    # LightGBM meta-learner
    meta_oob_lgbm = np.zeros((len(y), N_CLASSES))
    for tr, val in gkf.split(stacked_X, y, groups_stack):
        lgbm_meta = LGBMClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            objective="multiclass", num_class=N_CLASSES,
            verbose=-1, random_state=RANDOM_STATE,
        )
        lgbm_meta.fit(stacked_X[tr], y[tr], sample_weight=compute_sample_weights(y[tr]))
        meta_oob_lgbm[val] = lgbm_meta.predict_proba(stacked_X[val])

    stack_lgbm_biases, stack_lgbm_score = optimize_thresholds_4pass(y, meta_oob_lgbm)
    logger.info("Stacking (LGBM) meta-learner: global OOB=%.6f", stack_lgbm_score)

    # Pick the best method
    methods = [
        ("greedy", greedy_oob, greedy_biases, greedy_score),
        ("weighted", weighted_oob, weighted_biases, weighted_score),
        ("stack_lr", meta_oob, stack_biases, stack_score),
        ("stack_lgbm", meta_oob_lgbm, stack_lgbm_biases, stack_lgbm_score),
    ]
    best_method = max(methods, key=lambda x: x[3])
    logger.info("Best ensemble method: %s (%.6f)", best_method[0], best_method[3])

    return best_method[1], best_method[2], best_method[3], best_method[0]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Ultra-Fine Threshold Optimisation
# ═══════════════════════════════════════════════════════════════════════════════

def run_threshold_optimisation(y: np.ndarray, proba: np.ndarray,
                               label: str) -> tuple[np.ndarray, float]:
    """Full threshold optimisation with calibration and evolutionary search."""
    logger.info("=" * 72)
    logger.info("STAGE 4: ULTRA-FINE THRESHOLD OPTIMISATION (%s)", label)
    logger.info("=" * 72)

    # ── 4a: Standard 4-pass grid search ────────────────────────────────────
    biases_grid, score_grid = optimize_thresholds_4pass(y, proba)
    logger.info("4-pass grid search: %.6f  biases=[%.6f, %.6f, %.6f]",
                score_grid, *biases_grid)

    # ── 4b: Differential evolution on full 3-bias space ────────────────────
    lp = np.log(proba + 1e-12)

    def neg_recall_de(biases_3):
        preds = np.argmax(lp + biases_3, axis=1)
        return -min_class_recall(y, preds)

    try:
        de_result = differential_evolution(
            neg_recall_de, bounds=[(-6, 6), (-6, 6), (-6, 6)],
            seed=RANDOM_STATE, maxiter=500, tol=1e-8,
            mutation=(0.5, 1.5), recombination=0.9, popsize=30,
        )
        biases_de = de_result.x
        score_de = -de_result.fun
        logger.info("Differential evolution: %.6f  biases=[%.6f, %.6f, %.6f]",
                    score_de, *biases_de)
    except Exception as e:
        logger.warning("DE failed: %s", e)
        biases_de, score_de = biases_grid, score_grid

    # ── 4c: Isotonic calibration + re-threshold ────────────────────────────
    try:
        calibrated = np.zeros_like(proba)
        for cls in range(N_CLASSES):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(proba[:, cls], (y == cls).astype(float))
            calibrated[:, cls] = ir.predict(proba[:, cls])
        # Renormalise
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / np.clip(row_sums, 1e-12, None)

        biases_cal, score_cal = optimize_thresholds_4pass(y, calibrated)
        logger.info("Isotonic calibration + grid: %.6f  biases=[%.6f, %.6f, %.6f]",
                    score_cal, *biases_cal)
    except Exception as e:
        logger.warning("Isotonic calibration failed: %s", e)
        biases_cal, score_cal = biases_grid, score_grid
        calibrated = proba

    # ── 4d: Temperature scaling ────────────────────────────────────────────
    try:
        logits = np.log(proba + 1e-12)

        def neg_score_temp(T):
            T = max(T[0], 0.01)
            scaled = logits / T
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_s = np.exp(scaled)
            temp_proba = exp_s / exp_s.sum(axis=1, keepdims=True)
            _, s = optimize_thresholds_quick(y, temp_proba)
            return -s

        temp_result = minimize(neg_score_temp, x0=[1.0], method="Nelder-Mead",
                               options={"maxiter": 500})
        best_T = max(temp_result.x[0], 0.01)
        scaled = logits / best_T
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        temp_proba = exp_s / exp_s.sum(axis=1, keepdims=True)
        biases_temp, score_temp = optimize_thresholds_4pass(y, temp_proba)
        logger.info("Temperature scaling (T=%.4f): %.6f  biases=[%.6f, %.6f, %.6f]",
                    best_T, score_temp, *biases_temp)
    except Exception as e:
        logger.warning("Temperature scaling failed: %s", e)
        biases_temp, score_temp = biases_grid, score_grid

    # Pick the best
    results = [
        ("grid_4pass", biases_grid, score_grid),
        ("diff_evolution", biases_de, score_de),
        ("isotonic_cal", biases_cal, score_cal),
        ("temp_scaling", biases_temp, score_temp),
    ]
    best = max(results, key=lambda x: x[2])
    logger.info("Best threshold method: %s (%.6f)", best[0], best[2])

    # ── 4e: Hard case margin analysis ──────────────────────────────────────
    final_lp = np.log(proba + 1e-12)
    final_preds = np.argmax(final_lp + best[1], axis=1)
    errors = np.where(final_preds != y)[0]
    logger.info("Remaining errors: %d trajectories", len(errors))
    for idx in errors[:20]:
        true_cls = y[idx]
        pred_cls = final_preds[idx]
        p_true = proba[idx, true_cls]
        p_pred = proba[idx, pred_cls]
        margin = np.log(p_true + 1e-12) - np.log(p_pred + 1e-12)
        logger.info("  idx=%d  true=%d pred=%d  p_true=%.4f p_pred=%.4f  margin=%.4f",
                    idx, true_cls, pred_cls, p_true, p_pred, margin)

    return best[1], best[2]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("FINAL CEILING-BREAKER EXPERIMENT")
    logger.info("Budget: %d hours | Optuna trials: %d", TOTAL_TIME_BUDGET_S // 3600, N_OPTUNA_TRIALS)
    logger.info("=" * 72)

    # ── Load all features ──────────────────────────────────────────────────
    X_all, y, groups, all_feat = load_all_features()
    logger.info("Feature engineering complete. Elapsed: %.1f min", (time.time() - t0) / 60)

    # ── Stage 1: Feature selection ─────────────────────────────────────────
    ckpt_mask = CHECKPOINT_DIR / "feature_mask.npy"
    if ckpt_mask.exists():
        logger.info("Loading feature selection from checkpoint...")
        feature_mask = np.load(ckpt_mask)
        with open(CHECKPOINT_DIR / "selected_features.json") as fp:
            selected_features = json.load(fp)
    else:
        feature_mask, selected_features = run_feature_selection(X_all, y, groups, all_feat)

    X = X_all[:, feature_mask]
    logger.info("Proceeding with %d / %d features. Elapsed: %.1f min",
                X.shape[1], X_all.shape[1], (time.time() - t0) / 60)
    del X_all
    gc.collect()

    # ── Stage 2: Optuna ────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("STAGE 2: OPTUNA (%d trials, %d-fold, global OOB objective)",
                N_OPTUNA_TRIALS, N_SPLITS)
    logger.info("=" * 72)

    study = optuna.create_study(
        direction="maximize",
        study_name="ceiling-breaker",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=4),
        sampler=optuna.samplers.TPESampler(
            multivariate=True, seed=RANDOM_STATE,
        ),
    )

    stage2_start = time.time()
    try:
        study.optimize(
            create_objective(X, y, groups, stage2_start),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
    except optuna.exceptions.OptunaError as e:
        logger.info("Optuna stopped: %s", e)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info("Completed %d / %d trials in %.1f hours",
                len(completed), N_OPTUNA_TRIALS, (time.time() - stage2_start) / 3600)

    if completed:
        best = study.best_trial
        logger.info("Best trial: #%d  global_oob=%.6f  algorithm=%s",
                    best.number, best.value, best.params.get("algorithm"))
        logger.info("Hyperparameters:")
        for k, v in sorted(best.params.items()):
            logger.info("  %-25s = %s", k, v)

        logger.info("\nTop 10 trials:")
        for t in sorted(completed, key=lambda t: t.value, reverse=True)[:10]:
            logger.info("  #%3d  %-9s %-5s  global=%.6f  mean=%.4f  min_fold=%.4f",
                        t.number,
                        t.params.get("algorithm", "?"),
                        t.params.get("boosting_type", ""),
                        t.value,
                        t.user_attrs.get("mean_tuned", -1),
                        t.user_attrs.get("min_fold_tuned", -1))

    # Save Optuna results
    with open(CHECKPOINT_DIR / "optuna_results.json", "w") as fp:
        json.dump({
            "n_completed": len(completed),
            "best_value": study.best_value if completed else None,
            "best_params": study.best_params if completed else None,
        }, fp, indent=2)

    logger.info("Stage 2 elapsed: %.1f hours", (time.time() - stage2_start) / 3600)

    # ── Stage 3: Ensemble ──────────────────────────────────────────────────
    ens_oob, _ens_biases, ens_score, ens_method = run_ensemble(y)
    logger.info("Ensemble elapsed: %.1f min", (time.time() - t0) / 60 - (time.time() - stage2_start) / 60)

    # Also get the single best model's OOB
    if OOB_STORE:
        best_single_oob = max(OOB_STORE, key=lambda x: optimize_thresholds_quick(y, x[2])[1])
        _single_biases, _single_score = optimize_thresholds_4pass(y, best_single_oob[2])
    else:
        best_single_oob = (0, {}, np.zeros((len(y), N_CLASSES)))

    # ── Stage 4: Ultra-fine thresholds ─────────────────────────────────────
    # Run on both single best and ensemble
    logger.info("")
    single_biases_final, single_score_final = run_threshold_optimisation(
        y, best_single_oob[2], "single_best"
    )

    if ens_score > 0:
        ens_biases_final, ens_score_final = run_threshold_optimisation(
            y, ens_oob, f"ensemble_{ens_method}"
        )
    else:
        ens_biases_final, ens_score_final = single_biases_final, single_score_final

    # ── Final Report ───────────────────────────────────────────────────────
    total_hours = (time.time() - t0) / 3600

    # Per-fold breakdown of the best result
    best_proba = ens_oob if ens_score_final >= single_score_final else best_single_oob[2]
    best_biases = ens_biases_final if ens_score_final >= single_score_final else single_biases_final
    best_score = max(ens_score_final, single_score_final)
    best_label = f"ensemble_{ens_method}" if ens_score_final >= single_score_final else "single"

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_scores = []
    for _, val in gkf.split(np.zeros(len(y)), y, groups):
        preds = np.argmax(np.log(best_proba[val] + 1e-12) + best_biases, axis=1)
        fold_scores.append(min_class_recall(y[val], preds))

    logger.info("")
    logger.info("=" * 72)
    logger.info("FINAL REPORT")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  FEATURES:  %d selected / %d total", len(selected_features), len(all_feat))
    logger.info("  OPTUNA:    %d completed trials in %.1f hours", len(completed), total_hours)
    logger.info("")
    logger.info("  RESULTS:")
    logger.info("    Single best global OOB:    %.6f", single_score_final)
    logger.info("    Ensemble (%s) global OOB:  %.6f", ens_method, ens_score_final)
    logger.info("    ★ BEST global OOB:         %.6f  (%s)", best_score, best_label)
    logger.info("")
    logger.info("  PER-FOLD (best config):")
    for i, s in enumerate(fold_scores):
        logger.info("    Fold %2d: %.6f", i + 1, s)
    logger.info("    Mean:    %.6f", np.mean(fold_scores))
    logger.info("    Min:     %.6f", np.min(fold_scores))
    logger.info("")
    logger.info("  THRESHOLDS:")
    logger.info("    biases = [%.6f, %.6f, %.6f]", *best_biases)
    logger.info("")

    prev_global_oob = 0.9995
    delta = best_score - prev_global_oob
    logger.info("  COMPARISON vs PREVIOUS BEST (0.9995):")
    logger.info("    Previous global OOB: %.6f", prev_global_oob)
    logger.info("    This run global OOB: %.6f", best_score)
    logger.info("    Delta:               %+.6f", delta)
    if delta > 0:
        logger.info("    ★★★ NEW RECORD! ★★★")
    else:
        logger.info("    (no improvement — confirming 0.9995 is the empirical ceiling)")
    logger.info("")
    logger.info("  Total runtime: %.1f hours", total_hours)

    # ── Save final results ─────────────────────────────────────────────────
    results = {
        "selected_features": selected_features,
        "n_features_total": len(all_feat),
        "n_features_selected": len(selected_features),
        "n_optuna_completed": len(completed),
        "single_best_global_oob": float(single_score_final),
        "ensemble_method": ens_method,
        "ensemble_global_oob": float(ens_score_final),
        "best_global_oob": float(best_score),
        "best_method": best_label,
        "best_biases": best_biases.tolist(),
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_min": float(np.min(fold_scores)),
        "previous_best": prev_global_oob,
        "delta": float(delta),
        "total_runtime_hours": total_hours,
        "best_optuna_params": study.best_params if completed else None,
    }
    with open(CHECKPOINT_DIR / "final_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Results saved to %s", CHECKPOINT_DIR / "final_results.json")


if __name__ == "__main__":
    main()
