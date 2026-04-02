"""
SALVO CLUSTERING CEILING ATTEMPT — Google Colab (H100 GPU)
===========================================================

Full production-grade 4-stage experiment testing whether spatio-temporal
salvo clustering and rebel-group geographic features can break the 0.9995
min-recall ceiling established by kinematic features alone.

This is NOT a proof-of-concept. It matches the rigour of
colab_brute_force_optimization.py:

  Stage 0: Data prep   — engineer salvo + rebel-group features from train.csv
  Stage 1: Feature sel — backward elimination on all 69 candidate features
  Stage 2: Optuna      — 100-trial multi-algorithm search (LightGBM, XGBoost,
                         CatBoost, TabMLP), per-fold threshold tuning,
                         MedianPruner, GroupKFold grouped by rebel_group_id
  Stage 3: Final model — OOB global threshold tuning, 100% retrain, artifacts
  Stage 4: Report      — side-by-side baseline vs enriched, per-fold breakdown,
                         feature importances, salvo purity

Key design choices vs colab_salvo_clustering.py (the POC)
----------------------------------------------------------
1. GroupKFold by rebel_group_id (not salvo_id).
   A rebel group fires many salvos over time.  If salvo A and salvo B from
   the same rebel group land in different folds, the model learns
   rebel_group_id -> class from fold-train and leaks it to fold-val.
   Grouping by rebel_group_id guarantees each base is entirely in one fold.

2. rebel_group_id as LightGBM categorical feature.
   The DBSCAN integer ID is nominal (not ordinal).  Declaring it categorical
   lets LightGBM use one-vs-rest splits instead of range splits, which is
   the correct representation.

3. 100 Optuna trials with per-fold threshold tuning inside the objective.
   Same strategy as colab_brute_force_optimization.py.  The score being
   maximised is mean-tuned-min-recall across folds, not raw argmax recall.

4. MedianPruner after fold 2.
   Abandons trials whose tuned recall falls below the median of completed
   trials, saving ~30% of compute.

5. Side-by-side baseline (61 kinematic features) evaluated under the SAME
   GroupKFold by rebel_group_id so the delta is meaningful.
   Note: this CV is stricter than the original 0.9995 measurement (which
   used GroupKFold by traj_ind), so the baseline here may be slightly lower.

Business assumptions leveraged (assignment)
-------------------------------------------
3a. Launcher payload capacity   -> group_max_salvo_size
3b. Rockets fired in salvos     -> salvo_size, salvo_duration_s,
                                   salvo_spatial_spread_m, salvo_time_rank
3c. Geographically concentrated -> rebel_group_id (pure-spatial DBSCAN)
3d. Each group buys independently-> rebel_group_id is a class-label proxy

Why each algorithm:
  LightGBM  — best algorithm in previous experiments; fast GPU histogram building
  XGBoost   — strong alternative; CUDA device; different regularisation inductive bias
  CatBoost  — natively handles rebel_group_id as a true categorical (no hack needed)
  TabMLP    — entity embedding for rebel_group_id + BatchNorm MLP; architecturally
              correct NN for tabular + categorical data; tests whether H100 GPU
              matrix ops beat GBT cache-miss patterns on this dataset size

Setup (run in Colab first cell)
-------------------------------
    !pip install -q optuna lightgbm xgboost catboost scikit-learn pandas numpy pyarrow torch

Files to upload to /content/:
    - train.csv
    - cache_train_features.parquet   (or .feather)

Expected runtime on H100: ~90-120 min
"""

# %%
# ── Imports ────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as td
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
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
log = logging.getLogger("salvo-ceiling")

# %%
# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("/content")
TRAIN_CSV       = DATA_DIR / "train.csv"
FEATURE_CACHE   = DATA_DIR / "cache_train_features.parquet"
RESULTS_OUTPUT  = DATA_DIR / "salvo_ceiling_results.json"
ARTIFACTS_DIR   = DATA_DIR / "salvo_artifacts"

N_SPLITS_OPTUNA = 10      # folds for Optuna and final model
N_SPLITS_FEATSEL= 5       # folds for feature selection (speed)
N_TRIALS        = 100     # Optuna trials
N_CLASSES       = 3
RANDOM_SEED     = 42

# Threshold grid sizes — coarse -> fine -> ultra-fine (matches brute-force script)
GRID_COARSE     = 80
GRID_FINE       = 50
GRID_ULTRA      = 30

BASELINE_SCORE  = 0.9995  # kinematic-only ceiling to beat

# DBSCAN parameters
SALVO_EPS       = 0.5     # spatiotemporal (normalised space)
SALVO_MIN_SAMP  = 2
GROUP_EPS       = 1.0     # pure-spatial for rebel groups
GROUP_MIN_SAMP  = 3

# GPU detection — mirrors colab_brute_force_optimization.py
try:
    import subprocess as _sp
    _sp.run(["nvidia-smi"], check=True, capture_output=True)
    DEVICE      = "gpu"     # LightGBM / CatBoost device string
    XGB_DEVICE  = "cuda"    # XGBoost device string
    TORCH_DEVICE= torch.device("cuda")
except Exception:
    DEVICE      = "cpu"
    XGB_DEVICE  = "cpu"
    TORCH_DEVICE= torch.device("cpu")
N_JOBS = int(os.cpu_count() or 4)
print(f"LightGBM/CatBoost: {DEVICE}  |  XGBoost: {XGB_DEVICE}  |  PyTorch: {TORCH_DEVICE}  |  CPU cores: {N_JOBS}")

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
    "salvo_size",
    "salvo_duration_s",
    "salvo_spatial_spread_m",
    "salvo_time_rank",
]

GROUP_FEATURES = [
    "rebel_group_id",       # categorical: LightGBM must know this is nominal
    "group_total_rockets",
    "group_n_salvos",
    "group_max_salvo_size",
]

ALL_CANDIDATE_FEATURES = SELECTED_FEATURES + SALVO_FEATURES + GROUP_FEATURES

# %%
# ── Utility functions ──────────────────────────────────────────────────────────

def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Minimum per-class recall — the official evaluation metric."""
    recalls = []
    for cls in range(N_CLASSES):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(np.sum((y_pred == cls) & mask) / mask.sum()))
    return float(np.min(recalls))


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

    Identical to colab_brute_force_optimization.py.
    Returns (biases, best_score). biases[0] is fixed at 0.
    """
    lp = np.log(proba + 1e-12)
    best_s, best_b = -1.0, np.zeros(N_CLASSES)

    for b1 in np.linspace(-4, 4, GRID_COARSE):
        for b2 in np.linspace(-4, 4, GRID_COARSE):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    for b1 in np.linspace(best_b[1] - 0.12, best_b[1] + 0.12, GRID_FINE):
        for b2 in np.linspace(best_b[2] - 0.12, best_b[2] + 0.12, GRID_FINE):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    for b1 in np.linspace(best_b[1] - 0.02, best_b[1] + 0.02, GRID_ULTRA):
        for b2 in np.linspace(best_b[2] - 0.02, best_b[2] + 0.02, GRID_ULTRA):
            b = np.array([0.0, b1, b2])
            s = min_class_recall(y_true, np.argmax(lp + b, axis=1))
            if s > best_s:
                best_s, best_b = s, b.copy()

    return best_b, best_s


def make_lgbm(params: dict, cat_col_idx: int | None) -> LGBMClassifier:
    """Build a LGBMClassifier with GPU and categorical feature support."""
    return LGBMClassifier(
        **params,
        objective="multiclass",
        num_class=N_CLASSES,
        device=DEVICE,
        gpu_use_dp=False,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        verbose=-1,
        categorical_feature=[cat_col_idx] if cat_col_idx is not None else "auto",
    )


# ──────────────────────────────────────────────────────────────────────────────
# TabMLP: entity-embedding MLP for tabular + categorical data
# ──────────────────────────────────────────────────────────────────────────────
# Architecture (with rebel_group_id as the one categorical column):
#
#   rebel_group_id ──► Embedding(n_groups, embed_dim) ──► embed_vec
#   continuous     ──► BatchNorm ──────────────────────► cont_vec
#   [embed_vec || cont_vec] ──► MLP(hidden_dims, dropout) ──► logits(3)
#
# Why entity embeddings (not one-hot / integer):
#   rebel_group_id is nominal: group 5 is not "greater than" group 3.
#   An embedding lets the model learn a dense similarity space over groups,
#   so it can generalise "groups with similar launch patterns have similar
#   classes" — which is exactly what assumption 3d implies.
#
# If rebel_group_id was dropped by feature selection (cat_col is None),
# the model falls back to a pure continuous MLP (no embedding).

class TabMLP(nn.Module):
    """Entity-embedding MLP for one categorical + N continuous features."""

    def __init__(
        self,
        n_continuous: int,
        n_categories: int,     # number of unique rebel_group_id values + 1
        embed_dim: int,
        hidden_dims: list[int],
        dropout: float,
        cat_col: int | None,   # column index of rebel_group_id in X
    ) -> None:
        super().__init__()
        self.cat_col = cat_col
        if cat_col is not None:
            self.embed = nn.Embedding(n_categories, embed_dim, padding_idx=0)
            in_dim = (n_continuous - 1) + embed_dim
        else:
            self.embed = None
            in_dim = n_continuous

        layers: list[nn.Module] = [nn.BatchNorm1d(in_dim)]
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cat_col is not None and self.embed is not None:
            cat = x[:, self.cat_col].long().clamp(0)  # safe cast to index
            cont_cols = [i for i in range(x.shape[1]) if i != self.cat_col]
            cont = x[:, cont_cols]
            emb = self.embed(cat)
            x = torch.cat([cont, emb], dim=1)
        return self.net(x)


def _tabmlp_proba(model: TabMLP, X: np.ndarray) -> np.ndarray:
    """Run forward pass, return softmax probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(TORCH_DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy()


def _fit_tabmlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    sw: np.ndarray,
    cat_col: int | None,
    params: dict,
    n_categories: int,
) -> TabMLP:
    """Train a TabMLP for a fixed number of epochs (no early stopping).

    No early stopping: using the validation fold for stopping would leak
    label information. Optuna instead tunes `epochs` directly.
    """
    device = TORCH_DEVICE
    model = TabMLP(
        n_continuous=Xtr.shape[1],
        n_categories=n_categories,
        embed_dim=params["embed_dim"],
        hidden_dims=[params["hidden_dim"]] * params["n_layers"],
        dropout=params["dropout"],
        cat_col=cat_col,
    ).to(device)

    class_w = torch.tensor(
        [1.0 / max((ytr == c).sum(), 1) for c in range(N_CLASSES)],
        dtype=torch.float32, device=device,
    )
    class_w = class_w / class_w.sum() * N_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_w, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    dataset = td.TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
        torch.tensor(sw, dtype=torch.float32),
    )
    loader = td.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, drop_last=False)

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            loss = (criterion(model(xb), yb) * wb).sum() / wb.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def run_fold(
    algo: str,
    params: dict,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    sw: np.ndarray,
    Xv: np.ndarray,
    cat_col: int | None,
    n_cat: int = 1,
) -> tuple[object, np.ndarray]:
    """Fit one fold and return (fitted_model, proba_on_Xv).

    Unified interface for all algorithms — Optuna objective calls this.
    For TabMLP, n_cat = number of unique rebel_group_id values + 2 (padding).
    """
    if algo == "lightgbm":
        m = make_lgbm(params, cat_col)
        m.fit(Xtr, ytr, sample_weight=sw)
        return m, m.predict_proba(Xv)

    if algo == "xgboost":
        m = XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            device=XGB_DEVICE,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        m.fit(Xtr, ytr, sample_weight=sw)
        return m, m.predict_proba(Xv)

    if algo == "catboost":
        p = dict(params)
        m = CatBoostClassifier(
            iterations=p.pop("n_estimators", 600),
            depth=min(p.pop("max_depth", 6), 10),
            learning_rate=p.pop("learning_rate", 0.05),
            subsample=p.pop("subsample", 0.8),
            l2_leaf_reg=p.pop("reg_lambda", 1.0),
            bootstrap_type="Bernoulli",
            loss_function="MultiClass",
            classes_count=N_CLASSES,
            cat_features=[cat_col] if cat_col is not None else [],
            task_type="GPU" if DEVICE == "gpu" else "CPU",
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        m.fit(Xtr, ytr, sample_weight=sw)
        return m, m.predict_proba(Xv)

    # tabmlp
    m = _fit_tabmlp(Xtr, ytr, sw, cat_col, params, n_cat)
    return m, _tabmlp_proba(m, Xv)


# %%
# ── Stage 0: Data preparation and feature engineering ─────────────────────────

log.info("=" * 70)
log.info("STAGE 0: DATA PREPARATION & FEATURE ENGINEERING")
log.info("=" * 70)

# Load raw data to extract launch timestamps
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

# Load feature cache
feather_path = FEATURE_CACHE.with_suffix(".feather")
if feather_path.exists():
    import pyarrow.feather as pa_feather
    feats = pa_feather.read_table(str(feather_path), memory_map=True).to_pandas()
    log.info("Loaded features from Feather: %s", feather_path)
else:
    feats = pd.read_parquet(FEATURE_CACHE)
    log.info("Loaded features from Parquet: %s", FEATURE_CACHE)

feats = feats.join(launch_meta.set_index("traj_ind")[["launch_time"]], how="left")
assert "launch_x" in feats.columns and "launch_y" in feats.columns

# ── Salvo clustering (spatio-temporal) ────────────────────────────────────────

launch_time_s = feats["launch_time"].astype(np.int64) / 1e9
cluster_input = pd.DataFrame({
    "launch_x": feats["launch_x"],
    "launch_y": feats["launch_y"],
    "launch_time_s": launch_time_s,
}).fillna(0.0)

scaler_st = StandardScaler()
cluster_scaled = scaler_st.fit_transform(cluster_input)
raw_salvo = DBSCAN(eps=SALVO_EPS, min_samples=SALVO_MIN_SAMP, n_jobs=-1).fit_predict(cluster_scaled)

next_id = int(raw_salvo.max()) + 1
salvo_ids = raw_salvo.copy()
for i in range(len(salvo_ids)):
    if salvo_ids[i] == -1:
        salvo_ids[i] = next_id
        next_id += 1
feats["salvo_id"] = salvo_ids

n_salvos = len(set(raw_salvo[raw_salvo >= 0]))
log.info("Salvos: %d identified | %d noise rockets", n_salvos, (raw_salvo == -1).sum())

# Engineer salvo features
salvo_rows = []
for _sid, grp in feats.groupby("salvo_id"):
    n = len(grp)
    lx, ly = grp["launch_x"].values, grp["launch_y"].values
    lt = grp["launch_time"].values.astype(np.int64) / 1e9
    if n > 1:
        dx = lx[:, None] - lx[None, :]
        dy = ly[:, None] - ly[None, :]
        spread = float(np.sqrt(dx**2 + dy**2).max())
        dur = float(lt.max() - lt.min())
    else:
        spread = dur = 0.0
    ranks = pd.Series(lt).rank(method="first").astype(int).values
    for i, tid in enumerate(grp.index):
        salvo_rows.append(dict(
            traj_ind=tid,
            salvo_size=n,
            salvo_duration_s=dur,
            salvo_spatial_spread_m=spread,
            salvo_time_rank=int(ranks[i]),
        ))
feats = feats.join(pd.DataFrame(salvo_rows).set_index("traj_ind"), how="left")

# ── Rebel group clustering (pure-spatial) ─────────────────────────────────────

spatial_input = feats[["launch_x", "launch_y"]].fillna(0.0)
scaler_sp = StandardScaler()
spatial_scaled = scaler_sp.fit_transform(spatial_input)
raw_group = DBSCAN(eps=GROUP_EPS, min_samples=GROUP_MIN_SAMP, n_jobs=-1).fit_predict(spatial_scaled)

next_gid = int(raw_group.max()) + 1
group_ids = raw_group.copy()
for i in range(len(group_ids)):
    if group_ids[i] == -1:
        group_ids[i] = next_gid
        next_gid += 1
feats["rebel_group_id"] = group_ids

n_groups = len(set(raw_group[raw_group >= 0]))
log.info(
    "Rebel groups: %d identified | %d solo rockets (noise)",
    n_groups, (raw_group == -1).sum(),
)

group_stats = feats.groupby("rebel_group_id").agg(
    group_total_rockets=("rebel_group_id", "count"),
    group_n_salvos=("salvo_id", "nunique"),
    group_max_salvo_size=("salvo_size", "max"),
)
feats = feats.join(group_stats, on="rebel_group_id")

# Salvo purity (does same rebel group = same class?)
multi_mask = feats["salvo_size"] > 1
if multi_mask.sum() > 0:
    purity = feats[multi_mask].groupby("salvo_id")["label"].apply(
        lambda g: (g == g.mode().iloc[0]).mean()
    )
    log.info(
        "Multi-rocket salvo label purity: mean=%.3f  pure100%%=%.1f%%",
        purity.mean(), (purity == 1.0).mean() * 100,
    )

# ── Prepare arrays ─────────────────────────────────────────────────────────────

y = feats["label"].to_numpy(dtype=int)

# GroupKFold groups — by rebel_group_id (most conservative, prevents leakage)
# Adaptive n_splits: can't have more folds than unique groups
n_real_groups = len(np.unique(group_ids[raw_group >= 0]))
effective_splits_optuna = min(N_SPLITS_OPTUNA, n_real_groups)
effective_splits_featsel = min(N_SPLITS_FEATSEL, n_real_groups)
cv_groups = feats["rebel_group_id"].to_numpy(dtype=int)
log.info(
    "GroupKFold by rebel_group_id: %d unique groups -> using %d-fold CV",
    n_real_groups, effective_splits_optuna,
)

# Feature matrices
X_baseline = feats.reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32)
X_enriched = feats.reindex(columns=ALL_CANDIDATE_FEATURES).to_numpy(dtype=np.float32)
rebel_group_col_enriched = ALL_CANDIDATE_FEATURES.index("rebel_group_id")

log.info("Baseline  features: %s", X_baseline.shape)
log.info("Candidate features: %s (+%d new)", X_enriched.shape, len(SALVO_FEATURES + GROUP_FEATURES))

# %%
# ── Stage 1: Feature selection ─────────────────────────────────────────────────

log.info("")
log.info("=" * 70)
log.info("STAGE 1: BACKWARD FEATURE ELIMINATION ON %d CANDIDATE FEATURES", len(ALL_CANDIDATE_FEATURES))
log.info("=" * 70)


def cv_score_subset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    n_splits: int,
    cat_idx: int | None,
) -> float:
    """CV min-recall (threshold-tuned) on a feature subset."""
    X_sub = X[:, mask]
    gkf = GroupKFold(n_splits=n_splits)
    oob_proba = np.zeros((len(y), N_CLASSES))
    params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=10)
    for tr_idx, val_idx in gkf.split(X_sub, y, groups):
        Xtr, Xv = X_sub[tr_idx].copy(), X_sub[val_idx].copy()
        med = np.nanmedian(Xtr, axis=0)
        impute_nan(Xtr, med)
        impute_nan(Xv, med)
        sw = compute_sample_weights(y[tr_idx])
        # Remap categorical column index to subset space
        new_cat = None
        if cat_idx is not None:
            # Find position of cat column within the masked subset
            subset_positions = np.where(mask)[0]
            matches = np.where(subset_positions == cat_idx)[0]
            if len(matches) > 0:
                new_cat = int(matches[0])
        clf = make_lgbm(params, new_cat)
        clf.fit(Xtr, y[tr_idx], sample_weight=sw)
        oob_proba[val_idx] = clf.predict_proba(Xv)
    _, score = optimize_thresholds(y, oob_proba)
    return score


n_features = X_enriched.shape[1]
all_mask = np.ones(n_features, dtype=bool)

# Get initial importances to rank elimination candidates
log.info("Computing initial feature importances...")
X_imp = X_enriched.copy()
impute_nan(X_imp, np.nanmedian(X_imp, axis=0))
sw_all = compute_sample_weights(y)
init_clf = make_lgbm(
    dict(n_estimators=600, max_depth=6, learning_rate=0.05,
         subsample=0.8, colsample_bytree=0.8),
    rebel_group_col_enriched,
)
init_clf.fit(X_imp, y, sample_weight=sw_all)
importances = init_clf.feature_importances_
rank_order = np.argsort(importances)  # ascending: worst first

baseline_sel_score = cv_score_subset(
    X_enriched, y, cv_groups, all_mask, effective_splits_featsel, rebel_group_col_enriched,
)
log.info("Baseline score (all %d features): %.4f", n_features, baseline_sel_score)

current_mask = all_mask.copy()
current_score = baseline_sel_score
dropped: list[str] = []

for idx in rank_order:
    if current_mask.sum() <= 10:
        break
    trial_mask = current_mask.copy()
    trial_mask[idx] = False
    trial_score = cv_score_subset(
        X_enriched, y, cv_groups, trial_mask, effective_splits_featsel, rebel_group_col_enriched,
    )
    fname = ALL_CANDIDATE_FEATURES[idx]
    if trial_score >= current_score - 0.0001:
        current_mask = trial_mask
        current_score = trial_score
        dropped.append(fname)
        log.info(
            "  DROP %-30s (imp=%6.1f) -> %d features, score=%.4f",
            fname, importances[idx], current_mask.sum(), current_score,
        )
    else:
        log.info(
            "  KEEP %-30s (imp=%6.1f) -> drop would give %.4f",
            fname, importances[idx], trial_score,
        )

selected_features = [ALL_CANDIDATE_FEATURES[i] for i in range(n_features) if current_mask[i]]
X_sel = X_enriched[:, current_mask]

# Remap rebel_group_id column index to selected-feature space
if "rebel_group_id" in selected_features:
    rebel_group_col_sel = selected_features.index("rebel_group_id")
    log.info("rebel_group_id KEPT at column %d in selected features", rebel_group_col_sel)
else:
    rebel_group_col_sel = None
    log.info("rebel_group_id was DROPPED by feature selection")

log.info("")
log.info("Feature selection summary:")
log.info("  Started:  %d  |  Dropped: %d  |  Selected: %d", n_features, len(dropped), len(selected_features))
log.info("  Dropped features: %s", dropped if dropped else "none")
log.info("  Score: %.4f  (started at %.4f)", current_score, baseline_sel_score)
log.info("")
log.info("Selected features:")
for i, f in enumerate(selected_features, 1):
    tag = " [NEW]" if f in SALVO_FEATURES + GROUP_FEATURES else ""
    log.info("  %2d. %s%s", i, f, tag)

# %%
# ── Stage 2: Optuna hyperparameter search ─────────────────────────────────────

log.info("")
log.info("=" * 70)
log.info("STAGE 2: OPTUNA (%d trials, %d-fold GroupKFold by rebel_group_id)", N_TRIALS, effective_splits_optuna)
log.info("=" * 70)


def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    cat_col: int | None,
    n_cat: int,
) -> callable:
    def objective(trial: optuna.Trial) -> float:
        algo = trial.suggest_categorical(
            "algorithm", ["lightgbm", "xgboost", "catboost", "tabmlp"]
        )

        if algo == "lightgbm":
            hp = dict(
                n_estimators      = trial.suggest_int("n_estimators", 300, 3000),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth         = trial.suggest_int("max_depth", 4, 15),
                num_leaves        = trial.suggest_int("num_leaves", 31, 300),
                min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
                subsample         = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
        elif algo == "xgboost":
            hp = dict(
                n_estimators      = trial.suggest_int("n_estimators", 300, 3000),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth         = trial.suggest_int("max_depth", 3, 12),
                subsample         = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight  = trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
        elif algo == "catboost":
            hp = dict(
                n_estimators      = trial.suggest_int("n_estimators", 300, 2500),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                max_depth         = trial.suggest_int("max_depth", 4, 10),
                subsample         = trial.suggest_float("subsample", 0.6, 1.0),
                reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            )
        else:  # tabmlp
            hp = dict(
                n_layers   = trial.suggest_int("n_layers", 2, 5),
                hidden_dim = trial.suggest_int("hidden_dim", 64, 512),
                embed_dim  = trial.suggest_int("embed_dim", 4, 32),
                dropout    = trial.suggest_float("dropout", 0.0, 0.5),
                lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                batch_size = trial.suggest_categorical("batch_size", [128, 256, 512]),
                epochs     = trial.suggest_int("epochs", 20, 100),
            )

        gkf = GroupKFold(n_splits=n_splits)
        fold_raw, fold_tuned = [], []

        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            Xtr, Xv = X[tr_idx].copy(), X[val_idx].copy()
            ytr, yv = y[tr_idx], y[val_idx]
            med = np.nanmedian(Xtr, axis=0)
            impute_nan(Xtr, med)
            impute_nan(Xv, med)
            sw = compute_sample_weights(ytr)

            _, proba = run_fold(algo, hp, Xtr, ytr, sw, Xv, cat_col, n_cat)

            fold_raw.append(min_class_recall(yv, np.argmax(proba, axis=1)))
            _, ts = optimize_thresholds(yv, proba)
            fold_tuned.append(ts)

            if fold_idx >= 2:
                trial.report(float(np.mean(fold_tuned)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_raw   = float(np.mean(fold_raw))
        mean_tuned = float(np.mean(fold_tuned))
        min_tuned  = float(np.min(fold_tuned))

        trial.set_user_attr("algorithm", algo)
        trial.set_user_attr("mean_raw", mean_raw)
        trial.set_user_attr("mean_tuned", mean_tuned)
        trial.set_user_attr("min_fold_tuned", min_tuned)
        trial.set_user_attr("fold_scores_tuned", fold_tuned)

        log.info(
            "Trial %3d | %-9s | raw=%.4f | tuned=%.4f | min_fold=%.4f",
            trial.number, algo, mean_raw, mean_tuned, min_tuned,
        )
        return mean_tuned

    return objective


study = optuna.create_study(
    direction="maximize",
    study_name="salvo-ceiling",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
)
# n_cat: how many unique rebel_group_id values + 2 (padding for unseen groups)
n_cat_global = int(feats["rebel_group_id"].max()) + 2

study.optimize(
    create_objective(X_sel, y, cv_groups, effective_splits_optuna, rebel_group_col_sel, n_cat_global),
    n_trials=N_TRIALS,
    show_progress_bar=True,
)

best = study.best_trial
log.info("")
log.info("Optuna complete. Best trial #%d: tuned=%.4f  raw=%.4f  min_fold=%.4f",
         best.number, best.value,
         best.user_attrs.get("mean_raw", -1),
         best.user_attrs.get("min_fold_tuned", -1))

# Top 5 trials
log.info("Top 5 trials:")
for t in sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]:
    if t.value is not None:
        log.info("  #%3d %-9s tuned=%.4f  raw=%.4f  min_fold=%.4f",
                 t.number, t.user_attrs.get("algorithm", "?"), t.value,
                 t.user_attrs.get("mean_raw", -1),
                 t.user_attrs.get("min_fold_tuned", -1))

# %%
# ── Stage 3: Final model + OOB threshold tuning + 100% retrain ────────────────

log.info("")
log.info("=" * 70)
log.info("STAGE 3: FINAL MODEL — OOB THRESHOLD TUNING + 100%% RETRAIN")
log.info("=" * 70)

best_algo   = best.user_attrs.get("algorithm", best.params.get("algorithm", "lightgbm"))
best_params = {k: v for k, v in best.params.items() if k != "algorithm"}
log.info("Best algorithm: %s", best_algo)

gkf_final = GroupKFold(n_splits=effective_splits_optuna)
oob_proba = np.zeros((len(y), N_CLASSES))

for tr_idx, val_idx in gkf_final.split(X_sel, y, cv_groups):
    Xtr, Xv = X_sel[tr_idx].copy(), X_sel[val_idx].copy()
    ytr = y[tr_idx]
    med = np.nanmedian(Xtr, axis=0)
    impute_nan(Xtr, med)
    impute_nan(Xv, med)
    sw = compute_sample_weights(ytr)
    _, proba = run_fold(best_algo, dict(best_params), Xtr, ytr, sw, Xv, rebel_group_col_sel, n_cat_global)
    oob_proba[val_idx] = proba

biases, oob_global_score = optimize_thresholds(y, oob_proba)
log.info("OOB global threshold-tuned score: %.6f", oob_global_score)
log.info("Optimal biases: [%.6f, %.6f, %.6f]", *biases)

oob_fold_scores = []
for _, val_idx in gkf_final.split(X_sel, y, cv_groups):
    preds = np.argmax(np.log(oob_proba[val_idx] + 1e-12) + biases, axis=1)
    oob_fold_scores.append(min_class_recall(y[val_idx], preds))

# Retrain on 100%
X_full = X_sel.copy()
med_full = np.nanmedian(X_full, axis=0)
impute_nan(X_full, med_full)
sw_full = compute_sample_weights(y)
final_model, _ = run_fold(
    best_algo, dict(best_params), X_full, y, sw_full,
    X_full[:1], rebel_group_col_sel, n_cat_global,  # Xv dummy — not used
)

# Train recall sanity check
if best_algo == "tabmlp":
    train_proba = _tabmlp_proba(final_model, X_full)
else:
    train_proba = final_model.predict_proba(X_full)
train_raw   = min_class_recall(y, np.argmax(train_proba, axis=1))
train_tuned = min_class_recall(y, np.argmax(np.log(train_proba + 1e-12) + biases, axis=1))

# Save artifacts
ARTIFACTS_DIR.mkdir(exist_ok=True)
if best_algo == "lightgbm":
    joblib.dump(final_model, str(ARTIFACTS_DIR / "model_salvo.pkl"))
    final_model.booster_.save_model(str(ARTIFACTS_DIR / "model_salvo.lgb"))
elif best_algo == "xgboost":
    final_model.save_model(str(ARTIFACTS_DIR / "model_salvo.json"))
elif best_algo == "catboost":
    final_model.save_model(str(ARTIFACTS_DIR / "model_salvo.cbm"))
else:  # tabmlp
    torch.save(final_model.state_dict(), str(ARTIFACTS_DIR / "model_salvo_tabmlp.pt"))
np.save(str(ARTIFACTS_DIR / "train_medians_salvo.npy"), med_full)
np.save(str(ARTIFACTS_DIR / "threshold_biases_salvo.npy"), biases)
with open(str(ARTIFACTS_DIR / "selected_features_salvo.json"), "w") as f:
    json.dump({"algorithm": best_algo, "features": selected_features}, f, indent=2)
log.info("Artifacts saved to %s", ARTIFACTS_DIR)

# %%
# ── Stage 4: Comparison report ─────────────────────────────────────────────────

log.info("")
log.info("=" * 70)
log.info("STAGE 4: COMPARISON — BASELINE vs ENRICHED")
log.info("=" * 70)

# Run baseline (61 kinematic features) under the SAME GroupKFold by rebel_group_id
log.info("Evaluating baseline (61 kinematic features, same CV strategy)...")
oob_proba_base = np.zeros((len(y), N_CLASSES))
base_params = dict(
    n_estimators=2011, learning_rate=0.082, max_depth=12,
    subsample=0.913, colsample_bytree=0.679, min_child_samples=20,
)
for tr_idx, val_idx in GroupKFold(n_splits=effective_splits_optuna).split(X_baseline, y, cv_groups):
    Xtr, Xv = X_baseline[tr_idx].copy(), X_baseline[val_idx].copy()
    ytr = y[tr_idx]
    med = np.nanmedian(Xtr, axis=0)
    impute_nan(Xtr, med)
    impute_nan(Xv, med)
    sw = compute_sample_weights(ytr)
    clf_b = make_lgbm(base_params, None)
    clf_b.fit(Xtr, ytr, sample_weight=sw)
    oob_proba_base[val_idx] = clf_b.predict_proba(Xv)

base_biases, base_score = optimize_thresholds(y, oob_proba_base)
base_fold_scores = []
for _, val_idx in GroupKFold(n_splits=effective_splits_optuna).split(X_baseline, y, cv_groups):
    preds = np.argmax(np.log(oob_proba_base[val_idx] + 1e-12) + base_biases, axis=1)
    base_fold_scores.append(min_class_recall(y[val_idx], preds))

# Feature importances (tree models only — TabMLP uses attention weights not available here)
if best_algo != "tabmlp" and hasattr(final_model, "feature_importances_"):
    importances_final = pd.Series(
        final_model.feature_importances_, index=selected_features,
    ).sort_values(ascending=False)
else:
    importances_final = pd.Series(dtype=float)  # empty for TabMLP

# ── Print full report ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FINAL REPORT — SALVO CLUSTERING CEILING ATTEMPT")
print("=" * 70)

print(f"\n  CV strategy: GroupKFold by rebel_group_id ({effective_splits_optuna}-fold)")
print("  Note: stricter than original 0.9995 measurement (GroupKFold by traj_ind)")

print("\n  FEATURE SELECTION:")
print(f"    Candidate features : {len(ALL_CANDIDATE_FEATURES)}")
print(f"    Selected features  : {len(selected_features)}")
print(f"    Dropped            : {dropped if dropped else 'none'}")
new_kept = [f for f in selected_features if f in SALVO_FEATURES + GROUP_FEATURES]
print(f"    New features kept  : {new_kept if new_kept else 'none (all dropped)'}")

print("\n  SCORES (threshold-tuned OOB min-recall):")
print(f"    Kinematic baseline (61 feat, same CV): {base_score:.4f}")
print(f"    Enriched model ({len(selected_features):>2d} feat, Optuna best): {oob_global_score:.4f}")
print(f"    Reported ceiling (original CV)       : {BASELINE_SCORE:.4f}")
delta = oob_global_score - base_score
print(f"    Delta (enriched - baseline)          : {delta:+.4f}")

print("\n  Per-fold OOB (enriched):")
for i, s in enumerate(oob_fold_scores, 1):
    print(f"    Fold {i:>2d}: {s:.4f}")
print(f"    Mean: {np.mean(oob_fold_scores):.4f} +/- {np.std(oob_fold_scores):.4f}")
print(f"    Min:  {np.min(oob_fold_scores):.4f}")

print("\n  Train recall (enriched, sanity):")
print(f"    Raw:   {train_raw:.4f}")
print(f"    Tuned: {train_tuned:.4f}")
print(f"    Gap (train - OOB): {train_tuned - oob_global_score:.4f}")

print(f"\n  Optimal threshold biases (enriched): [{biases[0]:.6f}, {biases[1]:.6f}, {biases[2]:.6f}]")

print("\n  Feature importances (top 20):")
for name, imp in importances_final.head(20).items():
    tag = " [NEW]" if name in SALVO_FEATURES + GROUP_FEATURES else ""
    rank = list(importances_final.index).index(name) + 1
    print(f"    {rank:>2}. {name:<30} {imp:>8.1f}{tag}")

print("\n  New feature importance ranks:")
for feat in SALVO_FEATURES + GROUP_FEATURES:
    if feat in selected_features:
        rank = list(importances_final.index).index(feat) + 1
        print(f"    {feat:<30} rank={rank:>3}/{len(selected_features)}  imp={importances_final[feat]:.1f}")
    else:
        print(f"    {feat:<30} DROPPED by feature selection")

print("\n" + "=" * 70)
if oob_global_score > BASELINE_SCORE:
    print(f"\n  HYPOTHESIS CONFIRMED: {oob_global_score:.4f} > {BASELINE_SCORE}")
    print("  Salvo/rebel-group features BREAK the kinematic ceiling.")
    print("  -> Promote selected_features and new biases to production.")
    print(f"  -> Update SELECTED_FEATURES in model.py (add: {new_kept})")
elif oob_global_score > base_score:
    print(f"\n  MARGINAL: enriched ({oob_global_score:.4f}) > baseline ({base_score:.4f})")
    print(f"  but the delta ({delta:+.4f}) may reflect CV strategy difference,")
    print(f"  not a genuine breakthrough over the {BASELINE_SCORE} ceiling.")
    print("  -> Compare against GroupKFold-by-traj_ind baseline before promoting.")
else:
    print(f"\n  HYPOTHESIS REJECTED: {oob_global_score:.4f} <= {base_score:.4f}")
    print(f"  The {BASELINE_SCORE} ceiling is confirmed as fundamental data ambiguity.")
    print("  Salvo and rebel-group features add no discriminative signal.")
print("=" * 70)

# ── Save JSON results ──────────────────────────────────────────────────────────

results = dict(
    baseline_score=base_score,
    baseline_fold_scores=base_fold_scores,
    enriched_score=oob_global_score,
    enriched_fold_scores=oob_fold_scores,
    delta=delta,
    selected_features=selected_features,
    dropped_features=dropped,
    new_features_kept=new_kept,
    biases=biases.tolist(),
    best_params=best_params,
    best_optuna_value=best.value,
    n_rebel_groups=int(n_real_groups),
    effective_splits=effective_splits_optuna,
    salvo_purity_mean=float(purity.mean()) if multi_mask.sum() > 0 else None,
    train_raw=float(train_raw),
    train_tuned=float(train_tuned),
)
with open(RESULTS_OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
log.info("Results saved to %s", RESULTS_OUTPUT)
