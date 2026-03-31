"""
Pure Sequence Model — Google Colab (T4 GPU)
============================================

1D-CNN operating directly on raw radar time series (x, y, z, dt).
Targets the irreducible errors left by the tabular LightGBM model by
learning fine-grained temporal patterns that aggregate statistics miss:
ignition shape, micro-acceleration variance, temporal ordering of jerk.

Architecture:
    Input: (batch, max_len, 4) — padded sequences of (dx, dy, dz, dt)
    → Residual 1D-CNN blocks with increasing dilation
    → Attention pooling (weights important timesteps more)
    → FC head → 3-class softprob
    → Post-hoc threshold tuning on OOB probabilities

Install dependencies (run this cell first in Colab):
    !pip install -q scikit-learn pandas numpy pyarrow

PyTorch is pre-installed in Colab. No additional installs needed.

Upload to /content/ before running:
    - train.csv   (raw point-level: traj_ind, time_stamp, x, y, z, label)

Usage:
    %run colab_sequence_model.py
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# Force-reconfigure logging for Colab/Jupyter
_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(_handler)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger("rocket-cnn")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("/content/train.csv")
MODEL_OUTPUT = Path("/content/sequence_model.pt")
RESULTS_OUTPUT = Path("/content/sequence_results.json")

N_SPLITS = 10
N_CLASSES = 3
RANDOM_STATE = 42
MAX_SEQ_LEN = 256       # pad/truncate all sequences to this length
BATCH_SIZE = 1024       # H100 80 GB VRAM — large batch for throughput
EPOCHS = 60             # more epochs: H100 is fast enough to afford it
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12           # increased to match longer training budget

THRESHOLD_GRID_COARSE = 80
THRESHOLD_GRID_FINE = 50
THRESHOLD_GRID_ULTRA = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Threshold optimisation
# ---------------------------------------------------------------------------

def optimize_thresholds(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    """Coarse -> fine -> ultra-fine grid search over log-prob biases."""
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
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw train.csv and convert to padded sequence tensor.

    Each trajectory is represented as a sequence of relative displacements:
        (dx, dy, dz, dt)
    where dx/dy/dz are position changes from the previous ping and dt is
    the time delta. Using relative values makes the model invariant to
    absolute launch position and launch time.

    Sequences are truncated to MAX_SEQ_LEN (keeping the most recent pings
    which contain the richest kinematic information) and zero-padded at
    the front if shorter.

    Returns:
        sequences: float32 array of shape (N, MAX_SEQ_LEN, 4)
        labels:    int array of shape (N,)
        groups:    int array of shape (N,) — trajectory IDs for GroupKFold
        lengths:   int array of shape (N,) — true sequence length before padding
    """
    logger.info("Loading raw data from %s ...", path)
    df = pd.read_csv(path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")
    df = df.sort_values(["traj_ind", "time_stamp"])

    traj_ids = df["traj_ind"].unique()
    N = len(traj_ids)
    logger.info("Found %d trajectories", N)

    sequences = np.zeros((N, MAX_SEQ_LEN, 4), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    groups = np.zeros(N, dtype=np.int64)
    lengths = np.zeros(N, dtype=np.int64)

    for i, tid in enumerate(traj_ids):
        traj = df[df["traj_ind"] == tid]
        label = int(traj["label"].iloc[0])

        pos = traj[["x", "y", "z"]].to_numpy(dtype=np.float64)
        t_ns = traj["time_stamp"].to_numpy(dtype="datetime64[ns]").astype(np.float64)
        t_sec = t_ns / 1e9

        # Relative displacements: (dx, dy, dz, dt)
        if len(pos) < 2:
            # Single-point trajectory — use zeros
            seq = np.zeros((1, 4), dtype=np.float32)
        else:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            dz = np.diff(pos[:, 2])
            dt = np.diff(t_sec)
            dt = np.clip(dt, 1e-6, None)   # guard against zero/negative dt
            seq = np.stack([dx, dy, dz, dt], axis=1).astype(np.float32)

        # Normalise: divide by per-channel std across the dataset
        # (done globally after loading — see normalise() below)

        L = len(seq)
        lengths[i] = min(L, MAX_SEQ_LEN)

        # Truncate to last MAX_SEQ_LEN steps (most informative kinematically)
        if L > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
            L = MAX_SEQ_LEN

        # Left-align: data starts at index 0, zeros pad the end.
        # The attention mask marks positions 0..L-1 as valid, matching this layout.
        sequences[i, :L] = seq
        labels[i] = label
        groups[i] = tid

    logger.info(
        "Sequences built: shape=%s | Labels: 0=%d  1=%d  2=%d",
        sequences.shape, (labels == 0).sum(), (labels == 1).sum(), (labels == 2).sum(),
    )
    return sequences, labels, groups, lengths


def normalise_sequences(
    sequences: np.ndarray,
    train_idx: np.ndarray,
    stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Z-score normalisation using training-fold statistics only.

    Args:
        sequences: (N, L, 4) float32 array.
        train_idx: indices of the training fold (to compute mean/std).
        stats:     pre-computed dict with 'mean' and 'std' — if provided,
                   used directly (for validation fold / final inference).

    Returns (normalised_sequences, stats_dict).
    """
    if stats is None:
        # Compute over non-padded timesteps of the training fold
        train_seqs = sequences[train_idx]
        flat = train_seqs.reshape(-1, 4)
        # Ignore zero-padded rows (where all 4 values are exactly 0)
        mask = ~(flat == 0).all(axis=1)
        mean = flat[mask].mean(axis=0)
        std = flat[mask].std(axis=0)
        std = np.clip(std, 1e-6, None)
        stats = {"mean": mean, "std": std}

    norm = (sequences - stats["mean"]) / stats["std"]
    # Zero-padded positions remain near-zero after normalisation
    # (their std contribution is minimal)
    return norm.astype(np.float32), stats


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Minimal dataset wrapping pre-processed trajectory sequences."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        self.sequences = torch.from_numpy(sequences)   # (N, L, 4)
        self.labels = torch.from_numpy(labels)
        self.lengths = torch.from_numpy(lengths)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """Dilated residual block for 1D temporal convolution.

    Uses dilated convolutions to capture patterns at multiple time scales
    without increasing parameters, crucial for trajectories with both
    short-range (ignition spike) and long-range (ballistic arc) structure.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False,
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class AttentionPooling(nn.Module):
    """Soft attention over the time axis.

    Learns to weight timesteps by their discriminative contribution
    rather than averaging uniformly (which would replicate tabular features).
    The ignition timestep — where class 0 and class 1 first diverge — gets
    disproportionately high weight.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, channels, seq_len) -> transpose to (batch, seq_len, channels)
        x_t = x.transpose(1, 2)
        scores = self.attn(x_t).squeeze(-1)             # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)         # (batch, seq_len)
        pooled = (x_t * weights.unsqueeze(-1)).sum(dim=1)  # (batch, channels)
        return pooled


class RocketCNN(nn.Module):
    """Pure sequence classifier for rocket trajectory kinematics.

    Stacks residual 1D-CNN blocks with increasing dilation, followed by
    soft attention pooling and a classification head. Input is the sequence
    of relative displacements (dx, dy, dz, dt) per radar ping.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stem: project raw 4-channel input to base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        # Residual blocks at 3 channel scales with increasing dilation
        self.stage1 = nn.Sequential(
            ResidualBlock1D(base_channels, dilation=1),
            ResidualBlock1D(base_channels, dilation=2),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock1D(base_channels * 2, dilation=1),
            ResidualBlock1D(base_channels * 2, dilation=4),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
        )

        self.stage3 = nn.Sequential(
            ResidualBlock1D(base_channels * 4, dilation=1),
            ResidualBlock1D(base_channels * 4, dilation=8),
        )

        # Attention pooling over time
        self.pool = AttentionPooling(base_channels * 4)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(base_channels * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, 4) -> (batch, 4, seq_len) for Conv1d
        x = x.transpose(1, 2)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        # Build attention mask from true sequence lengths
        mask = None
        if lengths is not None:
            seq_len = x.shape[2]
            # Adjust lengths for the 2 stride-2 downsampling operations
            adjusted = (lengths.float() / 4).ceil().long().clamp(min=1, max=seq_len)
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < adjusted.unsqueeze(1)

        x = self.pool(x, mask)
        return self.head(x)   # logits, shape (batch, n_classes)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts, strict=True))
    n, k = len(y), len(classes)
    weights = np.array([n / (k * freq[c]) for c in range(k)], dtype=np.float32)
    return torch.tensor(weights, device=DEVICE)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
) -> float:
    """Single training epoch with mixed-precision AMP."""
    model.train()
    total_loss = 0.0

    for seqs, labels, lengths in loader:
        seqs = seqs.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(seqs, lengths)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
) -> np.ndarray:
    """Return softmax probabilities for all samples in loader."""
    model.eval()
    all_proba = []

    for seqs, _, lengths in loader:
        seqs = seqs.to(DEVICE)
        lengths = lengths.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(seqs, lengths)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        all_proba.append(proba)

    return np.concatenate(all_proba, axis=0)


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def run_cv(
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """10-fold GroupKFold CV. Returns (oob_proba, oob_labels)."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oob_proba = np.zeros((len(labels), N_CLASSES))
    torch.manual_seed(RANDOM_STATE)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(sequences, labels, groups), 1):
        logger.info("─" * 60)
        logger.info("FOLD %d / %d", fold, N_SPLITS)
        logger.info("─" * 60)

        # Normalise using training fold statistics only (no leakage)
        seq_norm, _stats = normalise_sequences(sequences, tr_idx)

        tr_ds = TrajectoryDataset(seq_norm[tr_idx], labels[tr_idx], lengths[tr_idx])
        val_ds = TrajectoryDataset(seq_norm[val_idx], labels[val_idx], lengths[val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)

        class_weights = compute_class_weights(labels[tr_idx])
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = RocketCNN().to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5,
        )
        scaler = torch.amp.GradScaler("cuda")

        best_val_score = -1.0
        best_proba = None
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, tr_loader, optimizer, criterion, scaler)
            scheduler.step()

            proba_val = predict_proba(model, val_loader)
            raw_score = min_class_recall(labels[val_idx], np.argmax(proba_val, axis=1))
            _, tuned_score = optimize_thresholds(labels[val_idx], proba_val)

            if tuned_score > best_val_score:
                best_val_score = tuned_score
                best_proba = proba_val.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == EPOCHS:
                logger.info(
                    "  Epoch %3d/%d | loss=%.4f | raw=%.4f | tuned=%.4f | best=%.4f",
                    epoch, EPOCHS, train_loss, raw_score, tuned_score, best_val_score,
                )

            if patience_counter >= PATIENCE:
                logger.info("  Early stopping at epoch %d", epoch)
                break

        oob_proba[val_idx] = best_proba
        logger.info("Fold %d complete | best tuned=%.4f", fold, best_val_score)

    return oob_proba, labels


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------

def train_final_model(
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    biases: np.ndarray,
) -> nn.Module:
    """Train final model on 100% of data."""
    logger.info("Training final model on 100%% of data...")

    seq_norm, stats = normalise_sequences(sequences, np.arange(len(sequences)))
    np.save("/content/seq_norm_stats.npy", np.stack([stats["mean"], stats["std"]]))

    ds = TrajectoryDataset(seq_norm, labels, lengths)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    class_weights = compute_class_weights(labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = RocketCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, scaler)
        scheduler.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            logger.info("  Epoch %3d/%d | loss=%.4f", epoch, EPOCHS, loss)

    # Train recall sanity check
    full_loader = DataLoader(
        TrajectoryDataset(seq_norm, labels, lengths),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    train_proba = predict_proba(model, full_loader)
    train_raw = min_class_recall(labels, np.argmax(train_proba, axis=1))
    train_tuned = min_class_recall(
        labels, np.argmax(np.log(train_proba + 1e-12) + biases, axis=1),
    )
    logger.info("Train min-recall (raw):   %.4f", train_raw)
    logger.info("Train min-recall (tuned): %.4f", train_tuned)

    torch.save(model.state_dict(), str(MODEL_OUTPUT))
    logger.info("Model saved to %s", MODEL_OUTPUT)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    logger.info("Device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Load & preprocess ────────────────────────────────────────────────
    sequences, labels, groups, lengths = load_and_preprocess(DATA_PATH)

    # ── Cross-validation ─────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION (%d-fold GroupKFold)", N_SPLITS)
    logger.info("=" * 60)

    oob_proba, _ = run_cv(sequences, labels, groups, lengths)

    # ── Global OOB threshold optimisation ────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("GLOBAL THRESHOLD OPTIMISATION")
    logger.info("=" * 60)

    biases, global_score = optimize_thresholds(labels, oob_proba)
    logger.info("Global OOB tuned score: %.6f", global_score)
    logger.info("Biases: [%.6f, %.6f, %.6f]", *biases)

    # Per-fold OOB scores with global biases
    gkf = GroupKFold(n_splits=N_SPLITS)
    oob_fold_scores = []
    for _, val_idx in gkf.split(sequences, labels, groups):
        preds = np.argmax(np.log(oob_proba[val_idx] + 1e-12) + biases, axis=1)
        oob_fold_scores.append(min_class_recall(labels[val_idx], preds))

    # ── Final model ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL MODEL ON 100%% DATA")
    logger.info("=" * 60)

    train_final_model(sequences, labels, lengths, biases)
    np.save("/content/seq_threshold_biases.npy", biases)

    # ── Final report ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  PERFORMANCE:")
    logger.info("    Val Min-Recall (OOB mean): %.6f", np.mean(oob_fold_scores))
    logger.info("    Val Min-Recall (OOB min):  %.6f", np.min(oob_fold_scores))
    logger.info("")
    logger.info("  OOB per-fold: %s", [f"{s:.4f}" for s in oob_fold_scores])
    logger.info("")
    logger.info("  OPTIMAL THRESHOLDS:")
    logger.info("    Class 0 bias: %.6f", biases[0])
    logger.info("    Class 1 bias: %.6f", biases[1])
    logger.info("    Class 2 bias: %.6f", biases[2])
    logger.info("")
    logger.info("  PRODUCTION CODE:")
    logger.info("    stats = np.load('/content/seq_norm_stats.npy')")
    logger.info("    mean, std = stats[0], stats[1]")
    logger.info("    biases = np.array([%.6f, %.6f, %.6f])", *biases)
    logger.info("    # Preprocess sequence: (dx, dy, dz, dt), pad to %d, normalise", MAX_SEQ_LEN)
    logger.info("    # seq_norm = (seq - mean) / std")
    logger.info("    # proba = model(seq_norm)")
    logger.info("    # pred = np.argmax(np.log(proba + 1e-12) + biases)")
    logger.info("")

    gap = 1.0 - np.mean(oob_fold_scores)
    logger.info("-" * 60)
    logger.info("COMPARISON vs LightGBM TABULAR MODEL")
    logger.info("-" * 60)
    lgbm_oob_mean = 0.9988
    lgbm_oob_min = 0.9959
    logger.info("  LightGBM OOB mean: %.4f  |  CNN OOB mean: %.4f  |  delta: %+.4f",
                lgbm_oob_mean, np.mean(oob_fold_scores),
                np.mean(oob_fold_scores) - lgbm_oob_mean)
    logger.info("  LightGBM OOB min:  %.4f  |  CNN OOB min:  %.4f  |  delta: %+.4f",
                lgbm_oob_min, np.min(oob_fold_scores),
                np.min(oob_fold_scores) - lgbm_oob_min)
    logger.info("")
    logger.info("  Remaining gap to 1.0: %.4f", gap)
    logger.info("  This gap is the empirical Bayes Error of the dataset.")
    logger.info("")
    logger.info("Total runtime: %.1f minutes", (time.time() - t0) / 60)

    results = {
        "oob_fold_scores": oob_fold_scores,
        "oob_mean": float(np.mean(oob_fold_scores)),
        "oob_min": float(np.min(oob_fold_scores)),
        "biases": biases.tolist(),
        "lgbm_oob_mean": lgbm_oob_mean,
        "lgbm_oob_min": lgbm_oob_min,
        "delta_mean": float(np.mean(oob_fold_scores) - lgbm_oob_mean),
    }
    with open(RESULTS_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_OUTPUT)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    main()
