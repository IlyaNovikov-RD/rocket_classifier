"""
Transformer Sequence Model — Google Colab (H100 GPU)
=====================================================

A Transformer encoder operating directly on raw radar time series
(x, y, z, dt). Self-attention allows any timestep to directly attend
to any other, making it strictly more expressive than the 1D-CNN for
capturing long-range dependencies (e.g. relating the ignition spike at
t=0 to the apogee shape at t=5s in a single attention head).

Architecture:
    Input: (batch, seq_len, 4) — left-aligned (dx, dy, dz, dt) sequences
    → Linear projection to d_model
    → Learnable positional embeddings
    → N Transformer encoder layers (multi-head self-attention + FFN)
    → [CLS] token pooling
    → FC head → 3-class softprob
    → Post-hoc OOB threshold tuning

Why Transformer over 1D-CNN:
    - Self-attention is O(L²) but for L=256 this is negligible on H100
    - Directly models interactions between any two timesteps in one layer
    - The [CLS] token learns a global trajectory summary without
      information loss from pooling
    - Multi-head attention can simultaneously specialise heads for
      velocity, acceleration, and jerk patterns

Install dependencies (run this cell first):
    !pip install -q scikit-learn pandas numpy pyarrow

PyTorch is pre-installed in Colab. No additional installs needed.

Upload to /content/ before running:
    - train.csv   (raw point-level: traj_ind, time_stamp, x, y, z, label)

Usage:
    %run colab_transformer_model.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# Disable Flash Attention / Memory-Efficient Attention — incompatible with
# this H100 + PyTorch 2.11 + cuDNN configuration under AMP. Falls back to
# the standard math SDPA kernel which is universally supported.
# Use bfloat16 for AMP — H100 has native BF16 hardware (different cuDNN path
# from float16, avoids the "No valid execution plans" SDPA error).
AMP_DTYPE = torch.bfloat16

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
logger = logging.getLogger("rocket-transformer")

# ---------------------------------------------------------------------------
# Configuration — tuned for H100 80 GB
# ---------------------------------------------------------------------------

DATA_PATH = Path("/content/train.csv")
MODEL_OUTPUT = Path("/content/transformer_model.pt")
RESULTS_OUTPUT = Path("/content/transformer_results.json")

N_SPLITS = 10
N_CLASSES = 3
RANDOM_STATE = 42
MAX_SEQ_LEN = 256

# Transformer hyperparameters
D_MODEL = 128           # embedding dimension
N_HEADS = 8             # attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS = 4            # transformer encoder layers
D_FF = 512              # feedforward hidden dimension
DROPOUT = 0.1

# Training
BATCH_SIZE = 512        # H100: large batch
EPOCHS = 80             # more epochs — transformers converge slower than CNNs
LEARNING_RATE = 1e-3    # higher LR with warmup works better for transformers
WARMUP_EPOCHS = 5       # linear LR warmup before cosine decay
WEIGHT_DECAY = 1e-2
PATIENCE = 15

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
    """Coarse → fine → ultra-fine grid search over log-prob biases."""
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
    """Load raw train.csv and build left-aligned padded sequence tensor.

    Each trajectory → sequence of relative displacements (dx, dy, dz, dt).
    Left-aligned: data at positions 0..L-1, zeros pad the end.
    Truncates long trajectories keeping the LAST MAX_SEQ_LEN steps
    (descent phase often most discriminative; ignition is captured via dt).

    Returns (sequences, labels, groups, lengths).
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

        if len(pos) < 2:
            seq = np.zeros((1, 4), dtype=np.float32)
        else:
            dx = np.diff(pos[:, 0])
            dy = np.diff(pos[:, 1])
            dz = np.diff(pos[:, 2])
            dt = np.diff(t_sec)
            dt = np.clip(dt, 1e-6, None)
            seq = np.stack([dx, dy, dz, dt], axis=1).astype(np.float32)

        L = len(seq)
        lengths[i] = min(L, MAX_SEQ_LEN)

        if L > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
            L = MAX_SEQ_LEN

        # Left-align: data at 0..L-1, zeros pad the end
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
    """Z-score normalisation using training-fold statistics only (no leakage)."""
    if stats is None:
        flat = sequences[train_idx].reshape(-1, 4)
        mask = ~(flat == 0).all(axis=1)
        mean = flat[mask].mean(axis=0)
        std = flat[mask].std(axis=0)
        std = np.clip(std, 1e-6, None)
        stats = {"mean": mean, "std": std}

    norm = (sequences - stats["mean"]) / stats["std"]
    return norm.astype(np.float32), stats


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.labels = torch.from_numpy(labels)
        self.lengths = torch.from_numpy(lengths)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class _PreLNLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer without SDPA.

    Uses need_weights=True in MultiheadAttention to force the legacy
    explicit-matmul attention path, bypassing scaled_dot_product_attention
    and the cuDNN Frontend entirely. This sacrifices the flash-attention
    speedup but runs correctly on all hardware/cuDNN versions.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN self-attention — need_weights=True bypasses SDPA/cuDNN
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=True)
        x = x + self.drop(attn_out)
        # Pre-LN feed-forward
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings (better than fixed sinusoidal for
    short fixed-length sequences where position matters absolutely)."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len + 1, d_model)  # +1 for CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class RocketTransformer(nn.Module):
    """Transformer encoder for rocket trajectory classification.

    Uses a prepended [CLS] token whose final hidden state is used for
    classification — the standard BERT-style approach. This is more
    expressive than mean/attention pooling because the CLS token can
    attend to all positions across all layers and aggregate selectively.

    Key design choices:
    - Pre-LN (LayerNorm before attention/FFN): more stable training than
      post-LN, especially important for small datasets
    - Learnable positional embeddings: better than sinusoidal for
      fixed-length sequences
    - GELU activation: smoother than ReLU, standard for transformers
    """

    def __init__(
        self,
        in_features: int = 4,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        max_len: int = MAX_SEQ_LEN,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        # Project raw 4-channel input to d_model
        self.input_proj = nn.Linear(in_features, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (seq_len + 1 for CLS position 0)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.dropout = nn.Dropout(dropout)

        # Custom Pre-LN encoder layers.
        # nn.TransformerEncoderLayer hardcodes need_weights=False which forces
        # SDPA → cuDNN and crashes on H100 + PyTorch 2.11. Our custom layer
        # calls need_weights=True to bypass SDPA entirely.
        self.encoder = nn.ModuleList([
            _PreLNLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head on CLS token
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       (batch, seq_len, 4)
            lengths: unused — kept for API compatibility with DataLoader

        Returns:
            logits: (batch, n_classes)
        """
        B, _L, _ = x.shape

        # Project input features
        x = self.input_proj(x)               # (B, L, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, L+1, d_model)

        # Add positional embeddings
        x = self.pos_enc(x)
        x = self.dropout(x)

        # Custom encoder — each layer uses need_weights=True to bypass SDPA
        for layer in self.encoder:
            x = layer(x)

        # Extract CLS token output
        cls_out = self.norm(x[:, 0])   # (B, d_model)
        return self.head(cls_out)       # (B, n_classes)


# ---------------------------------------------------------------------------
# LR Scheduler with Warmup
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing.

    Transformers train better with a brief warmup phase before the
    cosine decay — avoids early instability from large initial steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-5,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        e = self.last_epoch
        if e < self.warmup_epochs:
            factor = (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = self.min_lr / self.base_lrs[0] + 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (
                1 + math.cos(math.pi * progress)
            )
        return [base_lr * factor for base_lr in self.base_lrs]


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
    model.train()
    total_loss = 0.0

    for seqs, labels, lengths in loader:
        seqs = seqs.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
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
def predict_proba(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    all_proba = []

    for seqs, _, lengths in loader:
        seqs = seqs.to(DEVICE)
        lengths = lengths.to(DEVICE)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            logits = model(seqs, lengths)
        proba = torch.softmax(logits, dim=-1).float().cpu().numpy()
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
) -> np.ndarray:
    """10-fold GroupKFold CV. Returns OOB probability array."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oob_proba = np.zeros((len(labels), N_CLASSES))
    torch.manual_seed(RANDOM_STATE)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(sequences, labels, groups), 1):
        logger.info("─" * 60)
        logger.info("FOLD %d / %d", fold, N_SPLITS)
        logger.info("─" * 60)

        seq_norm, _stats = normalise_sequences(sequences, tr_idx)

        tr_ds = TrajectoryDataset(seq_norm[tr_idx], labels[tr_idx], lengths[tr_idx])
        val_ds = TrajectoryDataset(seq_norm[val_idx], labels[val_idx], lengths[val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)

        class_weights = compute_class_weights(labels[tr_idx])
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = RocketTransformer().to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))

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
                    "  Epoch %3d/%d | loss=%.4f | raw=%.4f | tuned=%.4f | best=%.4f | lr=%.2e",
                    epoch, EPOCHS, train_loss, raw_score, tuned_score, best_val_score,
                    scheduler.get_last_lr()[0],
                )

            if patience_counter >= PATIENCE:
                logger.info("  Early stopping at epoch %d", epoch)
                break

        oob_proba[val_idx] = best_proba
        logger.info("Fold %d complete | best tuned=%.4f", fold, best_val_score)

    return oob_proba


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------

def train_final_model(
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    biases: np.ndarray,
) -> nn.Module:
    """Train final model on 100% of data and save artifacts."""
    logger.info("Training final Transformer on 100%% of data...")

    seq_norm, stats = normalise_sequences(sequences, np.arange(len(sequences)))
    np.save("/content/transformer_norm_stats.npy", np.stack([stats["mean"], stats["std"]]))

    ds = TrajectoryDataset(seq_norm, labels, lengths)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    class_weights = compute_class_weights(labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = RocketTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS,
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, scaler)
        scheduler.step()
        if epoch % 10 == 0 or epoch == EPOCHS:
            logger.info("  Epoch %3d/%d | loss=%.4f | lr=%.2e",
                        epoch, EPOCHS, loss, scheduler.get_last_lr()[0])

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
    logger.info(
        "Model: d_model=%d  n_heads=%d  n_layers=%d  d_ff=%d  params=~%dk",
        D_MODEL, N_HEADS, N_LAYERS, D_FF,
        sum(p.numel() for p in RocketTransformer().parameters()) // 1000,
    )

    # Load data
    sequences, labels, groups, lengths = load_and_preprocess(DATA_PATH)

    # Cross-validation
    logger.info("")
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION (%d-fold GroupKFold)", N_SPLITS)
    logger.info("=" * 60)

    oob_proba = run_cv(sequences, labels, groups, lengths)

    # Global threshold optimisation
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

    # Final model
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL MODEL ON 100%% DATA")
    logger.info("=" * 60)

    train_final_model(sequences, labels, lengths, biases)
    np.save("/content/transformer_biases.npy", biases)

    # Train sanity
    seq_norm, _ = normalise_sequences(sequences, np.arange(len(sequences)))
    full_loader = DataLoader(
        TrajectoryDataset(seq_norm, labels, lengths),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    model = RocketTransformer().to(DEVICE)
    model.load_state_dict(torch.load(str(MODEL_OUTPUT), weights_only=True))
    train_proba = predict_proba(model, full_loader)
    train_raw = min_class_recall(labels, np.argmax(train_proba, axis=1))
    train_tuned = min_class_recall(
        labels, np.argmax(np.log(train_proba + 1e-12) + biases, axis=1),
    )

    # Final report
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  PERFORMANCE:")
    logger.info("    Train Min-Recall (raw):      %.6f", train_raw)
    logger.info("    Train Min-Recall (tuned):    %.6f", train_tuned)
    logger.info("    Val   Min-Recall (OOB mean): %.6f", np.mean(oob_fold_scores))
    logger.info("    Val   Min-Recall (OOB min):  %.6f", np.min(oob_fold_scores))
    logger.info("")
    logger.info("  OOB per-fold: %s", [f"{s:.4f}" for s in oob_fold_scores])
    logger.info("")
    logger.info("  OPTIMAL THRESHOLDS:")
    logger.info("    Class 0 bias: %.6f", biases[0])
    logger.info("    Class 1 bias: %.6f", biases[1])
    logger.info("    Class 2 bias: %.6f", biases[2])
    logger.info("")
    logger.info("  PRODUCTION CODE:")
    logger.info("    stats  = np.load('/content/transformer_norm_stats.npy')")
    logger.info("    biases = np.array([%.6f, %.6f, %.6f])", *biases)
    logger.info("    mean, std = stats[0], stats[1]")
    logger.info("    seq_norm = (seq - mean) / std  # (N, 256, 4), left-aligned")
    logger.info("    proba = model(seq_norm, lengths)")
    logger.info("    preds = np.argmax(np.log(proba + 1e-12) + biases, axis=1)")
    logger.info("")

    lgbm_mean, lgbm_min = 0.9988, 0.9959
    logger.info("-" * 60)
    logger.info("COMPARISON")
    logger.info("-" * 60)
    logger.info("  LightGBM tabular:  OOB mean=%.4f  OOB min=%.4f", lgbm_mean, lgbm_min)
    logger.info("  Transformer CNN:   OOB mean=%.4f  OOB min=%.4f",
                np.mean(oob_fold_scores), np.min(oob_fold_scores))
    logger.info("  Delta mean: %+.4f", np.mean(oob_fold_scores) - lgbm_mean)
    logger.info("")
    logger.info("  Remaining gap to 1.0: %.4f", 1.0 - np.mean(oob_fold_scores))
    logger.info("")
    logger.info("Total runtime: %.1f minutes", (time.time() - t0) / 60)

    results = {
        "model": "transformer",
        "d_model": D_MODEL, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "d_ff": D_FF,
        "oob_fold_scores": oob_fold_scores,
        "oob_mean": float(np.mean(oob_fold_scores)),
        "oob_min": float(np.min(oob_fold_scores)),
        "biases": biases.tolist(),
        "train_raw": float(train_raw),
        "train_tuned": float(train_tuned),
        "lgbm_oob_mean": lgbm_mean,
        "delta_mean": float(np.mean(oob_fold_scores) - lgbm_mean),
    }
    with open(RESULTS_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", RESULTS_OUTPUT)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    main()
