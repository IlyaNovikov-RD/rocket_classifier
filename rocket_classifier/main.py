"""Inference pipeline: Load features → Classify → Consensus → Export submission.csv.

This is the production entrypoint. It loads pre-computed feature matrices,
runs the pre-trained LightGBM model with threshold-tuned predictions, applies
proximity-based salvo consensus, and outputs ``submission.csv``.

No model training occurs here — training is done via ``research/train.py``.

Usage:
    python -m rocket_classifier.main
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rocket_classifier.features import build_features
from rocket_classifier.model import SELECTED_FEATURES, RocketClassifier
from rocket_classifier.schema import validate_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data"
MODELS_DIR = _ROOT / "models"
CACHE_DIR = _ROOT / "cache"
OUTPUTS_DIR = _ROOT / "outputs"
OUTPUT_PATH = OUTPUTS_DIR / "submission.csv"
MODEL_PATH = MODELS_DIR / "model.lgb"  # from_artifacts resolves .onnx/.lgb in order
MEDIANS_PATH = MODELS_DIR / "train_medians.npy"
BIASES_PATH = MODELS_DIR / "threshold_biases.npy"
FEATURE_CACHE_TRAIN = CACHE_DIR / "cache_train_features.parquet"
FEATURE_CACHE_TEST = CACHE_DIR / "cache_test_features.parquet"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV data from the data directory.

    Returns:
        A tuple of (train, test, sample_sub) DataFrames.
    """
    logger.info("Loading raw data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    logger.info("Train: %d rows, %d trajectories", len(train), train["traj_ind"].nunique())
    logger.info("Test:  %d rows, %d trajectories", len(test), test["traj_ind"].nunique())
    return train, test, sample_sub


def get_features(df: pd.DataFrame | None, cache_path: Path, name: str) -> pd.DataFrame:
    """Return per-trajectory feature matrix, loading from cache if available.

    Storage strategy follows the Apache Arrow recommendation:
    - Parquet  (.parquet) — canonical format for storage and download.
      Compressed, portable, used as the GitHub Release artifact.
    - Feather  (.feather) — Arrow IPC for fast repeated local reads.
      Written eagerly alongside Parquet at cache-build time so it is
      always ready; never written lazily during a pipeline run.
    """
    import pyarrow.feather as pa_feather

    feather_path = cache_path.with_suffix(".feather")

    if feather_path.exists():
        logger.info("Loading %s features from Feather cache: %s", name, feather_path)
        return pa_feather.read_table(str(feather_path), memory_map=True).to_pandas()

    if cache_path.exists():
        logger.info("Loading %s features from Parquet cache: %s", name, cache_path)
        df_feats = pd.read_parquet(cache_path)
        # Eagerly write Feather sidecar so all future reads are fast.
        pa_feather.write_feather(df_feats, str(feather_path), compression="lz4")
        logger.info("Wrote Feather sidecar: %s", feather_path)
        return df_feats

    if df is None:
        raise FileNotFoundError(
            f"Cache not found at {cache_path} and no raw data provided. Run: make download-all"
        )
    logger.info("Engineering %s features (this may take ~1-2 min)...", name)
    t0 = time.time()
    feats = build_features(df)
    logger.info("%s features done in %.1fs. Shape: %s", name, time.time() - t0, feats.shape)
    feats.to_parquet(cache_path)
    pa_feather.write_feather(feats, str(feather_path), compression="lz4")
    logger.info("Wrote Feather sidecar: %s", feather_path)
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Proximity consensus helpers (pure functions — no I/O, testable)
# ---------------------------------------------------------------------------

_PROX_POS_PRECISION = 2   # decimal places for rounding launch_x / launch_y
_PROX_TIME_WINDOW_S = 60  # max total span (s) of a salvo group


def build_proximity_groups(
    launch_x: "pd.Series",
    launch_y: "pd.Series",
    launch_lt_s: "pd.Series",
) -> np.ndarray:
    """Assign each trajectory to a proximity group.

    Rockets at the same rounded launch position fired within
    ``_PROX_TIME_WINDOW_S`` seconds of the group's first rocket belong to
    the same group (absolute-span split — no time-chaining).

    Args:
        launch_x: launch x-coordinates indexed by traj_ind.
        launch_y: launch y-coordinates indexed by traj_ind.
        launch_lt_s: launch timestamps in seconds since epoch, same index.

    Returns:
        Integer array of group IDs aligned with the input index.
    """
    lx_r = launch_x.fillna(0.0).round(_PROX_POS_PRECISION).values
    ly_r = launch_y.fillna(0.0).round(_PROX_POS_PRECISION).values
    lt_s = launch_lt_s.values

    # Sort by (lx_r, ly_r, lt_s) — one global sort instead of per-group sorts.
    order = np.lexsort((lt_s, ly_r, lx_r))
    lx_s, ly_s, lt_sorted = lx_r[order], ly_r[order], lt_s[order]

    # Single linear scan: new group starts when position changes or time exceeds window.
    new_group = np.ones(len(order), dtype=bool)
    salvo_start = lt_sorted[0]
    for i in range(1, len(order)):
        pos_same = lx_s[i] == lx_s[i - 1] and ly_s[i] == ly_s[i - 1]
        if pos_same and lt_sorted[i] - salvo_start <= _PROX_TIME_WINDOW_S:
            new_group[i] = False
        else:
            salvo_start = lt_sorted[i]

    gid_sorted = np.cumsum(new_group, dtype=np.int32) - 1

    # Map back to original positional order: gid_by_orig_pos[i] = group of position i.
    gid_by_orig_pos = np.empty(len(order), dtype=np.int32)
    gid_by_orig_pos[order] = gid_sorted
    return gid_by_orig_pos


def apply_salvo_consensus(y_pred: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Apply strict-majority mode vote within each proximity group of size ≥ 2.

    Args:
        y_pred: integer prediction array of shape (N,).
        group_ids: integer group ID array of shape (N,), aligned with y_pred.

    Returns:
        Copy of ``y_pred`` with borderline predictions corrected by consensus.
    """
    # Vectorized: count class votes per group with np.add.at, then apply
    # strict-majority (top_count > rest) to groups of size ≥ 2.
    _, gid_inverse, gid_counts = np.unique(group_ids, return_inverse=True, return_counts=True)

    n_classes = int(y_pred.max()) + 1
    group_class_votes = np.zeros((len(gid_counts), n_classes), dtype=np.int32)
    np.add.at(group_class_votes, (gid_inverse, y_pred), 1)

    top_class = np.argmax(group_class_votes, axis=1).astype(np.int32)
    top_count = group_class_votes[np.arange(len(gid_counts)), top_class]
    has_majority = top_count * 2 > gid_counts  # strict majority
    apply_consensus = (gid_counts >= 2) & has_majority

    result = y_pred.copy()
    update_mask = apply_consensus[gid_inverse]
    result[update_mask] = top_class[gid_inverse[update_mask]]
    return result


def main() -> None:
    """Run the inference pipeline end-to-end.

    Stages:
        1. Load raw CSV data (skipped for train if both caches exist).
        2. Validate training data against the Pydantic schema (skipped if cache exists).
        3. Engineer per-trajectory features (Feather for speed, Parquet for portability).
        4. Load the pre-trained LightGBM classifier.
        5. Generate predictions for the test set.
        5b. Apply proximity-based salvo consensus (assumptions 3b/3c):
            rockets fired from the same position within 60 s share a class —
            mode voting corrects borderline predictions without using labels.
        6. Export ``submission.csv``.
    """
    t_start = time.time()
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    both_caches_exist = (
        FEATURE_CACHE_TRAIN.exists() or FEATURE_CACHE_TRAIN.with_suffix(".feather").exists()
    ) and (FEATURE_CACHE_TEST.exists() or FEATURE_CACHE_TEST.with_suffix(".feather").exists())

    # --- Step 1: Load ---
    # When both feature caches are present, skip loading train.csv entirely —
    # it is only needed to build features, which are already cached.
    if both_caches_exist:
        logger.info("Feature caches found — skipping train.csv load and schema validation.")
        _sample_sub_path = DATA_DIR / "sample_submission.csv"
        sample_sub = pd.read_csv(_sample_sub_path)
        train_raw = None
        test_raw = None
    else:
        train_raw, test_raw, sample_sub = load_data()

        # --- Step 2: Schema validation (only when rebuilding features) ---
        _valid, errors = validate_dataframe(train_raw, has_label=True)
        if errors:
            logger.warning(
                "Schema validation (train): %d/%d rows have issues (pipeline continues)",
                len(errors),
                len(train_raw),
            )
        _valid_t, errors_t = validate_dataframe(test_raw, has_label=False)
        if errors_t:
            logger.warning(
                "Schema validation (test): %d/%d rows have issues (pipeline continues)",
                len(errors_t),
                len(test_raw),
            )

    # --- Step 3: Feature Engineering ---
    # train_feats is only needed when rebuilding caches (label distribution
    # logging). Skip loading it entirely in the fast path — inference only
    # touches test_feats. train_medians.npy handles NaN imputation.
    if not both_caches_exist:
        train_feats = get_features(train_raw, FEATURE_CACHE_TRAIN, "train")
    test_feats = get_features(test_raw, FEATURE_CACHE_TEST, "test")

    # --- Step 4: Load classifier ---
    if not MODEL_PATH.exists():
        logger.error(
            "Model not found at %s. Run: make download-models",
            MODEL_PATH,
        )
        return

    clf = RocketClassifier.from_artifacts(
        model_path=MODEL_PATH,
        medians_path=MEDIANS_PATH,
        biases_path=BIASES_PATH if BIASES_PATH.exists() else None,
    )

    # Select the 32 production features from the 83-column matrix.
    # If the cache was built with an older version of features.py that lacked
    # salvo/group columns, reindex fills them with NaN — they will be imputed
    # from train_medians.npy.  Delete cache/*.parquet and re-run to rebuild.
    X_test = test_feats.reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32)

    # Log label distribution only when train features were loaded (cache rebuild path)
    if not both_caches_exist and "label" in train_feats.columns:
        y_train = train_feats["label"].to_numpy(dtype=int)
        logger.info(
            "Label distribution — 0: %d, 1: %d, 2: %d",
            (y_train == 0).sum(),
            (y_train == 1).sum(),
            (y_train == 2).sum(),
        )

    logger.info("Test features: %s", X_test.shape)

    # --- Step 5: Predict ---
    y_pred = clf.predict(X_test)
    logger.info(
        "Test predictions — 0: %d, 1: %d, 2: %d",
        (y_pred == 0).sum(),
        (y_pred == 1).sum(),
        (y_pred == 2).sum(),
    )

    # --- Step 5b: Proximity consensus ---
    # Assumptions 3b/3c: rockets in a salvo share the same launcher → same type.
    # Group by launch position + 60 s window, apply mode vote (strict majority).
    # Parameters match research/train.py exactly.

    # Get launch timestamps for proximity grouping.
    # Preferred source: launch_time column in the feature cache (build_features
    # stores it so the raw CSV is not needed at inference time).
    # Fallback: re-derive from test_raw or test.csv for caches built before
    # launch_time was added to features.py.
    if "launch_time" in test_feats.columns:
        _launch_lt_s = test_feats["launch_time"].astype(np.int64) / 1e9
    else:
        _ts_src = test_raw if test_raw is not None else pd.read_csv(
            DATA_DIR / "test.csv", usecols=["traj_ind", "time_stamp"]
        )
        _ts_src = _ts_src.copy()
        _ts_src["time_stamp"] = pd.to_datetime(_ts_src["time_stamp"], format="mixed")
        _launch_lt_s = (
            _ts_src.groupby("traj_ind")["time_stamp"]
            .min()
            .reindex(test_feats.index)
            .astype(np.int64) / 1e9
        )

    _group_ids = build_proximity_groups(
        test_feats["launch_x"], test_feats["launch_y"], _launch_lt_s
    )
    y_pred = apply_salvo_consensus(y_pred, _group_ids)

    logger.info(
        "Post-consensus predictions — 0: %d, 1: %d, 2: %d",
        (y_pred == 0).sum(),
        (y_pred == 1).sum(),
        (y_pred == 2).sum(),
    )

    # --- Step 6: Export submission ---
    submission = pd.DataFrame(
        {
            "trajectory_ind": test_feats.index.tolist(),
            "label": y_pred,
        }
    )
    submission = (
        submission.set_index("trajectory_ind").reindex(sample_sub["trajectory_ind"]).reset_index()
    )
    submission = submission[["label", "trajectory_ind"]]
    submission.to_csv(OUTPUT_PATH, index=False)

    logger.info("submission.csv written to: %s", OUTPUT_PATH)
    logger.info(
        "Submission label dist — 0: %d, 1: %d, 2: %d",
        (submission["label"] == 0).sum(),
        (submission["label"] == 1).sum(),
        (submission["label"] == 2).sum(),
    )
    logger.info("Total pipeline time: %.1fs", time.time() - t_start)
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
