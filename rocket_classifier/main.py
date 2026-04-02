"""Inference pipeline: Load features → Classify → Export submission.csv.

This is the production entrypoint. It loads pre-computed feature matrices,
runs the pre-trained LightGBM model with threshold-tuned predictions, and
outputs ``submission.csv`` matching the competition format.

No model training occurs here — the model was trained on Colab H100 via
``research/colab_brute_force_optimization.py`` and is loaded from disk.

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
WEIGHTS_DIR = _ROOT / "weights"
CACHE_DIR = _ROOT / "cache"
OUTPUTS_DIR = _ROOT / "outputs"
OUTPUT_PATH = OUTPUTS_DIR / "submission.csv"
MODEL_PATH = WEIGHTS_DIR / "model.pkl"
MEDIANS_PATH = WEIGHTS_DIR / "train_medians.npy"
BIASES_PATH = WEIGHTS_DIR / "threshold_biases.npy"
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
    """Return per-trajectory feature matrix, loading from cache if available."""
    if cache_path.exists():
        logger.info("Loading %s features from cache: %s", name, cache_path)
        return pd.read_parquet(cache_path)
    if df is None:
        raise FileNotFoundError(
            f"Cache not found at {cache_path} and no raw data provided. "
            "Run: make download-all"
        )
    logger.info("Engineering %s features (this may take ~1-2 min)...", name)
    t0 = time.time()
    feats = build_features(df)
    logger.info("%s features done in %.1fs. Shape: %s", name, time.time() - t0, feats.shape)
    feats.to_parquet(cache_path)
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the inference pipeline end-to-end.

    Stages:
        1. Load raw CSV data (skipped for train if both caches exist).
        2. Validate training data against the Pydantic schema (skipped if cache exists).
        3. Engineer per-trajectory features (with Parquet caching).
        4. Load the pre-trained LightGBM classifier.
        5. Generate predictions for the test set.
        6. Export ``submission.csv``.
    """
    t_start = time.time()
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    both_caches_exist = FEATURE_CACHE_TRAIN.exists() and FEATURE_CACHE_TEST.exists()

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
                "Schema validation: %d/%d rows have issues (pipeline continues)",
                len(errors), len(train_raw),
            )

    # --- Step 3: Feature Engineering ---
    train_feats = get_features(train_raw, FEATURE_CACHE_TRAIN, "train")
    test_feats = get_features(test_raw, FEATURE_CACHE_TEST, "test")

    # --- Step 4: Load classifier ---
    if not MODEL_PATH.exists():
        logger.error(
            "Model not found at %s. Run: make download-weights",
            MODEL_PATH,
        )
        return

    clf = RocketClassifier.from_artifacts(
        model_path=MODEL_PATH,
        medians_path=MEDIANS_PATH,
        biases_path=BIASES_PATH if BIASES_PATH.exists() else None,
    )

    # Select the 61 production features from the 76-column matrix
    X_test = test_feats.reindex(columns=SELECTED_FEATURES).to_numpy(dtype=np.float32)

    # Log label distribution from training data (for reference)
    if "label" in train_feats.columns:
        y_train = train_feats["label"].to_numpy(dtype=int)
        logger.info(
            "Label distribution — 0: %d, 1: %d, 2: %d",
            (y_train == 0).sum(), (y_train == 1).sum(), (y_train == 2).sum(),
        )

    logger.info("Test features: %s", X_test.shape)

    # --- Step 5: Predict ---
    y_pred = clf.predict(X_test)
    logger.info(
        "Test predictions — 0: %d, 1: %d, 2: %d",
        (y_pred == 0).sum(), (y_pred == 1).sum(), (y_pred == 2).sum(),
    )

    # --- Step 6: Export submission ---
    submission = pd.DataFrame({
        "trajectory_ind": test_feats.index.tolist(),
        "label": y_pred,
    })
    submission = (
        submission.set_index("trajectory_ind")
        .reindex(sample_sub["trajectory_ind"])
        .reset_index()
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
