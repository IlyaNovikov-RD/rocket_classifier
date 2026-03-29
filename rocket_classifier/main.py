"""Pipeline orchestrator: Load -> Validate -> Features -> Train -> Predict -> Export.

Produces ``submission.csv`` in the project root matching the format of
``data/sample_submission.csv``:

    columns: label, trajectory_ind
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rocket_classifier.features import build_features
from rocket_classifier.model import predict, train_with_cv
from rocket_classifier.schema import validate_dataframe

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "submission.csv"
MODEL_PATH = Path(__file__).parent.parent / "model.pkl"
MEDIANS_PATH = Path(__file__).parent.parent / "train_medians.npy"
BIASES_PATH = Path(__file__).parent.parent / "threshold_biases.npy"
FEATURE_CACHE_TRAIN = Path(__file__).parent.parent / "cache_train_features.parquet"
FEATURE_CACHE_TEST = Path(__file__).parent.parent / "cache_test_features.parquet"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV data from the data directory.

    Returns:
        A tuple of three DataFrames:
            - train: Labelled point-level trajectory data with columns
              ``label``, ``time_stamp``, ``traj_ind``, ``x``, ``y``, ``z``.
            - test: Unlabelled point-level trajectory data (no ``label`` column).
            - sample_sub: Sample submission DataFrame with columns
              ``label`` and ``trajectory_ind``, used for output formatting.
    """
    logger.info("Loading raw data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    logger.info("Train: %d rows, %d trajectories", len(train), train["traj_ind"].nunique())
    logger.info("Test:  %d rows, %d trajectories", len(test), test["traj_ind"].nunique())
    return train, test, sample_sub


def get_features(df: pd.DataFrame, cache_path: Path, name: str) -> pd.DataFrame:
    """Return per-trajectory feature matrix, loading from Parquet cache if available.

    On first run the features are computed from raw point-level data and
    persisted to ``cache_path`` as a Parquet file. Subsequent runs load
    directly from cache, reducing runtime from ~96 seconds to under 1 second
    for the training split.

    Args:
        df: Raw point-level DataFrame passed to ``build_features`` when
            the cache does not exist.
        cache_path: Filesystem path where the Parquet cache is stored or
            will be written.
        name: Human-readable split name (e.g. ``"train"`` or ``"test"``)
            used in log messages.

    Returns:
        Per-trajectory feature DataFrame indexed by ``traj_ind``.
    """
    if cache_path.exists():
        logger.info("Loading %s features from cache: %s", name, cache_path)
        return pd.read_parquet(cache_path)
    logger.info("Engineering %s features (this may take ~1-2 min)...", name)
    t0 = time.time()
    feats = build_features(df)
    logger.info("%s features done in %.1fs. Shape: %s", name, time.time() - t0, feats.shape)
    feats.to_parquet(cache_path)
    return feats


def main() -> None:
    """Run the full classification pipeline end-to-end.

    Executes six sequential stages:
        1. Load raw CSV data.
        2. Validate training data against the Pydantic schema.
        3. Engineer per-trajectory features (with Parquet caching).
        4. Train XGBoost with 5-fold GroupKFold cross-validation.
        5. Generate predictions for the test set.
        6. Export ``submission.csv`` matching the sample submission format.
    """
    t_start = time.time()

    # --- Step 1: Load ---
    train_raw, test_raw, sample_sub = load_data()

    # --- Step 2: Schema validation ---
    _valid, errors = validate_dataframe(train_raw, has_label=True)
    if errors:
        logger.warning(
            "Schema validation: %d/%d rows have issues (pipeline continues with raw data)",
            len(errors),
            len(train_raw),
        )

    # --- Step 3: Feature Engineering ---
    train_feats = get_features(train_raw, FEATURE_CACHE_TRAIN, "train")
    test_feats = get_features(test_raw, FEATURE_CACHE_TEST, "test")

    # Align feature columns (train has 'label', test doesn't)
    feature_cols = [c for c in train_feats.columns if c != "label"]
    X_train = train_feats[feature_cols]
    y_train = train_feats["label"].to_numpy(dtype=int)
    groups = np.array(train_feats.index.tolist())  # traj_ind as group

    logger.info(
        "Label distribution — 0: %d, 1: %d, 2: %d",
        (y_train == 0).sum(),
        (y_train == 1).sum(),
        (y_train == 2).sum(),
    )

    # Align test columns to train (fill any structurally missing with NaN)
    X_test = test_feats.reindex(columns=feature_cols)

    logger.info("Feature matrix — train: %s, test: %s", X_train.shape, X_test.shape)

    # --- Step 4: Train with CV ---
    # NaN imputation now happens inside train_with_cv per-fold (no leakage)
    model, fold_scores, train_medians, biases = train_with_cv(X_train, y_train, groups, n_splits=5)
    logger.info("Final CV min-recall scores: %s", [f"{s:.4f}" for s in fold_scores])

    # Persist the trained model, imputation medians, and threshold biases
    model.save_model(str(MODEL_PATH))
    np.save(MEDIANS_PATH, train_medians)
    np.save(BIASES_PATH, biases)
    logger.info("Model saved to: %s", MODEL_PATH)
    logger.info("Train medians saved to: %s", MEDIANS_PATH)
    logger.info("Threshold biases saved to: %s", BIASES_PATH)

    # --- Step 5: Predict ---
    y_pred = predict(model, X_test, train_medians, biases)
    logger.info(
        "Test predictions — 0: %d, 1: %d, 2: %d",
        (y_pred == 0).sum(),
        (y_pred == 1).sum(),
        (y_pred == 2).sum(),
    )

    # --- Step 6: Export submission ---
    # sample_submission.csv columns: label, trajectory_ind
    submission = pd.DataFrame(
        {
            "trajectory_ind": test_feats.index.tolist(),
            "label": y_pred,
        }
    )

    # Ensure ordering matches sample_submission.csv
    submission = (
        submission.set_index("trajectory_ind").reindex(sample_sub["trajectory_ind"]).reset_index()
    )

    # Match exact column order of sample_submission: label, trajectory_ind
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
    logger.info("Done. submission.csv is ready for review.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
