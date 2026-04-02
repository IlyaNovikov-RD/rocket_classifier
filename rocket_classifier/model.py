"""Inference-only rocket trajectory classifier.

Loads a pre-trained LightGBM model and applies threshold-tuned predictions
optimised for the min-recall metric.

No training logic lives here — all training was performed in Colab on
H100 GPU. This module is the production inference entrypoint.

Production artifacts (served from the latest GitHub Release — run ``make download-weights``):
    - model.onnx          — ONNX format (fastest inference, preferred)
    - model.lgb           — native LightGBM text format (fast load, fallback)
    - model.pkl           — joblib-pickled LGBMClassifier (legacy fallback)
    - train_medians.npy   — per-feature NaN imputation medians (61 values)
    - threshold_biases.npy — per-class log-probability biases [0, 1.063, 2.177]

Backend selection order (automatic, first available wins):
    model.onnx  →  model.lgb  →  model.pkl
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# The 61 features the production model was trained on, in exact column order.
# Selected via automated backward elimination (see research/colab_brute_force_optimization.py).
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

# Production threshold biases found via OOB optimisation on Colab.
# Applied as: preds = argmax(log(proba) + BIASES)
PRODUCTION_BIASES: np.ndarray = np.array([0.000000, 1.063291, 2.177215])


# ---------------------------------------------------------------------------
# Backend wrappers — unified predict_proba interface
# ---------------------------------------------------------------------------


class _ONNXBackend:
    """ONNX Runtime inference backend (fastest — 2.6x over sklearn wrapper)."""

    def __init__(self, session: object) -> None:
        self._session = session
        self._input_name: str = session.get_inputs()[0].name  # type: ignore[union-attr]
        # Force JIT compilation now (during session init) rather than on
        # the first real inference call, so pipeline latency is predictable.
        n_feat: int = session.get_inputs()[0].shape[1]  # type: ignore[union-attr]
        self._session.run(  # type: ignore[union-attr]
            ["probabilities"],
            {self._input_name: np.zeros((1, n_feat), dtype=np.float32)},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._session.run(  # type: ignore[union-attr]
            ["probabilities"], {self._input_name: X.astype(np.float32)}
        )[0]


class _NativeLGBMBackend:
    """Native LightGBM Booster backend (1.6x over sklearn wrapper, no sklearn overhead)."""

    def __init__(self, booster: object) -> None:
        self._booster = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# RocketClassifier
# ---------------------------------------------------------------------------


class RocketClassifier:
    """Inference wrapper for the pre-trained LightGBM rocket classifier.

    Loads the model and imputation artifacts once, then provides a
    ``predict`` method that takes a feature array (61 columns, selected
    from the 76 engineered by ``build_features`` via ``SELECTED_FEATURES``)
    and returns threshold-tuned class predictions.

    Backend is selected automatically based on which artifact files exist:
    ``model.onnx`` (fastest) → ``model.lgb`` → ``model.pkl`` (legacy).

    Usage::

        clf = RocketClassifier.from_artifacts(
            model_path="weights/model.pkl",
            medians_path="weights/train_medians.npy",
        )
        preds = clf.predict(feature_df)
    """

    def __init__(
        self,
        model: object,
        medians: np.ndarray,
        biases: np.ndarray = PRODUCTION_BIASES,
    ) -> None:
        self.model = model
        self.medians = medians
        self.biases = biases
        self.feature_names = SELECTED_FEATURES

    @classmethod
    def from_artifacts(
        cls,
        model_path: str | Path,
        medians_path: str | Path,
        biases_path: str | Path | None = None,
    ) -> RocketClassifier:
        """Load a classifier from disk artifacts.

        Tries backends in order of performance: ONNX → native LightGBM → pkl.

        Args:
            model_path:   Path to the base model file (any of .onnx/.lgb/.pkl).
                          Sibling files with the other extensions are tried automatically.
            medians_path: Path to the 61-value NaN imputation medians (.npy).
            biases_path:  Optional path to threshold biases (.npy). If not
                          provided, uses the hardcoded production biases.

        Returns:
            A ready-to-use ``RocketClassifier`` instance.
        """
        base = Path(model_path).with_suffix("")
        onnx_path = base.with_suffix(".onnx")
        lgb_path = base.with_suffix(".lgb")
        pkl_path = base.with_suffix(".pkl")

        model: object
        backend_name: str

        if onnx_path.exists():
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.log_severity_level = 3  # suppress shape mismatch warnings
                # Use all available CPU cores — benchmarked optimal on this hardware.
                # Thread sweep (4-core machine, 30 runs each):
                #   1 thread: 2.29s | 2: 1.43s | 3: 1.33s | 4: 1.24s (best)
                # Beyond cpu_count() there are no more physical cores to use.
                so.intra_op_num_threads = os.cpu_count() or 1
                so.inter_op_num_threads = 1
                # Enable all graph optimisations and memory reuse patterns.
                so.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                so.enable_mem_pattern = True
                so.enable_cpu_mem_arena = True
                session = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                model = _ONNXBackend(session)
                backend_name = f"ONNX ({onnx_path.name})"
            except Exception as exc:
                logger.warning("ONNX backend unavailable (%s), trying lgb/pkl", exc)
                model = None  # type: ignore[assignment]
                backend_name = ""
            else:
                pass  # model set above
        else:
            model = None  # type: ignore[assignment]
            backend_name = ""

        if model is None and lgb_path.exists():
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(lgb_path))
            model = _NativeLGBMBackend(booster)
            backend_name = f"native LightGBM ({lgb_path.name})"

        if model is None:
            model = joblib.load(str(pkl_path))
            backend_name = f"joblib ({pkl_path.name})"

        medians = np.load(str(medians_path))
        biases = np.load(str(biases_path)) if biases_path else PRODUCTION_BIASES
        logger.info(
            "Loaded RocketClassifier [%s]: %d features, biases=[%.3f, %.3f, %.3f]",
            backend_name, len(SELECTED_FEATURES), *biases,
        )
        return cls(model=model, medians=medians, biases=biases)

    def _select_and_impute(self, X: np.ndarray) -> np.ndarray:
        """Select the 61 production features and impute NaN values."""
        X = X.copy()
        nan_mask = np.isnan(X)
        if nan_mask.any():
            for col in np.where(nan_mask.any(axis=0))[0]:
                X[nan_mask[:, col], col] = self.medians[col]
        return X

    def predict_proba(self, feature_df: np.ndarray) -> np.ndarray:
        """Return raw 3-class probabilities.

        Args:
            feature_df: Array of shape (N, 61) with the selected features
                in the order defined by ``SELECTED_FEATURES``.

        Returns:
            Probability array of shape (N, 3).
        """
        X = self._select_and_impute(feature_df)
        return self.model.predict_proba(X)  # type: ignore[union-attr]

    def predict_with_proba(self, feature_df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return threshold-tuned class predictions and raw probabilities in one pass.

        Runs imputation and inference once, then derives both the bias-adjusted
        class labels and the raw per-class probabilities from the same result.

        Args:
            feature_df: Array of shape (N, 61) with the selected features.

        Returns:
            A tuple of:
                - preds: Integer array of shape (N,) with class labels in {0, 1, 2}.
                - proba: Float array of shape (N, 3) with per-class probabilities.
        """
        proba = self.predict_proba(feature_df)
        adjusted = np.log(proba + 1e-12) + self.biases
        return np.argmax(adjusted, axis=1).astype(int), proba

    def predict(self, feature_df: np.ndarray) -> np.ndarray:
        """Return threshold-tuned class predictions.

        Applies the production biases to log-probabilities before argmax,
        shifting decision boundaries toward minority classes to maximise
        the min-recall metric.

        Args:
            feature_df: Array of shape (N, 61) with the selected features.

        Returns:
            Integer array of shape (N,) with class labels in {0, 1, 2}.
        """
        proba = self.predict_proba(feature_df)
        adjusted = np.log(proba + 1e-12) + self.biases
        return np.argmax(adjusted, axis=1).astype(int)


def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the minimum per-class recall across all classes.

    This is the official evaluation metric. The model is scored by its
    *worst* class — not its average.

    Args:
        y_true: Ground-truth integer labels array of shape (N,).
        y_pred: Predicted integer labels array of shape (N,).

    Returns:
        A scalar in [0.0, 1.0].
    """
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(np.sum((y_pred == cls) & mask) / mask.sum()))
    return float(np.min(recalls))
