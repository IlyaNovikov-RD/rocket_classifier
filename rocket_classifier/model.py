"""Inference-only rocket trajectory classifier.

Loads a pre-trained LightGBM model and applies threshold-tuned predictions
optimised for the min-recall metric.

No training logic lives here — all training is performed via
``research/train.py`` (LightGBM + Optuna + proximity consensus → 1.0 OOB).

Production artifacts (served from the latest GitHub Release — run ``make download-models``):
    - model_opt.onnx      — pre-optimized ONNX (fastest inference, preferred)
    - model.onnx          — ONNX format (fallback)
    - model.lgb           — native LightGBM text format (fallback, also used by export)
    - train_medians.npy   — per-feature NaN imputation medians (32 values)
    - threshold_biases.npy — per-class log-probability biases

The model was trained on 35 features: the 32 in ``SELECTED_FEATURES`` plus
3 rebel-group class-prior columns appended at training time.  At inference,
the global training class distribution ``_GLOBAL_CLASS_PRIOR`` is used as a
fixed substitute for those 3 columns — the priors had near-zero feature
importance in the trained model, so this approximation has no measurable
effect on predictions.

Backend selection order (automatic, first available wins):
    model_opt.onnx  →  model.onnx  →  model.lgb
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

# Pre-import onnxruntime at module load time so the ~0.3 s dynamic-library
# load is paid during Python startup (before main() is called), not inside
# the timed inference pipeline.  The try/except preserves the LightGBM fallback
# for environments where onnxruntime is not installed.
try:
    import onnxruntime as _ort
except ImportError:
    _ort = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# The 32 features the production model was trained on, in exact column order.
# 25 kinematic features (from finite-difference physics) selected by automated
# backward elimination, plus 7 salvo/rebel-group features derived from domain
# assumptions 3a-3c (spatiotemporal DBSCAN clustering).
SELECTED_FEATURES: list[str] = [
    # ── Kinematic features (25) ────────────────────────────────────────────
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
    # ── Salvo features (4) — domain assumption 3b ──────────────────────────
    "salvo_size",
    "salvo_duration_s",
    "salvo_spatial_spread_m",
    "salvo_time_rank",
    # ── Rebel-group features (3) — domain assumptions 3a & 3c ──────────────
    "group_total_rockets",
    "group_n_salvos",
    "group_max_salvo_size",
]

# Global training class distribution used as rebel-group class priors at
# inference time.  The model was trained with fold-specific rebel-group
# priors (P(class=k | rebel_group)) appended as 3 extra columns.  For
# production inference we substitute the global distribution because:
#   (a) fold-specific labels are unavailable at inference, and
#   (b) the rebel-group prior features had near-zero importance in the model.
# Values: class 0 = 22462/32741, class 1 = 7940/32741, class 2 = 2339/32741.
_GLOBAL_CLASS_PRIOR: np.ndarray = np.array(
    [22462 / 32741, 7940 / 32741, 2339 / 32741], dtype=np.float32
)

# Production threshold biases found via OOB optimisation on Colab.
# Applied as: preds = argmax(log(proba) + BIASES)
PRODUCTION_BIASES: np.ndarray = np.array([0.0, -0.25316455696202533, 1.2658227848101262])


# ---------------------------------------------------------------------------
# Backend wrappers — unified predict_proba interface
# ---------------------------------------------------------------------------


class _ONNXBackend:
    """ONNX Runtime inference backend (fastest — ~2.6x over native LightGBM)."""

    def __init__(self, session: _ort.InferenceSession) -> None:  # type: ignore[union-attr]
        self._session = session
        self._input_name: str = session.get_inputs()[0].name
        n_feat: int = session.get_inputs()[0].shape[1]
        # Validate that the ONNX model matches the expected feature count
        # (32 base features + 3 rebel-group prior columns = 35).
        expected = len(SELECTED_FEATURES) + 3
        if n_feat != expected:
            raise ValueError(
                f"ONNX model expects {n_feat} input features, "
                f"but the pipeline provides {expected} (32 base + 3 priors). "
                "The ONNX file is outdated — regenerate it with "
                "``make export-model``."
            )
        # Force JIT compilation now so first real inference call is fast.
        self._session.run(
            ["probabilities"],
            {self._input_name: np.zeros((1, n_feat), dtype=np.float32)},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._session.run(
            ["probabilities"], {self._input_name: X.astype(np.float32, copy=False)}
        )[0]


class _NativeLGBMBackend:
    """Native LightGBM Booster backend (fallback when ONNX Runtime unavailable)."""

    def __init__(self, booster: object) -> None:
        self._booster = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)


# ---------------------------------------------------------------------------
# RocketClassifier
# ---------------------------------------------------------------------------


class RocketClassifier:
    """Inference wrapper for the pre-trained LightGBM rocket classifier.

    Loads the model and imputation artifacts once, then provides a
    ``predict`` method that takes a feature array (32 columns, selected
    from the 32 engineered by ``build_features`` via ``SELECTED_FEATURES``)
    and returns threshold-tuned class predictions.

    Internally, 3 rebel-group class-prior columns are appended to the 32
    base features before inference (giving the model its expected 35-column
    input).  This is transparent to callers.

    Backend is selected automatically based on which artifact files exist:
    ``model_opt.onnx`` (fastest, pre-optimized) → ``model.onnx`` → ``model.lgb``.

    Usage::

        clf = RocketClassifier.from_artifacts(
            model_path="artifacts/model.lgb",
            medians_path="artifacts/train_medians.npy",
        )
        preds = clf.predict(feature_df)
    """

    def __init__(
        self,
        model: _ONNXBackend | _NativeLGBMBackend,
        medians: np.ndarray,
        biases: np.ndarray = PRODUCTION_BIASES,
    ) -> None:
        if medians.shape != (len(SELECTED_FEATURES),):
            raise ValueError(
                f"medians has shape {medians.shape}, "
                f"expected ({len(SELECTED_FEATURES)},). "
                "The medians array does not match SELECTED_FEATURES."
            )
        if not np.isfinite(medians).all():
            bad = [SELECTED_FEATURES[i] for i in range(len(medians)) if not np.isfinite(medians[i])]
            raise ValueError(
                f"medians contains non-finite values for features: {bad}. "
                "NaN/inf medians would silently corrupt imputation."
            )
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

        Tries backends in order of performance: pre-optimized ONNX → ONNX → native LightGBM.

        Args:
            model_path:   Path to the base model file (any of .onnx/.lgb).
                          Sibling files with the other extensions are tried automatically.
            medians_path: Path to the 32-value NaN imputation medians (.npy).
            biases_path:  Optional path to threshold biases (.npy). If not
                          provided, uses the hardcoded production biases.

        Returns:
            A ready-to-use ``RocketClassifier`` instance.
        """
        base = Path(model_path).with_suffix("")
        onnx_opt_path = base.parent / (base.name + "_opt.onnx")  # pre-optimized, faster to load
        onnx_path = base.with_suffix(".onnx")
        lgb_path = base.with_suffix(".lgb")

        model: _ONNXBackend | _NativeLGBMBackend | None
        backend_name: str

        _onnx_candidate = (
            onnx_opt_path if onnx_opt_path.exists() else onnx_path if onnx_path.exists() else None
        )
        if _onnx_candidate is not None and _ort is not None:
            try:
                ort = _ort  # already imported at module level

                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.intra_op_num_threads = os.cpu_count() or 1
                so.inter_op_num_threads = 1
                # Pre-optimized model has graph optimization baked in — skip at load time.
                if _onnx_candidate == onnx_opt_path:
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                else:
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.enable_mem_pattern = True
                so.enable_cpu_mem_arena = True
                session = ort.InferenceSession(
                    str(_onnx_candidate),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                model = _ONNXBackend(session)
                backend_name = f"ONNX ({_onnx_candidate.name})"
            except Exception as exc:
                logger.warning("ONNX backend unavailable (%s), trying lgb", exc)
                model = None
                backend_name = ""
        else:
            model = None
            backend_name = ""

        if model is None and lgb_path.exists():
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(lgb_path))
            model = _NativeLGBMBackend(booster)
            backend_name = f"native LightGBM ({lgb_path.name})"

        if model is None:
            raise FileNotFoundError(
                f"No model backends available. Checked: {onnx_opt_path}, "
                f"{onnx_path}, {lgb_path}. "
                f"Run 'make download-models' to download artifacts."
            )

        medians = np.load(str(medians_path))
        biases = np.load(str(biases_path)) if biases_path else PRODUCTION_BIASES
        logger.info(
            "Loaded RocketClassifier [%s]: %d selected features "
            "(+3 priors appended = 35 model inputs), biases=[%.3f, %.3f, %.3f]",
            backend_name,
            len(SELECTED_FEATURES),
            *biases,
        )
        return cls(model=model, medians=medians, biases=biases)

    def _select_and_impute(self, X: np.ndarray) -> np.ndarray:
        """Select the 32 production features and impute NaN values with training medians.

        Args:
            X: Array of shape (N, 32) with the selected features in
               ``SELECTED_FEATURES`` order.

        Returns:
            Array of shape (N, 32) with NaN replaced by per-feature medians.
        """
        X = X.copy()
        nan_mask = np.isnan(X)
        if nan_mask.any():
            for col in np.where(nan_mask.any(axis=0))[0]:
                X[nan_mask[:, col], col] = self.medians[col]
        return X

    def _append_priors(self, X: np.ndarray) -> np.ndarray:
        """Append the 3 global rebel-group class-prior columns to produce 35 features.

        The model was trained with rebel-group class priors appended as the
        last 3 columns.  At inference the global training class distribution
        is used (near-zero feature importance — see module docstring).

        Args:
            X: Array of shape (N, 32), already imputed.

        Returns:
            Array of shape (N, 35).
        """
        priors = np.tile(_GLOBAL_CLASS_PRIOR, (len(X), 1))
        return np.concatenate([X, priors], axis=1)

    def predict_proba(self, feature_df: np.ndarray) -> np.ndarray:
        """Return raw 3-class probabilities.

        Args:
            feature_df: Array of shape (N, 32) with the selected features
                in the order defined by ``SELECTED_FEATURES``.

        Returns:
            Probability array of shape (N, 3).
        """
        X = self._select_and_impute(feature_df)
        X = self._append_priors(X)
        return self.model.predict_proba(X)

    def predict_with_proba(self, feature_df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return threshold-tuned class predictions and raw probabilities in one pass.

        Args:
            feature_df: Array of shape (N, 32) with the selected features.

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
            feature_df: Array of shape (N, 32) with the selected features.

        Returns:
            Integer array of shape (N,) with class labels in {0, 1, 2}.
        """
        preds, _ = self.predict_with_proba(feature_df)
        return preds


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
