"""Unit tests for rocket_classifier.model.

Tests cover:
  - min_class_recall correctness (perfect, partial, worst-class bottleneck)
  - RocketClassifier._select_and_impute (no NaN, partial NaN, all-NaN column)
  - RocketClassifier.predict / predict_proba via a stub model (no disk artifacts)
  - Bias adjustment: large bias shifts argmax toward the favoured class
  - SELECTED_FEATURES contract: exactly 32 unique entries
  - _GLOBAL_CLASS_PRIOR contract: valid probability distribution
  - Consensus parameter contract: main.py values match training_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rocket_classifier.main import _PROX_POS_PRECISION, _PROX_TIME_WINDOW_S
from rocket_classifier.model import (
    _GLOBAL_CLASS_PRIOR,
    PRODUCTION_BIASES,
    SELECTED_FEATURES,
    RocketClassifier,
    min_class_recall,
)

N_FEATURES = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in for a LightGBM Booster.

    Returns fixed probabilities supplied at construction time so tests are
    fully deterministic without loading any model artifact from disk.
    """

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba.copy()


def _make_clf(proba: np.ndarray) -> RocketClassifier:
    """Build a RocketClassifier backed by a stub model."""
    medians = np.zeros(N_FEATURES)
    return RocketClassifier(model=_StubModel(proba), medians=medians)


# ---------------------------------------------------------------------------
# SELECTED_FEATURES contract
# ---------------------------------------------------------------------------


class TestSelectedFeatures:
    def test_exactly_32_features(self) -> None:
        assert len(SELECTED_FEATURES) == N_FEATURES

    def test_no_duplicates(self) -> None:
        assert len(set(SELECTED_FEATURES)) == len(SELECTED_FEATURES)

    def test_all_strings(self) -> None:
        assert all(isinstance(f, str) and f for f in SELECTED_FEATURES)


# ---------------------------------------------------------------------------
# min_class_recall
# ---------------------------------------------------------------------------


class TestMinClassRecall:
    def test_perfect_predictions(self) -> None:
        y = np.array([0, 0, 1, 1, 2, 2])
        assert min_class_recall(y, y) == pytest.approx(1.0)

    def test_one_class_all_wrong(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 0, 0])  # class 2 never predicted
        assert min_class_recall(y_true, y_pred) == pytest.approx(0.0)

    def test_worst_class_bottleneck(self) -> None:
        # class 0: 2/2=1.0, class 1: 1/2=0.5, class 2: 2/2=1.0 → min=0.5
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        assert min_class_recall(y_true, y_pred) == pytest.approx(0.5)

    def test_single_class(self) -> None:
        y = np.array([1, 1, 1])
        assert min_class_recall(y, y) == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        y = np.array([0, 1, 2])
        result = min_class_recall(y, y)
        assert isinstance(result, float)

    def test_empty_arrays_raises(self) -> None:
        """Empty inputs should raise ValueError (np.min of empty sequence)."""
        with pytest.raises(ValueError):
            min_class_recall(np.array([], dtype=int), np.array([], dtype=int))

    def test_absent_class_ignored(self) -> None:
        """Classes absent from y_true must not affect the metric.

        train.py has a Colab fallback that reimplements this function.
        Both must agree: absent classes are skipped, not counted as 0 recall.
        If they diverge, Optuna optimises a different metric than what is reported.
        """
        # Only class 1 present in y_true — classes 0 and 2 are absent.
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        assert min_class_recall(y_true, y_pred) == pytest.approx(1.0)

        # Partial correct: class 1 recall = 2/3.
        y_pred2 = np.array([1, 1, 0])
        assert min_class_recall(y_true, y_pred2) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# RocketClassifier._select_and_impute
# ---------------------------------------------------------------------------


class TestSelectAndImpute:
    def _make_clf_with_medians(self, medians: np.ndarray) -> RocketClassifier:
        proba = np.array([[1 / 3, 1 / 3, 1 / 3]])
        return RocketClassifier(model=_StubModel(proba), medians=medians)

    def test_no_nan_unchanged(self) -> None:
        medians = np.ones(N_FEATURES) * 99.0
        clf = self._make_clf_with_medians(medians)
        X = np.zeros((3, N_FEATURES))
        result = clf._select_and_impute(X)
        np.testing.assert_array_equal(result, X)

    def test_nan_replaced_by_median(self) -> None:
        medians = np.arange(N_FEATURES, dtype=float)
        clf = self._make_clf_with_medians(medians)
        X = np.zeros((2, N_FEATURES))
        X[0, 5] = np.nan  # single NaN in column 5
        result = clf._select_and_impute(X)
        assert result[0, 5] == pytest.approx(5.0)  # median for col 5 = 5.0
        assert result[1, 5] == pytest.approx(0.0)  # non-NaN row untouched

    def test_all_nan_column_imputed(self) -> None:
        medians = np.full(N_FEATURES, 7.0)
        clf = self._make_clf_with_medians(medians)
        X = np.zeros((3, N_FEATURES))
        X[:, 10] = np.nan  # entire column 10 is NaN
        result = clf._select_and_impute(X)
        np.testing.assert_array_almost_equal(result[:, 10], [7.0, 7.0, 7.0])

    def test_does_not_mutate_input(self) -> None:
        medians = np.zeros(N_FEATURES)
        clf = self._make_clf_with_medians(medians)
        X = np.full((2, N_FEATURES), np.nan)
        X_copy = X.copy()
        clf._select_and_impute(X)
        np.testing.assert_array_equal(X, X_copy)  # original untouched


# ---------------------------------------------------------------------------
# RocketClassifier._append_priors
# ---------------------------------------------------------------------------


class TestAppendPriors:
    def test_output_shape_adds_three_columns(self) -> None:
        """_append_priors must add exactly 3 columns (one per class).

        If _GLOBAL_CLASS_PRIOR shape is wrong or concatenation fails,
        the model receives wrong-shaped input and predicts silently incorrect classes.
        """
        clf = _make_clf(np.ones((1, 3)) / 3)
        X = np.zeros((5, N_FEATURES))
        out = clf._append_priors(X)
        assert out.shape == (5, N_FEATURES + 3)

    def test_appended_values_match_global_prior(self) -> None:
        """Every row of the appended prior columns must equal _GLOBAL_CLASS_PRIOR."""
        clf = _make_clf(np.ones((1, 3)) / 3)
        X = np.zeros((4, N_FEATURES))
        out = clf._append_priors(X)
        prior_cols = out[:, N_FEATURES:]  # last 3 columns
        for row in prior_cols:
            np.testing.assert_allclose(row, _GLOBAL_CLASS_PRIOR, rtol=1e-6)

    def test_original_columns_unchanged(self) -> None:
        """First 32 columns must be unchanged after appending priors."""
        clf = _make_clf(np.ones((1, 3)) / 3)
        X = np.arange(N_FEATURES * 3, dtype=float).reshape(3, N_FEATURES)
        out = clf._append_priors(X)
        np.testing.assert_array_equal(out[:, :N_FEATURES], X)


# ---------------------------------------------------------------------------
# RocketClassifier.predict_proba / predict
# ---------------------------------------------------------------------------


class TestRocketClassifierPredict:
    def test_predict_proba_shape(self) -> None:
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        clf = _make_clf(proba)
        X = np.zeros((2, N_FEATURES))
        out = clf.predict_proba(X)
        assert out.shape == (2, 3)

    def test_predict_returns_int_labels(self) -> None:
        proba = np.array([[0.9, 0.05, 0.05]])
        clf = _make_clf(proba)
        X = np.zeros((1, N_FEATURES))
        preds = clf.predict(X)
        assert np.issubdtype(preds.dtype, np.integer)
        assert preds[0] in {0, 1, 2}

    def test_predict_argmax_no_bias(self) -> None:
        # With zero biases the argmax of proba == argmax of log(proba)
        proba = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1]])
        clf = RocketClassifier(
            model=_StubModel(proba),
            medians=np.zeros(N_FEATURES),
            biases=np.zeros(3),
        )
        X = np.zeros((2, N_FEATURES))
        preds = clf.predict(X)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_bias_shifts_prediction(self) -> None:
        # Class 0 has highest raw probability, but a large bias on class 2
        # should flip the prediction to class 2.
        proba = np.array([[0.6, 0.3, 0.1]])
        biases = np.array([0.0, 0.0, 10.0])  # massive boost to class 2
        clf = RocketClassifier(
            model=_StubModel(proba),
            medians=np.zeros(N_FEATURES),
            biases=biases,
        )
        X = np.zeros((1, N_FEATURES))
        preds = clf.predict(X)
        assert preds[0] == 2

    def test_production_biases_shape(self) -> None:
        assert PRODUCTION_BIASES.shape == (3,)
        assert PRODUCTION_BIASES[0] == pytest.approx(0.0)

    def test_production_biases_all_finite(self) -> None:
        """No bias entry may be NaN or inf — that would silently break argmax."""
        assert np.isfinite(PRODUCTION_BIASES).all()

    def test_model_feature_count(self) -> None:
        """artifacts/model.lgb must expect exactly 35 features (32 selected + 3 priors).

        The LightGBM native backend does not validate feature count at inference
        time — wrong-shaped input produces silently incorrect predictions.
        This test catches the case where a model trained with different features
        is uploaded to artifacts/.

        Skipped if artifacts/model.lgb is not present (CI without downloaded artifacts).
        """
        import lightgbm as lgb

        model_path = Path(__file__).parent.parent / "artifacts" / "model.lgb"
        if not model_path.exists():
            pytest.skip("artifacts/model.lgb not found — skipping feature count check")
        booster = lgb.Booster(model_file=str(model_path))
        expected = len(SELECTED_FEATURES) + 3  # 32 base + 3 rebel-group priors
        assert booster.num_feature() == expected, (
            f"model.lgb expects {booster.num_feature()} features, "
            f"but pipeline provides {expected} (32 selected + 3 priors). "
            "Re-train or re-export the model."
        )

    def test_global_class_prior_is_valid_distribution(self) -> None:
        """_GLOBAL_CLASS_PRIOR must be a valid probability distribution.

        If the class counts in model.py are edited incorrectly, inference
        silently uses wrong priors — this catches that.
        """
        assert _GLOBAL_CLASS_PRIOR.shape == (3,), "Prior must have one entry per class"
        assert (_GLOBAL_CLASS_PRIOR >= 0).all(), "All prior probabilities must be >= 0"
        assert (_GLOBAL_CLASS_PRIOR <= 1).all(), "All prior probabilities must be <= 1"
        assert _GLOBAL_CLASS_PRIOR.sum() == pytest.approx(1.0, abs=1e-6), (
            "Prior probabilities must sum to 1.0"
        )

    def test_consensus_parameters_match_training_report(self) -> None:
        """Production consensus parameters must match what was used during training.

        If _PROX_POS_PRECISION or _PROX_TIME_WINDOW_S in main.py drifts from
        the values in training_report.json, production consensus behaves
        differently from how it was validated.

        Skipped if training_report.json does not exist (CI without artifacts).
        """
        report_path = Path(__file__).parent.parent / "training_report.json"
        if not report_path.exists():
            pytest.skip("training_report.json not found — skipping consensus sync check")
        with open(report_path) as f:
            report = json.load(f)
        assert _PROX_POS_PRECISION == report["prox_pos_precision"], (
            f"main.py _PROX_POS_PRECISION={_PROX_POS_PRECISION} != "
            f"training_report.json prox_pos_precision={report['prox_pos_precision']}"
        )
        assert _PROX_TIME_WINDOW_S == report["prox_time_window_s"], (
            f"main.py _PROX_TIME_WINDOW_S={_PROX_TIME_WINDOW_S} != "
            f"training_report.json prox_time_window_s={report['prox_time_window_s']}"
        )

    def test_predict_with_proba_matches_predict_and_predict_proba(self) -> None:
        """predict_with_proba must return the same results as calling predict and
        predict_proba separately — one pass, same outputs."""
        proba_fixed = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
        clf = _make_clf(proba_fixed)
        X = np.zeros((3, N_FEATURES))
        preds, proba = clf.predict_with_proba(X)
        np.testing.assert_array_equal(preds, clf.predict(X))
        np.testing.assert_array_almost_equal(proba, clf.predict_proba(X))

    def test_predict_with_proba_shapes(self) -> None:
        proba_fixed = np.array([[0.5, 0.3, 0.2]])
        clf = _make_clf(proba_fixed)
        X = np.zeros((1, N_FEATURES))
        preds, proba = clf.predict_with_proba(X)
        assert preds.shape == (1,)
        assert proba.shape == (1, 3)

    def test_predict_with_proba_bias_applied(self) -> None:
        """Bias on class 2 must flip the prediction just as predict() does."""
        proba_fixed = np.array([[0.6, 0.3, 0.1]])
        biases = np.array([0.0, 0.0, 10.0])
        clf = RocketClassifier(
            model=_StubModel(proba_fixed), medians=np.zeros(N_FEATURES), biases=biases
        )
        X = np.zeros((1, N_FEATURES))
        preds, _ = clf.predict_with_proba(X)
        assert preds[0] == 2
