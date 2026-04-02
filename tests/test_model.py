"""Unit tests for rocket_classifier.model.

Tests cover:
  - min_class_recall correctness (perfect, partial, worst-class bottleneck)
  - RocketClassifier._select_and_impute (no NaN, partial NaN, all-NaN column)
  - RocketClassifier.predict / predict_proba via a stub model (no disk artifacts)
  - Bias adjustment: large bias shifts argmax toward the favoured class
  - SELECTED_FEATURES contract: exactly 61 unique entries
"""

from __future__ import annotations

import numpy as np
import pytest

from rocket_classifier.model import (
    PRODUCTION_BIASES,
    SELECTED_FEATURES,
    RocketClassifier,
    min_class_recall,
)

N_FEATURES = 61


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
    def test_exactly_61_features(self) -> None:
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
        assert preds.dtype == np.dtype("int")
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
