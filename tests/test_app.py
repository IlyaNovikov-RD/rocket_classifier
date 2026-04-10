"""Unit tests for rocket_classifier.app demo functions.

Tests cover:
  - generate_trajectory: output shape, determinism, altitude clamping
  - classify: feature extraction + prediction via stub model
  - _ensure_artifact: download logic, size validation, skip-existing, error handling
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from conftest import N_FEATURES, StubModel

from rocket_classifier.app import _ensure_artifact, classify, generate_trajectory
from rocket_classifier.model import RocketClassifier

# ---------------------------------------------------------------------------
# generate_trajectory
# ---------------------------------------------------------------------------


class TestGenerateTrajectory:
    def test_output_shapes(self) -> None:
        """Position array is (n, 3) and time array is (n,)."""
        pos, t = generate_trajectory(initial_speed=50.0, thrust_accel=80.0, noise_sigma=0.1)
        assert pos.shape == (100, 3)
        assert t.shape == (100,)

    def test_custom_n_points(self) -> None:
        pos, t = generate_trajectory(50.0, 80.0, 0.1, n=50)
        assert pos.shape == (50, 3)
        assert t.shape == (50,)

    def test_altitude_non_negative(self) -> None:
        """z coordinate must be >= 0 (clamped to ground)."""
        pos, _ = generate_trajectory(50.0, 80.0, 0.1, n=200, dt=0.1)
        assert (pos[:, 2] >= 0).all()

    def test_deterministic_output(self) -> None:
        """Same parameters produce identical output (seeded RNG)."""
        pos1, t1 = generate_trajectory(50.0, 80.0, 0.1)
        pos2, t2 = generate_trajectory(50.0, 80.0, 0.1)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(t1, t2)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_returns_valid_class_and_probabilities(self) -> None:
        """classify returns an integer class in {0,1,2} and a (3,) probability array."""
        proba = np.array([[0.7, 0.2, 0.1]])
        clf = RocketClassifier(model=StubModel(proba), medians=np.zeros(N_FEATURES))
        pos, t = generate_trajectory(50.0, 80.0, 0.1)

        class_idx, prob = classify(clf, pos, t)
        assert class_idx in {0, 1, 2}
        assert prob.shape == (3,)

    def test_model_receives_correct_input_shape(self) -> None:
        """Model must receive (1, 35) input — 32 features + 3 class priors."""
        call_log: list[tuple[int, ...]] = []

        class SpyModel:
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                call_log.append(X.shape)
                return np.array([[0.5, 0.3, 0.2]])

        clf = RocketClassifier(model=SpyModel(), medians=np.zeros(N_FEATURES))
        pos, t = generate_trajectory(50.0, 80.0, 0.1)
        classify(clf, pos, t)

        assert len(call_log) == 1
        assert call_log[0] == (1, N_FEATURES + 3)


# ---------------------------------------------------------------------------
# _ensure_artifact
# ---------------------------------------------------------------------------


class TestEnsureArtifact:
    def test_skips_existing_file(self, tmp_path: Path) -> None:
        """Existing file returns True without any HTTP call."""
        path = tmp_path / "model.lgb"
        path.write_bytes(b"x" * 100)
        result = _ensure_artifact(path, "http://example.com/model.lgb")
        assert result is True

    @patch("rocket_classifier.app.st")
    @patch("rocket_classifier.app.requests.get")
    def test_rejects_small_download(
        self, mock_get: MagicMock, mock_st: MagicMock, tmp_path: Path
    ) -> None:
        """Downloaded file smaller than minimum threshold is rejected."""
        path = tmp_path / "model.lgb"  # min size = 3_000_000

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 100]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = _ensure_artifact(path, "http://example.com/model.lgb")
        assert result is False
        assert not path.exists()

    @patch("rocket_classifier.app.requests.get")
    def test_successful_download(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Successful download writes file atomically via tmp + rename."""
        path = tmp_path / "threshold_biases.npy"  # min size = 100
        content = b"x" * 200

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = _ensure_artifact(path, "http://example.com/threshold_biases.npy")
        assert result is True
        assert path.exists()
        assert path.read_bytes() == content

    @patch("rocket_classifier.app.st")
    @patch("rocket_classifier.app.requests.get")
    def test_handles_request_exception(
        self, mock_get: MagicMock, mock_st: MagicMock, tmp_path: Path
    ) -> None:
        """Network errors return False and leave no file behind."""
        import requests

        path = tmp_path / "model.lgb"
        mock_get.side_effect = requests.RequestException("connection failed")

        result = _ensure_artifact(path, "http://example.com/model.lgb")
        assert result is False
        assert not path.exists()
