"""Unit tests for rocket_classifier.main orchestration functions.

Tests cover:
  - get_features: Feather/Parquet cache loading, cache rebuild, error paths
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import make_raw_trajectory_df

from rocket_classifier.main import get_features
from rocket_classifier.model import SELECTED_FEATURES


class TestGetFeatures:
    """Tests for the get_features cache/rebuild logic."""

    @staticmethod
    def _make_cached_df() -> pd.DataFrame:
        """Minimal DataFrame that resembles build_features output."""
        data = {f: [1.0, 2.0] for f in SELECTED_FEATURES}
        data["launch_time"] = pd.to_datetime(["2024-01-01", "2024-01-02"])
        return pd.DataFrame(data, index=pd.Index([10, 20], name="traj_ind"))

    def test_loads_feather_cache(self, tmp_path: Path) -> None:
        """Feather cache takes priority over Parquet."""
        import pyarrow.feather as pa_feather

        feats = self._make_cached_df()
        parquet_path = tmp_path / "features.parquet"
        feather_path = tmp_path / "features.feather"
        pa_feather.write_feather(feats, str(feather_path), compression="lz4")

        result = get_features(None, parquet_path, "test")
        assert list(result.columns) == list(feats.columns)
        assert len(result) == 2

    def test_loads_parquet_and_creates_feather_sidecar(self, tmp_path: Path) -> None:
        """Parquet cache loads and eagerly writes a Feather sidecar."""
        feats = self._make_cached_df()
        parquet_path = tmp_path / "features.parquet"
        feather_path = tmp_path / "features.feather"
        feats.to_parquet(parquet_path)

        result = get_features(None, parquet_path, "test")
        assert feather_path.exists(), "Feather sidecar should be created"
        assert len(result) == 2

    def test_no_cache_no_data_raises(self, tmp_path: Path) -> None:
        """Missing cache with no raw DataFrame raises FileNotFoundError."""
        parquet_path = tmp_path / "nonexistent.parquet"
        with pytest.raises(FileNotFoundError, match="Cache not found"):
            get_features(None, parquet_path, "test")

    def test_builds_from_dataframe_and_caches(self, tmp_path: Path) -> None:
        """When no cache exists, builds features from raw data and writes both caches."""
        df = make_raw_trajectory_df(n_traj=5, n_points=4)
        parquet_path = tmp_path / "features.parquet"
        feather_path = tmp_path / "features.feather"

        result = get_features(df, parquet_path, "test")
        assert len(result) == 5
        assert parquet_path.exists(), "Parquet cache should be written"
        assert feather_path.exists(), "Feather sidecar should be written"
        # All SELECTED_FEATURES must be present in the output
        missing = set(SELECTED_FEATURES) - set(result.columns)
        assert not missing, f"Missing features in cache: {missing}"
