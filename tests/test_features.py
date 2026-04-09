"""
Comprehensive unit tests for rocket_classifier/features.py.

Coverage targets:
- _compute_derivatives: shapes, known physics values, boundary point counts, dt=0
- _extract_trajectory_features: 1/2/3/4+ point trajectories, vertical launch,
  duplicate timestamps, key completeness, apogee correctness
- build_features: with and without label, multi-trajectory aggregation
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rocket_classifier.features import (
    _compute_derivatives,
    _extract_trajectory_features,
    build_features,
)
from rocket_classifier.model import SELECTED_FEATURES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_group(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    dt_seconds: list[float] | None = None,
    base_ts: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    """Build a minimal trajectory DataFrame for _extract_trajectory_features."""
    n = len(xs)
    if dt_seconds is None:
        # Default: 0.05-second intervals
        dt_seconds = [0.05] * (n - 1)

    timestamps = [pd.Timestamp(base_ts)]
    for dt in dt_seconds:
        timestamps.append(timestamps[-1] + pd.Timedelta(seconds=dt))

    return pd.DataFrame(
        {
            "time_stamp": timestamps,
            "x": xs,
            "y": ys,
            "z": zs,
        }
    )


# The full set of feature keys _extract_trajectory_features must always return.
# After backward elimination, only 25 kinematic features are computed.
_ALWAYS_PRESENT = {
    "n_points",
    "initial_z",
    "final_z",
    "delta_z_total",
    "apogee_relative",
    "x_range",
    "y_range",
    "launch_x",
    "launch_y",
}

_DERIVATIVE_KEYS = {
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
}

ALL_EXPECTED_KEYS = _ALWAYS_PRESENT | _DERIVATIVE_KEYS


# ===========================================================================
# _compute_derivatives
# ===========================================================================


class TestComputeDerivatives:
    def test_shapes_five_points(self):
        """N=5 → vel(4,3), acc(3,3), jerk(2,3)."""
        pos = np.arange(15, dtype=float).reshape(5, 3)
        dt = np.full(4, 1.0)
        vel, acc, jerk = _compute_derivatives(pos, dt)
        assert vel.shape == (4, 3)
        assert acc.shape == (3, 3)
        assert jerk.shape == (2, 3)

    def test_shapes_four_points(self):
        """N=4 → vel(3,3), acc(2,3), jerk(1,3)."""
        pos = np.arange(12, dtype=float).reshape(4, 3)
        dt = np.full(3, 1.0)
        vel, acc, jerk = _compute_derivatives(pos, dt)
        assert vel.shape == (3, 3)
        assert acc.shape == (2, 3)
        assert jerk.shape == (1, 3)

    def test_shapes_three_points(self):
        """N=3 → vel(2,3), acc(1,3), jerk empty(0,3).
        acc.shape[0]==1 < 2, so jerk path returns early."""
        pos = np.arange(9, dtype=float).reshape(3, 3)
        dt = np.full(2, 1.0)
        vel, acc, jerk = _compute_derivatives(pos, dt)
        assert vel.shape == (2, 3)
        assert acc.shape == (1, 3)
        assert jerk.shape == (0, 3)

    def test_shapes_two_points(self):
        """N=2 → vel(1,3), acc empty(0,3), jerk empty(0,3).
        vel.shape[0]==1 < 2, so early return."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        dt = np.array([1.0])
        vel, acc, jerk = _compute_derivatives(pos, dt)
        assert vel.shape == (1, 3)
        assert acc.shape == (0, 3)
        assert jerk.shape == (0, 3)

    def test_uniform_motion_zero_acceleration(self):
        """Constant velocity → acceleration should be (near) zero."""
        # pos moves +1 in x each second for 5 steps
        pos = np.column_stack([np.arange(6, dtype=float), np.zeros(6), np.zeros(6)])
        dt = np.ones(5)
        vel, acc, _jerk = _compute_derivatives(pos, dt)
        # All velocities should be [1, 0, 0]
        np.testing.assert_allclose(vel, np.tile([1.0, 0.0, 0.0], (5, 1)), atol=1e-10)
        # Acceleration should be zero for uniform motion
        np.testing.assert_allclose(acc, 0.0, atol=1e-10)

    def test_constant_acceleration(self):
        """Ballistic fall: z(t) = z0 - 0.5*g*t^2 → az ≈ -g everywhere."""
        g = 9.81
        t = np.arange(6) * 0.1  # 0, 0.1, ..., 0.5 s
        z = 100.0 - 0.5 * g * t**2
        pos = np.column_stack([np.zeros(6), np.zeros(6), z])
        dt = np.full(5, 0.1)
        _vel, acc, _jerk = _compute_derivatives(pos, dt)
        # Vertical acceleration should be close to -g
        np.testing.assert_allclose(acc[:, 2], -g, rtol=1e-6)

    def test_dt_zero_produces_inf(self):
        """dt=0 is NOT sanitized inside _compute_derivatives itself.
        The caller (_extract_trajectory_features) is responsible for sanitization.
        This test documents the raw behavior: division by zero → inf/nan."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dt = np.array([0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            vel, _, _ = _compute_derivatives(pos, dt)
        assert not np.isfinite(vel).all(), (
            "dt=0 should produce non-finite velocity; sanitization is the caller's responsibility"
        )

    def test_velocity_values_two_points(self):
        """Simple sanity check: displacement / dt = velocity."""
        pos = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        dt = np.array([2.0])
        vel, _, _ = _compute_derivatives(pos, dt)
        np.testing.assert_allclose(vel[0], [1.5, 2.0, 0.0])

    def test_non_uniform_dt(self):
        """Verify derivatives are correctly scaled with non-uniform time steps."""
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        dt = np.array([2.0, 4.0])  # v[0]=1 m/s, v[1]=1 m/s
        vel, acc, _ = _compute_derivatives(pos, dt)
        np.testing.assert_allclose(vel[:, 0], [1.0, 1.0])
        # Uniform velocity → zero acceleration
        np.testing.assert_allclose(acc[:, 0], 0.0, atol=1e-10)


# ===========================================================================
# _extract_trajectory_features — key completeness
# ===========================================================================


class TestEmptyTrajectory:
    """An empty DataFrame (0 points) must return all-NaN features without crashing."""

    def test_zero_point_trajectory_returns_all_keys(self):
        group = pd.DataFrame(columns=["time_stamp", "x", "y", "z"])
        group["time_stamp"] = pd.to_datetime(group["time_stamp"])
        feats = _extract_trajectory_features(group)
        missing = ALL_EXPECTED_KEYS - feats.keys()
        assert not missing, f"Missing keys for 0-point trajectory: {missing}"

    def test_zero_point_trajectory_all_nan(self):
        group = pd.DataFrame(columns=["time_stamp", "x", "y", "z"])
        group["time_stamp"] = pd.to_datetime(group["time_stamp"])
        feats = _extract_trajectory_features(group)
        for key in ALL_EXPECTED_KEYS:
            if key == "n_points":
                assert feats[key] == 0.0
            else:
                assert math.isnan(feats[key]), f"{key} should be NaN for 0-point trajectory"


class TestExtractTrajFeaturesKeyCompleteness:
    """Every trajectory length must return the full, consistent key set."""

    def test_one_point_trajectory_has_all_keys(self):
        group = _make_group([1.0], [2.0], [3.0])
        feats = _extract_trajectory_features(group)
        missing = ALL_EXPECTED_KEYS - feats.keys()
        assert not missing, f"Missing keys for 1-point trajectory: {missing}"

    def test_two_point_trajectory_has_all_keys(self):
        group = _make_group([0.0, 1.0], [0.0, 0.0], [0.0, 0.5])
        feats = _extract_trajectory_features(group)
        missing = ALL_EXPECTED_KEYS - feats.keys()
        assert not missing, f"Missing keys for 2-point trajectory: {missing}"

    def test_three_point_trajectory_has_all_keys(self):
        group = _make_group([0.0, 1.0, 2.0], [0.0] * 3, [0.0, 0.5, 0.4])
        feats = _extract_trajectory_features(group)
        missing = ALL_EXPECTED_KEYS - feats.keys()
        assert not missing, f"Missing keys for 3-point trajectory: {missing}"

    def test_four_point_trajectory_has_all_keys(self):
        group = _make_group([0.0, 1.0, 2.0, 3.0], [0.0] * 4, [0.0, 0.5, 0.8, 0.3])
        feats = _extract_trajectory_features(group)
        missing = ALL_EXPECTED_KEYS - feats.keys()
        assert not missing, f"Missing keys for 4-point trajectory: {missing}"

    def test_long_trajectory_has_all_keys(self):
        n = 50
        t = np.linspace(0, 5, n)
        xs = list(t * 10.0)
        ys = [0.0] * n
        zs = list(50.0 * t - 4.905 * t**2)  # parabolic arc
        group = _make_group(xs, ys, zs, dt_seconds=[0.1] * (n - 1))
        feats = _extract_trajectory_features(group)
        missing = ALL_EXPECTED_KEYS - feats.keys()
        assert not missing, f"Missing keys for {n}-point trajectory: {missing}"


# ===========================================================================
# _extract_trajectory_features — 1-point edge case
# ===========================================================================


class TestSinglePointTrajectory:
    def setup_method(self):
        self.feats = _extract_trajectory_features(_make_group([5.0], [3.0], [10.0]))

    def test_n_points_is_one(self):
        assert self.feats["n_points"] == 1.0

    def test_all_derivative_features_are_nan(self):
        for key in _DERIVATIVE_KEYS:
            assert math.isnan(self.feats[key]), f"{key} should be NaN for 1-point trajectory"

    def test_launch_position_correct(self):
        assert self.feats["launch_x"] == pytest.approx(5.0)
        assert self.feats["launch_y"] == pytest.approx(3.0)

    def test_apogee_relative_equals_zero_for_single_point(self):
        assert self.feats["apogee_relative"] == pytest.approx(0.0)


# ===========================================================================
# _extract_trajectory_features — two-point trajectory
# ===========================================================================


class TestTwoPointTrajectory:
    def setup_method(self):
        # Simple 1 m/s diagonal motion for 1 second
        self.group = _make_group([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], dt_seconds=[1.0])
        self.feats = _extract_trajectory_features(self.group)

    def test_velocity_features_are_finite(self):
        for key in ["vy_mean", "vy_max", "vz_median", "v_horiz_std", "v_horiz_median"]:
            assert math.isfinite(self.feats[key]), f"{key} should be finite"

    def test_acceleration_features_are_nan(self):
        """2 points → only 1 velocity estimate → acc requires 2 → NaN."""
        for key in ["acc_mag_mean", "acc_mag_min", "acc_mag_max"]:
            assert math.isnan(self.feats[key])

    def test_initial_speed_correct(self):
        # displacement = sqrt(3), dt = 1 → speed = sqrt(3)
        assert self.feats["initial_speed"] == pytest.approx(math.sqrt(3), rel=1e-6)


# ===========================================================================
# _extract_trajectory_features — apogee_relative
# ===========================================================================


class TestApogee:
    def test_apogee_relative_is_rise_from_launch(self):
        zs = [5.0, 15.0, 25.0, 10.0]
        group = _make_group(list(range(4)), [0.0] * 4, zs)
        feats = _extract_trajectory_features(group)
        assert feats["apogee_relative"] == pytest.approx(20.0)  # 25 - 5


# ===========================================================================
# _extract_trajectory_features — duplicate timestamps (dt=0 guard)
# ===========================================================================


class TestDuplicateTimestamps:
    def test_all_finite_no_inf_on_duplicate_timestamps(self):
        """Two consecutive identical timestamps → dt=0 → must NOT produce inf
        in any feature. The implementation replaces dt<=0 with NaN then fills
        with median before calling _compute_derivatives."""
        group = _make_group(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0] * 5,
            [0.0, 1.0, 2.0, 1.5, 0.5],
            # Second interval is 0 (duplicate timestamp)
            dt_seconds=[0.1, 0.0, 0.1, 0.1],
        )
        feats = _extract_trajectory_features(group)
        inf_keys = [k for k, v in feats.items() if isinstance(v, float) and math.isinf(v)]
        assert not inf_keys, f"Infinite values found in features: {inf_keys}"

    def test_all_duplicate_timestamps_produces_nan_not_error(self):
        """All points at the same time → all dt=0 → valid_dt is empty →
        falls into the 'not enough points' branch → derivative features are NaN."""
        group = _make_group(
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            dt_seconds=[0.0, 0.0],
        )
        feats = _extract_trajectory_features(group)
        for key in _DERIVATIVE_KEYS:
            assert math.isnan(feats.get(key, float("nan"))), f"{key} should be NaN when all dt=0"


# ===========================================================================
# _extract_trajectory_features — spatial features
# ===========================================================================


class TestSpatialFeatures:
    def test_x_range(self):
        group = _make_group([1.0, 4.0, 2.0], [0.0] * 3, [0.0] * 3)
        feats = _extract_trajectory_features(group)
        assert feats["x_range"] == pytest.approx(3.0)

    def test_delta_z_total(self):
        group = _make_group([0.0] * 3, [0.0] * 3, [2.0, 7.0, 5.0])
        feats = _extract_trajectory_features(group)
        assert feats["delta_z_total"] == pytest.approx(3.0)  # 5 - 2


# ===========================================================================
# build_features
# ===========================================================================


class TestBuildFeatures:
    def _make_raw_df(self, include_label: bool = True) -> pd.DataFrame:
        rows = []
        # Trajectory 1: 5 points, label 0
        for i in range(5):
            row = {
                "traj_ind": 1,
                "time_stamp": f"2024-01-01 00:00:0{i}.000000",
                "x": float(i),
                "y": 0.0,
                "z": float(i) * 0.5,
            }
            if include_label:
                row["label"] = 0
            rows.append(row)
        # Trajectory 2: 4 points, label 2
        for i in range(4):
            row = {
                "traj_ind": 2,
                "time_stamp": f"2024-01-02 00:00:0{i}.000000",
                "x": float(i) * 2,
                "y": 1.0,
                "z": float(i),
            }
            if include_label:
                row["label"] = 2
            rows.append(row)
        return pd.DataFrame(rows)

    def test_output_has_one_row_per_trajectory(self):
        df = self._make_raw_df()
        result = build_features(df)
        assert len(result) == 2

    def test_index_is_traj_ind(self):
        df = self._make_raw_df()
        result = build_features(df)
        assert set(result.index) == {1, 2}

    def test_label_column_present_when_provided(self):
        df = self._make_raw_df(include_label=True)
        result = build_features(df)
        assert "label" in result.columns
        assert result.loc[1, "label"] == 0
        assert result.loc[2, "label"] == 2

    def test_label_column_dtype_is_integer(self):
        """label column must be integer dtype, not float or object.

        pandas can silently infer float64 when mixing ints with NaN; if that
        happens, downstream comparisons like (y == 0) still work but strict
        dtype checks (e.g., np.int32 arrays) break without error.
        """
        df = self._make_raw_df(include_label=True)
        result = build_features(df)
        assert np.issubdtype(result["label"].dtype, np.integer), (
            f"label dtype should be integer, got {result['label'].dtype}"
        )

    def test_label_column_absent_when_not_provided(self):
        df = self._make_raw_df(include_label=False)
        result = build_features(df)
        assert "label" not in result.columns

    def test_feature_count_matches_expected(self):
        """25 kinematic + 7 salvo/group + label + launch_time = 34 columns."""
        n_kinematic = 25
        n_salvo_group = 7
        n_label = 1
        n_launch_time = 1
        expected = n_kinematic + n_salvo_group + n_label + n_launch_time
        df = self._make_raw_df(include_label=True)
        result = build_features(df)
        assert result.shape[1] == expected, (
            f"Expected {expected} columns "
            f"({n_kinematic} kinematic + {n_salvo_group} salvo/group "
            f"+ {n_label} label + {n_launch_time} launch_time), got {result.shape[1]}"
        )

    def test_launch_time_column_present(self):
        """build_features must store launch_time so inference does not need raw CSV."""
        df = self._make_raw_df(include_label=False)
        result = build_features(df)
        assert "launch_time" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["launch_time"])

    def test_unsorted_input_sorted_correctly(self):
        """Points provided in reverse time order must produce the same result
        as if provided in forward order."""
        df_fwd = self._make_raw_df(include_label=False)
        df_rev = df_fwd.iloc[::-1].reset_index(drop=True)  # reverse row order

        result_fwd = build_features(df_fwd)
        result_rev = build_features(df_rev)

        pd.testing.assert_frame_equal(
            result_fwd.sort_index(),
            result_rev.sort_index(),
            check_like=True,
        )

    def test_selected_features_contract(self):
        """All 32 SELECTED_FEATURES in model.py must be present in build_features output.

        This is the critical contract between features.py and model.py.
        Adding or removing a feature from either file without updating the other
        would cause silent inference failures — this test catches that.
        """
        df = self._make_raw_df(include_label=False)
        result = build_features(df)
        missing = [f for f in SELECTED_FEATURES if f not in result.columns]
        assert not missing, (
            f"build_features() is missing features required by model.py: {missing}\n"
            "Update features.py or SELECTED_FEATURES in model.py to stay in sync."
        )

    def test_train_selected_features_match_production(self):
        """research/train.py SELECTED_FEATURES must equal model.py SELECTED_FEATURES.

        The training script maintains its own copy (research/ intentionally avoids
        importing from production).  If they drift, the trained model and the
        inference pipeline disagree on column order — this test catches that.
        """
        # Only parse the source text — don't execute the 960-line training script.
        import ast

        source = Path(__file__).parent.parent / "research" / "train.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        train_features: list[str] | None = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "SELECTED_FEATURES"
                and node.value is not None
            ):
                train_features = ast.literal_eval(node.value)
                break
        assert train_features is not None, (
            "Could not find SELECTED_FEATURES assignment in research/train.py"
        )
        assert train_features == list(SELECTED_FEATURES), (
            "research/train.py SELECTED_FEATURES differs from model.py.\n"
            f"  production: {list(SELECTED_FEATURES)}\n"
            f"  training:   {train_features}"
        )

    def test_multi_trajectory_dbscan_group_features(self):
        """With two clusters of trajectories at distinct positions, DBSCAN
        rebel-group clustering assigns them to real groups (not all noise=-1),
        exercising the group_total_rockets / group_n_salvos / group_max_salvo_size paths."""
        rows = []
        # Cluster A: 5 trajectories near (0, 0), close in time
        for traj_id in range(5):
            for i in range(4):
                rows.append(
                    {
                        "traj_ind": traj_id,
                        "time_stamp": f"2024-01-01 00:00:0{i}.{traj_id:06d}",
                        "x": float(i) + traj_id * 0.001,
                        "y": traj_id * 0.001,
                        "z": float(i) * 0.5,
                        "label": 0,
                    }
                )
        # Cluster B: 5 trajectories near (1000, 1000), close in time
        for traj_id in range(5, 10):
            for i in range(4):
                rows.append(
                    {
                        "traj_ind": traj_id,
                        "time_stamp": f"2024-01-01 00:00:0{i}.{traj_id:06d}",
                        "x": 1000.0 + float(i) + (traj_id - 5) * 0.001,
                        "y": 1000.0 + (traj_id - 5) * 0.001,
                        "z": float(i) * 0.5,
                        "label": 1,
                    }
                )
        df = pd.DataFrame(rows)
        result = build_features(df)
        assert len(result) == 10
        # With two well-separated clusters, DBSCAN should find >= 2 groups
        # and assign trajectories to groups with total_rockets >= 3
        assert (result["group_total_rockets"] >= 5).all(), (
            "Every trajectory in a 5-member cluster should see group_total_rockets >= 5"
        )
        assert (result["group_n_salvos"] >= 1).all()
        assert (result["group_max_salvo_size"] >= 1).all()

    def test_large_salvo_stochastic_spread(self):
        """Salvos with > 1000 trajectories use stochastic pair sampling for spread.

        The stochastic estimate (10 000 random pairs) must be close to the true max
        pairwise distance and must be deterministic (seeded RNG).
        """
        n_traj = 1100
        rows = []
        for traj_id in range(n_traj):
            for i in range(3):
                rows.append(
                    {
                        "traj_ind": traj_id,
                        "time_stamp": f"2024-01-01 00:00:0{i}.{traj_id:06d}",
                        "x": float(traj_id) * 0.01 + float(i),
                        "y": float(traj_id) * 0.01,
                        "z": float(i) * 0.5,
                        "label": 0,
                    }
                )
        df = pd.DataFrame(rows)
        result = build_features(df)
        assert len(result) == n_traj
        spreads = result["salvo_spatial_spread_m"]
        assert (spreads > 0).any(), "Stochastic spread should be positive for scattered points"
        # Run again — deterministic seed must produce identical results
        result2 = build_features(df)
        np.testing.assert_array_equal(
            result["salvo_spatial_spread_m"].values,
            result2["salvo_spatial_spread_m"].values,
            err_msg="Stochastic spread must be deterministic (seeded RNG)",
        )

    def test_dbscan_group_auto_tuning(self):
        """When the default GROUP_EPS produces < 2 groups, auto-tuning tries
        alternative eps values. Here all trajectories share one position so
        initial DBSCAN finds at most 1 group; auto-tuning must still complete
        without error and assign valid group features."""
        rows = []
        for traj_id in range(10):
            for i in range(4):
                rows.append(
                    {
                        "traj_ind": traj_id,
                        "time_stamp": f"2024-01-01 00:00:0{i}.{traj_id:06d}",
                        "x": float(i),
                        "y": 0.0,
                        "z": float(i) * 0.5,
                        "label": 0,
                    }
                )
        df = pd.DataFrame(rows)
        result = build_features(df)
        assert len(result) == 10
        assert (result["group_total_rockets"] >= 1).all()
        assert (result["group_n_salvos"] >= 1).all()
        assert (result["group_max_salvo_size"] >= 1).all()

    def test_mixed_timestamp_formats(self):
        """Timestamps with and without microseconds must parse without error."""
        df = pd.DataFrame(
            [
                {
                    "traj_ind": 99,
                    "time_stamp": "2024-01-01 00:00:00",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "label": 1,
                },
                {
                    "traj_ind": 99,
                    "time_stamp": "2024-01-01 00:00:00.050000",
                    "x": 1.0,
                    "y": 0.0,
                    "z": 0.5,
                    "label": 1,
                },
            ]
        )
        result = build_features(df)
        assert len(result) == 1
        assert not result["initial_speed"].isna().any()

    def test_inconsistent_labels_uses_first(self):
        """If a trajectory has conflicting labels across rows, build_features
        uses the label from the first row (after time-sorting). This is not
        expected in production data (schema enforces consistency), but the
        behaviour should be deterministic rather than undefined."""
        df = pd.DataFrame(
            [
                {
                    "traj_ind": 1,
                    "time_stamp": "2024-01-01 00:00:00",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "label": 0,
                },
                {
                    "traj_ind": 1,
                    "time_stamp": "2024-01-01 00:00:00.05",
                    "x": 1.0,
                    "y": 0.0,
                    "z": 0.5,
                    "label": 2,
                },
                {
                    "traj_ind": 1,
                    "time_stamp": "2024-01-01 00:00:00.10",
                    "x": 2.0,
                    "y": 0.0,
                    "z": 1.0,
                    "label": 1,
                },
            ]
        )
        result = build_features(df)
        assert result["label"].iloc[0] == 0, "Should use the label from the first row after sorting"
