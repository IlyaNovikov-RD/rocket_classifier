"""Unit tests for rocket_classifier.schema.

Tests cover TrajectoryPoint validation and validate_dataframe behaviour
for both training (with label) and inference (without label) inputs.
"""

from __future__ import annotations

import logging
import math

import pandas as pd
import pytest
from pydantic import ValidationError

from rocket_classifier.schema import TrajectoryPoint, validate_dataframe

# ---------------------------------------------------------------------------
# TrajectoryPoint field validation
# ---------------------------------------------------------------------------


class TestTrajectoryPointValid:
    def test_minimal_valid_point(self) -> None:
        p = TrajectoryPoint(
            traj_ind=1,
            time_stamp="2024-01-01T00:00:00",
            x=100.0,
            y=200.0,
            z=50.0,
        )
        assert p.traj_ind == 1
        assert p.z == 50.0
        assert p.label is None

    def test_valid_with_label(self) -> None:
        for label in (0, 1, 2):
            p = TrajectoryPoint(
                traj_ind=0,
                time_stamp="2024-06-15T12:30:00",
                x=0.0,
                y=0.0,
                z=0.0,
                label=label,
            )
            assert p.label == label

    def test_zero_altitude_allowed(self) -> None:
        p = TrajectoryPoint(
            traj_ind=1,
            time_stamp="2024-01-01",
            x=0.0,
            y=0.0,
            z=0.0,
        )
        assert p.z == 0.0


class TestTrajectoryPointInvalid:
    def test_negative_traj_ind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=-1,
                time_stamp="2024-01-01",
                x=0.0,
                y=0.0,
                z=0.0,
            )

    def test_large_negative_altitude_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=1,
                time_stamp="2024-01-01",
                x=0.0,
                y=0.0,
                z=-2.0,
            )

    def test_sensor_noise_z_clamped_to_zero(self) -> None:
        """Tiny negative z (sensor noise) is clamped to 0, not rejected."""
        pt = TrajectoryPoint(
            traj_ind=1,
            time_stamp="2024-01-01",
            x=0.0,
            y=0.0,
            z=-0.004,
        )
        assert pt.z == 0.0

    def test_invalid_label_rejected(self) -> None:
        for bad_label in (3, -1, 99):
            with pytest.raises(ValidationError):
                TrajectoryPoint(
                    traj_ind=1,
                    time_stamp="2024-01-01",
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    label=bad_label,
                )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=1,
                time_stamp="2024-01-01",
                x=0.0,
                y=0.0,
                z=0.0,
                unknown_field="bad",
            )

    def test_float_label_truncated_to_int(self) -> None:
        """label=0.7 is truncated to 0 via int(v) — verify this behaviour."""
        p = TrajectoryPoint(
            traj_ind=0,
            time_stamp="2024-01-01",
            x=0.0,
            y=0.0,
            z=0.0,
            label=0.7,
        )
        assert p.label == 0

    def test_float_label_out_of_range_rejected(self) -> None:
        """label=3.5 truncates to 3 which is not in {0,1,2} → rejected."""
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=0,
                time_stamp="2024-01-01",
                x=0.0,
                y=0.0,
                z=0.0,
                label=3.5,
            )

    def test_nan_coordinate_accepted(self) -> None:
        """NaN coordinates pass Pydantic float validation (no explicit NaN rejection)."""
        p = TrajectoryPoint(
            traj_ind=0,
            time_stamp="2024-01-01",
            x=float("nan"),
            y=0.0,
            z=0.0,
        )
        assert math.isnan(p.x)

    def test_high_altitude_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Altitude > 100 km should emit a warning but not reject the point."""
        with caplog.at_level(logging.WARNING):
            p = TrajectoryPoint(
                traj_ind=5,
                time_stamp="2024-01-01",
                x=0.0,
                y=0.0,
                z=150_000.0,
            )
        assert p.z == 150_000.0
        assert any("implausibly high altitude" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------


def _make_df(n: int = 3, has_label: bool = True) -> pd.DataFrame:
    data = {
        "traj_ind": list(range(n)),
        "time_stamp": ["2024-01-01T00:00:00"] * n,
        "x": [0.0] * n,
        "y": [0.0] * n,
        "z": [10.0] * n,
    }
    if has_label:
        data["label"] = [0] * n
    return pd.DataFrame(data)


class TestValidateDataframe:
    def test_valid_train_df(self) -> None:
        df = _make_df(has_label=True)
        _valid, errors = validate_dataframe(df, has_label=True)
        assert len(errors) == 0
        assert len(_valid) == 3

    def test_valid_test_df_no_label(self) -> None:
        df = _make_df(has_label=False)
        _valid, errors = validate_dataframe(df, has_label=False)
        assert len(errors) == 0

    def test_large_negative_altitude_row_flagged(self) -> None:
        df = _make_df(has_label=True)
        df.loc[1, "z"] = -5.0
        _valid, errors = validate_dataframe(df, has_label=True)
        assert len(errors) == 1
        assert errors[0][0] == 1  # row index

    def test_sensor_noise_z_row_passes(self) -> None:
        df = _make_df(has_label=True)
        df.loc[1, "z"] = -0.003
        _valid, errors = validate_dataframe(df, has_label=True)
        assert len(errors) == 0

    def test_missing_required_column(self) -> None:
        df = _make_df(has_label=False).drop(columns=["x"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, has_label=False)

    def test_invalid_label_row_flagged(self) -> None:
        df = _make_df(has_label=True)
        df.loc[2, "label"] = 9
        _valid, errors = validate_dataframe(df, has_label=True)
        assert len(errors) == 1
        assert errors[0][0] == 2
