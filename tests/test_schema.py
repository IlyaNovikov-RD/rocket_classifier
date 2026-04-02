"""Unit tests for rocket_classifier.schema.

Tests cover TrajectoryPoint validation and validate_dataframe behaviour
for both training (with label) and inference (without label) inputs.
"""

from __future__ import annotations

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
            x=100.0, y=200.0, z=50.0,
        )
        assert p.traj_ind == 1
        assert p.z == 50.0
        assert p.label is None

    def test_valid_with_label(self) -> None:
        for label in (0, 1, 2):
            p = TrajectoryPoint(
                traj_ind=0,
                time_stamp="2024-06-15T12:30:00",
                x=0.0, y=0.0, z=0.0,
                label=label,
            )
            assert p.label == label

    def test_zero_altitude_allowed(self) -> None:
        p = TrajectoryPoint(
            traj_ind=1, time_stamp="2024-01-01", x=0.0, y=0.0, z=0.0,
        )
        assert p.z == 0.0


class TestTrajectoryPointInvalid:
    def test_negative_traj_ind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=-1, time_stamp="2024-01-01", x=0.0, y=0.0, z=0.0,
            )

    def test_negative_altitude_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=1, time_stamp="2024-01-01", x=0.0, y=0.0, z=-1.0,
            )

    def test_invalid_label_rejected(self) -> None:
        for bad_label in (3, -1, 99):
            with pytest.raises(ValidationError):
                TrajectoryPoint(
                    traj_ind=1, time_stamp="2024-01-01",
                    x=0.0, y=0.0, z=0.0, label=bad_label,
                )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryPoint(
                traj_ind=1, time_stamp="2024-01-01",
                x=0.0, y=0.0, z=0.0, unknown_field="bad",
            )


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

    def test_negative_altitude_row_flagged(self) -> None:
        df = _make_df(has_label=True)
        df.loc[1, "z"] = -5.0
        _valid, errors = validate_dataframe(df, has_label=True)
        assert len(errors) == 1
        assert errors[0][0] == 1  # row index

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
