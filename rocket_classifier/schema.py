"""Pydantic data contracts for raw trajectory input validation.

Defines :class:`TrajectoryPoint`, the canonical schema for a single radar
observation, and :func:`validate_dataframe`, a helper that bulk-validates a
``pandas.DataFrame`` before it enters the feature engineering pipeline.

Typical usage::

    import pandas as pd
    from rocket_classifier.schema import validate_dataframe

    df = pd.read_csv("data/train.csv")
    valid_records, errors = validate_dataframe(df, has_label=True)
    # errors is a list of (row_index, ValidationError) for any bad rows
"""

import logging
from datetime import datetime
from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

VALID_LABELS: frozenset[int] = frozenset({0, 1, 2})


class TrajectoryPoint(BaseModel):
    """Validated schema for a single radar ping within a trajectory.

    Each row in ``train.csv`` and ``test.csv`` maps to one ``TrajectoryPoint``.
    Validation enforces physical plausibility (altitude cannot be negative),
    type safety, and label integrity before any feature engineering runs.

    Attributes:
        traj_ind: Non-negative integer trajectory identifier.  All rows
            sharing the same ``traj_ind`` belong to the same flight path.
        time_stamp: UTC datetime of the radar observation.  Accepts any
            format that ``datetime`` can parse (ISO 8601, mixed-microsecond,
            etc.).
        x: Horizontal position X-coordinate in metres.
        y: Horizontal position Y-coordinate in metres.
        z: Altitude in metres.  Must be ``>= 0``; negative altitude is
            physically impossible under the flat-terrain assumption.
        label: Integer class label in ``{0, 1, 2}``.  Set to ``None`` for
            test data where the true label is unknown.
    """

    model_config = ConfigDict(
        # Coerce compatible types (e.g. int → float for x/y/z) rather than
        # rejecting them outright.  This mirrors pandas' implicit casting.
        coerce_numbers_to_str=False,
        strict=False,
        # Forbid unexpected extra columns that could indicate a schema drift.
        extra="forbid",
    )

    traj_ind: int = Field(..., ge=0, description="Trajectory identifier (non-negative integer).")
    time_stamp: datetime = Field(..., description="UTC datetime of the radar observation.")
    x: float = Field(..., description="Horizontal X-position in metres.")
    y: float = Field(..., description="Horizontal Y-position in metres.")
    z: float = Field(..., ge=0.0, description="Altitude in metres (must be >= 0).")
    label: int | None = Field(
        default=None,
        description="Class label in {0, 1, 2}. None for unlabelled test rows.",
    )

    @field_validator("label", mode="before")
    @classmethod
    def label_must_be_valid_class(cls, v: object) -> int | None:
        """Reject label values outside the known class set {0, 1, 2}.

        Args:
            v: Raw value supplied for the ``label`` field before coercion.

        Returns:
            The validated integer label, or ``None`` if no label was provided.

        Raises:
            ValueError: If the label is not ``None`` and not in ``{0, 1, 2}``.
        """
        if v is None:
            return None
        label = int(v)
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {sorted(VALID_LABELS)}, got {label!r}.")
        return label

    @model_validator(mode="after")
    def z_consistent_with_altitude(self) -> Self:
        """Warn when altitude appears implausibly large (> 100 km).

        The operational scenario does not extend to orbital altitudes.
        Values above 100 km are not rejected outright (they may be sensor
        artefacts), but a warning is emitted so the anomaly is visible in
        structured logs.

        Returns:
            The validated ``TrajectoryPoint`` instance, unchanged.
        """
        if self.z > 100_000:
            logger.warning(
                "TrajectoryPoint traj_ind=%d has implausibly high altitude z=%.1f m "
                "(> 100 km). Possible sensor artefact.",
                self.traj_ind,
                self.z,
            )
        return self


# ---------------------------------------------------------------------------
# DataFrame bulk-validation helper
# ---------------------------------------------------------------------------


def validate_dataframe(
    df: pd.DataFrame,
    *,
    has_label: bool = False,
) -> tuple[list[TrajectoryPoint], list[tuple[int, Exception]]]:
    """Validate every row of a raw trajectory DataFrame against :class:`TrajectoryPoint`.

    Iterates over ``df`` row by row, attempting to parse each record as a
    ``TrajectoryPoint``.  Invalid rows are collected rather than raising
    immediately so that the caller can decide whether to abort or continue
    with the valid subset.

    Args:
        df: Raw point-level DataFrame with columns ``traj_ind``, ``time_stamp``,
            ``x``, ``y``, ``z``, and optionally ``label``.
        has_label: When ``True``, the ``label`` column is expected to be present
            and non-null in every row.  When ``False``, ``label`` is omitted from
            each row's validation payload so the ``extra="forbid"`` constraint
            does not trigger on its absence.

    Returns:
        A tuple of:
            - ``valid``: List of successfully parsed :class:`TrajectoryPoint`
              objects, one per valid row.
            - ``errors``: List of ``(original_row_index, exception)`` pairs for
              every row that failed validation.  Empty when all rows are valid.

    Example::

        valid, errors = validate_dataframe(pd.read_csv("data/train.csv"), has_label=True)
        if errors:
            for idx, exc in errors:
                logger.warning("Row %d: %s", idx, exc)
    """
    required_cols = {"traj_ind", "time_stamp", "x", "y", "z"}
    if has_label:
        required_cols.add("label")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    payload_cols = list(required_cols)  # only validate columns the schema expects

    valid: list[TrajectoryPoint] = []
    errors: list[tuple[int, Exception]] = []

    for idx, row in df[payload_cols].iterrows():
        try:
            point = TrajectoryPoint.model_validate(row.to_dict())
            valid.append(point)
        except Exception as exc:
            errors.append((int(idx), exc))

    if errors:
        logger.warning(
            "validate_dataframe: %d/%d rows failed validation.",
            len(errors),
            len(df),
        )
    else:
        logger.info("validate_dataframe: all %d rows passed schema validation.", len(df))

    return valid, errors
