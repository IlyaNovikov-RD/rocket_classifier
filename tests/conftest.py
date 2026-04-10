"""Shared test constants and helpers.

Provides StubModel and make_stub_clf for creating deterministic
RocketClassifier instances without loading model artifacts from disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rocket_classifier.model import RocketClassifier

N_FEATURES = 32


class StubModel:
    """Minimal stand-in for a LightGBM Booster.

    Returns fixed probabilities supplied at construction time so tests are
    fully deterministic without loading any model artifact from disk.
    """

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba.copy()


def make_stub_clf(proba: np.ndarray) -> RocketClassifier:
    """Build a RocketClassifier backed by a StubModel."""
    return RocketClassifier(model=StubModel(proba), medians=np.zeros(N_FEATURES))


def make_raw_trajectory_df(
    n_traj: int = 5,
    n_points: int = 4,
    include_label: bool = True,
) -> pd.DataFrame:
    """Create a multi-trajectory raw DataFrame suitable for build_features."""
    rows = []
    for traj_id in range(n_traj):
        for i in range(n_points):
            row = {
                "traj_ind": traj_id,
                "time_stamp": f"2024-01-01 00:00:0{i}.{traj_id:06d}",
                "x": float(i) + traj_id * 0.001,
                "y": traj_id * 0.001,
                "z": float(i) * 0.5,
            }
            if include_label:
                row["label"] = traj_id % 3
            rows.append(row)
    return pd.DataFrame(rows)
