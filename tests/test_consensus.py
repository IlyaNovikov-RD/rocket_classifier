"""Unit tests for proximity-based salvo consensus.

Tests cover:
  - build_proximity_groups: same position + close time → same group
  - build_proximity_groups: same position + far time → different groups (no chaining)
  - build_proximity_groups: different positions → different groups
  - apply_salvo_consensus: strict majority corrects borderline prediction
  - apply_salvo_consensus: tie → no override
  - apply_salvo_consensus: solo trajectory (group size 1) → unchanged
  - apply_salvo_consensus: already correct majority → unchanged
  - apply_salvo_consensus: n_broken = 0 when group is class-pure
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rocket_classifier.main import (
    _PROX_TIME_WINDOW_S,
    apply_salvo_consensus,
    build_proximity_groups,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(traj_inds: list[int], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.Index(traj_inds, name="traj_ind"))


# ---------------------------------------------------------------------------
# build_proximity_groups
# ---------------------------------------------------------------------------


class TestBuildProximityGroups:
    def test_same_position_close_time_same_group(self) -> None:
        """Rockets at the same position within 60 s → same group."""
        ids = [0, 1, 2]
        lx = _make_series(ids, [1.0, 1.0, 1.0])
        ly = _make_series(ids, [0.5, 0.5, 0.5])
        lt = _make_series(ids, [1000.0, 1010.0, 1020.0])  # 20 s span

        groups = build_proximity_groups(lx, ly, lt)

        assert groups[0] == groups[1] == groups[2]

    def test_same_position_far_time_different_groups(self) -> None:
        """Rockets at the same position but span > window → different groups."""
        ids = [0, 1, 2]
        lx = _make_series(ids, [1.0, 1.0, 1.0])
        ly = _make_series(ids, [0.5, 0.5, 0.5])
        # t=0, t=50 → same group; t=110 → new group (110 - 0 > WINDOW)
        lt = _make_series(ids, [0.0, _PROX_TIME_WINDOW_S - 10, _PROX_TIME_WINDOW_S + 50])

        groups = build_proximity_groups(lx, ly, lt)

        assert groups[0] == groups[1]
        assert groups[2] != groups[0]

    def test_no_chaining(self) -> None:
        """Consecutive gaps < window but total span > window must NOT chain."""
        w = _PROX_TIME_WINDOW_S
        gap = w - 10  # each step fits within the window
        ids = [0, 1, 2, 3]
        lx = _make_series(ids, [1.0] * 4)
        ly = _make_series(ids, [0.5] * 4)
        # Each consecutive gap = gap s < window, but cumulative span grows past it.
        lt = _make_series(ids, [0.0, gap, gap * 2, gap * 3])

        groups = build_proximity_groups(lx, ly, lt)

        # Absolute-span: group restarts whenever t - group_start > window
        assert groups[0] == groups[1]  # span = gap <= window → same group
        assert groups[2] != groups[0]  # span = gap*2 > window → new group
        assert groups[3] != groups[0]  # span = gap*3 > window → yet another group

    def test_different_positions_different_groups(self) -> None:
        """Rockets at different positions are never in the same group."""
        ids = [0, 1]
        lx = _make_series(ids, [1.0, 2.0])
        ly = _make_series(ids, [0.5, 0.5])
        lt = _make_series(ids, [1000.0, 1001.0])

        groups = build_proximity_groups(lx, ly, lt)

        assert groups[0] != groups[1]

    def test_nan_positions_handled(self) -> None:
        """NaN launch positions are filled to 0.0 and grouped together."""
        ids = [0, 1]
        lx = _make_series(ids, [float("nan"), float("nan")])
        ly = _make_series(ids, [float("nan"), float("nan")])
        lt = _make_series(ids, [1000.0, 1005.0])

        groups = build_proximity_groups(lx, ly, lt)

        assert groups[0] == groups[1]

    def test_returns_aligned_array(self) -> None:
        """Output array length matches input length."""
        n = 10
        ids = list(range(n))
        lx = _make_series(ids, [float(i % 3) for i in ids])
        ly = _make_series(ids, [0.0] * n)
        lt = _make_series(ids, [float(i * 100) for i in ids])

        groups = build_proximity_groups(lx, ly, lt)

        assert len(groups) == n


# ---------------------------------------------------------------------------
# apply_salvo_consensus
# ---------------------------------------------------------------------------


class TestApplySalvoConsensus:
    def test_strict_majority_corrects_miss(self) -> None:
        """3 class-0 correct + 1 class-0 mispredicted as class-1 → all corrected to 0."""
        # 4 rockets in the same salvo (group 0): true=0, preds=[0, 0, 0, 1]
        y_pred = np.array([0, 0, 0, 1], dtype=np.int32)
        group_ids = np.array([0, 0, 0, 0], dtype=np.int32)

        result = apply_salvo_consensus(y_pred, group_ids)

        assert (result == 0).all(), "All rockets should be corrected to class 0"

    def test_two_class_tie_no_override(self) -> None:
        """Equal votes between two classes (2 vs 2) → no change."""
        y_pred = np.array([0, 0, 1, 1], dtype=np.int32)
        group_ids = np.array([0, 0, 0, 0], dtype=np.int32)

        result = apply_salvo_consensus(y_pred, group_ids)

        np.testing.assert_array_equal(result, y_pred)

    def test_three_class_plurality_no_override(self) -> None:
        """Three classes with plurality (2/2/1) but no strict majority → no change.

        class 0: 2 votes, class 1: 2 votes, class 2: 1 vote.
        The top class (0 or 1 by argmax) has 2 votes but 2+1=3 others — not
        strictly greater, so the strict-majority rule must block the override.
        """
        y_pred = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        group_ids = np.array([0, 0, 0, 0, 0], dtype=np.int32)

        result = apply_salvo_consensus(y_pred, group_ids)

        np.testing.assert_array_equal(result, y_pred)

    def test_solo_trajectory_unchanged(self) -> None:
        """A trajectory in its own group (size 1) is never modified."""
        y_pred = np.array([1], dtype=np.int32)
        group_ids = np.array([0], dtype=np.int32)

        result = apply_salvo_consensus(y_pred, group_ids)

        assert result[0] == 1

    def test_does_not_modify_input(self) -> None:
        """Input array is not mutated."""
        y_pred = np.array([0, 0, 1], dtype=np.int32)
        group_ids = np.array([0, 0, 0], dtype=np.int32)
        original = y_pred.copy()

        apply_salvo_consensus(y_pred, group_ids)

        np.testing.assert_array_equal(y_pred, original)

    def test_pure_group_no_broken_predictions(self) -> None:
        """A class-pure group with all correct predictions → n_broken = 0."""
        y_pred = np.array([2, 2, 2, 2], dtype=np.int32)
        group_ids = np.array([0, 0, 0, 0], dtype=np.int32)
        y_true = np.array([2, 2, 2, 2], dtype=np.int32)

        result = apply_salvo_consensus(y_pred, group_ids)

        n_broken = int(((y_pred == y_true) & (result != y_true)).sum())
        assert n_broken == 0

    def test_multiple_independent_groups(self) -> None:
        """Each group votes independently; different groups don't interfere."""
        # Group 0: [0, 0, 1] → majority 0 → all become 0
        # Group 1: [2, 2, 2] → unanimous 2 → unchanged
        y_pred = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
        group_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        result = apply_salvo_consensus(y_pred, group_ids)

        assert (result[:3] == 0).all()
        assert (result[3:] == 2).all()

    def test_real_scenario_four_misses_corrected(self) -> None:
        """Stress test: 4 injected misses in salvos, all corrected by consensus."""
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.choice([0, 1, 2], size=n, p=[0.686, 0.243, 0.071])
        y_pred = y_true.copy()

        # Inject 4 misses in 4 different 3-rocket salvos (class-0 → predict 1)
        class0_idx = np.where(y_true == 0)[0]
        miss_positions = class0_idx[:4]
        for _i, pos in enumerate(miss_positions):
            y_pred[pos] = 1

        # Build groups: each miss trajectory has 2 correct class-0 neighbours
        group_ids = np.arange(n, dtype=np.int32)  # all solo initially
        next_gid = n
        for pos in miss_positions:
            # add two correct neighbours to the same group
            y_true = np.append(y_true, [0, 0])
            y_pred = np.append(y_pred, [0, 0])
            group_ids = np.append(group_ids, [next_gid, next_gid])
            group_ids[pos] = next_gid
            next_gid += 1

        result = apply_salvo_consensus(y_pred, group_ids)

        n_corrected = int(((y_pred != y_true) & (result == y_true)).sum())
        n_broken = int(((y_pred == y_true) & (result != y_true)).sum())
        assert n_corrected == 4
        assert n_broken == 0
