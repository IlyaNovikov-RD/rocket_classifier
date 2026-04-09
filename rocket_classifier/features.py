"""Feature engineering for rocket trajectory classification.

Assumptions:
    - z is altitude (height plane); x, y are horizontal coordinates.
    - Flat terrain: no terrain correction needed.
    - Rockets follow standard frictionless ballistic trajectories.
    - Per-trajectory aggregation: one feature vector per traj_ind.
    - Time deltas derived from time_stamp column (parsed as datetime).
    - Rockets are fired in spatiotemporal salvos by geographically
      concentrated rebel groups (domain assumptions 3a-3c).
"""

import logging

import numpy as np
import pandas as pd

# DBSCAN parameters for salvo and rebel-group clustering.
# Spatiotemporal (salvo): eps=0.5 in StandardScaler-normalised
# (launch_x, launch_y, launch_time_s) space; min_samples=2.
# Pure-spatial (rebel group): eps=0.25 in normalised (launch_x, launch_y)
# space; min_samples=3.  Both validated across multiple independent
# feature-selection runs on the full training set.
_SALVO_EPS = 0.5
_SALVO_MIN_SAMPLES = 2
_GROUP_EPS = 0.25
_GROUP_MIN_SAMPLES = 3

logger = logging.getLogger(__name__)


def _compute_derivatives(
    pos: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocity, acceleration, and jerk via finite differences.

    Uses forward differences on position to derive velocity, then applies
    midpoint-averaged time deltas for each subsequent derivative order to
    minimise discretisation error.

    Note:
        dt=0 is NOT sanitised here. The caller is responsible for replacing
        zero or negative time deltas before passing them to this function.
        Passing dt=0 will produce non-finite (inf/nan) velocity values.

    Args:
        pos: Position array of shape (N, 3) with columns [x, y, z],
            sorted in ascending time order.
        dt: Time delta array of shape (N-1,) in seconds between
            consecutive position samples.

    Returns:
        A tuple of (vel, acc, jerk) where:
            vel:  Velocity array of shape (N-1, 3) in units/s.
            acc:  Acceleration array of shape (N-2, 3), or (0, 3) if
                fewer than 2 velocity samples are available.
            jerk: Jerk array of shape (N-3, 3), or (0, 3) if fewer than
                2 acceleration samples are available.
    """
    vel = np.diff(pos, axis=0) / dt[:, np.newaxis]

    if vel.shape[0] < 2:
        return vel, np.empty((0, 3)), np.empty((0, 3))

    dt_acc = (dt[:-1] + dt[1:]) / 2.0
    acc = np.diff(vel, axis=0) / dt_acc[:, np.newaxis]

    if acc.shape[0] < 2:
        return vel, acc, np.empty((0, 3))

    dt_jerk = (dt_acc[:-1] + dt_acc[1:]) / 2.0
    jerk = np.diff(acc, axis=0) / dt_jerk[:, np.newaxis]

    return vel, acc, jerk


def _safe_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute descriptive statistics for a 1-D array with NaN fallback.

    Returns five statistics (mean, std, min, max, median) for the input
    array. If the array is empty, all five values are returned as NaN so
    that downstream feature matrices remain consistently shaped regardless
    of trajectory length.

    Args:
        arr: 1-D numeric array to summarise. May be empty.
        prefix: String prefix applied to every output key, e.g. ``"speed"``
            produces keys ``speed_mean``, ``speed_std``, etc.

    Returns:
        A dictionary with five entries keyed as ``{prefix}_mean``,
        ``{prefix}_std``, ``{prefix}_min``, ``{prefix}_max``, and
        ``{prefix}_median``. All values are Python floats.
    """
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }


def _extract_trajectory_features(group: pd.DataFrame) -> dict[str, float]:
    """Extract the 25 production kinematic features for one trajectory.

    Computes only the kinematic features that survived backward elimination
    and appear in ``SELECTED_FEATURES`` (see ``model.py``).  The 7
    salvo/group features are added later by ``_add_salvo_group_features``.

    Args:
        group: DataFrame of rows belonging to a single ``traj_ind``,
            pre-sorted by ``time_stamp`` in ascending order. Must contain
            columns ``time_stamp`` (datetime64), ``x``, ``y``, and ``z``
            (float64).

    Returns:
        A flat dictionary mapping feature name to float value. Contains
        exactly 25 keys regardless of trajectory length. NaN is used for
        features that cannot be computed (e.g. acceleration when N < 3).
    """
    feats: dict[str, float] = {}

    pos = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    times = group["time_stamp"].to_numpy(dtype="datetime64[ns]")

    n_points = len(pos)
    feats["n_points"] = float(n_points)

    _NAN_VELOCITY = {
        "vy_mean": np.nan,
        "vy_max": np.nan,
        "vz_median": np.nan,
        "v_horiz_std": np.nan,
        "v_horiz_median": np.nan,
        "initial_speed": np.nan,
        "initial_vz": np.nan,
        "final_vz": np.nan,
    }
    _NAN_ACCELERATION = {
        "acc_mag_mean": np.nan,
        "acc_mag_min": np.nan,
        "acc_mag_max": np.nan,
        "az_std": np.nan,
        "acc_horiz_std": np.nan,
        "acc_horiz_min": np.nan,
        "acc_horiz_max": np.nan,
        "mean_az": np.nan,
    }

    if n_points == 0:
        feats.update(_NAN_VELOCITY)
        feats.update(_NAN_ACCELERATION)
        for k in [
            "initial_z",
            "final_z",
            "delta_z_total",
            "apogee_relative",
            "x_range",
            "y_range",
            "launch_x",
            "launch_y",
        ]:
            feats[k] = np.nan
        return feats

    # --- Time deltas (needed for derivatives, not stored as features) ---
    dt_ns = np.diff(times).astype(np.float64)
    dt_sec = dt_ns / 1e9
    dt_sec = np.where(dt_sec <= 0, np.nan, dt_sec)
    valid_dt = dt_sec[~np.isnan(dt_sec)]

    # --- Velocity & Acceleration ---
    if n_points >= 2 and valid_dt.size > 0:
        dt_filled = np.where(np.isnan(dt_sec), np.nanmedian(dt_sec), dt_sec)
        vel, acc, _jerk = _compute_derivatives(pos, dt_filled)

        vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
        speed = np.linalg.norm(vel, axis=1)
        v_horiz = np.sqrt(vx**2 + vy**2)

        feats["vy_mean"] = float(np.mean(vy))
        feats["vy_max"] = float(np.max(vy))
        feats["vz_median"] = float(np.median(vz))
        feats["v_horiz_std"] = float(np.std(v_horiz))
        feats["v_horiz_median"] = float(np.median(v_horiz))
        feats["initial_speed"] = float(speed[0])
        feats["initial_vz"] = float(vz[0])
        feats["final_vz"] = float(vz[-1])

        if acc.shape[0] > 0:
            acc_mag = np.linalg.norm(acc, axis=1)
            ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
            acc_horiz = np.sqrt(ax**2 + ay**2)

            feats["acc_mag_mean"] = float(np.mean(acc_mag))
            feats["acc_mag_min"] = float(np.min(acc_mag))
            feats["acc_mag_max"] = float(np.max(acc_mag))
            feats["az_std"] = float(np.std(az))
            feats["acc_horiz_std"] = float(np.std(acc_horiz))
            feats["acc_horiz_min"] = float(np.min(acc_horiz))
            feats["acc_horiz_max"] = float(np.max(acc_horiz))
            feats["mean_az"] = float(np.mean(az))
        else:
            feats.update(_NAN_ACCELERATION)
    else:
        feats.update(_NAN_VELOCITY)
        feats.update(_NAN_ACCELERATION)

    # --- Altitude ---
    z_vals = pos[:, 2]
    feats["initial_z"] = float(z_vals[0])
    feats["final_z"] = float(z_vals[-1])
    feats["delta_z_total"] = float(z_vals[-1] - z_vals[0])
    feats["apogee_relative"] = float(z_vals.max() - z_vals[0])

    # --- Spatial extent & launch position ---
    feats["x_range"] = float(np.ptp(pos[:, 0]))
    feats["y_range"] = float(np.ptp(pos[:, 1]))
    feats["launch_x"] = float(pos[0, 0])
    feats["launch_y"] = float(pos[0, 1])

    return feats


def _add_salvo_group_features(
    result: pd.DataFrame,
    launch_times: pd.Series,
) -> pd.DataFrame:
    """Add salvo and rebel-group clustering features to the feature matrix.

    Implements domain assumptions 3a-3c from the problem specification:

    * **3b — Rockets fired in salvos**: spatiotemporal DBSCAN on
      ``(launch_x, launch_y, launch_time_s)`` groups trajectories fired
      together.  Produces ``salvo_size``, ``salvo_duration_s``,
      ``salvo_spatial_spread_m``, and ``salvo_time_rank``.

    * **3c — Geographically concentrated groups, each purchasing independently**:
      pure-spatial DBSCAN on ``(launch_x, launch_y)`` identifies persistent
      rebel bases.  Because each group purchases its own rocket supply
      independently, rockets from the same base are the same type.
      Produces ``group_total_rockets``, ``group_n_salvos``, and
      ``group_max_salvo_size`` (proxy for assumption 3a — launcher type).

    Args:
        result: Per-trajectory feature DataFrame (index=traj_ind) containing
            ``launch_x`` and ``launch_y`` columns.
        launch_times: Series indexed by ``traj_ind`` with the first
            timestamp of each trajectory.

    Returns:
        ``result`` with 7 new columns appended:
        ``salvo_size``, ``salvo_duration_s``, ``salvo_spatial_spread_m``,
        ``salvo_time_rank``, ``group_total_rockets``, ``group_n_salvos``,
        ``group_max_salvo_size``.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    result = result.copy()
    result["_lt_s"] = launch_times.reindex(result.index).astype(np.int64) / 1e9

    # ── Salvo clustering (spatiotemporal) ──────────────────────────────────
    cluster_input = result[["launch_x", "launch_y", "_lt_s"]].fillna(0.0)
    cluster_scaled = StandardScaler().fit_transform(cluster_input)
    raw_salvo = DBSCAN(eps=_SALVO_EPS, min_samples=_SALVO_MIN_SAMPLES, n_jobs=-1).fit_predict(
        cluster_scaled
    )

    # Noise points (label=-1) each become their own singleton salvo
    next_id = int(raw_salvo.max()) + 1
    salvo_ids = raw_salvo.copy()
    for i in range(len(salvo_ids)):
        if salvo_ids[i] == -1:
            salvo_ids[i] = next_id
            next_id += 1
    result["_salvo_id"] = salvo_ids

    # Per-salvo statistics
    salvo_rows: list[dict] = []
    for _sid, grp in result.groupby("_salvo_id"):
        n = len(grp)
        lx = grp["launch_x"].values
        ly = grp["launch_y"].values
        lt = grp["_lt_s"].values
        spread = dur = 0.0
        if n > 1:
            dur = float(lt.max() - lt.min())
            if n <= 1000:
                # Exact max pairwise distance for small salvos
                dx = lx[:, None] - lx[None, :]
                dy = ly[:, None] - ly[None, :]
                spread = float(np.sqrt(dx**2 + dy**2).max())
            else:
                # Stochastic estimate for large salvos — avoids O(n²) memory.
                # Sample 10 000 random pairs and take the max distance observed.
                rng = np.random.default_rng(seed=42)
                n_pairs = min(10_000, n * (n - 1) // 2)
                idx_a = rng.integers(0, n, size=n_pairs)
                idx_b = rng.integers(0, n, size=n_pairs)
                spread = float(
                    np.sqrt((lx[idx_a] - lx[idx_b]) ** 2 + (ly[idx_a] - ly[idx_b]) ** 2).max()
                )
        ranks = pd.Series(lt).rank(method="first").astype(int).values
        for i, tid in enumerate(grp.index):
            salvo_rows.append(
                {
                    "traj_ind": tid,
                    "salvo_size": n,
                    "salvo_duration_s": dur,
                    "salvo_spatial_spread_m": spread,
                    "salvo_time_rank": int(ranks[i]),
                }
            )
    result = result.join(pd.DataFrame(salvo_rows).set_index("traj_ind"))

    # ── Rebel-group clustering (pure-spatial) ──────────────────────────────
    spatial_input = result[["launch_x", "launch_y"]].fillna(0.0)
    spatial_scaled = StandardScaler().fit_transform(spatial_input)
    raw_group = DBSCAN(eps=_GROUP_EPS, min_samples=_GROUP_MIN_SAMPLES, n_jobs=-1).fit_predict(
        spatial_scaled
    )

    # Auto-tune GROUP_EPS if < 2 groups found (handles unusual spatial layouts)
    n_groups = len(set(raw_group[raw_group >= 0]))
    if n_groups < 2:
        for eps_try in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]:
            lbls = DBSCAN(eps=eps_try, min_samples=_GROUP_MIN_SAMPLES, n_jobs=-1).fit_predict(
                spatial_scaled
            )
            ng = len(set(lbls[lbls >= 0]))
            if 2 <= ng <= 20:
                raw_group = lbls
                logger.info("Auto-selected GROUP_EPS=%.2f → %d rebel groups", eps_try, ng)
                break
        else:
            logger.warning(
                "Rebel-group auto-tuning failed: no eps in [0.05..0.75] produced 2-20 groups "
                "(got %d). Proceeding with singleton groups — group features will be uninformative.",
                n_groups,
            )

    next_gid = int(raw_group.max()) + 1
    group_ids = raw_group.copy()
    for i in range(len(group_ids)):
        if group_ids[i] == -1:
            group_ids[i] = next_gid
            next_gid += 1
    result["_rebel_group_id"] = group_ids

    group_stats = result.groupby("_rebel_group_id").agg(
        group_total_rockets=("_rebel_group_id", "count"),
        group_n_salvos=("_salvo_id", "nunique"),
        group_max_salvo_size=("salvo_size", "max"),
    )
    result = result.join(group_stats, on="_rebel_group_id")

    result = result.drop(columns=["_lt_s", "_salvo_id", "_rebel_group_id"])
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate point-level radar data into a per-trajectory feature matrix.

    Parses timestamps, sorts each trajectory chronologically, then calls
    ``_extract_trajectory_features`` for every unique ``traj_ind``.  After
    kinematic feature extraction, domain-informed salvo and rebel-group
    features are appended via DBSCAN clustering (assumptions 3a-3c).

    Args:
        df: Raw point-level DataFrame with columns ``traj_ind``,
            ``time_stamp``, ``x``, ``y``, ``z``. An optional ``label``
            column is preserved when present (training data only).

    Returns:
        Per-trajectory feature DataFrame with ``traj_ind`` as the index.
        Shape is (n_trajectories, 33) without label, (n_trajectories, 34)
        with label — 25 kinematic + 7 salvo/group + 1 ``launch_time``.
        All values are float64; NaN indicates a feature that could not be
        computed for a given trajectory.
    """
    required_cols = {"traj_ind", "time_stamp", "x", "y", "z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"build_features(): input DataFrame is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")
    df = df.sort_values(["traj_ind", "time_stamp"])

    has_label = "label" in df.columns
    n_trajectories = df["traj_ind"].nunique()
    logger.info("Building features for %d trajectories...", n_trajectories)

    # Extract launch timestamp (first ping per trajectory, after time-sorting)
    launch_times = df.groupby("traj_ind")["time_stamp"].first()

    groups = df.groupby("traj_ind", sort=False)
    records: list[dict] = []
    for traj_id, group in groups:
        feats = _extract_trajectory_features(group.reset_index(drop=True))
        feats["traj_ind"] = traj_id
        if has_label:
            feats["label"] = int(group["label"].iloc[0])
        records.append(feats)

    result = pd.DataFrame(records).set_index("traj_ind")

    # Append salvo + rebel-group features (domain assumptions 3a-3c)
    result = _add_salvo_group_features(result, launch_times)

    # Store launch_time so callers don't need the raw CSV for proximity consensus.
    result["launch_time"] = launch_times.reindex(result.index)

    logger.info("Feature matrix built: shape=%s", result.shape)
    return result
