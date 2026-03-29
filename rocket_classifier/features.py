"""Feature engineering for rocket trajectory classification.

Assumptions:
    - z is altitude (height plane); x, y are horizontal coordinates.
    - Shtuchia has flat terrain: no terrain correction needed.
    - Rockets follow standard frictionless ballistic trajectories.
    - Per-trajectory aggregation: one feature vector per traj_ind.
    - Time deltas derived from time_stamp column (parsed as datetime).
"""

import logging

import numpy as np
import pandas as pd

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
    """Extract all physics-based and statistical features for one trajectory.

    Computes 76 scalar features from the raw point sequence of a single
    trajectory, including kinematic derivatives (velocity, acceleration, jerk),
    ballistic parameters (launch angle, apogee), and spatial statistics.
    All features are returned even for very short trajectories — derivative
    features are set to NaN when insufficient points are available.

    Args:
        group: DataFrame of rows belonging to a single ``traj_ind``,
            pre-sorted by ``time_stamp`` in ascending order. Must contain
            columns ``time_stamp`` (datetime64), ``x``, ``y``, and ``z``
            (float64).

    Returns:
        A flat dictionary mapping feature name to float value. Contains
        exactly 76 keys regardless of trajectory length. NaN is used for
        features that cannot be computed (e.g. acceleration when N < 3).
    """
    feats: dict[str, float] = {}

    pos = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    times = group["time_stamp"].to_numpy(dtype="datetime64[ns]")

    n_points = len(pos)
    feats["n_points"] = float(n_points)

    # --- Time features ---
    dt_ns = np.diff(times).astype(np.float64)  # nanoseconds
    dt_sec = dt_ns / 1e9  # seconds
    # Guard against zero or negative dt (duplicate timestamps)
    dt_sec = np.where(dt_sec <= 0, np.nan, dt_sec)
    valid_dt = dt_sec[~np.isnan(dt_sec)]

    total_duration = float(np.nansum(dt_sec)) if len(dt_sec) > 0 else 0.0
    feats["total_duration_s"] = total_duration
    feats.update(_safe_stats(valid_dt, "dt"))

    # --- 3D Velocity ---
    if n_points >= 2 and valid_dt.size > 0:
        # Replace nan dt with median for derivative computation
        dt_filled = np.where(np.isnan(dt_sec), np.nanmedian(dt_sec), dt_sec)
        vel, acc, jerk = _compute_derivatives(pos, dt_filled)

        speed = np.linalg.norm(vel, axis=1)
        feats.update(_safe_stats(speed, "speed"))

        vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
        feats.update(_safe_stats(vx, "vx"))
        feats.update(_safe_stats(vy, "vy"))
        feats.update(_safe_stats(vz, "vz"))

        # Horizontal speed
        v_horiz = np.sqrt(vx**2 + vy**2)
        feats.update(_safe_stats(v_horiz, "v_horiz"))

        # --- Launch angle (elevation angle of initial velocity vector) ---
        # Angle between initial velocity and horizontal plane
        feats["launch_angle_elev"] = (
            float(np.arctan2(vz[0], np.sqrt(vx[0] ** 2 + vy[0] ** 2))) if speed[0] > 0 else np.nan
        )

        # Azimuth of initial velocity (heading in horizontal plane)
        # When speed is zero, azimuth is undefined — arctan2(0,0)=0 is meaningless
        feats["launch_angle_azimuth"] = (
            float(np.arctan2(vy[0], vx[0])) if speed[0] > 0 else np.nan
        )

        # Initial speed components
        feats["initial_speed"] = float(speed[0])
        feats["initial_vz"] = float(vz[0])
        feats["initial_v_horiz"] = float(v_horiz[0])

        # Final speed components
        feats["final_speed"] = float(speed[-1])
        feats["final_vz"] = float(vz[-1])

        # --- 3D Acceleration ---
        if acc.shape[0] > 0:
            acc_mag = np.linalg.norm(acc, axis=1)
            feats.update(_safe_stats(acc_mag, "acc_mag"))

            ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
            feats.update(_safe_stats(az, "az"))  # vertical acceleration (gravity proxy)
            feats.update(_safe_stats(np.sqrt(ax**2 + ay**2), "acc_horiz"))

            feats["mean_az"] = float(np.mean(az))  # should be ~-g for ballistic

        else:
            for k in [
                "acc_mag_mean",
                "acc_mag_std",
                "acc_mag_min",
                "acc_mag_max",
                "acc_mag_median",
                "az_mean",
                "az_std",
                "az_min",
                "az_max",
                "az_median",
                "acc_horiz_mean",
                "acc_horiz_std",
                "acc_horiz_min",
                "acc_horiz_max",
                "acc_horiz_median",
                "mean_az",
            ]:
                feats[k] = np.nan

        # --- 3D Jerk ---
        if jerk.shape[0] > 0:
            jerk_mag = np.linalg.norm(jerk, axis=1)
            feats.update(_safe_stats(jerk_mag, "jerk_mag"))
        else:
            for k in [
                "jerk_mag_mean",
                "jerk_mag_std",
                "jerk_mag_min",
                "jerk_mag_max",
                "jerk_mag_median",
            ]:
                feats[k] = np.nan

    else:
        # Not enough points for derivatives
        for prefix in [
            "speed",
            "vx",
            "vy",
            "vz",
            "v_horiz",
            "acc_mag",
            "az",
            "acc_horiz",
            "jerk_mag",
        ]:
            for stat in ["mean", "std", "min", "max", "median"]:
                feats[f"{prefix}_{stat}"] = np.nan
        for k in [
            "launch_angle_elev",
            "launch_angle_azimuth",
            "initial_speed",
            "initial_vz",
            "initial_v_horiz",
            "final_speed",
            "final_vz",
            "mean_az",
        ]:
            feats[k] = np.nan

    # --- Apogee and Time-to-Apogee ---
    z_vals = pos[:, 2]
    apogee_idx = int(np.argmax(z_vals))
    feats["apogee_z"] = float(z_vals[apogee_idx])
    feats["initial_z"] = float(z_vals[0])
    feats["final_z"] = float(z_vals[-1])
    feats["delta_z_total"] = float(z_vals[-1] - z_vals[0])
    feats["apogee_relative"] = float(z_vals[apogee_idx] - z_vals[0])

    # Time to apogee (fraction of total trajectory)
    feats["apogee_time_frac"] = float(apogee_idx / max(n_points - 1, 1))
    # Absolute time to apogee — use direct timestamp subtraction to avoid
    # nansum gaps from duplicate-timestamp NaNs understating the true elapsed time
    if apogee_idx > 0:
        feats["time_to_apogee_s"] = float(
            (times[apogee_idx] - times[0]).astype(np.float64) / 1e9
        )
    else:
        feats["time_to_apogee_s"] = 0.0

    # --- Spatial extent ---
    feats["x_range"] = float(np.ptp(pos[:, 0]))
    feats["y_range"] = float(np.ptp(pos[:, 1]))
    feats["z_range"] = float(np.ptp(pos[:, 2]))

    # 2D horizontal range (max displacement in xy plane from launch)
    xy_disp = np.sqrt((pos[:, 0] - pos[0, 0]) ** 2 + (pos[:, 1] - pos[0, 1]) ** 2)
    feats["max_horiz_range"] = float(np.max(xy_disp))
    feats["final_horiz_range"] = float(xy_disp[-1])

    # --- Path length ---
    if n_points >= 2:
        segment_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        feats["path_length_3d"] = float(np.sum(segment_lengths))
    else:
        feats["path_length_3d"] = 0.0

    # --- Initial position (launch point) ---
    feats["launch_x"] = float(pos[0, 0])
    feats["launch_y"] = float(pos[0, 1])
    feats["launch_z"] = float(pos[0, 2])

    return feats


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate point-level radar data into a per-trajectory feature matrix.

    Parses timestamps, sorts each trajectory chronologically, then calls
    ``_extract_trajectory_features`` for every unique ``traj_ind``. The
    result is a single row per trajectory with 76 physics-derived features
    (plus an optional ``label`` column when training data is supplied).

    Args:
        df: Raw point-level DataFrame with columns ``traj_ind``,
            ``time_stamp``, ``x``, ``y``, ``z``. An optional ``label``
            column is preserved when present (training data only).

    Returns:
        Per-trajectory feature DataFrame with ``traj_ind`` as the index.
        Shape is (n_trajectories, 76) without label, (n_trajectories, 77)
        with label. All feature values are float64; NaN indicates a feature
        that could not be computed for a given trajectory.
    """
    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")

    # Sort within each trajectory by time
    df = df.sort_values(["traj_ind", "time_stamp"])

    has_label = "label" in df.columns
    n_trajectories = df["traj_ind"].nunique()
    logger.info("Building features for %d trajectories...", n_trajectories)

    groups = df.groupby("traj_ind", sort=False)

    records: list[dict] = []
    for traj_id, group in groups:
        feats = _extract_trajectory_features(group.reset_index(drop=True))
        feats["traj_ind"] = traj_id
        if has_label:
            feats["label"] = int(group["label"].iloc[0])
        records.append(feats)

    result = pd.DataFrame(records).set_index("traj_ind")
    logger.info("Feature matrix built: shape=%s", result.shape)
    return result
