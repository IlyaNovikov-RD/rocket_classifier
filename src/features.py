"""
Feature engineering for rocket trajectory classification.

Assumptions:
- z is altitude (height plane); x, y are horizontal coordinates.
- Shtuchia has flat terrain: no terrain correction needed.
- Rockets follow standard frictionless ballistic trajectories.
- Per-trajectory aggregation: one feature vector per traj_ind.
- Time deltas derived from time_stamp column (parsed as datetime).
"""

import numpy as np
import pandas as pd


def _compute_derivatives(
    pos: np.ndarray,
    dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocity, acceleration, jerk via finite differences.

    Args:
        pos: (N, 3) array of [x, y, z] positions, sorted by time.
        dt:  (N-1,) array of time deltas in seconds between consecutive points.

    Returns:
        vel:  (N-1, 3) velocity vectors [m/s equivalent].
        acc:  (N-2, 3) acceleration vectors.
        jerk: (N-3, 3) jerk vectors.
    """
    # Velocity: forward difference on positions
    vel = np.diff(pos, axis=0) / dt[:, np.newaxis]

    if vel.shape[0] < 2:
        return vel, np.empty((0, 3)), np.empty((0, 3))

    # Midpoint dt for acceleration (between consecutive velocity estimates)
    dt_acc = (dt[:-1] + dt[1:]) / 2.0
    acc = np.diff(vel, axis=0) / dt_acc[:, np.newaxis]

    if acc.shape[0] < 2:
        return vel, acc, np.empty((0, 3))

    dt_jerk = (dt_acc[:-1] + dt_acc[1:]) / 2.0
    jerk = np.diff(acc, axis=0) / dt_jerk[:, np.newaxis]

    return vel, acc, jerk


def _safe_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    """Return mean/std/min/max/median of 1-D array, with NaN fallback."""
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

    Args:
        group: DataFrame rows for a single traj_ind, pre-sorted by time_stamp.

    Returns:
        Dictionary of feature_name -> float value.
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
        feats["launch_angle_azimuth"] = float(np.arctan2(vy[0], vx[0]))

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
    # Absolute time to apogee
    if len(dt_sec) > 0 and apogee_idx > 0:
        feats["time_to_apogee_s"] = float(np.nansum(dt_sec[:apogee_idx]))
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
    """Aggregate point-level data into per-trajectory feature matrix.

    Args:
        df: Raw DataFrame with columns [traj_ind, time_stamp, x, y, z].
            Optionally includes 'label' (train only).

    Returns:
        Per-trajectory feature DataFrame indexed by traj_ind.
        Includes 'label' column if present in input.
    """
    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="mixed")

    # Sort within each trajectory by time
    df = df.sort_values(["traj_ind", "time_stamp"])

    has_label = "label" in df.columns

    groups = df.groupby("traj_ind", sort=False)

    records: list[dict] = []
    for traj_id, group in groups:
        feats = _extract_trajectory_features(group.reset_index(drop=True))
        feats["traj_ind"] = traj_id
        if has_label:
            feats["label"] = int(group["label"].iloc[0])
        records.append(feats)

    result = pd.DataFrame(records).set_index("traj_ind")
    return result
