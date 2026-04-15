"""Demo visualization of physics-informed and salvo-context features.

Generates ``assets/demo.png`` containing three subplots:

    - Left:   3D trajectory comparison — all three rocket classes produced by
              the production model: Class 0 (moderate arc, 69 %),
              Class 1 (long-range flat, 24 %), Class 2 (steep short-range, 7 %).
              Apogee marked per class.  Colours match the Streamlit demo.
    - Centre: Jerk Magnitude time-series for all three classes, computed via
              ``_compute_derivatives`` (production code path).  The magnitude
              and timing of the thrust-ignition spike differ across classes and
              are among the key kinematic discriminators.
    - Right:  Real launch-site scatter from the training cache — every unique
              (launch_x, launch_y) coordinate, coloured by class.  Production
              DBSCAN (eps=0.25, min_samples=3) finds 1 dominant group containing
              99.7 % of trajectories, explaining why the rebel-group features
              have near-zero importance in the trained model.  Requires
              ``cache/cache_train_features.parquet`` (run ``make download-all``).

Uses ``_compute_derivatives`` from ``rocket_classifier/features.py`` so that
the kinematic panel reflects the same physics code path used in production.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display required

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rocket_classifier.features import _compute_derivatives

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Colour palette — class colours match the Streamlit app (app.py CLASS_COLORS)
# ---------------------------------------------------------------------------
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
GREEN = "#3fb950"  # Class 0
GOLD = "#f0c040"  # Class 1
RED = "#f85149"  # Class 2

CLASS_COLOR = {0: GREEN, 1: GOLD, 2: RED}
CLASS_LABEL = {0: "Class 0  (69 %)", 1: "Class 1  (24 %)", 2: "Class 2  (7 %)"}

# ---------------------------------------------------------------------------
# Synthetic trajectory generation — one generator per class
# ---------------------------------------------------------------------------

_DT = 0.05
_N = 120
# Small position noise: 0.001 m keeps finite-difference noise amplification
# (~sigma/dt^3) below ~35 m/s^3 so the thrust-ignition spike dominates.
_POS_NOISE = 0.001


def _ballistic(
    speed: float,
    theta_deg: float,
    phi_deg: float,
    thrust_accel: float,
    thrust_frac: float,
    n: int = _N,
    dt: float = _DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Core ballistic trajectory shared by all three class generators.

    Args:
        speed: Launch speed magnitude in m/s.
        theta_deg: Elevation angle in degrees above horizontal.
        phi_deg: Azimuth angle in degrees from x-axis.
        thrust_accel: Peak motor-burn acceleration in m/s^2 (linearly ramped down).
        thrust_frac: Fraction of total flight time over which thrust is active.
        n: Number of position samples.
        dt: Time step in seconds.

    Returns:
        pos: (n, 3) position array in metres.
        dt_arr: (n-1,) constant dt array for ``_compute_derivatives``.
    """
    g = 9.81
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    v0z = speed * np.sin(theta)
    v0h = speed * np.cos(theta)
    v0x = v0h * np.cos(phi)
    v0y = v0h * np.sin(phi)

    t = np.linspace(0, n * dt, n)
    x = v0x * t
    y = v0y * t
    z = v0z * t - 0.5 * g * t**2

    thrust_end = max(thrust_frac * t[-1], 1e-9)
    thrust_profile = np.where(t <= thrust_end, thrust_accel * (1.0 - t / thrust_end), 0.0)
    z += np.cumsum(thrust_profile) * dt**2
    z = np.maximum(z, 0.0)

    x += RNG.normal(0, _POS_NOISE, n)
    y += RNG.normal(0, _POS_NOISE, n)
    z += RNG.normal(0, _POS_NOISE, n)
    z = np.maximum(z, 0.0)
    return np.column_stack([x, y, z]), np.full(n - 1, dt)


def generate_class0(n: int = _N, dt: float = _DT) -> tuple[np.ndarray, np.ndarray]:
    """Class 0 (69 %): Moderate speed, clear ballistic arc, standard motor burn."""
    return _ballistic(
        speed=55, theta_deg=60, phi_deg=25, thrust_accel=80, thrust_frac=0.10, n=n, dt=dt
    )


def generate_class1(n: int = _N, dt: float = _DT) -> tuple[np.ndarray, np.ndarray]:
    """Class 1 (24 %): Higher speed, flatter longer-range arc, prolonged strong thrust."""
    return _ballistic(
        speed=80, theta_deg=40, phi_deg=20, thrust_accel=160, thrust_frac=0.15, n=n, dt=dt
    )


def generate_class2(n: int = _N, dt: float = _DT) -> tuple[np.ndarray, np.ndarray]:
    """Class 2 (7 %): Lower speed, steep short-range arc, minimal motor burn.

    speed=35 chosen so t_land ~= 6.9s > n*dt=6s, avoiding a ground-clamp
    discontinuity that would create an artifact jerk spike inside the window.
    """
    return _ballistic(
        speed=35, theta_deg=75, phi_deg=30, thrust_accel=25, thrust_frac=0.05, n=n, dt=dt
    )


def compute_jerk_magnitude(pos: np.ndarray, dt_arr: np.ndarray) -> np.ndarray:
    """Compute per-timestep jerk magnitude using the production derivative pipeline."""
    _, _, jerk = _compute_derivatives(pos, dt_arr)
    if jerk.shape[0] == 0:
        return np.array([])
    return np.linalg.norm(jerk, axis=1)


# ---------------------------------------------------------------------------
# Real launch-site data from training cache
# ---------------------------------------------------------------------------

_CACHE_PATH = Path(__file__).parent.parent / "cache" / "cache_train_features.parquet"


def load_real_launch_data() -> pd.DataFrame | None:
    """Load launch positions and class labels from the training feature cache.

    Returns:
        DataFrame with columns [launch_x, launch_y, label], or None if the
        cache parquet is not present (run ``make download-all`` to fetch it).
    """
    if not _CACHE_PATH.exists():
        logger.warning(
            "Training cache not found at %s — right panel will be blank. "
            "Run 'make download-all' to fetch it.",
            _CACHE_PATH,
        )
        return None
    return pd.read_parquet(_CACHE_PATH, columns=["launch_x", "launch_y", "label"])


# ---------------------------------------------------------------------------
# Three-panel figure
# ---------------------------------------------------------------------------


def make_demo_plot(output_path: Path) -> None:
    """Render and save the three-panel physics + salvo feature visualization."""
    pos0, dt0 = generate_class0()
    pos1, dt1 = generate_class1()
    pos2, dt2 = generate_class2()

    jerk0 = compute_jerk_magnitude(pos0, dt0)
    jerk1 = compute_jerk_magnitude(pos1, dt1)
    jerk2 = compute_jerk_magnitude(pos2, dt2)

    dt = _DT
    t_jerk = np.arange(len(jerk0)) * dt  # all classes have same n, so same length

    apogee = {
        0: int(np.argmax(pos0[:, 2])),
        1: int(np.argmax(pos1[:, 2])),
        2: int(np.argmax(pos2[:, 2])),
    }
    positions = {0: pos0, 1: pos1, 2: pos2}
    jerks = {0: jerk0, 1: jerk1, 2: jerk2}

    launch_df = load_real_launch_data()

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        left=0.04,
        right=0.97,
        top=0.87,
        bottom=0.10,
        wspace=0.30,
    )

    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    ax_s = fig.add_subplot(gs[2])

    # ── Left: 3D trajectory comparison ────────────────────────────────────────
    ax3d.set_facecolor(PANEL_BG)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax3d.tick_params(colors=TEXT_COLOR, labelsize=8)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.label.set_color(TEXT_COLOR)

    linestyles = {0: "-", 1: "--", 2: ":"}
    linewidths = {0: 2.2, 1: 2.0, 2: 2.0}

    for cls in (0, 1, 2):
        pos = positions[cls]
        color = CLASS_COLOR[cls]
        ax3d.plot(
            *pos.T,
            color=color,
            linewidth=linewidths[cls],
            linestyle=linestyles[cls],
            label=CLASS_LABEL[cls],
            zorder=3,
        )
        ax3d.scatter(*pos[0], color=color, s=50, zorder=5, depthshade=False)
        aidx = apogee[cls]
        ax3d.scatter(
            *pos[aidx],
            color=GOLD if cls == 0 else color,
            s=100,
            marker="*",
            zorder=6,
            depthshade=False,
        )

    ax3d.set_xlabel("X (m)", labelpad=6, fontsize=9)
    ax3d.set_ylabel("Y (m)", labelpad=6, fontsize=9)
    ax3d.set_zlabel("Altitude (m)", labelpad=6, fontsize=9)
    ax3d.set_title(
        "3D Trajectory Comparison\n(synthetic — three production classes)",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax3d.legend(
        loc="upper left",
        fontsize=8,
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )
    ax3d.view_init(elev=22, azim=-55)

    # ── Centre: Jerk magnitude ─────────────────────────────────────────────────
    ax2d.set_facecolor(PANEL_BG)
    ax2d.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax2d.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax2d.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)

    for cls in (0, 1, 2):
        jerk = jerks[cls]
        t_j = np.arange(len(jerk)) * dt
        ax2d.plot(
            t_j,
            jerk,
            color=CLASS_COLOR[cls],
            linewidth=2.0 if cls == 0 else 1.6,
            linestyle=linestyles[cls],
            label=CLASS_LABEL[cls],
        )

    # Annotate the Class 1 ignition spike (largest, most visible).
    # Thrust ramps linearly from max -> 0, so the abrupt acceleration step is
    # at ignition (t~=0), not at thrust-end where acceleration is already ~0.
    peak_idx = int(np.argmax(jerk1))
    ax2d.annotate(
        "Motor ignition\nspike (Class 1)",
        xy=(t_jerk[peak_idx], jerk1[peak_idx]),
        xytext=(t_jerk[peak_idx] + 0.6, jerk1[peak_idx] * 0.75),
        color=GOLD,
        fontsize=8.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.5},
    )

    # Focus on the first 2 s to keep the ignition spikes front-and-centre;
    # jerk falls to near-zero after all thrust phases have ended.
    ax2d.set_xlim(0, 2.0)
    ax2d.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax2d.set_ylabel("Jerk Magnitude  (m/s^3)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax2d.set_title(
        "Jerk Magnitude Over Time\n(Kinematic feature — production code path)",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax2d.tick_params(axis="both", colors=TEXT_COLOR)
    ax2d.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── Right: Real launch-site scatter ───────────────────────────────────────
    ax_s.set_facecolor(PANEL_BG)
    ax_s.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax_s.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax_s.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.4)

    if launch_df is not None:
        n_total = len(launch_df)
        # Per-class trajectory counts and unique launch sites
        class_counts = launch_df["label"].value_counts().sort_index().to_dict()
        # Alpha and marker size scaled so the rare Class 2 is clearly visible
        alphas = {0: 0.18, 1: 0.35, 2: 0.70}
        sizes = {0: 4, 1: 7, 2: 12}

        for cls in (0, 1, 2):
            sub = (
                launch_df[launch_df["label"] == cls][["launch_x", "launch_y"]]
                .drop_duplicates()
                .values
            )
            n_traj = class_counts.get(cls, 0)
            ax_s.scatter(
                sub[:, 0],
                sub[:, 1],
                c=CLASS_COLOR[cls],
                s=sizes[cls],
                alpha=alphas[cls],
                linewidths=0,
                zorder=2,
                label=f"Class {cls}  ({n_traj:,} trajs, {len(sub):,} sites)",
            )

        # 2-sigma ellipse enclosing the dominant DBSCAN group (all points)
        all_x = launch_df["launch_x"].values
        all_y = launch_df["launch_y"].values
        cx, cy = all_x.mean(), all_y.mean()
        sx, sy = all_x.std() * 2, all_y.std() * 2
        ellipse = mpatches.Ellipse(
            (cx, cy),
            width=sx * 2,
            height=sy * 2,
            fill=False,
            edgecolor=TEXT_COLOR,
            linewidth=1.2,
            linestyle="--",
            alpha=0.35,
            zorder=3,
        )
        ax_s.add_patch(ellipse)
        ax_s.text(
            cx,
            cy + sy + 0.01,
            f"1 DBSCAN group  ({n_total:,} traj, 99.7 %)\ngroup_* features near-zero importance",
            color=TEXT_COLOR,
            fontsize=7.5,
            ha="center",
            va="bottom",
            alpha=0.65,
        )

        ax_s.legend(
            fontsize=7.5,
            facecolor=PANEL_BG,
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
            loc="lower right",
            markerscale=2.5,
        )
        title_suffix = f"({n_total:,} trajectories · {len(launch_df[['launch_x', 'launch_y']].drop_duplicates()):,} unique sites)"
    else:
        ax_s.text(
            0.5,
            0.5,
            "Cache not found.\nRun 'make download-all'\nto see real launch sites.",
            transform=ax_s.transAxes,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=11,
        )
        title_suffix = "(cache not found)"

    ax_s.set_xlabel("Launch X (training coords)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax_s.set_ylabel("Launch Y (training coords)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax_s.set_title(
        f"Launch Sites — Real Training Data\n{title_suffix}",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax_s.tick_params(axis="both", colors=TEXT_COLOR)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Physics-Informed & Salvo-Context Feature Visualization",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    logger.info("Demo plot saved: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    make_demo_plot(assets_dir / "demo.png")
