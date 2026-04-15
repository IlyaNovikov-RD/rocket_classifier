"""Demo visualization of physics-informed and salvo-context features.

Generates ``assets/demo.png`` containing three subplots:

    - Left:   3D trajectory comparison — all three rocket classes produced by
              the production model: Class 0 (moderate arc, 69 %),
              Class 1 (long-range flat, 24 %), Class 2 (steep short-range, 7 %).
              Apogee marked per class.  Colours match the Streamlit demo.
    - Centre: Real training-data scatter of the top two SHAP-confirmed
              non-geographic kinematic features: ``initial_z`` (SHAP rank 1,
              mean |SHAP| 2.01) and ``v_horiz_median`` (SHAP rank 3, 0.98).
              Per-class 2-sigma covariance ellipses show the class-conditional
              distributions.  Both features are in ``SELECTED_FEATURES`` and
              are used by the production model.
    - Right:  Real launch-site scatter from the training cache — every unique
              (launch_x, launch_y) coordinate, coloured by class.  Production
              DBSCAN (eps=0.25, min_samples=3) finds 1 dominant group containing
              99.7 % of trajectories, explaining why the rebel-group features
              have near-zero importance in the trained model.  Requires
              ``cache/cache_train_features.parquet`` (run ``make download-all``).
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
        dt_arr: (n-1,) constant dt array (uniform, for external use if needed).
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
    """Class 0 (69 %): Slowest, shortest range — the common low-cost rocket.

    Parameters scaled to match training-data medians (initial_speed ~1.0x baseline,
    x_range ~1.0x baseline).  Moderate elevation, standard motor burn.
    """
    return _ballistic(
        speed=38, theta_deg=62, phi_deg=25, thrust_accel=80, thrust_frac=0.10, n=n, dt=dt
    )


def generate_class1(n: int = _N, dt: float = _DT) -> tuple[np.ndarray, np.ndarray]:
    """Class 1 (24 %): 1.35x faster, similar range to Class 0, higher apogee.

    Real data: initial_speed 1.35x, x_range only 1.06x, apogee_relative 1.37x,
    acc_mag_max 0.60x (smoother flight).  Steeper angle keeps range close to
    Class 0 despite higher speed; lower thrust reflects smoother kinematics.
    """
    return _ballistic(
        speed=51, theta_deg=68, phi_deg=20, thrust_accel=55, thrust_frac=0.12, n=n, dt=dt
    )


def generate_class2(n: int = _N, dt: float = _DT) -> tuple[np.ndarray, np.ndarray]:
    """Class 2 (7 %): Fastest, longest range — the rare high-capability rocket.

    Real data: initial_speed 1.77x, x_range 2.96x, apogee_relative 3.14x,
    acc_mag_max 1.61x (stronger motor).  Flatter angle maximises range.
    All t_land values exceed 6 s (no ground-clamp artifact).
    """
    return _ballistic(
        speed=67, theta_deg=40, phi_deg=30, thrust_accel=160, thrust_frac=0.15, n=n, dt=dt
    )


# ---------------------------------------------------------------------------
# Real training-data loader
# ---------------------------------------------------------------------------

_CACHE_PATH = Path(__file__).parent.parent / "cache" / "cache_train_features.parquet"


def load_real_launch_data() -> pd.DataFrame | None:
    """Load training features and class labels from the training feature cache.

    Returns:
        DataFrame with columns [launch_x, launch_y, initial_z, v_horiz_median, label],
        or None if the cache parquet is not present (run ``make download-all``).
    """
    if not _CACHE_PATH.exists():
        logger.warning(
            "Training cache not found at %s — centre and right panels will be blank. "
            "Run 'make download-all' to fetch it.",
            _CACHE_PATH,
        )
        return None
    return pd.read_parquet(
        _CACHE_PATH,
        columns=["launch_x", "launch_y", "initial_z", "v_horiz_median", "label"],
    )


# ---------------------------------------------------------------------------
# Three-panel figure
# ---------------------------------------------------------------------------


def make_demo_plot(output_path: Path) -> None:
    """Render and save the three-panel physics + salvo feature visualization."""
    pos0, _ = generate_class0()
    pos1, _ = generate_class1()
    pos2, _ = generate_class2()

    apogee = {
        0: int(np.argmax(pos0[:, 2])),
        1: int(np.argmax(pos1[:, 2])),
        2: int(np.argmax(pos2[:, 2])),
    }
    positions = {0: pos0, 1: pos1, 2: pos2}

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

    # ── Centre: Real feature distributions (SHAP top kinematic features) ─────────
    ax2d.set_facecolor(PANEL_BG)
    ax2d.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax2d.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax2d.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)

    if launch_df is not None:
        # Subsample for visual clarity: cap Class 0 at 2000 so it doesn't drown
        # Classes 1 and 2; show all of Class 2 (only 2,339 trajectories).
        rng_sub = np.random.default_rng(seed=0)
        sub_n = {0: 2000, 1: 2000, 2: len(launch_df[launch_df["label"] == 2])}
        point_alphas = {0: 0.15, 1: 0.28, 2: 0.55}
        point_sizes = {0: 5, 1: 7, 2: 11}

        for cls in (0, 1, 2):
            cls_df = launch_df[launch_df["label"] == cls]
            n_draw = min(sub_n[cls], len(cls_df))
            idx = rng_sub.choice(len(cls_df), size=n_draw, replace=False)
            sub = cls_df.iloc[idx]

            ax2d.scatter(
                sub["initial_z"],
                sub["v_horiz_median"],
                c=CLASS_COLOR[cls],
                s=point_sizes[cls],
                alpha=point_alphas[cls],
                linewidths=0,
                zorder=2,
                label=CLASS_LABEL[cls],
            )

            # 2-sigma covariance ellipse — uses full class data, not just the subsample
            full = launch_df[launch_df["label"] == cls]
            mu = np.array([full["initial_z"].mean(), full["v_horiz_median"].mean()])
            cov = np.cov(full["initial_z"].values, full["v_horiz_median"].values)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eig_vecs[1, -1], eig_vecs[0, -1]))
            w, h = 2 * 2 * np.sqrt(np.abs(eig_vals))
            ellipse = mpatches.Ellipse(
                mu,
                width=w,
                height=h,
                angle=angle,
                fill=False,
                edgecolor=CLASS_COLOR[cls],
                linewidth=1.8,
                linestyle="--",
                alpha=0.85,
                zorder=3,
            )
            ax2d.add_patch(ellipse)

        center_title_suffix = f"({len(launch_df):,} trajectories · subsampled for display)"
        ax2d.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    else:
        ax2d.text(
            0.5,
            0.5,
            "Cache not found.\nRun 'make download-all'\nto see real features.",
            transform=ax2d.transAxes,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=11,
        )
        center_title_suffix = "(cache not found)"

    ax2d.set_xlabel(
        "initial_z  (SHAP rank 1 · mean|SHAP| 2.01)", color=TEXT_COLOR, fontsize=10, labelpad=6
    )
    ax2d.set_ylabel(
        "v_horiz_median  (SHAP rank 3 · mean|SHAP| 0.98)", color=TEXT_COLOR, fontsize=10, labelpad=6
    )
    ax2d.set_title(
        f"Key Kinematic Features — Real Training Data\n{center_title_suffix}",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax2d.tick_params(axis="both", colors=TEXT_COLOR)

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
