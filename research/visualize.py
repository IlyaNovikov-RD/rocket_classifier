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
    - Right:  Salvo context — top-down launch-position view of three synthetic
              rebel bases, each firing a 4-rocket salvo in sequence.
              Illustrates domain assumption 3b (rockets fired in salvos) and
              the ``salvo_time_rank`` feature (rank 12 in production model).

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
BLUE = "#58a6ff"  # salvo base accent
ORANGE = "#ffa657"  # salvo base accent

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

    speed=35 chosen so t_land ≈ 6.9s > n*dt=6s, avoiding a ground-clamp
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
# Salvo context data (synthetic)
# ---------------------------------------------------------------------------


def generate_salvo_data() -> tuple[list, list]:
    """Generate three synthetic rebel bases, each firing a 4-rocket salvo.

    Returns:
        (bases, salvos) where each base is a dict with 'center', 'color', and
        'label', and each salvo is a list of (launch_x, launch_y, rank) tuples.
    """
    bases = [
        {"center": np.array([1.5, 3.5]), "color": BLUE, "label": "Base A"},
        {"center": np.array([7.0, 1.5]), "color": ORANGE, "label": "Base B"},
        {"center": np.array([4.5, 7.0]), "color": GREEN, "label": "Base C"},
    ]

    salvos = []
    for base in bases:
        cx, cy = base["center"]
        angles = RNG.uniform(0, 2 * np.pi, 4)
        radii = RNG.uniform(0.05, 0.25, 4)
        points = [
            (cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles, strict=True)
        ]
        salvo = [(x, y, rank + 1) for rank, (x, y) in enumerate(points)]
        salvos.append(salvo)

    return bases, salvos


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

    bases, salvos = generate_salvo_data()

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
    # Thrust ramps linearly from max → 0, so the abrupt acceleration step is
    # at ignition (t≈0), not at thrust-end where acceleration is already ~0.
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

    # ── Right: Salvo context ───────────────────────────────────────────────────
    ax_s.set_facecolor(PANEL_BG)
    ax_s.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax_s.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax_s.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.4)

    rank_colors = [BLUE, GREEN, GOLD, RED]

    for base, salvo in zip(bases, salvos, strict=True):
        cx, cy = base["center"]
        color = base["color"]

        circle = mpatches.Circle(
            (cx, cy),
            radius=0.45,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.5,
        )
        ax_s.add_patch(circle)
        ax_s.text(
            cx, cy + 0.55, base["label"], color=color, fontsize=8.5, fontweight="bold", ha="center"
        )

        xs = [p[0] for p in salvo]
        ys = [p[1] for p in salvo]

        for i in range(len(salvo) - 1):
            ax_s.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops={"arrowstyle": "->", "color": TEXT_COLOR, "lw": 0.9, "alpha": 0.55},
            )

        for rx, ry, rank in salvo:
            rc = rank_colors[rank - 1]
            ax_s.scatter(rx, ry, s=120, color=rc, zorder=5, edgecolors=TEXT_COLOR, linewidths=0.6)
            ax_s.text(rx + 0.06, ry + 0.07, f"#{rank}", color=rc, fontsize=8, fontweight="bold")

    rank_handles = [
        mpatches.Patch(color=rank_colors[i], label=f"Salvo rank {i + 1}") for i in range(4)
    ]
    ax_s.legend(
        handles=rank_handles,
        fontsize=8,
        loc="lower right",
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    ax_s.set_xlabel("Launch X (normalised)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax_s.set_ylabel("Launch Y (normalised)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax_s.set_title(
        "Salvo Context: Launch Sequence\n(assumption 3b — salvo_time_rank, rank 12 in model)",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax_s.tick_params(axis="both", colors=TEXT_COLOR)
    ax_s.set_xlim(0, 10)
    ax_s.set_ylim(-0.5, 9)
    ax_s.set_aspect("equal")

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
