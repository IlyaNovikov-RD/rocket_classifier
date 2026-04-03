"""Demo visualization of physics-informed and salvo-context features.

Generates ``assets/demo.png`` containing three subplots:

    - Left:   3D trajectory comparison — ballistic Rocket (solid green) vs.
              stochastic Noise/Other (dashed red), with apogee marked.
    - Centre: Jerk Magnitude time-series computed via finite differences,
              illustrating the sharp ignition spike that distinguishes
              propelled rockets from passive or erratic objects.
    - Right:  Salvo context — top-down launch-position view of two synthetic
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
# Colour palette
# ---------------------------------------------------------------------------
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
GREEN = "#3fb950"
RED = "#f85149"
GOLD = "#f0c040"
BLUE = "#58a6ff"
ORANGE = "#ffa657"

# ---------------------------------------------------------------------------
# Synthetic trajectory generation
# ---------------------------------------------------------------------------


def generate_rocket(n: int = 120, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic ballistic trajectory with an initial thrust phase."""
    g = 9.81
    v0x, v0y, v0z = 18.0, 12.0, 55.0
    thrust_duration = 0.10

    t = np.linspace(0, n * dt, n)
    x = v0x * t
    y = v0y * t
    z = v0z * t - 0.5 * g * t**2

    thrust_mask = t < thrust_duration * t[-1]
    thrust_profile = np.where(thrust_mask, 80.0 * (1 - t / (thrust_duration * t[-1])), 0.0)
    z += np.cumsum(thrust_profile) * dt**2

    x += RNG.normal(0, 0.05, n)
    y += RNG.normal(0, 0.05, n)
    z += RNG.normal(0, 0.05, n)

    pos = np.column_stack([x, y, z])
    return pos, np.full(n - 1, dt)


def generate_noise(n: int = 120, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate a high-jitter stochastic trajectory (non-ballistic object)."""
    base_vx = RNG.uniform(5, 15)
    base_vz = RNG.uniform(2, 8)

    x = np.cumsum(base_vx * dt + RNG.normal(0, 1.2, n))
    y = np.cumsum(RNG.normal(0, 1.0, n))
    z = np.maximum(np.cumsum(base_vz * dt + RNG.normal(0, 1.8, n)), 0.0)

    pos = np.column_stack([x, y, z])
    return pos, np.full(n - 1, dt)


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
    """Generate two synthetic rebel bases, each firing a 4-rocket salvo.

    Returns:
        (bases, salvos) where each base is a dict with 'center' and 'color',
        and each salvo is a list of (launch_x, launch_y, rank) tuples.
    """
    bases = [
        {"center": np.array([2.0, 3.0]), "color": GREEN, "label": "Base A"},
        {"center": np.array([7.0, 1.5]), "color": ORANGE, "label": "Base B"},
    ]

    salvos = []
    for base in bases:
        cx, cy = base["center"]
        # 4 rockets per salvo: small spatial scatter around the base
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
    rocket_pos, rocket_dt = generate_rocket()
    noise_pos, noise_dt = generate_noise()

    rocket_jerk = compute_jerk_magnitude(rocket_pos, rocket_dt)
    noise_jerk = compute_jerk_magnitude(noise_pos, noise_dt)

    dt = rocket_dt[0]
    t_jerk_rocket = np.arange(len(rocket_jerk)) * dt
    t_jerk_noise = np.arange(len(noise_jerk)) * dt
    apogee_idx = int(np.argmax(rocket_pos[:, 2]))

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

    ax3d.plot(*rocket_pos.T, color=GREEN, linewidth=2.2, label="Rocket (Ballistic)", zorder=3)
    ax3d.plot(
        *noise_pos.T,
        color=RED,
        linewidth=1.6,
        linestyle="--",
        alpha=0.85,
        label="Other (Noise)",
        zorder=2,
    )
    ax3d.scatter(*rocket_pos[0], color=GREEN, s=60, zorder=5, depthshade=False)
    ax3d.scatter(*noise_pos[0], color=RED, s=60, zorder=5, depthshade=False)
    ax3d.scatter(
        *rocket_pos[apogee_idx],
        color=GOLD,
        s=120,
        marker="*",
        zorder=6,
        depthshade=False,
        label="Apogee",
    )
    ax3d.text(
        rocket_pos[apogee_idx, 0] + 1.5,
        rocket_pos[apogee_idx, 1] + 1.5,
        rocket_pos[apogee_idx, 2] + 2.0,
        "Apogee",
        color=GOLD,
        fontsize=9,
        fontweight="bold",
    )
    ax3d.set_xlabel("X (m)", labelpad=6, fontsize=9)
    ax3d.set_ylabel("Y (m)", labelpad=6, fontsize=9)
    ax3d.set_zlabel("Z — Altitude (m)", labelpad=6, fontsize=9)
    ax3d.set_title(
        "3D Trajectory Comparison", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12
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

    ax2d.plot(t_jerk_rocket, rocket_jerk, color=GREEN, linewidth=2.0, label="Rocket (Ballistic)")
    ax2d.plot(
        t_jerk_noise,
        noise_jerk,
        color=RED,
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
        label="Other (Noise)",
    )

    peak_idx = int(np.argmax(rocket_jerk))
    ax2d.annotate(
        "Thrust\nIgnition\nSpike",
        xy=(t_jerk_rocket[peak_idx], rocket_jerk[peak_idx]),
        xytext=(t_jerk_rocket[peak_idx] + 0.4, rocket_jerk[peak_idx] * 0.85),
        color=GOLD,
        fontsize=8.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.5},
    )
    ax2d.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax2d.set_ylabel("Jerk Magnitude  (m/s³)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax2d.set_title(
        "Jerk Magnitude Over Time\n(Kinematic feature — production code path)",
        color=TEXT_COLOR,
        fontsize=12,
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

        # Base perimeter circle (geographic concentration — assumption 3c)
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

        # Draw rockets in salvo order, connected by a dashed arrow sequence
        xs = [p[0] for p in salvo]
        ys = [p[1] for p in salvo]

        # Arrows connecting rank 1→2→3→4
        for i in range(len(salvo) - 1):
            ax_s.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops={"arrowstyle": "->", "color": TEXT_COLOR, "lw": 0.9, "alpha": 0.55},
            )

        # Scatter each rocket, coloured by rank
        for rx, ry, rank in salvo:
            rc = rank_colors[rank - 1]
            ax_s.scatter(rx, ry, s=120, color=rc, zorder=5, edgecolors=TEXT_COLOR, linewidths=0.6)
            ax_s.text(rx + 0.06, ry + 0.07, f"#{rank}", color=rc, fontsize=8, fontweight="bold")

    # Legend for rank colours
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
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_s.tick_params(axis="both", colors=TEXT_COLOR)
    ax_s.set_xlim(0, 10)
    ax_s.set_ylim(-0.5, 5)
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
