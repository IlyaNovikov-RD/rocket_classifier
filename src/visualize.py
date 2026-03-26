"""
Generate a professional demo visualization of physics-informed features.

Produces demo.png in the project root with:
  - Left:  3D trajectory plot comparing a ballistic Rocket vs. a Noisy/Other trajectory.
  - Right: Time-series of Jerk Magnitude, showing how clean ballistic physics
           produces a characteristic signature vs. high-frequency noise.

Uses _compute_derivatives from src/features.py to derive acceleration and jerk
directly from synthetic position/time data — the same code path used in production.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display required

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import _compute_derivatives

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Synthetic trajectory generation
# ---------------------------------------------------------------------------

def generate_rocket(n: int = 120, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Ballistic parabolic trajectory with initial high-thrust phase.

    Physics:
      x(t) = v0x * t
      y(t) = v0y * t
      z(t) = v0z * t - 0.5 * g * t^2   (standard ballistic, flat terrain)

    A brief thrust phase (first 10% of flight) adds extra upward acceleration,
    giving the jerk a sharp, distinctive spike at ignition — a real-world
    signature that differentiates propelled rockets from passive projectiles.

    Returns:
        pos: (n, 3) position array [x, y, z]
        dt_arr: (n-1,) uniform time delta array
    """
    g = 9.81
    v0x, v0y, v0z = 18.0, 12.0, 55.0
    thrust_duration = 0.10  # fraction of total flight

    t = np.linspace(0, n * dt, n)
    x = v0x * t
    y = v0y * t
    z = v0z * t - 0.5 * g * t**2

    # Thrust phase: additional upward kick in z (simulates motor burn)
    thrust_mask = t < thrust_duration * t[-1]
    thrust_profile = np.where(thrust_mask, 80.0 * (1 - t / (thrust_duration * t[-1])), 0.0)
    z += np.cumsum(thrust_profile) * dt**2

    # Small sensor noise (radar measurement error)
    x += RNG.normal(0, 0.05, n)
    y += RNG.normal(0, 0.05, n)
    z += RNG.normal(0, 0.05, n)

    pos = np.column_stack([x, y, z])
    dt_arr = np.full(n - 1, dt)
    return pos, dt_arr


def generate_noise(n: int = 120, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """High-jitter random-walk trajectory (non-ballistic / unknown object).

    Simulates an erratic trajectory with no clear ballistic arc:
    high-frequency direction changes, irregular speed, no smooth apogee.

    Returns:
        pos: (n, 3) position array [x, y, z]
        dt_arr: (n-1,) uniform time delta array
    """
    # Slow drift with large stochastic perturbations
    base_vx = RNG.uniform(5, 15)
    base_vz = RNG.uniform(2, 8)

    x = np.cumsum(base_vx * dt + RNG.normal(0, 1.2, n))
    y = np.cumsum(RNG.normal(0, 1.0, n))
    z = np.cumsum(base_vz * dt + RNG.normal(0, 1.8, n))

    # Keep z >= 0 (ground clamp)
    z = np.maximum(z, 0.0)

    pos = np.column_stack([x, y, z])
    dt_arr = np.full(n - 1, dt)
    return pos, dt_arr


# ---------------------------------------------------------------------------
# Feature extraction using production code
# ---------------------------------------------------------------------------

def compute_jerk_magnitude(pos: np.ndarray, dt_arr: np.ndarray) -> np.ndarray:
    """Return per-sample jerk magnitude using _compute_derivatives from features.py."""
    _, _, jerk = _compute_derivatives(pos, dt_arr)
    if jerk.shape[0] == 0:
        return np.array([])
    return np.linalg.norm(jerk, axis=1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_demo_plot(output_path: Path) -> None:
    rocket_pos, rocket_dt = generate_rocket()
    noise_pos, noise_dt = generate_noise()

    rocket_jerk = compute_jerk_magnitude(rocket_pos, rocket_dt)
    noise_jerk = compute_jerk_magnitude(noise_pos, noise_dt)

    # Time axes for jerk (jerk has n-3 samples from n points)
    dt = rocket_dt[0]
    # jerk starts at index 1.5 (midpoint of midpoint) — use a simple offset
    t_jerk_rocket = np.arange(len(rocket_jerk)) * dt
    t_jerk_noise = np.arange(len(noise_jerk)) * dt

    # --- Apogee ---
    apogee_idx = int(np.argmax(rocket_pos[:, 2]))

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(16, 7), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.04, right=0.97,
        top=0.88, bottom=0.10,
        wspace=0.32,
    )

    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])

    DARK_BG    = "#0d1117"
    PANEL_BG   = "#161b22"
    GRID_COLOR = "#30363d"
    TEXT_COLOR = "#e6edf3"
    GREEN      = "#3fb950"
    RED        = "#f85149"
    GOLD       = "#f0c040"
    BLUE_MUTED = "#58a6ff"

    # ---------------------------------------------------------------- 3D panel
    for ax in [ax3d]:
        ax.set_facecolor(PANEL_BG)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.zaxis.label.set_color(TEXT_COLOR)

    ax3d.plot(
        rocket_pos[:, 0], rocket_pos[:, 1], rocket_pos[:, 2],
        color=GREEN, linewidth=2.2, linestyle="-", label="Rocket (Ballistic)",
        zorder=3,
    )
    ax3d.plot(
        noise_pos[:, 0], noise_pos[:, 1], noise_pos[:, 2],
        color=RED, linewidth=1.6, linestyle="--", alpha=0.85, label="Other (Noise)",
        zorder=2,
    )

    # Launch markers
    ax3d.scatter(*rocket_pos[0], color=GREEN, s=60, zorder=5, depthshade=False)
    ax3d.scatter(*noise_pos[0],  color=RED,   s=60, zorder=5, depthshade=False)

    # Apogee marker
    ax3d.scatter(
        *rocket_pos[apogee_idx],
        color=GOLD, s=120, marker="*", zorder=6, depthshade=False, label="Apogee",
    )
    ax3d.text(
        rocket_pos[apogee_idx, 0] + 1.5,
        rocket_pos[apogee_idx, 1] + 1.5,
        rocket_pos[apogee_idx, 2] + 2.0,
        "Apogee", color=GOLD, fontsize=9, fontweight="bold",
    )

    ax3d.set_xlabel("X (m)", labelpad=6, fontsize=9)
    ax3d.set_ylabel("Y (m)", labelpad=6, fontsize=9)
    ax3d.set_zlabel("Z — Altitude (m)", labelpad=6, fontsize=9)
    ax3d.set_title("3D Trajectory Comparison", color=TEXT_COLOR,
                   fontsize=13, fontweight="bold", pad=12)

    leg3d = ax3d.legend(
        loc="upper left", fontsize=9,
        facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
    )

    ax3d.view_init(elev=22, azim=-55)

    # ---------------------------------------------------------------- 2D panel
    ax2d.set_facecolor(PANEL_BG)
    ax2d.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax2d.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax2d.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)

    ax2d.plot(
        t_jerk_rocket, rocket_jerk,
        color=GREEN, linewidth=2.0, linestyle="-", label="Rocket (Ballistic)",
    )
    ax2d.plot(
        t_jerk_noise, noise_jerk,
        color=RED, linewidth=1.4, linestyle="--", alpha=0.85, label="Other (Noise)",
    )

    # Annotate the thrust-ignition jerk spike
    peak_idx = int(np.argmax(rocket_jerk))
    ax2d.annotate(
        "Thrust\nIgnition\nSpike",
        xy=(t_jerk_rocket[peak_idx], rocket_jerk[peak_idx]),
        xytext=(t_jerk_rocket[peak_idx] + 0.4, rocket_jerk[peak_idx] * 0.85),
        color=GOLD, fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5),
    )

    ax2d.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax2d.set_ylabel("Jerk Magnitude  (m/s³)", color=TEXT_COLOR, fontsize=11, labelpad=6)
    ax2d.set_title("Jerk Magnitude Over Time\n(Computed via Finite Differences)",
                   color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax2d.tick_params(axis="x", colors=TEXT_COLOR)
    ax2d.tick_params(axis="y", colors=TEXT_COLOR)

    leg2d = ax2d.legend(
        fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
    )

    # ------------------------------------------------------------------ title
    fig.suptitle(
        "Physics-Informed Feature Visualization",
        color=TEXT_COLOR, fontsize=17, fontweight="bold", y=0.97,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    output = Path(__file__).parent.parent / "demo.png"
    make_demo_plot(output)
