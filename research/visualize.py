"""Demo visualization — all three panels use 100 % real training data.

Generates ``assets/demo.png`` containing three subplots:

    - Left:   3D trajectory comparison — one representative real radar track per
              class (traj #50 Class 0, #126 Class 1, #554 Class 2), loaded
              directly from ``data/train.csv``.  Trajectories are centred at the
              launch origin for horizontal comparison while retaining true terrain
              altitude so the ``initial_z`` (SHAP rank 1) difference is visible.
              Colours match the Streamlit demo.
    - Centre: Real training-data scatter of the top two SHAP-confirmed
              non-geographic kinematic features: ``initial_z`` (SHAP rank 1,
              mean |SHAP| 2.01) and ``v_horiz_median`` (SHAP rank 3, 0.98).
              Per-class 2-sigma covariance ellipses show the class-conditional
              distributions.  Both features are in ``SELECTED_FEATURES`` and
              are used by the production model.
    - Right:  Launch-site decision map — a 1-NN classifier trained on the
              8,016 class-exclusive (launch_x, launch_y) sites paints the
              background to show which class "owns" each location.  Unique
              sites are scattered on top.  Directly visualises why launch_x
              and launch_y are SHAP ranks 2 and 5: knowing where a rocket
              launched almost determines its class.

All three panels require ``cache/cache_train_features.parquet`` and/or
``data/train.csv`` (run ``make download-all`` to fetch both).
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
# Real training-data loaders
# ---------------------------------------------------------------------------

_TRAIN_CSV_PATH = Path(__file__).parent.parent / "data" / "train.csv"

# Representative trajectory IDs — one per class, chosen to be close to the
# class median on (x_range, apogee_relative, n_points, initial_z).
# Class 0 traj #50: 36m range, 7.6m apogee rise, 36 points, z0=9.4m
# Class 1 traj #126: 28m range, 22.6m apogee rise, 38 points, z0=28.6m (high arc)
# Class 2 traj #554: 115m range, 22.1m apogee rise, 67 points, z0=44.8m (long range)
_REPR_TRAJ_IDS: dict[int, int] = {0: 50, 1: 126, 2: 554}


def load_real_trajectories() -> dict[int, np.ndarray] | None:
    """Load one representative real radar trajectory per class from train.csv.

    Trajectories are converted from training coordinate units to metres
    (multiply by 300), with x/y centred at the launch point so all three
    arcs share a common horizontal origin for visual comparison.  The z
    coordinate retains its true terrain altitude so the ``initial_z``
    (SHAP rank 1) difference between classes remains visible.

    Returns:
        Dict mapping class label → (n, 3) position array in metres,
        or None if ``data/train.csv`` is not present.
    """
    if not _TRAIN_CSV_PATH.exists():
        logger.warning(
            "train.csv not found at %s — left panel will be blank. "
            "Run 'make download-all' to fetch it.",
            _TRAIN_CSV_PATH,
        )
        return None

    target_ids = set(_REPR_TRAJ_IDS.values())
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(_TRAIN_CSV_PATH, chunksize=50_000):
        hit = chunk[chunk["traj_ind"].isin(target_ids)]
        if len(hit):
            rows.append(hit)
        if rows and target_ids.issubset(set(pd.concat(rows)["traj_ind"].values)):
            break

    if not rows:
        return None

    df = pd.concat(rows)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    result: dict[int, np.ndarray] = {}
    for cls, tid in _REPR_TRAJ_IDS.items():
        traj = df[df["traj_ind"] == tid].sort_values("time_stamp")
        pos = traj[["x", "y", "z"]].values * 300.0  # training units → metres
        pos[:, 0] -= pos[0, 0]  # centre x at launch origin
        pos[:, 1] -= pos[0, 1]  # centre y at launch origin
        # z stays absolute (terrain altitude) — initial_z is SHAP rank 1
        result[cls] = pos
    return result


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
    """Render and save the three-panel real-data visualization."""
    traj_data = load_real_trajectories()
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

    # ── Left: 3D real radar tracks ─────────────────────────────────────────────
    ax3d.set_facecolor(PANEL_BG)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax3d.tick_params(colors=TEXT_COLOR, labelsize=8)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.label.set_color(TEXT_COLOR)

    linestyles = {0: "-", 1: "--", 2: ":"}
    linewidths = {0: 2.2, 1: 2.0, 2: 2.0}

    if traj_data is not None:
        for cls in (0, 1, 2):
            pos = traj_data[cls]
            color = CLASS_COLOR[cls]
            tid = _REPR_TRAJ_IDS[cls]
            n_pts = len(pos)
            x_rng = float(np.ptp(pos[:, 0]))
            label = f"{CLASS_LABEL[cls]}  (traj #{tid} · {x_rng:.0f} m range · {n_pts} pts)"
            ax3d.plot(
                *pos.T,
                color=color,
                linewidth=linewidths[cls],
                linestyle=linestyles[cls],
                label=label,
                zorder=3,
            )
            ax3d.scatter(*pos[0], color=color, s=50, zorder=5, depthshade=False)
            aidx = int(np.argmax(pos[:, 2]))
            ax3d.scatter(
                *pos[aidx],
                color=GOLD if cls == 0 else color,
                s=100,
                marker="*",
                zorder=6,
                depthshade=False,
            )
        left_title_suffix = "(real radar tracks — one per class)"
    else:
        ax3d.text2D(
            0.5,
            0.5,
            "train.csv not found.\nRun 'make download-all'.",
            transform=ax3d.transAxes,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=11,
        )
        left_title_suffix = "(train.csv not found)"

    ax3d.set_xlabel("X (m)", labelpad=6, fontsize=9)
    ax3d.set_ylabel("Y (m)", labelpad=6, fontsize=9)
    ax3d.set_zlabel("Altitude (m)", labelpad=6, fontsize=9)
    ax3d.set_title(
        f"3D Trajectory — Real Training Data\n{left_title_suffix}",
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax3d.legend(
        loc="upper left",
        fontsize=7.5,
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
        from matplotlib.colors import ListedColormap
        from sklearn.neighbors import KNeighborsClassifier

        n_total = len(launch_df)
        class_counts = launch_df["label"].value_counts().sort_index().to_dict()

        # ── Decision background: 1-NN on unique sites → class ────────────────
        # Paints a colour field showing which class "owns" each location.
        # Because sites are class-exclusive, 1-NN accuracy is ~100% — this
        # directly visualises why launch_x/y are SHAP ranks 2 & 5.
        unique_sites = launch_df[["launch_x", "launch_y", "label"]].drop_duplicates(
            subset=["launch_x", "launch_y"]
        )
        knn = KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree")
        knn.fit(unique_sites[["launch_x", "launch_y"]].values, unique_sites["label"].values)

        margin = 0.05
        x0, x1 = launch_df["launch_x"].min() - margin, launch_df["launch_x"].max() + margin
        y0, y1 = launch_df["launch_y"].min() - margin, launch_df["launch_y"].max() + margin
        xx, yy = np.meshgrid(np.linspace(x0, x1, 350), np.linspace(y0, y1, 350))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        cmap_bg = ListedColormap([GREEN, GOLD, RED])
        ax_s.pcolormesh(xx, yy, Z, cmap=cmap_bg, alpha=0.18, zorder=1, shading="auto")

        # ── Scatter: unique sites per class on top ────────────────────────────
        sizes = {0: 4, 1: 7, 2: 12}
        alphas = {0: 0.30, 1: 0.50, 2: 0.80}
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

        n_unique = len(unique_sites)
        ax_s.text(
            0.02,
            0.98,
            f"Background: 1-NN decision regions from {n_unique:,} class-exclusive sites\n"
            "launch_x / launch_y  —  SHAP ranks 2 & 5",
            transform=ax_s.transAxes,
            color=TEXT_COLOR,
            fontsize=7.5,
            ha="left",
            va="top",
            alpha=0.80,
            zorder=4,
        )

        ax_s.legend(
            fontsize=7.5,
            facecolor=PANEL_BG,
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
            loc="lower right",
            markerscale=2.5,
        )
        title_suffix = f"({n_total:,} trajectories · {n_unique:,} unique sites)"
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
        "Rocket Trajectory Classifier — Real Training Data (32,741 trajectories)",
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
