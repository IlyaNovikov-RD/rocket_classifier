"""Streamlit interactive demo for rocket trajectory classification.

Generates a synthetic 3D trajectory from physical parameters controlled
by sidebar sliders, extracts physics features, and classifies in real time
using a LightGBM model trained via GPU-accelerated Optuna + salvo consensus.

Model: LightGBM (32 selected features — 25 kinematic + 7 salvo/group).
       Trained via research/train.py → 1.000000 OOB min-recall with
       proximity consensus (0.999874 raw).

Note on salvo/group features in the demo:
    The demo classifies a single synthetic trajectory in isolation.  The 7
    salvo and rebel-group features require a multi-trajectory dataset for
    DBSCAN clustering and cannot be computed for one trajectory alone.
    Those 7 features are set to NaN and imputed from training medians,
    so the demo prediction uses kinematic features only.  Production
    inference on a full test set computes all 32 features correctly.

Model loading strategy (in priority order):
    1. Local artifacts in artifacts/ (ONNX → native LightGBM).
    2. Remote download from the GitHub Release asset (automatic fallback).

Run with:
    streamlit run rocket_classifier/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from rocket_classifier import RELEASE_BASE_URL
from rocket_classifier.features import _compute_derivatives, _extract_trajectory_features
from rocket_classifier.model import SELECTED_FEATURES, RocketClassifier

_ARTIFACTS = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = _ARTIFACTS / "model.lgb"  # base path — from_artifacts resolves .onnx/.lgb
MEDIANS_PATH = _ARTIFACTS / "train_medians.npy"
BIASES_PATH = _ARTIFACTS / "threshold_biases.npy"
_RELEASE_BASE = RELEASE_BASE_URL
# All backends in the order RocketClassifier.from_artifacts() tries them.
_RELEASE_ARTIFACTS: dict[Path, str] = {
    _ARTIFACTS / "model_opt.onnx": f"{_RELEASE_BASE}/model_opt.onnx",
    _ARTIFACTS / "model.onnx": f"{_RELEASE_BASE}/model.onnx",
    _ARTIFACTS / "model.lgb": f"{_RELEASE_BASE}/model.lgb",
    MEDIANS_PATH: f"{_RELEASE_BASE}/train_medians.npy",
    BIASES_PATH: f"{_RELEASE_BASE}/threshold_biases.npy",
}

# ── Display constants ──────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
CLASS_COLORS = {0: "#3fb950", 1: "#f0c040", 2: "#f85149"}
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"
GOLD = "#f0c040"
BLUE = "#58a6ff"


# ── Model + feature loading ────────────────────────────────────────────────────


_MIN_ARTIFACT_BYTES: dict[str, int] = {
    "model_opt.onnx": 2_000_000,
    "model.onnx": 2_000_000,
    "model.lgb": 3_000_000,
    "train_medians.npy": 150,
    "threshold_biases.npy": 100,
}


def _ensure_artifact(path: Path, url: str) -> bool:
    """Download artifact from url to path if not already present.

    Writes to a temporary file first and renames on success so that a
    connection drop mid-download never leaves a corrupt partial file.
    After download, verifies the file meets a minimum expected size to
    guard against truncated or empty responses.

    Returns True if the file is available (existed or downloaded successfully),
    False if the download failed.
    """
    if path.exists():
        return True
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        size = tmp.stat().st_size
        min_size = _MIN_ARTIFACT_BYTES.get(path.name, 0)
        if size < min_size:
            tmp.unlink(missing_ok=True)
            st.warning(
                f"Downloaded {path.name} is too small ({size} bytes, "
                f"expected >= {min_size}) — possible truncation."
            )
            return False
        tmp.replace(path)
        return True
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        st.warning(f"Could not download {path.name}: {exc}")
        return False


@st.cache_resource
def load_classifier() -> RocketClassifier | None:
    """Load the RocketClassifier, downloading artifacts from GitHub Release if needed.

    Downloads ONNX and native LightGBM backends so
    ``RocketClassifier.from_artifacts`` picks the fastest available one:
    model_opt.onnx → model.onnx → model.lgb.

    Returns:
        A ready-to-use ``RocketClassifier``, or ``None`` if artifacts are unavailable.
    """
    _ARTIFACTS.mkdir(exist_ok=True)
    for path, url in _RELEASE_ARTIFACTS.items():
        _ensure_artifact(path, url)

    if not MODEL_PATH.exists():
        return None

    return RocketClassifier.from_artifacts(
        model_path=MODEL_PATH,
        medians_path=MEDIANS_PATH,
        biases_path=BIASES_PATH if BIASES_PATH.exists() else None,
    )


@st.cache_data
def get_feature_names() -> list[str]:
    """Return the 32 production feature names in the order the model expects.

    Returns:
        Ordered list of 32 feature name strings (25 kinematic + 7 salvo/group,
        selected via automated backward elimination in research/).
    """
    return SELECTED_FEATURES


# ── Trajectory generation ──────────────────────────────────────────────────────


def generate_trajectory(
    initial_speed: float,
    thrust_accel: float,
    noise_sigma: float,
    n: int = 100,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3D trajectory from physical parameters.

    Models a frictionless ballistic flight with an optional motor-burn phase.
    Increasing ``thrust_accel`` produces a sharp jerk spike at ignition —
    the key discriminating feature for propelled rockets.  Increasing
    ``noise_sigma`` creates an erratic path representative of non-rocket objects.

    Args:
        initial_speed: Launch speed magnitude in m/s.
        thrust_accel: Peak motor-burn acceleration in m/s² (0 = passive object).
        noise_sigma: Gaussian position noise std-dev in m (radar measurement error).
        n: Number of position samples. Defaults to 100.
        dt: Time step in seconds between samples. Defaults to 0.05 s (20 Hz).

    Returns:
        A tuple of:
            - pos: Position array of shape (n, 3) with columns [x, y, z].
            - t: Time array of shape (n,) in seconds.
    """
    rng = np.random.default_rng(seed=7)
    g = 9.81
    theta, phi = np.radians(55), np.radians(25)
    v0z = initial_speed * np.sin(theta)
    v0h = initial_speed * np.cos(theta)
    v0x, v0y = v0h * np.cos(phi), v0h * np.sin(phi)

    t = np.linspace(0, n * dt, n)
    x = v0x * t
    y = v0y * t
    z = v0z * t - 0.5 * g * t**2

    # Thrust phase: additional upward acceleration over the first 15% of flight.
    thrust_end = max(0.15 * t[-1], 1e-9)
    thrust_profile = np.where(t <= thrust_end, thrust_accel * (1.0 - t / thrust_end), 0.0)
    z += np.cumsum(thrust_profile) * dt**2
    z = np.maximum(z, 0.0)

    # Radar measurement noise
    sigma = max(noise_sigma, 1e-9)
    x += rng.normal(0, sigma, n)
    y += rng.normal(0, sigma, n)
    z += rng.normal(0, sigma, n)
    z = np.maximum(z, 0.0)

    return np.column_stack([x, y, z]), t


# ── Feature extraction + prediction ───────────────────────────────────────────


def classify(
    clf: RocketClassifier,
    pos: np.ndarray,
    t: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Extract production features from a trajectory and classify it.

    Delegates to ``_extract_trajectory_features`` so the exact same physics
    code path is used as during training. NaN imputation and bias-adjusted
    prediction are handled internally by ``RocketClassifier``.

    Args:
        clf: Loaded ``RocketClassifier`` instance.
        pos: Position array of shape (n, 3).
        t: Time array of shape (n,) in seconds.

    Returns:
        A tuple of:
            - class_idx: Predicted integer class label (0, 1, or 2).
            - proba: Float array of shape (3,) with per-class probabilities.
    """
    times = pd.to_datetime(t, unit="s", origin="2024-01-01")
    df = pd.DataFrame({"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "time_stamp": times})
    feats = _extract_trajectory_features(df)

    # Build the 32-feature vector in SELECTED_FEATURES order; missing → NaN
    # (salvo/group features unavailable for a single trajectory — imputed from
    # training medians by RocketClassifier._select_and_impute).
    vec = np.array([feats.get(k, np.nan) for k in SELECTED_FEATURES], dtype=np.float32)
    X = vec.reshape(1, -1)

    preds, proba = clf.predict_with_proba(X)
    return int(preds[0]), proba[0]


# ── Plotly 3D chart ────────────────────────────────────────────────────────────


def make_3d_figure(pos: np.ndarray, class_idx: int, confidence: float) -> go.Figure:
    """Build a dark-themed Plotly 3D trajectory figure.

    Args:
        pos: Position array of shape (n, 3).
        class_idx: Predicted class label (0, 1, or 2).
        confidence: Predicted probability of the winning class (0-1).

    Returns:
        Plotly Figure with trajectory line, launch marker, and apogee marker.
    """
    color = CLASS_COLORS[class_idx]
    apogee_idx = int(np.argmax(pos[:, 2]))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            line=dict(color=color, width=4),
            name=CLASS_NAMES[class_idx],
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            z=[pos[0, 2]],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle"),
            name="Launch",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pos[apogee_idx, 0]],
            y=[pos[apogee_idx, 1]],
            z=[pos[apogee_idx, 2]],
            mode="markers+text",
            marker=dict(size=10, color=GOLD, symbol="diamond"),
            text=["Apogee"],
            textfont=dict(color=GOLD, size=11),
            textposition="top center",
            name="Apogee",
        )
    )

    axis_style = dict(
        color=TEXT_COLOR,
        gridcolor=GRID_COLOR,
        showbackground=False,
        zerolinecolor=GRID_COLOR,
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        scene=dict(
            bgcolor=PANEL_BG,
            xaxis=dict(title="X (m)", **axis_style),
            yaxis=dict(title="Y (m)", **axis_style),
            zaxis=dict(title="Altitude (m)", **axis_style),
        ),
        font=dict(color=TEXT_COLOR),
        legend=dict(
            font=dict(color=TEXT_COLOR, size=11),
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            x=0.01,
            y=0.99,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
        title=dict(
            text=(
                f"Predicted: <b>{CLASS_NAMES[class_idx]}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;Confidence: <b>{confidence:.1%}</b>"
            ),
            font=dict(color=TEXT_COLOR, size=14),
            x=0.5,
        ),
    )
    return fig


# ── UI helpers ─────────────────────────────────────────────────────────────────


def _metric_card(
    col: st.delta_generator.DeltaGenerator, label: str, value: str, color: str
) -> None:
    col.markdown(
        f"""
        <div style="background:{PANEL_BG};padding:18px 20px;border-radius:8px;
                    border-left:4px solid {color};margin-bottom:8px">
          <div style="color:#8b949e;font-size:11px;font-weight:600;
                      text-transform:uppercase;letter-spacing:1px">{label}</div>
          <div style="color:{color};font-size:26px;font-weight:700;
                      margin-top:6px;line-height:1.1">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _prob_bar(i: int, prob: float, class_idx: int) -> None:
    bar_color = CLASS_COLORS[i]
    pct = prob * 100
    weight = "font-weight:700;" if i == class_idx else ""
    st.markdown(
        f"""
        <div style="margin-bottom:16px">
          <div style="color:{bar_color};font-size:13px;{weight}margin-bottom:5px">
            {CLASS_NAMES[i]}
          </div>
          <div style="background:#30363d;border-radius:4px;height:8px">
            <div style="background:{bar_color};height:8px;border-radius:4px;
                        width:{min(pct, 100):.1f}%">
            </div>
          </div>
          <div style="color:#8b949e;font-size:12px;margin-top:3px">{pct:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Main app ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the Streamlit demo application."""
    st.set_page_config(
        page_title="Rocket Classifier — Live Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {DARK_BG}; color: {TEXT_COLOR}; }}
        section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; }}
        div[data-testid="stMetricValue"] {{ color: {TEXT_COLOR}; }}
        h1, h2, h3 {{ color: {TEXT_COLOR}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    clf = load_classifier()
    feature_names = get_feature_names()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🚀 Rocket Classifier")
        st.markdown(
            "Adjust the physical parameters below to update the synthetic "
            "trajectory and watch the model classify it in real time."
        )
        st.markdown("---")
        st.markdown("### Trajectory Parameters")

        initial_speed = st.slider(
            "Initial Speed  (m/s)",
            min_value=5.0,
            max_value=120.0,
            value=55.0,
            step=1.0,
            help="Launch velocity magnitude — higher values yield larger apogee and longer range.",
        )
        thrust_accel = st.slider(
            "Thrust Acceleration  (m/s²)",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            step=5.0,
            help=(
                "Motor-burn peak acceleration. Higher values produce a sharp jerk spike "
                "at ignition — the key discriminating feature for propelled rockets."
            ),
        )
        noise_sigma = st.slider(
            "Measurement Noise  (m)",
            min_value=0.0,
            max_value=5.0,
            value=0.1,
            step=0.1,
            help=(
                "Radar position noise. High values create an erratic path "
                "characteristic of non-rocket or jamming objects."
            ),
        )

        st.markdown("---")
        if clf is None:
            st.error(
                "**Model not found.**\n\nDownload the model first:\n```\nmake download-models\n```"
            )
        else:
            st.success("✓ Model loaded")
            st.caption(f"{len(feature_names)} selected features (25 kinematic + 7 salvo/group)")

    # ── Compute ────────────────────────────────────────────────────────────────
    pos, t = generate_trajectory(initial_speed, thrust_accel, noise_sigma)

    if clf is not None:
        class_idx, proba = classify(clf, pos, t)
        confidence = float(proba[class_idx])
    else:
        class_idx = 0
        proba = np.full(3, 1.0 / 3.0)
        confidence = 1.0 / 3.0

    color = CLASS_COLORS[class_idx]
    apogee_z = float(np.max(pos[:, 2]))

    # Peak jerk via the shared finite-difference function in features.py
    _, _, jerk = _compute_derivatives(pos, np.diff(t))
    peak_jerk = float(np.max(np.linalg.norm(jerk, axis=1)))

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("# Rocket Trajectory Classifier — Live Demo")
    st.markdown(
        "Drag the sidebar sliders to modify the synthetic trajectory. "
        "The LightGBM model re-classifies on every change using the same "
        "kinematic feature pipeline as production (salvo/group features are "
        "unavailable for a single synthetic trajectory and are imputed from medians)."
    )

    # ── Metric cards ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "Predicted Class", CLASS_NAMES[class_idx], color)
    _metric_card(c2, "Confidence", f"{confidence:.1%}", color)
    _metric_card(c3, "Apogee Altitude", f"{apogee_z:.0f} m", GOLD)
    _metric_card(c4, "Peak Jerk", f"{peak_jerk:.0f} m/s³", BLUE)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3D chart + probability bars ────────────────────────────────────────────
    col_main, col_prob = st.columns([3, 1])

    with col_main:
        fig = make_3d_figure(pos, class_idx, confidence)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_prob:
        st.markdown(
            f"<div style='color:{TEXT_COLOR};font-size:14px;font-weight:600;"
            f"margin-bottom:16px'>Class Probabilities</div>",
            unsafe_allow_html=True,
        )
        for i in range(3):
            _prob_bar(i, float(proba[i]), class_idx)


if __name__ == "__main__":
    main()
