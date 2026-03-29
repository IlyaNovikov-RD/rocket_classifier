"""Streamlit interactive demo for rocket trajectory classification.

Generates a synthetic 3D trajectory from physical parameters controlled
by sidebar sliders, extracts the same 76 physics features used in
production, and classifies in real time using the trained XGBoost model.

Model loading strategy (in priority order):
    1. Local file ``model.pkl`` in the project root (fastest — created by
       running ``python src/main.py``).
    2. Remote download from the GitHub Release asset at
       ``MODEL_RELEASE_URL`` (automatic fallback when no local file exists,
       e.g. in a fresh clone or cloud deployment).  The download is cached
       by ``@st.cache_resource`` so it only happens once per session.

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
from xgboost import XGBClassifier

from rocket_classifier.features import _extract_trajectory_features

MODEL_PATH = Path(__file__).parent.parent / "model.pkl"
MEDIANS_PATH = Path(__file__).parent.parent / "train_medians.npy"
BIASES_PATH = Path(__file__).parent.parent / "threshold_biases.npy"
MODEL_RELEASE_URL = (
    "https://github.com/IlyaNovikov-RD/rocket_classifier"
    "/releases/download/v1.0.0/model.pkl"
)
MEDIANS_RELEASE_URL = (
    "https://github.com/IlyaNovikov-RD/rocket_classifier"
    "/releases/download/v1.0.0/train_medians.npy"
)
BIASES_RELEASE_URL = (
    "https://github.com/IlyaNovikov-RD/rocket_classifier"
    "/releases/download/v1.0.0/threshold_biases.npy"
)

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


@st.cache_resource
def _download_file(url: str) -> bytes | None:
    """Download a binary file from a URL, returning raw bytes or None on failure."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        chunks = []
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)
    except requests.RequestException as exc:
        st.warning(f"Could not download from {url}: {exc}")
        return None


@st.cache_resource
def load_model() -> XGBClassifier | None:
    """Load the trained XGBoost classifier using native XGBoost format.

    Resolution order:
        1. Local ``model.pkl`` in the project root.
        2. Remote download from GitHub Release.

    Returns:
        The XGBClassifier, or ``None`` if neither source is available.
    """
    if MODEL_PATH.exists():
        model = XGBClassifier()
        model.load_model(str(MODEL_PATH))
        return model

    # Fallback: fetch from GitHub Release and write to temp file for native load
    model_bytes = _download_file(MODEL_RELEASE_URL)
    if model_bytes is None:
        return None
    # Write to a temp path so XGBoost's native loader can read it
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        f.write(model_bytes)
        tmp_path = f.name
    model = XGBClassifier()
    model.load_model(tmp_path)
    return model


@st.cache_resource
def load_train_medians() -> np.ndarray | None:
    """Load per-feature training medians for consistent NaN imputation.

    Returns:
        NumPy array of per-column medians, or ``None`` if unavailable.
    """
    if MEDIANS_PATH.exists():
        return np.load(MEDIANS_PATH)

    medians_bytes = _download_file(MEDIANS_RELEASE_URL)
    if medians_bytes is None:
        return None
    import io

    return np.load(io.BytesIO(medians_bytes))


@st.cache_resource
def load_threshold_biases() -> np.ndarray | None:
    """Load per-class log-probability biases for threshold-tuned prediction.

    Returns:
        NumPy array of shape (3,), or ``None`` if unavailable.
    """
    if BIASES_PATH.exists():
        return np.load(BIASES_PATH)

    biases_bytes = _download_file(BIASES_RELEASE_URL)
    if biases_bytes is None:
        return None
    import io

    return np.load(io.BytesIO(biases_bytes))


@st.cache_data
def get_feature_names() -> list[str]:
    """Derive the canonical 76-feature order by calling the production extractor.

    Generates a minimal dummy trajectory, calls ``_extract_trajectory_features``
    once, and returns the dict keys in insertion order.  Because Python 3.7+
    preserves dict insertion order, this is deterministic and matches the column
    order used during model training.

    Returns:
        Ordered list of 76 feature name strings.
    """
    rng = np.random.default_rng(0)
    n = 20
    t = np.linspace(0, 1, n)
    pos = rng.random((n, 3))
    times = pd.to_datetime(t, unit="s", origin="2024-01-01")
    df = pd.DataFrame({"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "time_stamp": times})
    return list(_extract_trajectory_features(df).keys())


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
    model: XGBClassifier,
    pos: np.ndarray,
    t: np.ndarray,
    feature_names: list[str],
    train_medians: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    """Extract production features from a trajectory and classify it.

    Delegates to ``_extract_trajectory_features`` so the exact same physics
    code path is used as during training.

    Args:
        model: Fitted XGBClassifier.
        pos: Position array of shape (n, 3).
        t: Time array of shape (n,) in seconds.
        feature_names: Ordered list of 76 feature names matching training order.
        train_medians: Per-feature median array from training. Used to fill
            NaN values consistently with training. Falls back to 0.0 if None.

    Returns:
        A tuple of:
            - class_idx: Predicted integer class label (0, 1, or 2).
            - proba: Float array of shape (3,) with per-class probabilities.
    """
    times = pd.to_datetime(t, unit="s", origin="2024-01-01")
    df = pd.DataFrame({"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "time_stamp": times})
    feats = _extract_trajectory_features(df)
    vec = np.array([feats.get(k, 0.0) for k in feature_names], dtype=np.float32)

    # Fill NaN with train medians to match training imputation (prevent train-serve skew)
    nan_mask = np.isnan(vec)
    if nan_mask.any() and train_medians is not None:
        vec[nan_mask] = train_medians[nan_mask]
    else:
        vec = np.nan_to_num(vec, nan=0.0)

    X = vec.reshape(1, -1)

    proba = model.predict_proba(X)[0]
    biases = load_threshold_biases()
    if biases is not None:
        adjusted = np.log(proba + 1e-12) + biases
        class_idx = int(np.argmax(adjusted))
    else:
        class_idx = int(np.argmax(proba))
    return class_idx, proba


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


def _metric_card(col: st.delta_generator.DeltaGenerator, label: str, value: str, color: str) -> None:
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

    model = load_model()
    train_medians = load_train_medians()
    feature_names = get_feature_names()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"## 🚀 Rocket Classifier")  # noqa: F541
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
        if model is None:
            st.error(
                "**model.pkl not found.**\n\n"
                "Train the model first:\n"
                "```\nuv run python src/main.py\n```"
            )
        else:
            st.success("✓ Model loaded")
            st.caption(f"{len(feature_names)} physics features")

    # ── Compute ────────────────────────────────────────────────────────────────
    pos, t = generate_trajectory(initial_speed, thrust_accel, noise_sigma)

    if model is not None:
        class_idx, proba = classify(model, pos, t, feature_names, train_medians)
        confidence = float(proba[class_idx])
    else:
        class_idx = 0
        proba = np.full(3, 1.0 / 3.0)
        confidence = 1.0 / 3.0

    color = CLASS_COLORS[class_idx]
    apogee_z = float(np.max(pos[:, 2]))

    # Peak jerk via finite differences (mirrors _compute_derivatives logic)
    dt_arr = np.diff(t)
    vel = np.diff(pos, axis=0) / dt_arr[:, np.newaxis]
    dt_acc = (dt_arr[:-1] + dt_arr[1:]) / 2.0
    acc = np.diff(vel, axis=0) / dt_acc[:, np.newaxis]
    dt_jk = (dt_acc[:-1] + dt_acc[1:]) / 2.0
    jerk = np.diff(acc, axis=0) / dt_jk[:, np.newaxis]
    peak_jerk = float(np.max(np.linalg.norm(jerk, axis=1)))

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("# Rocket Trajectory Classifier — Live Demo")
    st.markdown(
        "Drag the sidebar sliders to modify the synthetic trajectory. "
        "The XGBoost model re-classifies on every change using the same "
        "76-feature physics pipeline as production."
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
