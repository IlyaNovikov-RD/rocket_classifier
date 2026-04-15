"""Streamlit interactive demo for rocket trajectory classification.

Generates a synthetic 3D trajectory from physical parameters controlled
by sidebar sliders, extracts physics features, and classifies in real time
using a LightGBM model trained via GPU-accelerated Optuna + salvo consensus.

Model: LightGBM (32 selected features — 25 kinematic + 7 salvo/group).
       Trained via research/train.py → 1.000000 OOB min-recall with
       proximity consensus (0.999874 raw).

Note on features and biases in the demo:
    The demo classifies a single synthetic trajectory in isolation.

    Coordinate scaling:
      Training data uses projected geographic coordinates where 1 unit ≈ 300 m.
      The demo generates trajectories in metres; classify() scales positions by
      1/300 before feature extraction so kinematic features (speeds, accels)
      land in the training distribution (initial_speed p5-p95 ≈ 0.08-0.33
      units/s ≈ 24-99 m/s).  The 3D visualisation always shows unscaled metres.

    Rebel group context:
      launch_x and launch_y (SHAP ranks 2 and 5, near-deterministic class proxies)
      are set to the class-specific training medians chosen by the "Rebel Group"
      selector.  Individual launch sites are class-exclusive (zero cross-class
      overlap across 32,741 trajectories), so geographic position is the dominant
      classifier signal.  Using the global training median would silently inject a
      Class 0 prior (it equals the Class 0 median because Class 0 is 69 % of data).

    Features set to NaN (imputed from global training medians):
      - 7 salvo/group features: require multi-trajectory DBSCAN clustering.

    Threshold biases not applied:
      The production biases [0.0, -0.253, +1.266] correct for the 7 %
      class-2 prevalence in the production population.  Applied to a single
      synthetic trajectory with no prior context, the +1.266 class-2 bias
      dominates and locks the prediction to class 2 regardless of the
      kinematic features.  The demo uses raw argmax(proba) so slider changes
      produce visible class transitions driven by kinematic signal.

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

# Per-class training-data medians for contextual feature imputation.
# Individual launch sites are class-exclusive (zero cross-class overlap across
# 32,741 trajectories); geographic position is the top-2 SHAP feature and
# acts as a near-deterministic class proxy.  The demo sets launch_x/launch_y
# to the class-specific median so that kinematic features can drive the result.
_CLASS_LAUNCH_X = {0: 0.5410, 1: 0.8499, 2: 0.6965}
_CLASS_LAUNCH_Y = {0: 0.0996, 1: 0.1551, 2: 0.2768}
# Training-data median initial_z (launch-site terrain altitude) per class, metres.
# initial_z is the single most important SHAP feature (rank 1, mean|SHAP| 2.01).
_CLASS_INIT_ALT_M = {0: 12.0, 1: 24.0, 2: 39.0}
# Training-data typical speed range for the sidebar hint, m/s.
_CLASS_SPEED_HINT = {0: "30-50", 1: "45-70", 2: "60-100"}
# Training-data typical launch angle range for the sidebar hint, degrees.
_CLASS_ANGLE_HINT = {0: "50-75", 1: "55-80", 2: "10-35"}


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
    launch_angle_deg: float = 45.0,
    initial_alt_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3D trajectory from physical parameters.

    Models a frictionless ballistic flight with an optional motor-burn phase.
    The trajectory starts at ``initial_alt_m`` metres above ground, matching
    the terrain elevation of a real launch site (training-data median: Class 0
    ~12 m, Class 1 ~24 m, Class 2 ~39 m).  ``initial_z`` is the single most
    important SHAP feature (rank 1, mean|SHAP| 2.01), so setting it correctly
    is critical for accurate classification.

    Args:
        initial_speed: Launch speed magnitude in m/s.
        thrust_accel: Peak motor-burn acceleration in m/s² (0 = passive object).
        noise_sigma: Gaussian position noise std-dev in m (radar measurement error).
        n: Number of position samples. Defaults to 100.
        dt: Time step in seconds between samples. Defaults to 0.05 s (20 Hz).
        launch_angle_deg: Elevation angle above horizontal in degrees. Defaults to 45°.
        initial_alt_m: Terrain altitude of the launch site in metres. Shifts the
            entire trajectory vertically so ``initial_z`` matches training data.

    Returns:
        A tuple of:
            - pos: Position array of shape (n, 3) with columns [x, y, z].
            - t: Time array of shape (n,) in seconds.
    """
    rng = np.random.default_rng(seed=7)
    g = 9.81
    theta, phi = np.radians(launch_angle_deg), np.radians(25)
    v0z = initial_speed * np.sin(theta)
    v0h = initial_speed * np.cos(theta)
    v0x, v0y = v0h * np.cos(phi), v0h * np.sin(phi)

    t = np.linspace(0, n * dt, n)
    x = v0x * t
    y = v0y * t
    z = initial_alt_m + v0z * t - 0.5 * g * t**2

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
    rebel_class: int = 0,
) -> tuple[int, np.ndarray]:
    """Extract production features from a trajectory and classify it.

    Delegates to ``_extract_trajectory_features`` so the exact same physics
    code path is used as during training.  NaN imputation is handled by
    ``RocketClassifier._select_and_impute``; threshold biases are intentionally
    not applied (see module docstring).

    Args:
        clf: Loaded ``RocketClassifier`` instance.
        pos: Position array of shape (n, 3).
        t: Time array of shape (n,) in seconds.
        rebel_class: Rebel group context (0, 1, or 2).  Sets ``launch_x`` and
            ``launch_y`` to the class-specific training-data medians.  Launch
            position is the top-2 SHAP feature and near-deterministic for class;
            imputing from the global median (dominated by Class 0, 69 % of data)
            would inject a Class 0 geographic prior regardless of kinematics.

    Returns:
        A tuple of:
            - class_idx: Predicted integer class label (0, 1, or 2).
            - proba: Float array of shape (3,) with per-class probabilities.
    """
    # Scale from metres to training coordinate units (1 unit ≈ 300 m).
    # Training data uses projected geographic coordinates; kinematic features
    # computed directly from metre-scale positions would be ~300x out of the
    # training distribution, causing trees to boundary-clamp and ignore sliders.
    _COORD_SCALE = 1.0 / 300.0
    pos_scaled = pos * _COORD_SCALE
    times = pd.to_datetime(t, unit="s", origin="2024-01-01")
    df = pd.DataFrame(
        {"x": pos_scaled[:, 0], "y": pos_scaled[:, 1], "z": pos_scaled[:, 2], "time_stamp": times}
    )
    feats = _extract_trajectory_features(df)

    # Override launch_x/launch_y with the class-specific training medians.
    # The synthetic trajectory originates at (0, 0) which is not a real
    # rebel-base coordinate.  Individual launch sites are class-exclusive
    # (zero cross-class overlap), so geographic position is a near-deterministic
    # class proxy.  Using the global median would silently inject a Class 0
    # prior (global median ≈ Class 0 median because Class 0 is 69 % of data).
    feats["launch_x"] = _CLASS_LAUNCH_X[rebel_class]
    feats["launch_y"] = _CLASS_LAUNCH_Y[rebel_class]

    # Build the 32-feature vector; salvo/group keys missing → NaN, imputed later.
    vec = np.array([feats.get(k, np.nan) for k in SELECTED_FEATURES], dtype=np.float32)
    X = vec.reshape(1, -1)

    # Do not apply production threshold biases.  The +1.266 class-2 bias
    # corrects for the 7 % class-2 prevalence in the production population;
    # on a single synthetic trajectory it dominates and locks the prediction
    # to class 2.  Raw argmax(proba) lets kinematic features drive the result.
    proba = clf.predict_proba(X)
    pred = int(np.argmax(proba[0]))
    return pred, proba[0]


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
            "Adjust the parameters below to update the synthetic trajectory "
            "and watch the model classify it in real time."
        )
        st.markdown("---")
        st.markdown("### Rebel Group Context")
        rebel_class = st.selectbox(
            "Rebel Group",
            options=[0, 1, 2],
            format_func=lambda c: (
                f"Class {c}  —  {['69 %', '24 %', '7 %'][c]}  "
                f"(launch alt ~{_CLASS_INIT_ALT_M[c]:.0f} m, "
                f"speed {_CLASS_SPEED_HINT[c]} m/s, "
                f"angle {_CLASS_ANGLE_HINT[c]}°)"
            ),
            index=0,
            help=(
                "Sets the geographic launch-site context. Launch position is the "
                "top-2 SHAP feature and near-deterministic for class — each launch "
                "site fires only one rocket type. The global training median would "
                "silently inject a Class 0 prior (69 % of data)."
            ),
        )
        st.markdown("---")
        st.markdown("### Trajectory Parameters")

        initial_alt_m = st.slider(
            "Launch Altitude  (m)",
            min_value=0.0,
            max_value=80.0,
            value=float(_CLASS_INIT_ALT_M[rebel_class]),
            step=1.0,
            help=(
                "Terrain elevation of the launch site. "
                "initial_z is the single most important SHAP feature (rank 1, mean|SHAP| 2.01). "
                "Training medians: Class 0 ~12 m · Class 1 ~24 m · Class 2 ~39 m."
            ),
        )
        initial_speed = st.slider(
            "Initial Speed  (m/s)",
            min_value=5.0,
            max_value=200.0,
            value=55.0,
            step=1.0,
            help=(
                f"Launch velocity magnitude. Class {rebel_class} typical range: "
                f"{_CLASS_SPEED_HINT[rebel_class]} m/s."
            ),
        )
        thrust_accel = st.slider(
            "Thrust Acceleration  (m/s²)",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            step=5.0,
            help="Motor-burn peak acceleration over the first 15 % of flight.",
        )
        noise_sigma = st.slider(
            "Measurement Noise  (m)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            help="Radar position noise std-dev (~2 m matches typical training levels).",
        )
        launch_angle_deg = st.slider(
            "Launch Angle  (°)",
            min_value=5.0,
            max_value=80.0,
            value=45.0,
            step=1.0,
            help=(
                f"Elevation angle above horizontal. "
                f"Class {rebel_class} typical range: {_CLASS_ANGLE_HINT[rebel_class]}°. "
                "Steeper = higher apogee; shallower = longer range."
            ),
        )

        st.markdown("---")
        if clf is None:
            st.error(
                "**Model not found.**\n\nDownload the model first:\n```\nmake download-models\n```"
            )
        else:
            st.success("✓ Model loaded")
            st.caption(
                f"{len(feature_names)} features · "
                f"launch pos: Class {rebel_class} median · "
                "salvo/group imputed · biases not applied"
            )

    # ── Compute ────────────────────────────────────────────────────────────────
    # n=50 matches the training-data observation window (~2.5 s at 20 Hz).
    # The full training trajectory duration is 1.3-2.5 s (n_points p5-p95: 14-88).
    # Using n=100 (5 s) pushes apogee_relative, x_range, and final_z far OOD.
    pos, t = generate_trajectory(
        initial_speed,
        thrust_accel,
        noise_sigma,
        n=50,
        launch_angle_deg=launch_angle_deg,
        initial_alt_m=initial_alt_m,
    )

    if clf is not None:
        class_idx, proba = classify(clf, pos, t, rebel_class=rebel_class)
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
        "Select a **Rebel Group** context and adjust the trajectory parameters. "
        "The LightGBM model re-classifies on every change using the same "
        "kinematic feature pipeline as production. "
        "Launch position (SHAP rank 1-2) is set to the class-specific training median; "
        "salvo/group features are imputed from training medians. "
        "Production threshold biases are not applied — raw probabilities let "
        "kinematic changes produce visible class transitions."
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
