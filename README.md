# Rocket Trajectory Classifier

[![CI](https://github.com/IlyaNovikov-RD/rocket_classifier/actions/workflows/python-tests.yml/badge.svg)](https://github.com/IlyaNovikov-RD/rocket_classifier/actions/workflows/python-tests.yml)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit&logoColor=white)](https://novikov-rocket-lab.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)

> **[Live Demo — novikov-rocket-lab.streamlit.app](https://novikov-rocket-lab.streamlit.app/)**
> Adjust physics sliders and watch the XGBoost model classify trajectories in real time.

![Physics-Informed Feature Visualization](demo.png)

---

## What This Is

A production-grade ML pipeline that classifies rocket types from radar-tracked 3D flight trajectories. Given a variable-length sequence of `(x, y, z, time_stamp)` radar observations, the system extracts 76 physics-derived features and predicts the rocket class (0, 1, or 2).

### The Core Challenge: Worst-Class Recall Under Severe Imbalance

Standard accuracy is the wrong metric here. Class 2 comprises only **7.1% of training data** (2,339 of 32,741 trajectories), yet the evaluation metric is **minimum per-class recall** — the system is scored by whichever class it handles *worst*. A model that perfectly identifies classes 0 and 1 but misses every class 2 rocket scores **0.0**.

```math
\text{score} = \min_{j \in \{0,1,2\}} \frac{\sum_i \mathbf{1}[y_i = j \;\wedge\; \hat{y}_i = j]}{\sum_i \mathbf{1}[y_i = j]}
```

This metric demands that every design decision — feature engineering, class weighting, objective function, and post-hoc threshold tuning — be oriented toward equalising recall across all classes, deliberately sacrificing majority-class precision where necessary to protect minority-class recall.

**Result:** 5-fold GroupKFold cross-validation min-recall of **0.9984 ± 0.0007** — fewer than 1 in 400 rockets misclassified in the worst-performing class.

| Fold | Min-Recall (tuned) |
|------|--------------------|
| 1    | 0.9979             |
| 2    | 0.9981             |
| 3    | 0.9993             |
| 4    | 0.9975             |
| 5    | 0.9991             |
| **Mean** | **0.9984 ± 0.0007** |

---

## Solution Design

### Why XGBoost

The problem is **tabular classification on 76 hand-engineered features** — the exact regime where gradient-boosted trees dominate. XGBoost was chosen over alternatives for specific reasons:

- **Over neural networks:** With ~32k trajectories and 76 scalar features, a deep model would overfit without extensive regularisation. XGBoost's histogram binning and built-in L1/L2 regularisation handle this natively. There is no sequential/spatial structure in the aggregated features that a CNN or LSTM could exploit (the raw time series is already reduced to summary statistics).
- **Over Random Forest:** XGBoost's sequential boosting focuses capacity on hard-to-classify trajectories (the ~33 borderline cases between class 0 and 1), whereas Random Forest treats all samples equally. This matters when the metric penalises the worst class.
- **Over logistic regression / SVM:** The class boundaries in this feature space are non-linear (e.g., class 2 occupies a specific region of the launch_z × v_horiz plane). Tree-based models handle this without manual feature crosses.

### Pipeline Overview

```
Raw radar pings (x, y, z, t)
    │
    ▼
Feature Engineering ──► 76 physics features per trajectory
    │
    ▼
XGBoost (multi:softprob) ──► 3-class probability estimates
    │
    ▼
OOB Threshold Tuning ──► bias-adjusted predictions optimised for min-recall
    │
    ▼
submission.csv
```

The pipeline has three stages, each addressing a different challenge:

1. **Feature engineering** converts variable-length radar sequences into fixed-size vectors that encode the physics of each rocket type.
2. **XGBoost with softprob** learns the class boundaries and outputs calibrated probabilities rather than hard labels.
3. **Threshold tuning** post-processes these probabilities with per-class biases to directly optimise the min-recall metric, recovering borderline cases that raw argmax misclassifies.

---

## How It Works

### Feature Engineering

Raw radar pings are aggregated into **76 scalar features** per trajectory via finite-difference kinematics:

- **Velocity, Acceleration, Jerk** (45 features) — 3D derivatives with midpoint-averaged time deltas. Jerk magnitude distinguishes propelled rockets (sharp ignition spike) from passive objects.
- **Launch Angle** (2 features) — elevation and azimuth of the initial velocity vector via `atan2`. Invariant to launch position.
- **Apogee** (7 features) — peak altitude, relative rise, time fraction. Encodes the ballistic arc shape.
- **Spatial Extent** (9 features) — ranges, path length, horizontal displacement. Trajectory footprint.
- **Launch Position** (3 features) — geography clusters by threat group.
- **Temporal** (3 features) — point count, duration, sampling interval statistics.

### Why These Features Work

Different rocket families have different propellant charges (muzzle velocity), motor types (thrust profile / jerk), airframes (ballistic coefficient / velocity decay), and launch geometry. The 76 features capture exactly these physical quantities. The model achieves near-ceiling recall because the physics are deterministic — a rocket's class is written into its kinematics from the moment of launch.

### Model Configuration

The XGBoost classifier uses `multi:softprob` with 600 histogram-based trees at depth 6:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 600 | Enough capacity for 76 features; diminishing returns beyond this |
| `max_depth` | 6 | Deep enough for feature interactions, shallow enough to avoid overfitting 32k samples |
| `learning_rate` | 0.05 | Conservative shrinkage for stable convergence |
| `subsample` | 0.8 | Row sampling reduces variance on the minority class |
| `colsample_bytree` | 0.8 | Feature sampling forces diverse trees |
| `min_child_weight` | 5 | Prevents splits on tiny leaf groups (important for class 2 with only 2,339 samples) |
| `objective` | `multi:softprob` | Outputs calibrated probabilities for downstream threshold tuning |

### Class Imbalance Strategy

Inverse-frequency sample weights (`w_i = N / (K * N_j)`) passed to XGBoost. Preferred over SMOTE because synthesizing trajectory feature vectors produces physically implausible combinations — a trajectory cannot have high jerk but zero acceleration.

### Threshold Tuning

The model uses `multi:softprob` to output calibrated per-class probabilities instead of hard predictions. Per-class log-probability biases are then optimised on out-of-bag (OOB) predictions from cross-validation to maximise the min-recall metric. This shifts decision boundaries toward minority classes without retraining, improving CV min-recall from 0.9966 to 0.9984.

### Data Leakage Prevention

Three layers:
1. **GroupKFold** on `traj_ind` — all radar pings from one trajectory stay in the same fold. Mirrors deployment where the model sees entirely new flights.
2. **Per-fold NaN imputation** — column medians are computed from the training fold only. Validation fold data never leaks into imputation statistics.
3. **OOB threshold tuning** — biases are optimised on OOB predictions only (each sample's probability comes from a model that never saw it during training).

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Runtime** | Python 3.12 | PEP 709 comprehension inlining, improved error messages |
| **ML** | XGBoost 2.x, scikit-learn | Histogram-based gradient boosting, GroupKFold |
| **Validation** | Pydantic v2 | Schema enforcement on raw radar data before feature engineering |
| **Explainability** | SHAP TreeExplainer | Exact Shapley values in O(TLD) time |
| **Demo** | Streamlit, Plotly | Real-time 3D trajectory visualization |
| **Package management** | [uv](https://docs.astral.sh/uv/) | Deterministic lockfile, 10-100x faster than pip/poetry |
| **Container** | Docker (python:3.12-slim + uv) | Reproducible builds, no resolver in CI |
| **CI** | GitHub Actions | Ruff lint + 56 pytest tests on every push/PR |
| **Caching** | Parquet | Feature matrices cached to disk; reloads in <1s vs ~96s to recompute |

---

## Getting Started

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

No system Python required. uv manages its own Python 3.12 installation.

### Install & Run

```bash
# Clone and install
git clone https://github.com/IlyaNovikov-RD/rocket_classifier.git
cd rocket_classifier
uv sync

# Run the full training pipeline (outputs submission.csv + model.pkl)
uv run python -m rocket_classifier.main

# Launch the interactive demo (opens localhost:8501)
uv run streamlit run rocket_classifier/app.py
```

### Make Targets

```bash
make install   # uv sync
make test      # 56 unit tests
make lint      # ruff check
make format    # ruff format
make demo      # streamlit demo
make lock      # regenerate uv.lock
```

### Docker

```bash
docker build -t rocket-classifier .
docker run rocket-classifier
```

The Dockerfile uses `COPY --from=ghcr.io/astral-sh/uv:latest` for a single-binary uv install, then `uv sync --frozen --no-dev` for a reproducible build with no dependency resolution at build time.

---

## Project Structure

```
rocket_classifier/          # Installable Python package
├── __init__.py
├── features.py             # 76 physics-derived features (velocity, jerk, apogee, ...)
├── model.py                # XGBoost + GroupKFold CV + per-fold imputation + threshold tuning
├── schema.py               # Pydantic v2 data contracts (TrajectoryPoint)
├── main.py                 # Pipeline orchestrator: load → validate → train → predict
├── app.py                  # Streamlit interactive demo
├── interpret.py            # SHAP explainability (shap_summary.png)
└── visualize.py            # Physics feature visualization (demo.png)

tests/
└── test_features.py        # 56 unit tests (derivatives, edge cases, key completeness)

data/
├── train.csv               # Labeled radar trajectories
├── test.csv                # Unlabeled trajectories for inference
└── sample_submission.csv   # Expected output format

pyproject.toml              # PEP 621 metadata, uv/hatchling build
uv.lock                     # Deterministic dependency lockfile
Dockerfile                  # Python 3.12-slim + uv
Makefile                    # Developer automation
ruff.toml                   # Linter/formatter config (target: py312)
```

---

## Model Interpretability

![SHAP Feature Importance](shap_summary.png)

`rocket_classifier/interpret.py` computes exact SHAP values via `TreeExplainer` on a 500-trajectory test sample. Top discriminators:

- **Launch position** (`launch_x`, `launch_z`) — geography clusters by threat group
- **Horizontal speed** (`v_horiz_median`) — muzzle velocity is propellant-charge dependent
- **Kinematic derivatives** (`acc_horiz_min`, `vz_mean`) — propulsion physics separate the classes
- **Apogee features** (`apogee_relative`, `apogee_time_frac`) — ballistic arc shape differs between rocket families

```bash
uv run python -m rocket_classifier.interpret   # regenerates shap_summary.png
```

---

## Assumptions

1. **Flat terrain** — `z` is absolute altitude; no terrain correction.
2. **Frictionless ballistic physics** — features are physically meaningful under point-mass assumption.
3. **One label per trajectory** — all pings in a `traj_ind` share the same class.
4. **Trajectory independence** — each flight is independent; no cross-trajectory temporal features.
