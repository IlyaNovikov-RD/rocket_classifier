# Rocket Trajectory Classifier

[![CI](https://github.com/IlyaNovikov-RD/rocket_classifier/actions/workflows/python-tests.yml/badge.svg)](https://github.com/IlyaNovikov-RD/rocket_classifier/actions/workflows/python-tests.yml)
[![Streamlit — App: Online](https://img.shields.io/badge/Streamlit-App%3A%20Online-FF4B4B?logo=streamlit&logoColor=white)](https://ilya-rocket-classifier.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)

> ### 🚀 [Live Demo — ilya-rocket-classifier.streamlit.app](https://ilya-rocket-classifier.streamlit.app/)
> Adjust physics sliders and watch the XGBoost model classify trajectories in real time.

![Physics-Informed Feature Visualization](demo.png)

---

## Overview

Rocket Trajectory Classifier is a production-grade machine learning pipeline that classifies airborne objects from radar-tracked 3D flight data. Given a sequence of positional readings `(x, y, z, time_stamp)` belonging to a single trajectory, the system predicts the rocket class (0, 1, or 2) with a **CV min-recall of 0.9949 ± 0.0022** — meaning it misses fewer than 1 in 200 threats across every class simultaneously.

**The core insight** is that different rocket families leave distinct physics fingerprints in kinematic derivatives. Rather than learning raw coordinates, the pipeline computes 76 physics-derived features per trajectory — velocity, acceleration, jerk, launch angle, apogee, and spatial extent — all grounded in frictionless ballistic mechanics. These features are invariant to launch position and time, making the model robust to deployment at unseen sites.

The metric is **minimum per-class recall** (`min_j recall_j`): a model that perfectly classifies two classes but misses every instance of the third scores **0.0**. This forces the classifier to be simultaneously competent across all threat types.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML** | XGBoost 2.x · scikit-learn (GroupKFold, metrics) |
| **Explainability** | SHAP (TreeExplainer) |
| **Data validation** | Pydantic v2 |
| **Web demo** | Streamlit · Plotly |
| **Packaging** | Poetry · Parquet (feature caching) |
| **Containerisation** | Docker |
| **CI/CD** | GitHub Actions |
| **Automation** | GNU Make |

---

## Engineering Highlights

### CI/CD Pipeline
Every push to `main` and every Pull Request triggers the GitHub Actions workflow (`.github/workflows/python-tests.yml`): ruff lints the entire codebase, then all 55 pytest unit tests run on `ubuntu-latest`. The `pull_request` trigger means no PR can be merged with a broken build.

### Physics-Based Feature Engineering
Raw radar pings are aggregated into 76 scalar features per trajectory using finite-difference kinematics (velocity → acceleration → jerk), `atan2`-based launch angles, apogee detection, and spatial extent metrics. All computations are fully vectorised with NumPy. The feature pipeline is hardened against edge cases — single-point trajectories, duplicate timestamps (`dt = 0`), zero-velocity states — all covered by the 55-test suite.

### Data Leakage Prevention
`GroupKFold` on `traj_ind` guarantees that every radar ping from a given trajectory appears exclusively in training or validation within each fold — never both. This mirrors real-world deployment where the model encounters entirely unseen flights.

### SHAP Explainability
`src/interpret.py` runs `TreeExplainer` on a representative test sample and produces `shap_summary.png` — a per-class SHAP bar chart showing which features drive each classification decision. Top drivers: launch position geography (`launch_x`, `launch_z`) and muzzle velocity (`v_horiz_median`), consistent with independent rocket supply chains per threat group.

### Artifact Retrieval from GitHub Releases
`model.pkl` is published as a binary asset on the [v1.0.0 GitHub Release](https://github.com/IlyaNovikov-RD/rocket_classifier/releases/tag/v1.0.0). The Streamlit demo automatically downloads and caches it at startup when no local file is present — making the app deployable to any cloud environment (Streamlit Community Cloud, Docker, bare VM) without pre-running the training pipeline.

### Dynamic Artifact Retrieval
To maintain a lightweight repository and follow MLOps best practices, the XGBoost model binary is NOT tracked in Git. Instead, the Streamlit application features a dynamic retrieval layer that fetches the `model.pkl` directly from GitHub Release assets (v1.0.0) upon startup. This ensures a clean separation between code and model artifacts while enabling seamless one-click deployment to any environment.

### Pydantic Data Contracts
`src/schema.py` defines `TrajectoryPoint` (Pydantic v2 `BaseModel`) with enforced constraints: `z ≥ 0` (altitude cannot be negative), `label ∈ {0, 1, 2}`, `extra = "forbid"` to catch schema drift immediately. `validate_dataframe()` bulk-validates any raw DataFrame and returns `(valid_records, errors)` for graceful handling of bad rows.

---

## How to Run

**Prerequisites:** Python 3.11, [Poetry](https://python-poetry.org/docs/#installation), `make`

```bash
# 1. Install all dependencies
make install

# 2. Run the full training pipeline
#    Generates submission.csv and model.pkl
poetry run python src/main.py

# 3. Launch the interactive Streamlit demo
make demo
```

All developer workflows in one place:

```bash
make install   # poetry install
make test      # pytest -v  (55 unit tests)
make lint      # ruff check .
make format    # ruff format .
make demo      # streamlit run src/app.py → localhost:8501
```

> **Windows users:** `make` is available via [Git for Windows](https://gitforwindows.org/),
> `choco install make`, or `winget install GnuWin32.Make`.

---

## Project Structure

```
rocket_classifier/
├── data/
│   ├── train.csv                  # Labeled trajectory point data
│   ├── test.csv                   # Unlabeled trajectory point data
│   └── sample_submission.csv      # Expected submission format
├── src/
│   ├── schema.py                  # Pydantic data contracts (TrajectoryPoint)
│   ├── features.py                # Physics-based feature engineering (76 features)
│   ├── model.py                   # XGBoost classifier + GroupKFold CV
│   ├── main.py                    # Pipeline orchestrator (train → submission)
│   ├── app.py                     # Streamlit interactive demo
│   ├── interpret.py               # SHAP model interpretability
│   └── visualize.py               # Physics feature visualization (demo.png)
├── tests/
│   └── test_features.py           # 55 unit tests for feature engineering
├── .github/workflows/
│   └── python-tests.yml           # CI: ruff + pytest on push and PR
├── Makefile                       # Developer automation
├── Dockerfile                     # Containerized pipeline runtime
├── pyproject.toml                 # Poetry dependency manifest
└── poetry.lock                    # Pinned dependency versions
```

---

## Evaluation Metric

The official metric is **minimum per-class recall** across all rocket types:

```math
\text{score} = \min_{j \in \{0,1,2\}} \frac{\sum_i \mathbf{1}[y_i = j \wedge \hat{y}_i = j]}{\sum_i \mathbf{1}[y_i = j]}
```

This penalizes models that perform well on average but fail on any single class. A model that perfectly classifies classes 0 and 1 but misses every class 2 rocket scores **0.0**.

### Cross-Validation Results

| Fold | Min-Recall | Class 0 | Class 1 | Class 2 |
|------|-----------|---------|---------|---------|
| 1    | 0.9915    | 1.000   | 0.998   | 0.992   |
| 2    | 0.9978    | 0.999   | 0.999   | 0.998   |
| 3    | 0.9958    | 0.999   | 1.000   | 0.996   |
| 4    | 0.9958    | 0.999   | 0.998   | 0.996   |
| 5    | 0.9934    | 1.000   | 0.998   | 0.993   |
| **CV** | **0.9949 ± 0.0022** | | | |

---

## Model Interpretability

![SHAP Feature Importance](shap_summary.png)

`src/interpret.py` runs `TreeExplainer` on a 500-trajectory sample from the test set and produces the stacked bar chart above. Key findings:

- **Launch position** (`launch_x`, `launch_z`) and **horizontal speed** (`v_horiz_median`) are the dominant discriminators — launcher geography and muzzle velocity encode which group fired the weapon.
- **Kinematic derivatives** (`acc_horiz_min`, `vz_mean`, `speed_median`) rank in the top 10, confirming that propulsion physics — not just trajectory shape — separate the classes.
- **Apogee features** (`apogee_relative`, `apogee_time_frac`) appear mid-table, consistent with ballistic arc differences between rocket families.

```bash
poetry run python src/interpret.py   # regenerates shap_summary.png
```

---

## Key Architectural Decisions

### 1. Vectorized Physics Feature Engineering (`src/features.py`)

76 scalar features per trajectory, fully vectorised with NumPy:

- **3D Velocity, Acceleration, Jerk** — finite differences on `(x, y, z)` with midpoint-averaged time deltas to minimise discretisation error.
- **Launch Angle** — elevation `atan2(vz, v_horiz)` and azimuth `atan2(vy, vx)` of the initial velocity vector.
- **Apogee and Time-to-Apogee** — maximum altitude and the fractional trajectory position at which it occurs.
- **Horizontal Range, Path Length, Spatial Extent** — trajectory footprint and kinetic energy proxies.

### 2. Data Leakage Prevention (`src/model.py`)

`GroupKFold(traj_ind)` ensures all radar pings of a given trajectory are never split across train and validation in the same fold — mirroring real-world deployment where the model always sees fully new flights.

### 3. Imbalanced Data Handling (`src/model.py`)

Training set skew: class 0 (69%), class 1 (24%), class 2 (7%). Solution: inverse-frequency sample weights passed directly to `XGBClassifier.fit()`:

$$w_i = \frac{N}{K \cdot N_j} \quad \text{where } j = \text{class of sample } i$$

Preferred over SMOTE because synthesising trajectory-level feature vectors can produce physically implausible combinations.

### 4. Parquet Feature Caching (`src/main.py`)

Feature engineering over 32,741 training trajectories takes ~96 s on first run. Results are cached to `cache_train_features.parquet` and `cache_test_features.parquet`. Subsequent runs skip directly to training, reducing total runtime from ~6 min to ~4 min. Parquet preserves exact float64 precision and columnar compression; CSV would not.

---

## Assumptions

1. **Flat terrain** — `z` is absolute altitude; no terrain correction required.
2. **Standard ballistic physics** — frictionless point-mass trajectories; derived features are physically meaningful.
3. **One label per trajectory** — all pings share the same `label`; taken from the first row of each group.
4. **Trajectory independence** — each `traj_ind` is a fully independent flight; no cross-trajectory temporal features.
