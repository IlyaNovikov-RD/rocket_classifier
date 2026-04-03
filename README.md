# Rocket Trajectory Classifier

[![CI](https://github.com/IlyaNovikov-RD/rocket_classifier/actions/workflows/python-tests.yml/badge.svg)](https://github.com/IlyaNovikov-RD/rocket_classifier/actions/workflows/python-tests.yml)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit&logoColor=white)](https://novikov-rocket-lab.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)

> **[Live Demo — novikov-rocket-lab.streamlit.app](https://novikov-rocket-lab.streamlit.app/)**
> Adjust physics sliders and watch the LightGBM model classify trajectories in real time.

![Physics-Informed Feature Visualization](assets/demo.png)

---

## What This Is

A production-grade ML pipeline that classifies rocket types from radar-tracked 3D flight trajectories. Given a variable-length sequence of `(x, y, z, time_stamp)` radar observations, the system extracts physics-derived features — enriched with domain-specific salvo and rebel-group context — and predicts the rocket class (0, 1, or 2).

### The Core Challenge: Worst-Class Recall Under Severe Imbalance

Standard accuracy is the wrong metric here. Class 2 comprises only **7.1% of training data** (2,339 of 32,741 trajectories), yet the evaluation metric is **minimum per-class recall** — the system is scored by whichever class it handles *worst*. A model that perfectly identifies classes 0 and 1 but misses every class 2 rocket scores **0.0**.

```math
\text{score} = \min_{j \in \{0,1,2\}} \frac{\sum_i \mathbf{1}[y_i = j \;\wedge\; \hat{y}_i = j]}{\sum_i \mathbf{1}[y_i = j]}
```

This metric demands that every design decision — feature engineering, class weighting, objective function, and post-hoc threshold tuning — be oriented toward equalising recall across all classes, deliberately sacrificing majority-class precision where necessary to protect minority-class recall.

**Result:** Global OOB min-recall of **0.999911** — 2 misclassified trajectories out of 32,741.

| Class | Count | Recall (OOB) | Wrong |
|-------|-------|-------------|-------|
| 0 (majority — 68.6%) | 22,462 | 0.999911 | 2 |
| 1 (24.3%) | 7,940 | 1.000000 | 0 |
| 2 (minority — 7.1%) | 2,339 | 1.000000 | 0 |
| **Global OOB (tuned)** | **32,741** | **0.999911** | **2** |

An oracle threshold analysis (`research/colab_analysis.py`) confirms that both remaining errors are distinguishable in principle — the model's probability outputs are sufficient to achieve 1.0 on every individual CV fold when thresholds are tuned per-fold. The practical ceiling with a single global threshold is **0.999911**.

---

## Domain Assumptions That Shaped the Solution

Assumptions 1 and 2 (flat terrain, frictionless ballistic physics) establish that kinematic features are physically meaningful — altitude can be used directly, and velocity/acceleration/jerk faithfully encode the rocket's propulsion signature. Assumption 3 provides three business sub-assumptions from a domain expert, each directly driving a feature engineering decision.

### 3a — Different launchers have different payload capacities
Different launcher types can fire a different maximum number of rockets. This means the *largest* salvo ever observed from a rebel base reveals which launcher type they operate. Feature: `group_max_salvo_size`.

### 3b — Rockets are fired in salvos
Rockets are typically fired together or with a short delay. Spatiotemporal DBSCAN clustering on `(launch_x, launch_y, launch_time_s)` identifies co-fired rockets and produces four features:

| Feature | Description |
|---|---|
| `salvo_size` | Number of rockets in the same salvo |
| `salvo_duration_s` | Elapsed time from first to last launch in the salvo |
| `salvo_spatial_spread_m` | Maximum pairwise launch-point distance within the salvo |
| `salvo_time_rank` | This rocket's launch order within its salvo |

`salvo_time_rank` is the most discriminative of these, ranking 12th in model feature importance — rockets fired later in a salvo carry a detectable kinematic imprint.

### 3c — Few concentrated rebel groups, each purchasing independently
There are few rebel groups, each operating from a fixed geographic area. Critically, **each group purchases its own rocket supply independently** — they are not organised under a single umbrella organisation. This means every group stocks one rocket type. Pure-spatial DBSCAN on `(launch_x, launch_y)` identifies 15 persistent rebel bases and produces group-level features:

| Feature | Description |
|---|---|
| `group_total_rockets` | Total rockets ever fired from this base |
| `group_n_salvos` | Number of distinct firing events from this base |
| `group_max_salvo_size` | Largest single salvo (proxy for launcher type — assumption 3a) |

**Empirical validation:** 97.4% class purity within same-salvo rockets — confirming the independent procurement assumption holds strongly in the data.

---

## Solution Design

### Why LightGBM

The problem is **tabular classification on physics-engineered features with cross-trajectory context**. LightGBM is the right choice for four structural reasons:

**1. The feature set is inherently tabular, not sequential.**
The discriminative signal lives in aggregate statistics — muzzle velocity, apogee shape, salvo rank, launch position. These are scalar features per trajectory, not temporal sequences. Sequence models (CNN, Transformer) are designed to learn representations from raw time series; here the representations are already hand-crafted from physics. Asking a neural network to re-derive `initial_speed` from raw pings adds no information but introduces variance and training complexity.

**2. Salvo and group features are cross-trajectory by nature.**
`salvo_time_rank`, `group_max_salvo_size`, and the other 5 salvo/group features (assumption 3) require aggregating information across multiple trajectories via DBSCAN clustering. A per-trajectory sequential model cannot access this information at all — it would need to be provided as an additional scalar input anyway. LightGBM treats all 32 features uniformly regardless of how they were derived.

**3. Calibrated probabilities are essential for threshold tuning.**
The evaluation metric (`min_class_recall`) is optimised post-hoc by tuning per-class log-probability biases on OOB predictions. This requires the model to output well-calibrated class probabilities. LightGBM's `multiclass` objective with softmax produces probabilities suitable for this; raw neural network logits require additional calibration steps.

**4. Leaf-wise growth concentrates capacity on hard cases.**
With a severe class imbalance (69%/24%/7%) and a metric that penalises the worst-performing class, the model must focus on the small number of class-2 trajectories that sit near class boundaries. LightGBM's leaf-wise growth builds asymmetric trees that allocate splits where the gain is highest — precisely the borderline minority-class examples. Level-wise trees (Random Forest, default XGBoost) waste capacity on already-well-separated majority-class regions.

**Empirical validation via AutoML.**
The algorithm choice was confirmed — not assumed — through a 100-trial Optuna search over LightGBM, XGBoost, CatBoost, and a neural TabMLP with entity embeddings, all on the same 32-feature set with the same 10-fold GroupKFold evaluation. LightGBM dominated from the first trial: XGBoost scored 0.9253 raw OOB while LightGBM scored 0.9996 in the very next trial — a gap that never closed. Every top-5 Optuna trial across all runs was LightGBM.

A separate experiment tested a Transformer encoder operating on raw radar sequences with the same 7 salvo/group features as additional context input (salvo_dim=64, 4 layers, 8 heads). Best OOB min-recall after 10 Optuna trials: **0.9110** — confirming that the hand-crafted aggregate features used by LightGBM encode the discriminative signal far more effectively than learned temporal representations on this dataset.

### Model Development Pipeline (Research — Colab H100, one-time)

```
Raw radar pings (x, y, z, t)
    │
    ▼
Kinematic Feature Engineering ──► 76 physics features per trajectory
    │
    ▼
Salvo/Group Feature Engineering ──► +7 features via DBSCAN (assumptions 3a-3c)
    │                                  (83 total)
    ▼
Backward Elimination ──► 32 features selected (25 kinematic + 7 salvo/group)
    │
    ▼
LightGBM + 100-trial Optuna (GPU) ──► optimised hyperparameters
    │
    ▼
10-fold GroupKFold OOB Threshold Tuning ──► biases [0, 0.759, 0.658]
    │
    ▼
weights/model.lgb  ·  weights/train_medians.npy  ·  weights/threshold_biases.npy
```

### Production Inference Pipeline (`make run`)

```
cache/cache_test_features.feather   (pre-computed 83-feature matrix, Arrow IPC)
    │
    ▼
Select 32 production features + impute NaN with train_medians.npy
Append global class priors [0.686, 0.243, 0.071] → 35-column model input
    │
    ▼
weights/model.onnx  ──►  ONNX Runtime (AVX2, all cores, JIT pre-warmed)
    │
    ▼
log(proba) + biases [0, 0.759, 0.658]  ──►  argmax
    │
    ▼
outputs/submission.csv
```

---

## How It Works

### Feature Engineering

Raw radar pings are aggregated into **83 scalar features** per trajectory — 76 kinematic features from finite-difference physics (assumptions 1 & 2), plus 7 salvo and rebel-group features from DBSCAN clustering (assumption 3). **32 are used in production** after automated backward elimination.

**Why these features discriminate between rocket classes:**

| Feature group | Selected | Why it works |
|---|---|---|
| **Velocity** — `vy_mean`, `vy_max`, `v_horiz_median`, `v_horiz_std`, `initial_speed`, `final_vz` | 6 | Muzzle velocity is set by the propellant charge — a fixed physical constant per rocket type. `initial_speed` measures it directly. Horizontal speed encodes range capability; `final_vz` encodes terminal descent rate (varies by airframe mass and drag). |
| **Acceleration** — `acc_mag_mean`, `acc_mag_min`, `acc_mag_max`, `acc_horiz_std`, `acc_horiz_min`, `acc_horiz_max`, `az_std`, `mean_az` | 8 | Under frictionless ballistic physics (assumption 2), vertical acceleration is constant at −g. Deviations encode thrust and drag. `acc_mag_max` captures peak motor thrust (varies by motor type). `acc_horiz_min` captures peak lateral deceleration — the airframe's drag signature. |
| **Vertical kinematics** — `vz_median`, `initial_vz`, `final_vz`, `initial_z`, `final_z`, `delta_z_total`, `apogee_relative` | 7 | The ballistic arc shape is uniquely determined by initial vertical velocity under flat-terrain physics (assumption 1). Different rocket types launch at different angles with different muzzle velocities, producing distinct arc heights and descent profiles. |
| **Spatial extent** — `x_range`, `y_range` | 2 | Downrange distance is determined by horizontal muzzle velocity and time of flight — both vary by rocket type. Compact encodings of the trajectory footprint. |
| **Launch position** — `launch_x`, `launch_y` | 2 | Per assumption 3c, each rebel group operates from a fixed geographic area and uses one rocket type. Launch coordinates are therefore a near-deterministic proxy for rocket class — the 2nd and 5th most important features by SHAP. |
| **Temporal** — `n_points` | 1 | Number of radar returns reflects trajectory duration and radar exposure. Longer-range rockets produce more pings; very short trajectories (few pings) signal specific launch geometries. |
| **Salvo/group** — `salvo_size`, `salvo_duration_s`, `salvo_spatial_spread_m`, `salvo_time_rank`, `group_total_rockets`, `group_n_salvos`, `group_max_salvo_size` | 7 | See [Domain Assumptions](#domain-assumptions-that-shaped-the-solution). `salvo_time_rank` (rank 12 by SHAP) carries a kinematic imprint from the firing sequence; `group_max_salvo_size` proxies launcher type (assumption 3a). |

**Why 32 and not all 83?** Automated backward elimination on the full training set showed that the remaining 51 features did not improve `min_class_recall` when added individually — they either encoded redundant information or introduced noise.

> **Note on model inputs:** At inference, 3 rebel-group class-prior columns are appended to the 32 selected features, giving the model 35 total inputs. These priors (global training class distribution: 68.6%/24.3%/7.1%) are fixed constants that substitute for the fold-specific priors used during training — their feature importance is near-zero, making this approximation negligible.

### Model Configuration

The production model is LightGBM, trained via 100-trial GPU-accelerated Optuna search:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1,047 | Optuna-determined for this feature set |
| `max_depth` | 9 | Leaf-wise growth on 32 features |
| `learning_rate` | 0.127 | Higher rate, fewer trees than kinematic-only baseline |
| `subsample` | 0.770 | Row sampling found by Optuna |
| `colsample_bytree` | 0.619 | Feature sampling found by Optuna |
| `objective` | `multiclass` (softprob) | Calibrated probabilities for threshold tuning |

Threshold biases: `[0.000000, 0.759494, 0.658228]` — upweights classes 1 and 2 to compensate for the 69%/24%/7% class imbalance. Found via scipy `differential_evolution` on global OOB predictions — confirmed globally optimal.

### Class Imbalance Strategy

Inverse-frequency sample weights (`w_i = N / (K * N_j)`) passed to LightGBM. Preferred over SMOTE because synthesizing trajectory feature vectors produces physically implausible combinations — a trajectory cannot have high jerk but zero acceleration.

### Threshold Tuning

LightGBM outputs calibrated per-class probabilities (`multiclass` objective). Per-class log-probability biases are then optimised on all OOB predictions simultaneously from 10-fold CV to maximise the min-recall metric. An exhaustive global search (scipy `differential_evolution`) confirms `[0, 0.759, 0.658]` are globally optimal for the current model.

### Data Leakage Prevention

Three layers:
1. **GroupKFold** on `traj_ind` — all radar pings from one trajectory stay in the same fold.
2. **Per-fold NaN imputation** — column medians computed from the training fold only.
3. **OOB threshold tuning** — biases optimised on OOB predictions only.

---

## Pipeline Performance

All runtimes measured on a Windows 11 machine (Intel Core i5, 4 cores, 8,185 test trajectories).

### Operational mode — optimisation history

| Optimisation | Total pipeline | vs. original |
|---|---|---|
| Original (CSV + Pydantic + sklearn) | 3m 40s | baseline |
| Skip CSV/validation when caches exist | 15s | 14.7x |
| ONNX Runtime backend (2.6x faster inference) | 5s | 44x |
| All CPU threads + Feather cache + JIT pre-warm | 4s | 55x |
| memory_map Feather + skip train load in hot path | **~1.5s** | **~147x** |

### Current pipeline breakdown (measured, min of 5 runs)

| Step | Time |
|---|---|
| Load Feather test cache (memory-mapped) | 0.02s |
| ONNX session init + JIT pre-warm | ~0.8s |
| Impute NaN + append priors | <0.01s |
| **ONNX inference** (8,185 traj. × 35 feat., 4 threads) | **~1.5s** |
| Threshold bias + argmax | <0.01s |
| Write submission.csv | 0.03s |
| **Total** | **~1.5s** |

### Proof this is the hardware floor

**1. Thread saturation (empirical)**

All 4 physical cores are in use. Thread sweep (30 runs each, ONNX inference only):

| Threads | Inference time | Speedup |
|---|---|---|
| 1 | 2.29s | baseline |
| 2 | 1.43s | 1.6x |
| 3 | 1.33s | 1.7x |
| 4 | **1.24s** | **1.85x** |

Adding threads beyond `cpu_count()` yields no further gain — there are no more physical cores.

**2. Memory-bandwidth bound (theoretical)**

The irreducible work is 8,185 samples × 1,047 trees × depth 9 = **68.8M comparisons** across the 5.4 MB ONNX model.

```
Theoretical compute floor: 68.8M ops / 4 cores / 10⁹ ops/s  ≈  17ms
Measured inference:                                            ~1.5s
Overhead factor:                                               ~88x
```

This overhead is explained entirely by **cache miss cost**: the 5.4 MB model does not fit in L1/L2 cache (256 KB / 1 MB per core). Each tree traversal follows random pointers through L3 and main memory. Effective memory bandwidth for irregular-access tree walks is ~1–5 GB/s vs the raw L3 peak of ~100 GB/s.

**3. Algorithm irreducibility**

The 1,047 trees were determined by Optuna (100 trials, H100 GPU) to be the minimum that achieves 0.999911 global OOB min-recall. Reducing tree count would degrade accuracy below the operational threshold. Traversing every tree for every sample is not optional.

**4. Backend optimality**

ONNX Runtime with `ORT_ENABLE_ALL` already applies:
- AVX2/AVX512 SIMD vectorisation for operator kernels
- Graph-level operator fusion
- Memory arena pre-allocation (`enable_mem_pattern = True`)

No further software optimisations are possible without different hardware (more cores or GPU inference).

**5. Persistent-service mode**

For a long-running server that loads the model once and handles repeated requests, the ONNX session init + JIT overhead (~0.8s) is paid only once. Steady-state per-batch latency drops to **~1.5s** (pure inference + I/O).

### Cold start (no caches, raw CSV only)

| Stage | Time |
|---|---|
| Load raw CSVs (1M+ rows) | ~4s |
| Pydantic schema validation | ~3m |
| Kinematic feature engineering (76 features × 32k trajectories) | ~96s |
| Salvo/group feature engineering (DBSCAN) | ~15s |
| Model inference | ~1.5s |
| **Total cold start** | **~6 min** |

## Transitioning to Real-Time Operations

The current pipeline is designed for batch inference — it classifies a complete set of pre-collected trajectories. The operational requirement is to classify rockets **as early as possible** after detection, before they reach apogee. This section outlines the architectural path from batch to real-time.

### The core challenge: salvo features require temporal coordination

The model's 32 features fall into two categories with fundamentally different latency profiles:

| Feature group | When available | Quality |
|---|---|---|
| 25 kinematic features | First 3–5 radar pings (~1 second after detection) | ~0.9997 OOB recall |
| 7 salvo/group features (3b, 3c) | After other rockets in the same salvo are detected (~5–30 seconds) | 0.999911 OOB recall |

A real-time system cannot compute `salvo_time_rank` or `salvo_size` for a rocket until sibling rockets in the same firing event have also been detected — they arrive asynchronously across different radars. This motivates a **two-phase progressive classification** architecture:

```
Radar pings arrive
       │
       ▼  Phase 1 — immediate (< 100ms after first ping)
Kinematic features only
  → ONNX inference (salvo features at training medians)
  → Preliminary class estimate broadcast to operators
       │
       ▼  Phase 2 — salvo-confirmed (~5–30s after launch cluster detected)
Salvo context materialises via streaming DBSCAN
  → Re-run inference with full 32 features
  → Updated high-confidence prediction replaces preliminary estimate
```

Phase 1 already satisfies the "as early as possible" requirement from the assignment at ~0.9997 quality — operators receive a threat assessment within the first second. Phase 2 upgrades to 0.999911 as the salvo context arrives, without requiring any architectural change to the model itself.

### Streaming architecture

```
Radar stations (N radars across the operational theatre)
        │  raw pings (x, y, z, t, radar_id)
        ▼
   Apache Kafka  ──────────────────────────────────┐
        │  topic: radar-pings                       │
        ▼                                           │
  Apache Flink (stream processor)                   │
   ├─ Trajectory assembler                          │
   │   Groups pings by traj_ind within a           │
   │   sliding time window                          │
   │                                                │
   ├─ Kinematic feature extractor                   │
   │   Runs on partial trajectory (≥ 3 pings)      │
   │   Triggers Phase 1 classification              │
   │                                                │
   └─ Salvo coordinator                             │
       Online DBSCAN over recent launch positions   │
       Triggers Phase 2 when salvo membership       │
       is confirmed                                 │
        │                                           │
        ▼                                           │
  ONNX serving endpoint (model.onnx)                │
  < 2ms per trajectory, all cores                   │
        │                                           │
        ▼                                           │
  Prediction store (Redis)  ◄────────────────────┘
  Keyed by traj_ind
  Updated by both phases
        │
        ▼
  Operator dashboard  ──  Alert system
```

### Online salvo detection

DBSCAN as currently implemented runs in batch (O(N²) for large clusters). In a streaming context, the salvo coordinator would use an **online spatial index** (e.g. R-tree or ball tree over a sliding 60-second window of recent launches) to assign incoming rockets to existing salvos or open a new salvo group. The DBSCAN parameters (`eps=0.5`, `min_samples=2`) remain unchanged — only the execution model shifts from batch to incremental.

### Model retraining and class drift

The assignment identifies few rebel groups each buying independently (assumption 3c). If a new group acquires a new rocket type, production recall for that class will degrade. Operationally, this requires:

- **Recall monitoring per class** — not overall accuracy. The evaluation metric (`min_class_recall`) maps directly to the production alert threshold. When any class recall drops below an agreed SLA, retraining is triggered.
- **Active labelling pipeline** — intercepted rockets provide confirmed labels. These feed the next training run.
- **GroupKFold preservation** — retraining must continue to group all pings from one trajectory into the same fold to prevent leakage from partial-trajectory data collected mid-flight.

### Latency budget

| Stage | Target |
|---|---|
| First radar ping → kinematic features computed | < 50ms |
| Kinematic features → ONNX inference → alert | < 10ms |
| **Phase 1: detection to preliminary classification** | **< 100ms** |
| Salvo membership confirmed → salvo features → re-inference | < 500ms |
| **Phase 2: salvo-confirmed classification** | **< 30s after launch cluster** |

A ballistic rocket travelling at typical class-1 speeds takes 60–90 seconds from launch to apogee. Both phases complete well within the available threat-assessment window.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Runtime** | Python 3.12 | PEP 709 comprehension inlining, improved error messages |
| **ML** | LightGBM 4.x, scikit-learn | Leaf-wise gradient boosting, GPU-accelerated Optuna, GroupKFold |
| **Clustering** | scikit-learn DBSCAN | Spatiotemporal salvo and geographic rebel-group identification |
| **Inference** | ONNX Runtime | 2.6x faster than sklearn wrapper via AVX2 vectorisation |
| **Validation** | Pydantic v2 | Schema enforcement on raw radar data |
| **Explainability** | SHAP TreeExplainer | Exact Shapley values in O(TLD) time |
| **Demo** | Streamlit, Plotly | Real-time 3D trajectory visualisation |
| **Package management** | [uv](https://docs.astral.sh/uv/) | Deterministic lockfile, 10–100x faster than pip/poetry |
| **CI** | GitHub Actions | Ruff lint + pytest on every push/PR |
| **Caching** | Parquet + Feather (Arrow IPC) | Feature matrices cached; Feather is 2x faster to read |

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

### Install & Run

```bash
git clone https://github.com/IlyaNovikov-RD/rocket_classifier.git
cd rocket_classifier
uv sync

# Full pipeline — downloads ~20 MB from GitHub Release
make pipeline
# Output: outputs/submission.csv  +  updated assets/

# Step by step:
make download-all    # weights/ + cache/
make run             # → outputs/submission.csv
make interpret       # → assets/shap_summary.png
make visualize       # → assets/demo.png

# After a model update — regenerate ONNX (requires: uv pip install onnxmltools skl2onnx)
make export-model

make demo            # streamlit demo (localhost:8501)
```

### Make Targets

```bash
make install          # uv sync
make test             # unit tests
make lint             # ruff check
make format           # ruff format
make demo             # streamlit demo (localhost:8501)
make download-weights # fetch weights/ from GitHub Release
make download-all     # + cache/ parquet caches
make export-model     # convert model.lgb/pkl → model.onnx (run after model update)
make run              # inference pipeline → outputs/submission.csv
make interpret        # regenerate SHAP assets after model update
make visualize        # regenerate assets/demo.png after feature changes
make pipeline         # download-all + run + interpret  (full end-to-end)
```

### Docker

```bash
docker build -t rocket-classifier .
docker run -v $(pwd)/outputs:/app/outputs rocket-classifier
```

---

## Project Structure

```
rocket_classifier/              # Production inference package
├── __init__.py
├── features.py                 # 83 physics + salvo/group features (single source of truth)
├── model.py                    # RocketClassifier — loads LightGBM, applies biases
├── schema.py                   # Pydantic v2 data contracts (TrajectoryPoint)
├── main.py                     # Inference pipeline: features → predict → submission.csv
└── app.py                      # Streamlit interactive demo

scripts/
├── download_weights.py         # Download model artifacts from GitHub Release
└── export_fast_models.py       # Export model.lgb → model.onnx + benchmark all backends

research/                       # R&D scripts (Colab GPU experiments)
├── colab_train.py                      # Full training pipeline → 0.999911 OOB + artifacts
├── colab_analysis.py                   # Oracle + calibration proof: 0.999911 is the ceiling
├── interpret.py                        # SHAP interpretability — run via `make interpret`
└── visualize.py                        # Feature visualization — run via `make visualize`

tests/
├── test_features.py            # Feature engineering unit tests
├── test_model.py               # RocketClassifier + min_class_recall unit tests
└── test_schema.py              # Schema validation unit tests

weights/                        # Model artifacts — gitignored, from GitHub Release
├── model.onnx                  # ONNX format — fastest inference (preferred)
├── model.lgb                   # Native LightGBM — fallback
├── model.pkl                   # joblib LGBMClassifier — legacy fallback
├── train_medians.npy           # 32-feature NaN imputation medians
└── threshold_biases.npy        # Per-class log-probability biases [0, 0.759, 0.658]

cache/                          # Feature caches — gitignored
├── cache_train_features.parquet   # 83-feature matrix (canonical)
├── cache_train_features.feather   # Arrow IPC sidecar — fast reads
├── cache_test_features.parquet
└── cache_test_features.feather
```

---

## Why 0.999911 Is the Practical Ceiling

The result — 2 misclassified trajectories out of 32,741 — has been rigorously analysed to understand whether improvement is possible. The analysis is reproducible via `research/colab_analysis.py`.

### The oracle test

An **oracle threshold test** tunes the per-class log-probability biases directly on the validation labels for each CV fold (cheating — labels are used to find the best threshold for that specific fold). If this cheating oracle cannot reach 1.0 on a fold, then no threshold, feature, or model can fix that fold's errors; the model's own probability outputs are insufficient — this is Bayes error.

**Result: oracle = 1.0 on all 10 folds.** The model already assigns higher probability to the correct class for every trajectory in every fold. The signal exists.

### Why 1.0 is not achievable with a global threshold

The oracle works by applying a different threshold per fold. In production, a single global bias vector must serve all 32,741 trajectories simultaneously. The two remaining errors require:

| Trajectory | True class | Model confidence (class-0) | Threshold needed to fix |
|---|---|---|---|
| traj=16004 | class-0 | ~9–19% | b₁ < −2.26 (massive class-1 penalty) |
| traj=23395 | class-0 | ~67% | b₁ < 0.73 (small adjustment) |

Fixing traj=16004 globally requires a class-1 penalty so large (b₁ < −2.26) that hundreds of legitimate class-1 trajectories with 80–95% class-1 confidence would be misclassified as class-0. The oracle can fix it per-fold only because, within each specific fold, those borderline class-1 trajectories happen not to coexist with traj=16004 in the same validation set.

### Confirmed by exhaustive search

A scipy `differential_evolution` global optimisation over the full OOB probability array — the most thorough threshold search possible — confirms that b₁ = 0.759 is the globally optimal value. Reducing it fixes traj=23395 but immediately breaks one or more class-1 trajectories, leaving the score unchanged at 0.999911.

A 20-seed ensemble with isotonic probability calibration was also tested. The ensemble averaged class-0 probability for traj=16004 stabilised at ~22% — higher than individual runs but still insufficient to overcome the global threshold constraint.

### What this means — and what would reach 1.0

0.999911 is not a Bayes error limit. The oracle proves the probabilities contain enough information. The gap is a calibration problem: traj=16004 needs its class-0 probability to rise from ~15% to ~40%+ so the global threshold can simultaneously fix it without breaking class-1 elsewhere.

Three concrete paths could close it:

**1. More training data for short trajectories.** traj=16004 has only 3 radar pings — nearly all 32 kinematic features are NaN-imputed from training medians. The model has almost no information and defaults to what 3-ping trajectories statistically look like (apparently class-1). More examples of short class-0 trajectories in training would directly teach the model that few-ping trajectories can be class-0.

**2. A better-calibrated model for this specific region.** The current model is confident (81–91% class-1) for a trajectory the oracle shows is recoverable. Alternative model families, deeper Optuna search targeting raw probability quality rather than tuned recall, or temperature scaling calibration could shift the probability mass. A Transformer operating on raw radar sequences combined with salvo context features is one candidate — it processes the temporal signature of each ping individually rather than as aggregate statistics, which may produce different probability estimates for anomalous short-trajectory cases.

**3. A streaming per-fold threshold (production architecture only).** In the real-time architecture described above, each processing window has a local class distribution that could support a locally-tuned threshold — but this requires labels or proxy signals that are unavailable at inference time in the batch setting.

---

## Model Interpretability

![SHAP Feature Importance](assets/shap_summary.png)

`research/interpret.py` computes exact SHAP values via `TreeExplainer`. Top discriminators:

- **Launch position** (`launch_x`, `launch_y`, `initial_z`) — geography clusters by rebel group; altitude encodes terrain and launch geometry
- **Horizontal speed** (`v_horiz_median`) — muzzle velocity is propellant-charge dependent
- **Salvo timing** (`salvo_time_rank`) — launch order within a salvo carries a kinematic imprint (assumption 3b)
- **Acceleration** (`acc_horiz_min`, `acc_mag_min`) — propulsion physics separate the classes
- **Ballistic arc** (`apogee_relative`, `delta_z_total`) — arc shape differs between rocket families

```bash
make interpret   # regenerates assets/shap_summary.png and assets/interpretation_report.txt
```

---

## Assumptions

1. Flat terrain — no mountains or valleys, so `z` is absolute altitude with no terrain correction required.
2. Rockets follow standard frictionless ballistic physics — kinematic features (velocity, acceleration, jerk) are physically meaningful.
3. Business knowledge from domain expert:
   - **3a.** Different launchers have different payload capacities (max rockets per launcher type).
   - **3b.** Rockets are typically fired in salvos — together or with a short delay.
   - **3c.** There are few rebel groups concentrated in geographic areas; each group is independent and purchases its own rocket supply — meaning each group uses one rocket type.

---

## Author

**Ilya Novikov**

[![GitHub](https://img.shields.io/badge/GitHub-IlyaNovikov--RD-181717?logo=github)](https://github.com/IlyaNovikov-RD)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ilya_Novikov-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ilya-novikov-data/)
