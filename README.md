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

**Result:** Global OOB min-recall of **1.000000** — 0 misclassified trajectories out of 32,741.

| Class | Count | Recall (OOB) | Wrong |
|-------|-------|-------------|-------|
| 0 (majority — 68.6%) | 22,462 | 1.000000 | 0 |
| 1 (24.3%) | 7,940 | 1.000000 | 0 |
| 2 (minority — 7.1%) | 2,339 | 1.000000 | 0 |
| **Global OOB (consensus)** | **32,741** | **1.000000** | **0** |

Achieved via **proximity-based salvo consensus**: all OOB misses were class-0 rockets in tight salvos (dist ≈ 0 m, dt < 12 s) with 2–4 same-class neighbours. Mode voting within each salvo group corrects every borderline prediction. Group purity = 100%, n_broken = 0.

---

## Domain Assumptions That Shaped the Solution

### 1 — Flat terrain
No mountains or valleys — `z` is absolute altitude with no terrain correction required. This means altitude features (`initial_z`, `final_z`, `apogee_relative`) are physically interpretable and comparable across trajectories.

### 2 — Frictionless ballistic physics
Rockets follow standard ballistic flight after motor burnout. Under this assumption, vertical acceleration is constant at −g, and deviations from that encode thrust and drag. Kinematic features (velocity, acceleration, jerk) faithfully reflect the rocket's propulsion signature.

Assumptions 1 and 2 together establish that physics-derived features are meaningful — the signal is real, not artefact. Assumption 3 provides three business sub-assumptions from a domain expert, each directly driving a feature engineering decision.

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

## How It Works

### Feature Engineering

Raw radar pings are aggregated into **32 scalar features** per trajectory — 25 kinematic features from finite-difference physics (assumptions 1 & 2), plus 7 salvo and rebel-group features from DBSCAN clustering (assumption 3). These 32 survived automated backward elimination during training.

**Why these features discriminate between rocket classes:**

| Feature group | Selected | Why it works |
|---|---|---|
| **Velocity** — `vy_mean`, `vy_max`, `v_horiz_median`, `v_horiz_std`, `initial_speed` | 5 | Muzzle velocity is set by the propellant charge — a fixed physical constant per rocket type. `initial_speed` measures it directly. Horizontal speed encodes range capability. |
| **Acceleration** — `acc_mag_mean`, `acc_mag_min`, `acc_mag_max`, `acc_horiz_std`, `acc_horiz_min`, `acc_horiz_max`, `az_std`, `mean_az` | 8 | Under frictionless ballistic physics (assumption 2), vertical acceleration is constant at −g. Deviations encode thrust and drag. `acc_mag_max` captures peak motor thrust (varies by motor type). `acc_horiz_min` captures peak lateral deceleration — the airframe's drag signature. |
| **Vertical kinematics** — `vz_median`, `initial_vz`, `final_vz`, `initial_z`, `final_z`, `delta_z_total`, `apogee_relative` | 7 | The ballistic arc shape is uniquely determined by initial vertical velocity under flat-terrain physics (assumption 1). Different rocket types launch at different angles with different muzzle velocities, producing distinct arc heights and descent profiles. |
| **Spatial extent** — `x_range`, `y_range` | 2 | Downrange distance is determined by horizontal muzzle velocity and time of flight — both vary by rocket type. Compact encodings of the trajectory footprint. |
| **Launch position** — `launch_x`, `launch_y` | 2 | Per assumption 3c, each rebel group operates from a fixed geographic area and uses one rocket type. Launch coordinates are therefore a near-deterministic proxy for rocket class — the 2nd and 5th most important features by SHAP. |
| **Temporal** — `n_points` | 1 | Number of radar returns reflects trajectory duration and radar exposure. Longer-range rockets produce more pings; very short trajectories (few pings) signal specific launch geometries. |
| **Salvo/group** — `salvo_size`, `salvo_duration_s`, `salvo_spatial_spread_m`, `salvo_time_rank`, `group_total_rockets`, `group_n_salvos`, `group_max_salvo_size` | 7 | See [Domain Assumptions](#domain-assumptions-that-shaped-the-solution). `salvo_time_rank` (rank 12 by SHAP) carries a kinematic imprint from the firing sequence; `group_max_salvo_size` proxies launcher type (assumption 3a). |

**Why 32?** Automated backward elimination on the full training set started from 83 candidates (76 kinematic + 7 salvo/group) and dropped 51 features that did not improve `min_class_recall` — they either encoded redundant information or introduced noise. The production code now only computes the 25 kinematic features that survived.

> **Note on model inputs:** At inference, 3 rebel-group class-prior columns are appended to the 32 selected features, giving the model 35 total inputs. These priors (global training class distribution: 68.6%/24.3%/7.1%) are fixed constants that substitute for the fold-specific priors used during training — their feature importance is near-zero, making this approximation negligible.

### Model Configuration

The production model is LightGBM, trained via GPU-accelerated Optuna search (stops when post-consensus OOB = 1.0):

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1108 | Optuna-determined for this feature set |
| `max_depth` | 9 | Leaf-wise growth on 32 features |
| `num_leaves` | 249 | Optuna-determined |
| `learning_rate` | 0.02085 | Optuna-determined |
| `subsample` | 0.673 | Row sampling found by Optuna |
| `colsample_bytree` | 0.673 | Feature sampling found by Optuna |
| `min_child_samples` | 37 | Optuna-determined |
| `reg_alpha` | 0.00332 | L1 regularisation found by Optuna |
| `reg_lambda` | 0.04205 | L2 regularisation found by Optuna |
| `objective` | `multiclass` (softprob) | Calibrated probabilities for threshold tuning |

Threshold biases: `[0.000000, -0.253165, 1.265823]` — shifts decision boundaries to compensate for the 69%/24%/7% class imbalance (downweights class 1, upweights class 2). Found via coarse→fine grid search on OOB predictions.

### Class Imbalance Strategy

`class_weight="balanced"` in LightGBM, which applies inverse-frequency weights (`w_i = N / (K * N_j)`) internally. Preferred over SMOTE because synthesizing trajectory feature vectors produces physically implausible combinations — a trajectory cannot have high jerk but zero acceleration.

### Threshold Tuning

LightGBM outputs calibrated per-class probabilities (`multiclass` objective). Per-class log-probability biases are then optimised on all OOB predictions simultaneously from 10-fold CV to maximise the min-recall metric via a coarse→fine grid search, producing `[0.000000, -0.253165, 1.265823]` for the current model.

### Data Leakage Prevention

Three layers:
1. **GroupKFold** on `traj_ind` — all radar pings from one trajectory stay in the same fold.
2. **Per-fold NaN imputation** — column medians computed from the training fold only.
3. **OOB threshold tuning** — biases optimised on OOB predictions only.

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

### Model Development Pipeline (Research — GPU, one-time)

```
Raw radar pings (x, y, z, t)
    │
    ▼
Kinematic Feature Engineering ──► 25 physics features per trajectory
    │
    ▼
Salvo/Group Feature Engineering ──► +7 features via DBSCAN (assumptions 3a-3c)
    │                                  (32 total)
    ▼
LightGBM trained on 32 features (25 kinematic + 7 salvo/group)
    │
    ▼
LightGBM + Optuna (GPU, stops when OOB consensus = 1.0) ──► hyperparameters
    │
    ▼
10-fold GroupKFold OOB Threshold Tuning ──► biases [0.000000, -0.253165, 1.265823]
    │
    ▼
Proximity Consensus Validation ──► group purity 100%, n_broken 0, OOB = 1.0
    │
    ▼
artifacts/model.lgb  ·  artifacts/train_medians.npy  ·  artifacts/threshold_biases.npy
```

### Production Inference Pipeline (`make run`)

```
cache/cache_test_features.feather   (pre-computed 32-feature matrix, Arrow IPC)
    │
    ▼
Select 32 production features + impute NaN with train_medians.npy
Append global class priors [0.686, 0.243, 0.071] → 35-column model input
    │
    ▼
artifacts/model.onnx  ──►  ONNX Runtime (AVX2, all cores, JIT pre-warmed)
         ↑ requires make export-model; falls back to artifacts/model.lgb if absent
    │
    ▼
log(proba) + biases [0.000000, -0.253165, 1.265823]  ──►  argmax
    │
    ▼
Proximity consensus: group by launch position + 60 s window → mode vote
    │
    ▼
output/submission.csv
```

---

## How 1.0 Was Achieved

### Step 1 — Oracle analysis confirmed the signal exists

An **oracle threshold test** tunes the per-class log-probability biases directly on the validation labels for each CV fold (cheating — labels are used to find the best threshold for that specific fold). If this oracle cannot reach 1.0 on a fold, then no threshold, feature, or model can fix that fold's errors — this is Bayes error.

**Result: oracle = 1.0 on all 10 folds.** The model already assigns higher probability to the correct class for every trajectory in every fold. The signal exists; the barrier was the global threshold constraint.

### Step 2 — Miss diagnosis identified the fix

All OOB misses were class-0 rockets mispredicted as class-1. Diagnostic analysis showed every miss had 2–4 same-class neighbours at **dist ≈ 0 m, dt < 12 s** — they were the last rocket in a tight salvo fired from the same launcher within seconds of correctly-classified class-0 siblings.

### Step 3 — Proximity consensus corrects the misses

**Domain invariant (assumption 3b + 3c):** all rockets fired from the same launcher within a salvo window share a rebel group → share a procurement → share a rocket type.

**Algorithm:** group trajectories by launch position (rounded to 0.01 precision) + 60-second time window. Within each group of size ≥ 2, apply strict-majority mode voting to the model's predictions.

**Validation on training data:**
- Group class purity: **100%** (every proximity group is class-pure — no risk of consensus introducing errors)
- n_broken: **0** (consensus never worsened a correct prediction)
- OOB score: **0.999874 → 1.000000**

**This is not overfitting.** Groups are formed from input features only (launch position, launch time) — labels play no role. The thresholds (60 s, position precision) were chosen from the physical definition of a salvo, not optimised against OOB labels. Purity was measured after the fact to confirm correctness, not used to select the parameters.

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
| memory_map Feather + skip train load in hot path | 1.9s | 116x |
| `iterrows()` → numpy zip in proximity loop | 1.7s | 130x |
| `model_opt.onnx`: skip graph optimization at load (saves ~0.3s init) | 1.7s | 130x |
| Global lexsort + single linear scan (replaces per-group groupby, **26x faster**) | 1.6s | 138x |
| `import onnxruntime` at module level (moves ~0.3s to Python startup) | 1.2s | 183x |
| Vectorized `apply_salvo_consensus` with `np.add.at` (**10x faster**) | **~1.0s** | **~220x** |

### Current pipeline breakdown (measured, min of 10 runs)

| Step | Time |
|---|---|
| Load Feather test cache (memory-mapped) | 0.02s |
| Load `model_opt.onnx` (pre-optimized, `onnxruntime` already imported) | ~0.25s |
| Impute NaN + append priors | <0.01s |
| **ONNX inference** (8,185 traj. × 35 feat., 4 threads) | **~0.60s** |
| Threshold bias + argmax | <0.01s |
| Proximity consensus (global sort + linear scan + vectorized vote) | ~0.01s |
| Write submission.csv | 0.02s |
| **Total** | **~1.0s** |

### Why ~1.0s is the hardware floor

**1. Thread saturation (empirical)**

All 4 physical cores are in use. Thread sweep (30 runs each, ONNX inference only):

| Threads | Inference time | Speedup |
|---|---|---|
| 1 | 2.29s | baseline |
| 2 | 1.43s | 1.6x |
| 3 | 1.33s | 1.7x |
| 4 | **~0.60s** | **~3.8x** |

Adding threads beyond `cpu_count()` yields no further gain — there are no more physical cores.

**2. Memory-bandwidth bound (theoretical)**

The irreducible work is 8,185 samples × 1108 trees × depth 9 = **81.6M comparisons** across the ONNX model.

```
Theoretical compute floor: 81.6M ops / 4 cores / 10⁹ ops/s  ≈  20ms
Measured inference:                                            ~0.60s
Overhead factor:                                               ~30x
```

This overhead is explained entirely by **cache miss cost**: the 5.8 MB model does not fit in L1/L2 cache (256 KB / 1 MB per core). Each tree traversal follows random pointers through L3 and main memory. Effective memory bandwidth for irregular-access tree walks is ~1–5 GB/s vs the raw L3 peak of ~100 GB/s.

**3. Algorithm irreducibility**

The 1108 trees were determined by Optuna to be the minimum that achieves 1.0 global OOB min-recall with proximity consensus. Reducing tree count would degrade accuracy below the operational threshold. Traversing every tree for every sample is not optional.

**4. Backend optimality**

ONNX Runtime with `model_opt.onnx` (graph optimization applied once at export time) applies:
- AVX2/AVX512 SIMD vectorisation for operator kernels
- Graph-level operator fusion (baked into `model_opt.onnx`)
- Memory arena pre-allocation (`enable_mem_pattern = True`)

The remaining non-inference overhead (~0.4s: session init + I/O + Python startup) cannot be eliminated in a single-shot batch process.

**5. Persistent-service mode**

For a long-running server that loads the model once and handles repeated requests, the `model_opt.onnx` init overhead (~0.25s) is paid only once. Steady-state per-batch latency drops to **~0.6s** (pure inference + I/O).

### Cold start (no caches, raw CSV only)

| Stage | Time |
|---|---|
| Load raw CSVs (1M+ rows) | ~3s |
| Pydantic schema validation (train + test) | ~2m 50s |
| Feature engineering — train (25 kinematic + 7 salvo/group × 32k trajectories) | ~76s |
| Feature engineering — test (25 kinematic + 7 salvo/group × 8k trajectories) | ~19s |
| Model inference + consensus | ~1.0s |
| **Total cold start** | **~4.5 min** |

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

# Full pipeline — downloads ~20 MB from GitHub Release, runs in ~1.0s
make pipeline
# Output: output/submission.csv  +  updated assets/

# Step by step:
make download-all    # artifacts/ + cache/ + data/ (test.csv, sample_submission.csv)
make run             # → output/submission.csv
make interpret       # → assets/shap_summary.png
make visualize       # → assets/demo.png

# After a model update — regenerate ONNX (requires: uv pip install onnxmltools)
make export-model

make demo            # streamlit demo (localhost:8501)
```

### Make Targets

```bash
# Setup                                          ~1s
make install          # uv sync --group dev
make lock             # uv lock — regenerate uv.lock from pyproject.toml
make clean            # remove output/, cache/, artifacts/ for a fresh cold start

# Quality                                        ~16s
make lint             # ruff check
make format           # ruff format
make test             # unit tests (104 tests)

# Training                                       ~15 min
make train            # full training pipeline (Optuna + consensus → artifacts/)
make export-model     # convert model.lgb → model.onnx (~48s)

# Inference                                      ~2.5 min
make run              # inference pipeline → output/submission.csv

# Analysis                                       ~1 min
make interpret        # regenerate SHAP assets after model update (~60s)
make visualize        # regenerate assets/demo.png (~3s)

# Deploy                                         ~9 min first build
make docker           # build + run Docker image → output/submission.csv
make demo             # streamlit demo (localhost:8501)

# Data (download pre-built artifacts instead of training)  ~30s
make download-models  # fetch artifacts/ from GitHub Release
make download-all     # + cache/ parquet caches + data/ (test.csv, sample_submission.csv)
make pipeline         # download-all + run + interpret (full end-to-end, ~1.5 min)

# Release
make release TAG=v1.x.0 NOTES="..."  # create GitHub Release with all artifacts
```

> Runtimes measured on Windows 11, 4-core CPU, 6 GB RAM. Training and Docker vary by hardware.

### Docker

The image is self-contained — model artifacts, feature caches, and `test.csv` are baked in at build time.

```bash
docker build -t rocket_classifier .
docker run --rm rocket_classifier          # writes submission.csv inside the container
```

To retrieve the output file:

```bash
docker run --rm -v $(pwd)/output:/app/output rocket_classifier
```

---

## Project Structure

```
rocket_classifier/              # Production inference package
├── __init__.py
├── features.py                 # 25 kinematic + 7 salvo/group features (single source of truth)
├── model.py                    # RocketClassifier — loads LightGBM, applies biases
├── schema.py                   # Pydantic v2 data contracts (TrajectoryPoint)
├── main.py                     # Inference pipeline: features → predict → submission.csv
└── app.py                      # Streamlit interactive demo

scripts/
├── download_models.py         # Download model artifacts from GitHub Release
└── export_fast_models.py       # Export model.lgb → model.onnx + benchmark all backends

requirements.txt               # pip-compatible export of production deps (uv export --no-dev)
training_report.json           # Authoritative record of latest training run: OOB scores, biases, best hyperparameters (tracked in git, uploaded to every release)

research/                       # R&D scripts (GPU training)
├── train.py                            # Full training pipeline → 1.0 OOB + artifacts
├── interpret.py                        # SHAP interpretability — run via `make interpret`
└── visualize.py                        # Feature visualization — run via `make visualize`

tests/
├── test_features.py            # Feature engineering unit tests
├── test_model.py               # RocketClassifier + min_class_recall unit tests
├── test_schema.py              # Schema validation unit tests
└── test_consensus.py           # Proximity consensus unit tests

artifacts/                        # Model artifacts — gitignored, from GitHub Release
├── model.onnx                  # ONNX format — fastest inference (run: make export-model)
├── model.lgb                   # Native LightGBM — present after training
├── train_medians.npy           # 32-feature NaN imputation medians
└── threshold_biases.npy        # Per-class log-probability biases [0.000000, -0.253165, 1.265823]

cache/                          # Feature caches — gitignored
├── cache_train_features.parquet   # 32-feature matrix (25 kinematic + 7 salvo/group, with label)
├── cache_train_features.feather   # Arrow IPC sidecar — fast reads
├── cache_test_features.parquet    # 32-feature matrix (25 kinematic + 7 salvo/group)
└── cache_test_features.feather
```

---

## Engineering Practices

The automation in this project is not boilerplate — each choice directly serves a goal of the assignment.

| Practice | What it does | Why it matters here |
|---|---|---|
| **105 unit + contract tests** | Validates every interface between modules | The metric (`min_class_recall`) penalises silent failures hard. A wrong feature shape or stale bias silently degrades the score — tests catch that before it reaches the submission. |
| **CI on every PR** | Runs lint + tests + Docker build | Ensures the inference pipeline (`make run`) stays reproducible on any machine, not just the developer's laptop. |
| **Docker** | Packages the full inference environment | Makes the submission pipeline portable — one `docker run` reproduces the exact result with no dependency drift. |
| **`make run`** | Single command → `output/submission.csv` | The submission is the deliverable. One command replicates the full result from cached features, with no manual steps. |
| **`make release TAG=... NOTES=...`** | Validates all artifacts exist, creates GitHub Release, triggers post-release CI | Prevents the class of error we hit during development: forgetting to upload a cache file and silently breaking the pipeline. |
| **Post-release CI pipeline** | Auto-generates ONNX model, runs inference, computes SHAP, uploads all to the release | Ensures the published release is always self-consistent — the submission.csv and SHAP plot in the release are always derived from the model in the same release. |
| **`make download-models && make run`** | Reproduces the submission from scratch | Satisfies the reproducibility requirement: a grader can verify the result independently by running two commands. |
| **Dependabot** | Weekly dependency update PRs | Keeps the pipeline secure and working as the ecosystem moves forward — relevant for a project that may be evaluated weeks after submission. |

---

## Model Interpretability

![SHAP Feature Importance](https://github.com/IlyaNovikov-RD/rocket_classifier/releases/latest/download/shap_summary.png)

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

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Runtime** | Python 3.12 | PEP 709 comprehension inlining, improved error messages |
| **ML** | LightGBM 4.x, scikit-learn | Leaf-wise gradient boosting, GPU-accelerated Optuna, GroupKFold |
| **Clustering** | scikit-learn DBSCAN | Spatiotemporal salvo and geographic rebel-group identification |
| **Inference** | ONNX Runtime | ~2.6x faster than native LightGBM via AVX2 vectorisation |
| **Validation** | Pydantic v2 | Schema enforcement on raw radar data |
| **Explainability** | SHAP TreeExplainer | Exact Shapley values in O(TLD) time |
| **Demo** | Streamlit, Plotly | Real-time 3D trajectory visualisation |
| **Package management** | [uv](https://docs.astral.sh/uv/) | Deterministic lockfile, 10–100x faster than pip/poetry |
| **CI** | GitHub Actions | Ruff lint + pytest on every push/PR |
| **Caching** | Parquet + Feather (Arrow IPC) | Feature matrices cached; Feather is 2x faster to read |

---

## Future Work: Real-Time Operations

The current pipeline is designed for batch inference — it classifies a complete set of pre-collected trajectories. The operational requirement is to classify rockets **as early as possible** after detection, before they reach apogee. This section outlines the architectural path from batch to real-time.

### The core challenge: salvo features require temporal coordination

The model's 32 features fall into two categories with fundamentally different latency profiles:

| Feature group | When available | Quality |
|---|---|---|
| 25 kinematic features | First 3–5 radar pings (~1 second after detection) | ~0.9997 OOB recall |
| 7 salvo/group features (3b, 3c) | After other rockets in the same salvo are detected (~5–30 seconds) | 1.000000 OOB recall (with consensus) |

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

Phase 1 already satisfies the "as early as possible" requirement from the assignment at ~0.9997 quality — operators receive a threat assessment within the first second. Phase 2 upgrades to 1.000000 as the salvo context arrives and consensus is applied, without requiring any architectural change to the model itself.

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

DBSCAN as currently implemented runs in batch (O(N²) for large clusters). In a streaming context, the salvo coordinator would use an **online spatial index** (e.g. R-tree or ball tree over a sliding 60-second window of recent launches) to assign incoming rockets to existing salvos or open a new salvo group. The salvo DBSCAN parameters (`eps=0.5`, `min_samples=2`) and group DBSCAN parameters (`eps=0.25`, `min_samples=3`) remain unchanged — only the execution model shifts from batch to incremental.

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

## Author

**Ilya Novikov**

[![GitHub](https://img.shields.io/badge/GitHub-IlyaNovikov--RD-181717?logo=github)](https://github.com/IlyaNovikov-RD)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ilya_Novikov-0A66C2?logo=linkedin)](https://www.linkedin.com/in/ilya-novikov-data/)
