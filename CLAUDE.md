# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Top-level
make all              # full validation: setup → quality → train → test → run → analysis
make all-full         # all + cold Docker rebuild + Streamlit demo
# Setup
make install          # uv sync --group dev — install all dependencies
make lock             # uv lock — regenerate uv.lock from pyproject.toml
make clean            # remove output/, cache/, artifacts/ for a fresh cold start
# Quality
make lint             # uv run ruff check .
make format           # uv run ruff format .
# Training
make train            # full training pipeline (Optuna + consensus → artifacts/)
make export-model     # convert model.lgb → model.onnx + model_opt.onnx (requires onnxmltools)
make test             # uv run pytest tests/ -v (runs after artifacts exist → 105/105)
# Inference
make run              # inference pipeline → output/submission.csv
# Analysis
make interpret        # regenerate SHAP assets after model update
make visualize        # regenerate assets/demo.png after feature changes
# Deploy
make docker           # build + run Docker image → output/submission.csv
make docker-clean     # remove image and rebuild from scratch
make demo             # launch Streamlit app (localhost:8501)
# Data (download pre-built artifacts instead of training)
make download-models  # fetch model artifacts from latest GitHub Release into artifacts/
make download-all     # + feature caches into cache/ + test.csv/sample_submission.csv into data/
make pipeline         # download-all + run + interpret (full end-to-end)
# Release
make release TAG=v1.x.0 NOTES="..."  # create GitHub Release with all artifacts
```

Run a single test: `uv run pytest tests/test_model.py::TestMinClassRecall::test_perfect_predictions -v`

## Architecture

**Production package** (`rocket_classifier/`): Inference-only. No training code.

- `features.py` — Extracts 32 features per trajectory: 25 kinematic features from raw `(x, y, z, time_stamp)` radar pings via finite-difference kinematics, plus 7 salvo/rebel-group features from DBSCAN clustering (domain assumptions 3a-3c). Single source of truth for feature engineering.
- `model.py` — `RocketClassifier` class. Backend selected automatically: `model_opt.onnx` (pre-optimized ONNX, fastest) → `model.onnx` → `model.lgb`. Contains `SELECTED_FEATURES` (32 features used in production: 25 kinematic + 7 salvo/group), `PRODUCTION_BIASES` (threshold-tuned log-probability biases), and `_GLOBAL_CLASS_PRIOR` (appended rebel-group prior columns). These are the single source of truth — never duplicate them.
- `schema.py` — Pydantic v2 validation for raw radar data (`TrajectoryPoint`, `validate_dataframe`).
- `main.py` — Orchestrates: load data → validate → featurize → predict → proximity consensus → write `output/submission.csv`.
- `app.py` — Streamlit demo. Downloads model from GitHub Release if not local. Uses `_extract_trajectory_features` from `features.py` directly (salvo features unavailable for single-trajectory demo — imputed from medians).

**Research scripts** (`research/`): Training and analysis scripts. These import `optuna`, `lightgbm` and other libraries not in production deps — that's intentional. Never move these into `rocket_classifier/`.
- `train.py` — full training pipeline: feature engineering → Optuna → proximity consensus → artifacts → 1.000000 OOB

**Model artifacts** (`artifacts/`): Gitignored. Downloaded from the latest GitHub Release via `download_models.py`. Uses `releases/latest/download` URLs — never hardcode version numbers.

**Feature caches** (`cache/`): Gitignored parquet files. Regenerated from `data/` or downloaded from release.

## Key Design Decisions

- **Metric**: `min_class_recall` — worst-class recall. Every design choice optimises for this, not accuracy.
- **32 features**: 25 kinematic features that survived backward elimination, plus 7 salvo/group features from domain assumptions. Production code only computes these 32 — the 51 eliminated features are no longer extracted.
- **Model input = 35**: The model was trained with 32 base features + 3 rebel-group class-prior columns appended per fold. At production inference `_GLOBAL_CLASS_PRIOR` (the training class distribution) substitutes for those 3 columns — they have near-zero feature importance.
- **Threshold biases** `[0.000000, -0.253165, 1.265823]`: Applied as `argmax(log(proba) + biases)` to shift decision boundaries toward minority classes (class distribution: 69%/24%/7%). Exact values saved in `artifacts/threshold_biases.npy` and `training_report.json`.
- **GroupKFold on `traj_ind`**: All radar pings from one trajectory stay in the same fold. Prevents data leakage.
- **No training in production**: Model was trained via `research/train.py`. `rocket_classifier/` only does inference.
- **ONNX regeneration**: After any model update, run `make export-model` (requires `onnxmltools`) to rebuild `artifacts/model.onnx` and `artifacts/model_opt.onnx` (pre-graph-optimized, ~0.3s faster to load) and benchmark all backends.

## Linting

Ruff with `line-length = 100`, target `py312`. Intentionally suppressed: `N803`/`N806` (ML convention allows uppercase `X`, `X_train`), `C408` (Plotly uses `dict()`), `B008`.

## CI

GitHub Actions on push/PR to main: ruff check + pytest + Docker build.
Post-release workflow (`update-interpretability.yml`) triggers on every GitHub Release: exports `model.onnx` + `model_opt.onnx`, runs inference, generates SHAP assets, and uploads `submission.csv` + all artifacts to the release.
Use `make release TAG=v1.x.0 NOTES="..."` to create a release with all required artifacts validated.

## Workflow Rules

- **Never modify `rocket_classifier/` for experiments** — only after an experiment proves improvement.
- **Don't push/PR until explicitly told** the task is done.
- **Cascade check on any change**: if you update one thing (e.g., model, features, biases), check whether README, app.py, docstrings, release assets, or deps also need updating.
