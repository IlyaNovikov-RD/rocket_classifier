# Rocket Classifier — project automation
#
# Prerequisites: uv must be installed.
#   https://docs.astral.sh/uv/getting-started/installation/
#
# ── Top-level ──
#   make all                Run full validation: setup → quality → train → test → run → analysis
#   make all-full           all + Docker build + Streamlit demo
#
# ── Setup ──
#   make install            Install all dependencies via uv
#   make lock               Regenerate uv.lock from pyproject.toml
#   make clean              Remove output/, cache/, artifacts/ for a fresh cold start
#
# ── Quality ──
#   make lint               Check code quality with ruff
#   make format             Auto-format source files with ruff
#
# ── Training ──
#   make train              Run full training pipeline (Optuna + consensus → artifacts/)
#   make export-model       Build model.onnx + model_opt.onnx from model.lgb
#   make test               Run the full pytest suite (after artifacts exist)
#
# ── Inference ──
#   make run                Run inference pipeline → output/submission.csv
#
# ── Analysis ──
#   make interpret          Regenerate SHAP plot + report after a new model is deployed
#   make visualize          Regenerate demo.png (physics feature visualization)
#
# ── Deploy ──
#   make docker             Build + run Docker image → output/submission.csv
#   make docker-clean       Remove image and rebuild from scratch
#   make demo               Launch the Streamlit interactive demo (localhost:8501)
#
# ── Data (download pre-built artifacts instead of training) ──
#   make download-models    Download model + medians + biases from GitHub Release
#   make download-all       Download model artifacts + feature caches from GitHub Release
#   make pipeline           Full local pipeline: download-all → run → interpret
#
# ── Release ──
#   make release TAG=v1.x.0 NOTES="..."   Upload all artifacts + trigger CI
#
#   ONNX files are included in the release from the start (no download gap).

.PHONY: all all-full install lock clean lint format train export-model test run interpret visualize docker docker-clean demo download-models download-all pipeline release

# ── Top-level ─────────────────────────────────────────────────────────────────

all: install lock clean lint format train export-model test run interpret visualize
	@echo "All targets passed."

all-full: all docker-clean demo
	@echo "All targets (including Docker and demo) passed."

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	uv sync --group dev

lock:
	uv lock
	uv export --no-dev --no-hashes -o requirements.txt

clean:
	rm -rf output/ cache/ artifacts/ __pycache__ rocket_classifier/__pycache__ tests/__pycache__
	@echo "Cleaned output/, cache/, artifacts/, and __pycache__. Run make download-all to re-fetch artifacts."

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	uv run ruff check .

format:
	uv run ruff format .

# ── Training ──────────────────────────────────────────────────────────────────

train:
	uv sync --group research
	uv run python research/train.py

export-model:
	uv run python scripts/export_fast_models.py

test:
	uv run pytest tests/ -v

# ── Inference ─────────────────────────────────────────────────────────────────

run:
	uv run python -m rocket_classifier.main

# ── Analysis ──────────────────────────────────────────────────────────────────

interpret:
	uv sync --group research
	uv run python research/interpret.py

visualize:
	uv sync --group research
	uv run python research/visualize.py

# ── Deploy ────────────────────────────────────────────────────────────────────

docker:
	docker build -t rocket_classifier .
	docker run --rm --mount type=bind,source=$$(pwd)/output,target=/app/output rocket_classifier

docker-clean:
	docker rmi rocket_classifier 2>/dev/null || true
	$(MAKE) docker

demo:
	uv run streamlit run rocket_classifier/app.py --server.headless=true

# ── Data (download pre-built artifacts instead of training) ───────────────────

download-models:
	uv run python scripts/download_models.py

download-all:
	uv run python scripts/download_models.py --with-caches

pipeline: download-all run interpret
	@echo "Pipeline complete. submission.csv and assets/ are up to date."

# ── Release ───────────────────────────────────────────────────────────────────
# Create a GitHub Release with all required artifacts.
# Usage: make release TAG=v1.x.0 NOTES="What changed"
# Requires: artifacts/ and cache/ populated (run research/train.py first).
# After publishing, the post-release CI pipeline triggers automatically.
release:
	@test -n "$(TAG)"   || (echo "Usage: make release TAG=v1.x.0 NOTES='...'"; exit 1)
	@test -n "$(NOTES)" || (echo "Usage: make release TAG=v1.x.0 NOTES='...'"; exit 1)
	@git diff --quiet --exit-code && git diff --cached --quiet --exit-code || (echo "Error: working tree is dirty — commit or stash changes first"; exit 1)
	@test -f artifacts/model.lgb           || (echo "artifacts/model.lgb not found — run research/train.py first"; exit 1)
	@test -f artifacts/train_medians.npy   || (echo "artifacts/train_medians.npy not found"; exit 1)
	@test -f artifacts/threshold_biases.npy || (echo "artifacts/threshold_biases.npy not found"; exit 1)
	@test -f artifacts/model_opt.onnx      || (echo "artifacts/model_opt.onnx not found — run: make export-model"; exit 1)
	@test -f artifacts/model.onnx          || (echo "artifacts/model.onnx not found — run: make export-model"; exit 1)
	@test -f training_report.json          || (echo "training_report.json not found — run research/train.py first"; exit 1)
	@test -f cache/cache_train_features.parquet || (echo "cache/cache_train_features.parquet not found"; exit 1)
	@test -f cache/cache_test_features.parquet  || (echo "cache/cache_test_features.parquet not found"; exit 1)
	@test -f data/test.csv              || (echo "data/test.csv not found"; exit 1)
	@test -f data/sample_submission.csv || (echo "data/sample_submission.csv not found"; exit 1)
	gh release create $(TAG) \
	  artifacts/model.lgb \
	  artifacts/model.onnx \
	  artifacts/model_opt.onnx \
	  artifacts/train_medians.npy \
	  artifacts/threshold_biases.npy \
	  training_report.json \
	  cache/cache_train_features.parquet \
	  cache/cache_test_features.parquet \
	  data/test.csv \
	  data/sample_submission.csv \
	  --title "$(TAG): $(NOTES)" \
	  --notes "$(NOTES)"
	@echo "Release $(TAG) created. Post-release pipeline will run automatically."
	@echo "Creating PR to commit training_report.json provenance..."
	git show-ref --verify --quiet refs/heads/chore/training-report-$(TAG) && \
	  (echo "Error: branch chore/training-report-$(TAG) already exists — delete it first or use a new TAG"; exit 1) || true
	git checkout -b chore/training-report-$(TAG)
	git config user.name  "$$(git log -1 --format='%an')" 2>/dev/null || true
	git config user.email "$$(git log -1 --format='%ae')" 2>/dev/null || true
	git add training_report.json
	git diff --cached --quiet && echo "training_report.json unchanged, skipping PR" || \
	  (git commit -m "chore: update training_report.json for $(TAG)" && \
	   git push -u origin chore/training-report-$(TAG) && \
	   gh pr create \
	     --title "chore: update training_report.json for $(TAG)" \
	     --base main \
	     --body "Training provenance for $(TAG): OOB scores, hyperparameters, and consensus parameters.")
	git checkout main
	@echo "Done. Merge the PR to record provenance in main."
