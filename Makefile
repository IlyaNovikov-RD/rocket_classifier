# Rocket Classifier — project automation
#
# Prerequisites: uv must be installed.
#   https://docs.astral.sh/uv/getting-started/installation/
#
# Usage:
#   make install       Install all dependencies via uv
#   make test          Run the full pytest suite
#   make lint          Check code quality with ruff
#   make format        Auto-format source files with ruff
#   make demo          Launch the Streamlit interactive demo
#   make lock          Regenerate uv.lock from pyproject.toml
#   make download-models   Download model + medians + biases from GitHub Release
#   make download-all      Download model artifacts + feature caches from GitHub Release
#   make run               Run inference pipeline → submission.csv
#   make interpret         Regenerate SHAP plot + report after a new model is deployed
#   make visualize         Regenerate demo.png (physics feature visualization)
#   make pipeline          Full local pipeline: download-all → run → interpret
#   make docker            Build + run Docker image → output/submission.csv
#   make clean             Remove output/, cache/, artifacts/ for a fresh cold start
#
# After training a new model (research/train.py):
#   make export-model                         # build model.onnx + model_opt.onnx
#   make release TAG=v1.x.0 NOTES="..."      # upload all artifacts + trigger CI
#   ONNX files are included in the release from the start (no download gap).

.PHONY: install test lint format demo lock download-models download-all run interpret visualize pipeline export-model release docker clean

install:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

demo:
	uv run streamlit run rocket_classifier/app.py

lock:
	uv lock

download-models:
	uv run python scripts/download_models.py

download-all:
	uv run python scripts/download_models.py --with-caches

run:
	uv run python -m rocket_classifier.main

interpret:
	uv sync --group research
	uv run python research/interpret.py

visualize:
	uv sync --group research
	uv run python research/visualize.py

export-model:
	uv run python scripts/export_fast_models.py

clean:
	rm -rf output/ cache/ artifacts/ __pycache__ rocket_classifier/__pycache__ tests/__pycache__
	@echo "Cleaned output/, cache/, artifacts/, and __pycache__. Run make download-all to re-fetch artifacts."

docker:
	docker build -t rocket_classifier .
	docker run --rm -v $$(pwd)/output:/app/output rocket_classifier

pipeline: download-all run interpret
	@echo "Pipeline complete. submission.csv and assets/ are up to date."

# Create a GitHub Release with all required artifacts.
# Usage: make release TAG=v1.x.0 NOTES="What changed"
# Requires: artifacts/ and cache/ populated (run research/train.py first).
# After publishing, the post-release CI pipeline triggers automatically.
release:
	@test -n "$(TAG)"   || (echo "Usage: make release TAG=v1.x.0 NOTES='...'"; exit 1)
	@test -n "$(NOTES)" || (echo "Usage: make release TAG=v1.x.0 NOTES='...'"; exit 1)
	@test -f artifacts/model.lgb           || (echo "artifacts/model.lgb not found — run research/train.py first"; exit 1)
	@test -f artifacts/train_medians.npy   || (echo "artifacts/train_medians.npy not found"; exit 1)
	@test -f artifacts/threshold_biases.npy || (echo "artifacts/threshold_biases.npy not found"; exit 1)
	@test -f artifacts/model_opt.onnx      || (echo "artifacts/model_opt.onnx not found — run: make export-model"; exit 1)
	@test -f artifacts/model.onnx          || (echo "artifacts/model.onnx not found — run: make export-model"; exit 1)
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
	git checkout -B chore/training-report-$(TAG)
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
