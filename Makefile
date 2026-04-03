# Rocket Classifier — project automation
#
# Prerequisites: uv must be installed.
#   https://docs.astral.sh/uv/getting-started/installation/
#
# Usage:
#   make install    Install all dependencies via uv
#   make test       Run the full pytest suite
#   make lint       Check code quality with ruff
#   make format     Auto-format source files with ruff
#   make demo       Launch the Streamlit interactive demo
#   make lock       Regenerate uv.lock from pyproject.toml
#   make download-weights  Download model + medians + biases from GitHub Release
#   make download-all      Download model artifacts + feature caches from GitHub Release
#   make run               Run inference pipeline → submission.csv
#   make interpret         Regenerate SHAP plot + report after a new model is deployed
#   make visualize         Regenerate demo.png (physics feature visualization)
#   make pipeline          Full local pipeline: download-all → run → interpret

.PHONY: install test lint format demo lock download-weights download-all run interpret visualize pipeline export-model

install:
	uv sync

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

download-weights:
	uv run python scripts/download_weights.py

download-all:
	uv run python scripts/download_weights.py --with-caches

run:
	uv run python -m rocket_classifier.main

interpret:
	uv run python research/interpret.py

visualize:
	uv run python research/visualize.py

export-model:
	uv run python scripts/export_fast_models.py

pipeline: download-all run interpret
	@echo "Pipeline complete. submission.csv and assets/ are up to date."
