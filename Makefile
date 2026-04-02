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
#   make download-weights  Download model artifacts from GitHub Release
#   make interpret         Regenerate SHAP plot + report after a new model is deployed

.PHONY: install test lint format demo lock download-weights interpret

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
	uv run python download_weights.py

interpret:
	uv run python research/interpret.py
