# Rocket Classifier — project automation
#
# Prerequisites: Poetry must be installed and available as `poetry`.
#   https://python-poetry.org/docs/#installation
#
# Usage:
#   make install    Install all dependencies into the Poetry virtualenv
#   make test       Run the full pytest suite (55 unit tests)
#   make lint       Check code quality with ruff
#   make format     Auto-format source files with ruff
#   make demo       Launch the Streamlit interactive demo in your browser

.PHONY: install test lint format demo

install:
	poetry install

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

demo:
	poetry run streamlit run src/app.py
