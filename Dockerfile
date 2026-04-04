FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv — single static binary, no pip overhead
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy production source only (no data, no research, no models)
COPY rocket_classifier/ ./rocket_classifier/
COPY scripts/download_models.py ./scripts/

# Install the project itself
RUN uv sync --frozen --no-dev

# Download model artifacts from GitHub Release at build time
RUN uv run python scripts/download_models.py

# Pre-create runtime directories
RUN mkdir -p cache models outputs

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uv", "run", "python", "-m", "rocket_classifier.main"]
