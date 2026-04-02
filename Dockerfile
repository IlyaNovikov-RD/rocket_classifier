FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv — single static binary, no pip overhead
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy production source only (no data, no research, no weights)
COPY rocket_classifier/ ./rocket_classifier/
COPY download_weights.py ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Download model artifacts from GitHub Release at build time
# weights/ is created by download_weights.py
RUN uv run python download_weights.py

# Pre-create runtime directories
RUN mkdir -p cache outputs

CMD ["uv", "run", "python", "-m", "rocket_classifier.main"]
