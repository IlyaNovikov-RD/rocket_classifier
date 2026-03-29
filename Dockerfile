FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv — single static binary, no pip overhead
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifest first for layer caching.
# uv sync --frozen uses the lockfile directly — no resolver run in CI/CD.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source and data
COPY rocket_classifier/ ./rocket_classifier/
COPY data/ ./data/

# Install the project itself (now that source is available)
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "-m", "rocket_classifier.main"]
