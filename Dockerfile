# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# SmartGrid KG — FastAPI prediction service (Stage 3).
#
# No builder stage on purpose: numpy / pandas / scikit-learn / asyncpg publish
# manylinux wheels for CPython 3.11, so no compiler toolchain is required.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Unbuffered logs so `docker logs` streams immediately; no .pyc files in layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Only curl, and only for the HEALTHCHECK below. Runtime wheels cover the
# scientific stack, so build-essential/gcc is deliberately not installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer stays cached until requirements.txt changes.
# For byte-reproducible builds swap in requirements.lock.txt (exact pins).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code. .dockerignore keeps venv/, .git/, *.db, .env and caches out.
COPY . .

# Writable location for the SQLite database (./data is mounted by compose).
RUN mkdir -p /app/data /app/app/ml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/healthcheck" || exit 1

# Shell form: ${PORT:-8000} is expanded at container start (Render/Railway-style
# PaaS injects PORT; plain `docker run` falls back to 8000).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
