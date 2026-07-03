# Project Sentinel — single-image demo (FastAPI serves the built React SPA).
# Build context = project_sentinel/. Local:  docker build -t sentinel .
#                                     run:    docker run -p 7860:7860 sentinel
# Hugging Face Docker Spaces run this automatically on push (app_port 7860).

# ---- Stage 1: build the React frontend -> /fe/dist ----
FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: python runtime serving API + built SPA ----
FROM python:3.12-slim
# libgomp1: OpenMP runtime that LightGBM/XGBoost link against at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# uv: fast, lockfile-faithful installs (reproducible from uv.lock).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
# deps first (cached layer); --no-install-project: it's an app, not a package.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project
# app code + artifacts (test.parquet is git-ignored; copied explicitly)
COPY src/ ./src/
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/processed/test.parquet ./data/processed/test.parquet
COPY bench_latency.py ./
COPY --from=frontend /fe/dist ./frontend/dist
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
