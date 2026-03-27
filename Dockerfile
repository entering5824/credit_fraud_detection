# ──────────────────────────────────────────────────────────────────────────────
# Fraud Detection AI Agent – production API image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---------- system dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies ----------
COPY requirements.txt requirements-extra.txt* ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install prometheus-client opentelemetry-sdk \
       opentelemetry-exporter-otlp-proto-grpc \
       opentelemetry-instrumentation-fastapi \
       httpx

# ---------- source code ----------
COPY . .

# ---------- non-root user ----------
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ---------- model directory (mounted at runtime or via docker-compose volume) ----------
RUN mkdir -p /app/models /app/results/monitoring /app/data/raw

EXPOSE 8000

# Warm-start SHAP explainer at process start if model files are present
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
