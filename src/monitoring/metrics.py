"""
Prometheus metrics for the fraud investigation agent.

Usage
-----
Import `record_*` helpers anywhere in the codebase.
Expose /metrics by adding `router` to the FastAPI app (see src/api/main.py).

If prometheus_client is not installed, all calls are silently no-ops (graceful degradation).
"""

from __future__ import annotations

import time
from typing import Any, Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric definitions (created only when prometheus_client is available)
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    _agent_requests_total = Counter(
        "agent_requests_total",
        "Total fraud investigation requests",
        ["endpoint", "request_type"],
    )
    _tool_calls_total = Counter(
        "tool_calls_total",
        "Total tool invocations by the agent",
        ["tool_name"],
    )
    _tool_errors_total = Counter(
        "tool_errors_total",
        "Total tool failures",
        ["tool_name"],
    )
    _partial_reports_total = Counter(
        "partial_reports_total",
        "Investigations that returned a degraded partial report",
        [],
    )
    _inference_latency_ms = Histogram(
        "inference_latency_ms",
        "End-to-end agent investigation latency in milliseconds",
        ["request_type"],
        buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
    )
    _fraud_probability_gauge = Histogram(
        "fraud_probability_distribution",
        "Distribution of fraud probability scores",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    _model_prediction_total = Counter(
        "model_prediction_total",
        "Fraud/legitimate predictions from the ML model",
        ["prediction"],   # "fraud" or "legit"
    )


# ---------------------------------------------------------------------------
# Public helpers (no-op when prometheus_client is absent)
# ---------------------------------------------------------------------------

def record_request(endpoint: str, request_type: str) -> None:
    if _PROMETHEUS_AVAILABLE:
        _agent_requests_total.labels(endpoint=endpoint, request_type=request_type).inc()


def record_tool_call(tool_name: str, success: bool) -> None:
    if _PROMETHEUS_AVAILABLE:
        _tool_calls_total.labels(tool_name=tool_name).inc()
        if not success:
            _tool_errors_total.labels(tool_name=tool_name).inc()


def record_latency(request_type: str, elapsed_ms: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        _inference_latency_ms.labels(request_type=request_type).observe(elapsed_ms)


def record_prediction(fraud_probability: float, prediction: int) -> None:
    if _PROMETHEUS_AVAILABLE:
        _fraud_probability_gauge.observe(fraud_probability)
        label = "fraud" if prediction == 1 else "legit"
        _model_prediction_total.labels(prediction=label).inc()


def record_partial_report() -> None:
    if _PROMETHEUS_AVAILABLE:
        _partial_reports_total.inc()


# ---------------------------------------------------------------------------
# FastAPI router for /metrics endpoint
# ---------------------------------------------------------------------------

def _build_metrics_router():
    """Return a FastAPI APIRouter that exposes the /metrics endpoint."""
    from fastapi import APIRouter
    from fastapi.responses import Response

    metrics_router = APIRouter(tags=["observability"])

    @metrics_router.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        if not _PROMETHEUS_AVAILABLE:
            return Response(
                content="# prometheus_client not installed\n",
                media_type="text/plain",
            )
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return metrics_router


try:
    metrics_router = _build_metrics_router()
except Exception:
    metrics_router = None
