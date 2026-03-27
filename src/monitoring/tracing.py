"""
OpenTelemetry distributed tracing for the fraud investigation agent.

Setup
-----
Set OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://jaeger:4317) before starting the server.
If opentelemetry packages are not installed, all helpers are transparent no-ops.

Usage
-----
    from src.monitoring.tracing import start_span, setup_tracing

    setup_tracing(service_name="fraud-agent-api")

    with start_span("my_operation", attributes={"key": "value"}) as span:
        do_work()
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_tracer: Optional[Any] = None


def setup_tracing(
    service_name: str = "fraud-agent-api",
    otlp_endpoint: Optional[str] = None,
) -> None:
    """
    Initialise the global tracer.  Call once at application startup.

    Tries OTLP exporter first; falls back to console logging if unavailable.
    """
    global _tracer
    if not _OTEL_AVAILABLE:
        return

    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        except ImportError:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


@contextmanager
def start_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Context manager that wraps a code block in an OTel span.
    If tracing is not configured or unavailable, yields None transparently.
    """
    if not _OTEL_AVAILABLE or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(str(k), str(v))
        yield span


def instrument_fastapi(app: Any) -> None:
    """
    Auto-instrument a FastAPI application with OpenTelemetry.
    Call after app is created and tracing is set up.
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass
