# src/serving/metrics.py
import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Request, Response
from fastapi.routing import APIRouter

REQUESTS = Counter("api_requests_total", "Total API requests", ["path", "method"])
RESPONSES = Counter(
    "api_responses_total", "Total API responses", ["path", "method", "status"]
)
LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["path", "method"]
)
INFLIGHT = Gauge("api_inflight_requests", "Inflight requests")

router = APIRouter()


@router.get("/metrics")
def metrics_endpoint():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        REQUESTS.labels(path=path, method=method).inc()
        start = time.perf_counter()
        INFLIGHT.inc()

        status_holder = {"status": "200"}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = str(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            LATENCY.labels(path=path, method=method).observe(
                time.perf_counter() - start
            )
            RESPONSES.labels(
                path=path, method=method, status=status_holder["status"]
            ).inc()
            INFLIGHT.dec()
