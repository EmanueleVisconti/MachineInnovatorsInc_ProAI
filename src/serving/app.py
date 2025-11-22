from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time
from ..features.preprocess import normalize_text
from .load_model import predict_fn

app = FastAPI(title="Sentiment Service", version="0.1.0")

PRED_COUNT = Counter("predictions_total", "Total predictions", ["label"])
LATENCY = Histogram("inference_latency_seconds", "Latency seconds")


class Item(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Item):
    start = time.time()
    clean = normalize_text(item.text)
    label, score = predict_fn(clean)
    LATENCY.observe(time.time() - start)
    PRED_COUNT.labels(label).inc()
    return {"label": label, "score": score}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
