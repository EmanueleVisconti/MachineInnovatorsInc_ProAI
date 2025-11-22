import os
import mlflow
import mlflow.pyfunc
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_label_map = {0: "negative", 1: "neutral", 2: "positive"}

MODEL_URI = os.getenv("MODEL_URI")  # es. models:/Sentiment/Production
STRICT_REGISTRY = os.getenv("STRICT_REGISTRY", "0") == "1"

_tokenizer = _model = _pipeline = _mlflow_model = None


def _normalize_label(raw_label: str) -> str:
    if isinstance(raw_label, str) and raw_label.startswith("LABEL_"):
        idx = int(raw_label.split("_")[-1])
        return _label_map.get(idx, raw_label)
    return str(raw_label).lower()


def _get_hf_pipeline():
    global _pipeline, _tokenizer, _model
    if _pipeline is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        _pipeline = TextClassificationPipeline(
            model=_model, tokenizer=_tokenizer, return_all_scores=False
        )
    return _pipeline


def _try_get_mlflow_model():
    global _mlflow_model
    if MODEL_URI and _mlflow_model is None:
        try:
            _mlflow_model = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as e:
            # niente Production o modello assente → fallback
            if STRICT_REGISTRY:
                raise
            _mlflow_model = None
    return _mlflow_model


def predict_fn(text: str):
    m = _try_get_mlflow_model()
    if m is not None:
        out = m.predict([text])[0]
        label = _normalize_label(out["label"])  # type: ignore[index]
        score = float(out["score"])  # type: ignore[index]
        return label, score
    # fallback HF
    pipe = _get_hf_pipeline()
    res = pipe(text, truncation=True)
    first = res[0] if isinstance(res, list) else res
    if isinstance(first, list):
        first = first[0]
    label = _normalize_label(first["label"])  # type: ignore[index]
    score = float(first["score"])  # type: ignore[index]
    return label, score
