from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Lazy singletons
_tokenizer = None
_model = None
_pipeline = None

def get_pipeline() -> TextClassificationPipeline:
    global _pipeline, _tokenizer, _model
    if _pipeline is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        # NOTE: do NOT set top_k=None here (can return a list depending on HF version)
        _pipeline = TextClassificationPipeline(
            model=_model,
            tokenizer=_tokenizer,
            return_all_scores=False,
        )
    return _pipeline


def _normalize_label(raw_label: str) -> str:
    if raw_label.startswith("LABEL_"):
        idx = int(raw_label.split("_")[-1])
        return _label_map.get(idx, raw_label)
    return raw_label.lower()


def predict_fn(text: str) -> tuple[str, float]:
    """Robusta alla variazione di output tra versioni di Transformers.

    Possibili forme di output per input singolo:
    - [ {"label": "LABEL_2", "score": 0.99} ]
    - [ [ {"label": "LABEL_2", "score": 0.99}, ... ] ]  (se top_k implicito)
    """
    pipe = get_pipeline()
    result = pipe(text, truncation=True)
    # Batched outer list
    first = result[0] if isinstance(result, list) else result
    # Se è una lista (top_k), prendi il top-1
    if isinstance(first, list):
        first = first[0]
    label = _normalize_label(first["label"])  # type: ignore[index]
    score = float(first["score"])             # type: ignore[index]
    return label, score