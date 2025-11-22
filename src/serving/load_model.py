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
        _pipeline = TextClassificationPipeline(
            model=_model,
            tokenizer=_tokenizer,
            return_all_scores=False,
            top_k=None,
        )
    return _pipeline


def predict_fn(text: str) -> tuple[str, float]:
    pipe = get_pipeline()
    out = pipe(text, truncation=True)[0]
    # HF pipeline with this model returns label indices like 'LABEL_0' or human labels depending on config
    label = out["label"]
    score = float(out["score"])

    # Normalize label to {negative, neutral, positive}
    if label.startswith("LABEL_"):
        idx = int(label.split("_")[-1])
        norm_label = _label_map.get(idx, label)
    else:
        norm_label = label.lower()
    return norm_label, score