import argparse
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from src.utils.mlflow_utils import get_or_create_experiment, REGISTERED_NAME

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class HFTextClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.pipe = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer, return_all_scores=False
        )

    def predict(self, context, model_input):
        outputs = []
        for text in model_input:
            res = self.pipe(text, truncation=True)
            first = res[0] if isinstance(res, list) else res
            if isinstance(first, list):
                first = first[0]
            outputs.append({"label": first["label"], "score": float(first["score"])})
        return outputs


def _train_sklearn_model(csv_path: str):
    df = pd.read_csv(csv_path)
    if not set(["text", "label"]).issubset(df.columns):
        raise ValueError("train_csv must contain 'text' and 'label' columns")
    texts = df["text"].astype(str).tolist()
    y = df["label"].astype(str).str.lower().tolist()
    vec = TfidfVectorizer(max_features=2048)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    metrics = {
        "train_size": len(df),
        "classes": len(set(y)),
    }
    return clf, vec, metrics


def main(experiment: str = "sentiment", train_csv: str | None = None) -> int:
    exp_id = get_or_create_experiment(experiment)
    mlflow.set_experiment(experiment)
    # If a train CSV is given, run the small training loop (sklearn)
    sklearn_model = None
    vectorizer = None
    metrics = {}
    if train_csv:
        sklearn_model, vectorizer, metrics = _train_sklearn_model(train_csv)

    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_param("base_model", MODEL_ID)
        if train_csv:
            mlflow.log_param("train_csv", train_csv)
            # registra la versione sklearn per testing e debug
            if sklearn_model:
                mlflow.sklearn.log_model(sklearn_model, artifact_path="sklearn_model")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HFTextClassifier(),
            registered_model_name=REGISTERED_NAME,
        )
        # Log some metrics discovered in the CSV
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        print(f"Run logged: {run.info.run_id}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="sentiment")
    parser.add_argument("--train_csv", default=None, help="CSV per training con colonne: text,label")
    args = parser.parse_args()
    raise SystemExit(main(args.experiment, args.train_csv))
