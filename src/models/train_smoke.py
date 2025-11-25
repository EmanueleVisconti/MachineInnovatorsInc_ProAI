"""
Simple smoke training script for fast dev/demo runs.
This script trains a tiny sklearn sentinel model on a head of the csv and registers
it under a dev model name (REGISTERED_NAME + '-dev'), so it doesn't affect
production models.

Usage:
    python -m src.models.train_smoke --n_samples 32 --train_csv /path/to/csv

The script is intentionally lightweight and avoids heavy imports (transformers)
so it can run in limited resource environments.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils.mlflow_utils import get_or_create_experiment, REGISTERED_NAME


def _train_sklearn_model(csv_path: str) -> Tuple[object, object, dict]:
    df = pd.read_csv(csv_path)
    if not set(["text", "label"]).issubset(df.columns):
        raise ValueError("train_csv must contain 'text' and 'label' columns")
    texts = df["text"].astype(str).tolist()
    y = df["label"].astype(str).str.lower().tolist()
    vec = TfidfVectorizer(max_features=2048)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    metrics = {"train_size": len(df), "classes": len(set(y))}
    return clf, vec, metrics


def main(experiment: str = "sentiment", train_csv: str | None = None, n_samples: int = 1, dev_suffix: str = "-dev") -> int:
    exp_id = get_or_create_experiment(experiment)
    mlflow.set_experiment(experiment)

    DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
    HOLDOUT = os.path.join(DATA_DIR, "holdout.csv")
    CUR = os.path.join(DATA_DIR, "raw", "current.csv")

    if not train_csv:
        # prefer current if present, otherwise fallback to holdout
        if os.path.exists(CUR):
            train_csv = CUR
        elif os.path.exists(HOLDOUT):
            train_csv = HOLDOUT
        else:
            raise FileNotFoundError("No train_csv specified and no current/holdout CSV found")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        df = pd.read_csv(train_csv)
        if n_samples and n_samples > 0:
            df = df.head(n_samples)
        df.to_csv(tmp.name, index=False)

        sklearn_model, vectorizer, metrics = _train_sklearn_model(tmp.name)

        target_model_name = f"{REGISTERED_NAME}{dev_suffix}"
        with mlflow.start_run() as run:
            mlflow.log_param("train_csv", train_csv)
            mlflow.log_param("n_samples", n_samples)
            mlflow.sklearn.log_model(
                sklearn_model,
                artifact_path="sklearn_model",
                registered_model_name=target_model_name,
            )
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            print(f"Run logged: {run.info.run_id}")
            print(f"Registered model: {target_model_name}")
            print(f"Used train csv: {tmp.name}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="sentiment")
    parser.add_argument("--train_csv", default=None, help="Path to CSV with columns text,label")
    parser.add_argument("--n_samples", default=1, type=int, help="Number of rows to sample from head")
    parser.add_argument("--dev_suffix", default="-dev", help="Suffix to append to REGISTERED_NAME for dev models")
    args = parser.parse_args()
    raise SystemExit(main(args.experiment, args.train_csv, args.n_samples, args.dev_suffix))
