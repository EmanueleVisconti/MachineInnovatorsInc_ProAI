import argparse
import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import f1_score
from src.utils.mlflow_utils import (
    get_production_model_uri,
    promote_to_stage,
    REGISTERED_NAME,
)

_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


def _normalize(label: str) -> str:
    if label.startswith("LABEL_"):
        return _LABEL_MAP.get(label, label)
    return label.lower()


def _predict_df(model, df: pd.DataFrame) -> list[str]:
    preds = []
    for text in df["text"].tolist():
        out = model.predict([text])[0]
        preds.append(_normalize(out["label"]))
    return preds


def evaluate_and_maybe_promote(
    new_model_uri: str, eval_csv: str, min_improvement: float = 0.0
) -> None:
    df = pd.read_csv(eval_csv)
    y_true = df["label"].astype(str).str.lower().tolist()

    # Valuta nuovo modello
    new_model = mlflow.pyfunc.load_model(new_model_uri)
    new_pred = _predict_df(new_model, df)
    new_f1 = f1_score(y_true, new_pred, average="macro")

    # Valuta production corrente (se esiste)
    prod_uri = get_production_model_uri(REGISTERED_NAME)
    if prod_uri:
        prod_model = mlflow.pyfunc.load_model(prod_uri)
        prod_pred = _predict_df(prod_model, df)
        prod_f1 = f1_score(y_true, prod_pred, average="macro")
    else:
        prod_f1 = -1.0  # forza promozione alla prima

    print({"new_f1": new_f1, "prod_f1": prod_f1})

    if new_f1 >= prod_f1 + min_improvement:
        # Estrai version da new_model_uri: runs:/.../model oppure models:/...
        client = mlflow.tracking.MlflowClient()
        # Recupera l'ultima versione registrata (quella appena loggata)
        versions = client.get_latest_versions(REGISTERED_NAME, stages=["None"]) or []
        if not versions:
            # fallback: prendi la più recente tra tutte
            allv = client.search_model_versions(f"name='{REGISTERED_NAME}'")
            versions = sorted(allv, key=lambda v: int(v.version), reverse=True)
        version = int(versions[0].version)
        promote_to_stage(REGISTERED_NAME, version, stage="Production")
        print(f"Promosso {REGISTERED_NAME} v{version} → Production")
    else:
        print("Nessuna promozione: nuovo modello non migliora.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--new_model_uri",
        required=True,
        help="es. runs:/<run_id>/model oppure models:/Sentiment/None",
    )
    ap.add_argument("--eval_csv", required=True, help="CSV con colonne: text,label")
    ap.add_argument("--min_improvement", type=float, default=0.0)
    args = ap.parse_args()
    evaluate_and_maybe_promote(args.new_model_uri, args.eval_csv, args.min_improvement)
