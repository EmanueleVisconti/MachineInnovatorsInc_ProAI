# src/monitoring/drift_report.py
import argparse
import json
import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def _pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    return TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=False)


def _predict(pipe, s: pd.Series) -> list[str]:
    out = []
    for t in s.tolist():
        res = pipe(t, truncation=True)
        first = res[0] if isinstance(res, list) else res
        if isinstance(first, list):
            first = first[0]
        lab = first["label"]
        if isinstance(lab, str) and lab.startswith("LABEL_"):
            idx = int(lab.split("_")[-1])
            lab = LABELS.get(idx, lab)
        out.append(str(lab).lower())
    return out


def main(ref_csv: str, cur_csv: str, out_dir: str = "artifacts") -> int:
    os.makedirs(out_dir, exist_ok=True)
    ref = pd.read_csv(ref_csv).dropna(subset=["text"]).copy()
    cur = pd.read_csv(cur_csv).dropna(subset=["text"]).copy()

    # feature semplice + predizione categoriale
    ref["len"] = ref["text"].str.len()
    cur["len"] = cur["text"].str.len()

    pipe = _pipeline()
    ref["pred"] = _predict(pipe, ref["text"])
    cur["pred"] = _predict(pipe, cur["text"])

    # In Evidently 0.4.x, ColumnDriftMetric funziona anche su colonne categoriali
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="len"),
            ColumnDriftMetric(column_name="pred"),
        ]
    )
    report.run(reference_data=ref[["len", "pred"]], current_data=cur[["len", "pred"]])

    js = report.as_dict()
    with open(os.path.join(out_dir, "drift_report.json"), "w") as f:
        json.dump(js, f, indent=2)
    report.save_html(os.path.join(out_dir, "drift_report.html"))

    # Ritorna 0 se nessun drift “forte”, 1 se drift su una delle due colonne
    # La chiave nei dict di Evidently 0.4.x è "drift_detected"
    metrics = {m["metric"]: m for m in js.get("metrics", [])}

    # Ogni m ha "column_name": "len"/"pred"
    def drift_for(col):
        for m in js.get("metrics", []):
            res = m.get("result", {})
            if res.get("column_name") == col:
                return bool(res.get("drift_detected", False))
        return False

    len_drift = drift_for("len")
    pred_drift = drift_for("pred")
    return 1 if (len_drift or pred_drift) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    raise SystemExit(main(args.reference, args.current, args.out))
