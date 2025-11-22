# src/monitoring/drift_report.py
import argparse, json, os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, CategoricalDriftMetric

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
    ref = pd.read_csv(ref_csv)
    cur = pd.read_csv(cur_csv)
    ref = ref.dropna(subset=["text"]).copy()
    cur = cur.dropna(subset=["text"]).copy()

    ref["len"], cur["len"] = ref.text.str.len(), cur.text.str.len()

    pipe = _pipeline()
    ref["pred"], cur["pred"] = _predict(pipe, ref.text), _predict(pipe, cur.text)

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="len"),
            CategoricalDriftMetric(column_name="pred"),
        ]
    )
    report.run(reference_data=ref[["len", "pred"]], current_data=cur[["len", "pred"]])

    js = report.as_dict()
    with open(os.path.join(out_dir, "drift_report.json"), "w") as f:
        json.dump(js, f, indent=2)
    report.save_html(os.path.join(out_dir, "drift_report.html"))

    # Ritorna 0 se nessun drift forte, 1 se drift
    metrics = {m["metric"]: m for m in js.get("metrics", [])}
    len_drift = (
        metrics.get("ColumnDriftMetric", {})
        .get("result", {})
        .get("drift_detected", False)
    )
    cat_drift = (
        metrics.get("CategoricalDriftMetric", {})
        .get("result", {})
        .get("drift_detected", False)
    )
    return 1 if (len_drift or cat_drift) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    code = main(args.reference, args.current, args.out)
    raise SystemExit(code)
