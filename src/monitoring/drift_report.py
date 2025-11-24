# src/monitoring/drift_report.py
"""Heuristic drift detection that works offline and surfaces a clear flag.

- If a `label` column is present we use it directly to compare class
  distributions. Otherwise we fall back to the serving pipeline
  (`src.serving.load_model.get_pipeline`) which includes a stub for offline
  environments.
- Drift is flagged when either the text-length median shifts materially or the
  class distribution drifts (total variation distance) beyond a configurable
  threshold. Both metrics are exported to JSON for inspection.
"""

import argparse
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd

from src.serving.load_model import get_pipeline, _normalize_label  # type: ignore

LENGTH_SHIFT_THRESHOLD = 0.35  # 35% median-length shift
CLASS_DRIFT_THRESHOLD = 0.25  # TV distance on label distribution


def _class_distribution(labels: Iterable[str]) -> dict[str, float]:
    dist: dict[str, float] = {}
    total = 0
    for lab in labels:
        lab_norm = str(lab).strip().lower()
        dist[lab_norm] = dist.get(lab_norm, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in dist.items()}


def _tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _predict_labels(texts: pd.Series) -> list[str]:
    """Predict sentiment labels using the serving pipeline or its stub."""

    pipe = get_pipeline()
    outputs = []
    for txt in texts.tolist():
        res = pipe(txt, truncation=True)
        first = res[0] if isinstance(res, list) else res
        if isinstance(first, list):
            first = first[0]
        outputs.append(_normalize_label(first["label"]))  # type: ignore[index]
    return outputs


def _pick_labels(df: pd.DataFrame) -> list[str]:
    if "label" in df.columns:
        return df["label"].astype(str).str.lower().tolist()
    return _predict_labels(df["text"])


def main(ref_csv: str, cur_csv: str, out_dir: str = "artifacts") -> int:
    os.makedirs(out_dir, exist_ok=True)
    ref = pd.read_csv(ref_csv).dropna(subset=["text"]).copy()
    cur = pd.read_csv(cur_csv).dropna(subset=["text"]).copy()

    ref_len = ref["text"].str.len()
    cur_len = cur["text"].str.len()

    ref_labels = _pick_labels(ref)
    cur_labels = _pick_labels(cur)

    len_shift = abs(np.median(cur_len) - np.median(ref_len)) / max(
        np.median(ref_len), 1
    )
    dist_ref = _class_distribution(ref_labels)
    dist_cur = _class_distribution(cur_labels)
    cls_tv = _tv_distance(dist_ref, dist_cur)

    drift_flag = int(
        (len_shift >= LENGTH_SHIFT_THRESHOLD) or (cls_tv >= CLASS_DRIFT_THRESHOLD)
    )

    summary = {
        "length_median_reference": float(np.median(ref_len)),
        "length_median_current": float(np.median(cur_len)),
        "length_shift_ratio": float(len_shift),
        "class_distribution_reference": dist_ref,
        "class_distribution_current": dist_cur,
        "class_tv_distance": float(cls_tv),
        "drift_flag": drift_flag,
        "length_shift_threshold": LENGTH_SHIFT_THRESHOLD,
        "class_drift_threshold": CLASS_DRIFT_THRESHOLD,
    }

    with open(os.path.join(out_dir, "drift_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # HTML placeholder to avoid breaking viewers expecting a file
    html_path = os.path.join(out_dir, "drift_report.html")
    with open(html_path, "w") as f:
        f.write("<html><body><pre>{}</pre></body></html>".format(
            json.dumps(summary, indent=2)
        ))

    return drift_flag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    raise SystemExit(main(args.reference, args.current, args.out))
