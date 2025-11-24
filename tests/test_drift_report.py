import os
import pandas as pd

from src.monitoring import drift_report


def _write(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return str(path)


def test_drift_detected_on_length_and_label(tmp_path):
    ref = _write(tmp_path, "ref.csv", [["short", "positive"], ["tiny", "negative"]])
    cur = _write(tmp_path, "cur.csv", [["this is a very very long off-topic text", "neutral"]] * 5)

    code = drift_report.main(ref, cur, out_dir=tmp_path)
    assert code == 1

    with open(tmp_path / "drift_report.json") as f:
        summary = f.read()
    assert "drift_flag" in summary


def test_no_drift_when_distributions_match(tmp_path):
    rows = [["good", "positive"], ["bad", "negative"], ["ok", "neutral"]]
    ref = _write(tmp_path, "ref.csv", rows)
    cur = _write(tmp_path, "cur.csv", rows)

    code = drift_report.main(ref, cur, out_dir=tmp_path)
    assert code == 0
