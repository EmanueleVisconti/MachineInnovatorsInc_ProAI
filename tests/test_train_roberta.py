import pandas as pd
import io
import pytest

from src.models import train_roberta


def _write_csv(tmp_path, rows, filename="train.csv"):
    path = tmp_path / filename
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return str(path)


def test__train_sklearn_model_invalid_columns(tmp_path):
    # Missing 'label' column
    path = _write_csv(tmp_path, [["hello", "x"]], filename="bad.csv")
    # Tamper columns: remove label
    df = pd.read_csv(path)
    df = df.drop(columns=["label"])
    df.to_csv(path, index=False)

    with pytest.raises(ValueError):
        train_roberta._train_sklearn_model(path)


def test__train_sklearn_model_valid(tmp_path):
    rows = [["I love this", "positive"], ["I hate this", "negative"], ["meh", "neutral"]]
    path = _write_csv(tmp_path, rows)
    clf, vec, metrics = train_roberta._train_sklearn_model(path)
    assert hasattr(clf, "predict")
    assert hasattr(vec, "transform")
    assert metrics["train_size"] == 3
    assert metrics["classes"] == 3


def test_main_with_train_csv_logs(monkeypatch, tmp_path):
    rows = [["I love this", "positive"], ["I hate this", "negative"], ["meh", "neutral"]]
    path = _write_csv(tmp_path, rows)

    called = {}

    # Patch utils.get_or_create_experiment to avoid MLflow server calls
    monkeypatch.setattr(
        "src.utils.mlflow_utils.get_or_create_experiment", lambda name: "fake_exp_id"
    )

    # Patch mlflow functions to avoid side effects and capture calls
    import mlflow

    def fake_log_model(*args, **kwargs):
        called.setdefault("log_model", 0)
        called["log_model"] += 1

    def fake_log_param(*args, **kwargs):
        called.setdefault("log_param", []).append((args, kwargs))

    def fake_log_metric(*args, **kwargs):
        called.setdefault("log_metric", []).append((args, kwargs))

    class DummyRun:
        class Info:
            run_id = "rid"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", fake_log_model)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", fake_log_model)
    monkeypatch.setattr(mlflow, "log_param", fake_log_param)
    monkeypatch.setattr(mlflow, "log_metric", fake_log_metric)

    # Run main with train_csv -- should return 0 and invoke logging
    ret = train_roberta.main("sentiment", train_csv=path)
    assert ret == 0
    assert called.get("log_model", 0) >= 1
    assert "train_csv" in [p[0][0] for p in called.get("log_param", [])]
    assert any(k in [m[0][0] for m in called.get("log_metric", []) if m] for k in ["train_size"]) or called.get(
        "log_metric"
    )
import os
import pandas as pd
from types import SimpleNamespace

import mlflow

from src.models import train_roberta


class DummyRun:
    def __init__(self):
        self.info = SimpleNamespace(run_id="dummy")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _write_csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(path, index=False)
    return str(path)


def test_main_with_train_csv(monkeypatch, tmp_path):
    csv = _write_csv(tmp_path, "train.csv", [["hello there", "positive"], ["this is bad", "negative"]])
    calls = {"pyfunc": 0, "sklearn": 0}

    monkeypatch.setattr(train_roberta, "get_or_create_experiment", lambda x: "expid")
    monkeypatch.setattr(mlflow, "set_experiment", lambda x: None)
    monkeypatch.setattr(mlflow, "start_run", lambda experiment_id=None: DummyRun())
    monkeypatch.setattr(mlflow.pyfunc, "log_model", lambda **kwargs: calls.__setitem__("pyfunc", calls["pyfunc"] + 1))
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda **kwargs: calls.__setitem__("sklearn", calls["sklearn"] + 1))
    monkeypatch.setattr(mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kwargs: None)

    code = train_roberta.main(experiment="sentiment_test", train_csv=csv)
    assert code == 0
    assert calls["pyfunc"] == 1
    assert calls["sklearn"] == 1


def test_main_invalid_csv(monkeypatch, tmp_path):
    # write CSV without required columns
    path = tmp_path / "bad.csv"
    pd.DataFrame([[1, 2], [3, 4]]).to_csv(path, index=False)
    monkeypatch.setattr(train_roberta, "get_or_create_experiment", lambda x: "expid")
    try:
        train_roberta.main(experiment="sentiment_test", train_csv=str(path))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid CSV")


def test_main_without_train_csv(monkeypatch):
    calls = {"pyfunc": 0}
    monkeypatch.setattr(train_roberta, "get_or_create_experiment", lambda x: "expid")
    monkeypatch.setattr(mlflow, "set_experiment", lambda x: None)
    monkeypatch.setattr(mlflow, "start_run", lambda experiment_id=None: DummyRun())
    monkeypatch.setattr(mlflow.pyfunc, "log_model", lambda **kwargs: calls.__setitem__("pyfunc", calls["pyfunc"] + 1))
    monkeypatch.setattr(mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kwargs: None)

    code = train_roberta.main(experiment="sentiment_test", train_csv=None)
    assert code == 0
    assert calls["pyfunc"] == 1
