import pytest
from src.serving import load_model


class _FakePipeTop1:
    def __call__(self, *args, **kwargs):
        # Forma: [ {"label": "LABEL_2", "score": 0.99} ]
        return [{"label": "LABEL_2", "score": 0.99}]


class _FakePipeTopK:
    def __call__(self, *args, **kwargs):
        # Forma: [ [ {"label": "LABEL_1", "score": 0.51}, {"label": "LABEL_2", "score": 0.49} ] ]
        return [
            [{"label": "LABEL_1", "score": 0.51}, {"label": "LABEL_2", "score": 0.49}]
        ]


@pytest.mark.parametrize(
    "fake_pipe, expected",
    [(_FakePipeTop1(), "positive"), (_FakePipeTopK(), "neutral")],
)
def test_predict_fn_handles_shapes(monkeypatch, fake_pipe, expected):
    # monkeypatch get_pipeline per non scaricare modelli reali
    monkeypatch.setattr(load_model, "get_pipeline", lambda: fake_pipe)
    label, score = load_model.predict_fn("text")
    assert label == expected
    assert isinstance(score, float) and 0.0 <= score <= 1.0
