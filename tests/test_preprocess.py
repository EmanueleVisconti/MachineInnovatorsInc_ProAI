from src.features.preprocess import normalize_text


def test_normalize_text_basic():
    s = "Hey @user check https://example.com #GreatDay"
    out = normalize_text(s)
    assert "<USER>" in out
    assert "<URL>" in out
    assert "#" not in out
