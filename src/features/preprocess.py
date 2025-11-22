import re

_url = re.compile(r"https?://\S+")
_mention = re.compile(r"@[A-Za-z0-9_]+")
_hashtag = re.compile(r"#[\w_]+")
_whitespace = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Sostituzioni leggere e conservative per social text."""
    t = text.strip()
    t = _url.sub("<URL>", t)
    t = _mention.sub("<USER>", t)
    t = _hashtag.sub(lambda m: m.group(0)[1:], t)  # drop '#'
    t = _whitespace.sub(" ", t)
    return t
