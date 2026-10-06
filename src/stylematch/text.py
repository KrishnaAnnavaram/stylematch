"""Tokenizer shared by BM25, TF-IDF and the query parser."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to with your you our "
    "has have was were will one two".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def fold(token: str) -> str:
    """Light plural folding: dresses -> dress, shirts -> shirt, jeans stays jeans."""
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "jeans", "shorts", "trousers")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stopwords, with plural folding."""
    text = str(text).lower().replace("t-shirt", "tshirt").replace("t shirt", "tshirt")
    return [fold(t) for t in _TOKEN.findall(text) if t not in STOPWORDS]
