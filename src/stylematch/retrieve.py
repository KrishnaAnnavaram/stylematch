"""Retrievers over one shared candidate pool.

Every method gets the same intent and the same hard filters (gender and price),
so the comparison is fair:

* ``tfidf``: cosine similarity of TF-IDF vectors (the baseline).
* ``bm25``: Okapi BM25.
* ``dense``: cosine similarity of embedder vectors.
* ``hybrid``: reciprocal-rank fusion (RRF) of the BM25 list and the dense list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .index import Index
from .intent import Intent

METHODS = ("tfidf", "bm25", "dense", "hybrid")
RRF_K = 60


@dataclass
class Candidate:
    product_id: str
    row: int
    score: float
    method: str


@dataclass
class Retrieval:
    candidates: list[Candidate]
    pool_size: int
    filters: list[str] = field(default_factory=list)


def filter_rows(products: pd.DataFrame, intent: Intent) -> tuple[np.ndarray, list[str]]:
    """Rows that pass the hard filters, and a description of each filter."""
    mask = np.ones(len(products), dtype=bool)
    applied = []
    allowed = intent.allowed_genders()
    if allowed:
        mask &= products["gender"].isin(allowed).to_numpy()
        applied.append(f"gender in {list(allowed)}")
    if intent.min_price is not None:
        mask &= (products["price"] >= intent.min_price).to_numpy()
        applied.append(f"price >= {intent.min_price:g}")
    if intent.max_price is not None:
        mask &= (products["price"] <= intent.max_price).to_numpy()
        applied.append(f"price <= {intent.max_price:g}")
    return np.flatnonzero(mask), applied


def _top(rows: np.ndarray, scores: np.ndarray, k: int, positive_only: bool) -> list[tuple[int, float]]:
    order = np.argsort(-scores, kind="stable")
    out = []
    for i in order[:k]:
        if positive_only and scores[i] <= 0:
            break
        out.append((int(rows[i]), float(scores[i])))
    return out


def _scores(index: Index, text: str, rows: np.ndarray, method: str) -> np.ndarray:
    if method == "tfidf":
        query = index.tfidf.transform([text])
        return np.asarray((index.tfidf_matrix[rows] @ query.T).todense()).ravel()
    if method == "bm25":
        return index.bm25.scores(text, rows)
    if method == "dense":
        return index.vectors[rows] @ index.embedder.embed([text])[0]
    raise ValueError(f"unknown method {method!r}")


def retrieve(index: Index, intent: Intent, method: str = "hybrid", k: int = 50, use_filters: bool = True) -> Retrieval:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if use_filters:
        rows, applied = filter_rows(index.products, intent)
    else:
        rows, applied = np.arange(len(index.products)), []
    if rows.size == 0:
        return Retrieval([], 0, applied)

    text = intent.search_text
    ids = index.products["product_id"].astype(str).to_numpy()
    if method in ("tfidf", "bm25", "dense"):
        top = _top(rows, _scores(index, text, rows, method), k, positive_only=method != "dense")
        return Retrieval([Candidate(ids[r], r, s, method) for r, s in top], int(rows.size), applied)

    fused: dict[int, float] = {}
    for part in ("bm25", "dense"):
        ranked = _top(rows, _scores(index, text, rows, part), k, positive_only=part == "bm25")
        for rank, (row, _) in enumerate(ranked, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + rank)
    order = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
    return Retrieval([Candidate(ids[r], r, s, "hybrid") for r, s in order], int(rows.size), applied)
