"""Offline evaluation against labelled queries (the gold set).

The relevance labels come from the gold set, never from another model. For each
query and each system the evaluation measures precision@k, recall@k, nDCG@k
(graded gains ``2^grade - 1``), MRR and the hit rate. Over all queries it gives
the catalog coverage and the brand diversity of the lists. A paired bootstrap
over queries gives a 95 % interval for the nDCG difference against the baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .index import Index
from .recommender import Recommender

GOLD_FIELDS = ("query_id", "query", "relevant")


def load_gold(path) -> list[dict]:
    """Read a gold set: JSONL (``relevant`` = {id: grade}) or CSV (query_id, query, product_id, relevance)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"product_id": str, "query_id": str})
        missing = {"query_id", "query", "product_id", "relevance"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = frame.dropna(subset=["relevance"])
        gold = []
        for (qid, query), group in frame.groupby(["query_id", "query"], sort=True):
            relevant = {str(p): int(r) for p, r in zip(group["product_id"], group["relevance"]) if int(r) > 0}
            gold.append({"query_id": str(qid), "query": str(query), "relevant": relevant})
        return gold
    gold = []
    with open(path, encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [f for f in GOLD_FIELDS if f not in row]
            if missing:
                raise ValueError(f"{path}:{n}: missing fields {missing}")
            row["relevant"] = {str(k): int(v) for k, v in row["relevant"].items() if int(v) > 0}
            gold.append(row)
    return gold


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    return sum(1 for pid in ranked[:k] if relevant.get(pid, 0) > 0) / k


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    total = sum(1 for g in relevant.values() if g > 0)
    return sum(1 for pid in ranked[:k] if relevant.get(pid, 0) > 0) / total if total else 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    gains = [2 ** relevant.get(pid, 0) - 1 for pid in ranked[:k]]
    ideal = sorted((2**g - 1 for g in relevant.values()), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0


def reciprocal_rank(ranked: list[str], relevant: dict[str, int]) -> float:
    for i, pid in enumerate(ranked, start=1):
        if relevant.get(pid, 0) > 0:
            return 1.0 / i
    return 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Mean of (a - b) and its 95 % percentile interval, resampling queries."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def default_systems(recommender: Recommender, k: int) -> dict[str, Callable[[str], list[str]]]:
    """The systems in the comparison. All except ``tfidf_unfiltered`` share the same filters."""

    def make(method: str, reranker: str, use_filters: bool = True):
        def run(query: str) -> list[str]:
            result = recommender.recommend(query, k=k, method=method, reranker=reranker, use_filters=use_filters)
            return [item["product_id"] for item in result.items]

        return run

    systems = {
        "tfidf_unfiltered": make("tfidf", "none", use_filters=False),
        "tfidf": make("tfidf", "none"),
        "bm25": make("bm25", "none"),
        "dense": make("dense", "none"),
        "hybrid": make("hybrid", "none"),
        "hybrid_attribute": make("hybrid", "attribute"),
    }
    if recommender.client is not None:
        systems["hybrid_llm"] = make("hybrid", "llm")
    return systems


def evaluate(index: Index, gold: list[dict], systems: dict[str, Callable[[str], list[str]]], k: int = 10, baseline: str = "tfidf", seed: int = 42) -> dict:
    rows = []
    shown: dict[str, set[str]] = {name: set() for name in systems}
    for item in gold:
        relevant = item["relevant"]
        for name, run in systems.items():
            ranked = run(item["query"])
            shown[name].update(ranked[:k])
            brands = index.products.set_index("product_id").loc[ranked[:k], "brand"] if ranked else pd.Series(dtype=str)
            rows.append(
                {
                    "query_id": item["query_id"],
                    "kind": item.get("kind", ""),
                    "system": name,
                    f"precision@{k}": precision_at_k(ranked, relevant, k),
                    f"recall@{k}": recall_at_k(ranked, relevant, k),
                    f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
                    "mrr": reciprocal_rank(ranked[:k], relevant),
                    "hit": float(any(relevant.get(p, 0) > 0 for p in ranked[:k])),
                    "brand_diversity": (brands.nunique() / len(brands)) if len(brands) else 0.0,
                    "returned": len(ranked[:k]),
                }
            )
    per_query = pd.DataFrame(rows)
    metric_cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "mrr", "hit", "brand_diversity"]
    summary = per_query.groupby("system", sort=False)[metric_cols].mean().reset_index()
    summary["coverage"] = summary["system"].map(lambda s: len(shown[s]) / len(index.products))

    comparisons = []
    if baseline in systems:
        base = per_query[per_query["system"] == baseline].set_index("query_id")[f"ndcg@{k}"]
        for name in systems:
            if name == baseline:
                continue
            other = per_query[per_query["system"] == name].set_index("query_id")[f"ndcg@{k}"].loc[base.index]
            mean, low, high = paired_bootstrap(other.to_numpy(), base.to_numpy(), seed=seed)
            comparisons.append({"system": name, "baseline": baseline, "ndcg_diff": mean, "ci_low": low, "ci_high": high})
    by_kind = per_query.groupby(["kind", "system"], sort=False)[f"ndcg@{k}"].mean().unstack(0).reset_index()
    return {"per_query": per_query, "summary": summary, "comparisons": pd.DataFrame(comparisons), "by_kind": by_kind, "k": k}


def make_pool(recommender: Recommender, queries: list[dict], depth: int = 20, seed: int = 42) -> pd.DataFrame:
    """A blind labelling sheet: the union of the top ``depth`` items of every system, shuffled."""
    rng = np.random.default_rng(seed)
    systems = default_systems(recommender, depth)
    rows = []
    for item in queries:
        pooled: list[str] = []
        for run in systems.values():
            pooled += [pid for pid in run(item["query"]) if pid not in pooled]
        rng.shuffle(pooled)
        for pid in pooled:
            product = recommender.index.products.iloc[recommender.index.position(pid)]
            rows.append(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "product_id": pid,
                    "name": product["name"],
                    "brand": product["brand"],
                    "price": product["price"],
                    "colour": product["colour"],
                    "relevance": "",
                }
            )
    return pd.DataFrame(rows)


def run_evaluation(index: Index, gold_path, settings: Settings | None = None, k: int = 10, client=None) -> dict:
    recommender = Recommender(index, settings or Settings(), client=client)
    return evaluate(index, load_gold(gold_path), default_systems(recommender, k), k=k, seed=(settings or Settings()).seed)
