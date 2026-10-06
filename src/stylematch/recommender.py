"""The recommender facade: intent -> filtered retrieval -> re-rank -> explanation.

Every item in a result is a row of the catalog, found by its ID. The result
shows the retrieval score and the reason. It never shows a made-up rating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .index import Index
from .intent import Intent, parse_query
from .llm import ChatClient, make_client
from .rerank import make_reranker
from .retrieve import retrieve


@dataclass
class Recommendation:
    intent: Intent
    method: str
    reranker: str
    items: list[dict]
    pool_size: int
    filters: list[str]
    notes: list[str] = field(default_factory=list)


def template_reason(intent: Intent, row) -> str:
    """A reason from the facts of the product that match the intent."""
    facts = []
    if intent.gender:
        facts.append(f"for {row['gender']}")
    if intent.categories and row["category"] in intent.categories:
        facts.append(f"category {row['category']}")
    if intent.colours and row["colour"] in intent.colours:
        facts.append(f"colour {row['colour']}")
    if intent.max_price is not None or intent.min_price is not None:
        facts.append(f"price {row['price']:,.0f} inside the budget")
    text = f"{row['name']} {row['description']}".lower()
    hits = [term for term in intent.expansion if term in text]
    if hits:
        facts.append("occasion words: " + ", ".join(hits[:3]))
    return "Matches " + "; ".join(facts) if facts else "Closest text match in the filtered catalog"


class Recommender:
    def __init__(self, index: Index, settings: Settings | None = None, client: ChatClient | None = None):
        self.index = index
        self.settings = settings or Settings()
        self.client = client if client is not None else make_client(self.settings)

    def recommend(
        self,
        query: str,
        k: int | None = None,
        method: str = "hybrid",
        reranker: str = "attribute",
        gender: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        use_filters: bool = True,
    ) -> Recommendation:
        k = k or self.settings.top_k
        intent = parse_query(query, gender=gender, min_price=min_price, max_price=max_price)
        found = retrieve(self.index, intent, method=method, k=max(self.settings.candidates, k), use_filters=use_filters)
        notes = []
        if not found.candidates:
            notes.append("no product passes the filters: change the budget or the gender")
            return Recommendation(intent, method, reranker, [], found.pool_size, found.filters, notes)

        ranked = make_reranker(reranker, self.client).rerank(intent, found.candidates, self.index.products)
        if ranked.fallback:
            notes.append(ranked.reasons.get("_error", "re-ranker failed: retrieval order used"))
        if ranked.dropped_ids:
            notes.append(f"{len(ranked.dropped_ids)} ID(s) from the re-ranker are not candidates and were removed")

        by_id = {c.product_id: c for c in found.candidates}
        items = []
        for rank, pid in enumerate(ranked.product_ids[:k], start=1):
            if not self.index.has(pid):  # defence in depth: never show an unknown product
                continue
            row = self.index.products.iloc[self.index.position(pid)]
            reason = ranked.reasons.get(pid) or template_reason(intent, row)
            items.append(
                {
                    "rank": rank,
                    "product_id": pid,
                    "name": row["name"],
                    "brand": row["brand"],
                    "gender": row["gender"],
                    "category": row["category"],
                    "colour": row["colour"],
                    "price": float(row["price"]),
                    "retrieval_score": round(by_id[pid].score, 6),
                    "reason": reason,
                }
            )
        return Recommendation(intent, method, reranker, items, found.pool_size, found.filters, notes)
