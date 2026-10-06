"""Re-rankers. They only reorder retrieved candidates. They cannot add a product.

* ``none``: keep the retrieval order.
* ``attribute``: add a bonus for the category, colour and occasion words of the intent.
* ``llm``: a chat model orders the candidate IDs and gives a short reason for each.
  ``validate_ranking`` removes every ID that is not a candidate, removes repeats
  and appends the candidates that the model left out, in retrieval order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from .intent import Intent
from .llm import ChatClient
from .retrieve import Candidate

RERANKERS = ("none", "attribute", "llm")
CATEGORY_BONUS, COLOUR_BONUS, OCCASION_BONUS = 1.0, 0.5, 0.3


@dataclass
class Ranked:
    product_ids: list[str]
    reasons: dict[str, str] = field(default_factory=dict)
    dropped_ids: list[str] = field(default_factory=list)
    fallback: bool = False


def validate_ranking(proposed: list[str], candidates: list[str]) -> tuple[list[str], list[str]]:
    """Keep only candidate IDs, once each. Append the missing candidates. Return (order, dropped)."""
    allowed = set(candidates)
    order, seen, dropped = [], set(), []
    for pid in proposed:
        pid = str(pid)
        if pid not in allowed:
            dropped.append(pid)
            continue
        if pid not in seen:
            order.append(pid)
            seen.add(pid)
    order += [pid for pid in candidates if pid not in seen]
    return order, dropped


class Reranker:
    name = "none"

    def rerank(self, intent: Intent, candidates: list[Candidate], products: pd.DataFrame) -> Ranked:
        return Ranked([c.product_id for c in candidates])


class AttributeReranker(Reranker):
    name = "attribute"

    def rerank(self, intent, candidates, products):
        if not candidates:
            return Ranked([])
        top = max(c.score for c in candidates) or 1.0
        scored = []
        for position, cand in enumerate(candidates):
            row = products.iloc[cand.row]
            bonus = 0.0
            if intent.categories and row["category"] in intent.categories:
                bonus += CATEGORY_BONUS
            if intent.colours and str(row["colour"]) in intent.colours:
                bonus += COLOUR_BONUS
            text = f"{row['name']} {row['description']}".lower()
            if intent.expansion and any(term in text for term in intent.expansion):
                bonus += OCCASION_BONUS
            scored.append((cand.score / top + bonus, -position, cand.product_id))
        scored.sort(reverse=True)
        return Ranked([pid for _, _, pid in scored])


SYSTEM_PROMPT = (
    "You re-rank fashion products for a shopper. Use only the candidate products given. "
    "Never invent a product or an ID. Reply with JSON: "
    '{"ranking": [{"id": "<candidate id>", "reason": "<max 25 words>"}]} in order of fit.'
)


class LLMReranker(Reranker):
    name = "llm"

    def __init__(self, client: ChatClient, max_candidates: int = 20):
        self.client = client
        self.max_candidates = max_candidates

    def _messages(self, intent: Intent, candidates: list[Candidate], products: pd.DataFrame) -> list[dict]:
        lines = []
        for cand in candidates[: self.max_candidates]:
            row = products.iloc[cand.row]
            lines.append(
                json.dumps(
                    {"id": cand.product_id, "name": row["name"], "brand": row["brand"], "gender": row["gender"],
                     "category": row["category"], "colour": row["colour"], "price": float(row["price"])}
                )
            )
        user = f"Shopper request: {intent.query}\nParsed intent: {json.dumps(intent.to_dict())}\nCandidates:\n" + "\n".join(lines)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def rerank(self, intent, candidates, products):
        ids = [c.product_id for c in candidates]
        head = ids[: self.max_candidates]
        try:
            reply = json.loads(self.client.complete(self._messages(intent, candidates, products)))
            items = reply["ranking"]
            proposed = [str(item["id"]) for item in items]
            reasons = {str(item["id"]): str(item.get("reason", ""))[:300] for item in items}
        except (ValueError, KeyError, TypeError) as exc:
            return Ranked(ids, fallback=True, reasons={"_error": f"LLM reply not usable: {type(exc).__name__}"})
        order, dropped = validate_ranking(proposed, head)
        valid_reasons = {pid: reasons[pid] for pid in order if pid in reasons and reasons[pid]}
        return Ranked(order + ids[self.max_candidates :], reasons=valid_reasons, dropped_ids=dropped)


def make_reranker(name: str, client: ChatClient | None = None) -> Reranker:
    if name == "none":
        return Reranker()
    if name == "attribute":
        return AttributeReranker()
    if name == "llm":
        if client is None:
            raise ValueError("the llm reranker needs a configured LLM (STYLEMATCH_LLM_API_KEY)")
        return LLMReranker(client)
    raise ValueError(f"reranker must be one of {RERANKERS}")
