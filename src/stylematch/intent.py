"""Query understanding: a rule-based parser that turns a free-text query into an intent.

The intent holds hard filters (gender, price range) and soft preferences
(categories, colours), plus expansion terms for occasions. A query such as
"outfit for a birthday party" shares few words with product descriptions. The
occasion map adds the product words that such an occasion needs. The same intent
goes to every retriever, so the comparison between retrievers stays fair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import CATEGORY_KEYWORDS
from .text import tokenize

GENDER_WORDS = {
    "women": ("women", "woman", "womens", "ladies", "lady", "female", "her"),
    "men": ("men", "man", "mens", "gents", "male", "him"),
    "kids": ("kids", "kid", "child", "children"),
    "boys": ("boys", "boy"),
    "girls": ("girls", "girl"),
}
# a query gender -> the catalog genders that match it
GENDER_MATCH = {
    "women": ("women", "unisex"),
    "men": ("men", "unisex"),
    "boys": ("boys", "unisex kids"),
    "girls": ("girls", "unisex kids"),
    "kids": ("boys", "girls", "unisex kids"),
}
COLOURS = (
    "black", "white", "blue", "navy", "red", "green", "grey", "brown", "yellow", "pink", "beige", "purple",
    "orange", "maroon", "gold", "silver", "olive", "teal", "cream", "khaki", "multi",
)
OCCASIONS = {
    "party": ("dress", "party", "sequinned", "shimmer", "heels", "festive", "celebration", "blazer"),
    "birthday": ("dress", "party", "festive", "celebration"),
    "wedding": ("saree", "lehenga", "sherwani", "kurta", "ethnic", "festive", "embroidered"),
    "festival": ("kurta", "ethnic", "festive", "saree"),
    "office": ("shirt", "trousers", "formal", "blazer", "workwear"),
    "work": ("shirt", "trousers", "formal", "workwear"),
    "interview": ("shirt", "trousers", "formal", "blazer"),
    "gym": ("sports", "training", "trackpants", "tshirt", "running", "workout"),
    "workout": ("sports", "training", "trackpants", "running"),
    "running": ("running", "sports", "shoes", "sneaker"),
    "beach": ("shorts", "sandals", "summer", "linen", "sunglasses"),
    "winter": ("jacket", "sweater", "sweatshirt", "warm", "fleece"),
    "casual": ("tshirt", "jeans", "casual", "sneaker"),
    "travel": ("bag", "backpack", "trolley", "travel"),
}
CATEGORY_WORDS = {tok: cat for tok, cat in CATEGORY_KEYWORDS}

_NUMBER = r"(?:rs\.?|inr|₹)?\s*(\d[\d,]*)"
_UNDER = re.compile(rf"(?:under|below|less than|upto|up to|within|max(?:imum)?|<)\s*{_NUMBER}", re.I)
_OVER = re.compile(rf"(?:over|above|more than|at least|min(?:imum)?|>)\s*{_NUMBER}", re.I)
_BETWEEN = re.compile(rf"between\s*{_NUMBER}\s*(?:and|to|-)\s*{_NUMBER}", re.I)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


@dataclass
class Intent:
    query: str
    gender: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    categories: list[str] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    expansion: list[str] = field(default_factory=list)

    @property
    def search_text(self) -> str:
        """The query text plus the expansion terms."""
        return " ".join([self.query, *self.expansion])

    def allowed_genders(self) -> tuple[str, ...] | None:
        return GENDER_MATCH.get(self.gender) if self.gender else None

    def to_dict(self) -> dict:
        return {
            "gender": self.gender,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "categories": self.categories,
            "colours": self.colours,
            "occasions": self.occasions,
            "expansion": self.expansion,
        }


def parse_query(query: str, gender: str | None = None, min_price: float | None = None, max_price: float | None = None) -> Intent:
    """Rule-based intent. Explicit arguments override what the text says."""
    text = str(query).strip()
    if not text:
        raise ValueError("the query is empty")
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)
    intent = Intent(query=text)

    for canon in ("boys", "girls", "kids", "women", "men"):
        if any(w in GENDER_WORDS[canon] for w in words):
            intent.gender = canon
            break

    between = _BETWEEN.search(lowered)
    if between:
        low, high = sorted((_num(between.group(1)), _num(between.group(2))))
        intent.min_price, intent.max_price = low, high
    else:
        under, over = _UNDER.search(lowered), _OVER.search(lowered)
        if under:
            intent.max_price = _num(under.group(1))
        if over:
            intent.min_price = _num(over.group(1))

    tokens = tokenize(text)
    for tok in tokens:
        for keyword, category in CATEGORY_KEYWORDS:
            if tok.startswith(keyword) and category not in intent.categories:
                intent.categories.append(category)
                break
    intent.colours = [c for c in COLOURS if c in words]
    intent.occasions = [o for o in OCCASIONS if o in words]
    expansion: list[str] = []
    for occasion in intent.occasions:
        expansion += [w for w in OCCASIONS[occasion] if w not in expansion]
    intent.expansion = expansion

    if gender:
        intent.gender = gender.lower()
        if intent.gender not in GENDER_MATCH:
            raise ValueError(f"gender must be one of {sorted(GENDER_MATCH)}")
    if min_price is not None:
        intent.min_price = float(min_price)
    if max_price is not None:
        intent.max_price = float(max_price)
    if intent.min_price is not None and intent.max_price is not None and intent.min_price > intent.max_price:
        raise ValueError("min_price is above max_price")
    return intent
