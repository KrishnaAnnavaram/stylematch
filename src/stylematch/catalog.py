"""Catalog loading: a column contract, normalization and derived attributes.

The loader accepts the headers of the public fashion catalog (``ProductID``,
``ProductName``, ``ProductBrand``, ``Gender``, ``Price (INR)``, ``Description``,
``PrimaryColor``) and the snake_case names of the synthetic catalog.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .text import tokenize


class CatalogError(ValueError):
    """The catalog does not match its column contract."""


COLUMNS = {
    "product_id": ("productid", "id", "product_id"),
    "name": ("productname", "name", "title"),
    "brand": ("productbrand", "brand"),
    "gender": ("gender",),
    "price": ("priceinr", "price"),
    "description": ("description", "desc"),
    "colour": ("primarycolor", "primarycolour", "colour", "color"),
}
REQUIRED = ("product_id", "name", "gender", "price")

GENDERS = ("women", "men", "unisex", "boys", "girls", "unisex kids")
GENDER_ALIASES = {"unisexkids": "unisex kids", "kids": "unisex kids", "female": "women", "male": "men"}

# keyword in product name -> category; the first match wins, so put specific names first
CATEGORY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("tshirt", "tshirt"), ("kurta", "kurta"), ("saree", "saree"), ("lehenga", "lehenga"), ("sherwani", "sherwani"),
    ("jean", "jeans"), ("trouser", "trousers"), ("chino", "trousers"), ("short", "shorts"), ("skirt", "skirt"),
    ("dress", "dress"), ("gown", "dress"), ("shirt", "shirt"), ("top", "top"), ("blazer", "blazer"), ("jacket", "jacket"),
    ("sweatshirt", "sweatshirt"), ("hoodie", "sweatshirt"), ("sweater", "sweater"), ("track", "trackpants"),
    ("legging", "leggings"), ("sneaker", "shoes"), ("shoe", "shoes"), ("heel", "heels"), ("sandal", "sandals"),
    ("flip", "sandals"), ("boot", "shoes"), ("watch", "watch"), ("bag", "bag"), ("backpack", "bag"), ("wallet", "wallet"),
    ("sunglass", "sunglasses"), ("bra", "innerwear"), ("brief", "innerwear"), ("sock", "socks"), ("cap", "cap"),
)

_CONTACT = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|(?:\+?\d[\d\s-]{8,}\d)|(?:toll[- ]free|customer care|helpline)[^.]*",
    re.IGNORECASE,
)


def _key(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum() or ch == "_")


def scrub_contacts(text: str) -> str:
    """Remove e-mail addresses, phone numbers and helpline sentences from product text."""
    return re.sub(r"\s{2,}", " ", _CONTACT.sub(" ", str(text))).strip()


def infer_category(name: str) -> str:
    tokens = tokenize(name)
    joined = " ".join(tokens)
    for keyword, category in CATEGORY_KEYWORDS:
        if any(tok.startswith(keyword) for tok in tokens) or keyword in joined.split():
            return category
    return "other"


def normalize_gender(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = GENDER_ALIASES.get(text.replace(" ", ""), text)
    return text if text in GENDERS else "unisex"


def price_band(price: float) -> str:
    if price < 500:
        return "under 500"
    if price < 1000:
        return "500-999"
    if price < 2500:
        return "1000-2499"
    if price < 5000:
        return "2500-4999"
    return "5000+"


def normalize(frame: pd.DataFrame, source: str = "catalog") -> pd.DataFrame:
    lookup = {_key(c): c for c in frame.columns}
    out = pd.DataFrame(index=frame.index)
    for canon, aliases in COLUMNS.items():
        found = next((lookup[a] for a in aliases if a in lookup), None)
        if found is None:
            if canon in REQUIRED:
                raise CatalogError(f"{source}: missing required column {canon!r} (accepted: {', '.join(aliases)})")
            out[canon] = ""
            continue
        out[canon] = frame[found]

    out["product_id"] = out["product_id"].astype(str).str.strip()
    if (out["product_id"] == "").any() or out["product_id"].isin(["nan", "None"]).any():
        raise CatalogError(f"{source}: empty product_id")
    price = pd.to_numeric(out["price"], errors="coerce")
    if price.isna().any() or (price < 0).any():
        bad = (np.flatnonzero((price.isna() | (price < 0)).to_numpy()) + 2)[:5].tolist()
        raise CatalogError(f"{source}: price must be a number >= 0 (file rows {bad})")
    out["price"] = price.astype(float)
    for col in ("name", "brand", "description", "colour"):
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["colour"] = out["colour"].str.lower().replace("", "unknown")
    out["gender"] = out["gender"].map(normalize_gender)
    out["description"] = out["description"].map(scrub_contacts)
    out["category"] = out["name"].map(infer_category)
    out["price_band"] = out["price"].map(price_band)

    before = len(out)
    out = out.drop_duplicates("product_id", keep="first").reset_index(drop=True)
    out.attrs["duplicates_removed"] = before - len(out)
    return out


def load_catalog(path) -> pd.DataFrame:
    return normalize(pd.read_csv(path), source=str(path))


def document_text(row: pd.Series) -> str:
    """The text that the retrievers index for one product."""
    return " ".join(
        str(part)
        for part in (row["name"], row["brand"], row["gender"], row["category"], row["colour"], row["description"])
        if part
    )
