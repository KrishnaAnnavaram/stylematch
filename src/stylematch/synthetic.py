"""Synthetic catalog and labelled queries for the offline demo and the tests.

All brands, products and phone numbers are made up. The generator knows the
true category, gender, colour, price and occasions of each product, so it can
write graded relevance labels for each query (2 = full match, 1 = partial match).
Descriptions often describe an occasion with other words than the query uses
(for example "celebrations" for a party), as real product text does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .intent import GENDER_MATCH

BRANDS = ["Aurel", "Kestra", "Norvane", "Pellio", "Quorra", "Saffra", "Tavik", "Ulmo", "Vessa", "Zandor"]
COLOURS = ["black", "white", "blue", "red", "green", "grey", "pink", "beige", "yellow", "maroon"]
GENDER_WORD = {"women": "Women", "men": "Men", "boys": "Boys", "girls": "Girls", "unisex": "Unisex"}

# category, genders, name nouns, price range, occasions
SPECS = [
    ("dress", ["women", "girls"], ["Dress", "Maxi Dress", "Bodycon Dress", "Fit and Flare Dress"], (700, 4500), ["party", "casual"]),
    ("kurta", ["women", "men"], ["Kurta", "Straight Kurta", "Anarkali Kurta"], (500, 3500), ["wedding", "festival", "casual"]),
    ("saree", ["women"], ["Saree", "Silk Saree"], (900, 9000), ["wedding", "festival"]),
    ("sherwani", ["men"], ["Sherwani"], (4000, 15000), ["wedding"]),
    ("shirt", ["men", "women", "boys"], ["Formal Shirt", "Casual Shirt", "Slim Fit Shirt"], (500, 2500), ["office", "casual"]),
    ("trousers", ["men", "women"], ["Formal Trousers", "Chinos"], (700, 3000), ["office"]),
    ("blazer", ["men", "women"], ["Blazer"], (2000, 8000), ["office", "party"]),
    ("tshirt", ["men", "women", "boys", "girls"], ["T-shirt", "Polo T-shirt", "Graphic T-shirt"], (250, 1500), ["casual", "gym"]),
    ("trackpants", ["men", "women"], ["Track Pants", "Joggers"], (400, 2000), ["gym"]),
    ("shoes", ["men", "women", "unisex"], ["Running Shoes", "Sneakers"], (1200, 7000), ["gym", "running", "casual"]),
    ("heels", ["women"], ["Heels", "Stilettos"], (800, 4000), ["party"]),
    ("jeans", ["men", "women", "boys"], ["Jeans", "Slim Fit Jeans"], (700, 3500), ["casual"]),
    ("jacket", ["men", "women", "unisex"], ["Puffer Jacket", "Bomber Jacket"], (1200, 6000), ["winter"]),
    ("bag", ["unisex", "women"], ["Backpack", "Trolley Bag", "Tote Bag"], (600, 9000), ["travel"]),
    ("watch", ["men", "women", "unisex"], ["Analogue Watch", "Smart Watch"], (900, 12000), ["office", "party"]),
]
OCCASION_TEXT = {
    "party": ["sequinned with a shimmer finish for celebrations", "a glamorous look for evening events", "festive sparkle for celebrations"],
    "casual": ["easy everyday comfort", "relaxed casual styling", "soft fabric for daily wear"],
    "wedding": ["rich embroidery for ceremonies", "an ethnic look with zari detail for grand functions"],
    "festival": ["a festive ethnic look", "bright festive colours with mirror work"],
    "office": ["a crisp tailored look for workwear", "formal styling for meetings"],
    "gym": ["quick-dry sports fabric for training", "stretch fabric for workouts"],
    "running": ["a cushioned sole for running", "a lightweight build for long runs"],
    "winter": ["a warm fleece lining for cold days", "insulated padding for chilly weather"],
    "travel": ["spacious compartments for trips", "a sturdy build for journeys"],
}
FABRICS = ["cotton", "polyester", "viscose", "linen blend", "denim", "silk blend", "knit", "leather"]


def generate_catalog(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """The catalog with hidden truth columns ``occasions`` (``|``-joined)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        category, genders, nouns, (low, high), occasions = SPECS[int(rng.integers(len(SPECS)))]
        gender = str(rng.choice(genders))
        colour = str(rng.choice(COLOURS))
        noun = str(rng.choice(nouns))
        brand = str(rng.choice(BRANDS))
        price = float(round(np.exp(rng.uniform(np.log(low), np.log(high))) / 10) * 10 - 1)
        own = [o for o in occasions if rng.random() < 0.75] or [occasions[0]]
        phrases = [str(rng.choice(OCCASION_TEXT[o])) for o in own]
        description = f"{colour.title()} {noun.lower()} in {rng.choice(FABRICS)}, " + ", ".join(phrases) + "."
        if rng.random() < 0.05:
            description += " For queries call customer care 1800 000 0000."
        rows.append(
            {
                "ProductID": 20_000_000 + i,
                "ProductName": f"{brand} {GENDER_WORD[gender]} {colour.title()} {noun}",
                "ProductBrand": brand,
                "Gender": GENDER_WORD[gender],
                "Price (INR)": price,
                "Description": description,
                "PrimaryColor": colour.title(),
                "true_category": category,
                "true_occasions": "|".join(own),
            }
        )
    return pd.DataFrame(rows)


def _gender_ok(product_gender: str, query_gender: str | None) -> bool:
    return query_gender is None or product_gender.lower() in GENDER_MATCH[query_gender]


def generate_gold(catalog: pd.DataFrame, n_queries: int = 40, seed: int = 42) -> list[dict]:
    """Labelled queries: attribute queries, budget queries and occasion queries."""
    rng = np.random.default_rng(seed + 7)
    occasion_queries = [
        ("an outfit for a birthday party", "party", "women"),
        ("what to wear to a party", "party", "men"),
        ("wedding wear", "wedding", "men"),
        ("something for a wedding", "wedding", "women"),
        ("clothes for the gym", "gym", "men"),
        ("workout clothes", "gym", "women"),
        ("office wear", "office", "women"),
        ("outfit for a job interview at the office", "office", "men"),
        ("keep warm in winter", "winter", None),
        ("bag for travel", "travel", None),
        ("festival outfit", "festival", "women"),
        ("shoes for running", "running", None),
    ]
    gold = []
    qid = 0
    for text, occasion, gender in occasion_queries:
        query = f"{text} for {gender}" if gender else text
        relevant = {}
        for _, row in catalog.iterrows():
            if occasion in row["true_occasions"].split("|") and _gender_ok(row["Gender"], gender):
                relevant[str(row["ProductID"])] = 2
        gold.append({"query_id": f"q{qid:03d}", "kind": "occasion", "query": query, "relevant": relevant})
        qid += 1

    while len(gold) < n_queries:
        category, genders, nouns, (low, high), _ = SPECS[int(rng.integers(len(SPECS)))]
        gender = str(rng.choice([g for g in genders if g != "unisex"] or ["women"]))
        noun = str(rng.choice(nouns)).lower()
        if rng.random() < 0.5:
            colour = str(rng.choice(COLOURS))
            query = f"{colour} {noun} for {gender}"
            relevant = {}
            for _, row in catalog.iterrows():
                if row["true_category"] == category and _gender_ok(row["Gender"], gender):
                    relevant[str(row["ProductID"])] = 2 if row["PrimaryColor"].lower() == colour else 1
            kind = "attribute"
        else:
            budget = float(int(np.exp(rng.uniform(np.log(low), np.log(high))) / 100) * 100 + 100)
            query = f"{noun} for {gender} under {budget:.0f}"
            relevant = {
                str(row["ProductID"]): 2
                for _, row in catalog.iterrows()
                if row["true_category"] == category and _gender_ok(row["Gender"], gender) and row["Price (INR)"] <= budget
            }
            kind = "budget"
        if relevant:
            gold.append({"query_id": f"q{qid:03d}", "kind": kind, "query": query, "relevant": relevant})
            qid += 1
    return gold


def write_synthetic(out_dir, n_products: int = 1500, n_queries: int = 40, seed: int = 42) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = generate_catalog(n_products, seed)
    gold = generate_gold(catalog, n_queries, seed)
    paths = {"catalog": out / "catalog.csv", "gold": out / "gold.jsonl"}
    catalog.drop(columns=["true_category", "true_occasions"]).to_csv(paths["catalog"], index=False)
    with open(paths["gold"], "w", encoding="utf-8") as handle:
        for row in gold:
            handle.write(json.dumps(row) + "\n")
    return paths
