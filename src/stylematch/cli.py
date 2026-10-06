"""Command line interface: ``stylematch <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import Settings
from .evaluate import default_systems, evaluate, load_gold, make_pool
from .index import build_index, build_or_load, load_index
from .catalog import load_catalog
from .embed import make_embedder
from .recommender import Recommender
from .rerank import RERANKERS
from .retrieve import METHODS
from .study import analyze, make_plan, simulate_ratings
from .synthetic import write_synthetic


def _settings(args) -> Settings:
    settings = Settings.from_env()
    changes = {}
    for name in ("catalog", "index_dir", "embedder"):
        value = getattr(args, name, None)
        if value:
            changes[name] = Path(value) if name != "embedder" else value
    return dataclasses.replace(settings, **changes)


def _index(settings: Settings):
    index, built = build_or_load(settings.catalog, settings.index_dir, settings)
    print(f"index {'built' if built else 'reused'}: {len(index)} products, embedder {index.manifest['embedder']}", file=sys.stderr)
    return index


def _print(frame: pd.DataFrame) -> None:
    with pd.option_context("display.width", 180, "display.max_columns", 20, "display.max_colwidth", 60):
        print(frame.to_string(index=False))


def cmd_synth(args) -> int:
    paths = write_synthetic(args.out, n_products=args.products, n_queries=args.queries, seed=args.seed)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_index(args) -> int:
    settings = _settings(args)
    if args.rebuild:
        index = build_index(load_catalog(settings.catalog), make_embedder(settings.embedder, settings), settings.embed_model)
        index.save(settings.index_dir)
        print(f"index built: {len(index)} products in {settings.index_dir}")
    else:
        index = _index(settings)
    print(json.dumps(index.manifest, indent=2))
    return 0


def cmd_recommend(args) -> int:
    settings = _settings(args)
    rec = Recommender(_index(settings), settings).recommend(
        args.query, k=args.k, method=args.method, reranker=args.rerank, gender=args.gender,
        min_price=args.min_price, max_price=args.max_price, use_filters=not args.no_filters,
    )
    if args.json:
        print(json.dumps({"intent": rec.intent.to_dict(), "filters": rec.filters, "items": rec.items, "notes": rec.notes}, indent=2))
        return 0
    print(f"intent: {rec.intent.to_dict()}")
    print(f"filters: {rec.filters or 'none'}   pool: {rec.pool_size} products   method: {rec.method} + {rec.reranker}")
    for item in rec.items:
        print(f"{item['rank']:>2}. [{item['product_id']}] {item['name']} | {item['price']:,.0f} | score {item['retrieval_score']:.4f}\n    {item['reason']}")
    for note in rec.notes:
        print(f"note: {note}")
    return 0 if rec.items else 1


def cmd_evaluate(args) -> int:
    settings = _settings(args)
    index = _index(settings)
    recommender = Recommender(index, settings)
    result = evaluate(index, load_gold(args.gold), default_systems(recommender, args.k), k=args.k, seed=settings.seed)
    _print(result["summary"].round(3))
    print()
    _print(result["comparisons"].round(3))
    print()
    _print(result["by_kind"].round(3))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result["per_query"].to_csv(out / "per_query.csv", index=False)
        result["summary"].to_csv(out / "summary.csv", index=False)
        result["comparisons"].to_csv(out / "comparisons.csv", index=False)
        print(f"wrote {out}")
    return 0


def _queries(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({"query_id": str(row["query_id"]), "query": row["query"]})
    return rows


def cmd_pool(args) -> int:
    settings = _settings(args)
    pool = make_pool(Recommender(_index(settings), settings), _queries(args.queries), depth=args.depth, seed=settings.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.out, index=False)
    print(f"wrote {len(pool)} rows to {args.out}. Fill the 'relevance' column with 0, 1 or 2.")
    return 0


def _study_lists(settings: Settings, sheet: pd.DataFrame, key: pd.DataFrame, k: int) -> pd.DataFrame:
    recommender = Recommender(_index(settings), settings)
    systems = default_systems(recommender, k)
    products = recommender.index.products.set_index("product_id")
    cache: dict[tuple[str, str], str] = {}

    def names(system: str, query: str) -> str:
        if (system, query) not in cache:
            ids = systems[system](query)
            cache[(system, query)] = " | ".join(products.loc[ids, "name"].tolist())
        return cache[(system, query)]

    merged = sheet.merge(key, on=["participant", "query_id"])
    sheet = sheet.copy()
    sheet["list_1"] = [names(s, q) for s, q in zip(merged["list_1_system"], merged["query"])]
    sheet["list_2"] = [names(s, q) for s, q in zip(merged["list_2_system"], merged["query"])]
    return sheet


def cmd_study_plan(args) -> int:
    settings = _settings(args)
    sheet, key = make_plan(_queries(args.queries), args.participants, (args.a, args.b), seed=settings.seed)
    sheet = _study_lists(settings, sheet, key, args.k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out / "sheet.csv", index=False)
    key.to_csv(out / "key.csv", index=False)
    print(f"wrote {out / 'sheet.csv'} (give to participants) and {out / 'key.csv'} (keep hidden until analysis)")
    return 0


def cmd_study_analyze(args) -> int:
    result = analyze(pd.read_csv(args.ratings), pd.read_csv(args.key), args.a, args.b)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    paths = write_synthetic(out / "synthetic_data", seed=args.seed)
    settings = dataclasses.replace(Settings.from_env(), catalog=paths["catalog"], index_dir=out / "index", seed=args.seed)
    index, _ = build_or_load(settings.catalog, settings.index_dir, settings)
    recommender = Recommender(index, settings, client=None)
    print("SYNTHETIC DATA. Example: 'an outfit for a birthday party for women under 3000'")
    for item in recommender.recommend("an outfit for a birthday party for women under 3000", k=5).items:
        print(f"  {item['rank']}. {item['name']} | {item['price']:,.0f} | {item['reason']}")
    gold = load_gold(paths["gold"])
    result = evaluate(index, gold, default_systems(recommender, 10), k=10, seed=args.seed)
    print("\nOffline evaluation, k = 10 (synthetic labels):")
    _print(result["summary"].round(3))
    print()
    _print(result["comparisons"].round(3))
    print()
    _print(result["by_kind"].round(3))
    sheet, key = make_plan(gold[:10], 24, ("tfidf", "hybrid_attribute"), seed=args.seed)
    ratings = simulate_ratings(sheet, key, {"tfidf": 5.5, "hybrid_attribute": 6.5}, seed=args.seed)
    study = analyze(ratings, key, "hybrid_attribute", "tfidf")
    print(f"\nSimulated blinded study (SYNTHETIC ratings, 24 participants x 10 queries): {study.verdict}")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in study.to_dict().items()}))
    out.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(out / "summary.csv", index=False)
    result["comparisons"].to_csv(out / "comparisons.csv", index=False)
    return 0


def cmd_ui(args) -> int:  # pragma: no cover - starts a server
    app = Path(__file__).parent / "app" / "streamlit_app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stylematch", description="Hybrid fashion recommender with offline evaluation")
    parser.add_argument("--version", action="version", version=f"stylematch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_index(p):
        p.add_argument("--catalog")
        p.add_argument("--index-dir", dest="index_dir")
        p.add_argument("--embedder", choices=("lsa", "openai", "sentence-transformers"))

    p = sub.add_parser("synth", help="write a synthetic catalog and labelled queries")
    p.add_argument("--out", default="data")
    p.add_argument("--products", type=int, default=1500)
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("index", help="build the index once, or reuse it if the catalog did not change")
    with_index(p)
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("recommend", help="recommend products for a query")
    with_index(p)
    p.add_argument("query")
    p.add_argument("--k", type=int)
    p.add_argument("--method", choices=METHODS, default="hybrid")
    p.add_argument("--rerank", choices=RERANKERS, default="attribute")
    p.add_argument("--gender", choices=("women", "men", "boys", "girls", "kids"))
    p.add_argument("--min-price", dest="min_price", type=float)
    p.add_argument("--max-price", dest="max_price", type=float)
    p.add_argument("--no-filters", dest="no_filters", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("evaluate", help="offline metrics on a labelled gold set")
    with_index(p)
    p.add_argument("--gold", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("pool", help="write a blind labelling sheet from the top items of every system")
    with_index(p)
    p.add_argument("--queries", required=True, help="JSONL with query_id and query")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--out", default="reports/pool.csv")
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("study-plan", help="write a blinded, randomized user-study sheet and its key")
    with_index(p)
    p.add_argument("--queries", required=True)
    p.add_argument("--participants", type=int, default=20)
    p.add_argument("--a", default="hybrid_attribute")
    p.add_argument("--b", default="tfidf")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default="study")
    p.set_defaults(func=cmd_study_plan)

    p = sub.add_parser("study-analyze", help="unblind and test the user-study ratings")
    p.add_argument("--ratings", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--a", default="hybrid_attribute")
    p.add_argument("--b", default="tfidf")
    p.set_defaults(func=cmd_study_analyze)

    p = sub.add_parser("demo", help="synthetic catalog + index + evaluation + simulated study, offline")
    p.add_argument("--out", default="reports/demo")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("ui", help="start the Streamlit page (needs the 'ui' extra)")
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
