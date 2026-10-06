"""Reference problems 7 (index rebuilt per query) and 8 (refit per click, no filters, weak baseline)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stylematch.embed import Embedder, LsaEmbedder
from stylematch.index import BM25, build_index, build_or_load, load_index
from stylematch.intent import parse_query
from stylematch.retrieve import METHODS, retrieve
from stylematch.catalog import load_catalog


class CountingEmbedder(Embedder):
    name = "counting"
    persist = True

    def __init__(self):
        self.inner = LsaEmbedder(dims=16)
        self.calls = []

    def fit(self, texts):
        self.inner.fit(texts)
        return self

    def embed(self, texts):
        self.calls.append(len(texts))
        return self.inner.embed(texts)


def test_index_is_reused_when_the_catalog_did_not_change(settings, index):
    again, built = build_or_load(settings.catalog, settings.index_dir, settings)
    assert built is False
    assert np.allclose(again.vectors, index.vectors)
    assert again.manifest["fingerprint"] == index.manifest["fingerprint"]


def test_changed_catalog_gives_a_new_index(settings, tmp_path):
    frame = pd.read_csv(settings.catalog)
    frame.loc[0, "Description"] = "a different description"
    path = tmp_path / "changed.csv"
    frame.to_csv(path, index=False)
    changed = dataclasses.replace(settings, catalog=path, index_dir=tmp_path / "idx")
    _, first = build_or_load(path, changed.index_dir, changed)
    _, second = build_or_load(path, changed.index_dir, changed)
    assert first is True and second is False


def test_a_query_embeds_only_the_query_text(settings):
    products = load_catalog(settings.catalog)
    embedder = CountingEmbedder()
    idx = build_index(products, embedder)
    assert embedder.calls == [len(products)]
    retrieve(idx, parse_query("red dress for women"), method="dense")
    retrieve(idx, parse_query("black jeans"), method="hybrid")
    assert embedder.calls[1:] == [1, 1]


def test_saved_index_round_trip(index, settings, tmp_path):
    index.save(tmp_path / "copy")
    loaded = load_index(tmp_path / "copy", settings)
    query = parse_query("blue shirt for men")
    for method in METHODS:
        a = [c.product_id for c in retrieve(index, query, method).candidates]
        b = [c.product_id for c in retrieve(loaded, query, method).candidates]
        assert a == b


@pytest.mark.parametrize("method", METHODS)
def test_every_method_obeys_the_same_hard_filters(index, method):
    intent = parse_query("dress for women under 1500")
    found = retrieve(index, intent, method=method, k=30)
    rows = index.products.set_index("product_id").loc[[c.product_id for c in found.candidates]]
    assert len(rows) > 0
    assert rows["gender"].isin(["women", "unisex"]).all()
    assert (rows["price"] <= 1500).all()
    assert found.filters == ["gender in ['women', 'unisex']", "price <= 1500"]


def test_pool_is_identical_for_every_method(index):
    intent = parse_query("shoes for men under 3000")
    sizes = {retrieve(index, intent, method=m).pool_size for m in METHODS}
    assert len(sizes) == 1


def test_empty_pool_returns_no_candidates(index):
    found = retrieve(index, parse_query("watch for men under 5"), method="hybrid")
    assert found.candidates == [] and found.pool_size == 0


def test_bm25_prefers_documents_with_the_query_term():
    bm25 = BM25().fit(["red dress party", "blue jeans", "red red shirt", "green cap"])
    scores = bm25.scores("red")
    assert scores[1] == 0 and scores[3] == 0
    assert scores[2] > scores[0] > 0


def test_occasion_query_without_literal_overlap_finds_occasion_products(index):
    found = retrieve(index, parse_query("what to wear to a party for men"), method="hybrid", k=10)
    text = " ".join(index.products.set_index("product_id").loc[[c.product_id for c in found.candidates], "description"]).lower()
    assert "celebration" in text or "evening" in text
