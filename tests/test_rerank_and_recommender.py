"""Reference problems 1 (key in source), 2 (random products to GPT), 3 (fake ratings) and 6 (invented products)."""

import json
import pickle

import pytest

from stylematch.config import Settings
from stylematch.embed import OpenAIEmbedder
from stylematch.llm import OpenAICompatibleClient, ScriptedChatClient
from stylematch.recommender import Recommender
from stylematch.rerank import LLMReranker, make_reranker, validate_ranking


def test_validate_ranking_removes_invented_and_repeated_ids():
    order, dropped = validate_ranking(["b", "zzz", "b", "a"], ["a", "b", "c"])
    assert order == ["b", "a", "c"]
    assert dropped == ["zzz"]


def test_llm_sees_only_retrieved_candidates_and_cannot_add_products(index, settings):
    def reply(messages):
        lines = messages[1]["content"].split("Candidates:\n")[1].splitlines()
        ids = [json.loads(line)["id"] for line in lines]
        return json.dumps({"ranking": [{"id": "INVENTED-1", "reason": "made up"}, {"id": ids[2], "reason": "best fit"}]})

    client = ScriptedChatClient(reply)
    rec = Recommender(index, settings, client=client).recommend("black jeans for men", k=5, reranker="llm")
    retrieved = Recommender(index, settings, client=None).recommend("black jeans for men", k=50, reranker="none")
    allowed = {item["product_id"] for item in retrieved.items}
    assert all(item["product_id"] in allowed for item in rec.items)
    assert all(index.has(item["product_id"]) for item in rec.items)
    assert rec.items[0]["reason"] == "best fit"
    assert any("not candidates" in note for note in rec.notes)
    sent = client.calls[0][1]["content"]
    assert all(json.loads(line)["id"] in allowed for line in sent.split("Candidates:\n")[1].splitlines())


def test_malformed_llm_reply_falls_back_to_retrieval_order(index, settings):
    rec = Recommender(index, settings, client=ScriptedChatClient("not json")).recommend("red dress", k=3, reranker="llm")
    base = Recommender(index, settings, client=None).recommend("red dress", k=3, reranker="none")
    assert [i["product_id"] for i in rec.items] == [i["product_id"] for i in base.items]
    assert any("not usable" in note for note in rec.notes)


def test_llm_reranker_needs_a_client():
    with pytest.raises(ValueError):
        make_reranker("llm", None)
    assert isinstance(make_reranker("llm", ScriptedChatClient("{}")), LLMReranker)


def test_results_depend_on_the_query_and_are_deterministic(recommender):
    a1 = recommender.recommend("blue kurta for women", k=5)
    a2 = recommender.recommend("blue kurta for women", k=5)
    b = recommender.recommend("running shoes for men", k=5)
    assert [i["product_id"] for i in a1.items] == [i["product_id"] for i in a2.items]
    assert {i["product_id"] for i in a1.items}.isdisjoint({i["product_id"] for i in b.items})
    assert all(i["category"] == "kurta" for i in a1.items)


def test_items_show_scores_and_reasons_not_ratings(recommender):
    rec = recommender.recommend("green tshirt for boys under 1000", k=5)
    for item in rec.items:
        assert "rating" not in item
        assert item["retrieval_score"] > 0
        assert item["reason"].startswith("Matches")
        assert item["price"] <= 1000


def test_no_match_gives_a_clear_note(recommender):
    rec = recommender.recommend("sherwani for men under 10")
    assert rec.items == [] and "no product passes the filters" in rec.notes[0]


def test_api_key_never_appears_in_repr_or_pickle():
    settings = Settings(llm_provider="openai", llm_api_key="secret-value-123")
    assert "secret-value-123" not in repr(settings)
    client = OpenAICompatibleClient("secret-value-123", "m", "https://example.invalid/v1")
    assert "secret-value-123" not in repr(client)
    embedder = OpenAIEmbedder("secret-value-123")
    assert b"secret-value-123" not in pickle.dumps(embedder)


def test_settings_read_the_key_from_the_environment(monkeypatch):
    monkeypatch.delenv("STYLEMATCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("STYLEMATCH_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    settings = Settings.from_env()
    assert settings.has_llm and settings.llm_provider == "openai"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert not Settings.from_env().has_llm


def test_invalid_settings_are_refused(monkeypatch):
    monkeypatch.setenv("STYLEMATCH_EMBEDDER", "word2vec")
    with pytest.raises(ValueError):
        Settings.from_env()
