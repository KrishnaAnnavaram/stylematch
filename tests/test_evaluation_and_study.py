"""Reference problems 4 (undocumented study, fixed verdict), 5 (NDCG misuse) and 10 (no offline evaluation)."""

import numpy as np
import pandas as pd
import pytest

from stylematch.evaluate import (
    default_systems,
    evaluate,
    load_gold,
    make_pool,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from stylematch.study import analyze, make_plan, simulate_ratings, unblind


def test_ranking_metrics_on_a_hand_example():
    relevant = {"a": 2, "b": 1}
    ranked = ["x", "a", "b"]
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, relevant) == pytest.approx(0.5)
    expected = (3 / np.log2(3) + 1 / np.log2(4)) / (3 / np.log2(2) + 1 / np.log2(3))
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(expected)


def test_ndcg_uses_gold_labels_and_is_one_for_the_ideal_order():
    relevant = {"a": 2, "b": 1, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a", "c"], relevant, 3) < 1.0
    assert ndcg_at_k(["x", "y"], relevant, 2) == 0.0
    assert ndcg_at_k(["a"], {}, 1) == 0.0


def test_paired_bootstrap_interval_contains_the_mean():
    a = np.linspace(0.5, 0.9, 30)
    mean, low, high = paired_bootstrap(a, a - 0.1)
    assert mean == pytest.approx(0.1) and low <= mean <= high


def test_evaluation_compares_all_systems_on_gold_labels(index, recommender, data_paths):
    gold = load_gold(data_paths["gold"])[:10]
    result = evaluate(index, gold, default_systems(recommender, 10), k=10)
    summary = result["summary"].set_index("system")
    assert set(summary.index) == {"tfidf_unfiltered", "tfidf", "bm25", "dense", "hybrid", "hybrid_attribute"}
    assert {"precision@10", "recall@10", "ndcg@10", "mrr", "hit", "brand_diversity", "coverage"} <= set(summary.columns)
    assert summary.loc["tfidf", "ndcg@10"] > summary.loc["tfidf_unfiltered", "ndcg@10"]
    assert summary.loc["hybrid_attribute", "ndcg@10"] >= summary.loc["tfidf", "ndcg@10"]
    assert set(result["comparisons"]["baseline"]) == {"tfidf"}


def test_gold_csv_from_a_labelled_pool(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"query_id": ["q1", "q1", "q2"], "query": ["a", "a", "b"], "product_id": ["1", "2", "3"], "relevance": [2, 0, 1]}
    ).to_csv(path, index=False)
    gold = load_gold(path)
    assert gold == [{"query_id": "q1", "query": "a", "relevant": {"1": 2}}, {"query_id": "q2", "query": "b", "relevant": {"3": 1}}]


def test_pool_sheet_is_blind_and_has_an_empty_label_column(recommender):
    pool = make_pool(recommender, [{"query_id": "q1", "query": "blue jeans for men"}], depth=5)
    assert "system" not in pool.columns
    assert (pool["relevance"] == "").all()
    assert pool["product_id"].is_unique


QUERIES = [{"query_id": f"q{i}", "query": f"query {i}"} for i in range(8)]


def test_plan_is_blind_and_balanced():
    sheet, key = make_plan(QUERIES, 40, ("A", "B"), seed=1)
    assert not {"list_1_system", "list_2_system"} & set(sheet.columns)
    share = (key["list_1_system"] == "A").mean()
    assert 0.4 < share < 0.6
    orders = sheet.groupby("participant")["query_id"].apply(tuple)
    assert orders.nunique() > 1


def test_unblind_maps_ratings_to_systems():
    sheet, key = make_plan(QUERIES[:2], 2, ("A", "B"), seed=2)
    sheet["rating_list_1"], sheet["rating_list_2"] = 9, 3
    paired = unblind(sheet, key, "A", "B")
    merged = paired.merge(key, on=["participant", "query_id"])
    expected_a = np.where(merged["list_1_system"] == "A", 9, 3)
    assert (merged["rating_a"].to_numpy() == expected_a).all()


def test_verdict_comes_from_the_data():
    sheet, key = make_plan(QUERIES, 30, ("A", "B"), seed=3)
    same = analyze(simulate_ratings(sheet, key, {"A": 6.0, "B": 6.0}, seed=3), key, "A", "B")
    assert same.verdict.startswith("no evidence")
    better = analyze(simulate_ratings(sheet, key, {"A": 7.5, "B": 5.0}, seed=3), key, "A", "B")
    assert better.verdict.startswith("participants preferred A")
    assert better.ci_low > 0 and better.cohens_dz > 0
    worse = analyze(simulate_ratings(sheet, key, {"A": 5.0, "B": 7.5}, seed=3), key, "A", "B")
    assert worse.verdict.startswith("participants preferred B")


def test_each_participant_counts_once():
    sheet, key = make_plan(QUERIES, 5, ("A", "B"), seed=4)
    result = analyze(simulate_ratings(sheet, key, {"A": 7, "B": 5}, seed=4), key, "A", "B")
    assert result.participants == 5 and result.ratings == 40


def test_ratings_outside_the_scale_are_refused():
    sheet, key = make_plan(QUERIES[:2], 2, ("A", "B"))
    sheet["rating_list_1"], sheet["rating_list_2"] = 11, 3
    with pytest.raises(ValueError, match="1 to 10"):
        analyze(sheet, key, "A", "B")
