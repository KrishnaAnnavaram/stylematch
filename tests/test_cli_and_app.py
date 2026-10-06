"""CLI commands, the light core (reference problem 9) and the optional Streamlit page."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stylematch.cli import main


@pytest.fixture
def env(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("STYLEMATCH_CATALOG", str(settings.catalog))
    monkeypatch.setenv("STYLEMATCH_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STYLEMATCH_LLM_API_KEY", raising=False)
    return tmp_path


def test_synth_writes_catalog_and_gold(tmp_path, capsys):
    assert main(["synth", "--out", str(tmp_path), "--products", "200", "--queries", "15"]) == 0
    assert (tmp_path / "catalog.csv").exists()
    assert sum(1 for _ in open(tmp_path / "gold.jsonl", encoding="utf-8")) == 15


def test_index_is_built_then_reused(env, capsys):
    assert main(["index"]) == 0
    assert "index built" in capsys.readouterr().err
    assert main(["index"]) == 0
    assert "index reused" in capsys.readouterr().err


def test_recommend_json(env, capsys):
    assert main(["recommend", "red dress for women under 2000", "--k", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["items"]) == 3 and payload["intent"]["max_price"] == 2000


def test_recommend_without_matches_returns_1(env):
    assert main(["recommend", "sherwani for men under 5"]) == 1


def test_evaluate_and_pool(env, data_paths, capsys):
    out = env / "eval"
    assert main(["evaluate", "--gold", str(data_paths["gold"]), "--k", "5", "--out", str(out)]) == 0
    assert (out / "summary.csv").exists() and (out / "comparisons.csv").exists()
    assert main(["pool", "--queries", str(data_paths["gold"]), "--depth", "3", "--out", str(env / "pool.csv")]) == 0
    assert set(pd.read_csv(env / "pool.csv").columns) >= {"query_id", "product_id", "relevance"}


def test_study_plan_and_analysis(env, data_paths, capsys):
    out = env / "study"
    assert main(["study-plan", "--queries", str(data_paths["gold"]), "--participants", "4", "--out", str(out)]) == 0
    sheet = pd.read_csv(out / "sheet.csv")
    assert {"list_1", "list_2"} <= set(sheet.columns)
    sheet["rating_list_1"], sheet["rating_list_2"] = 6, 6
    sheet.to_csv(out / "ratings.csv", index=False)
    capsys.readouterr()
    assert main(["study-analyze", "--ratings", str(out / "ratings.csv"), "--key", str(out / "key.csv")]) == 0
    assert "no evidence" in json.loads(capsys.readouterr().out)["verdict"]


def test_demo_runs_offline(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "hybrid_attribute" in out


def test_core_import_does_not_load_optional_frameworks():
    code = (
        "import stylematch.cli, sys; "
        "print([m for m in ('streamlit', 'sentence_transformers', 'openai', 'torch') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_streamlit_page_renders(env):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "src" / "stylematch" / "app" / "streamlit_app.py"
    page = AppTest.from_file(str(app), default_timeout=120).run()
    assert not page.exception
    assert page.title[0].value == "stylematch"
    assert len(page.subheader) == 2
