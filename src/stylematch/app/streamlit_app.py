"""Streamlit page: one query, two systems side by side, scores and reasons. No ratings are invented.

Run with ``stylematch ui`` or ``streamlit run src/stylematch/app/streamlit_app.py``.
"""

from __future__ import annotations

import streamlit as st

from stylematch.config import Settings
from stylematch.index import build_or_load
from stylematch.recommender import Recommender
from stylematch.rerank import RERANKERS
from stylematch.retrieve import METHODS


@st.cache_resource
def _recommender() -> Recommender:
    settings = Settings.from_env()
    index, _ = build_or_load(settings.catalog, settings.index_dir, settings)
    return Recommender(index, settings)


def _show(column, title: str, rec) -> None:
    column.subheader(title)
    column.caption(f"filters: {', '.join(rec.filters) or 'none'} | pool: {rec.pool_size}")
    for item in rec.items:
        column.markdown(f"**{item['rank']}. {item['name']}**")
        column.text(f"{item['brand']} | {item['colour']} | {item['price']:,.0f} | score {item['retrieval_score']:.4f}")
        column.caption(item["reason"])
    for note in rec.notes:
        column.warning(note)


def main() -> None:
    st.set_page_config(page_title="stylematch", layout="wide")
    st.title("stylematch")
    query = st.text_input("What are you looking for?", "an outfit for a birthday party for women under 3000")
    k = st.slider("Number of products", 1, 10, 5)
    left_method = st.selectbox("Left system", METHODS, index=METHODS.index("tfidf"))
    right_method = st.selectbox("Right system", METHODS, index=METHODS.index("hybrid"))
    rerank = st.selectbox("Re-ranker for the right system", RERANKERS, index=1)
    if query.strip():
        recommender = _recommender()
        left, right = st.columns(2)
        _show(left, f"{left_method}", recommender.recommend(query, k=k, method=left_method, reranker="none"))
        try:
            _show(right, f"{right_method} + {rerank}", recommender.recommend(query, k=k, method=right_method, reranker=rerank))
        except ValueError as exc:
            right.error(str(exc))


main()
