"""The search index: built once from the catalog, saved to disk and loaded on later runs.

The index holds three structures over the same product texts:

* TF-IDF vectors (the baseline retriever),
* BM25 term counts (the lexical retriever),
* dense vectors from the embedder (the semantic retriever).

A fingerprint of the normalized catalog, the embedder and the index version is
saved in ``manifest.json``. If the fingerprint is unchanged, ``build_or_load``
loads the saved index and embeds only the query at search time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .catalog import document_text, load_catalog
from .embed import Embedder, make_embedder
from .text import tokenize

INDEX_VERSION = "stylematch-index-1"


class BM25:
    """Okapi BM25 over a document-term count matrix."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, texts: list[str]) -> "BM25":
        self.vectorizer_ = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        counts = self.vectorizer_.fit_transform(texts)
        self._set(counts.tocsc(), self.vectorizer_.vocabulary_)
        return self

    def _set(self, counts: sparse.csc_matrix, vocabulary: dict[str, int]) -> None:
        self.counts_ = counts
        self.vocabulary_ = {str(k): int(v) for k, v in vocabulary.items()}
        n_docs = counts.shape[0]
        df = np.diff(counts.indptr)
        self.idf_ = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.doc_len_ = np.asarray(counts.sum(axis=1)).ravel().astype(float)
        self.avgdl_ = float(self.doc_len_.mean()) if n_docs else 0.0

    def scores(self, query: str, rows: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(self.counts_.shape[0]) if rows is None else np.asarray(rows)
        out = np.zeros(rows.size)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len_[rows] / max(self.avgdl_, 1e-9))
        for term in set(tokenize(query)):
            col = self.vocabulary_.get(term)
            if col is None:
                continue
            tf = self.counts_[:, col].toarray().ravel()[rows]
            out += self.idf_[col] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def save(self, folder: Path) -> None:
        sparse.save_npz(folder / "bm25_counts.npz", self.counts_.tocsr())
        (folder / "bm25.json").write_text(json.dumps({"k1": self.k1, "b": self.b, "vocabulary": self.vocabulary_}), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "BM25":
        meta = json.loads((folder / "bm25.json").read_text(encoding="utf-8"))
        model = cls(meta["k1"], meta["b"])
        model._set(sparse.load_npz(folder / "bm25_counts.npz").tocsc(), meta["vocabulary"])
        return model


@dataclass
class Index:
    products: pd.DataFrame
    tfidf: TfidfVectorizer
    tfidf_matrix: sparse.csr_matrix
    bm25: BM25
    embedder: Embedder
    vectors: np.ndarray
    manifest: dict

    def __len__(self) -> int:
        return len(self.products)

    def position(self, product_id: str) -> int:
        return int(self._positions[str(product_id)])

    def __post_init__(self):
        self._positions = {pid: i for i, pid in enumerate(self.products["product_id"].astype(str))}

    def has(self, product_id: str) -> bool:
        return str(product_id) in self._positions

    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.products.to_csv(folder / "products.csv", index=False)
        joblib.dump(self.tfidf, folder / "tfidf.joblib")
        sparse.save_npz(folder / "tfidf_matrix.npz", self.tfidf_matrix)
        self.bm25.save(folder)
        np.save(folder / "vectors.npy", self.vectors)
        if self.embedder.persist:
            joblib.dump(self.embedder, folder / "embedder.joblib")
        (folder / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return folder


def fingerprint(products: pd.DataFrame, embedder_name: str, embed_model: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(INDEX_VERSION.encode())
    digest.update(f"{embedder_name}|{embed_model}".encode())
    digest.update(products.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def build_index(products: pd.DataFrame, embedder: Embedder, embed_model: str = "") -> Index:
    texts = [document_text(row) for _, row in products.iterrows()]
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(texts).tocsr()
    bm25 = BM25().fit(texts)
    embedder.fit(texts)
    vectors = embedder.embed(texts)
    manifest = {
        "version": INDEX_VERSION,
        "fingerprint": fingerprint(products, embedder.name, embed_model),
        "embedder": embedder.name,
        "embed_model": embed_model,
        "products": int(len(products)),
        "dims": int(vectors.shape[1]),
    }
    return Index(products, tfidf, tfidf_matrix, bm25, embedder, vectors, manifest)


def load_index(folder, settings=None) -> Index:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != INDEX_VERSION:
        raise ValueError(f"index version {manifest.get('version')!r} is not {INDEX_VERSION!r}: build the index again")
    products = pd.read_csv(folder / "products.csv", dtype={"product_id": str}, keep_default_na=False)
    products["price"] = products["price"].astype(float)
    if (folder / "embedder.joblib").exists():
        embedder = joblib.load(folder / "embedder.joblib")
    else:
        embedder = make_embedder(manifest["embedder"], settings)
    return Index(
        products=products,
        tfidf=joblib.load(folder / "tfidf.joblib"),
        tfidf_matrix=sparse.load_npz(folder / "tfidf_matrix.npz").tocsr(),
        bm25=BM25.load(folder),
        embedder=embedder,
        vectors=np.load(folder / "vectors.npy"),
        manifest=manifest,
    )


def build_or_load(catalog_path, folder, settings) -> tuple[Index, bool]:
    """Return ``(index, built)``. ``built`` is False if the saved index was reused."""
    products = load_catalog(catalog_path)
    folder = Path(folder)
    expected = fingerprint(products, settings.embedder, settings.embed_model)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == expected:
            return load_index(folder, settings), False
    index = build_index(products, make_embedder(settings.embedder, settings), settings.embed_model)
    index.save(folder)
    return index, True
