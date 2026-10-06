"""Embedders behind one small interface: ``fit(texts)`` once, then ``embed(texts)``.

* ``lsa`` (default, offline): TF-IDF word n-grams reduced with truncated SVD.
* ``openai``: any OpenAI-compatible ``/embeddings`` endpoint (standard library HTTP).
* ``sentence-transformers``: a local model, imported only when you use it.

Every embedder returns rows of length 1, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import tokenize


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


class Embedder:
    name = "base"
    persist = False  # True if the fitted object must be saved with the index

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LsaEmbedder(Embedder):
    """Latent semantic analysis: words that occur together get close vectors."""

    name = "lsa"
    persist = True

    def __init__(self, dims: int = 128, seed: int = 42):
        self.dims = dims
        self.seed = seed

    def fit(self, texts):
        self.vectorizer_ = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        matrix = self.vectorizer_.fit_transform(texts)
        dims = max(2, min(self.dims, matrix.shape[1] - 1, matrix.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=dims, random_state=self.seed)
        self.svd_.fit(matrix)
        return self

    def embed(self, texts):
        if not hasattr(self, "svd_"):
            raise RuntimeError("LsaEmbedder.fit must run before embed")
        return _normalize(self.svd_.transform(self.vectorizer_.transform(texts)))


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = "https://api.openai.com/v1", timeout_s: float = 30.0, batch: int = 256):
        if not api_key:
            raise ValueError("the openai embedder needs STYLEMATCH_LLM_API_KEY or OPENAI_API_KEY")
        self._key = api_key
        self.model = model or "text-embedding-3-small"
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.batch = batch

    def __getstate__(self):  # never pickle the key
        state = dict(self.__dict__)
        state["_key"] = ""
        return state

    def embed(self, texts):
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            body = json.dumps({"model": self.model, "input": list(texts[start : start + self.batch])}).encode()
            request = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=body,
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - configured URL
                payload = json.loads(response.read())
            rows += [item["embedding"] for item in sorted(payload["data"], key=lambda d: d["index"])]
        return _normalize(np.array(rows))


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = model or "all-MiniLM-L6-v2"
        self._model = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_model"] = None
        return state

    def embed(self, texts):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("install the extra: pip install 'stylematch[sentence-transformers]'") from exc
            self._model = SentenceTransformer(self.model)
        return _normalize(self._model.encode(list(texts), batch_size=64, show_progress_bar=False))


def make_embedder(name: str, settings=None) -> Embedder:
    if name == "lsa":
        return LsaEmbedder(seed=getattr(settings, "seed", 42))
    if name == "openai":
        return OpenAIEmbedder(
            api_key=getattr(settings, "llm_api_key", ""),
            model=getattr(settings, "embed_model", "") or "text-embedding-3-small",
            base_url=getattr(settings, "llm_base_url", "https://api.openai.com/v1"),
            timeout_s=getattr(settings, "llm_timeout_s", 30.0),
        )
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder(getattr(settings, "embed_model", ""))
    raise ValueError(f"unknown embedder {name!r}")
