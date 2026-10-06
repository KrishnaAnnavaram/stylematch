"""Settings from environment variables. The API key is read from the environment only."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

EMBEDDERS = ("lsa", "openai", "sentence-transformers")
LLM_PROVIDERS = ("none", "openai")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    catalog: Path = Path("data/catalog.csv")
    index_dir: Path = Path("artifacts/index")
    embedder: str = "lsa"
    embed_model: str = ""
    llm_provider: str = "none"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = field(default="", repr=False)
    llm_timeout_s: float = 30.0
    top_k: int = 5
    candidates: int = 50
    seed: int = 42

    def __post_init__(self):
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"STYLEMATCH_EMBEDDER must be one of {EMBEDDERS}, got {self.embedder!r}")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"STYLEMATCH_LLM_PROVIDER must be one of {LLM_PROVIDERS}, got {self.llm_provider!r}")

    @property
    def has_llm(self) -> bool:
        return self.llm_provider != "none" and bool(self.llm_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        key = _get("STYLEMATCH_LLM_API_KEY") or _get("OPENAI_API_KEY")
        provider = _get("STYLEMATCH_LLM_PROVIDER", "openai" if key else "none")
        return cls(
            catalog=Path(_get("STYLEMATCH_CATALOG", "data/catalog.csv")),
            index_dir=Path(_get("STYLEMATCH_INDEX_DIR", "artifacts/index")),
            embedder=_get("STYLEMATCH_EMBEDDER", "lsa"),
            embed_model=_get("STYLEMATCH_EMBED_MODEL"),
            llm_provider=provider,
            llm_base_url=_get("STYLEMATCH_LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=_get("STYLEMATCH_LLM_MODEL", "gpt-4o-mini"),
            llm_api_key=key,
            llm_timeout_s=_float("STYLEMATCH_LLM_TIMEOUT_S", 30.0),
            top_k=_int("STYLEMATCH_TOP_K", 5),
            candidates=_int("STYLEMATCH_CANDIDATES", 50),
            seed=_int("STYLEMATCH_SEED", 42, minimum=0),
        )
