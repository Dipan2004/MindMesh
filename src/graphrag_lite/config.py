"""
config.py — Single source of truth for every tunable variable.

Reads from environment variables (or a .env file), falling back to the
defaults documented in CONFIG.md. Every value used anywhere in the
pipeline must be imported from here — no inline magic numbers.

Usage:
    from graphrag_lite.config import get_config
    cfg = get_config()
    print(cfg.CHUNK_SIZE_EXTRACTION)
"""

from __future__ import annotations

import functools
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    All configuration variables for graphrag_lite.

    Pydantic-settings reads from environment variables (case-insensitive)
    and from a .env file in the working directory, then falls back to the
    defaults below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",        # ignore unrecognised env vars silently
    )

    # ------------------------------------------------------------------ #
    # §1  Chunking (CONFIG.md §1)
    # ------------------------------------------------------------------ #
    CHUNK_SIZE_EXTRACTION: int = Field(
        default=450,
        description="Token budget for extraction chunks (300-600).",
    )
    CHUNK_OVERLAP_PCT: float = Field(
        default=0.12,
        description="Fraction of chunk size used as overlap (0.10-0.15).",
    )
    CHUNK_SIZE_CITATION: int = Field(
        default=1200,
        description="Token budget for citation-display chunks (1000-1500).",
    )
    TOKENIZER: str = Field(
        default="qwen",
        description=(
            "Tokenizer family: 'qwen' uses cl100k_base via tiktoken "
            "(closest public approximation to Qwen's tokenizer)."
        ),
    )
    CHUNKING_STRATEGY: str = Field(
        default="markdown_aware",
        description=(
            "Active chunking strategy: sliding_window | recursive_char | "
            "markdown_aware | small_window. "
            "CONFIRMED default after chunking ablation (Phase 1 extension Task 1.6): "
            "lowest fail rate (4.8%), highest unique/1kTok (12.38 measured), "
            "non-overlapping sections = zero redundant token processing."
        ),
    )

    # ------------------------------------------------------------------ #
    # §2  Extraction
    # ------------------------------------------------------------------ #
    MAX_GLEANINGS: int = Field(
        default=1,
        description="Re-prompts after first extraction pass. Raise only after measuring recall.",
    )
    EXTRACTION_MODEL: str = Field(
        default="",
        description=(
            "Model used for entity/relationship extraction. "
            "Defaults to SLM_MODEL when empty."
        ),
    )
    EXTRACTION_TEMPERATURE: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for extraction and gleaning LLM calls. "
            "0.0 = greedy / deterministic. Keep low so results are stable "
            "across re-runs and cache hits are meaningful."
        ),
    )
    EXTRACTION_SEED: int = Field(
        default=42,
        description=(
            "Fixed RNG seed for Ollama extraction calls. "
            "Combined with temperature=0.0 this produces deterministic output. "
            "temperature=0.0 alone is insufficient on some models/platforms."
        ),
    )

    # ------------------------------------------------------------------ #
    # §3  Embedding
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Ollama model name or sentence-transformers model id.",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        description="Texts per embed_batch call. Tune for CPU RAM budget.",
    )

    # ------------------------------------------------------------------ #
    # §4  Vector Store (Zvec)
    # ------------------------------------------------------------------ #
    ZVEC_COLLECTION_ENTITIES: str = Field(default="entities_v1")
    ZVEC_COLLECTION_CHUNKS: str = Field(default="chunks_v1")
    ZVEC_TOP_K_ENTRY: int = Field(
        default=8,
        description="Top-k entities returned by Pipeline B entry-point search.",
    )
    ZVEC_TOP_N_FLAT: int = Field(
        default=8,
        description="Top-N chunks returned by Pipeline A flat search.",
    )
    ZVEC_FUSION_METHOD: str = Field(
        default="rrf",
        description="Zvec fusion strategy: 'rrf' or 'weighted'.",
    )

    # ------------------------------------------------------------------ #
    # §5  Graph Retrieval (Pipeline B)
    # ------------------------------------------------------------------ #
    BFS_HOP_DEPTH: int = Field(
        default=2,
        description="BFS expansion depth from entry-point entities (2-3).",
    )
    RANKING_METHOD: Literal["pagerank", "degree"] = Field(
        default="pagerank",
        description="Node-ranking strategy used in Pipeline B candidate selection.",
    )
    PAGERANK_DAMPING: float = Field(
        default=0.85,
        description=(
            "NetworkX PageRank damping factor. Standard default is 0.85. "
            "Read from config so ablations can vary it without code changes."
        ),
    )
    TOP_N_CANDIDATES_AFTER_RANKING: int = Field(
        default=20,
        description="Candidates kept after structural ranking, before second Zvec pass.",
    )
    TOP_N_FINAL_EVIDENCE: int = Field(
        default=6,
        description="Final evidence pieces passed to the LLM.",
    )

    # ------------------------------------------------------------------ #
    # §6  Serialization
    # ------------------------------------------------------------------ #
    SERIALIZATION_FORMAT: Literal["toon", "json"] = Field(
        default="toon",
        description="Evidence serialization format sent to the LLM.",
    )

    # ------------------------------------------------------------------ #
    # §7  LLM / Mode Toggle
    # ------------------------------------------------------------------ #
    LLM_MODE: Literal["local", "api"] = Field(
        default="local",
        description=(
            "'local' → Ollama/SLM_MODEL + LangSmith tracing ON. "
            "'api' → Grok API + tracing OFF."
        ),
    )
    SLM_MODEL: str = Field(
        default="qwen3:4b",
        description="Ollama model tag used in local mode.",
    )
    API_MODEL: str = Field(
        default="",
        description=(
            "Grok model name for api mode. Check current Grok docs at build time; "
            "do not hardcode a version that may be deprecated."
        ),
    )
    LLM_TIMEOUT_S: float = Field(
        default=300.0,
        description=(
            "Seconds before an Ollama generate() call raises TimeoutError. "
            "CPU inference on qwen2.5:0.5b takes 30-120s+ per call; gleaning "
            "prompts with 6+ entities reach 1500+ tokens and can exceed 120s. "
            "Raised to 300s after httpcore.ReadTimeout at chunk 32/42. "
            "Also surfaces stuck Ollama queue states from abandoned requests."
        ),
    )

    # ------------------------------------------------------------------ #
    # §8  Observability
    # ------------------------------------------------------------------ #
    LANGCHAIN_PROJECT: str = Field(
        default="graphrag-lite-eval",
        description="LangSmith project name. Keep eval runs grouped separately.",
    )
    LANGCHAIN_API_KEY: str = Field(
        default="",
        description="LangSmith API key. Only needed in local mode with tracing enabled.",
    )

    # ------------------------------------------------------------------ #
    # §9  Caching
    # ------------------------------------------------------------------ #
    CACHE_BACKEND: str = Field(
        default="diskcache",
        description="Cache backend. Only 'diskcache' is implemented.",
    )
    CACHE_DIR: str = Field(
        default="data/cache",
        description="Directory for the disk-cache SQLite store.",
    )

    # ------------------------------------------------------------------ #
    # §10  Stress Test
    # ------------------------------------------------------------------ #
    STRESS_CONCURRENCY_LEVELS: List[int] = Field(
        default=[10, 50, 200, 1000],
        description="Ramp levels for the async stress test (Task 9.1).",
    )
    STRESS_TIMEOUT_S: int = Field(
        default=30,
        description="Per-request timeout in seconds during the stress test.",
    )
    STRESS_SEMAPHORE_LIMIT: int = Field(
        default=50,
        description=(
            "Max in-flight Grok requests. Must not exceed Grok's actual "
            "rate limit — verify against current API docs before first run."
        ),
    )

    # ------------------------------------------------------------------ #
    # §11  Production Hardening
    # ------------------------------------------------------------------ #
    UVICORN_WORKERS: int = Field(
        default=0,
        description=(
            "Uvicorn worker count. 0 = auto (os.cpu_count()). "
            "Set explicitly to override."
        ),
    )
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        description="Consecutive failures before the circuit breaker opens.",
    )
    RETRY_BACKOFF_BASE_S: float = Field(
        default=0.5,
        description="Base sleep time (seconds) for exponential backoff retry.",
    )
    RETRY_MAX_ATTEMPTS: int = Field(
        default=3,
        description="Max retry attempts for outbound Grok calls.",
    )

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #

    def effective_extraction_model(self) -> str:
        """
        Return the model to use for extraction.
        Falls back to SLM_MODEL when EXTRACTION_MODEL is not explicitly set.
        Use this instead of reading EXTRACTION_MODEL directly.
        """
        return self.EXTRACTION_MODEL if self.EXTRACTION_MODEL else self.SLM_MODEL


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Return the singleton Config instance.

    Cached after the first call so every import across the process shares
    the same object and reads the same values. Call ``get_config.cache_clear()``
    in tests that need a fresh read.
    """
    return Config()
