"""
env.py -- point MindMesh's config at an isolated benchmark index.

Pipelines A and B read their storage through graphrag_lite.config (an
lru_cached pydantic-settings singleton driven by environment variables).
``activate_index`` sets those variables to a benchmark index directory,
clears the config cache and resets Pipeline B's graph singleton, so the
canonical data/cache is never read or written during a benchmark run.
"""

from __future__ import annotations

import os
from pathlib import Path

CHUNKING_TAG = "hotpotqa_paragraph"


def index_env(index_dir: Path, top_k: int) -> dict[str, str]:
    index_dir = Path(index_dir).resolve()
    return {
        "ZVEC_ROOT": str(index_dir),
        "CACHE_DIR": str(index_dir),          # extractions.jsonl + LLM disk cache
        "ZVEC_COLLECTION_CHUNKS": "chunks",
        "ZVEC_COLLECTION_ENTITIES": "entities",
        "CHUNKING_STRATEGY": CHUNKING_TAG,
        "ZVEC_TOP_N_FLAT": str(top_k),        # Pipeline A depth
        "TOP_N_FINAL_EVIDENCE": str(top_k),   # Pipeline B depth
        "ZVEC_ADOPT_UNTAGGED": "0",
    }


def activate_index(index_dir: Path, top_k: int = 10) -> dict[str, str]:
    env = index_env(index_dir, top_k)
    os.environ.update(env)
    from graphrag_lite.config import get_config
    get_config.cache_clear()
    try:
        import graphrag_lite.retrieval.pipeline_b as pb
        pb._G = pb._PR = pb._PR_MAX = None
    except Exception:
        pass
    try:
        import graphrag_lite.cache.llm_cache as lc
        lc._cache = None
    except Exception:
        pass
    cfg = get_config()
    assert Path(cfg.ZVEC_ROOT).resolve() == Path(index_dir).resolve(), "benchmark isolation failed"
    return env
