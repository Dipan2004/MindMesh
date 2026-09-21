"""
llm_cache.py — Disk-backed memoization for every LLM call.

Every extraction call, gleaning re-prompt, and generation call goes through
``cached_call``. The cache key is sha256(prompt + model_name), stored in a
diskcache SQLite database at CACHE_DIR. This means re-running the indexing
pipeline after a crash or code change never re-pays for LLM calls that
already completed.

Usage:
    from graphrag_lite.cache.llm_cache import cached_call, cache_stats

    result = cached_call(
        prompt="Extract entities from: ...",
        model="qwen3:4b",
        fn=lambda p: my_llm.generate(p),
    )
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable

import diskcache

from graphrag_lite.config import get_config

# ------------------------------------------------------------------ #
# Module-level cache instance (one per process, lazily initialised)
# ------------------------------------------------------------------ #

_cache: diskcache.Cache | None = None
_hits: int = 0
_misses: int = 0


def _get_cache() -> diskcache.Cache:
    """
    Return (or create) the process-level diskcache.Cache instance.

    The cache directory is created automatically if it doesn't exist.
    """
    global _cache
    if _cache is None:
        cfg = get_config()
        cache_dir = cfg.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        _cache = diskcache.Cache(cache_dir)
    return _cache


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def _make_key(prompt: str, model: str) -> str:
    """Return a stable sha256 hex digest for (prompt, model)."""
    raw = f"{model}\x00{prompt}"          # null byte separator avoids collisions
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_call(
    prompt: str,
    model: str,
    fn: Callable[[str], str],
) -> str:
    """
    Return the cached LLM response for (prompt, model), calling fn on a miss.

    Parameters
    ----------
    prompt:
        The exact prompt string sent to the LLM.
    model:
        The model name/tag (e.g. "qwen3:4b", "grok-3"). Included in the
        cache key so the same prompt against a different model stores a
        separate entry.
    fn:
        A callable that accepts the prompt string and returns the LLM
        response string. Only called on a cache miss.

    Returns
    -------
    str
        The LLM response, either freshly generated or recalled from cache.

    Raises
    ------
    Any exception raised by ``fn`` propagates normally — this wrapper does
    NOT swallow errors from the underlying LLM call.
    """
    global _hits, _misses

    if not prompt:
        raise ValueError("prompt must not be empty")
    if not model:
        raise ValueError("model must not be empty")

    cache = _get_cache()
    key = _make_key(prompt, model)

    result: str | None = cache.get(key, default=None)
    if result is not None:
        _hits += 1
        return result

    # Cache miss — call the real function (let any exception propagate)
    _misses += 1
    result = fn(prompt)

    if not isinstance(result, str):
        raise TypeError(
            f"cached_call: fn must return str, got {type(result).__name__!r}"
        )

    cache.set(key, result)
    return result


def cache_stats() -> dict[str, int]:
    """
    Return hit/miss counters accumulated since process start.

    Returns
    -------
    dict with keys "hits", "misses", "total".
    """
    return {
        "hits": _hits,
        "misses": _misses,
        "total": _hits + _misses,
    }


def clear_cache() -> None:
    """
    Wipe all entries from the disk cache.

    Intended for test teardown or forced re-extraction. Not called during
    normal pipeline operation.
    """
    _get_cache().clear()


def reset_stats() -> None:
    """Reset hit/miss counters. Useful for isolated test assertions."""
    global _hits, _misses
    _hits = 0
    _misses = 0
