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
from pathlib import Path
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

def _make_key(prompt: str, model: str, cache_tag: str = "") -> str:
    """
    Return a stable sha256 hex digest for (prompt, model, cache_tag).

    With an EMPTY ``cache_tag`` this is exactly the historical key,
    sha256(model + "\\x00" + prompt), so the cache entries that survived
    the 2026-10-22 wipe stay reachable. A non-empty tag adds a second
    separator, keeping tagged entries in their own namespace.
    """
    if not cache_tag:
        raw = f"{model}\x00{prompt}"   # historical key — do not change
    else:
        raw = f"{model}\x00{cache_tag}\x00{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_call(
    prompt: str,
    model: str,
    fn: Callable[[str], str],
    cache_tag: str = "",
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
    cache_tag:
        Optional extra discriminator folded into the cache key. Use it when
        the SAME (prompt, model) pair can legitimately be answered with
        DIFFERENT generation settings — for example a retry that escalates
        ``frequency_penalty`` after a parse failure. Without a tag the
        retry would hit the first attempt's cached (failed) response and
        change nothing. The default "" keeps the historical key exactly, so
        existing cache entries stay reachable.

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
    key = _make_key(prompt, model, cache_tag)

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


def _protected_paths(cache_dir: str) -> list[str]:
    """
    Return names inside ``cache_dir`` that must never be wiped by a blind
    cache clear.

    A directory qualifies as protected if it holds a Zvec collection (any
    ``manifest.*`` file) or if it holds a canonical extraction dump. Those
    are expensive to rebuild and are not LLM-response cache entries.
    """
    root = Path(cache_dir)
    if not root.exists():
        return []
    protected: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(p.glob("manifest.*")):
            protected.append(p.name + "/")
        elif p.is_file() and p.name.startswith("extractions") and p.suffix == ".jsonl":
            protected.append(p.name)
    return protected


def clear_cache(force: bool = False) -> None:
    """
    Wipe all entries from the disk cache.

    Intended for test teardown or forced re-extraction. Not called during
    normal pipeline operation.

    Parameters
    ----------
    force:
        When False (the default) this refuses to clear a cache directory
        that also holds Zvec collections or canonical extraction dumps —
        clearing those directories blind destroyed the canonical LLM
        response cache on 2026-10-22. Pass ``force=True`` only when you
        have confirmed the directory holds nothing but LLM responses.

    Raises
    ------
    RuntimeError
        If the directory holds protected artefacts and ``force`` is False.
    """
    cache = _get_cache()
    protected = _protected_paths(cache.directory)
    if protected and not force:
        raise RuntimeError(
            f"clear_cache: refusing to wipe {cache.directory!r} — it also holds "
            f"protected artefacts: {protected}. The canonical LLM response "
            f"cache was destroyed this way before. Point CACHE_DIR at a "
            f"private directory, or call clear_cache(force=True) if you are "
            f"certain."
        )
    cache.clear()


def reset_stats() -> None:
    """Reset hit/miss counters. Useful for isolated test assertions."""
    global _hits, _misses
    _hits = 0
    _misses = 0
