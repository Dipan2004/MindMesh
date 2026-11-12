"""
test_model_cache_isolation.py — Regression tests for model/cache identity.

These tests prove that two different OllamaClient instances configured for
different models cannot share cache entries accidentally.  They were added
as part of the 2026-11-09 fix that changed extractor.py and gleaning.py to
derive the cache-key model from ``llm_client.model`` rather than
``cfg.effective_extraction_model()``.

Run:
    python tests/test_model_cache_isolation.py
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphrag_lite.cache.llm_cache import _make_key, reset_stats, cache_stats
from graphrag_lite.llm.ollama_client import OllamaClient


# ── helpers ──────────────────────────────────────────────────────────────────


def _client(model: str, fp: float = 0.25) -> OllamaClient:
    return OllamaClient(
        model=model,
        temperature=0.0,
        seed=42,
        num_predict=4096,
        num_ctx=8192,
        frequency_penalty=fp,
    )


def _key(client: OllamaClient, prompt: str) -> str:
    return _make_key(prompt, client.model, client.cache_tag)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_different_models_produce_different_keys():
    """
    Two clients configured for different models must produce different cache
    keys for the same prompt.  This is the core regression: before the fix,
    extractor.py used cfg.effective_extraction_model() as the model key, so
    a client built with qwen2.5:3b but cfg saying qwen2.5:0.5b would store
    its response under the wrong key.
    """
    c1 = _client("qwen2.5:0.5b")
    c2 = _client("qwen2.5:1.5b")
    c3 = _client("qwen3:4b")

    prompt = "Extract entities: HNSW is an approximate nearest-neighbor algorithm."

    k1 = _key(c1, prompt)
    k2 = _key(c2, prompt)
    k3 = _key(c3, prompt)

    assert k1 != k2, "qwen2.5:0.5b and qwen2.5:1.5b must not share cache keys"
    assert k1 != k3, "qwen2.5:0.5b and qwen3:4b must not share cache keys"
    assert k2 != k3, "qwen2.5:1.5b and qwen3:4b must not share cache keys"

    print("PASS test_different_models_produce_different_keys")


def test_same_model_same_settings_same_key():
    """
    Two clients with identical model + settings must produce the same cache
    key so that a cache hit works correctly.
    """
    c1 = _client("qwen2.5:0.5b")
    c2 = _client("qwen2.5:0.5b")  # identical

    prompt = "Extract entities: HNSW is an algorithm."

    assert _key(c1, prompt) == _key(c2, prompt), (
        "Identical (model, settings, prompt) must produce identical cache key"
    )

    print("PASS test_same_model_same_settings_same_key")


def test_different_frequency_penalty_different_key():
    """
    The retry ladder escalates frequency_penalty. Each rung must produce a
    distinct cache key so the retry gets a fresh decode, not the failed
    response from the previous attempt.
    """
    c_base  = _client("qwen2.5:0.5b", fp=0.25)
    c_rung1 = _client("qwen2.5:0.5b", fp=0.50)
    c_rung2 = _client("qwen2.5:0.5b", fp=1.00)

    prompt = "Extract entities: HNSW test."

    k_base  = _key(c_base,  prompt)
    k_rung1 = _key(c_rung1, prompt)
    k_rung2 = _key(c_rung2, prompt)

    assert k_base  != k_rung1, "fp=0.25 and fp=0.5 must not share cache keys"
    assert k_base  != k_rung2, "fp=0.25 and fp=1.0 must not share cache keys"
    assert k_rung1 != k_rung2, "fp=0.5 and fp=1.0 must not share cache keys"

    print("PASS test_different_frequency_penalty_different_key")


def test_extractor_uses_client_model_not_config_model(tmp_path):
    """
    End-to-end: build a fake chunk, run extractor with two different clients
    and verify that each gets a separate cache entry (miss, not hit).

    This is the exact regression scenario: cfg.EXTRACTION_MODEL = 'qwen2.5:0.5b'
    but the experiment runner passes an OllamaClient('qwen2.5:1.5b', ...).
    Before the fix both clients would share the same cache key (keyed to the
    cfg model).  After the fix, each client's .model attribute determines the
    key, so they are always isolated.

    We validate isolation via miss counts: if client_b reused client_a's
    entry it would get a HIT (misses unchanged); a proper miss means they
    are isolated.
    """
    import diskcache
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.chunker import Chunk
    import graphrag_lite.cache.llm_cache as lc_mod

    chunk: Chunk = {
        "chunk_id": "test_isolation_chunk",
        "doc_id":   "test_doc",
        "chunk_index": 0,
        "text": "HNSW is an approximate nearest-neighbor algorithm used in vector databases.",
    }

    RESPONSE = (
        '{"entities": [{"name": "HNSW", "type": "TECHNOLOGY", '
        '"description": "ANN algorithm.", '
        '"source_sentence": "HNSW is an approximate nearest-neighbor algorithm."}], '
        '"relationships": []}'
    )

    class FakeClient:
        def __init__(self, model: str):
            self.model = model
            self.frequency_penalty = 0.0
            self.cache_tag = f"t=0.0;seed=42;np=4096;ctx=8192;fp=0.0;think=None"

        def generate(self, prompt: str) -> str:
            return RESPONSE

        def with_overrides(self, **kw):
            c = FakeClient(self.model)
            c.cache_tag = self.cache_tag
            return c

    cache_dir = str(tmp_path / "isolation_test_cache")
    old_cache = lc_mod._cache
    lc_mod._cache = diskcache.Cache(cache_dir)
    lc_mod._hits = 0
    lc_mod._misses = 0

    try:
        client_a = FakeClient("qwen2.5:0.5b")
        client_b = FakeClient("qwen2.5:1.5b")

        # Run extraction with client_a → 1 miss
        result_a = extract_entities_relationships(chunk, client_a)
        assert len(result_a["entities"]) > 0, "client_a extraction should succeed"
        assert lc_mod._misses == 1, f"Expected 1 miss after client_a, got {lc_mod._misses}"
        assert lc_mod._hits == 0, f"Expected 0 hits after client_a, got {lc_mod._hits}"

        # Run extraction with client_b → must be another MISS (not a hit on client_a's entry)
        result_b = extract_entities_relationships(chunk, client_b)
        assert len(result_b["entities"]) > 0, "client_b extraction should succeed"
        assert lc_mod._misses == 2, (
            f"Expected 2 misses after client_b, got {lc_mod._misses}. "
            "client_b must NOT reuse client_a's cache entry."
        )
        assert lc_mod._hits == 0, (
            f"Expected 0 hits, got {lc_mod._hits}. "
            "Different models must not share cache entries."
        )

        # Run client_a again → must be a HIT now (same model, same settings)
        result_a2 = extract_entities_relationships(chunk, client_a)
        assert lc_mod._hits == 1, (
            f"Expected 1 hit on second client_a call, got {lc_mod._hits}."
        )
        assert lc_mod._misses == 2, "Miss count must not change on cache hit"

        print("PASS test_extractor_uses_client_model_not_config_model")

    finally:
        if lc_mod._cache is not old_cache:
            lc_mod._cache.close()
        lc_mod._cache = old_cache


def test_gleaning_uses_client_model_not_config_model(tmp_path):
    """
    Same as above but for gleaning.py: gleaning must use the client's model
    for its cache key, not cfg.effective_extraction_model().
    """
    import diskcache
    from graphrag_lite.ingestion.gleaning import glean
    from graphrag_lite.ingestion.extractor import ExtractionResult
    from graphrag_lite.ingestion.chunker import Chunk
    import graphrag_lite.cache.llm_cache as lc_mod

    chunk: Chunk = {
        "chunk_id": "gleaning_test_chunk",
        "doc_id":   "test_doc",
        "chunk_index": 0,
        "text": "HNSW is used in vector databases. Faiss is another library.",
    }
    first_pass: ExtractionResult = {
        "entities": [
            {"name": "HNSW", "type": "TECHNOLOGY", "description": "ANN.",
             "source_sentence": "HNSW is used in vector databases."}
        ],
        "relationships": [],
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
    }

    class FakeGleanClient:
        def __init__(self, model: str):
            self.model = model
            self.frequency_penalty = 0.0
            self.cache_tag = "t=0.0;seed=42;np=4096;ctx=None;fp=0.0;think=None"

        def generate(self, prompt: str) -> str:
            return '{"entities": [], "relationships": []}'

    cache_dir = str(tmp_path / "gleaning_test_cache")
    old_cache = lc_mod._cache
    lc_mod._cache = diskcache.Cache(cache_dir)
    old_hits, old_misses = lc_mod._hits, lc_mod._misses
    lc_mod._hits = 0
    lc_mod._misses = 0

    try:
        client_a = FakeGleanClient("qwen2.5:0.5b")
        client_b = FakeGleanClient("qwen2.5:1.5b")

        # Run gleaning with client_a
        glean(chunk, first_pass, client_a, max_gleanings=1)
        misses_after_a = lc_mod._misses
        assert misses_after_a == 1, "Should be exactly 1 miss for client_a gleaning"

        # Run gleaning with client_b — must be another MISS
        glean(chunk, first_pass, client_b, max_gleanings=1)
        misses_after_b = lc_mod._misses
        assert misses_after_b == 2, (
            "client_b gleaning must get a MISS — model isolation requires "
            "separate cache entries per model"
        )

        print("PASS test_gleaning_uses_client_model_not_config_model")

    finally:
        if lc_mod._cache is not old_cache:
            lc_mod._cache.close()
        lc_mod._cache = old_cache
        lc_mod._hits = old_hits
        lc_mod._misses = old_misses


# ── Run all tests ─────────────────────────────────────────────────────────────


def main():
    import tempfile, traceback

    tests = [
        test_different_models_produce_different_keys,
        test_same_model_same_settings_same_key,
        test_different_frequency_penalty_different_key,
    ]
    # tests that need tmp_path
    path_tests = [
        test_extractor_uses_client_model_not_config_model,
        test_gleaning_uses_client_model_not_config_model,
    ]

    passed = 0
    failed = 0

    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    for fn in path_tests:
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            try:
                fn(Path(tmp))
                passed += 1
            except Exception as exc:
                print(f"FAIL {fn.__name__}: {exc}")
                traceback.print_exc()
                failed += 1

    print()
    print(f"{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
