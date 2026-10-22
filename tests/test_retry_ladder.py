"""
test_retry_ladder.py — Offline tests for the retry-with-escalation fix.

Three things are covered:

1. ``cached_call(cache_tag=...)`` — the cache key must distinguish the
   generation settings, so a retry is never served the first attempt's
   failed response.
2. ``OllamaClient.cache_tag`` / ``OllamaClient.with_overrides`` — the tag
   must change when the settings change, and ``with_overrides`` must not
   mutate the original client.
3. ``extract_entities_relationships`` — the retry ladder itself: a chunk
   that fails to parse must be retried with escalated settings, and a
   chunk that parses must NOT be retried.

No network: every LLM call is a stub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag_lite.cache.llm_cache import _make_key, cached_call  # noqa: E402
from graphrag_lite.ingestion.extractor import extract_entities_relationships  # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """
    Point the LLM cache at a throwaway directory for every test in this
    module.

    Without this the stub clients' responses are written to the REAL
    canonical cache at data/cache, and a later test with the same
    (prompt, model, tag) is served from it instead of calling the stub
    — which is exactly the conflation bug this module exists to cover.
    """
    import graphrag_lite.cache.llm_cache as llm_cache

    cache = llm_cache.diskcache.Cache(str(tmp_path / "cache"))
    monkeypatch.setattr(llm_cache, "_get_cache", lambda: cache)
    yield cache


# ------------------------------------------------------------------ #
# 1. Cache key separation
# ------------------------------------------------------------------ #

def test_cache_tag_changes_the_key():
    """A different cache_tag must produce a different cache key."""
    a = _make_key("prompt", "model", "fp=0.25")
    b = _make_key("prompt", "model", "fp=0.5")
    assert a != b


def test_empty_cache_tag_preserves_the_historical_key():
    """
    The default (empty) tag must keep the legacy key exactly, so the
    canonical cache entries that survived the 2026-10-22 wipe stay
    reachable.

    The legacy key was sha256(model + "\\x00" + prompt). A non-empty tag
    adds a second separator and lands in a separate namespace.
    """
    import hashlib

    legacy = hashlib.sha256("model\x00prompt".encode("utf-8")).hexdigest()
    assert _make_key("prompt", "model") == legacy
    assert _make_key("prompt", "model", "") == legacy
    # A tagged key must differ from the legacy one.
    assert _make_key("prompt", "model", "fp=0.25") != legacy


def test_cached_call_with_a_tag_does_not_serve_the_untagged_entry(tmp_path, monkeypatch):
    """
    The failure this fixes: a retry with escalated settings must not be
    handed the first attempt's (failed) response.
    """
    import graphrag_lite.cache.llm_cache as llm_cache

    monkeypatch.setattr(llm_cache, "_cache", llm_cache.diskcache.Cache(str(tmp_path)))
    monkeypatch.setattr(llm_cache, "_get_cache", lambda: llm_cache._cache)

    calls: list[str] = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return "FAILED-RESPONSE"

    cached_call(prompt="p", model="m", fn=fake)
    # Same prompt+model, different tag -> must call fn again.
    got = cached_call(prompt="p", model="m", fn=fake, cache_tag="fp=0.5")
    assert got == "FAILED-RESPONSE"
    assert len(calls) == 2

    # Same prompt+model+tag -> served from cache, fn not called again.
    cached_call(prompt="p", model="m", fn=fake, cache_tag="fp=0.5")
    assert len(calls) == 2


# ------------------------------------------------------------------ #
# 2. OllamaClient.cache_tag / with_overrides
# ------------------------------------------------------------------ #

def test_cache_tag_reflects_every_setting_that_changes_the_response():
    base = OllamaClient(model="m", temperature=0.0, seed=42, num_predict=4096)
    assert base.cache_tag != OllamaClient(model="m", temperature=0.5, seed=42,
                                         num_predict=4096).cache_tag
    assert base.cache_tag != OllamaClient(model="m", temperature=0.0, seed=7,
                                         num_predict=4096).cache_tag
    assert base.cache_tag != OllamaClient(model="m", temperature=0.0, seed=42,
                                         num_predict=2048).cache_tag
    assert base.cache_tag != OllamaClient(model="m", temperature=0.0, seed=42,
                                         num_predict=4096, num_ctx=8192).cache_tag
    assert base.cache_tag != OllamaClient(model="m", temperature=0.0, seed=42,
                                         num_predict=4096, frequency_penalty=0.5).cache_tag
    assert base.cache_tag != OllamaClient(model="m", temperature=0.0, seed=42,
                                         num_predict=4096, think=False).cache_tag


def test_with_overrides_returns_a_new_client_and_leaves_the_original_alone():
    original = OllamaClient(model="m", num_predict=4096, frequency_penalty=0.25)
    escalated = original.with_overrides(frequency_penalty=1.0)

    assert escalated is not original
    assert escalated.frequency_penalty == 1.0
    assert original.frequency_penalty == 0.25          # unchanged
    assert escalated.num_predict == 4096               # other settings carried over
    assert escalated.model == "m"
    assert escalated.cache_tag != original.cache_tag    # distinct cache entry


def test_with_overrides_keeps_think_and_num_ctx():
    c = OllamaClient(model="m", think=False, num_ctx=8192)
    d = c.with_overrides(frequency_penalty=0.5)
    assert d.think is False
    assert d.num_ctx == 8192


# ------------------------------------------------------------------ #
# 3. The retry ladder in extract_entities_relationships
# ------------------------------------------------------------------ #

class _StubClient:
    """
    A client that fails to parse on the first N calls, then succeeds.

    Records the frequency_penalty used for each attempt.
    """

    def __init__(self, failures: int, good: str) -> None:
        self.failures = failures
        self.good = good
        self.attempts: list[float] = []
        self.penalty = 0.25
        self._current_penalty = 0.25

    @property
    def cache_tag(self) -> str:
        return f"fp={self._current_penalty}"

    def with_overrides(self, **overrides):
        new = _StubClient(self.failures, self.good)
        new.attempts = self.attempts          # share the attempt log
        new._current_penalty = overrides.get("frequency_penalty", self._current_penalty)
        new.penalty = new._current_penalty
        new.failures = self.failures
        return new

    def generate(self, prompt: str) -> str:
        self.attempts.append(self._current_penalty)
        if len(self.attempts) <= self.failures:
            return "not json at all"
        return self.good


_GOOD_JSON = (
    '{"entities": [{"name": "Transformer", "type": "CONCEPT", '
    '"description": "d", "source_sentence": "s"}], '
    '"relationships": []}'
)


def _chunk():
    return {
        "chunk_id": "abc123",
        "doc_id": "doc1",
        "text": "The Transformer is a model.",
        "index": 0,
    }


def test_a_parsing_chunk_is_not_retried():
    client = _StubClient(failures=0, good=_GOOD_JSON)
    result = extract_entities_relationships(_chunk(), client)
    assert len(result["entities"]) == 1
    assert len(client.attempts) == 1


def test_a_failing_chunk_is_retried_with_escalated_penalties(monkeypatch):
    """
    The first attempt fails to parse; the ladder must then try the
    configured escalation values and return the first success.
    """
    import graphrag_lite.ingestion.extractor as extractor

    cfg = type("C", (), {"effective_extraction_model": lambda self: "m",
                         "LLM_RETRY_FREQUENCY_PENALTIES": [0.5, 1.0]})()
    monkeypatch.setattr(extractor, "get_config", lambda: cfg)

    client = _StubClient(failures=1, good=_GOOD_JSON)
    with pytest.warns(UserWarning):
        result = extract_entities_relationships(_chunk(), client)

    assert len(result["entities"]) == 1
    # attempt 0 = as-configured (0.25) fails; attempt 1 = 0.5 succeeds.
    # The ladder stops at the first success — 1.0 is never tried.
    assert client.attempts == [0.25, 0.5]


def test_the_ladder_stops_at_the_first_success(monkeypatch):
    import graphrag_lite.ingestion.extractor as extractor

    cfg = type("C", (), {"effective_extraction_model": lambda self: "m",
                         "LLM_RETRY_FREQUENCY_PENALTIES": [0.5, 1.0]})()
    monkeypatch.setattr(extractor, "get_config", lambda: cfg)

    client = _StubClient(failures=0, good=_GOOD_JSON)
    extract_entities_relationships(_chunk(), client)
    assert client.attempts == [0.25]           # no retry needed


def test_the_ladder_exhaustion_reraises_the_last_error(monkeypatch):
    import graphrag_lite.ingestion.extractor as extractor

    cfg = type("C", (), {"effective_extraction_model": lambda self: "m",
                         "LLM_RETRY_FREQUENCY_PENALTIES": [0.5, 1.0]})()
    monkeypatch.setattr(extractor, "get_config", lambda: cfg)

    client = _StubClient(failures=99, good=_GOOD_JSON)
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            extract_entities_relationships(_chunk(), client)
    assert len(client.attempts) == 3             # all three attempts made


def test_a_client_without_with_overrides_gets_exactly_one_attempt(monkeypatch):
    """Backwards compatibility: a plain LLMClient has no escalation."""
    import graphrag_lite.ingestion.extractor as extractor

    cfg = type("C", (), {"effective_extraction_model": lambda self: "m",
                         "LLM_RETRY_FREQUENCY_PENALTIES": [0.5, 1.0]})()
    monkeypatch.setattr(extractor, "get_config", lambda: cfg)

    class Plain:
        def __init__(self):
            self.attempts = 0

        def generate(self, prompt: str) -> str:
            self.attempts += 1
            return "not json"

    client = Plain()
    with pytest.raises(ValueError):
        extract_entities_relationships(_chunk(), client)
    assert client.attempts == 1


def test_a_rung_equal_to_the_current_penalty_is_skipped(monkeypatch):
    """
    A rung that repeats the client's current penalty would be the SAME
    decode under the SAME cache key — the "retry" would be served the
    first attempt's failed response and change nothing. It must be
    skipped, and the next distinct rung used instead.
    """
    import graphrag_lite.ingestion.extractor as extractor

    cfg = type("C", (), {"effective_extraction_model": lambda self: "m",
                         "LLM_RETRY_FREQUENCY_PENALTIES": [0.25, 0.5, 1.0]})()
    monkeypatch.setattr(extractor, "get_config", lambda: cfg)

    client = _StubClient(failures=99, good=_GOOD_JSON)
    client.penalty = 0.25
    client._current_penalty = 0.25

    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            extract_entities_relationships(_chunk(), client)

    # 0.25 (the current one) is skipped, so the ladder is 0.25 -> 0.5 -> 1.0
    assert client.attempts == [0.25, 0.5, 1.0]
