"""
test_phase7_circuit_breaker_dod.py — Task 7.3 DoD integration test.

Drives the circuit breaker to OPEN by injecting consecutive failures,
then verifies fast-fail behaviour and measures the latency difference.

DoD requirement: paste the failure log showing the threshold being hit and
the fast-fail latency after.

Uses in-process ASGI transport so mocks apply.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import logging

from graphrag_lite.api.middleware import _breaker, _CircuitBreaker


@pytest.fixture(autouse=True)
def reset_breaker():
    """Always start each test with a CLOSED breaker."""
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0
    _breaker._opened_at = 0.0
    yield
    # Restore after test
    _breaker._state = _CircuitBreaker._CLOSED
    _breaker._consecutive_failures = 0


def _app_client(patches):
    """Return an httpx.AsyncClient with ASGI transport under given patches."""
    from graphrag_lite.api.main import app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold_and_fast_fails(caplog):
    """
    Task 7.3 DoD test.

    Steps:
    1. Make CIRCUIT_BREAKER_FAILURE_THRESHOLD calls through mode='api'
       each of which fails (GrokClientStub raises).
    2. Verify the breaker is now OPEN.
    3. Make one more call — must return 503 instantly (fast-fail).
    4. Measure and print: slow retry latency vs fast-fail latency.
    """
    from graphrag_lite.config import get_config
    from graphrag_lite.llm.generator import Answer

    cfg = get_config()
    threshold = cfg.CIRCUIT_BREAKER_FAILURE_THRESHOLD

    # Evidence for the retrieval mock
    from graphrag_lite.retrieval.pipeline_a import Evidence
    ev = [Evidence(chunk_id="c001", source_doc="doc", chunk_index=0,
                   text="text", score=0.03, source="chunk")]

    # Generator that always raises (simulates broken Grok API)
    def failing_generate(*args, **kwargs):
        raise ConnectionError("Grok endpoint unreachable")

    base_patches = [
        patch("graphrag_lite.api.main.set_tracing"),
        patch("graphrag_lite.api.main.init_http_client"),
        patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock),
        patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev),
        patch("graphrag_lite.llm.client_factory.get_llm_client",
              return_value=MagicMock(model="stub")),
        patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer",
              side_effect=failing_generate),
    ]

    # Install a real semaphore — init_http_client is mocked so the singleton
    # would otherwise be absent, causing api_call_with_retry to raise.
    import graphrag_lite.api.middleware as _mw
    _orig_sem = _mw._api_semaphore
    _mw._api_semaphore = asyncio.Semaphore(100)

    print(f"\n[circuit_breaker] threshold={threshold}  "
          f"retry_max={cfg.RETRY_MAX_ATTEMPTS}  "
          f"backoff_base={cfg.RETRY_BACKOFF_BASE_S}s")
    print(f"[circuit_breaker] Semantics: each failed logical request (exhausting "
          f"all retries) counts as ONE breaker failure.")
    print("-" * 60)

    retried_latencies: list[float] = []
    statuses: list[int] = []

    with patch("graphrag_lite.api.middleware.asyncio.sleep", new_callable=AsyncMock):
        for p in base_patches:
            p.start()
        try:
            from graphrag_lite.api.main import app
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://testserver") as client:

                # ── Drive to threshold ────────────────────────────────
                # Each logical call that exhausts all retries records ONE
                # breaker failure. We need exactly `threshold` logical calls.
                for call_n in range(1, threshold + 1):
                    t0 = time.monotonic()
                    resp = await client.post(
                        "/query",
                        json={"query": "test", "pipeline": "b", "mode": "api"},
                        timeout=30.0,
                    )
                    elapsed = time.monotonic() - t0
                    retried_latencies.append(elapsed)
                    statuses.append(resp.status_code)
                    print(f"  call {call_n:>2}: status={resp.status_code}  "
                          f"latency={elapsed*1000:.1f}ms  "
                          f"breaker={_breaker._state}  "
                          f"logical_failures={_breaker._consecutive_failures}")

                breaker_state_after_driving = _breaker._state

                # ── Fast-fail calls (breaker OPEN) ─────────────────────
                fast_fail_latencies: list[float] = []
                print(f"\n[circuit_breaker] breaker is now {breaker_state_after_driving}")
                print("[circuit_breaker] Next 3 calls should be instant 503:")
                for call_n in range(threshold + 1, threshold + 4):
                    t0 = time.monotonic()
                    resp = await client.post(
                        "/query",
                        json={"query": "test", "pipeline": "b", "mode": "api"},
                        timeout=30.0,
                    )
                    elapsed = time.monotonic() - t0
                    fast_fail_latencies.append(elapsed)
                    statuses.append(resp.status_code)
                    print(f"  call {call_n:>2}: status={resp.status_code}  "
                          f"latency={elapsed*1000:.1f}ms  (fast-fail)")

        finally:
            for p in reversed(base_patches):
                p.stop()
            _mw._api_semaphore = _orig_sem

    avg_retried = sum(retried_latencies) / len(retried_latencies) if retried_latencies else 0
    avg_fast = sum(fast_fail_latencies) / len(fast_fail_latencies) if fast_fail_latencies else 0

    print(f"\n[circuit_breaker] avg retried call: {avg_retried*1000:.1f}ms")
    print(f"[circuit_breaker] avg fast-fail:    {avg_fast*1000:.1f}ms")
    print(f"[circuit_breaker] speedup:          {avg_retried/avg_fast:.1f}×")

    # ── Assertions ────────────────────────────────────────────────────────
    # All retry-path calls should be 500 (generation failed) or 503 (breaker opened mid-series)
    for i, s in enumerate(statuses[:threshold]):
        assert s in (500, 503), f"call {i+1}: expected 500 or 503, got {s}"

    # After threshold is hit, breaker must be OPEN
    assert breaker_state_after_driving == "OPEN", (
        f"Breaker should be OPEN after {threshold} failures, "
        f"got {breaker_state_after_driving}"
    )

    # All fast-fail calls must return 503
    for i, s in enumerate(statuses[threshold:]):
        assert s == 503, f"fast-fail call {i+1}: expected 503, got {s}"

    # Fast-fail must be substantially faster than retried calls
    assert avg_fast < avg_retried, (
        f"Fast-fail ({avg_fast*1000:.1f}ms) not faster than retried ({avg_retried*1000:.1f}ms)"
    )
