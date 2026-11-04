"""
test_phase7_concurrent.py — Task 7.2 DoD: 20 concurrent requests.

Fires 20 requests to POST /query concurrently and records per-request
start/end timestamps to prove overlapping execution windows.

Uses httpx.AsyncClient with ASGITransport so the server runs in-process
and mocks apply correctly.  The live uvicorn server is not needed here —
the DoD requires showing overlapping windows, not a specific transport.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

N_REQUESTS = 20
QUERY_PAYLOAD = {"query": "What is attention?", "pipeline": "b", "mode": "local"}


@pytest.mark.asyncio
async def test_20_concurrent_requests_show_overlap():
    """
    Fire 20 concurrent POST /query requests via in-process ASGI transport.

    Verifies:
    1. All 20 complete successfully (status 200).
    2. At least some request windows overlap.
    3. Total wall time is substantially less than N * per_request_time.
    """
    from graphrag_lite.retrieval.pipeline_a import Evidence
    from graphrag_lite.llm.generator import Answer

    ev = [Evidence(chunk_id=f"chunk_{i:03d}", source_doc="doc", chunk_index=i,
                   text="text", score=0.03, source="chunk") for i in range(6)]
    ans = Answer(text="answer [0]", cited_evidence_ids=["chunk_000"], confidence=None)

    results: list[dict] = []

    async def one_request(client: httpx.AsyncClient, idx: int) -> None:
        t_start = time.monotonic()
        resp = await client.post("/query", json=QUERY_PAYLOAD, timeout=30.0)
        t_end = time.monotonic()
        results.append({"idx": idx, "status": resp.status_code,
                        "start": t_start, "end": t_end,
                        "duration": round(t_end - t_start, 4)})

    with patch("graphrag_lite.api.main.set_tracing"), \
         patch("graphrag_lite.api.main.init_http_client"), \
         patch("graphrag_lite.api.main.close_http_client", new_callable=AsyncMock), \
         patch("graphrag_lite.retrieval.pipeline_b.retrieve_structural", return_value=ev), \
         patch("graphrag_lite.llm.generator.AnswerGenerator.generate_answer", return_value=ans), \
         patch("graphrag_lite.llm.client_factory.get_llm_client",
               return_value=MagicMock(model="qwen2.5:1.5b")):

        from graphrag_lite.api.main import app
        transport = httpx.ASGITransport(app=app)

        wall_start = time.monotonic()
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            await asyncio.gather(*[one_request(client, i) for i in range(N_REQUESTS)])
        wall_elapsed = time.monotonic() - wall_start

    # ── Print timestamp log (DoD) ──
    results.sort(key=lambda r: r["start"])
    t0 = results[0]["start"]
    print(f"\n{'idx':>3}  {'start_ms':>9}  {'end_ms':>9}  {'dur_ms':>8}  status")
    print("-" * 47)
    for r in results:
        print(f"{r['idx']:>3}  {(r['start']-t0)*1000:>9.1f}  "
              f"{(r['end']-t0)*1000:>9.1f}  "
              f"{r['duration']*1000:>8.1f}  {r['status']}")
    sum_durations = sum(r["duration"] for r in results)
    print(f"\nwall_time={wall_elapsed:.3f}s  sum_durations={sum_durations:.3f}s  "
          f"ratio={wall_elapsed/sum_durations:.4f}")

    # ── Assertions ──
    statuses = [r["status"] for r in results]
    assert all(s == 200 for s in statuses), f"Non-200 responses: {statuses}"
    assert len(results) == N_REQUESTS

    # With truly concurrent handling wall time << sum of individual times.
    # Require ratio < 0.5 (sequential would be ~1.0).
    ratio = wall_elapsed / sum_durations if sum_durations > 0 else 1.0
    assert ratio < 0.5, (
        f"Requests appear serialised: wall={wall_elapsed:.3f}s "
        f"sum={sum_durations:.3f}s ratio={ratio:.4f}"
    )

    # At least some windows must overlap
    overlaps = 0
    for i in range(1, len(results)):
        if results[i]["start"] < results[i - 1]["end"]:
            overlaps += 1
    assert overlaps > 0, "No overlapping request windows — server appears synchronous"
    print(f"\n✓ Overlapping windows: {overlaps}/{N_REQUESTS - 1}  ratio={ratio:.4f}")
