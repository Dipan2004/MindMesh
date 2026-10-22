"""
Offline tests for the ladder-measurement tooling.

`tests/exp_ladder_hard_summary.py` produces the headline rescue-rate
number (P(attempt 0 fails), P(rescued | attempt 0 failed), P(ultimately
fails)), so it is worth a test that the arithmetic is right. These tests
feed it a synthetic records file and check the reported rates.

They also guard the trap this session fell into: an arm whose record
carries `raw_relationships` (pre-filter) must not be compared against an
arm whose record carries only the post-filter `relationships`.

Run:  python -m pytest tests/test_ladder_summary.py -q
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _write_records(tmp_path: Path, recs: list[dict]) -> Path:
    p = tmp_path / "exp_ladder_hard.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return p


def _rec(rep, cid, status, attempts, modes, ents=5, rels=3):
    return {
        "rep": rep,
        "chunk_id": cid,
        "doc_id": "d",
        "chunk_index": 0,
        "status": status,
        "attempts_used": attempts,
        "rescued": status == "ok" and attempts > 1,
        "entities": ents,
        "relationships": rels,
        "seconds": 10.0,
        "per_attempt": [
            {"attempt": i, "penalty": p, "mode": m, "chars": 100}
            for i, (p, m) in enumerate(modes)
        ],
        "error": "" if status == "ok" else "ValueError",
    }


def test_rescue_rate_arithmetic(tmp_path, monkeypatch, capsys):
    """3 of 12 attempt-0 failures, all 3 rescued -> 100% rescue rate."""
    recs = []
    for rep in (0, 1, 2):
        # one chunk loops at attempt 0 and is rescued at attempt 1
        recs.append(_rec(rep, "aaaa", "ok", 2, [(0.25, "loop"), (0.5, "ok")]))
        # three chunks pass at attempt 0
        recs.append(_rec(rep, "bbbb", "ok", 1, [(0.25, "ok")]))
        recs.append(_rec(rep, "cccc", "ok", 1, [(0.25, "ok")]))
        recs.append(_rec(rep, "dddd", "ok", 1, [(0.25, "ok")]))
    path = _write_records(tmp_path, recs)

    script = ROOT / "tests/exp_ladder_hard_summary.py"
    monkeypatch.setattr(
        sys, "argv", ["exp_ladder_hard_summary.py"],
    )
    # point the script at the synthetic file
    src = script.read_text(encoding="utf-8")
    src = src.replace(
        'OUT = ROOT / "data/cache/exp_ladder_hard.jsonl"',
        f'OUT = Path(r"{path}")',
    )
    tmp_script = tmp_path / "summary_run.py"
    tmp_script.write_text(src, encoding="utf-8")

    runpy.run_path(str(tmp_script), run_name="__main__")
    out = capsys.readouterr().out

    assert "P(attempt 0 fails)          = 3/12" in out
    assert "P(rescued | attempt 0 failed) = 3/3" in out
    assert "P(chunk ultimately fails)   = 0/12" in out


def test_raw_counts_are_preferred_over_filtered():
    """
    The trap: an arm's `relationships` may be post-filter while another
    arm's is pre-filter. `arm_stats` must prefer `raw_relationships`.
    """
    src = (ROOT / "tests/exp_compare_arms.py").read_text(encoding="utf-8")
    assert "raw_relationships" in src, "compare_arms must read the raw counts"
    assert "raw_entities" in src

    # And the ladder records must actually carry them.
    path = ROOT / "data/cache/exp_reextract_ladder.jsonl"
    if not path.exists():
        pytest.skip("ladder run not present")
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "raw_relationships" in first, (
        "ladder records must carry raw_relationships; re-run "
        "tests/exp_backfill_raw_counts.py"
    )
    assert "raw_entities" in first


def test_ladder_records_are_complete():
    """42 chunks, all with a raw-count backfill, and the raw >= kept."""
    path = ROOT / "data/cache/exp_reextract_ladder.jsonl"
    if not path.exists():
        pytest.skip("ladder run not present")
    recs = [
        json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(recs) == 42
    assert all("raw_relationships" in r for r in recs)
    assert all(r["raw_relationships"] >= max(r["relationships"], 0) for r in recs)
    assert sum(1 for r in recs if r["status"] == "ok") == 42
    retried = [r["chunk_id"] for r in recs if r.get("retries")]
    assert len(retried) == 2, f"expected 2 retried chunks, got {retried}"
