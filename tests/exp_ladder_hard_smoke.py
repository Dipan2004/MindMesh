"""
exp_ladder_hard_smoke.py — plumbing check for exp_ladder_hard.py, with NO
LLM calls.

The real `exp_ladder_hard.py` is queued behind a multi-hour corpus run. If
it has a bug, that run is wasted. This script exercises the same code path
with `OllamaClient.generate` stubbed out, so the whole chain runs for real:

    real extract_entities_relationships
      -> real ladder, real warnings, real cached_call, real cache keys
        -> stubbed generate (no network, no server)

The stub returns a LOOPING response at fp=0.25 and a valid JSON at fp=0.5,
so the expected result for every chunk is:
    attempt 0 (fp=0.25) -> parse failure, loop mode
    attempt 1 (fp=0.5)  -> parse success
    => status=ok, attempts_used=2, rescued=True

Run:  python tests/exp_ladder_hard_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCRATCH = ROOT / "data" / "cache" / "smoke_ladder_hard"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True, exist_ok=True)

os.environ["EXP_LADDER_CACHE_DIR"] = str(SCRATCH)
os.environ["EXP_REP"] = "99"
os.environ["CACHE_DIR"] = str(SCRATCH)
os.environ["ZVEC_ROOT"] = "data/cache"

# --- stub the network layer BEFORE the script imports anything -------- #
from graphrag_lite.llm.ollama_client import OllamaClient  # noqa: E402

LOOP = "\n".join(['{"name": "Google", "type": "ORG"}'] * 12)
GOOD = json.dumps({
    "entities": [
        {"name": "A", "type": "CONCEPT", "description": "d", "source_sentence": "s"},
        {"name": "B", "type": "CONCEPT", "description": "d", "source_sentence": "s"},
    ],
    "relationships": [
        {"source": "A", "target": "B", "type": "REL",
         "description": "d", "source_sentence": "s"},
    ],
})

calls = []


def _fake_generate(self, prompt: str) -> str:
    calls.append((self.frequency_penalty, self.cache_tag))
    if abs(float(self.frequency_penalty) - 0.25) < 1e-9:
        return LOOP          # -> parse failure, loop mode
    return GOOD              # -> parses


OllamaClient.generate = _fake_generate

# --- run the script under test --------------------------------------- #
import runpy  # noqa: E402

script = ROOT / "tests" / "exp_ladder_hard.py"
runpy.run_path(str(script), run_name="__main__")

# --- check what it wrote ---------------------------------------------- #
print()
print("=" * 78)
print("SMOKE CHECK")
print("=" * 78)

out = ROOT / "data" / "cache" / "exp_ladder_hard.jsonl"
if out.exists():
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # keep only the smoke records (rep 99)
    recs = [json.loads(ln) for ln in lines]
    recs = [r for r in recs if r.get("rep") == 99]
    print(f"records written      : {len(recs)}")
    ok = True
    for r in recs:
        modes = [a["mode"] for a in r["per_attempt"]]
        good = (r["status"] == "ok" and r["attempts_used"] == 2
                and r["rescued"] is True and modes[0] == "loop"
                and modes[1] == "ok")
        ok = ok and good
        print(f"  {r['chunk_id'][:12]} status={r['status']} attempts={r['attempts_used']} "
              f"rescued={r['rescued']} modes={modes} "
              f"ent={r['entities']} rel={r['relationships']}  "
              f"{'OK' if good else 'MISMATCH'}")
    # drop the smoke records so the real run's file is clean
    keep = [ln for ln in lines if ln.strip() and json.loads(ln).get("rep") != 99]
    out.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    print(f"smoke records removed : {len(recs)}")
else:
    ok = False
    print("NO OUTPUT FILE — the script did not run to completion")

print(f"generate() calls      : {len(calls)} "
      f"(penalties {[c[0] for c in calls]})")
print(f"distinct cache tags   : {len({c[1] for c in calls})}")
print("=" * 78)
print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
sys.exit(0 if ok else 1)
