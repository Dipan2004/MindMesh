"""
exp_reextract_corpus_ladder_smoke.py — plumbing check for
exp_reextract_corpus_ladder.py, with NO LLM calls.

The real corpus ladder run is queued behind the fp=0 control arm. If the
script has a bug, that run is wasted. This exercises the same code path
with `OllamaClient.generate` stubbed:

    real extract_entities_relationships (the production ladder)
      -> real cached_call, real cache keys, real warnings, real raw dumps

The stub makes the FIRST distinct prompt fail at fp=0.25 (a looping
response) and succeed at fp=0.5, and every other prompt succeed at
fp=0.25. The expected result is therefore 42 records: 41 with retries=0
and 1 with retries=1, all status=ok, and one raw dump on disk.

IMPORTANT: this runs with its own CACHE_DIR and deletes every file it
touches afterwards, so it cannot poison the real run's cache or its
resume file.

Run:  python tests/exp_reextract_corpus_ladder_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCRATCH = ROOT / "data" / "cache" / "smoke_corpus_ladder"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True, exist_ok=True)

os.environ["EXP_LADDER_CACHE_DIR"] = str(SCRATCH)
os.environ["CACHE_DIR"] = str(SCRATCH)
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["EXP_RESUME"] = "0"

# Files the script under test writes; all removed afterwards.
OUT_JSONL = ROOT / "data" / "cache" / "exp_reextract_ladder.jsonl"
OUT_JSON = ROOT / "data" / "cache" / "exp_reextract_ladder.json"
LOG = ROOT / "data" / "cache" / "exp_reextract_ladder_log.txt"
RAW_DIR = ROOT / "data" / "cache" / "exp_reextract_ladder_raw"
SAVED = {p: p.exists() for p in (OUT_JSONL, OUT_JSON, LOG, RAW_DIR)}

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

_hard: set[str] = set()
_seen: list[str] = []


def _fake_generate(self, prompt: str) -> str:
    if prompt not in _seen:
        _seen.append(prompt)
        if len(_seen) == 1:
            _hard.add(prompt)      # the first distinct prompt is the hard one
    if prompt in _hard and abs(float(self.frequency_penalty) - 0.25) < 1e-9:
        return LOOP              # -> parse failure, escalates
    return GOOD


OllamaClient.generate = _fake_generate

import runpy  # noqa: E402

_g = runpy.run_path(str(ROOT / "tests" / "exp_reextract_corpus_ladder.py"),
                     run_name="__main__")

print()
print("=" * 78)
print("SMOKE CHECK")
print("=" * 78)

ok = True
recs = []
if OUT_JSONL.exists():
    recs = [json.loads(ln) for ln in OUT_JSONL.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    print(f"records written      : {len(recs)}")
    ok = ok and len(recs) == 42
    retried = [r for r in recs if r.get("retries", 0) > 0]
    failed = [r for r in recs if r["status"] != "ok"]
    print(f"records that retried : {len(retried)}  {[r['chunk_id'][:12] for r in retried]}")
    print(f"records that failed  : {len(failed)}")
    ok = ok and len(retried) == 1 and not failed
    for r in recs:
        if r.get("retries", 0) > 0:
            print(f"  rescued chunk      : {r['chunk_id'][:12]} "
                  f"status={r['status']} ent={r['entities']} rel={r['relationships']}")
else:
    ok = False
    print("NO OUTPUT FILE — the script did not run to completion")

summary_ok = False
if OUT_JSON.exists():
    blob = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    s = blob["summary"]
    summary_ok = (s["chunks"] == 42 and s["parsed_ok"] == 42
                  and s["failures"] == 0 and s["chunks_that_retried"] == 1)
    print(f"summary              : chunks={s['chunks']} ok={s['parsed_ok']} "
          f"failures={s['failures']} retried={s['chunks_that_retried']} "
          f"ents={s['entities']} rels={s['relationships']}")
    ok = ok and summary_ok

dumps = list(RAW_DIR.glob("*.txt")) if RAW_DIR.exists() else []
print(f"raw dumps written    : {len(dumps)}  {[d.name for d in dumps]}")
# The rescued chunk made 2 attempts (fp=0.25 then fp=0.5), so BOTH must be
# dumped -- the point of the dump is that the failed attempt is preserved,
# not just the successful one.
ok = ok and len(dumps) == 2
loops = [d for d in dumps if d.name.endswith("fp0.25.txt")]
if loops:
    body = loops[0].read_text(encoding="utf-8")
    is_loop = body.count('{"name": "Google", "type": "ORG"}') >= 5
    print(f"fp0.25 dump is a loop: {is_loop}")
    ok = ok and is_loop

# --- clean up: the real run must not see any of this ------------------- #
for p in (OUT_JSONL, OUT_JSON, LOG):
    if p.exists():
        p.unlink()
if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
# Close the diskcache handles first: on Windows an open SQLite file cannot
# be removed, and the scratch dir would survive with the stubbed responses
# still in it.
for holder in (_g.get("cache"),):
    if holder is not None:
        try:
            holder.close()
        except Exception:
            pass
try:
    from graphrag_lite.cache.llm_cache import _get_cache
    _get_cache().close()
except Exception:
    pass
shutil.rmtree(SCRATCH, ignore_errors=True)
print("smoke artifacts removed")
print("=" * 78)
print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
sys.exit(0 if ok else 1)
