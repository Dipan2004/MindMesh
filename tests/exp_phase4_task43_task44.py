"""
exp_phase4_task43_task44.py -- the Phase 4 DoD for Tasks 4.3 and 4.4.

Task 4.3 (scoped re-query)
    * Pipeline A and Pipeline B evidence side by side for 3 queries, from the
      SAME gold queries the earlier evaluation used.
    * the three possible scopes of the second (scoped) query measured against
      the 10-query gold set:
        - 'docs'      : the earlier implementation -- restricted to the source
                         documents of the candidates
        - 'candidates' : the re-query is restricted to the chunks in which the
                         ranked candidate nodes were mentioned
        - 'verbalized' : PRD S9 Step 3 verbatim -- the candidates are
                         VERBALIZED and that text becomes the dense side of
                         the second query, scoped to the candidate chunks
    * recall@6, the rank of the gold chunk, and how often the two pipelines
      return the identical evidence set.

Task 4.4 (serialization)
    * the TOON and the JSON string of the same Pipeline B evidence, both
      printed
    * the token count of each, from the REAL tokenizer in config
    * the round trip: from_toon(to_toon(e)) and from_json(to_json(e)) must
      equal the evidence that went in, field for field

Isolation: own CACHE_DIR (data/cache/exp_phase4), ZVEC_ROOT=data/cache (the
canonical collections, read-only).  The canonical cache is never cleared.

Run:  python tests/exp_phase4_task43_task44.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

CACHE_DIR = "data/cache/exp_phase4"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

from graphrag_lite.cache.llm_cache import _get_cache  # noqa: E402
from graphrag_lite.ingestion.chunker import get_tokenizer  # noqa: E402
from graphrag_lite.retrieval.pipeline_a import retrieve_flat  # noqa: E402
from graphrag_lite.retrieval.pipeline_b import get_candidates, retrieve_structural  # noqa: E402
from graphrag_lite.retrieval.serialize import (  # noqa: E402
    compare_formats,
    count_tokens,
    from_json,
    from_toon,
    to_json,
    to_toon,
)

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)

GOLD_FILE = "eval/gold_qa_set.json"
gold_doc = json.load(open(GOLD_FILE, encoding="utf-8"))
GOLD = gold_doc["queries"]
K = 6

# The three queries the DoD asks for: the first three of the gold set, so
# the side-by-side output is against a written, verified gold list.
SHOWCASE = ["Q01", "Q02", "Q03"]

print("=" * 100)
print("TASK 4.3 + 4.4 -- scoped re-query and serialization")
print("=" * 100)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"ZVEC_ROOT              : {cfg.ZVEC_ROOT}")
print(f"Gold file              : {GOLD_FILE} ({len(GOLD)} queries)")
print(f"K                      : {K}")
print(f"Tokenizer              : {cfg.TOKENIZER} (tiktoken cl100k_base)")
print("=" * 100)


def show_evidence(title: str, ev) -> None:
    print(f"\n  {title}  ({len(ev)} pieces)")
    for i, e in enumerate(ev, 1):
        snippet = e.text.replace("\n", " ")[:110]
        print(f"    [{i}] score={e.score:.5f}  {e.chunk_id}  "
              f"doc={e.source_doc}  idx={e.chunk_index}")
        print(f"        {snippet}")


# ====================================================================== #
# PART 1 -- the two scopes of the second query, over the whole gold set
# ====================================================================== #

def measure_scope(scope: str) -> dict:
    object.__setattr__(cfg, "PIPELINE_B_SCOPE", scope)
    rows = []
    for q in GOLD:
        qid, query = q["query_id"], q["query"]
        gold8 = {g[:8] for g in q["gold_chunk_ids"]}
        ev_b = retrieve_structural(query, top_k=K)
        b6 = [e.chunk_id for e in ev_b]
        rec = len(gold8 & {c[:8] for c in b6}) / len(gold8)
        ranks = [i + 1 for i, c in enumerate(b6) if c[:8] in gold8]
        rows.append({
            "query_id": qid, "query": query, "gold": sorted(gold8),
            "b6": [c[:8] for c in b6], "recall": rec,
            "best_gold_rank": min(ranks) if ranks else None,
            "n_evidence": len(b6),
        })
    avg = sum(r["recall"] for r in rows) / len(rows)
    at1 = sum(1 for r in rows if r["best_gold_rank"] == 1)
    ranks = [r["best_gold_rank"] for r in rows if r["best_gold_rank"]]
    mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
    return {"rows": rows, "avg_recall": avg, "gold_at_rank1": at1,
            "mean_best_gold_rank": mean_rank}


print("\n" + "=" * 100)
print("PART 1 -- the scope of the second (scoped) query, over the gold set")
print("=" * 100)
scopes = {}
for scope in ("docs", "candidates", "verbalized"):
    scopes[scope] = measure_scope(scope)
    s = scopes[scope]
    print(f"\n  scope = {scope!r}")
    print(f"    {'QID':<5}{'recall@6':>10}{'best gold rank':>17}{'n_ev':>6}   evidence")
    for r in s["rows"]:
        print(f"    {r['query_id']:<5}{r['recall']:>10.2f}"
              f"{str(r['best_gold_rank']):>17}{r['n_evidence']:>6}   {r['b6']}")
    print(f"    {'AVG':<5}{s['avg_recall']:>10.2f}"
          f"{s['mean_best_gold_rank']:>17.2f}"
          f"   gold at rank 1: {s['gold_at_rank1']}/{len(GOLD)}")

scope_names = list(scopes)
for i in range(len(scope_names)):
    for j in range(i + 1, len(scope_names)):
        a, b = scope_names[i], scope_names[j]
        same = sum(1 for x, y in zip(scopes[a]["rows"], scopes[b]["rows"])
                   if x["b6"] == y["b6"])
        print(f"\n  Queries where {a!r} and {b!r} return the SAME evidence list: "
              f"{same}/{len(GOLD)}")
same = sum(1 for x, y in zip(scopes["docs"]["rows"], scopes["candidates"]["rows"])
           if x["b6"] == y["b6"])

# ====================================================================== #
# PART 2 -- Pipeline A vs Pipeline B, side by side (the 4.3 DoD)
# ====================================================================== #

print("\n" + "=" * 100)
print("PART 2 -- Pipeline A vs Pipeline B, side by side (Task 4.3 DoD)")
print("=" * 100)

by_qid = {q["query_id"]: q for q in GOLD}
for qid in SHOWCASE:
    q = by_qid[qid]
    query = q["query"]
    gold8 = {g[:8] for g in q["gold_chunk_ids"]}

    ev_a = retrieve_flat(query)[:K]
    ev_by_scope = {}
    for sc in ("docs", "candidates", "verbalized"):
        object.__setattr__(cfg, "PIPELINE_B_SCOPE", sc)
        ev_by_scope[sc] = retrieve_structural(query, top_k=K)
    ev_b_cand = ev_by_scope["candidates"]

    cands = get_candidates(query)
    print("\n" + "-" * 100)
    print(f"{qid}: {query}")
    print(f"  gold chunk(s): {sorted(gold8)}")
    print(f"  entry points ({len(cands['entry_points'])}): {cands['entry_points']}")
    print(f"  BFS nodes: {len(cands['bfs_nodes'])}   "
          f"ranked candidates: {len(cands['ranked_candidates'])}")
    print(f"  top-5 candidates: "
          f"{[(n[:28], round(s, 4)) for n, s in cands['ranked_candidates'][:5]]}")

    def col(ev, i):
        if i >= len(ev):
            return ""
        e = ev[i]
        return (f"[{i+1}] {e.chunk_id[:16]} s={e.score:.5f} "
                f"{'GOLD' if e.chunk_id[:8] in gold8 else '    '} "
                f"{e.text.replace(chr(10), ' ')[:20]}")

    W = 46
    print("\n  " + "PIPELINE A (flat)".ljust(W) + "| " + "B (scope=docs)".ljust(W)
          + "| " + "B (scope=candidates)".ljust(W) + "| B (scope=verbalized)")
    print("  " + "-" * W + "+" + "-" * W + "+" + "-" * W + "+" + "-" * W)
    for i in range(max(len(ev_a), *(len(v) for v in ev_by_scope.values()))):
        print("  " + col(ev_a, i).ljust(W) + "| "
              + col(ev_by_scope["docs"], i).ljust(W) + "| "
              + col(ev_by_scope["candidates"], i).ljust(W) + "| "
              + col(ev_by_scope["verbalized"], i))
    print("  " + "-" * W + "+" + "-" * W + "+" + "-" * W + "+" + "-" * W)
    a_ids = [e.chunk_id for e in ev_a]
    b_ids = [e.chunk_id for e in ev_b_cand]
    d_ids = [e.chunk_id for e in ev_by_scope["docs"]]
    v_ids = [e.chunk_id for e in ev_by_scope["verbalized"]]
    print(f"  A@{K} ids: {[c[:8] for c in a_ids]}")
    print(f"  B@{K} (docs)        ids: {[c[:8] for c in d_ids]}")
    print(f"  B@{K} (candidates)  ids: {[c[:8] for c in b_ids]}")
    print(f"  B@{K} (verbalized)  ids: {[c[:8] for c in v_ids]}")
    print(f"  identical evidence sets A vs B(candidates): {set(a_ids) == set(b_ids)}")

# ====================================================================== #
# PART 3 -- serialization, both ways, with token counts (the 4.4 DoD)
# ====================================================================== #

print("\n" + "=" * 100)
print("PART 3 -- serialization of Task 4.3's evidence (Task 4.4 DoD)")
print("=" * 100)

q = by_qid["Q01"]
object.__setattr__(cfg, "PIPELINE_B_SCOPE", "candidates")
ev_b = retrieve_structural(q["query"], top_k=K)

toon = to_toon(ev_b)
js = to_json(ev_b)
toon_meta = to_toon(ev_b, include_text=False)
js_meta = to_json(ev_b, include_text=False)

print("\n--- TOON ---")
print(toon)
print("\n--- JSON ---")
print(js)
print("\n--- TOKEN COUNTS (real tokenizer: tiktoken cl100k_base) ---")
print(f"  TOON  full payload : {count_tokens(toon, tokenizer):>6} tokens "
      f"({len(toon):>6} chars)")
print(f"  JSON  full payload : {count_tokens(js, tokenizer):>6} tokens "
      f"({len(js):>6} chars)")
print(f"  TOON  metadata only: {count_tokens(toon_meta, tokenizer):>6} tokens "
      f"({len(toon_meta):>6} chars)")
print(f"  JSON  metadata only: {count_tokens(js_meta, tokenizer):>6} tokens "
      f"({len(js_meta):>6} chars)")
cmp = compare_formats(ev_b, tokenizer)
print(f"  saved on the full payload : "
      f"{cmp['with_text']['saved_tokens']} tokens "
      f"({cmp['with_text']['saved_pct']:.1f}%)")
print(f"  saved on the metadata only: "
      f"{cmp['metadata_only']['saved_tokens']} tokens "
      f"({cmp['metadata_only']['saved_pct']:.1f}%)")

print("\n--- ROUND TRIP (must be lossless) ---")
rt_toon = from_toon(toon)
rt_json = from_json(js)
orig = [asdict(e) for e in ev_b]


def same_records(a, b) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


print(f"  from_toon(to_toon(e)) == e        : {same_records([asdict(e) for e in rt_toon], orig)}")
print(f"  from_json(to_json(e)) == e        : {same_records([asdict(e) for e in rt_json], orig)}")
print(f"  metadata-only round trip (text=''): "
      f"{same_records([asdict(e) for e in from_toon(toon_meta)], [dict(d, text='') for d in orig])}")

out = "data/cache/exp_phase4_task43_task44.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump({
        "k": K,
        "scopes": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} | {"rows": v["rows"]}
                   for k, v in scopes.items()},
        "identical_scope_lists": same,
        "token_counts": cmp,
    }, fh, indent=2, ensure_ascii=False)

print(f"\nCACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("TASK 4.3 + 4.4 DoD complete.")
