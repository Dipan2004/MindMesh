"""
exp_pipeline_b_eval.py -- Pipeline A vs Pipeline B at equal k, against the
gold set written by exp_gold_set.py.

The gold set (eval/gold_qa_set.json) is read from disk. It was written by a
script that performs NO retrieval, so the gold ids cannot have been tuned
against retrieval output. This script is the first thing that runs either
pipeline.

What it measures:
  * A@6 vs B@6            -- equal k, the comparison the paper hinges on
  * A@8 (Pipeline A's native top_k) vs B@6
  * B chunks NOT in A's top-8   -- what the structural filter surfaces that
                                    flat search misses
  * top-1 / top-5 candidate identity across the 10 queries -- is the
    ranking actually query-dependent, or does one node always win?
  * type-label entry points that were filtered, and type-label nodes that
    survive into the ranked candidates

Isolation:
  * own CACHE_DIR  (data/cache/exp_pipeline_b) -- never the canonical cache
  * ZVEC_ROOT      data/cache (canonical collections, read-only)
  * own output file: data/cache/exp_pipeline_b_eval.json
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_pipeline_b_eval.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_pipeline_b"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"

from graphrag_lite.config import get_config        # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache   # noqa: E402
from graphrag_lite.retrieval.pipeline_a import retrieve_flat   # noqa: E402
from graphrag_lite.retrieval.pipeline_b import (              # noqa: E402
    get_candidates, retrieve_structural, _TYPE_LABEL_NAMES, _norm,
)

cfg = get_config()
cache = _get_cache()

GOLD_FILE = "eval/gold_qa_set.json"
if not os.path.exists(GOLD_FILE):
    print(f"{GOLD_FILE} not found -- run tests/exp_gold_set.py FIRST.")
    sys.exit(1)
gold_doc = json.load(open(GOLD_FILE, encoding="utf-8"))
GOLD = gold_doc["queries"]

K = 6
A_NATIVE = cfg.ZVEC_TOP_N_FLAT      # Pipeline A's configured top_k (8)

print("=" * 78)
print("EXP: Pipeline A vs Pipeline B at equal k, against a pre-written gold set")
print("=" * 78)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"ZVEC_ROOT              : {cfg.ZVEC_ROOT}")
print(f"Gold file              : {GOLD_FILE}  "
      f"({len(GOLD)} queries, {sum(len(q['gold_chunk_ids']) for q in GOLD)} gold chunks)")
print(f"K (equal k)            : {K}")
print(f"A native top_k         : {A_NATIVE}")
print(f"BFS_HOP_DEPTH          : {cfg.BFS_HOP_DEPTH}")
print(f"ZVEC_TOP_K_ENTRY       : {cfg.ZVEC_TOP_K_ENTRY}")
print("=" * 78)
print()

rows = []
top1 = []
top5_sets = []

for q in GOLD:
    qid, query = q["query_id"], q["query"]
    gold = set(q["gold_chunk_ids"])
    gold8 = {g[:8] for g in gold}

    ev_a_all = retrieve_flat(query)             # A's native list (top 8)
    ev_b = retrieve_structural(query, top_k=K)

    a8 = [e.chunk_id for e in ev_a_all]
    a6 = a8[:K]
    b6 = [e.chunk_id for e in ev_b]

    a8s, a6s, b6s = set(a8), set(a6), set(b6)

    rec_a = len(gold8 & {c[:8] for c in a6}) / len(gold8)
    rec_b = len(gold8 & {c[:8] for c in b6}) / len(gold8)
    rec_a8 = len(gold8 & {c[:8] for c in a8}) / len(gold8)

    # Rank of each gold chunk inside each pipeline's list (1-based). Recall
    # is saturated at 1.00 for both pipelines, so the rank position is the
    # only metric that can still discriminate between them.
    def ranks(lst):
        out = {}
        for g in sorted(gold8):
            pos = next((i + 1 for i, c in enumerate(lst) if c[:8] == g), None)
            out[g] = pos
        return out

    rank_a = ranks(a6)
    rank_b = ranks(b6)
    best_a = min((v for v in rank_a.values() if v), default=None)
    best_b = min((v for v in rank_b.values() if v), default=None)
    identical_sets = set(a6s) == set(b6s)

    # B chunks that flat search at its native top-k did not surface
    b_not_in_a8 = [c for c in b6 if c not in a8s]
    b_not_in_a6 = [c for c in b6 if c not in a6s]

    cands = get_candidates(query)
    ep = cands["entry_points"]
    ep_f = cands["entry_points_filtered"]
    bfs = cands["bfs_nodes"]
    ranked = cands["ranked_candidates"]

    if ranked:
        top1.append(ranked[0][0])
        top5_sets.append(tuple(n for n, _ in ranked[:5]))
    else:
        top1.append(None)
        top5_sets.append(())

    # type-label nodes that survived into the ranked candidates
    tl_in_ranked = [n for n, _ in ranked if _norm(n) in _TYPE_LABEL_NAMES]

    rows.append({
        "query_id": qid,
        "query": query,
        "gold": sorted(gold8),
        "entry_points": ep,
        "entry_points_filtered": ep_f,
        "bfs_count": len(bfs),
        "ranked_count": len(ranked),
        "top5_candidates": [(n, round(s, 6)) for n, s in ranked[:5]],
        "type_labels_in_ranked": tl_in_ranked,
        "a8": a8,
        "a6": a6,
        "b6": b6,
        "rank_of_gold_in_a6": rank_a,
        "rank_of_gold_in_b6": rank_b,
        "best_gold_rank_a6": best_a,
        "best_gold_rank_b6": best_b,
        "identical_evidence_sets": identical_sets,
        "recall_a6": rec_a,
        "recall_b6": rec_b,
        "recall_a8": rec_a8,
        "b_not_in_a8": b_not_in_a8,
        "b_not_in_a6": b_not_in_a6,
    })

    print(f"{qid}: {query}")
    print(f"  gold            : {sorted(gold8)}")
    print(f"  entry points    : {len(ep)}  {ep[:6]}")
    if ep_f:
        print(f"  filtered labels : {ep_f}")
    print(f"  BFS nodes       : {len(bfs)}   ranked: {len(ranked)}")
    print(f"  top-5 candidates: {[(n[:24], round(s, 5)) for n, s in ranked[:5]]}")
    if tl_in_ranked:
        print(f"  type-label nodes surviving into ranking: {tl_in_ranked}")
    print(f"  A@{K}  recall {rec_a:.2f}  {sorted(c[:8] for c in a6)}")
    print(f"  B@{K}  recall {rec_b:.2f}  {sorted(c[:8] for c in b6)}")
    print(f"  best gold rank  A@{K}: {best_a}   B@{K}: {best_b}   "
          f"identical evidence sets: {identical_sets}")
    print(f"  A@{A_NATIVE} recall {rec_a8:.2f}")
    print(f"  B chunks NOT in A's top-{A_NATIVE}: {len(b_not_in_a8)} "
          f"{[c[:8] for c in b_not_in_a8]}")
    print()

# ── summary ────────────────────────────────────────────────────────────────
avg_a6 = sum(r["recall_a6"] for r in rows) / len(rows)
avg_b6 = sum(r["recall_b6"] for r in rows) / len(rows)
avg_a8 = sum(r["recall_a8"] for r in rows) / len(rows)
a_wins = sum(1 for r in rows if r["recall_a6"] > r["recall_b6"])
b_wins = sum(1 for r in rows if r["recall_b6"] > r["recall_a6"])
ties = len(rows) - a_wins - b_wins
total_b_not_in_a8 = sum(len(r["b_not_in_a8"]) for r in rows)
total_b_chunks = sum(len(r["b6"]) for r in rows)

print("=" * 78)
print(f"SUMMARY -- A@{K} vs B@{K} over {len(rows)} gold queries")
print("=" * 78)
print(f"{'QID':<5}{'rec_A@6':>10}{'rec_B@6':>10}{'rec_A@8':>10}{'B>A':>6}"
      f"{'B_not_in_A@8':>15}")
print("-" * 66)
for r in rows:
    better = "YES" if r["recall_b6"] > r["recall_a6"] else (
        "TIE" if r["recall_b6"] == r["recall_a6"] else "no")
    print(f"{r['query_id']:<5}{r['recall_a6']:>10.2f}{r['recall_b6']:>10.2f}"
          f"{r['recall_a8']:>10.2f}{better:>6}{len(r['b_not_in_a8']):>15}")
print("-" * 66)
print(f"{'AVG':<5}{avg_a6:>10.2f}{avg_b6:>10.2f}{avg_a8:>10.2f}")
print()
print(f"A wins={a_wins}  B wins={b_wins}  ties={ties}")
print(f"B chunks total: {total_b_chunks}   NOT in A's top-{A_NATIVE}: "
      f"{total_b_not_in_a8} ({100 * total_b_not_in_a8 / total_b_chunks:.1f}%)")
print()
print("=" * 78)
print("DISCRIMINATION CHECK -- recall is saturated, so use gold RANK")
print("=" * 78)
ident = sum(1 for r in rows if r["identical_evidence_sets"])
print(f"Queries where A@{K} and B@{K} return the IDENTICAL set of chunks: "
      f"{ident}/{len(rows)}")
a_ranks = [r["best_gold_rank_a6"] for r in rows if r["best_gold_rank_a6"]]
b_ranks = [r["best_gold_rank_b6"] for r in rows if r["best_gold_rank_b6"]]
print(f"Best gold rank in A@{K}: {a_ranks}  mean {sum(a_ranks) / len(a_ranks):.2f}")
print(f"Best gold rank in B@{K}: {b_ranks}  mean {sum(b_ranks) / len(b_ranks):.2f}")
a_at1 = sum(1 for r in a_ranks if r == 1)
b_at1 = sum(1 for r in b_ranks if r == 1)
print(f"Gold at rank 1:  A@{K} = {a_at1}/{len(rows)}   B@{K} = {b_at1}/{len(rows)}")
if a_ranks == b_ranks and ident == len(rows):
    print("=> The two pipelines are INDISTINGUISHABLE on this gold set. "
          "Recall@1.00 for both is a property of the benchmark being too easy, "
          "NOT evidence that Pipeline B is equivalent to Pipeline A.")
print()
print(f"{'=' * 78}")
print("IS THE RANKING QUERY-DEPENDENT?")
print("=" * 78)
uniq_top1 = [t for t in dict.fromkeys(top1)]
print(f"Distinct top-1 candidates across {len(rows)} queries: {len(uniq_top1)}")
for t in uniq_top1:
    qs = [rows[i]["query_id"] for i in range(len(rows)) if top1[i] == t]
    print(f"    {t!r}  -> {qs}")
print()
uniq_top5 = len(set(top5_sets))
print(f"Distinct top-5 candidate sets: {uniq_top5}/{len(rows)}")
print()
tl_counter = Counter(n for r in rows for n in r["type_labels_in_ranked"])
if tl_counter:
    print("Type-label nodes surviving into the ranked candidates:")
    for n, c in tl_counter.most_common():
        print(f"    {n!r}: {c} query/queries")
else:
    print("No type-label nodes survive into the ranked candidates.")
print()

out = "data/cache/exp_pipeline_b_eval.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump({
        "gold_file": GOLD_FILE,
        "k": K,
        "a_native_top_k": A_NATIVE,
        "rows": rows,
        "summary": {
            "avg_recall_a6": avg_a6,
            "avg_recall_b6": avg_b6,
            "avg_recall_a8": avg_a8,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "b_chunks_total": total_b_chunks,
            "b_not_in_a8": total_b_not_in_a8,
            "identical_evidence_sets": ident,
            "best_gold_rank_a6": a_ranks,
            "best_gold_rank_b6": b_ranks,
            "gold_at_rank1_a6": a_at1,
            "gold_at_rank1_b6": b_at1,
            "distinct_top1": len(uniq_top1),
            "top1": top1,
            "distinct_top5_sets": uniq_top5,
            "type_labels_in_ranked": dict(tl_counter),
        },
    }, fh, indent=2, ensure_ascii=False)

print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("EXP pipeline-B eval complete.")
