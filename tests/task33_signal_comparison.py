"""
Task 3.3 addendum: 3 queries against chunks_v1, each run dense-only, FTS-only,
and fused (RRF), printed side by side. Tests that both signals contribute.
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import zvec
from graphrag_lite.vectorstore.zvec_client import _chunks_path
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.config import get_config

cfg = get_config()
path = _chunks_path()

def query_signals(query_text: str, top_k: int = 5):
    """Run dense-only, FTS-only, and fused against chunks_v1. Returns 3 lists."""
    q_vec = embed_batch([query_text])[0]
    col = zvec.open(path)
    try:
        dense  = col.query(queries=[zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64))], topk=top_k)
        fts    = col.query(queries=[zvec.Query("text", fts=zvec.Fts(query_text))], topk=top_k)
        fused  = col.query(queries=[
                     zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64)),
                     zvec.Query("text",     fts=zvec.Fts(query_text)),
                 ], topk=top_k, reranker=zvec.RrfReRanker())
    finally:
        col.close()
    def fmt(results):
        return [(r.id[:12], round(r.score, 5), r.fields.get("source_doc","?")) for r in results]
    return fmt(dense), fmt(fts), fmt(fused)


def print_side_by_side(label: str, query: str, dense, fts, fused):
    print(f"\n{'='*70}")
    print(f"QUERY ({label}): {query!r}")
    print(f"{'='*70}")
    print(f"{'#':<3}  {'DENSE':38}  {'FTS':38}  {'FUSED (RRF)':38}")
    print("-"*120)
    for i in range(5):
        d = f"{dense[i][0]} {dense[i][1]:>9.5f} {dense[i][2][:14]}" if i < len(dense) else "-"
        f_ = f"{fts[i][0]} {fts[i][1]:>9.5f} {fts[i][2][:14]}" if i < len(fts) else "-"
        u = f"{fused[i][0]} {fused[i][1]:>9.5f} {fused[i][2][:14]}" if i < len(fused) else "-"
        print(f"  {i+1}  {d:<38}  {f_:<38}  {u:<38}")
    # Analysis
    dense_ids  = {r[0] for r in dense}
    fts_ids    = {r[0] for r in fts}
    fused_ids  = {r[0] for r in fused}
    only_dense = fused_ids & dense_ids - fts_ids
    only_fts   = fused_ids & fts_ids - dense_ids
    both       = fused_ids & dense_ids & fts_ids
    print(f"\n  Lists differ (dense vs FTS): {dense_ids != fts_ids}")
    print(f"  In fused only via dense (not in FTS top-5) : {sorted(fused_ids & dense_ids - fts_ids)}")
    print(f"  In fused only via FTS   (not in dense top-5): {sorted(fused_ids & fts_ids - dense_ids)}")
    print(f"  In both (fused from either signal)          : {sorted(both)}")


# ── Query (a): lexical-only -- rare exact token "HNSW" ──────────────────────
dense_a, fts_a, fused_a = query_signals("HNSW approximate nearest neighbor index")

# ── Query (b): semantic-only paraphrase ─────────────────────────────────────
dense_b, fts_b, fused_b = query_signals("ways to find similar items quickly in huge collections")

# ── Query (c): cross-document ────────────────────────────────────────────────
dense_c, fts_c, fused_c = query_signals("Qwen language model embeddings knowledge graph")

print_side_by_side("(a) lexical/rare token",
    "HNSW approximate nearest neighbor index", dense_a, fts_a, fused_a)
print_side_by_side("(b) semantic paraphrase",
    "ways to find similar items quickly in huge collections", dense_b, fts_b, fused_b)
print_side_by_side("(c) cross-document",
    "Qwen language model embeddings knowledge graph", dense_c, fts_c, fused_c)

print("\n\nSUMMARY")
print("-"*40)
for label, d, f, u in [
    ("(a) lexical",   dense_a, fts_a,  fused_a),
    ("(b) semantic",  dense_b, fts_b,  fused_b),
    ("(c) cross-doc", dense_c, fts_c,  fused_c),
]:
    d_ids = {r[0] for r in d}
    f_ids = {r[0] for r in f}
    differ = d_ids != f_ids
    fusion_adds = bool((({r[0] for r in u} & d_ids - f_ids) | ({r[0] for r in u} & f_ids - d_ids)))
    print(f"  {label}: dense!=FTS={differ}  fusion_adds_unique={fusion_adds}")
print("\nDONE")
