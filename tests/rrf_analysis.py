"""
Item 2: RRF analysis with topk=20 per signal.
For each fused result, print dense rank, FTS rank, 1/(60+r) sum.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import zvec
from graphrag_lite.vectorstore.zvec_client import _chunks_path
from graphrag_lite.embeddings.embedder import embed_batch

path = _chunks_path()
POOL = 20   # per-signal pool fed into RRF

def run_rrf_analysis(query_label: str, query_text: str):
    q_vec = embed_batch([query_text])[0]
    col = zvec.open(path)
    try:
        dense_r = col.query(
            queries=[zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64))],
            topk=POOL)
        fts_r = col.query(
            queries=[zvec.Query("text", fts=zvec.Fts(query_text))],
            topk=POOL)
        fused_r = col.query(
            queries=[
                zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64)),
                zvec.Query("text", fts=zvec.Fts(query_text)),
            ],
            topk=10, reranker=zvec.RrfReRanker())
    finally:
        col.close()

    dense_ids  = [r.id[:12] for r in dense_r]
    fts_ids    = [r.id[:12] for r in fts_r]
    fused_ids  = [r.id[:12] for r in fused_r]
    fused_scores = {r.id[:12]: r.score for r in fused_r}

    print(f"\n{'='*70}")
    print(f"QUERY {query_label}: {query_text!r}")
    print(f"Per-signal pool depth: dense={len(dense_ids)}  fts={len(fts_ids)}")
    print(f"{'='*70}")
    print(f"{'#':<3}  {'ID':12}  {'doc':22}  {'d_rank':>7}  {'f_rank':>7}  "
          f"{'RRF_sum':>10}  {'reported':>10}  {'diff':>8}")
    print("-"*85)

    RRF_K = 60
    for i, r in enumerate(fused_r):
        rid = r.id[:12]
        doc = r.fields.get("source_doc", "?")[:20]
        d_rank = dense_ids.index(rid) + 1 if rid in dense_ids else None
        f_rank = fts_ids.index(rid) + 1   if rid in fts_ids   else None
        rrf = (1/(RRF_K + d_rank) if d_rank else 0) + \
              (1/(RRF_K + f_rank) if f_rank else 0)
        d_str = str(d_rank) if d_rank else "-"
        f_str = str(f_rank) if f_rank else "-"
        diff = r.score - rrf
        print(f"  {i+1:<2}  {rid:12}  {doc:22}  {d_str:>7}  {f_str:>7}  "
              f"{rrf:>10.6f}  {r.score:>10.6f}  {diff:>+8.2e}")

    in_both = [rid for rid in fused_ids if rid in dense_ids and rid in fts_ids]
    dense_only = [rid for rid in fused_ids if rid in dense_ids and rid not in fts_ids]
    fts_only   = [rid for rid in fused_ids if rid in fts_ids  and rid not in dense_ids]
    print(f"\n  In both signals: {sorted(in_both)}")
    print(f"  Via dense only : {sorted(dense_only)}")
    print(f"  Via FTS only   : {sorted(fts_only)}")
    print(f"  FTS result count: {len(fts_ids)}")

    # Doc distribution in fused
    docs_in_fused = {}
    for r in fused_r:
        d = r.fields.get("source_doc", "?")
        docs_in_fused[d] = docs_in_fused.get(d, 0) + 1
    print(f"  Docs in fused top-10: {dict(sorted(docs_in_fused.items()))}")

run_rrf_analysis("(a) bare HNSW", "HNSW")
run_rrf_analysis("(b) semantic paraphrase", "ways to find similar items quickly in huge collections")
run_rrf_analysis("(c) cross-document",      "Qwen language model embeddings knowledge graph")
