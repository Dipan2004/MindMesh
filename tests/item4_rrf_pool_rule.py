"""Item 4: Find the deepest dense/FTS rank contributing at each topk value."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import zvec
from graphrag_lite.vectorstore.zvec_client import _chunks_path
from graphrag_lite.embeddings.embedder import embed_batch

path = _chunks_path()
QUERY = "ways to find similar items quickly in huge collections"
q_vec = embed_batch([QUERY])[0]

print(f"Query: {QUERY!r}")
print(f"{'topk':>6}  {'fused_count':>11}  {'max_dense_rank':>14}  {'max_fts_rank':>12}  max(topk,10)?")
print("-" * 65)

for topk in list(range(1, 13)) + [15, 20]:
    col = zvec.open(path)
    dense = col.query(queries=[zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64))], topk=topk)
    fts   = col.query(queries=[zvec.Query("text", fts=zvec.Fts(QUERY))], topk=topk)
    fused = col.query(queries=[
        zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64)),
        zvec.Query("text", fts=zvec.Fts(QUERY)),
    ], topk=topk, reranker=zvec.RrfReRanker())
    col.close()

    d_ids = [r.id[:12] for r in dense]
    f_ids = [r.id[:12] for r in fts]
    fused_ids = [r.id[:12] for r in fused]

    max_d_rank = max((d_ids.index(fid)+1 for fid in fused_ids if fid in d_ids), default=0)
    max_f_rank = max((f_ids.index(fid)+1 for fid in fused_ids if fid in f_ids), default=0)
    expected   = max(topk, 10)
    consistent = max_d_rank <= expected and max_f_rank <= expected

    print(f"{topk:>6}  {len(fused_ids):>11}  {max_d_rank:>14}  {max_f_rank:>12}  {'yes' if consistent else 'NO'} (rule={expected})")
