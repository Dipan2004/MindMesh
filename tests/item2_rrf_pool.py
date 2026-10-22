"""Item 2: RRF pool depth at topk=5,10,20,50 for query (b)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import zvec
from graphrag_lite.vectorstore.zvec_client import _chunks_path
from graphrag_lite.embeddings.embedder import embed_batch

path = _chunks_path()
QUERY = "ways to find similar items quickly in huge collections"
q_vec = embed_batch([QUERY])[0]

for topk in [5, 10, 20, 50]:
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
    print(f"\ntopk={topk}  dense_pool={len(d_ids)}  fts_pool={len(f_ids)}  fused={len(list(fused))}")
    for i, r in enumerate(fused):
        rid = r.id[:12]
        dr = d_ids.index(rid)+1 if rid in d_ids else "-"
        fr = f_ids.index(rid)+1 if rid in f_ids else "-"
        rrf = ((1/(60+d_ids.index(rid)+1) if rid in d_ids else 0) +
               (1/(60+f_ids.index(rid)+1) if rid in f_ids else 0))
        doc = r.fields.get("source_doc","?")[:18]
        print(f"  {i+1:2d}  {rid}  d={str(dr):>4}  f={str(fr):>4}  rrf={rrf:.6f}  rep={r.score:.6f}  doc={doc}")
