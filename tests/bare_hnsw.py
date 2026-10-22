import sys; sys.path.insert(0, "src")
import zvec
from graphrag_lite.vectorstore.zvec_client import _chunks_path
from graphrag_lite.embeddings.embedder import embed_batch

path = _chunks_path()
q_vec = embed_batch(["HNSW"])[0]
col = zvec.open(path)
dense = col.query(queries=[zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64))], topk=5)
fts   = col.query(queries=[zvec.Query("text", fts=zvec.Fts("HNSW"))], topk=5)
fused = col.query(queries=[
    zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=64)),
    zvec.Query("text", fts=zvec.Fts("HNSW")),
], topk=5, reranker=zvec.RrfReRanker())
col.close()

for label, r in [("dense", dense), ("fts  ", fts), ("fused", fused)]:
    rows = [(x.id[:12], round(x.score, 5), x.fields.get("source_doc", "?")[:18]) for x in r]
    print(f"{label}: {rows}")

d_ids = set(x.id[:12] for x in dense)
f_ids = set(x.id[:12] for x in fts)
u_ids = set(x.id[:12] for x in fused)
print("Lists differ (dense vs FTS):", d_ids != f_ids)
print("In fused via dense only:", sorted(u_ids & d_ids - f_ids))
print("In fused via FTS only:  ", sorted(u_ids & f_ids - d_ids))
