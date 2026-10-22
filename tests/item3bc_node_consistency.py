"""Item 3b/c: assert graph node set == entities_v1 name field set."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import _entities_path
import zvec

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

graph_nodes = set(G.nodes())

# entities_v1 stores original name in the 'name' field
_ = embed_batch(["seed"])
path = _entities_path()
col = zvec.open(path)
# Fetch all docs by querying with a broad vector
from graphrag_lite.embeddings.embedder import embed_batch
q_vec = embed_batch(["entity"])[0]
results = col.query(
    queries=[zvec.Query("text_vec", vector=q_vec, param=zvec.HnswQueryParam(ef=1000))],
    topk=200,
    output_fields=["name"]
)
col.close()

zvec_names = {r.fields.get("name", r.id.replace("_", " ")) for r in results}

only_graph = graph_nodes - zvec_names
only_zvec  = zvec_names - graph_nodes

print(f"Graph nodes:       {len(graph_nodes)}")
print(f"Zvec entity names: {len(zvec_names)}")
print(f"Only in graph (not in Zvec): {sorted(only_graph)}")
print(f"Only in Zvec (not in graph): {sorted(only_zvec)}")

if not only_graph and not only_zvec:
    print("Set difference EMPTY -- graph and entities_v1 are consistent.")
else:
    print("MISMATCH -- sets differ.")
