import sys, time, io, contextlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"LightRAG")
from pathlib import Path
from graphrag_lite.graph.builder import build_graph, load_extractions
from graphrag_lite.entity_resolution.resolver import EntityResolver
from graphrag_lite.entity_resolution.aliases import AliasRegistry

ext = Path("data/experiments/qwen2_5_1_5b_genuine_gleaning_on/extractions.jsonl")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    recs = load_extractions(str(ext))
print(f"Records loaded: {len(recs)}")

t0 = time.monotonic()
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    G = build_graph(recs)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.monotonic()-t0:.1f}s)")

t1 = time.monotonic()
reg = AliasRegistry.with_builtins()
resolver = EntityResolver(G, registry=reg)
configs = resolver.run_all_configs()
print(f"Configs built in {time.monotonic()-t1:.1f}s")
for name, cfg in configs.items():
    m = cfg.metrics()
    print(f"  {name:22}: nodes={m['nodes']} edges={m['edges']} cross_doc_edges={m['cross_doc_edges']} merges={m['merges_applied']}")

# Now evaluate retrieval
from graphrag_lite.graph.ranking import rank_pagerank
import json
dev = json.loads(Path("eval/gold_dev_set.json").read_text(encoding="utf-8"))["queries"]
exp_dir = Path("data/experiments/qwen2_5_1_5b_validate")
print(f"\nEvaluating LightRAG Local on {len(dev)} queries...")

from graphrag_lite.retrieval.lightrag import retrieve_local

for name, cfg in configs.items():
    pr = rank_pagerank(cfg.G)
    recalls, mrrs = [], []
    d18_hit = False
    for q in dev:
        gold = q.get("gold_chunk_ids", [])
        try:
            ev = retrieve_local(q["query"], cfg.G, pr, exp_dir, top_k=6)
            ids = [e["chunk_id"] for e in ev]
        except Exception:
            ids = []
        rec = sum(1 for g in gold if g in ids)/len(gold) if gold else 1.0
        mrr = next((1.0/i for i, c in enumerate(ids, 1) if c in set(gold)), 0.0)
        recalls.append(rec)
        mrrs.append(mrr)
        if q["query_id"] == "D18":
            d18_hit = any(g in ids for g in gold)
    n = len(dev)
    avg_rec = sum(recalls)/n
    avg_mrr = sum(mrrs)/n
    ug = len({cid for q in dev for cid, r in zip(ids, [True]) if r})
    print(f"  {name:22}: Recall={avg_rec:.4f} MRR={avg_mrr:.4f} D18={d18_hit}")

print("\nDone.")
