"""Rebuild entities_v1 with alias-merged graph."""
import sys, os, io, re, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import defaultdict
from graphrag_lite.graph.builder import build_graph, load_extractions, _norm
from graphrag_lite.graph.ranking import rank_pagerank
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import insert_entities, _entities_path, reset_collection

def _norm_alias(name):
    s = name.casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1] if s.endswith("s") else s

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

# Apply alias merge
groups = defaultdict(list)
for n in G.nodes(): groups[_norm_alias(n)].append(n)
merge_map = {}
for nodes in groups.values():
    if len(nodes) > 1:
        canon = sorted(nodes)[0]
        for n in nodes:
            if n != canon: merge_map[n] = canon

for old, canon in merge_map.items():
    if old not in G: continue
    od, cd = G.nodes[old], G.nodes[canon]
    cd["descriptions"].extend(od.get("descriptions", []))
    cd["source_sentences"].extend(od.get("source_sentences", []))
    cd["source_docs"].update(od.get("source_docs", set()))
    if cd.get("type") == "OTHER" and od.get("type") != "OTHER":
        cd["type"] = od["type"]
    for _, tgt, ed in list(G.out_edges(old, data=True)):
        rt = merge_map.get(tgt, tgt)
        if rt != canon and not G.has_edge(canon, rt): G.add_edge(canon, rt, **ed)
    for src, _, ed in list(G.in_edges(old, data=True)):
        rs = merge_map.get(src, src)
        if rs != canon and not G.has_edge(rs, canon): G.add_edge(rs, canon, **ed)
    G.remove_node(old)

print(f"Merged graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pr = rank_pagerank(G)
entity_dicts = []
for node in G.nodes():
    nd = dict(G.nodes[node]); nd["name"] = node
    entity_dicts.append(nd)

entity_texts = []
for ed in entity_dicts:
    d = ed.get("descriptions", [])
    desc = d[0]["text"] if d else ""
    entity_texts.append(f"{ed['name']}: {desc}" if desc else ed["name"])

print("Embedding...", flush=True)
vecs = embed_batch(entity_texts)

epath = _entities_path()
reset_collection(epath)
insert_entities(entity_dicts, vecs, pagerank_scores=pr)
print(f"entities_v1 rebuilt: {len(entity_dicts)} entities inserted")

import zvec
col = zvec.open(epath)
print(f"doc_count in entities_v1: {col.stats.doc_count}")
col.close()
