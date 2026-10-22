"""Collect data for items 2-6: gleaning call count, type-label stoplist impact,
alias before/after, coverage table, confound list."""
import sys, os, re, json, io, contextlib
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE
from graphrag_lite.ingestion.gleaning import _GLEANING_PROMPT_TEMPLATE
from graphrag_lite.cache.llm_cache import _get_cache, _make_key
from graphrag_lite.graph.builder import build_graph, load_extractions, _norm, _norm_alias
import networkx as nx

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs: all_chunks.extend(chunk_document(d, cfg, tokenizer))
model = cfg.effective_extraction_model()
cache = _get_cache()

# ── Item 4: Gleaning call count reconciliation ────────────────────────────────
print("="*60)
print("ITEM 4: Gleaning cache call counts")
print("="*60)
extract_hits = extract_misses = 0
glean_hits = glean_misses = 0

for chunk in all_chunks:
    ep = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    eraw = cache.get(_make_key(ep, model))
    if eraw is None: extract_misses += 1
    else:
        extract_hits += 1
        # Gleaning prompt
        try:
            from graphrag_lite.ingestion.extractor import _extract_json
            erj = _extract_json(eraw)
        except: erj = {"entities":[], "relationships":[]}
        import json as _json
        existing = _json.dumps({"entities": erj.get("entities",[]), "relationships": erj.get("relationships",[])}, indent=2)
        gp = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing, chunk_text=chunk["text"])
        graw = cache.get(_make_key(f"GLEANING_PASS_0::{gp}", model))
        if graw is None: glean_misses += 1
        else: glean_hits += 1

total_cache = len(cache)
print(f"Cache entries total:         {total_cache}")
print(f"Extraction prompts: hits={extract_hits}  misses={extract_misses}")
print(f"Gleaning prompts:   hits={glean_hits}  misses={glean_misses}")
print(f"Gleaning call count: {glean_hits+glean_misses} (not 23 or 41 -- exact value above)")
print(f"Share of total calls: {(glean_hits+glean_misses)}/{(extract_hits+extract_misses+glean_hits+glean_misses)} = "
      f"{(glean_hits+glean_misses)*100//(extract_hits+extract_misses+glean_hits+glean_misses+1)}%")
print()

# ── Item 5: Stoplist impact without Edge ─────────────────────────────────────
print("="*60)
print("ITEM 5: Type-label stoplist impact (Edge excluded)")
print("="*60)
STOPLIST = {"person","organization","location","event","technology","other"}
# Note: Edge excluded per item 5 instruction; concept/entity/node/relationship also excluded

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G_full = build_graph(records)

stop_nodes = [n for n in G_full.nodes() if _norm(n) in STOPLIST]
print(f"Nodes matching stoplist (Edge excluded): {stop_nodes}")
G_filtered = G_full.copy()
for n in stop_nodes:
    if n in G_filtered:
        G_filtered.remove_node(n)

comps_full     = list(nx.connected_components(G_full.to_undirected()))
comps_filtered = list(nx.connected_components(G_filtered.to_undirected()))
sizes_full     = sorted([len(c) for c in comps_full], reverse=True)
sizes_filtered = sorted([len(c) for c in comps_filtered], reverse=True)

print(f"Full graph:     nodes={G_full.number_of_nodes()}  edges={G_full.number_of_edges()}  "
      f"components={len(comps_full)}  largest={sizes_full[0]}  singletons={sizes_full.count(1)}")
print(f"Filtered graph: nodes={G_filtered.number_of_nodes()}  edges={G_filtered.number_of_edges()}  "
      f"components={len(comps_filtered)}  largest={sizes_filtered[0]}  singletons={sizes_filtered.count(1)}")
print(f"Delta: nodes -{len(stop_nodes)}  edges -{G_full.number_of_edges()-G_filtered.number_of_edges()}  "
      f"components -{len(comps_full)-len(comps_filtered)}")
print()

# ── Item 6: Alias before/after table ─────────────────────────────────────────
print("="*60)
print("ITEM 6: Alias merge before/after table")
print("="*60)
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    raw_records = load_extractions("data/cache/extractions.jsonl")
    G_raw = build_graph(raw_records)
print(f"{'Metric':<30} {'Before (75)':>12} {'After (72)':>12}")
print("-"*56)
comps_raw = sorted(nx.connected_components(G_raw.to_undirected()), key=len, reverse=True)
sizes_raw = [len(c) for c in comps_raw]
comps_merged = sorted(nx.connected_components(G_full.to_undirected()), key=len, reverse=True)
sizes_merged = [len(c) for c in comps_merged]
for label, v_raw, v_merged in [
    ("Nodes", G_raw.number_of_nodes(), G_full.number_of_nodes()),
    ("Edges", G_raw.number_of_edges(), G_full.number_of_edges()),
    ("Components", len(comps_raw), len(comps_merged)),
    ("Largest component", sizes_raw[0], sizes_merged[0]),
    ("Singletons", sizes_raw.count(1), sizes_merged.count(1)),
]:
    print(f"  {label:<28} {v_raw:>12} {v_merged:>12}")
print()

# ── Item 6: Coverage table ────────────────────────────────────────────────────
print("="*60)
print("ITEM 6: Per-document coverage table")
print("="*60)

# Chunk counts per doc
chunks_by_doc = defaultdict(list)
for c in all_chunks: chunks_by_doc[c["doc_id"]].append(c)

# JSONL records
with open("data/cache/extractions.jsonl") as f:
    jsonl = [json.loads(l) for l in f if l.strip()]
by_chunk = {r["chunk_id"]: r for r in jsonl}

doc_tokens = {d["doc_id"]: len(tokenizer.encode(d["text"])) for d in docs}
total_src  = sum(doc_tokens.values())

nodes_by_doc = defaultdict(set)
for n in G_full.nodes():
    for doc in G_full.nodes[n].get("source_docs", set()):
        nodes_by_doc[doc].add(n)
edges_by_doc = defaultdict(set)
for u,v,data in G_full.edges(data=True):
    edges_by_doc[data.get("source_doc","")].add((u,v))

print(f"{'Doc':<38} {'Tok':>5} {'%':>4} {'Chu':>4} {'0ent':>5} {'Fail':>5} {'Ent':>5} {'Node':>5} {'Edge':>5}")
print("-"*80)
for doc in docs:
    did = doc["doc_id"]
    toks = doc_tokens[did]
    pct  = toks*100//total_src
    chuks = chunks_by_doc[did]
    zero = sum(1 for c in chuks if by_chunk.get(c["chunk_id"]) and
               not by_chunk[c["chunk_id"]].get("_parse_error") and
               len(by_chunk[c["chunk_id"]].get("entities",[])) == 0)
    fail = sum(1 for c in chuks if by_chunk.get(c["chunk_id"]) and
               "_parse_error" in by_chunk[c["chunk_id"]])
    raw_ent = sum(len(by_chunk[c["chunk_id"]].get("entities",[])) for c in chuks
                  if by_chunk.get(c["chunk_id"]) and "_parse_error" not in by_chunk[c["chunk_id"]])
    gr_nodes = len(nodes_by_doc[did])
    gr_edges = len(edges_by_doc[did])
    print(f"  {did:<36} {toks:>5} {pct:>4} {len(chuks):>4} {zero:>5} {fail:>5} {raw_ent:>5} {gr_nodes:>5} {gr_edges:>5}")
