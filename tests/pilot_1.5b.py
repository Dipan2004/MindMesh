"""
Pilot extraction: qwen2.5:1.5b on transformer_architecture +
retrieval_augmented_generation, MAX_GLEANINGS=1 and 0.

Uses data/cache/pilot_1.5b as its own CACHE_DIR -- never touches the
canonical cache at data/cache. Writes per-setting JSONL files.
Prints cache dir and entry count at startup.
"""
import sys, os, json, warnings, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Must be set BEFORE importing graphrag_lite so get_config() picks it up
os.makedirs("data/cache/pilot_1.5b", exist_ok=True)
os.environ["CACHE_DIR"] = "data/cache/pilot_1.5b"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["SLM_MODEL"] = "qwen2.5:1.5b"

from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.graph.builder import build_graph, _norm
from graphrag_lite.cache.llm_cache import reset_stats, cache_stats, _get_cache
import networkx as nx

cfg = get_config()
cache = _get_cache()
print(f"CACHE_DIR : {cfg.CACHE_DIR}")
print(f"Cache entries at start: {len(cache)}")
print(f"Model: {cfg.effective_extraction_model()}")
print(f"num_predict: {cfg.LLM_NUM_PREDICT}")
print()

try:
    import httpx as _httpx
    import httpcore as _httpcore
    CATCH = (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError,
             _httpx.ReadTimeout, _httpcore.ReadTimeout)
except ImportError:
    CATCH = (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError)

TARGET_DOCS = {"transformer_architecture", "retrieval_augmented_generation"}

tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = [d for d in load_documents("data/raw") if d["doc_id"] in TARGET_DOCS]
print(f"Pilot docs: {[d['doc_id'] for d in docs]}")
print()

all_results = {}   # max_g -> list of records

for max_g in [1, 0]:
    print(f"\n{'='*55}")
    print(f"MAX_GLEANINGS={max_g}")
    print(f"{'='*55}")
    reset_stats()
    # NOTE: never call clear_cache() here; this run accumulates into pilot cache
    client = OllamaClient(
        cfg.effective_extraction_model(),
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S,
        num_predict=cfg.LLM_NUM_PREDICT,
    )
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d, cfg, tokenizer))

    records = []
    failed = 0
    t0 = time.perf_counter()
    for c in all_chunks:
        t_chunk = time.perf_counter()
        print(f"  [{c['doc_id']}/{c['chunk_index']}] tokens={c['token_count']}", end=" ", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = extract_entities_relationships(c, client)
            e1, r1 = len(result["entities"]), len(result["relationships"])
            t_extract = time.perf_counter() - t_chunk

            if max_g > 0:
                try:
                    result = glean(c, result, client, max_gleanings=max_g)
                except CATCH as e:
                    print(f"(glean fail: {type(e).__name__})", end=" ")
            e2, r2 = len(result["entities"]), len(result["relationships"])
            t_total = time.perf_counter() - t_chunk
            print(f"e={e1}->{e2} r={r1}->{r2} t={t_total:.1f}s", flush=True)
        except CATCH as e:
            failed += 1
            t_total = time.perf_counter() - t_chunk
            print(f"FAILED ({type(e).__name__}) t={t_total:.1f}s", flush=True)
            result = {"entities": [], "relationships": [],
                      "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                      "_parse_error": str(e)[:100]}
        records.append(result)

    elapsed = time.perf_counter() - t0
    stats = cache_stats()

    clean = [r for r in records if "_parse_error" not in r]
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(clean)
    comps = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    edge_types = []
    for u, v, data in G.edges(data=True):
        edge_types.extend(data.get("types", []))
    works_at_pct = 100 * edge_types.count("WORKS_AT") // (len(edge_types) or 1)
    self_loops = sum(
        1 for r in clean for rel in r.get("relationships", [])
        if _norm(rel.get("source","")) == _norm(rel.get("target","")) and rel.get("source","")
    )
    total_ents = sum(len(r.get("entities",[])) for r in clean)
    total_rels = sum(len(r.get("relationships",[])) for r in clean)

    print(f"\n--- MAX_GLEANINGS={max_g} summary ---")
    print(f"chunks={len(all_chunks)}  failed={failed}  elapsed={elapsed:.0f}s")
    print(f"raw_entities={total_ents}  raw_rels={total_rels}")
    print(f"self_loops(pre-graph)={self_loops}")
    print(f"graph: nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  "
          f"components={len(comps)}  largest={sizes[0] if sizes else 0}  singletons={sizes.count(1)}")
    print(f"WORKS_AT share: {edge_types.count('WORKS_AT')}/{len(edge_types)} = {works_at_pct}%")
    print(f"cache hits={stats['hits']}  misses={stats['misses']}")
    print(f"CACHE_DIR: {cfg.CACHE_DIR}")

    # Save this setting's results
    out_path = f"data/cache/extractions_pilot_1.5b_glean{max_g}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {out_path} ({len(records)} records)")
    all_results[max_g] = records
