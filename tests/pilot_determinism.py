"""Item 2b: Determinism check + gleaning analysis for 1.5b pilot."""
import sys, os, json, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
from graphrag_lite.cache.llm_cache import _get_cache

cfg = get_config()
print(f"CACHE_DIR: {cfg.CACHE_DIR}  entries: {len(_get_cache())}")
print()

tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
ta = next(d for d in docs if d["doc_id"] == "transformer_architecture")
ta_chunks = chunk_document(ta, cfg, tokenizer)
client = OllamaClient(cfg.effective_extraction_model(), temperature=0.0, seed=42,
                      timeout=cfg.LLM_TIMEOUT_S, num_predict=cfg.LLM_NUM_PREDICT)

# ── Determinism: run 3 chunks first-pass twice ────────────────────────────────
print("="*60)
print("DETERMINISM: 3 chunks extracted twice (same result?)")
print("="*60)

DET_IDXS = [0, 1, 2]   # skip timed-out 3 and 4
for idx in DET_IDXS:
    c = ta_chunks[idx]
    print(f"\nchunk idx={idx}  id={c['chunk_id'][:12]}  tokens={c['token_count']}")
    runs = []
    for run in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = extract_entities_relationships(c, client)
        names = [e["name"] for e in r["entities"]]
        runs.append(names)
        print(f"  run {run+1}: {len(r['entities'])} entities {names[:5]}")
    same = runs[0] == runs[1]
    print(f"  Deterministic: {same}")

# ── Gleaning analysis: print before/after for idx 0 and 2 ────────────────────
print()
print("="*60)
print("GLEANING: first-pass vs post-glean for idx=0 and idx=2")
print("="*60)
for idx in [0, 2]:
    c = ta_chunks[idx]
    print(f"\nchunk idx={idx}  id={c['chunk_id'][:12]}  tokens={c['token_count']}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp = extract_entities_relationships(c, client)
    print(f"  First-pass:  {len(fp['entities'])} entities, {len(fp['relationships'])} relationships")
    for e in fp["entities"]:
        print(f"    E: {e['name']!r} ({e['type']})")
    for r in fp["relationships"]:
        print(f"    R: {r['source']!r} --{r['type']}--> {r['target']!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pg = glean(c, fp, client, max_gleanings=1)
    new_e = len(pg["entities"]) - len(fp["entities"])
    new_r = len(pg["relationships"]) - len(fp["relationships"])
    print(f"  Post-glean:  {len(pg['entities'])} entities (+{new_e}), {len(pg['relationships'])} relationships (+{new_r})")
    for e in pg["entities"][len(fp["entities"]):]:
        print(f"    NEW E: {e['name']!r} ({e['type']})")
    for r in pg["relationships"][len(fp["relationships"]):]:
        print(f"    NEW R: {r['source']!r} --{r['type']}--> {r['target']!r}")
