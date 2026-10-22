"""Item 2b/c: Show the 9 formerly-dropped relationships and whether they duplicate existing first-pass triples."""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE, _extract_json
from graphrag_lite.ingestion.gleaning import _GLEANING_PROMPT_TEMPLATE
from graphrag_lite.cache.llm_cache import _get_cache, _make_key
import time

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs: all_chunks.extend(chunk_document(d, cfg, tokenizer))
model = cfg.effective_extraction_model()
cache = _get_cache()

def norm(n): return n.strip().lower()

newly_valid = []          # 9 relationships now passing with fix
entities_added = 0
rels_added = 0
glean_calls = 0
t0 = time.perf_counter()

for chunk in all_chunks:
    eprompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    eraw = cache.get(_make_key(eprompt, model))
    if eraw is None: continue
    try: erj = _extract_json(eraw)
    except: continue

    first_pass_ents = {norm(e.get("name","")) for e in erj.get("entities",[]) if e.get("name","")}
    first_pass_triples = {(norm(r.get("source","")), norm(r.get("target","")))
                          for r in erj.get("relationships",[])
                          if r.get("source") and r.get("target")}

    existing = json.dumps({"entities": erj.get("entities",[]), "relationships": erj.get("relationships",[])}, indent=2)
    gprompt = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing, chunk_text=chunk["text"])
    graw = cache.get(_make_key(f"GLEANING_PASS_0::{gprompt}", model))
    if graw is None: continue
    glean_calls += 1

    try: grj = _extract_json(graw)
    except: continue

    gleaned_ents = {norm(e.get("name","")) for e in grj.get("entities",[]) if e.get("name","")}
    new_ents = gleaned_ents - first_pass_ents
    entities_added += len(new_ents)

    merged_ents = first_pass_ents | gleaned_ents

    for rel in grj.get("relationships",[]):
        src = rel.get("source","").strip()
        tgt = rel.get("target","").strip()
        if not src or not tgt: continue
        rtype = rel.get("type","")

        old_passes  = norm(src) in gleaned_ents and norm(tgt) in gleaned_ents
        new_passes  = norm(src) in merged_ents and norm(tgt) in merged_ents

        if new_passes and not old_passes:
            triple = (norm(src), norm(tgt))
            is_dup = triple in first_pass_triples
            newly_valid.append({
                "chunk": chunk["chunk_id"][:8], "doc": chunk["doc_id"],
                "src": src, "tgt": tgt, "type": rtype,
                "duplicates_first_pass": is_dup,
            })
            if not is_dup:
                rels_added += 1

elapsed = time.perf_counter() - t0
total_glean_ents_added = entities_added
total_new_rels = rels_added

print(f"Gleaning calls (cache hits): {glean_calls}")
print(f"Wall time (cache only):       {elapsed:.2f}s")
print(f"Entities added by gleaning:   {entities_added}")
print(f"New unique (src,tgt) pairs not in first pass: {total_new_rels}")
print()
print(f"9 formerly-dropped relationships now passing with fix:")
for r in newly_valid:
    dup_note = "DUPLICATE of first-pass triple" if r["duplicates_first_pass"] else "NEW pair"
    print(f"  [{r['doc']}/{r['chunk']}] {r['src']!r} --{r['type']}--> {r['tgt']!r}")
    print(f"    => {dup_note}")
print()
print(f"Summary: {len(newly_valid)} now valid; "
      f"{sum(1 for r in newly_valid if r['duplicates_first_pass'])} duplicate first-pass triples; "
      f"{total_new_rels} genuinely new (src,tgt) pairs")
