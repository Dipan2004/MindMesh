"""Item 1b: Measure gleaning validation bug impact from cached responses."""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE, _extract_json
from graphrag_lite.ingestion.gleaning import _GLEANING_PROMPT_TEMPLATE
from graphrag_lite.cache.llm_cache import _get_cache, _make_key

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs: all_chunks.extend(chunk_document(d, cfg, tokenizer))
model = cfg.effective_extraction_model()
cache = _get_cache()

def norm(n): return n.strip().lower()

total_glean_rels = 0
would_pass_merged = 0    # passes if validated against merged entity set
currently_passes  = 0    # passes with current code (only gleaned entities)
entities_added_by_gleaning = 0

for chunk in all_chunks:
    # Get first-pass result
    eprompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    eraw = cache.get(_make_key(eprompt, model))
    if eraw is None: continue
    try: erj = _extract_json(eraw)
    except: continue
    first_pass_ents = {norm(e.get("name","")) for e in erj.get("entities",[]) if e.get("name","")}

    # Get gleaning result
    existing_json = json.dumps({"entities": erj.get("entities",[]), "relationships": erj.get("relationships",[])}, indent=2)
    gprompt = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing_json, chunk_text=chunk["text"])
    graw = cache.get(_make_key(f"GLEANING_PASS_0::{gprompt}", model))
    if graw is None: continue
    try: grj = _extract_json(graw)
    except: continue

    gleaned_ents = {norm(e.get("name","")) for e in grj.get("entities",[]) if e.get("name","")}
    new_ents = gleaned_ents - first_pass_ents
    entities_added_by_gleaning += len(new_ents)
    merged_ents = first_pass_ents | gleaned_ents

    for rel in grj.get("relationships",[]):
        src = rel.get("source","").strip()
        tgt = rel.get("target","").strip()
        if not src or not tgt: continue
        total_glean_rels += 1

        # Current behaviour: validate against gleaned_ents only
        if norm(src) in gleaned_ents and norm(tgt) in gleaned_ents:
            currently_passes += 1

        # Fixed behaviour: validate against merged set
        if norm(src) in merged_ents and norm(tgt) in merged_ents:
            would_pass_merged += 1

print(f"Total gleaned relationships (before any validation):  {total_glean_rels}")
print(f"Currently pass (validated against gleaned_ents only): {currently_passes}")
print(f"Would pass with fix (vs merged first_pass+gleaned):   {would_pass_merged}")
print(f"Additional relationships unlocked by fix:             {would_pass_merged - currently_passes}")
print(f"Total new entities added by gleaning across corpus:   {entities_added_by_gleaning}")
print()
print("Bug confirmed: gleaning validates relationships against ONLY the gleaned")
print("entity set, not the merged (first-pass + gleaned) set.")
print("Relationships where source is in first-pass entities but not gleaned")
print("entities are incorrectly dropped.")
