"""
Rebuild extractions_v2.jsonl from cached LLM responses only.
Uses the fixed gleaning validation (prior_entity_names fix).
No live LLM calls -- skips chunks whose prompts are not cached.
"""
import sys, os, json, warnings, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import (
    _EXTRACTION_PROMPT_TEMPLATE, _extract_json, _parse_extraction
)
from graphrag_lite.ingestion.gleaning import _GLEANING_PROMPT_TEMPLATE, _merge
from graphrag_lite.cache.llm_cache import _get_cache, _make_key

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs: all_chunks.extend(chunk_document(d, cfg, tokenizer))
model = cfg.effective_extraction_model()
cache = _get_cache()

OUT = "data/cache/extractions_v2.jsonl"
total_ents = total_rels = skipped = failed = 0
records_written = 0

with open(OUT, "w", encoding="utf-8") as fh:
    for chunk in all_chunks:
        eprompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        eraw = cache.get(_make_key(eprompt, model))
        if eraw is None:
            skipped += 1
            fh.write(json.dumps({
                "entities": [], "relationships": [],
                "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                "_parse_error": "cache_miss: no cached response"
            }) + "\n")
            continue
        try:
            erj = _extract_json(eraw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _parse_extraction(erj, chunk["chunk_id"], chunk["doc_id"])
        except Exception as e:
            failed += 1
            fh.write(json.dumps({
                "entities": [], "relationships": [],
                "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                "_parse_error": str(e)[:200]
            }) + "\n")
            continue

        # Gleaning with fix
        for pass_num in range(cfg.MAX_GLEANINGS):
            existing = json.dumps({"entities": erj.get("entities",[]), "relationships": erj.get("relationships",[])}, indent=2)
            gprompt = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing, chunk_text=chunk["text"])
            graw = cache.get(_make_key(f"GLEANING_PASS_{pass_num}::{gprompt}", model))
            if graw is None: break
            try:
                grj = _extract_json(graw)
                prior_names = {e["name"].lower() for e in result["entities"]}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    new_result = _parse_extraction(grj, chunk["chunk_id"], chunk["doc_id"],
                                                   prior_entity_names=prior_names)
                if new_result["entities"] or new_result["relationships"]:
                    result = _merge(result, new_result["entities"], new_result["relationships"])
            except Exception:
                break

        total_ents += len(result["entities"])
        total_rels += len(result["relationships"])
        fh.write(json.dumps(result) + "\n")
        records_written += 1

print(f"extractions_v2.jsonl written: {records_written} records, {skipped} cache-miss, {failed} parse-failed")
print(f"Total entities: {total_ents}  relationships: {total_rels}")
print()

# Compare with canonical extractions.jsonl
import json as json2
v1_ents = v1_rels = 0
with open("data/cache/extractions.jsonl") as f:
    for line in f:
        if not line.strip(): continue
        r = json2.loads(line)
        if "_parse_error" not in r:
            v1_ents += len(r.get("entities",[]))
            v1_rels += len(r.get("relationships",[]))
print(f"extractions.jsonl (v1): entities={v1_ents}  relationships={v1_rels}")
print(f"extractions_v2.jsonl:   entities={total_ents}  relationships={total_rels}")
print(f"Delta: entities={total_ents-v1_ents:+d}  relationships={total_rels-v1_rels:+d}")
