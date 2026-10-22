"""
Item 3b redo: re-run _parse_extraction on every cached raw LLM response
for the 41 clean markdown_aware chunks. Log every dropped relationship.
"""
import sys, os, json, warnings, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import (
    _EXTRACTION_PROMPT_TEMPLATE,
    _extract_json, _parse_extraction
)
from graphrag_lite.ingestion.gleaning import _GLEANING_PROMPT_TEMPLATE
from graphrag_lite.cache.llm_cache import _get_cache, _make_key
from graphrag_lite.graph.builder import _norm

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs:
    all_chunks.extend(chunk_document(d, cfg, tokenizer))

model = cfg.effective_extraction_model()
cache = _get_cache()

# Build per-doc and per-chunk entity sets for resolution testing
def _norm_alias(name):
    s = name.casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1] if s.endswith("s") else s

# First pass: collect all entities per doc from cached extraction results
doc_entities = {}   # doc_id -> set of _norm(name)
doc_alias = {}      # doc_id -> set of _norm_alias(name)

for chunk in all_chunks:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    key = _make_key(prompt, model)
    raw = cache.get(key)
    if raw is None:
        continue
    try:
        rj = _extract_json(raw)
        entities = [e.get("name","").strip() for e in rj.get("entities", []) if e.get("name","").strip()]
    except Exception:
        entities = []
    doc_id = chunk["doc_id"]
    if doc_id not in doc_entities:
        doc_entities[doc_id] = set()
        doc_alias[doc_id] = set()
    for n in entities:
        doc_entities[doc_id].add(_norm(n))
        doc_alias[doc_id].add(_norm_alias(n))

# Second pass: collect dropped relationships
dropped = []
total_rels = 0
total_dropped = 0

for chunk in all_chunks:
    chunk_id = chunk["chunk_id"]
    doc_id   = chunk["doc_id"]

    # Extraction pass
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    key = _make_key(prompt, model)
    raw = cache.get(key)
    if raw is None:
        continue
    try:
        rj = _extract_json(raw)
    except Exception:
        continue

    # Chunk-level entity set
    chunk_entities = {_norm(e.get("name","")) for e in rj.get("entities",[]) if e.get("name","")}

    for rel in rj.get("relationships", []):
        src = rel.get("source","").strip()
        tgt = rel.get("target","").strip()
        if not src or not tgt:
            continue
        total_rels += 1
        src_ok = _norm(src) in chunk_entities
        tgt_ok = _norm(tgt) in chunk_entities
        if src_ok and tgt_ok:
            continue
        total_dropped += 1
        missing = []
        if not src_ok: missing.append(f"src={src!r}")
        if not tgt_ok: missing.append(f"tgt={tgt!r}")

        # Resolution checks
        src_doc = _norm(src) in doc_entities.get(doc_id, set())
        tgt_doc = _norm(tgt) in doc_entities.get(doc_id, set())
        src_alias = _norm_alias(src) in doc_alias.get(doc_id, set())
        tgt_alias = _norm_alias(tgt) in doc_alias.get(doc_id, set())

        resolve_doc   = (src_doc or src_ok) and (tgt_doc or tgt_ok)
        resolve_alias = (src_alias or src_ok) and (tgt_alias or tgt_ok)

        dropped.append({
            "chunk_id": chunk_id[:8], "doc_id": doc_id,
            "src": src, "tgt": tgt, "type": rel.get("type",""),
            "missing": missing,
            "resolve_doc":   resolve_doc,
            "resolve_alias": resolve_alias,
        })

    # Gleaning pass (also check for dropped relationships)
    for pass_num in range(cfg.MAX_GLEANINGS):
        # Build existing JSON for gleaning prompt
        existing = json.dumps({"entities": rj.get("entities",[]), "relationships": rj.get("relationships",[])}, indent=2)
        g_prompt = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing, chunk_text=chunk["text"])
        g_key = _make_key(f"GLEANING_PASS_{pass_num}::{g_prompt}", model)
        g_raw = cache.get(g_key)
        if g_raw is None:
            continue
        try:
            g_rj = _extract_json(g_raw)
        except Exception:
            continue
        for rel in g_rj.get("relationships", []):
            src = rel.get("source","").strip()
            tgt = rel.get("target","").strip()
            if not src or not tgt: continue
            total_rels += 1
            # Use combined chunk+gleaning entities for chunk set
            all_chunk_ents = chunk_entities | {_norm(e.get("name","")) for e in g_rj.get("entities",[]) if e.get("name","")}
            src_ok = _norm(src) in all_chunk_ents
            tgt_ok = _norm(tgt) in all_chunk_ents
            if src_ok and tgt_ok: continue
            total_dropped += 1
            missing = []
            if not src_ok: missing.append(f"src={src!r}")
            if not tgt_ok: missing.append(f"tgt={tgt!r}")
            src_doc   = _norm(src) in doc_entities.get(doc_id, set())
            tgt_doc   = _norm(tgt) in doc_entities.get(doc_id, set())
            src_alias = _norm_alias(src) in doc_alias.get(doc_id, set())
            tgt_alias = _norm_alias(tgt) in doc_alias.get(doc_id, set())
            resolve_doc   = (src_doc or src_ok) and (tgt_doc or tgt_ok)
            resolve_alias = (src_alias or src_ok) and (tgt_alias or tgt_ok)
            dropped.append({
                "chunk_id": chunk_id[:8], "doc_id": doc_id,
                "src": src, "tgt": tgt, "type": rel.get("type",""),
                "missing": missing,
                "resolve_doc": resolve_doc, "resolve_alias": resolve_alias,
                "from_gleaning": True,
            })

resolve_doc_count   = sum(1 for d in dropped if d["resolve_doc"])
resolve_alias_count = sum(1 for d in dropped if not d["resolve_doc"] and d["resolve_alias"])
pure_hallucinated   = sum(1 for d in dropped if not d["resolve_doc"] and not d["resolve_alias"])

print(f"Total relationships seen (before chunk-level validation): {total_rels}")
print(f"Dropped by chunk-level endpoint validation:               {total_dropped}")
print(f"  Would resolve with doc-level entity set:                {resolve_doc_count}")
print(f"  Would resolve with alias normalisation only:            {resolve_alias_count}")
print(f"  Genuinely unresolvable (hallucinated endpoints):        {pure_hallucinated}")
print()
print("15 examples of dropped relationships (judged real vs hallucinated):")
for ex in dropped[:15]:
    gl = " [gleaning]" if ex.get("from_gleaning") else ""
    if ex["resolve_doc"]:
        judgment = "REAL (doc-level match)"
    elif ex["resolve_alias"]:
        judgment = "REAL (alias match)"
    else:
        judgment = "HALLUCINATED"
    print(f"  [{ex['doc_id']}/{ex['chunk_id']}]{gl}")
    print(f"    {ex['src']!r} --{ex['type']}--> {ex['tgt']!r}")
    print(f"    missing: {ex['missing']}  => {judgment}")
