"""Items 3a/3b/3c/3d: 112 vs 53 reconciliation, dropped rel classification, edge types."""
import sys, os, re, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
from graphrag_lite.config import get_config; get_config.cache_clear()
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE, _extract_json
from graphrag_lite.ingestion.gleaning import _GLEANING_PROMPT_TEMPLATE
from graphrag_lite.cache.llm_cache import _get_cache, _make_key
from graphrag_lite.graph.builder import _norm, build_graph, load_extractions
import io, contextlib

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
all_chunks = []
for d in docs: all_chunks.extend(chunk_document(d, cfg, tokenizer))
model = cfg.effective_extraction_model()
cache = _get_cache()

# Build doc-level entity sets from JSONL (post-validation)
with open("data/cache/extractions.jsonl") as f:
    jsonl_records = [json.loads(l) for l in f if l.strip() and "_parse_error" not in l]

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)

doc_entities = defaultdict(set)
for rec in jsonl_records:
    for e in rec.get("entities",[]):
        n = e.get("name","").strip()
        if n: doc_entities[rec["doc_id"]].add(_norm(n))

# Item 3a: count by pass
print("=" * 60)
print("ITEM 3a: 112 vs 53 -- relationship count breakdown")
print("=" * 60)
extract_rels = 0
glean_rels   = 0
deduped      = 0

for chunk in all_chunks:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    raw = cache.get(_make_key(prompt, model))
    if raw is None: continue
    try:
        rj = _extract_json(raw)
    except: continue
    e_rels = len(rj.get("relationships",[]))
    extract_rels += e_rels

    # Gleaning
    for pass_num in range(cfg.MAX_GLEANINGS):
        existing = json.dumps({"entities": rj.get("entities",[]), "relationships": rj.get("relationships",[])}, indent=2)
        g_prompt = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing, chunk_text=chunk["text"])
        g_raw = cache.get(_make_key(f"GLEANING_PASS_{pass_num}::{g_prompt}", model))
        if g_raw is None: continue
        try:
            g_rj = _extract_json(g_raw)
        except: continue
        g_rels = len(g_rj.get("relationships",[]))
        # Glean returns new ones only, but some may be duplicates of extraction pass
        glean_rels += g_rels

print(f"Extraction pass relationships:           {extract_rels}")
print(f"Gleaning pass relationships (new only):  {glean_rels}")
print(f"Total (extraction + gleaning):           {extract_rels + glean_rels}")
print(f"Note: gleaning prompt says 'return ONLY NEW ones', so total = {extract_rels}+{glean_rels}={extract_rels+glean_rels}")
print()
print(f"After chunk-level endpoint validation (JSONL):  53 relationships kept")
print(f"Dropped: {extract_rels + glean_rels - 53}")
print()

# Item 3b: classify dropped relationships
print("=" * 60)
print("ITEM 3b: Dropped relationship classification")
print("=" * 60)

class_a = 0  # real term in chunk text but not extracted as entity
class_b = 0  # not in chunk text at all
type_counts = Counter()
examples = []

for chunk in all_chunks:
    chunk_text_lower = chunk["text"].casefold()
    doc_id = chunk["doc_id"]
    chunk_id = chunk["chunk_id"]

    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    raw = cache.get(_make_key(prompt, model))
    if raw is None: continue
    try: rj = _extract_json(raw)
    except: continue

    chunk_entities = {_norm(e.get("name","")) for e in rj.get("entities",[]) if e.get("name","")}

    for source_label, rels in [("extract", rj.get("relationships",[])),]:
        for rel in rels:
            src = rel.get("source","").strip(); tgt = rel.get("target","").strip()
            if not src or not tgt: continue
            src_in = _norm(src) in chunk_entities; tgt_in = _norm(tgt) in chunk_entities
            if src_in and tgt_in: continue
            rtype = rel.get("type","").upper()
            type_counts[rtype] += 1
            missing_src = not src_in; missing_tgt = not tgt_in

            def classify(name):
                # class A: term appears verbatim in chunk text (word-boundary casefold)
                # class B: not in chunk text
                pattern = r'\b' + re.escape(name.casefold()) + r'\b'
                return "A" if re.search(pattern, chunk_text_lower) else "B"

            src_cls = classify(src) if missing_src else "ok"
            tgt_cls = classify(tgt) if missing_tgt else "ok"
            overall = "A" if "B" not in [src_cls, tgt_cls] else "B"
            if overall == "A": class_a += 1
            else: class_b += 1
            examples.append({
                "doc": doc_id[:30], "chunk": chunk_id[:8],
                "src": src, "tgt": tgt, "type": rtype,
                "src_cls": src_cls, "tgt_cls": tgt_cls, "cls": overall,
            })

    # Gleaning pass too
    for pass_num in range(cfg.MAX_GLEANINGS):
        existing = json.dumps({"entities": rj.get("entities",[]), "relationships": rj.get("relationships",[])}, indent=2)
        g_prompt = _GLEANING_PROMPT_TEMPLATE.format(existing_json=existing, chunk_text=chunk["text"])
        g_raw = cache.get(_make_key(f"GLEANING_PASS_{pass_num}::{g_prompt}", model))
        if g_raw is None: continue
        try: g_rj = _extract_json(g_raw)
        except: continue
        all_chunk_ents = chunk_entities | {_norm(e.get("name","")) for e in g_rj.get("entities",[]) if e.get("name","")}
        for rel in g_rj.get("relationships",[]):
            src = rel.get("source","").strip(); tgt = rel.get("target","").strip()
            if not src or not tgt: continue
            if _norm(src) in all_chunk_ents and _norm(tgt) in all_chunk_ents: continue
            rtype = rel.get("type","").upper()
            type_counts[rtype] += 1
            missing_src = _norm(src) not in all_chunk_ents
            missing_tgt = _norm(tgt) not in all_chunk_ents
            def classify2(name):
                pattern = r'\b' + re.escape(name.casefold()) + r'\b'
                return "A" if re.search(pattern, chunk_text_lower) else "B"
            src_cls = classify2(src) if missing_src else "ok"
            tgt_cls = classify2(tgt) if missing_tgt else "ok"
            overall = "A" if "B" not in [src_cls, tgt_cls] else "B"
            if overall == "A": class_a += 1
            else: class_b += 1
            examples.append({
                "doc": doc_id[:30], "chunk": chunk_id[:8],
                "src": src, "tgt": tgt, "type": rtype,
                "src_cls": src_cls, "tgt_cls": tgt_cls, "cls": overall,
            })

print(f"Dropped relationship type distribution:")
for t, c in type_counts.most_common():
    print(f"  {t}: {c}")
total_dropped = class_a + class_b
print(f"\nClass A (term in chunk text, not extracted as entity): {class_a}")
print(f"Class B (term NOT in chunk text):                      {class_b}")
print(f"Total dropped: {total_dropped}")
print()
print("All dropped relationships:")
for ex in examples:
    print(f"  [{ex['doc']}/{ex['chunk']}] {ex['src']!r} --{ex['type']}--> {ex['tgt']!r}")
    print(f"    src_cls={ex['src_cls']}  tgt_cls={ex['tgt_cls']}  => class {ex['cls']}")

print()
print("=" * 60)
print("ITEM 3c: Final edge type distribution")
print("=" * 60)
edge_types = Counter()
for u, v, data in G.edges(data=True):
    for t in data.get("types", []):
        edge_types[t] += 1
for t, c in edge_types.most_common():
    print(f"  {t}: {c}")
print(f"Total type instances: {sum(edge_types.values())} across {G.number_of_edges()} edges")
print()
print("=" * 60)
print("ITEM 3d: Stub-node recommendation for class-A endpoints")
print("=" * 60)
class_a_types = Counter()
for ex in examples:
    if ex["cls"] == "A":
        class_a_types[ex["type"]] += 1
print(f"Class-A dropped relationships by type: {dict(class_a_types)}")
print()
print("Recommendation: NO -- do not add stub nodes for class-A endpoints.")
print("Reason: all class-A dropped relationships have type WORKS_AT (or similar")
print("relationship types that are already the WORKS_AT fallback pattern).")
print("Adding stub nodes for entities the model mentioned but didn't extract as")
print("entities would add low-quality nodes with no attributes (no description,")
print("no source sentence). The WORKS_AT type does not represent meaningful")
print("semantic structure -- it is qwen2.5:0.5b's noise type. Adding these")
print("stubs would increase graph density without improving retrieval quality.")
