"""
Item 3: Cross-doc names from raw extractions + dropped relationship analysis.
No new LLM calls. Reads extractions.jsonl and cache only.
"""
import sys, os, re, json
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.graph.builder import _norm

def _norm_alias(name: str) -> str:
    s = name.casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("s"):
        s = s[:-1]
    return s

JSONL = "data/cache/extractions.jsonl"

with open(JSONL) as f:
    records = [json.loads(l) for l in f if l.strip() and "_parse_error" not in l]

print("=" * 65)
print("ITEM 3a: Normalised entity names appearing in >1 doc")
print("=" * 65)

# Per-doc entity sets (raw names, normalised)
doc_entities: dict[str, set] = defaultdict(set)
for rec in records:
    doc_id = rec.get("doc_id", "")
    for e in rec.get("entities", []):
        name = e.get("name", "").strip()
        if name:
            doc_entities[doc_id].add(_norm(name))

# Find names in >1 doc
all_norms = defaultdict(set)  # norm_name -> set of doc_ids
for doc_id, names in doc_entities.items():
    for n in names:
        all_norms[n].add(doc_id)

cross_doc = {k: v for k, v in all_norms.items() if len(v) > 1}
print(f"Normalised entity names appearing in >1 doc: {len(cross_doc)}")
for name, docs in sorted(cross_doc.items(), key=lambda x: -len(x[1]))[:20]:
    print(f"  {name!r:40s}  docs={sorted(docs)}")
print()

print("=" * 65)
print("ITEM 3b: Dropped relationship analysis")
print("=" * 65)

# Build per-doc entity set (all entities extracted, normalised)
doc_entity_set: dict[str, set] = defaultdict(set)  # doc_id -> set of norm names
for rec in records:
    doc_id = rec.get("doc_id", "")
    for e in rec.get("entities", []):
        name = e.get("name", "").strip()
        if name:
            doc_entity_set[doc_id].add(_norm(name))

# Also build alias-normalised per-doc set
doc_alias_set: dict[str, set] = defaultdict(set)
for doc_id, names in doc_entity_set.items():
    for n in names:
        doc_alias_set[doc_id].add(_norm_alias(n))

total_rels = 0
dropped_in_builder = 0
would_resolve_doc_level = 0
would_resolve_alias = 0
examples = []

for rec in records:
    doc_id = rec.get("doc_id", "")
    chunk_id = rec.get("chunk_id", "")
    chunk_entities = {_norm(e.get("name","")) for e in rec.get("entities", []) if e.get("name")}
    chunk_alias = {_norm_alias(e.get("name","")) for e in rec.get("entities", []) if e.get("name")}

    for rel in rec.get("relationships", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        if not src or not tgt:
            continue
        total_rels += 1

        src_norm = _norm(src)
        tgt_norm = _norm(tgt)
        src_in_chunk = src_norm in chunk_entities
        tgt_in_chunk = tgt_norm in chunk_entities

        if src_in_chunk and tgt_in_chunk:
            continue  # would be kept by builder

        # Was dropped by builder (at least one endpoint missing from chunk entities)
        dropped_in_builder += 1

        # Would it resolve at doc level?
        src_in_doc = src_norm in doc_entity_set.get(doc_id, set())
        tgt_in_doc = tgt_norm in doc_entity_set.get(doc_id, set())
        if src_in_doc and tgt_in_doc:
            would_resolve_doc_level += 1
            judgment = "REAL (doc-level match)"
        else:
            # Would alias normalisation help?
            src_alias = _norm_alias(src)
            tgt_alias = _norm_alias(tgt)
            src_in_alias = src_alias in doc_alias_set.get(doc_id, set())
            tgt_in_alias = tgt_alias in doc_alias_set.get(doc_id, set())
            if src_in_alias and tgt_in_alias:
                would_resolve_alias += 1
                judgment = "REAL (alias match)"
            else:
                judgment = "HALLUCINATED (no match)"

        if len(examples) < 15:
            missing = []
            if not src_in_chunk:
                missing.append(f"src={src!r}")
            if not tgt_in_chunk:
                missing.append(f"tgt={tgt!r}")
            examples.append({
                "doc": doc_id, "chunk": chunk_id[:8],
                "src": src, "tgt": tgt,
                "type": rel.get("type",""),
                "missing": missing,
                "judgment": judgment,
            })

print(f"Total relationships in JSONL:                   {total_rels}")
print(f"Dropped by chunk-level endpoint validation:     {dropped_in_builder}")
print(f"  Would resolve with doc-level entity set:      {would_resolve_doc_level}")
print(f"  Would resolve with alias normalisation:       {would_resolve_alias}")
print(f"  Genuinely unresolvable (hallucinated):        "
      f"{dropped_in_builder - would_resolve_doc_level - would_resolve_alias}")
print()
print("15 examples of dropped relationships:")
for ex in examples:
    print(f"  [{ex['doc']}/{ex['chunk']}] "
          f"{ex['src']!r} --{ex['type']}--> {ex['tgt']!r}")
    print(f"    missing: {ex['missing']}  judgment: {ex['judgment']}")
