"""Item 2d: Judge 15 random relationships from pilot JSONL; report type shares."""
import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter

# Use whichever pilot JSONL exists
for path in ["data/cache/extractions_pilot_1.5b_glean1.jsonl",
             "data/cache/extractions_pilot_1.5b_glean0.jsonl",
             "data/cache/extractions_pilot_1.5b.jsonl"]:
    if os.path.exists(path):
        JSONL = path
        break
else:
    print("No pilot JSONL found yet -- run pilot_1.5b.py first")
    sys.exit(0)

print(f"Reading: {JSONL}")
records = [json.loads(l) for l in open(JSONL) if l.strip() and "_parse_error" not in l]
all_rels = []
for r in records:
    for rel in r.get("relationships", []):
        all_rels.append({**rel, "_doc": r["doc_id"], "_chunk": r["chunk_id"][:8]})

print(f"Total relationships: {len(all_rels)}")
type_counts = Counter(r.get("type","?") for r in all_rels)
print(f"Type distribution:")
for t, c in type_counts.most_common():
    print(f"  {t}: {c}")

random.seed(42)
sample = random.sample(all_rels, min(15, len(all_rels)))
print(f"\n15 sampled relationships:")
for i, r in enumerate(sample):
    src = r.get("source","?")
    tgt = r.get("target","?")
    rtype = r.get("type","?")
    ssent = r.get("source_sentence","")[:80]
    # Judgement: is src and tgt real named entities (not schema terms or generic)?
    schema = {"person","organization","location","entity","other","technology","event",
              "concept","node","edge","relationship"}
    src_schema = src.lower() in schema
    tgt_schema = tgt.lower() in schema
    if src_schema or tgt_schema:
        verdict = "NOISE (schema endpoint)"
    elif rtype == "WORKS_AT":
        verdict = "NOISE (WORKS_AT fallback type)"
    elif len(ssent) < 10:
        verdict = "NOISE (no source sentence)"
    else:
        verdict = "REAL"
    print(f"  {i+1:2d}. [{r['_doc'][:25]}/{r['_chunk']}] "
          f"{src!r} --{rtype}--> {tgt!r}")
    print(f"       sent: {ssent!r}")
    print(f"       => {verdict}")
