"""
Fix 2 — proves relationship validation (option a).

Injects a raw_json dict where two relationships reference endpoints that
are NOT in the entities list, and confirms:
  - Those relationships are dropped
  - A UserWarning is emitted for each dropped relationship
  - Valid relationships (both endpoints in entities) are kept
"""
import json
import warnings

from graphrag_lite.ingestion.extractor import _parse_extraction

# Synthetic raw_json: one good relationship + two dangling ones
raw_json = {
    "entities": [
        {"name": "Wikidata", "type": "ORGANIZATION",
         "description": "A free knowledge base.", "source_sentence": "Wikidata is a free knowledge base."},
        {"name": "Wikimedia Foundation", "type": "ORGANIZATION",
         "description": "Nonprofit that runs Wikipedia.", "source_sentence": "Wikimedia Foundation runs Wikipedia."},
    ],
    "relationships": [
        # GOOD — both endpoints are in entities
        {"source": "Wikidata", "target": "Wikimedia Foundation",
         "type": "MAINTAINED_BY", "description": "Wikidata is maintained by Wikimedia Foundation.",
         "source_sentence": "Wikidata is maintained by Wikimedia Foundation."},
        # BAD — "Wikipedia" not in entities
        {"source": "Wikidata", "target": "Wikipedia",
         "type": "USED_FOR", "description": "Wikidata provides data for Wikipedia.",
         "source_sentence": "Wikidata provides data for Wikipedia."},
        # BAD — "Google Knowledge Graph" not in entities
        {"source": "Google Knowledge Graph", "target": "Wikidata",
         "type": "SOURCED_FROM", "description": "Google KG drew from Wikidata.",
         "source_sentence": "Google KG drew from Wikidata."},
    ],
}

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = _parse_extraction(raw_json, chunk_id="test_chunk", doc_id="test_doc")

print("=== Entities kept ===")
for e in result["entities"]:
    print(f"  {e['name']} ({e['type']})")

print()
print("=== Relationships kept ===")
for r in result["relationships"]:
    print(f"  {r['source']} --{r['type']}--> {r['target']}")

print()
print("=== Warnings emitted ===")
for w in caught:
    print(f"  {w.category.__name__}: {w.message}")

print()
# Assertions
assert len(result["relationships"]) == 1, (
    f"Expected 1 relationship kept, got {len(result['relationships'])}"
)
assert result["relationships"][0]["source"] == "Wikidata"
assert result["relationships"][0]["target"] == "Wikimedia Foundation"

assert len(caught) == 2, f"Expected 2 warnings, got {len(caught)}"
warning_messages = [str(w.message) for w in caught]
assert any("Wikipedia" in m for m in warning_messages), "Expected warning about Wikipedia"
assert any("Google Knowledge Graph" in m for m in warning_messages), \
    "Expected warning about Google Knowledge Graph"

print("ALL ASSERTIONS PASSED")
print()
print("Decision: Option (a) chosen over (b).")
print("Reason: Dangling relationship endpoints are most likely hallucinated entity names")
print("from a small model — silently promoting them to graph nodes (option b) would")
print("inject fabricated entities into the knowledge graph. Dropping with a warning")
print("is the safer choice; the builder (Task 2.1) only works with entities that were")
print("actually extracted and verified against the source text.")
