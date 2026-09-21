"""
Temperature stability test — Point 1.
Runs extraction on transformer_architecture chunk_idx=0 twice in a row
with temperature=0.0, bypassing cache each time, and confirms identical output.
"""
import json
import warnings
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships, _EXTRACTION_PROMPT_TEMPLATE, _extract_json, _parse_extraction
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.config import get_config

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
ta_doc = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta_doc, cfg.CHUNK_SIZE_EXTRACTION, cfg.CHUNK_OVERLAP_PCT, tokenizer)
chunk = chunks[0]

print("chunk_id :", chunk["chunk_id"])
print("doc_id   :", chunk["doc_id"])
print("chunk_idx:", chunk["chunk_index"])
print("temperature:", cfg.EXTRACTION_TEMPERATURE)
print("seed       :", cfg.EXTRACTION_SEED)
print()

client = OllamaClient(cfg.effective_extraction_model(), temperature=cfg.EXTRACTION_TEMPERATURE, seed=cfg.EXTRACTION_SEED)
prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])

results = []
for run in range(1, 3):
    print(f"=== RUN {run} (direct LLM call, no cache) ===")
    raw = client.generate(prompt)
    print("Raw response:")
    print(raw)
    print()
    try:
        parsed = _extract_json(raw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _parse_extraction(parsed, chunk["chunk_id"], chunk["doc_id"])
        entities = len(result["entities"])
        rels = len(result["relationships"])
        print(f"Parsed: entities={entities}  relationships={rels}")
        if caught:
            for w in caught:
                print(f"  Warning: {w.message}")
        results.append({"entities": entities, "rels": rels, "raw": raw})
    except ValueError as e:
        print(f"PARSE FAILED: {e}")
        results.append({"entities": "FAILED", "rels": "FAILED", "raw": raw})
    print()

print("=== STABILITY CHECK ===")
print(f"Run 1: entities={results[0]['entities']}  rels={results[0]['rels']}")
print(f"Run 2: entities={results[1]['entities']}  rels={results[1]['rels']}")
stable = (results[0]["raw"] == results[1]["raw"])
print(f"Raw responses identical: {stable}")
if stable:
    print("STABLE — temperature=0.0 produces deterministic output")
else:
    print("NOT STABLE — responses differ despite temperature=0.0")
    print("Diff (first 200 chars where they diverge):")
    for i, (a, b) in enumerate(zip(results[0]["raw"], results[1]["raw"])):
        if a != b:
            print(f"  char {i}: run1={repr(a)}  run2={repr(b)}")
            print(f"  context: ...{results[0]['raw'][max(0,i-20):i+50]}...")
            break
