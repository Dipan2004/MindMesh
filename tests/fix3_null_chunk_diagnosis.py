"""
Fix 3 — diagnose null extraction for transformer_architecture chunk_idx=0.
Prints the source chunk text and the raw (unparsed) LLM response.
Bypasses cache to force a fresh call so we can see the actual model output.
"""
import json
import warnings

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE
from graphrag_lite.config import get_config

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
ta_doc = next(d for d in docs if d["doc_id"] == "transformer_architecture")
chunks = chunk_document(ta_doc, cfg.CHUNK_SIZE_EXTRACTION, cfg.CHUNK_OVERLAP_PCT, tokenizer)

# chunk_idx=0 is the one that returned 0 entities, 0 rels
chunk = chunks[0]

print("=== SOURCE CHUNK TEXT (chunk_id:", chunk["chunk_id"],
      "token_count:", chunk["token_count"], ") ===")
print(chunk["text"])
print()

# Call LLM directly (bypassing cache) to get the raw response
client = OllamaClient(cfg.SLM_MODEL)
prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
raw_response = client.generate(prompt)

print("=== RAW LLM RESPONSE (unprocessed) ===")
print(raw_response)
print()

# Now show what the parser makes of it
import re
from graphrag_lite.ingestion.extractor import _extract_json, _parse_extraction

try:
    raw_json = _extract_json(raw_response)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _parse_extraction(raw_json, chunk["chunk_id"], chunk["doc_id"])
    print("=== PARSED RESULT ===")
    print(json.dumps(result, indent=2))
    if caught:
        print("=== WARNINGS ===")
        for w in caught:
            print(" ", w.message)
    print()
    print(f"Entities: {len(result['entities'])}  Relationships: {len(result['relationships'])}")
except ValueError as e:
    print("=== PARSE FAILED ===")
    print(str(e))
