"""
Task 1.3 verification — runs extraction on one real chunk against the local SLM.
Prints the full ExtractionResult JSON plus the source chunk text for manual verification.
"""
import json

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.config import get_config
from graphrag_lite.cache.llm_cache import clear_cache

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)

# Use the knowledge_graphs doc — richest entity/relationship density
docs = load_documents("data/raw")
kg_doc = next(d for d in docs if d["doc_id"] == "knowledge_graphs")

chunks = chunk_document(kg_doc, cfg.CHUNK_SIZE_EXTRACTION, cfg.CHUNK_OVERLAP_PCT, tokenizer)
chunk = chunks[0]   # first chunk — intro + history section

print("=" * 60)
print("SOURCE CHUNK TEXT:")
print("=" * 60)
print(chunk["text"])
print()
print("chunk_id :", chunk["chunk_id"])
print("token_count:", chunk["token_count"])
print()

client = OllamaClient(cfg.SLM_MODEL)
print(f"Running extraction with model: {cfg.SLM_MODEL}")
print("(first call hits the LLM; subsequent runs will be cached)")
print()

result = extract_entities_relationships(chunk, client)

print("=" * 60)
print("EXTRACTION RESULT:")
print("=" * 60)
print(json.dumps(result, indent=2))
print()
print(f"Entities found    : {len(result['entities'])}")
print(f"Relationships found: {len(result['relationships'])}")

assert len(result["entities"]) >= 2, "Expected at least 2 entities"
assert len(result["relationships"]) >= 1, "Expected at least 1 relationship"
print()
print("ALL ASSERTIONS PASSED")
