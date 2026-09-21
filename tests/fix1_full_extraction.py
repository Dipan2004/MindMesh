"""Fix 1 — print full untruncated source chunk text + ExtractionResult JSON."""
import json
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.config import get_config

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
kg_doc = next(d for d in docs if d["doc_id"] == "knowledge_graphs")
chunks = chunk_document(kg_doc, cfg.CHUNK_SIZE_EXTRACTION, cfg.CHUNK_OVERLAP_PCT, tokenizer)
chunk = chunks[0]

client = OllamaClient(cfg.SLM_MODEL)
result = extract_entities_relationships(chunk, client)

print("=== SOURCE CHUNK TEXT (chunk_id:", chunk["chunk_id"], "token_count:", chunk["token_count"], ") ===")
print(chunk["text"])
print()
print("=== EXTRACTION RESULT JSON ===")
print(json.dumps(result, indent=2, ensure_ascii=False))
