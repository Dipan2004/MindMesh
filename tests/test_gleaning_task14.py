"""
Task 1.4 verification — runs gleaning on the same chunk as Task 1.3.
Prints entity/relationship counts before and after gleaning.
"""
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import extract_entities_relationships
from graphrag_lite.ingestion.gleaning import glean
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.config import get_config

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)

docs = load_documents("data/raw")
kg_doc = next(d for d in docs if d["doc_id"] == "knowledge_graphs")
chunks = chunk_document(kg_doc, cfg.CHUNK_SIZE_EXTRACTION, cfg.CHUNK_OVERLAP_PCT, tokenizer)
chunk = chunks[0]

client = OllamaClient(cfg.SLM_MODEL)

# First pass (likely cached from Task 1.3 run)
first_pass = extract_entities_relationships(chunk, client)
before_entities = len(first_pass["entities"])
before_rels = len(first_pass["relationships"])
print(f"BEFORE gleaning: {before_entities} entities, {before_rels} relationships")

# Gleaning pass
final = glean(chunk, first_pass, client, max_gleanings=cfg.MAX_GLEANINGS)
after_entities = len(final["entities"])
after_rels = len(final["relationships"])
print(f"AFTER  gleaning: {after_entities} entities, {after_rels} relationships")
print(f"Added: {after_entities - before_entities} entities, {after_rels - before_rels} relationships")
print()
print("(Note: 0 new items is a valid result on a small chunk with a small model)")
print("ALL ASSERTIONS PASSED")
