"""Task 1.2 verification script — run directly with python."""
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.config import get_config

cfg = get_config()
tokenizer = get_tokenizer(cfg.TOKENIZER)
docs = load_documents("data/raw")
doc = docs[0]

chunks = chunk_document(doc, cfg, tokenizer)

doc_id = doc["doc_id"]
print("Doc:", doc_id)
print("Total chunks:", len(chunks))
print()

overlap_tokens = int(cfg.CHUNK_SIZE_EXTRACTION * cfg.CHUNK_OVERLAP_PCT)
print(
    "Config: size=%d  overlap_pct=%.2f  expected_overlap_tokens=%d"
    % (cfg.CHUNK_SIZE_EXTRACTION, cfg.CHUNK_OVERLAP_PCT, overlap_tokens)
)
print()

all_size_ok = True
for i, c in enumerate(chunks):
    over = 0
    if i > 0:
        prev = chunks[i - 1]
        over = prev["end_offset"] - c["start_offset"]
    within_size = c["token_count"] <= cfg.CHUNK_SIZE_EXTRACTION
    if not within_size:
        all_size_ok = False
    print(
        "  chunk[%d] tokens=%4d  start=%5d  end=%5d  overlap_with_prev=%3d  size_ok=%s"
        % (i, c["token_count"], c["start_offset"], c["end_offset"], over, within_size)
    )

print()
assert all_size_ok, "Some chunks exceeded CHUNK_SIZE_EXTRACTION!"
print("All chunks within size limit:", all_size_ok)
print("ALL ASSERTIONS PASSED")
