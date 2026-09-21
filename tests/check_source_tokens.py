"""Item 1 — verify true source token count vs ablation figure."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import get_tokenizer

tokenizer = get_tokenizer("qwen")
docs = load_documents("data/raw")

total = 0
for d in docs:
    tc = len(tokenizer.encode(d["text"]))
    print(f"  {d['doc_id']:<45}  {tc} tokens")
    total += tc

print()
print(f"TRUE total source tokens (raw docs, no overlap): {total}")
print(f"Ablation 'dedup_check' reported:                6058")
print(f"Match: {total == 6058}")
print()

# Now check what the ablation script actually does
# It calls: sum(len(tokenizer.encode(d["text"])) for d in docs)
# which is exactly the same — so if total != 6058, the ablation was wrong
# Let's also check what summing CHUNK token counts would give for each strategy
import os
for strategy, size in [("sliding_window","450"), ("recursive_char","450"),
                        ("markdown_aware","450"), ("small_window","250")]:
    os.environ["CHUNKING_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE_EXTRACTION"] = size
    from graphrag_lite.config import get_config
    get_config.cache_clear()
    cfg = get_config()
    from graphrag_lite.ingestion.chunker import chunk_document
    chunk_total = 0
    for d in docs:
        for c in chunk_document(d, cfg, tokenizer):
            chunk_total += c["token_count"]
    print(f"  {strategy:<20}  sum(chunk token counts)={chunk_total}  "
          f"ratio={chunk_total/total:.3f}x source tokens")
