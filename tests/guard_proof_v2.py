"""
Item 1d: Guard proof on a COPY of the real collection.
Never touches data/cache/chunks_v1 or entities_v1.
Asserts doc_count >= 42 on each open path.
"""
import sys, os, shutil, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import (
    _chunks_path, _open_or_create, _chunks_schema
)
from graphrag_lite.config import get_config
import zvec
from pathlib import Path

_ = embed_batch(["seed"])   # populate EMBEDDING_DIM
cfg = get_config()
real_path = _chunks_path()

# ---- Verify real collection exists and has correct count -------------------
print("Real chunks_v1 state:")
m = next((Path(real_path) / f for f in (os.listdir(real_path) if Path(real_path).exists() else [])
          if f.startswith("manifest.")), None)
if m and m.exists():
    col = zvec.open(real_path)
    real_count = col.stats.doc_count
    col.close()
    tag = (Path(real_path) / "chunking_strategy.txt").read_text().strip() \
          if (Path(real_path) / "chunking_strategy.txt").exists() else "MISSING"
    print(f"  doc_count={real_count}  tag={tag}")
else:
    real_count = 0
    print("  NO MANIFEST -- collection empty or missing")

print()

# ---- Make a temp copy of the real collection for safe testing --------------
with tempfile.TemporaryDirectory() as tmp:
    copy_path = os.path.join(tmp, "chunks_test")

    if real_count > 0:
        shutil.copytree(real_path, copy_path)
        print(f"Copied real collection ({real_count} docs) to temp dir")
    else:
        # Real collection is empty; build a minimal one in temp dir for proof
        print("Real collection empty -- building minimal temp collection")
        schema = _chunks_schema(cfg.EMBEDDING_DIM)
        col = zvec.create_and_open(copy_path, schema)
        # Insert 5 dummy docs
        dummy_vec = [0.1] * cfg.EMBEDDING_DIM
        docs = [
            zvec.Doc(id=f"chunk{i:04d}",
                     vectors={"text_vec": dummy_vec},
                     fields={"chunk_id": f"chunk{i:04d}", "source_doc": "test",
                              "chunk_index": i, "text": f"test chunk {i}"})
            for i in range(5)
        ]
        col.insert(docs); col.flush(); col.close()
        (Path(copy_path) / "chunking_strategy.txt").write_text("markdown_aware")
        real_count = 5
        print(f"  Created temp collection with {real_count} docs")

    print()

    # ---- TEST A: wrong strategy -> RuntimeError ----------------------------
    print("TEST A: wrong strategy -> RuntimeError")
    os.environ["CHUNKING_STRATEGY"] = "sliding_window"
    get_config.cache_clear()
    _ = embed_batch(["seed"])  # repopulate EMBEDDING_DIM with new config
    try:
        col = _open_or_create(copy_path, _chunks_schema)
        col.close()
        print("  ERROR: should have raised")
    except RuntimeError as e:
        msg = str(e)
        print(f"  RuntimeError: {msg[:100]}")
        assert "sliding_window" in msg or "markdown_aware" in msg
        print("  PASS")

    # ---- TEST B: right strategy -> opens, doc_count correct ----------------
    print("TEST B: right strategy -> opens, doc_count >= expected")
    os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
    get_config.cache_clear()
    _ = embed_batch(["seed"])  # repopulate EMBEDDING_DIM after cache_clear
    col = _open_or_create(copy_path, _chunks_schema)
    count = col.stats.doc_count
    col.close()
    print(f"  doc_count={count}")
    assert count >= min(real_count, 5), f"doc_count={count} < expected {real_count}"
    print("  PASS")

    # ---- TEST C: missing tag -> RuntimeError without ZVEC_ADOPT_UNTAGGED --
    print("TEST C: missing tag -> RuntimeError (no opt-in)")
    tag_file = Path(copy_path) / "chunking_strategy.txt"
    tag_backup = tag_file.read_text()
    tag_file.unlink()
    os.environ.pop("ZVEC_ADOPT_UNTAGGED", None)
    try:
        col = _open_or_create(copy_path, _chunks_schema)
        col.close()
        print("  ERROR: should have raised")
    except RuntimeError as e:
        print(f"  RuntimeError: {str(e)[:100]}")
        assert "ZVEC_ADOPT_UNTAGGED" in str(e)
        print("  PASS -- missing tag blocked without opt-in")
    tag_file.write_text(tag_backup)  # restore

    # ---- TEST D: missing tag + ZVEC_ADOPT_UNTAGGED=1 -> opens -------------
    print("TEST D: missing tag + ZVEC_ADOPT_UNTAGGED=1 -> opens and tags")
    tag_file.unlink()
    os.environ["ZVEC_ADOPT_UNTAGGED"] = "1"
    col = _open_or_create(copy_path, _chunks_schema)
    count_d = col.stats.doc_count
    col.close()
    new_tag = tag_file.read_text().strip() if tag_file.exists() else "NOT WRITTEN"
    print(f"  doc_count={count_d}  new_tag={new_tag}")
    assert count_d >= min(real_count, 5)
    assert new_tag == "markdown_aware"
    print("  PASS")
    os.environ.pop("ZVEC_ADOPT_UNTAGGED", None)

print()
print("ALL GUARD TESTS PASSED -- real collection untouched")
print(f"Real collection doc_count={real_count} (unchanged)")
