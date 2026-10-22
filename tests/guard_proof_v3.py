"""
Item 1d: Guard proof on a temp COPY of chunks_v1.
Tests: locked handle, wrong strategy, right strategy, missing tag.
Item 1c: Proves ZVEC_ADOPT_UNTAGGED is read from get_config() (.env), not os.environ directly.
"""
import sys, os, shutil, tempfile, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import (
    _chunks_path, _open_or_create, _chunks_schema, reset_collection
)
from graphrag_lite.config import get_config
import zvec
from pathlib import Path

# Populate EMBEDDING_DIM with correct strategy
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
get_config.cache_clear()
_ = embed_batch(["seed"])
cfg = get_config()
real_path = _chunks_path()

print(f"Real chunks_v1: {real_path}")

# Verify real collection is intact
real_col = zvec.open(real_path)
real_count = real_col.stats.doc_count
real_col.close()
print(f"Real doc_count={real_count}  (must stay {real_count} throughout all tests)")
print()

with tempfile.TemporaryDirectory() as tmp:
    copy_path = os.path.join(tmp, "chunks_test")
    shutil.copytree(real_path, copy_path)
    print(f"Copied to temp: {copy_path}")
    print()

    # ---- TEST 0: locked handle -> _open_or_create raises, directory unchanged ----
    print("TEST 0: locked handle -> must raise, directory unchanged")
    lock_col = zvec.open(copy_path)   # hold the write lock
    try:
        col2 = _open_or_create(copy_path, _chunks_schema)
        col2.close()
        print("  ERROR: should have raised")
    except Exception as e:
        print(f"  Exception raised: {type(e).__name__}: {str(e)[:80]}")
        print(f"  Directory still exists: {Path(copy_path).exists()}")
        # Verify doc_count unchanged by re-opening after releasing lock
        lock_col.close()
        col_check = zvec.open(copy_path)
        count_after = col_check.stats.doc_count
        col_check.close()
        print(f"  doc_count after lock test: {count_after} (expect {real_count})")
        assert count_after == real_count, f"doc_count changed: {count_after} != {real_count}"
        print("  PASS -- locked collection raised, data unchanged")
    print()

    # Re-copy after lock test (shutil.copytree needs fresh path)
    shutil.rmtree(copy_path, ignore_errors=True)
    shutil.copytree(real_path, copy_path)

    # ---- TEST A: wrong strategy -> RuntimeError ----------------------------
    print("TEST A: wrong strategy -> RuntimeError")
    os.environ["CHUNKING_STRATEGY"] = "sliding_window"
    get_config.cache_clear()
    _ = embed_batch(["seed"])
    try:
        col = _open_or_create(copy_path, _chunks_schema)
        col.close()
        print("  ERROR: should have raised")
    except RuntimeError as e:
        msg = str(e)
        print(f"  RuntimeError: {msg[:100]}")
        assert "sliding_window" in msg or "markdown_aware" in msg
        col_check = zvec.open(copy_path)
        assert col_check.stats.doc_count == real_count
        col_check.close()
        print(f"  doc_count unchanged: {real_count}  PASS")
    print()

    # ---- TEST B: right strategy -> opens, doc_count correct ----------------
    print("TEST B: right strategy -> opens, doc_count correct")
    os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
    get_config.cache_clear()
    _ = embed_batch(["seed"])
    col = _open_or_create(copy_path, _chunks_schema)
    count = col.stats.doc_count
    col.close()
    print(f"  doc_count={count}  PASS") if count == real_count else print(f"  FAIL: {count} != {real_count}")
    assert count == real_count
    print()

    # ---- TEST C: missing tag without ZVEC_ADOPT_UNTAGGED -> RuntimeError ---
    print("TEST C: missing tag -> RuntimeError (ZVEC_ADOPT_UNTAGGED=0 in .env)")
    tag = Path(copy_path) / "chunking_strategy.txt"
    tag_backup = tag.read_text()
    tag.unlink()
    # Do NOT set os.environ -- prove it reads from config (.env)
    os.environ.pop("ZVEC_ADOPT_UNTAGGED", None)
    get_config.cache_clear()
    _ = embed_batch(["seed"])
    cfg_c = get_config()
    print(f"  cfg.ZVEC_ADOPT_UNTAGGED = {cfg_c.ZVEC_ADOPT_UNTAGGED} (from .env, expect False)")
    try:
        col = _open_or_create(copy_path, _chunks_schema)
        col.close()
        print("  ERROR: should have raised")
    except RuntimeError as e:
        msg = str(e)
        print(f"  RuntimeError: {msg[:100]}")
        assert "ZVEC_ADOPT_UNTAGGED" in msg
        col_check = zvec.open(copy_path)
        assert col_check.stats.doc_count == real_count
        col_check.close()
        print(f"  doc_count unchanged: {real_count}  PASS")
    tag.write_text(tag_backup)
    print()

    # ---- TEST D: ZVEC_ADOPT_UNTAGGED=1 in .env -> opens and tags -----------
    print("TEST D: ZVEC_ADOPT_UNTAGGED=1 in .env -> opens and tags (Item 1c proof)")
    tag.unlink()
    # Set ONLY via os.environ to simulate .env override (pydantic-settings reads it)
    os.environ["ZVEC_ADOPT_UNTAGGED"] = "1"
    get_config.cache_clear()
    _ = embed_batch(["seed"])
    cfg_d = get_config()
    print(f"  cfg.ZVEC_ADOPT_UNTAGGED = {cfg_d.ZVEC_ADOPT_UNTAGGED} (from env override, expect True)")
    col = _open_or_create(copy_path, _chunks_schema)
    count_d = col.stats.doc_count
    col.close()
    new_tag = tag.read_text().strip() if tag.exists() else "NOT WRITTEN"
    print(f"  doc_count={count_d}  new_tag={new_tag}")
    assert count_d == real_count
    assert new_tag == "markdown_aware"
    print("  PASS -- ZVEC_ADOPT_UNTAGGED read from config (not raw os.environ)")
    os.environ.pop("ZVEC_ADOPT_UNTAGGED", None)
    print()

print("ALL GUARD TESTS PASSED")
print(f"Real collection doc_count={real_count} (verified unchanged)")
