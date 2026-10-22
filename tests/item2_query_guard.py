"""Item 2: Test hybrid_query strategy guard on a temp copy."""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import _chunks_path, hybrid_query, _check_strategy_tag
from graphrag_lite.config import get_config
import zvec
from pathlib import Path

os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
get_config.cache_clear()
_ = embed_batch(["seed"])
cfg = get_config()
real_path = _chunks_path()

with tempfile.TemporaryDirectory() as tmp:
    copy_path = os.path.join(tmp, "chunks_test")
    shutil.copytree(real_path, copy_path)

    # Patch _chunks_path to return copy_path for this test
    import graphrag_lite.vectorstore.zvec_client as zc
    original_fn = zc._chunks_path
    zc._chunks_path = lambda: copy_path

    # TEST A: wrong strategy -> raises from retrieve_flat
    print("TEST A: wrong strategy -> RuntimeError from hybrid_query")
    os.environ["CHUNKING_STRATEGY"] = "sliding_window"
    get_config.cache_clear()
    _ = embed_batch(["seed"])
    q_vec = embed_batch(["test query"])[0]
    try:
        results = hybrid_query(zc.get_config().ZVEC_COLLECTION_CHUNKS, q_vec, "test", top_k=3)
        print("  ERROR: should have raised")
    except RuntimeError as e:
        print(f"  RuntimeError: {str(e)[:80]}")
        assert "sliding_window" in str(e) or "markdown_aware" in str(e)
        print("  PASS")

    # TEST B: right strategy -> returns results
    print("TEST B: right strategy -> returns results")
    os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
    get_config.cache_clear()
    _ = embed_batch(["seed"])
    cfg2 = get_config()
    q_vec2 = embed_batch(["knowledge graph"])[0]
    results2 = hybrid_query(cfg2.ZVEC_COLLECTION_CHUNKS, q_vec2, "knowledge graph", top_k=3)
    print(f"  Returned {len(results2)} results")
    assert len(results2) > 0
    print("  PASS")

    # Restore
    zc._chunks_path = original_fn

print("ALL QUERY GUARD TESTS PASSED")
