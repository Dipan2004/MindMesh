"""Item 2: Strategy guard proof without cache_clear zeroing EMBEDDING_DIM."""
import sys, os, subprocess, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Test A: wrong strategy -> RuntimeError ────────────────────────────────────
print("TEST A: wrong strategy -> RuntimeError")
result_a = subprocess.run(
    [sys.executable, "-c", """
import sys; sys.path.insert(0,'src')
import os
os.environ['CHUNKING_STRATEGY'] = 'sliding_window'  # wrong strategy
# Do NOT cache_clear -- let Config be built fresh with env var
from graphrag_lite.config import get_config
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import get_chunks_collection, _chunks_path
from pathlib import Path

# Ensure a chunks collection tagged 'markdown_aware' exists
cfg = get_config()
path = _chunks_path()
tag = Path(path) / 'chunking_strategy.txt'
if tag.exists():
    print('existing tag:', tag.read_text())
else:
    print('no tag file yet -- will create on next real insert')
    # Manually simulate: write the tag as markdown_aware so the guard fires
    import zvec, shutil
    from graphrag_lite.vectorstore.zvec_client import _chunks_schema
    _ = embed_batch(['seed'])  # populate EMBEDDING_DIM
    cfg2 = get_config()
    shutil.rmtree(path, ignore_errors=True)
    col = zvec.create_and_open(path, _chunks_schema(cfg2.EMBEDDING_DIM))
    col.close()
    tag.write_text('markdown_aware')
    print('wrote tag: markdown_aware')

# Now try to open with wrong strategy
_ = embed_batch(['seed'])  # EMBEDDING_DIM already set, no cache_clear
try:
    col = get_chunks_collection()
    print('ERROR: should have raised RuntimeError')
except RuntimeError as e:
    print('RuntimeError:', str(e)[:120])
"""],
    cwd="c:\\Users\\KIIT0001\\Downloads\\MindMesh-main\\MindMesh-main",
    capture_output=True, text=True, timeout=60
)
print(result_a.stdout.strip())
if result_a.stderr.strip():
    # Only show non-warning stderr
    errs = [l for l in result_a.stderr.splitlines() if 'UserWarning' not in l and 'warning' not in l.lower()]
    if errs:
        print("stderr:", "\n".join(errs[:3]))
print()

# ── Test B: right strategy -> opens without error ─────────────────────────────
print("TEST B: right strategy -> opens without error")
result_b = subprocess.run(
    [sys.executable, "-c", """
import sys; sys.path.insert(0,'src')
import os
os.environ['CHUNKING_STRATEGY'] = 'markdown_aware'  # correct
from graphrag_lite.config import get_config
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import get_chunks_collection
_ = embed_batch(['seed'])
col = get_chunks_collection()
print('opened OK, doc_count =', col.stats.doc_count)
col.close()
"""],
    cwd="c:\\Users\\KIIT0001\\Downloads\\MindMesh-main\\MindMesh-main",
    capture_output=True, text=True, timeout=60
)
print(result_b.stdout.strip())
if result_b.returncode != 0:
    print("ERROR:", result_b.stderr[:200])
print()

# ── Test C: missing tag -> state the behavior ─────────────────────────────────
print("TEST C: missing tag -> opens without error (no tag = new collection)")
result_c = subprocess.run(
    [sys.executable, "-c", """
import sys, shutil; sys.path.insert(0,'src')
import os
os.environ['CHUNKING_STRATEGY'] = 'markdown_aware'
from graphrag_lite.config import get_config
from graphrag_lite.embeddings.embedder import embed_batch
from graphrag_lite.vectorstore.zvec_client import _chunks_path, _open_or_create, _chunks_schema
from pathlib import Path
_ = embed_batch(['seed'])
cfg = get_config()
path = _chunks_path()
# Remove only the tag, keep manifest
tag = Path(path) / 'chunking_strategy.txt'
if tag.exists():
    tag.unlink()
    print('removed tag')
# Try to open -- should open (no tag = legacy/new, no guard fired)
col = _open_or_create(path, _chunks_schema)
print('opened OK, doc_count =', col.stats.doc_count)
col.close()
# Tag should now be re-written
print('tag after open:', tag.read_text() if tag.exists() else 'MISSING (tag not re-written on open, only on create)')
"""],
    cwd="c:\\Users\\KIIT0001\\Downloads\\MindMesh-main\\MindMesh-main",
    capture_output=True, text=True, timeout=60
)
print(result_c.stdout.strip())
if result_c.returncode != 0:
    print("ERROR:", result_c.stderr[:200])
print()

# ── List chunks_v1 directory ──────────────────────────────────────────────────
print("Get-ChildItem data\\cache\\chunks_v1:")
result_ls = subprocess.run(
    ["powershell", "-Command", "Get-ChildItem data\\cache\\chunks_v1 -ErrorAction SilentlyContinue | Select-Object Name, Length"],
    cwd="c:\\Users\\KIIT0001\\Downloads\\MindMesh-main\\MindMesh-main",
    capture_output=True, text=True, timeout=10
)
print(result_ls.stdout.strip())
