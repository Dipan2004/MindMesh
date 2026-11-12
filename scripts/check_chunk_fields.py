import sys, pathlib
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
import zvec
col = zvec.open(str(ROOT / "data/cache/chunks_v1"))
res = col.query(queries=[zvec.Query("text_vec", vector=[0.0]*768,
                param=zvec.HnswQueryParam(ef=10))], topk=2)
for r in res:
    print("id:", r.id)
    print("fields:", dict(r.fields) if r.fields else {})
    break
col.close()

# Check what the Chunk TypedDict expects in extractor.py
from graphrag_lite.ingestion.chunker import Chunk
import inspect
print("Chunk annotation:", Chunk.__annotations__ if hasattr(Chunk, '__annotations__') else "N/A")
