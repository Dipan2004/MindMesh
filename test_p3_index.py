"""
test_p3_index.py — Validate genuine upstream LightRAG indexing and query modes.

Uses qwen2.5:1.5b for LLM calls and nomic-embed-text for embeddings.
Indexes data/raw/ into an isolated directory: data/experiments/p3_validation/

GPU-intensive: each document requires LLM extraction.
Run manually: python test_p3_index.py
"""
import sys, asyncio, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "LightRAG")
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT / "data" / "experiments" / "p3_validation"
WORKDIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = ROOT / "data" / "raw"

async def main():
    from lightrag import LightRAG, QueryParam
    import httpx

    print("=== Pipeline 3 Genuine LightRAG v1.5.8 Validation ===\n")

    # Build async LLM function using qwen2.5:1.5b via Ollama
    async def ollama_llm(prompt, system_prompt=None, history_messages=[], **kw):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in (history_messages or []):
            messages.append(m)
        messages.append({"role": "user", "content": prompt})
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                "http://localhost:11434/api/chat",
                json={"model": "qwen2.5:1.5b", "messages": messages,
                      "stream": False, "options": {"temperature": 0.0, "seed": 42}},
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def ollama_embed_raw(texts):
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "http://localhost:11434/api/embed",
                json={"model": "nomic-embed-text", "input": texts},
            )
            r.raise_for_status()
            return r.json()["embeddings"]

    from lightrag.utils import EmbeddingFunc
    ollama_embed = EmbeddingFunc(embedding_dim=768, func=ollama_embed_raw)

    # Init LightRAG
    print("Initializing LightRAG...")
    rag = LightRAG(
        working_dir=str(WORKDIR),
        llm_model_func=ollama_llm,
        embedding_func=ollama_embed,
    )
    print(f"Working dir: {WORKDIR}")
    print(f"KV storage: {type(rag.full_docs).__name__}")
    print(f"Graph storage: {type(rag.chunk_entity_relation_graph).__name__}")
    print()

    # Index documents
    docs = sorted([f for f in RAW_DIR.iterdir() if f.suffix in (".txt", ".md")])
    print(f"Indexing {len(docs)} documents...")
    index_start = time.monotonic()
    for i, doc_path in enumerate(docs, 1):
        text = doc_path.read_text(encoding="utf-8")
        print(f"  [{i}/{len(docs)}] {doc_path.name} ({len(text)} chars)...", end="", flush=True)
        t0 = time.monotonic()
        await rag.ainsert(text)
        print(f" done ({time.monotonic()-t0:.1f}s)")
    index_total = time.monotonic() - index_start
    print(f"\nIndexing complete: {index_total:.1f}s\n")

    # Validate storage
    print("Storage validation:")
    kv_files = list(WORKDIR.glob("kv_store_*.json"))
    graph_files = list(WORKDIR.glob("graph_chunk_entity_relation.graphml")) + \
                  list(WORKDIR.glob("*.graphml"))
    print(f"  KV store files: {len(kv_files)}")
    print(f"  Graph files: {len(graph_files)}")
    for kf in kv_files:
        data = json.loads(kf.read_text(encoding="utf-8"))
        print(f"  {kf.name}: {len(data)} entries")
    print()

    # Test all query modes
    test_q = "What is hybrid search and how does it combine vector search with keyword search?"
    gold_related = "895ee8b439ed7d7b"  # D18 gold chunk
    modes = ["naive", "local", "global", "hybrid", "mix"]
    results = {}

    print(f"Test query: {test_q[:70]}...\n")
    for mode in modes:
        print(f"  Mode={mode}...", end="", flush=True)
        t0 = time.monotonic()
        try:
            result = await rag.aquery(
                test_q,
                param=QueryParam(mode=mode, top_k=6, only_need_context=True)
            )
            lat = time.monotonic() - t0
            results[mode] = {"ok": True, "latency_s": round(lat,3),
                             "result_type": type(result).__name__,
                             "result_len": len(str(result)),
                             "preview": str(result)[:150]}
            print(f" OK ({lat:.1f}s, {len(str(result))} chars)")
        except Exception as exc:
            lat = time.monotonic() - t0
            results[mode] = {"ok": False, "latency_s": round(lat,3), "error": str(exc)[:100]}
            print(f" ERROR: {exc}")

    print()
    print("Query mode results:")
    for mode, r in results.items():
        status = "OK" if r["ok"] else "FAIL"
        lat = r["latency_s"]
        if r["ok"]:
            print(f"  {status} {mode:8} {lat:.1f}s  {r['result_len']} chars")
            print(f"       Preview: {r['preview'][:100]}")
        else:
            print(f"  {status} {mode:8} {lat:.1f}s  ERROR: {r.get('error','')[:80]}")

    # Save results
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "qwen2.5:1.5b", "embed": "nomic-embed-text",
        "workdir": str(WORKDIR),
        "docs_indexed": len(docs),
        "index_time_s": round(index_total, 1),
        "kv_files": len(kv_files), "graph_files": len(graph_files),
        "query_modes": results,
    }
    out_path = WORKDIR / "p3_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")

asyncio.run(main())
