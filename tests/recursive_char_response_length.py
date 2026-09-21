"""
Item 2 — test output-length hypothesis for recursive_char.
Runs 3 chunks from recursive_char and 3 from sliding_window,
measures raw LLM response length and per-chunk generation time.
Cache cleared so we see live generation time per chunk.
All chunks selected to be similar input size (~430-450 tokens).
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer
from graphrag_lite.ingestion.extractor import _EXTRACTION_PROMPT_TEMPLATE
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.cache.llm_cache import clear_cache
from graphrag_lite.config import get_config

tokenizer = get_tokenizer("qwen")
docs = load_documents("data/raw")

results = []
for strategy, size in [("recursive_char", "450"), ("sliding_window", "450")]:
    os.environ["CHUNKING_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE_EXTRACTION"] = size
    get_config.cache_clear()
    cfg = get_config()
    client = OllamaClient(cfg.effective_extraction_model(),
                          temperature=cfg.EXTRACTION_TEMPERATURE,
                          seed=cfg.EXTRACTION_SEED)

    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d, cfg, tokenizer))

    # Pick 3 chunks with token_count >= 400 (comparable input size)
    target_chunks = [c for c in all_chunks if c["token_count"] >= 400][:3]

    print(f"\nStrategy: {strategy}  (testing {len(target_chunks)} chunks)")
    print(f"{'idx':>4}  {'input_tok':>10}  {'resp_chars':>11}  {'resp_tok':>9}  {'time_s':>7}  {'status'}")
    print("-" * 60)

    for c in target_chunks:
        clear_cache()
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=c["text"])
        t0 = time.perf_counter()
        raw_response = client.generate(prompt)
        elapsed = time.perf_counter() - t0
        resp_chars = len(raw_response)
        resp_tokens = len(tokenizer.encode(raw_response))
        # Try parsing
        try:
            import json, re
            cleaned = re.sub(r"```(?:json)?", "", raw_response, flags=re.IGNORECASE).strip()
            start = cleaned.find("{"); end = cleaned.rfind("}")
            parsed = json.loads(cleaned[start:end+1]) if start != -1 else {}
            n_ent = len(parsed.get("entities", []))
            status = f"ok(ent={n_ent})"
        except Exception as e:
            status = f"FAIL({str(e)[:30]})"

        print(f"{c['chunk_index']:>4}  {c['token_count']:>10}  {resp_chars:>11}  "
              f"{resp_tokens:>9}  {elapsed:>7.1f}  {status}")
        results.append({
            "strategy": strategy,
            "input_tokens": c["token_count"],
            "resp_chars": resp_chars,
            "resp_tokens": resp_tokens,
            "time_s": round(elapsed, 1),
            "status": status,
        })

print()
print("Summary:")
for strat in ["recursive_char", "sliding_window"]:
    strat_results = [r for r in results if r["strategy"] == strat]
    if strat_results:
        avg_resp = sum(r["resp_tokens"] for r in strat_results) / len(strat_results)
        avg_time = sum(r["time_s"] for r in strat_results) / len(strat_results)
        print(f"  {strat:<20}  avg_resp_tokens={avg_resp:.0f}  avg_time={avg_time:.1f}s")
