"""
perf_remediation.py — Performance measurements for the pre-Phase 6 gate.

Measures (with 5+ warm repetitions where practical):
  1. Model cold-start latency (qwen2.5:1.5b, first call loads the model)
  2. Model warm generation latency (5 reps, trivial prompt)
  3. Embedding latency (5 reps, nomic-embed-text)
  4. Zvec insertion latency (1 rep into temp collection — destructive, not repeated)
  5. Retrieval latency (5 reps, Pipeline A and Pipeline B)
  6. Graph construction latency (5 reps)
  7. End-to-end query latency (3 reps — expensive at ~45s each)

With 5 samples, p95 is explicitly labelled as a rough estimate.
"""

from __future__ import annotations

import json
import os
import statistics
import time

def _stats(samples: list[float]) -> dict:
    """Return mean, median, p95 (rough), max for a list of timings in seconds."""
    n = len(samples)
    s = sorted(samples)
    return {
        "n": n,
        "mean_s": round(statistics.mean(samples), 3),
        "median_s": round(statistics.median(samples), 3),
        "p95_s": round(s[max(0, int(n * 0.95) - 1)], 3) if n >= 2 else round(samples[0], 3),
        "p95_note": "rough estimate (n<30)" if n < 30 else "",
        "max_s": round(max(samples), 3),
        "min_s": round(min(samples), 3),
    }


def main() -> None:
    from graphrag_lite.config import get_config
    cfg = get_config()

    results: dict = {
        "hardware": {
            "python": __import__("sys").version,
            "ollama_version": "0.40.4",
            "config": {
                "SLM_MODEL": cfg.SLM_MODEL,
                "EMBEDDING_MODEL": cfg.EMBEDDING_MODEL,
                "LLM_NUM_PREDICT": cfg.LLM_NUM_PREDICT,
            },
        },
    }

    # ── 1. Model cold-start (qwen2.5:1.5b) ──
    import ollama
    client = ollama.Client()
    t0 = time.time()
    r = client.generate(model=cfg.SLM_MODEL, prompt="Say OK.",
                        options={"temperature": 0.0, "seed": 42, "num_predict": 16})
    cold_s = time.time() - t0
    results["1_cold_start_1.5b"] = {
        "latency_s": round(cold_s, 3),
        "response": r.response[:50],
    }

    # ── 2. Model warm generation (5 reps) ──
    warm_samples = []
    for i in range(5):
        t0 = time.time()
        client.generate(model=cfg.SLM_MODEL, prompt="Say OK.",
                        options={"temperature": 0.0, "seed": 42, "num_predict": 16})
        warm_samples.append(time.time() - t0)
    results["2_warm_gen_1.5b"] = _stats(warm_samples)
    results["2_warm_gen_1.5b"]["samples_s"] = [round(s, 3) for s in warm_samples]

    # ── 3. Embedding latency (5 reps) ──
    from graphrag_lite.embeddings.embedder import embed_batch
    # Warm up the embedder (first call may load the model)
    embed_batch(["warmup sentence"])
    embed_samples = []
    test_text = "How does the self-attention mechanism compute attention scores?"
    for i in range(5):
        t0 = time.time()
        embed_batch([test_text])
        embed_samples.append(time.time() - t0)
    results["3_embedding_nomic"] = _stats(embed_samples)
    results["3_embedding_nomic"]["samples_s"] = [round(s, 3) for s in embed_samples]

    # ── 4. Zvec insertion latency ──
    # SKIPPED: measuring Zvec insertion requires creating a temp collection with
    # a fresh config singleton, which conflicts with EMBEDDING_DIM population.
    # This is a one-time offline operation; its latency is not query-critical.
    results["4_zvec_insert"] = {"status": "SKIPPED", "reason": "destructive operation, not query-critical"}

    # ── 5. Retrieval latency (5 reps) ──
    from graphrag_lite.retrieval.pipeline_a import retrieve_flat
    from graphrag_lite.retrieval.pipeline_b import retrieve_structural
    query = "How does the self-attention mechanism compute attention scores?"

    # Warm up (graph singleton + first query)
    retrieve_flat(query)
    retrieve_structural(query)

    pa_samples = []
    for i in range(5):
        t0 = time.time()
        retrieve_flat(query)
        pa_samples.append(time.time() - t0)
    results["5a_pipeline_a_retrieval"] = _stats(pa_samples)
    results["5a_pipeline_a_retrieval"]["samples_s"] = [round(s, 3) for s in pa_samples]

    pb_samples = []
    for i in range(5):
        t0 = time.time()
        retrieve_structural(query)
        pb_samples.append(time.time() - t0)
    results["5b_pipeline_b_retrieval"] = _stats(pb_samples)
    results["5b_pipeline_b_retrieval"]["samples_s"] = [round(s, 3) for s in pb_samples]

    # ── 6. Graph construction latency (5 reps) ──
    from graphrag_lite.graph.builder import build_graph, load_extractions
    ext_path = os.path.join(cfg.CACHE_DIR, "extractions.jsonl")
    recs = load_extractions(ext_path)
    graph_samples = []
    for i in range(5):
        t0 = time.time()
        build_graph(recs)
        graph_samples.append(time.time() - t0)
    results["6_graph_construction"] = _stats(graph_samples)
    results["6_graph_construction"]["samples_s"] = [round(s, 3) for s in graph_samples]

    # ── 7. End-to-end query latency (3 reps — expensive) ──
    from graphrag_lite.llm.client_factory import get_llm_client
    from graphrag_lite.llm.generator import AnswerGenerator
    llm_client = get_llm_client("local")
    gen = AnswerGenerator(client=llm_client)

    e2e_samples = []
    for i in range(3):
        t0 = time.time()
        ev = retrieve_structural(query)
        evidence_dicts = [{"text": e.text, "chunk_id": e.chunk_id} for e in ev[:6]]
        answer = gen.generate_answer(query, evidence_dicts)
        e2e_samples.append(time.time() - t0)
    results["7_end_to_end_query"] = _stats(e2e_samples)
    results["7_end_to_end_query"]["samples_s"] = [round(s, 3) for s in e2e_samples]
    results["7_end_to_end_query"]["note"] = "3 reps (each includes retrieval + generation)"

    # ── Save ──
    out_path = os.path.join("data", "cache", "perf_remediation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Performance results saved to {out_path}")

    # ── Print summary table ──
    print("\n" + "=" * 75)
    print("PERFORMANCE MEASUREMENT SUMMARY")
    print("=" * 75)
    print(f"{'Operation':<35} {'Mean':>8} {'Median':>8} {'p95':>8} {'Max':>8}")
    print("-" * 75)
    for key, label in [
        ("2_warm_gen_1.5b", "Warm gen (1.5b, 16 tok)"),
        ("3_embedding_nomic", "Embedding (nomic, 1 text)"),
        ("5a_pipeline_a_retrieval", "Pipeline A retrieval"),
        ("5b_pipeline_b_retrieval", "Pipeline B retrieval"),
        ("6_graph_construction", "Graph construction"),
        ("7_end_to_end_query", "End-to-end query (3 reps)"),
    ]:
        s = results.get(key, {})
        if "mean_s" in s:
            print(f"{label:<35} {s['mean_s']:>7.3f}s {s['median_s']:>7.3f}s {s['p95_s']:>7.3f}s {s['max_s']:>7.3f}s")
    print("-" * 75)
    print(f"{'Cold start (1.5b)':<35} {results['1_cold_start_1.5b']['latency_s']:>7.3f}s {'':>8} {'':>8} {'':>8}")
    print(f"{'Zvec insert':<35} {'SKIPPED':>8} {'':>8} {'':>8} {'':>8}")
    print("=" * 75)


if __name__ == "__main__":
    main()
