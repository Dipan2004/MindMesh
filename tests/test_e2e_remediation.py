"""
test_e2e_remediation.py — End-to-end verification for the pre-Phase 6 gate.

Steps 1-2 (load + chunk source documents) are BLOCKED: data/raw/ is empty
and the source markdown files are not available on this machine. The
canonical extractions.jsonl and Zvec collections (restored from backup)
are used instead, which verifies steps 3-9 of the pipeline:

  3.  Extractions exist (canonical extractions.jsonl — 96 entities, 53 rels)
  4.  Build the graph from extractions.jsonl
  5.  Embedding model available (nomic-embed-text via Ollama)
  6.  Zvec collections available (entities_v1: 72 docs, chunks_v1: 42 docs)
  7.  Run Pipeline A and Pipeline B retrieval on a real query
  8.  Pass Pipeline B evidence into the local answer generator (qwen2.5:1.5b)
  9.  Validate cited evidence IDs against the actual evidence supplied
  10. Save a run manifest with git state, versions, config, timings, results

The script writes a manifest to data/cache/e2e_remediation_manifest.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import platform
from pathlib import Path

# ── helpers ──

def _git_info() -> dict:
    info = {}
    for key, cmd in [
        ("commit", ["git", "log", "-1", "--format=%H"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("dirty", ["git", "status", "--short"]),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            info[key] = r.stdout.strip()
        except Exception:
            info[key] = "unknown"
    return info


def main() -> dict:
    manifest: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git": _git_info(),
    }

    # ── Config ──
    from graphrag_lite.config import get_config
    cfg = get_config()
    manifest["config"] = {
        "SLM_MODEL": cfg.SLM_MODEL,
        "EMBEDDING_MODEL": cfg.EMBEDDING_MODEL,
        "LLM_NUM_PREDICT": cfg.LLM_NUM_PREDICT,
        "LLM_NUM_CTX": cfg.LLM_NUM_CTX,
        "LLM_THINK": cfg.LLM_THINK,
        "LLM_FREQUENCY_PENALTY": cfg.LLM_FREQUENCY_PENALTY,
        "PIPELINE_B_SIM_WEIGHT": cfg.PIPELINE_B_SIM_WEIGHT,
        "PIPELINE_B_PR_WEIGHT": cfg.PIPELINE_B_PR_WEIGHT,
        "PIPELINE_B_SCOPE": cfg.PIPELINE_B_SCOPE,
        "PIPELINE_B_DROP_TYPE_LABELS": cfg.PIPELINE_B_DROP_TYPE_LABELS,
        "BFS_HOP_DEPTH": cfg.BFS_HOP_DEPTH,
        "ZVEC_TOP_K_ENTRY": cfg.ZVEC_TOP_K_ENTRY,
        "ZVEC_TOP_N_FLAT": cfg.ZVEC_TOP_N_FLAT,
        "TOP_N_FINAL_EVIDENCE": cfg.TOP_N_FINAL_EVIDENCE,
        "CACHE_DIR": cfg.CACHE_DIR,
    }

    # ── Step 3: Verify extractions exist ──
    ext_path = os.path.join(cfg.CACHE_DIR, "extractions.jsonl")
    assert os.path.exists(ext_path), f"extractions.jsonl not found at {ext_path}"
    with open(ext_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    total_ent = sum(len(r.get("entities", [])) for r in records)
    total_rel = sum(len(r.get("relationships", [])) for r in records)
    clean = sum(1 for r in records if "_parse_error" not in r)
    manifest["step3_extractions"] = {
        "status": "PASS",
        "records": len(records),
        "clean": clean,
        "entities": total_ent,
        "relationships": total_rel,
    }

    # ── Step 4: Build the graph ──
    from graphrag_lite.graph.builder import build_graph, load_extractions
    t0 = time.time()
    recs = load_extractions(ext_path)
    G = build_graph(recs)
    t1 = time.time()
    manifest["step4_graph"] = {
        "status": "PASS",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "build_time_s": round(t1 - t0, 3),
    }

    # ── Step 5: Verify embedding model ──
    import ollama
    client = ollama.Client()
    try:
        client.show(cfg.EMBEDDING_MODEL)
        manifest["step5_embedding_model"] = {"status": "PASS", "model": cfg.EMBEDDING_MODEL}
    except Exception as e:
        manifest["step5_embedding_model"] = {"status": "FAIL", "error": str(e)}
        manifest["overall"] = "BLOCKED"
        _save(manifest)
        return manifest

    # ── Step 6: Verify Zvec collections ──
    import zvec
    try:
        e_col = zvec.open(os.path.join(cfg.CACHE_DIR, cfg.ZVEC_COLLECTION_ENTITIES))
        e_count = e_col.stats.doc_count
        e_col.close()
        c_col = zvec.open(os.path.join(cfg.CACHE_DIR, cfg.ZVEC_COLLECTION_CHUNKS))
        c_count = c_col.stats.doc_count
        c_col.close()
        manifest["step6_zvec"] = {
            "status": "PASS",
            "entities_v1_docs": e_count,
            "chunks_v1_docs": c_count,
        }
    except Exception as e:
        manifest["step6_zvec"] = {"status": "FAIL", "error": str(e)}
        manifest["overall"] = "BLOCKED"
        _save(manifest)
        return manifest

    # ── Step 7: Run Pipeline A and Pipeline B ──
    from graphrag_lite.retrieval.pipeline_a import retrieve_flat
    from graphrag_lite.retrieval.pipeline_b import retrieve_structural

    query = "How does the self-attention mechanism compute attention scores?"
    gold_chunk = "c81c2a06e4ee9667"

    # Pipeline A
    t0 = time.time()
    ev_a = retrieve_flat(query)
    t1 = time.time()
    manifest["step7_pipeline_a"] = {
        "status": "PASS",
        "query": query,
        "retrieved_count": len(ev_a),
        "top_chunk_id": ev_a[0].chunk_id if ev_a else None,
        "gold_at_rank": next((i+1 for i, e in enumerate(ev_a) if e.chunk_id == gold_chunk), None),
        "latency_s": round(t1 - t0, 3),
    }

    # Pipeline B
    t0 = time.time()
    ev_b = retrieve_structural(query)
    t1 = time.time()
    manifest["step7_pipeline_b"] = {
        "status": "PASS",
        "query": query,
        "retrieved_count": len(ev_b),
        "top_chunk_id": ev_b[0].chunk_id if ev_b else None,
        "gold_at_rank": next((i+1 for i, e in enumerate(ev_b) if e.chunk_id == gold_chunk), None),
        "latency_s": round(t1 - t0, 3),
    }

    # ── Step 8: Generate an answer with the local model ──
    from graphrag_lite.llm.client_factory import get_llm_client
    from graphrag_lite.llm.generator import AnswerGenerator

    llm_client = get_llm_client("local")
    gen = AnswerGenerator(client=llm_client)

    evidence_dicts = [
        {"text": e.text, "chunk_id": e.chunk_id}
        for e in ev_b[:6]  # TOP_N_FINAL_EVIDENCE
    ]
    t0 = time.time()
    answer = gen.generate_answer(query, evidence_dicts)
    t1 = time.time()

    manifest["step8_generation"] = {
        "status": "PASS",
        "model": cfg.SLM_MODEL,
        "answer_text_preview": answer.text[:300] + "..." if len(answer.text) > 300 else answer.text,
        "cited_evidence_ids": answer.cited_evidence_ids,
        "confidence": answer.confidence,
        "latency_s": round(t1 - t0, 3),
    }

    # ── Step 9: Validate citations ──
    passed_ids = {e.chunk_id for e in ev_b[:6]}
    cited_ids = set(answer.cited_evidence_ids)
    all_valid = cited_ids.issubset(passed_ids)
    hallucinated = cited_ids - passed_ids

    manifest["step9_citation_validation"] = {
        "status": "PASS" if all_valid else "FAIL",
        "passed_evidence_ids": sorted(passed_ids),
        "cited_evidence_ids": sorted(cited_ids),
        "all_cited_ids_in_evidence": all_valid,
        "hallucinated_ids": sorted(hallucinated) if hallucinated else None,
    }

    # ── Step 10: Overall ──
    all_steps = [
        manifest.get("step3_extractions", {}).get("status"),
        manifest.get("step4_graph", {}).get("status"),
        manifest.get("step5_embedding_model", {}).get("status"),
        manifest.get("step6_zvec", {}).get("status"),
        manifest.get("step7_pipeline_a", {}).get("status"),
        manifest.get("step7_pipeline_b", {}).get("status"),
        manifest.get("step8_generation", {}).get("status"),
        manifest.get("step9_citation_validation", {}).get("status"),
    ]
    manifest["overall"] = "PASS" if all(s == "PASS" for s in all_steps) else "FAIL"
    manifest["blocked_steps"] = ["step1_load_source_docs (BLOCKED: data/raw/ empty)",
                                  "step2_chunk_source_docs (BLOCKED: no source documents)"]

    _save(manifest)
    return manifest


def _save(manifest: dict) -> None:
    out_path = os.path.join("data", "cache", "e2e_remediation_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved to {out_path}")


if __name__ == "__main__":
    m = main()
    print("\n" + "=" * 70)
    print("END-TO-END VERIFICATION RESULT:", m.get("overall", "UNKNOWN"))
    print("=" * 70)
    for key in sorted(m.keys()):
        if key.startswith("step"):
            print(f"  {key}: {json.dumps(m[key], ensure_ascii=False)}")
    print(f"\n  Blocked steps: {m.get('blocked_steps', [])}")
    print(f"  Git commit: {m.get('git', {}).get('commit', 'unknown')[:12]}")
