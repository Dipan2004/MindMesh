"""
eval/hotpotqa/record_environment.py -- write experiment_environment.json for the expanded experiments.

    python eval/hotpotqa/record_environment.py [--out data/benchmarks/hotpotqa/expanded_20261119]

Records git state, model digests, hardware, every retrieval / generation setting
the four pipelines use, and SHA-256 hashes of all pipeline and benchmark code
(so any later code change is detectable).  Read-only apart from the output file.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/benchmarks/hotpotqa/expanded_20261119")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from mindmesh.benchmarks.hotpotqa.env import index_env
    from mindmesh.benchmarks.hotpotqa.generation import ANSWER_SCHEMA, PROMPT_VERSION
    from mindmesh.benchmarks.hotpotqa.runtime import hardware_info, installed_models, ollama_ps
    from graphrag_lite.retrieval.lightrag import retrievers as R

    os.environ.update(index_env(args.out / "_probe_index", 10))   # benchmark config view, nothing written
    from graphrag_lite.config import get_config
    get_config.cache_clear()
    cfg = get_config()

    files = sorted(set(glob.glob(str(ROOT / "src/graphrag_lite/**/*.py"), recursive=True)
                       + glob.glob(str(ROOT / "src/mindmesh/**/*.py"), recursive=True)
                       + glob.glob(str(ROOT / "eval/hotpotqa/*.py")) + [str(ROOT / "eval/run_model_experiment.py")]))
    hashes = {Path(f).relative_to(ROOT).as_posix(): hashlib.sha256(Path(f).read_bytes()).hexdigest()
              for f in files if "__pycache__" not in f}

    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    hw = hardware_info()
    hw["cpu_name"] = cpu
    models = installed_models()

    env = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repository": {"path": str(ROOT), "branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
                       "commit_subject": git("log", "-1", "--format=%s"),
                       "uncommitted_changes": git("status", "--porcelain").splitlines()},
        "hardware": hw,
        "ollama_ps_at_record": ollama_ps(),
        "models": {
            "generator": {"tag": "mistral:latest", **models.get("mistral:latest", {})},
            "extractor": {"tag": "mistral:latest", **models.get("mistral:latest", {})},
            "embedding": {"tag": cfg.EMBEDDING_MODEL + (":latest" if ":" not in cfg.EMBEDDING_MODEL else ""),
                          **models.get("nomic-embed-text:latest", {}), "dimensions": 768},
        },
        "generation": {"prompt_version": PROMPT_VERSION, "format": "JSON schema", "schema": ANSWER_SCHEMA,
                       "temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256,
                       "context_paragraphs": 5, "same_for_all_pipelines": True},
        "extraction": {"temperature": cfg.EXTRACTION_TEMPERATURE, "seed": cfg.EXTRACTION_SEED,
                       "num_predict": cfg.LLM_NUM_PREDICT, "num_ctx": cfg.LLM_NUM_CTX,
                       "frequency_penalty": cfg.LLM_FREQUENCY_PENALTY,
                       "retry_frequency_penalties": cfg.LLM_RETRY_FREQUENCY_PENALTIES, "gleanings": 0},
        "retrieval": {
            "evaluation_top_k": 10,
            "chunking": "one chunk per Wikipedia paragraph",
            "zvec": {"index": "HNSW (text_vec, 768-d FP32) + BM25 FTS, fused with RRF", "hnsw_ef": "max(4*top_k, 64)"},
            "pipeline_a (MindMesh Dense)": {"query": "one hybrid HNSW+BM25 query on chunks", "top_k": cfg.ZVEC_TOP_N_FLAT},
            "pipeline_b (MindMesh Graph)": {"entry_entities": cfg.ZVEC_TOP_K_ENTRY, "bfs_hops": cfg.BFS_HOP_DEPTH,
                                            "ranking": cfg.RANKING_METHOD, "pagerank_damping": cfg.PAGERANK_DAMPING,
                                            "candidates_after_ranking": cfg.TOP_N_CANDIDATES_AFTER_RANKING,
                                            "sim_weight": cfg.PIPELINE_B_SIM_WEIGHT, "pr_weight": cfg.PIPELINE_B_PR_WEIGHT,
                                            "drop_type_labels": cfg.PIPELINE_B_DROP_TYPE_LABELS,
                                            "scope": cfg.PIPELINE_B_SCOPE, "final_top_k": cfg.TOP_N_FINAL_EVIDENCE},
            "lightrag_mix (MindMesh Mix)": {"merge": "round-robin vector -> local(entity) -> global(relationship), dedup",
                                            "dense_pool": "max(top_k, ZVEC_TOP_N_FLAT)",
                                            "graph_list_pool": "max(3*top_k, 20)",
                                            "edge_weight": R._EDGE_WEIGHT, "endpoint_weight": R._ENDPOINT_WEIGHT},
            "lightrag_mix_opt (MindMesh Mix Optimized)": {"merge": "dense-anchored: dense top (k-g), graph fills g slots",
                                                          "graph_slots_at_k10": R._graph_slots(10),
                                                          "graph_rank": "RRF(local_opt, global_opt), k=60",
                                                          "opt_primary": R._OPT_PRIMARY, "opt_edge": R._OPT_EDGE,
                                                          "opt_endpoint": R._OPT_ENDPOINT,
                                                          "degree_normalisation": "1/log2(2+degree)"},
        },
        "code_sha256": hashes,
    }
    out = args.out / "experiment_environment.json"
    out.write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}  ({len(hashes)} code files hashed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
