"""
eval/hotpotqa/run_c5_ondemand.py -- Dense vs query-focused C5 with genuine on-demand extraction.

    python eval/hotpotqa/run_c5_ondemand.py --out <exp dir> [--limit N] [--run-id ID]

Inputs (read-only): <bench-dir>/questions.json, corpus.jsonl.
Requires <out>/dense_index (eval/hotpotqa/build_dense_index.py).

Phase 1 (retrieval, both pipelines, every question, questions.json order):
  pipeline_a    MindMesh Dense: the existing PipelineAAdapter on the Dense-only index
  c5_ondemand   C5 (prototype logic unchanged) with a qwen2.5:3b extraction cache;
                cache misses are extracted on demand and timed
Phase 2 (generation): the same generator, prompt and settings for both pipelines.

Both phases are append-only and resumable: rows already present are skipped.
GPU telemetry is sampled every 15 s for the whole run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mindmesh.benchmarks.hotpotqa.env import activate_index  # noqa: E402
from mindmesh.benchmarks.hotpotqa.generation import PROMPT_VERSION, generate  # noqa: E402
from mindmesh.benchmarks.hotpotqa.metrics import (  # noqa: E402
    answer_metrics, joint_metrics, retrieval_metrics, sp_metrics)
from mindmesh.benchmarks.hotpotqa.runtime import (  # noqa: E402
    GpuLock, hardware_info, ollama_ps, resolve_local_model, unload_model)

DEF_BENCH = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/dev_distractor_100q_seed20261119"
DEF_OUT = ROOT / "data/benchmarks/hotpotqa/c5_ondemand_qwen3b_20261119"
LABELS = {"pipeline_a": "MindMesh Dense", "c5_ondemand": "MindMesh C5 on-demand (qwen2.5:3b extraction)"}
GEN_OPTIONS = {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256}
TOP_K, CONTEXT_K = 10, 5


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def append(p: Path, row: dict) -> None:
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def telemetry(path: Path, stop: threading.Event) -> None:
    q = "timestamp,temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw,clocks_event_reasons.active"
    while not stop.is_set():
        try:
            line = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
            if line:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            pass
        stop.wait(15)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", type=Path, default=DEF_BENCH)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--generator", default="qwen2.5:3b")
    ap.add_argument("--extractor", default="qwen2.5:3b")
    ap.add_argument("--extractor-provider", choices=["ollama", "openrouter"], default="ollama",
                    help="openrouter: config/extraction_provider.json model, NO local fallback")
    ap.add_argument("--cache-root", type=Path, default=None, help="default <out>/extraction_cache")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    dense_dir = out / "dense_index"
    if not (dense_dir / "dense_index_meta.json").exists():
        print("[REFUSED] build the Dense-only index first (build_dense_index.py)", file=sys.stderr)
        return 1
    gen_model, gen_info = resolve_local_model(args.generator)
    if args.extractor_provider == "openrouter":
        from graphrag_lite.retrieval import provider_extraction as PE
        or_settings = PE.load_provider_settings().openrouter
        if not PE.api_key_available(or_settings):
            print(f"[REFUSED] {or_settings.api_key_env} not set", file=sys.stderr)
            return 1
        ext_model, ext_info = or_settings.model, {"digest": None, "provider": "openrouter"}
        LABELS["c5_ondemand"] = f"MindMesh C5 on-demand ({ext_model} extraction)"
    else:
        ext_model, ext_info = resolve_local_model(args.extractor)
    activate_index(dense_dir, top_k=TOP_K)

    from graphrag_lite.retrieval.extraction_cache import ExtractionCache
    from graphrag_lite.retrieval.query_focused_ondemand import OnDemandIndex, retrieve_c5_ondemand
    from mindmesh.pipelines.pipeline_a_adapter.adapter import PipelineAAdapter

    qs = json.loads((args.bench_dir / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[: args.limit]
    paras = {json.loads(l)["chunk_id"]: json.loads(l)
             for l in (args.bench_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    if args.extractor_provider == "openrouter":
        cache = PE.OpenRouterExtractionCache(args.cache_root or out / "extraction_cache", or_settings)
    else:
        cache = ExtractionCache(args.cache_root or out / "extraction_cache", ext_model, ext_info["digest"])
    idx = OnDemandIndex(paras, cache)
    dense = PipelineAAdapter(model=gen_model, exp_dir=dense_dir, top_k=TOP_K)

    core = {"generator": gen_model, "generator_digest": gen_info["digest"], "extractor": ext_model,
            "extractor_digest": ext_info["digest"], "cache_version": cache.version, "top_k": TOP_K,
            "context_k": CONTEXT_K, "generation_options": GEN_OPTIONS, "prompt_version": PROMPT_VERSION,
            "question_ids": [q["_id"] for q in qs], "pipelines": list(LABELS)}
    if args.extractor_provider != "ollama":          # key absent for Ollama runs: old config hashes unchanged
        core["extractor_provider"] = args.extractor_provider
    h = hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]
    run_dir = out / "runs" / (args.run_id or f"dense_vs_c5_{h}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({**core, "config_hash": h, "bench_dir": str(args.bench_dir),
                                        "dense_index": json.loads((dense_dir / "dense_index_meta.json").read_text()),
                                        "cache_dir": str(cache.dir), "cache_meta": json.loads((cache.dir / "cache_meta.json").read_text()),
                                        "hardware": hardware_info(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                       indent=2, default=str), encoding="utf-8")
    elif json.loads(cfg_path.read_text())["config_hash"] != h:
        print(f"[REFUSED] {run_dir} holds a different configuration", file=sys.stderr)
        return 1

    ret_path, res_path = run_dir / "retrieval.jsonl", run_dir / "results.jsonl"
    stop = threading.Event()
    th = threading.Thread(target=telemetry, args=(run_dir / "gpu_telemetry.csv", stop), daemon=True)
    th.start()
    try:
        with GpuLock(f"c5 on-demand {run_dir.name}"):
            # ── Phase 1: retrieval ────────────────────────────────────────────
            done = {(r["pipeline"], r["qid"]) for r in jsonl(ret_path)}
            for i, q in enumerate(qs, 1):
                for p in LABELS:
                    if (p, q["_id"]) in done:
                        continue
                    row = {"pipeline": p, "qid": q["_id"], "type": q["type"], "question": q["question"], "errors": []}
                    t0 = time.monotonic()
                    try:
                        if p == "pipeline_a":
                            res = dense.query(q["question"])
                            row["retrieved_chunk_ids"] = res.chunk_ids[:TOP_K]
                            row["retrieval_latency_s"] = round(res.retrieval_latency_s, 4)
                        else:
                            ids, info = retrieve_c5_ondemand(q["question"], q["_id"], idx, TOP_K)
                            row["retrieved_chunk_ids"] = ids[:TOP_K]
                            row["retrieval_latency_s"] = info["timing_s"]["retrieval_total"]
                            row["diagnostics"] = info
                    except Exception as exc:  # noqa: BLE001
                        row["retrieved_chunk_ids"] = []
                        row["retrieval_latency_s"] = round(time.monotonic() - t0, 4)
                        row["errors"].append(f"retrieval: {type(exc).__name__}: {exc}")
                    append(ret_path, row)
                    c = row.get("diagnostics", {}).get("cache", {})
                    print(f"[R {i}/{len(qs)}] {p:12s} {q['_id']} {row['retrieval_latency_s']:7.2f}s"
                          + (f" miss={c.get('misses')} hit={c.get('hits_ok')} ext={c.get('extraction_wall_s', 0):.1f}s"
                             if c else "") + (" ERR" if row["errors"] else ""), flush=True)
            # ── Phase 2: generation (same generator/settings for both pipelines) ──
            if gen_model != ext_model and args.extractor_provider == "ollama":
                unload_model(ext_model)
            rrows = {(r["pipeline"], r["qid"]): r for r in jsonl(ret_path)}
            gdone = {(r["pipeline"], r["qid"]) for r in jsonl(res_path)}
            for i, q in enumerate(qs, 1):
                for p in LABELS:
                    if (p, q["_id"]) in gdone:
                        continue
                    rr = rrows[(p, q["_id"])]
                    rec = {"pipeline": p, "pipeline_label": LABELS[p], "qid": q["_id"], "type": q["type"],
                           "question": q["question"], "gold_answer": q["answer"], "gold_chunk_ids": q["gold_chunk_ids"],
                           "gold_sp": q["supporting_facts"], "errors": list(rr["errors"]),
                           "retrieved_chunk_ids": rr["retrieved_chunk_ids"], "retrieval_latency_s": rr["retrieval_latency_s"]}
                    rec["retrieved_titles"] = [paras[c]["title"] if c in paras else None for c in rec["retrieved_chunk_ids"]]
                    rec.update(retrieval_metrics(rec["retrieved_chunk_ids"], q["gold_chunk_ids"]))
                    ctx = [{"title": paras[c]["title"], "sentences": paras[c]["sentences"]}
                           for c in rec["retrieved_chunk_ids"][:CONTEXT_K] if c in paras]
                    try:
                        g = generate(gen_model, q["question"], ctx, GEN_OPTIONS)
                    except Exception as exc:  # noqa: BLE001
                        g = {"answer": "", "sp_pred": [], "raw": "", "parse_ok": False, "latency_s": 0.0}
                        rec["errors"].append(f"generation: {type(exc).__name__}: {exc}")
                    rec.update(pred_answer=g["answer"], pred_sp=g["sp_pred"], generation_raw=g["raw"],
                               generation_parse_ok=g["parse_ok"], generation_latency_s=round(g["latency_s"], 4),
                               generation_model=gen_model)
                    for k in ("prompt_tokens", "output_tokens", "tokens_per_s", "load_s"):
                        rec[k] = g.get(k)
                    a = answer_metrics(g["answer"], q["answer"]); s = sp_metrics(g["sp_pred"], q["supporting_facts"])
                    j = joint_metrics(a, s)
                    rec.update({f"ans_{k}": v for k, v in a.items()}); rec.update({f"sp_{k}": v for k, v in s.items()})
                    rec.update({f"joint_{k}": v for k, v in j.items()})
                    rec["total_latency_s"] = round(rec["retrieval_latency_s"] + rec["generation_latency_s"], 4)
                    append(res_path, rec)
                    print(f"[G {i}/{len(qs)}] {p:12s} {q['_id']} EM={rec['ans_em']:.0f} F1={rec['ans_f1']:.2f} "
                          f"SP-F1={rec['sp_f1']:.2f} R@10={rec['recall@10']:.2f}{' ERR' if rec['errors'] else ''}", flush=True)
            ps = ollama_ps()
            unload_model(gen_model)
    finally:
        stop.set()
    cfg = json.loads(cfg_path.read_text())
    cfg["ollama_ps_end"] = ps
    cfg["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    rows = jsonl(res_path)
    summ = {}
    for p in LABELS:
        pr = [r for r in rows if r["pipeline"] == p]
        if pr:
            summ[p] = {"n": len(pr), **{m: round(statistics.fmean(r[m] for r in pr), 4) for m in
                       ("recall@5", "recall@10", "mrr", "all_gold_in_top10", "ans_em", "ans_f1", "sp_prec",
                        "sp_recall", "sp_f1", "joint_em", "joint_f1", "retrieval_latency_s", "generation_latency_s",
                        "total_latency_s")}}
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"[done] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
