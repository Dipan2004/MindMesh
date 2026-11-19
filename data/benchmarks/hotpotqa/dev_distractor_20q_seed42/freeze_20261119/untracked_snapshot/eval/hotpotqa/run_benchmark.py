"""
eval/hotpotqa/run_benchmark.py -- run the four MindMesh pipelines on the HotpotQA sample.

    python eval/hotpotqa/run_benchmark.py --model mistral:latest
    python eval/hotpotqa/run_benchmark.py --model mistral:latest --limit 2          # smoke test
    python eval/hotpotqa/run_benchmark.py --model qwen2.5:7b --index-model mistral:latest

Pipelines (identical questions, corpus, index, embedding model, generator
model and generation settings):
    pipeline_a        MindMesh Dense
    pipeline_b        MindMesh Graph
    lightrag_mix      MindMesh Mix
    lightrag_mix_opt  MindMesh Mix Optimized

Resumable: results.jsonl is append-only; a (pipeline, question) pair already
recorded is never re-run or overwritten.  Re-running the same command
continues an interrupted run.  A different configuration gets its own run
directory; an existing run directory with a different configuration is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mindmesh.benchmarks.hotpotqa.dataset import load_benchmark  # noqa: E402
from mindmesh.benchmarks.hotpotqa.env import activate_index  # noqa: E402
from mindmesh.benchmarks.hotpotqa.generation import PROMPT_VERSION, generate  # noqa: E402
from mindmesh.benchmarks.hotpotqa.metrics import (  # noqa: E402
    answer_metrics, joint_metrics, retrieval_metrics, sp_metrics,
)
from mindmesh.benchmarks.hotpotqa.runtime import (  # noqa: E402
    GpuLock, hardware_info, ollama_ps, resolve_local_model, unload_model,
)

DEFAULT_BENCH = ROOT / "data" / "benchmarks" / "hotpotqa" / "dev_distractor_20q_seed42"
PIPELINES = {
    "pipeline_a": "MindMesh Dense",
    "pipeline_b": "MindMesh Graph",
    "lightrag_mix": "MindMesh Mix",
    "lightrag_mix_opt": "MindMesh Mix Optimized",
}
METRIC_KEYS = [
    "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "all_gold_in_top10",
    "ans_em", "ans_f1", "sp_prec", "sp_recall", "sp_f1", "sp_em", "joint_em", "joint_f1",
    "retrieval_latency_s", "generation_latency_s", "total_latency_s",
]


def slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def run_question(adapter, pname, q, paras_by_chunk, model, top_k, context_k, gen_options) -> dict:
    t0 = time.monotonic()
    rec = {"pipeline": pname, "pipeline_label": PIPELINES[pname], "qid": q["_id"],
           "type": q["type"], "question": q["question"], "gold_answer": q["answer"],
           "gold_chunk_ids": q["gold_chunk_ids"], "gold_sp": q["supporting_facts"], "errors": []}
    try:
        res = adapter.query(q["question"], generate_answer=False)
        retrieved = res.chunk_ids[:top_k]
        rec["retrieval_latency_s"] = round(res.retrieval_latency_s, 4)
        rec["errors"].extend(res.errors)
    except Exception as exc:  # noqa: BLE001
        retrieved = []
        rec["retrieval_latency_s"] = round(time.monotonic() - t0, 4)
        rec["errors"].append(f"retrieval: {type(exc).__name__}: {exc}")
    rec["retrieved_chunk_ids"] = retrieved
    rec["retrieved_titles"] = [paras_by_chunk[c].title if c in paras_by_chunk else None for c in retrieved]
    rec.update(retrieval_metrics(retrieved, q["gold_chunk_ids"]))

    ctx = [{"title": paras_by_chunk[c].title, "sentences": paras_by_chunk[c].sentences}
           for c in retrieved[:context_k] if c in paras_by_chunk]
    try:
        g = generate(model, q["question"], ctx, gen_options)
    except Exception as exc:  # noqa: BLE001
        g = {"answer": "", "sp_pred": [], "raw": "", "parse_ok": False, "latency_s": 0.0}
        rec["errors"].append(f"generation: {type(exc).__name__}: {exc}")
    rec["pred_answer"] = g["answer"]
    rec["pred_sp"] = g["sp_pred"]
    rec["generation_raw"] = g["raw"]
    rec["generation_parse_ok"] = g["parse_ok"]
    rec["generation_latency_s"] = round(g["latency_s"], 4)
    for k in ("prompt_tokens", "output_tokens", "tokens_per_s", "load_s"):
        rec[k] = g.get(k)
    a = answer_metrics(g["answer"], q["answer"])
    s = sp_metrics(g["sp_pred"], q["supporting_facts"])
    j = joint_metrics(a, s)
    rec.update({f"ans_{k}": v for k, v in a.items()})
    rec.update({f"sp_{k}": v for k, v in s.items()})
    rec.update({f"joint_{k}": v for k, v in j.items()})
    rec["total_latency_s"] = round(rec["retrieval_latency_s"] + rec["generation_latency_s"], 4)
    rec["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def summarize(rows: list[dict]) -> dict:
    out = {}
    for pname in PIPELINES:
        prow = [r for r in rows if r["pipeline"] == pname]
        if not prow:
            continue
        agg = {"label": PIPELINES[pname], "n": len(prow),
               "n_errors": sum(1 for r in prow if r["errors"]),
               "n_generation_parse_fail": sum(1 for r in prow if not r.get("generation_parse_ok"))}
        for k in METRIC_KEYS:
            vals = [r[k] for r in prow if r.get(k) is not None]
            agg[k] = round(statistics.fmean(vals), 4) if vals else None
        for t in ("bridge", "comparison"):
            sub = [r for r in prow if r["type"] == t]
            if sub:
                agg[f"{t}_n"] = len(sub)
                for k in ("recall@10", "mrr", "ans_em", "ans_f1", "sp_f1", "joint_f1"):
                    agg[f"{t}_{k}"] = round(statistics.fmean(r[k] for r in sub), 4)
        out[pname] = agg
    return out


def summary_markdown(summary: dict, cfg: dict) -> str:
    cols = [("R@1", "recall@1"), ("R@3", "recall@3"), ("R@5", "recall@5"), ("R@10", "recall@10"),
            ("MRR", "mrr"), ("EM", "ans_em"), ("F1", "ans_f1"), ("SP-P", "sp_prec"),
            ("SP-R", "sp_recall"), ("SP-F1", "sp_f1"), ("J-EM", "joint_em"), ("J-F1", "joint_f1"),
            ("Ret s", "retrieval_latency_s"), ("Gen s", "generation_latency_s"), ("Tot s", "total_latency_s")]
    lines = [
        f"# HotpotQA run `{cfg['run_id']}`", "",
        f"- Benchmark: {cfg['benchmark']['corpus_scope']}",
        f"- Questions: {cfg['n_questions']} ({cfg['type_counts']})",
        f"- Generator: `{cfg['generation_model']}`; index/extraction: `{cfg['index_model']}`; "
        f"embedding: `{cfg['embedding_model']}`",
        f"- Retrieval depth {cfg['top_k']}, generation context top-{cfg['context_k']}, "
        f"options {json.dumps(cfg['generation_options'])}",
        f"- Hardware: {cfg['hardware'].get('gpus')} | CPU {cfg['hardware'].get('cpu')}", "",
        "| Pipeline | n | " + " | ".join(c for c, _ in cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for p, a in summary.items():
        vals = ["-" if a.get(k) is None else f"{a[k]:.3f}" for _, k in cols]
        lines.append(f"| {a['label']} (`{p}`) | {a['n']} | " + " | ".join(vals) + " |")
    lines += ["", "Metrics: answer EM/F1, SP and joint per the official HotpotQA script; "
              "Recall@k/MRR over the supporting-fact paragraphs (MindMesh addition). "
              "Retrieval recall and citation validity do not establish semantic correctness."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="exact local Ollama tag for answer generation")
    ap.add_argument("--index-model", default=None, help="extraction model of the index to use (default: --model)")
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES), choices=list(PIPELINES))
    ap.add_argument("--top-k", type=int, default=10, help="retrieval depth (Recall@10 needs >=10)")
    ap.add_argument("--context-k", type=int, default=5, help="retrieved paragraphs passed to the generator")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    ap.add_argument("--run-id", default=None, help="explicit run directory name")
    args = ap.parse_args(argv)

    args.bench_dir = args.bench_dir.resolve()
    model, model_info = resolve_local_model(args.model)
    index_model, _ = resolve_local_model(args.index_model or args.model)
    index_dir = args.bench_dir / "index" / slug(index_model)
    if not (index_dir / "index_meta.json").exists():
        print(f"[REFUSED] No index at {index_dir}. Run eval/hotpotqa/build_index.py --model {index_model}", file=sys.stderr)
        return 1
    index_meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    if index_meta.get("smoke_test_limit") is not None:
        print(f"[WARN] index at {index_dir} is a smoke-test index ({index_meta['smoke_test_limit']} chunks)")

    activate_index(index_dir, top_k=args.top_k)
    from graphrag_lite.config import get_config
    cfg = get_config()

    bench = load_benchmark(args.bench_dir)
    questions = bench["questions"][: args.limit] if args.limit else bench["questions"]
    paras_by_chunk = {p.chunk_id: p for p in bench["paragraphs"]}
    gen_options = {"temperature": 0.0, "seed": 42, "num_ctx": args.num_ctx, "num_predict": args.num_predict}

    run_cfg_core = {
        "generation_model": model, "generation_model_digest": model_info.get("digest"),
        "index_model": index_model, "embedding_model": cfg.EMBEDDING_MODEL,
        "pipelines": args.pipelines, "top_k": args.top_k, "context_k": args.context_k,
        "generation_options": gen_options, "prompt_version": PROMPT_VERSION,
        "question_ids": [q["_id"] for q in questions],
    }
    cfg_hash = hashlib.sha1(json.dumps(run_cfg_core, sort_keys=True).encode()).hexdigest()[:8]
    run_id = args.run_id or f"gen-{slug(model)}__idx-{slug(index_model)}__{cfg_hash}"
    run_dir = args.bench_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
        if prev.get("config_hash") != cfg_hash:
            print(f"[REFUSED] {run_dir} holds a run with a different configuration.", file=sys.stderr)
            return 1

    type_counts = {}
    for q in questions:
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1
    run_cfg = dict(run_cfg_core, run_id=run_id, config_hash=cfg_hash, n_questions=len(questions),
                   type_counts=type_counts, benchmark=bench["config"], corpus_stats=bench["stats"],
                   index_dir=str(index_dir.relative_to(ROOT)),
                   index_stats={k: index_meta.get(k) for k in (
                       "chunks_indexed", "entities_indexed", "relationships_indexed", "documents",
                       "chunks_parse_failed", "graph", "timings", "gleanings", "extraction_model")},
                   pipeline_labels={p: PIPELINES[p] for p in args.pipelines},
                   hardware=hardware_info(), started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf-8")

    results_path = run_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["pipeline"], r["qid"]))
    todo = [(p, q) for p in args.pipelines for q in questions if (p, q["_id"]) not in done]
    print(f"[run] {run_dir}\n[run] {len(done)} done, {len(todo)} to go | gen={model} index={index_model}")

    from mindmesh.pipelines.registry import build_adapter
    adapters = {}
    ps_snapshot = None
    with GpuLock(f"hotpotqa run {run_id}"):
        with open(results_path, "a", encoding="utf-8") as fh:
            for i, (pname, q) in enumerate(todo, 1):
                if pname not in adapters:
                    adapters[pname] = build_adapter(pname, model=model, exp_dir=index_dir, top_k=args.top_k)
                rec = run_question(adapters[pname], pname, q, paras_by_chunk, model,
                                   args.top_k, args.context_k, gen_options)
                rec["generation_model"] = model
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if ps_snapshot is None:
                    ps_snapshot = ollama_ps()
                print(f"  [{i}/{len(todo)}] {PIPELINES[pname]:<24} {q['_id']} R@10={rec['recall@10']:.2f} "
                      f"EM={rec['ans_em']:.0f} F1={rec['ans_f1']:.2f} SP-F1={rec['sp_f1']:.2f} "
                      f"pred={rec['pred_answer'][:40]!r} gold={q['answer'][:30]!r} "
                      f"({rec['total_latency_s']:.1f}s){' ERR' if rec['errors'] else ''}", flush=True)
        unload_model(model)

    rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = summarize(rows)
    run_cfg_full = json.loads(cfg_path.read_text(encoding="utf-8"))
    if ps_snapshot:
        run_cfg_full.setdefault("ollama_ps_snapshots", []).append(
            {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": ps_snapshot})
    run_cfg_full["last_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg_path.write_text(json.dumps(run_cfg_full, indent=2, default=str), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_markdown(summary, run_cfg_full), encoding="utf-8")
    print(summary_markdown(summary, run_cfg_full))
    return 0


if __name__ == "__main__":
    sys.exit(main())
