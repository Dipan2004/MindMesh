"""
analyze_main5.py -- verification + analysis for the main five-pipeline Gemini-extraction benchmark (100 questions).

Inputs (read-only):
  four full-graph pipelines: <bench>/runs/main5_gemini_4pipe_100q/results.jsonl
  C5 (own on-demand Gemini cache) + its Dense consistency row: main5_gemini_20261120/c5/runs/main5_gemini_c5_100q/
  offline build report + extraction failure / gold check
Output: main5_gemini_20261120/analysis.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "hotpotqa"))
from analyze_expanded import paired_bootstrap  # noqa: E402

B = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/dev_distractor_100q_seed20261119"
IDX = B / "index/openrouter__google_gemini-2.5-flash-lite"
M = ROOT / "data/benchmarks/hotpotqa/main5_gemini_20261120"
R4 = B / "runs/main5_gemini_4pipe_100q"
RC = M / "c5/runs/main5_gemini_c5_100q"
METRICS = ["recall@5", "recall@10", "mrr", "all_gold_in_top10", "ans_em", "ans_f1", "sp_prec", "sp_recall",
           "sp_f1", "joint_em", "joint_f1"]
ORDER = ["pipeline_a", "pipeline_b", "lightrag_mix", "lightrag_mix_opt", "c5_ondemand"]
LABEL = {"pipeline_a": "MindMesh Dense", "pipeline_b": "MindMesh Graph", "lightrag_mix": "MindMesh Mix",
         "lightrag_mix_opt": "MindMesh Mix Optimized", "c5_ondemand": "query_focused_c5 (on-demand)"}


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    qids = [q["_id"] for q in json.loads((B / "questions.json").read_text(encoding="utf-8"))]
    r4 = jl(R4 / "results.jsonl")
    rc = jl(RC / "results.jsonl")
    ret_c = {(r["pipeline"], r["qid"]): r for r in jl(RC / "retrieval.jsonl")}
    rows = {(r["pipeline"], r["qid"]): r for r in r4}
    rows.update({(r["pipeline"], r["qid"]): r for r in rc if r["pipeline"] == "c5_ondemand"})
    dense_c5drv = {r["qid"]: r for r in rc if r["pipeline"] == "pipeline_a"}

    # ── verification ──
    ver = {"expected_question_ids": len(qids)}
    for p in ORDER:
        src = r4 if p != "c5_ondemand" else [r for r in rc if r["pipeline"] == p]
        ids = [r["qid"] for r in src if r["pipeline"] == p]
        ver[p] = {"rows": len(ids), "unique": len(set(ids)), "missing": sorted(set(qids) - set(ids)),
                  "extra": sorted(set(ids) - set(qids)),
                  "rows_with_errors": sum(bool(r["errors"]) for r in src if r["pipeline"] == p),
                  "generation_parse_failures": sum(not r["generation_parse_ok"] for r in src if r["pipeline"] == p),
                  "generation_model": sorted({r["generation_model"] for r in src if r["pipeline"] == p})}
    ver["all_complete"] = all(ver[p]["rows"] == 100 and ver[p]["unique"] == 100 and not ver[p]["missing"]
                              and not ver[p]["extra"] for p in ORDER)
    ver["same_generator_everywhere"] = len({m for p in ORDER for m in ver[p]["generation_model"]}) == 1
    cfg4 = json.loads((R4 / "run_config.json").read_text(encoding="utf-8"))
    cfgc = json.loads((RC / "run_config.json").read_text(encoding="utf-8"))
    ver["generation_settings"] = {"four_pipelines": {k: cfg4.get(k) for k in ("generation_options", "prompt_version", "top_k", "context_k")},
                                  "c5_driver": {k: cfgc.get(k) for k in ("generation_options", "prompt_version", "top_k", "context_k")}}
    # Dense in the C5 driver (Dense-only index) vs Dense in the full index: same retrieval expected
    ver["dense_consistency_c5_driver_vs_full_index"] = {
        "same_top10": sum(dense_c5drv[q]["retrieved_chunk_ids"] == rows[("pipeline_a", q)]["retrieved_chunk_ids"] for q in qids if q in dense_c5drv),
        "same_answer": sum(dense_c5drv[q]["pred_answer"] == rows[("pipeline_a", q)]["pred_answer"] for q in qids if q in dense_c5drv),
        "n": sum(q in dense_c5drv for q in qids)}

    # ── metrics ──
    summary = {}
    for p in ORDER:
        pr = [rows[(p, q)] for q in qids if (p, q) in rows]
        s = {"label": LABEL[p], "n": len(pr), **{m: round(st.fmean(r[m] for r in pr), 4) for m in METRICS}}
        for t in ("bridge", "comparison"):
            tr = [r for r in pr if r["type"] == t]
            s[f"{t}_n"] = len(tr)
            s[f"{t}_recall@5"] = round(st.fmean(r["recall@5"] for r in tr), 4)
            s[f"{t}_ans_f1"] = round(st.fmean(r["ans_f1"] for r in tr), 4)
        gen = [r["generation_latency_s"] for r in pr]
        if p == "c5_ondemand":
            T = [ret_c[(p, q)]["diagnostics"]["timing_s"] for q in qids]
            online = [t["retrieval_total"] - t["extraction"] for t in T]
            ext = [t["extraction"] for t in T]
            s["latency_s"] = {"online_retrieval_excl_extraction_mean": round(st.fmean(online), 3),
                              "online_retrieval_excl_extraction_median": round(st.median(online), 3),
                              "on_demand_extraction_mean_per_question": round(st.fmean(ext), 3),
                              "on_demand_extraction_total": round(sum(ext), 1),
                              "retrieval_incl_extraction_mean": round(st.fmean(t["retrieval_total"] for t in T), 3),
                              "generation_mean": round(st.fmean(gen), 3), "generation_median": round(st.median(gen), 3)}
        else:
            rl = [r["retrieval_latency_s"] for r in pr]
            s["latency_s"] = {"online_retrieval_mean": round(st.fmean(rl), 3), "online_retrieval_median": round(st.median(rl), 3),
                              "online_retrieval_p90": round(pct(rl, 0.9), 3),
                              "generation_mean": round(st.fmean(gen), 3), "generation_median": round(st.median(gen), 3)}
        summary[p] = s

    vs_dense = {p: {m: paired_bootstrap([rows[("pipeline_a", q)][m] for q in qids], [rows[(p, q)][m] for q in qids])
                    for m in ("recall@5", "recall@10", "mrr", "ans_em", "ans_f1", "sp_f1", "joint_f1")}
                for p in ORDER[1:]}

    # ── C5 on-demand cache accounting ──
    C = [ret_c[("c5_ondemand", q)]["diagnostics"]["cache"] for q in qids]
    cache = {k: sum(c[k] for c in C) for k in ("lookups", "hits_ok", "hits_failed", "misses", "extracted_ok",
                                               "extracted_failed_model", "extracted_failed_infra")}
    cache.update(extraction_wall_s=round(sum(c["extraction_wall_s"] for c in C), 1),
                 cost_usd=round(sum(c.get("extraction_cost_usd", 0) for c in C), 5),
                 prompt_tokens=sum(c.get("extraction_prompt_tokens", 0) for c in C),
                 completion_tokens=sum(c["extraction_eval_tokens"] for c in C),
                 unique_paragraphs_extracted=len({x for c in C for x in c["extracted_chunk_ids"]}),
                 hit_rate=round((cache["hits_ok"] + cache["hits_failed"]) / cache["lookups"], 4) if cache["lookups"] else None)
    ce = []
    for f in (M / "c5/extraction_cache/openrouter").glob("*/cache.jsonl"):
        ce += jl(f)
    cache["cache_file_entries"] = len(ce)
    cache["failures_by_kind"] = {"truncated_length": sum("finish_reason=length" in e["error"] for e in ce if e["status"] != "ok"),
                                 "other": sum("finish_reason=length" not in e["error"] for e in ce if e["status"] != "ok")}
    qs = {q["_id"]: q for q in json.loads((B / "questions.json").read_text(encoding="utf-8"))}
    c5fail = {e["chunk_id"] for e in ce if e["status"] != "ok"}
    cache["gold_paragraphs_failed_in_c5_cache"] = sorted({c for q in qids for c in qs[q]["gold_chunk_ids"] if c in c5fail})

    offline = json.loads((IDX / "offline_build_report.json").read_text(encoding="utf-8"))
    meta = json.loads((IDX / "index_meta.json").read_text(encoding="utf-8"))
    fail = json.loads((M / "extraction_failures_gold_check.json").read_text(encoding="utf-8"))
    out = {
        "label": "Main five-pipeline benchmark, 100 HotpotQA dev-distractor questions, one run per pipeline",
        "verification": ver,
        "summary": summary,
        "paired_bootstrap_vs_dense": vs_dense,
        "offline_full_graph_build": {
            "extraction_model": offline["extraction_model"], "provider": "openrouter",
            "paragraphs": offline["extraction"]["paragraphs"], "failed": offline["extraction"]["failed"],
            "failure_kinds": {"truncated_length": sum(f["kind"].startswith("truncated") for f in fail["failures"]),
                              "http_429_after_retries": sum("429" in f["kind"] for f in fail["failures"])},
            "extraction_wall_clock": f"{fail['extraction_first']} -> {fail['extraction_last']} (8 parallel requests)",
            "sum_of_request_times_s": offline["extraction"]["time_s_total"],
            "mean_request_s": offline["extraction"]["time_s_mean_per_paragraph"],
            "prompt_tokens": offline["extraction"]["prompt_tokens"], "completion_tokens": offline["extraction"]["completion_tokens"],
            "cost_usd": offline["extraction"]["cost_usd"],
            "graph_construction_s": offline["graph_construction_s"], "embedding_and_collections_s": offline["embedding_and_collections_s"],
            "graph": meta["graph"], "entities_indexed": meta["entities_indexed"], "relationships_indexed": meta["relationships_indexed"],
            "gold_paragraphs_failed": fail["gold_paragraphs_failed"], "gold_paragraphs_total": fail["gold_paragraphs_total"],
            "affected_questions": fail["affected_questions"]},
        "c5_on_demand_extraction": cache,
    }
    (M / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verification": {k: v for k, v in ver.items() if k != "generation_settings"}}, indent=1))
    for p in ORDER:
        s = summary[p]
        print(f"{s['label']:34} " + " ".join(f"{m}={s[m]:.3f}" for m in METRICS) + f" | {s['latency_s']}")
    print(json.dumps(vs_dense, indent=0)[:3000])
    print(json.dumps(cache, indent=1))


if __name__ == "__main__":
    main()
