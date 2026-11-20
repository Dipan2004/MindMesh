"""
eval/hotpotqa/analyze_c5_ondemand.py -- analysis of the Dense vs on-demand C5 experiment.

    python eval/hotpotqa/analyze_c5_ondemand.py --run-dir <exp>/runs/<run_id>

Reads results.jsonl, retrieval.jsonl, the run's extraction cache and GPU telemetry.
Writes analysis.json, c5_ondemand_comparison.csv and c5_ondemand_per_question.csv into the run dir.
All quality and cost numbers are MEASURED from raw rows; the only estimate (full-corpus
extraction time = corpus size x measured mean per-paragraph time) is labelled as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "hotpotqa"))
from analyze_expanded import paired_bootstrap  # noqa: E402

PIPES = ["pipeline_a", "c5_ondemand"]
METRICS = ["recall@5", "recall@10", "mrr", "all_gold_in_top10", "ans_em", "ans_f1", "sp_prec", "sp_recall", "sp_f1",
           "joint_em", "joint_f1", "retrieval_latency_s", "generation_latency_s", "total_latency_s"]


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def q(vals):
    vals = list(vals)
    return {"mean": round(st.fmean(vals), 4), "median": round(st.median(vals), 4), "max": round(max(vals), 4),
            "total": round(sum(vals), 3)} if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    rd = args.run_dir
    cfg = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    res = jl(rd / "results.jsonl"); ret = jl(rd / "retrieval.jsonl")
    qids = cfg["question_ids"]
    by = {(r["pipeline"], r["qid"]): r for r in res}
    rby = {(r["pipeline"], r["qid"]): r for r in ret}
    out: dict = {"run_dir": str(rd), "n_questions": len(qids), "generator": cfg["generator"],
                 "extractor": cfg["extractor"], "cache_version": cfg["cache_version"], "pipelines": {}, "by_type": {},
                 "c5_minus_dense": {}, "failures": {}}
    types = {r["qid"]: r["type"] for r in res}
    for p in PIPES:
        pr = [by[(p, x)] for x in qids if (p, x) in by]
        out["pipelines"][p] = {"n": len(pr), **{m: round(st.fmean(r[m] for r in pr), 4) for m in METRICS}}
        out["failures"][p] = {"failed_retrieval": sum(any(e.startswith("retrieval") for e in r["errors"]) for r in pr),
                              "failed_generation": sum(any(e.startswith("generation") for e in r["errors"]) for r in pr),
                              "invalid_output": sum(not r.get("generation_parse_ok") for r in pr),
                              "empty_retrieval": sum(not r["retrieved_chunk_ids"] for r in pr)}
        for t in ("bridge", "comparison"):
            sub = [by[(p, x)] for x in qids if (p, x) in by and types[x] == t]
            if not sub:
                continue
            out["by_type"].setdefault(t, {})[p] = {"n": len(sub), **{m: round(st.fmean(r[m] for r in sub), 4)
                                                                    for m in ("recall@5", "recall@10", "mrr", "all_gold_in_top10",
                                                                              "ans_f1", "sp_f1")}}
    common = [x for x in qids if ("pipeline_a", x) in by and ("c5_ondemand", x) in by]
    for m in ("recall@5", "recall@10", "mrr", "all_gold_in_top10", "ans_em", "ans_f1", "sp_f1", "joint_f1",
              "retrieval_latency_s", "total_latency_s"):
        out["c5_minus_dense"][m] = paired_bootstrap([by[("pipeline_a", x)][m] for x in common],
                                                    [by[("c5_ondemand", x)][m] for x in common])

    # C5 diagnostics and MEASURED extraction cost
    diags = [rby[("c5_ondemand", x)]["diagnostics"] for x in qids if ("c5_ondemand", x) in rby and "diagnostics" in rby[("c5_ondemand", x)]]
    cache_rows = jl(Path(cfg["cache_dir"]) / "cache.jsonl")
    run_rows = [e for e in cache_rows if e["triggered_by_question"] in set(qids)]
    gold_cov = []
    for x in qids:
        r = rby.get(("c5_ondemand", x))
        if r and "diagnostics" in r:
            g = by[("c5_ondemand", x)]["gold_chunk_ids"]
            gold_cov.append(sum(c in r["diagnostics"]["selected_paragraphs"] for c in g) / len(g))
    cum_lookups = cum_hits = 0
    hit_curve = []
    for d in diags:
        c = d["cache"]; cum_lookups += c["lookups"]; cum_hits += c["hits_ok"] + c["hits_failed"]
        hit_curve.append(round(cum_hits / cum_lookups, 4) if cum_lookups else 0.0)
    ok = [e for e in run_rows if e["status"] == "ok"]
    out["c5_diagnostics"] = {
        "paragraphs_selected_per_question": q(d["n_selected"] for d in diags),
        "valid_records_used_per_question": q(d["valid_records_used"] for d in diags),
        "selected_but_failed_extraction_per_question": q(d["selected_failed_or_missing"] for d in diags),
        "expansion_from_question_titles_per_question": q(len(d["from_question"]) for d in diags),
        "expansion_from_entity_titles_per_question": q(len(d["from_entities"]) for d in diags),
        "graph_nodes": q(d["graph_nodes"] for d in diags), "graph_edges": q(d["graph_edges"] for d in diags),
        "questions_with_question_entity_anchor": sum("fallback" not in d for d in diags),
        "questions_falling_back_to_dense_order": sum("fallback" in d for d in diags),
        "gold_paragraph_coverage_of_selection": round(st.fmean(gold_cov), 4) if gold_cov else None,
        "all_gold_selected": sum(v == 1.0 for v in gold_cov),
    }
    out["measured_extraction_cost"] = {
        "label": "MEASURED: on-demand extraction actually performed in this run",
        "unique_paragraphs_extracted": len(run_rows),
        "share_of_corpus_extracted": round(len(run_rows) / 985, 4),
        "status_counts": {s: sum(e["status"] == s for e in run_rows) for s in ("ok", "failed_model", "failed_infra")},
        "failed_entries": [{"title": e["title"], "status": e["status"], "error": e["error"][:120]}
                           for e in run_rows if e["status"] != "ok"],
        "extraction_wall_s_total": round(sum(e["extraction_wall_s"] for e in run_rows), 1),
        "extraction_wall_s_per_paragraph": q(e["extraction_wall_s"] for e in run_rows),
        "extraction_output_tokens_per_paragraph": q(e["eval_tokens"] for e in run_rows),
        "decode_tok_per_s": round(sum(e["eval_tokens"] for e in run_rows) / max(sum(e["eval_s"] for e in run_rows), 1e-9), 1),
        "entities_per_ok_paragraph": round(st.fmean(len(e["entities"]) for e in ok), 2) if ok else None,
        "relationships_per_ok_paragraph": round(st.fmean(len(e["relationships"]) for e in ok), 2) if ok else None,
        "cache_lookups_total": sum(d["cache"]["lookups"] for d in diags),
        "cache_hits_total": sum(d["cache"]["hits_ok"] + d["cache"]["hits_failed"] for d in diags),
        "cache_misses_total": sum(d["cache"]["misses"] for d in diags),
        "cumulative_hit_rate_at_question": {str(i + 1): hit_curve[i] for i in (0, 9, 24, 49, 74, 99) if i < len(hit_curve)},
        "per_question_extraction_s": q(d["cache"]["extraction_wall_s"] for d in diags),
        "questions_with_zero_extraction": sum(d["cache"]["misses"] == 0 for d in diags),
    }
    mean_s = out["measured_extraction_cost"]["extraction_wall_s_per_paragraph"]["mean"] if run_rows else None
    out["estimated_full_corpus_extraction"] = {
        "label": "ESTIMATE (not measured): 985 paragraphs x the measured mean qwen2.5:3b time per paragraph",
        "estimated_h": round(985 * mean_s / 3600, 2) if mean_s else None,
        "measured_query_focused_h": round(out["measured_extraction_cost"]["extraction_wall_s_total"] / 3600, 2),
    }
    out["c5_latency_breakdown_s"] = {k: q(d["timing_s"][k] for d in diags) for k in
                                     ("dense_retrieval", "extraction", "graph_construction", "traversal_and_ranking",
                                      "retrieval_total")}
    tf = rd / "gpu_telemetry.csv"
    if tf.exists():
        temps, clocks, mem = [], [], []
        for row in csv.reader(tf.read_text(encoding="utf-8", errors="ignore").splitlines()):
            try:
                temps.append(float(row[1])); clocks.append(float(row[2].split()[0])); mem.append(float(row[4].split()[0]))
            except (ValueError, IndexError):
                continue
        if temps:
            out["gpu"] = {"samples_15s": len(temps), "temp_c_mean": round(st.fmean(temps), 1), "temp_c_max": max(temps),
                          "share_ge_87c": round(sum(t >= 87 for t in temps) / len(temps), 3),
                          "sm_clock_mhz_median": st.median(clocks), "vram_used_mib_max": max(mem)}
    (rd / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(rd / "c5_ondemand_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pipeline", "n"] + METRICS + list(out["failures"]["pipeline_a"]))
        for p in PIPES:
            w.writerow([p, out["pipelines"][p]["n"]] + [out["pipelines"][p][m] for m in METRICS] + list(out["failures"][p].values()))
    with open(rd / "c5_ondemand_per_question.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["qid", "type", "pipeline", "gold_answer", "pred_answer"] + METRICS + ["c5_selected", "c5_misses", "c5_extraction_s", "errors"]
        w = csv.writer(f); w.writerow(cols)
        for x in qids:
            for p in PIPES:
                r = by.get((p, x))
                if not r:
                    continue
                d = rby.get((p, x), {}).get("diagnostics", {})
                w.writerow([x, r["type"], p, r["gold_answer"], r["pred_answer"]] + [r[m] for m in METRICS]
                           + [d.get("n_selected", ""), d.get("cache", {}).get("misses", ""),
                              d.get("cache", {}).get("extraction_wall_s", ""), " | ".join(r["errors"])])
    print(json.dumps({k: v for k, v in out.items() if k not in ("c5_minus_dense",)}, indent=1)[:5000])
    print("C5 - Dense:", {m: (v["diff"], v["ci95"]) for m, v in out["c5_minus_dense"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
