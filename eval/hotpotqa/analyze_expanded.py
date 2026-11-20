"""
eval/hotpotqa/analyze_expanded.py -- tables, paired bootstrap CIs and CSVs for a HotpotQA run.

    python eval/hotpotqa/analyze_expanded.py --bench-dir <dir> --tag 100q --out <exp dir>

Reads only the raw per-question rows (results.jsonl) of the run in <bench-dir>/runs/.
Writes:
  <out>/hotpotqa_<tag>_analysis.json        all numbers below
  <out>/hotpotqa_pipeline_comparison_<tag>.csv
  <out>/hotpotqa_per_question_results_<tag>.csv

Paired bootstrap: questions are resampled with replacement (10,000 resamples,
seed 12345); for each resample the mean difference (pipeline - Dense) is taken
over the same questions, giving a 95% percentile interval. An interval that
excludes 0 is reported as "CI excludes 0" -- not as a significance claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

PIPES = ["pipeline_a", "pipeline_b", "lightrag_mix", "lightrag_mix_opt"]
LABEL = {"pipeline_a": "MindMesh Dense", "pipeline_b": "MindMesh Graph",
         "lightrag_mix": "MindMesh Mix", "lightrag_mix_opt": "MindMesh Mix Optimized"}
METRICS = ["recall@5", "recall@10", "mrr", "all_gold_in_top10", "ans_em", "ans_f1",
           "sp_prec", "sp_recall", "sp_f1", "joint_em", "joint_f1",
           "retrieval_latency_s", "generation_latency_s", "total_latency_s"]
CI_METRICS = ["recall@5", "recall@10", "mrr", "all_gold_in_top10", "ans_em", "ans_f1", "sp_f1",
              "joint_f1", "retrieval_latency_s", "total_latency_s"]


def load_rows(bench: Path, run_id: str | None) -> tuple[Path, list[dict]]:
    runs = [d for d in (bench / "runs").iterdir() if d.is_dir() and d.name.startswith("gen-")]
    rd = (bench / "runs" / run_id) if run_id else max(runs, key=lambda d: d.stat().st_mtime)
    rows = [json.loads(l) for l in (rd / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return rd, rows


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 12345) -> dict:
    rng = random.Random(seed)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    obs = statistics.fmean(diffs)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {"diff": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)], "ci_excludes_0": lo > 0 or hi < 0,
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    rd, rows = load_rows(args.bench_dir, args.run_id)
    qs = json.loads((args.bench_dir / "questions.json").read_text(encoding="utf-8"))
    qids = [q["_id"] for q in qs]
    by = {(r["pipeline"], r["qid"]): r for r in rows}

    res = {"run_dir": str(rd), "n_questions": len(qids), "pipelines": {}, "by_type": {}, "vs_dense": {}, "failures": {}}
    for p in PIPES:
        pr = [by[(p, q)] for q in qids]
        res["pipelines"][p] = {m: round(statistics.fmean(r[m] for r in pr), 4) for m in METRICS}
        res["failures"][p] = {
            "failed_retrieval": sum(any(e.startswith("retrieval") for e in r["errors"]) for r in pr),
            "failed_generation": sum(any(e.startswith("generation") for e in r["errors"]) for r in pr),
            "invalid_or_truncated_output": sum(not r.get("generation_parse_ok") for r in pr),
            "empty_retrieval": sum(not r["retrieved_chunk_ids"] for r in pr),
        }
        for t in ("bridge", "comparison"):
            sub = [by[(p, q["_id"])] for q in qs if q["type"] == t]
            res["by_type"].setdefault(t, {})[p] = {"n": len(sub), **{m: round(statistics.fmean(r[m] for r in sub), 4)
                                                                   for m in ("recall@10", "mrr", "ans_em", "ans_f1", "sp_f1", "joint_f1")}}
    for p in PIPES[1:]:
        res["vs_dense"][p] = {}
        for m in CI_METRICS:
            a = [by[("pipeline_a", q)][m] for q in qids]
            b = [by[(p, q)][m] for q in qids]
            res["vs_dense"][p][m] = paired_bootstrap(a, b)
        for t in ("bridge", "comparison"):
            sub = [q["_id"] for q in qs if q["type"] == t]
            for m in ("recall@10", "ans_f1", "sp_f1"):
                a = [by[("pipeline_a", q)][m] for q in sub]
                b = [by[(p, q)][m] for q in sub]
                res["vs_dense"][p][f"{t}:{m}"] = paired_bootstrap(a, b)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"hotpotqa_{args.tag}_analysis.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    with open(args.out / f"hotpotqa_pipeline_comparison_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "pipeline", "label", "n"] + METRICS + list(res["failures"]["pipeline_a"]))
        for p in PIPES:
            w.writerow([args.tag, p, LABEL[p], len(qids)] + [res["pipelines"][p][m] for m in METRICS]
                       + list(res["failures"][p].values()))

    with open(args.out / f"hotpotqa_per_question_results_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["experiment", "qid", "type", "pipeline", "gold_answer", "pred_answer", "recall@5", "recall@10", "mrr",
                "all_gold_in_top10", "ans_em", "ans_f1", "sp_prec", "sp_recall", "sp_f1", "joint_em", "joint_f1",
                "retrieval_latency_s", "generation_latency_s", "total_latency_s", "generation_parse_ok", "errors",
                "retrieved_chunk_ids", "gold_chunk_ids", "pred_sp", "gold_sp"]
        w.writerow(cols)
        for q in qs:
            for p in PIPES:
                r = by[(p, q["_id"])]
                w.writerow([args.tag, q["_id"], q["type"], p, q["answer"], r["pred_answer"]]
                           + [r[c] for c in cols[6:20]] + [r.get("generation_parse_ok"), " | ".join(r["errors"]),
                                                          " ".join(r["retrieved_chunk_ids"]), " ".join(r["gold_chunk_ids"]),
                                                          json.dumps(r["pred_sp"], ensure_ascii=False), json.dumps(r["gold_sp"], ensure_ascii=False)])
    print(json.dumps({p: {k: res["pipelines"][p][k] for k in ("recall@10", "mrr", "ans_f1", "sp_f1")} for p in PIPES}, indent=1))
    print("vs dense (diff, CI):", {p: {m: (v["diff"], v["ci95"]) for m, v in d.items() if ":" not in m and m in ("recall@10", "ans_f1", "sp_f1")}
                                   for p, d in res["vs_dense"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
