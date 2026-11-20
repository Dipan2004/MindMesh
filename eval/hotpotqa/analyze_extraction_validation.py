"""
eval/hotpotqa/analyze_extraction_validation.py -- paired analysis of the extraction-variant validation.

    python eval/hotpotqa/analyze_extraction_validation.py

Reads data/benchmarks/hotpotqa/expanded_20261119/extraction_validation/{results.jsonl,
gpu_telemetry.csv, config.json} and writes analysis.json there.

Quality is measured as AGREEMENT with the baseline extraction (same model, current
prompt), plus reference-free grounding checks. There are no gold entity or
relationship annotations for HotpotQA paragraphs, so agreement is not correctness.
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphrag_lite.graph.builder import _norm  # noqa: E402

VD = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/extraction_validation"
PILOT = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42"
ARMS = ["baseline", "C", "BC"]


def prf(base: set, other: set) -> dict:
    tp = len(base & other)
    p = tp / len(other) if other else (1.0 if not base else 0.0)
    r = tp / len(base) if base else (1.0 if not other else 0.0)
    return {"tp": tp, "base": len(base), "other": len(other), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def micro(items: list[dict]) -> dict:
    tp = sum(i["tp"] for i in items); b = sum(i["base"] for i in items); o = sum(i["other"] for i in items)
    p = tp / o if o else 0.0; r = tp / b if b else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "baseline_items": b, "variant_items": o}


def boot_ci(vals: list[float], n: int = 10000, seed: int = 7) -> list[float]:
    rng = random.Random(seed); k = len(vals)
    ms = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(ms[int(0.025 * n)], 4), round(ms[int(0.975 * n) - 1], 4)]


def grounded(name: str, text: str) -> bool:
    return _norm(name) in _norm(text)


def main() -> int:
    rows = [json.loads(l) for l in (VD / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["arm"], r["chunk_id"]): r for r in rows}
    cids = [c for c in dict.fromkeys(r["chunk_id"] for r in rows) if all((a, c) in by for a in ARMS)]
    paras = {json.loads(l)["chunk_id"]: json.loads(l) for l in (PILOT / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    out: dict = {"n_paired_paragraphs": len(cids), "limitations": [
        "No gold entity/relationship annotations exist for these paragraphs: quality = agreement with the "
        "baseline extraction (same model, current prompt) plus reference-free grounding checks, not correctness.",
        "Entity matching uses MindMesh's normalised name (_norm); relationships match on normalised (source, target) "
        "pairs, with type compared separately. Paraphrased names count as disagreements.",
        "Timing includes thermal throttling; arms are interleaved per paragraph to share thermal conditions."]}

    # timing / tokens
    out["arms"] = {}
    for a in ARMS:
        rs = [by[(a, c)] for c in cids]
        w = [r["wall_s"] for r in rs]; t = [r["eval_count"] for r in rs]
        out["arms"][a] = {
            "wall_s_mean": round(st.fmean(w), 2), "wall_s_median": round(st.median(w), 2), "wall_s_total": round(sum(w), 1),
            "output_tokens_mean": round(st.fmean(t), 1), "output_tokens_median": st.median(t), "output_tokens_total": sum(t),
            "decode_tok_per_s": round(sum(t) / max(sum(r["eval_s"] for r in rs), 1e-9), 1),
            "parse_failures": sum(not r["parse_ok"] for r in rs),
            "paragraphs_with_retries": sum(r["attempts"] > 1 for r in rs),
            "entities_total": sum(len(r["entities"]) for r in rs),
            "relationships_total": sum(len(r["relationships"]) for r in rs)}
    b = out["arms"]["baseline"]
    for a in ("C", "BC"):
        x = out["arms"][a]
        x["time_reduction_total_pct"] = round(100 * (1 - x["wall_s_total"] / b["wall_s_total"]), 1)
        x["token_reduction_total_pct"] = round(100 * (1 - x["output_tokens_total"] / b["output_tokens_total"]), 1)
        # paired per-paragraph consistency
        tr = [1 - by[(a, c)]["wall_s"] / by[("baseline", c)]["wall_s"] for c in cids]
        kr = [1 - by[(a, c)]["eval_count"] / max(by[("baseline", c)]["eval_count"], 1) for c in cids]
        dec = [by[(a, c)]["eval_count"] / by[(a, c)]["eval_s"] for c in cids if by[(a, c)]["eval_s"]]
        x["paired"] = {
            "time_reduction_per_paragraph_median_pct": round(100 * st.median(tr), 1),
            "time_reduction_per_paragraph_mean_pct": round(100 * st.fmean(tr), 1),
            "time_reduction_mean_ci95_pct": [round(100 * v, 1) for v in boot_ci(tr)],
            "paragraphs_faster": sum(v > 0 for v in tr), "paragraphs_slower": sum(v < 0 for v in tr),
            "token_reduction_per_paragraph_median_pct": round(100 * st.median(kr), 1),
            "token_reduction_mean_ci95_pct": [round(100 * v, 1) for v in boot_ci(kr)],
            "paragraphs_fewer_tokens": sum(v > 0 for v in kr), "paragraphs_more_tokens": sum(v < 0 for v in kr),
            "decode_tok_per_s_median": round(st.median(dec), 1) if dec else None,
        }
        # outlier sensitivity: drop the 3 paragraphs with the largest baseline time
        top = sorted(cids, key=lambda c: -by[("baseline", c)]["wall_s"])[:3]
        rest = [c for c in cids if c not in top]
        x["paired"]["time_reduction_total_pct_excluding_3_slowest_baseline"] = round(
            100 * (1 - sum(by[(a, c)]["wall_s"] for c in rest) / sum(by[("baseline", c)]["wall_s"] for c in rest)), 1)
        x["paired"]["share_of_time_saving_from_top3_paragraphs_pct"] = round(
            100 * sum(by[("baseline", c)]["wall_s"] - by[(a, c)]["wall_s"] for c in top)
            / max(sum(by[("baseline", c)]["wall_s"] - by[(a, c)]["wall_s"] for c in cids), 1e-9), 1)

    # quality vs fresh baseline (paired, only where both parsed)
    out["quality_vs_baseline"] = {}
    for a in ("C", "BC"):
        ent, rel, rel_typed, diffs = [], [], [], []
        dup_e = dup_r = type_changed_e = type_changed_r = sent_changed = 0
        for c in cids:
            B, V = by[("baseline", c)], by[(a, c)]
            if not (B["parse_ok"] and V["parse_ok"]):
                continue
            be = {_norm(e["name"]): e for e in B["entities"]}; ve = {_norm(e["name"]): e for e in V["entities"]}
            br = {(_norm(r["source"]), _norm(r["target"])): r for r in B["relationships"]}
            vr = {(_norm(r["source"]), _norm(r["target"])): r for r in V["relationships"]}
            ent.append(prf(set(be), set(ve))); rel.append(prf(set(br), set(vr)))
            rel_typed.append(prf({k + (str(v.get("type", "")).upper(),) for k, v in br.items()},
                                 {k + (str(v.get("type", "")).upper(),) for k, v in vr.items()}))
            dup_e += len(V["entities"]) - len(ve); dup_r += len(V["relationships"]) - len(vr)
            for k in set(be) & set(ve):
                type_changed_e += str(be[k].get("type")).upper() != str(ve[k].get("type")).upper()
                sent_changed += " ".join(str(be[k].get("source_sentence", "")).split()) != " ".join(str(ve[k].get("source_sentence", "")).split())
            for k in set(br) & set(vr):
                type_changed_r += str(br[k].get("type")).upper() != str(vr[k].get("type")).upper()
            missing_r = sorted(set(br) - set(vr)); added_r = sorted(set(vr) - set(br))
            missing_e = sorted(set(be) - set(ve)); added_e = sorted(set(ve) - set(be))
            if missing_e or added_e or missing_r or added_r:
                diffs.append({"chunk_id": c, "title": paras[c]["title"],
                              "missing_entities": missing_e, "added_entities": added_e,
                              "missing_relationships": [f"{s} -> {t}" for s, t in missing_r],
                              "added_relationships": [f"{s} -> {t}" for s, t in added_r]})
        per_para_rel_recall = [i["recall"] for i in rel if i["base"]]
        out["quality_vs_baseline"][a] = {
            "entities_micro": micro(ent), "relationships_micro": micro(rel), "relationships_typed_micro": micro(rel_typed),
            "relationship_recall_per_paragraph_median": round(st.median(per_para_rel_recall), 4) if per_para_rel_recall else None,
            "paragraphs_losing_over_half_of_baseline_relationships": sum(1 for v in per_para_rel_recall if v < 0.5),
            "duplicate_entities_within_paragraph": dup_e, "duplicate_relationships_within_paragraph": dup_r,
            "same_entity_different_type": type_changed_e, "same_relationship_pair_different_type": type_changed_r,
            "same_entity_different_source_sentence": sent_changed,
            "paragraphs_with_any_difference": len(diffs), "differences": diffs}

    # reference-free grounding per arm
    out["grounding"] = {}
    for a in ARMS:
        ne = ng = nr = nrg = ns = nsv = title_hit = 0
        for c in cids:
            r = by[(a, c)]; txt = paras[c]["text"]; norm_txt = " ".join(txt.split())
            for e in r["entities"]:
                ne += 1; ng += grounded(e["name"], txt)
            for rl in r["relationships"]:
                nr += 1; nrg += grounded(rl["source"], txt) and grounded(rl["target"], txt)
            for it in r["entities"] + r["relationships"]:
                if it.get("source_sentence"):
                    ns += 1; nsv += " ".join(it["source_sentence"].split()) in norm_txt
            title_hit += any(_norm(e["name"]) == _norm(paras[c]["title"]) or _norm(paras[c]["title"]).startswith(_norm(e["name"]))
                             for e in r["entities"])
        out["grounding"][a] = {"entity_name_in_text": round(ng / ne, 4) if ne else None,
                               "relationship_endpoints_in_text": round(nrg / nr, 4) if nr else None,
                               "source_sentence_verbatim": round(nsv / ns, 4) if ns else None,
                               "paragraph_title_entity_extracted": f"{title_hit}/{len(cids)}"}

    # thermal
    th = {}
    tf = VD / "gpu_telemetry.csv"
    if tf.exists():
        temps, clocks, reasons = [], [], Counter()
        for row in csv.reader(tf.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if len(row) < 6:
                continue
            try:
                temps.append(float(row[1])); clocks.append(float(row[2].split()[0])); reasons[row[5].strip()] += 1
            except ValueError:
                continue
        if temps:
            th = {"samples_5s": len(temps), "temp_c_mean": round(st.fmean(temps), 1), "temp_c_max": max(temps),
                  "share_samples_ge_87c": round(sum(t >= 87 for t in temps) / len(temps), 3),
                  "sm_clock_mhz_median": st.median(clocks), "sm_clock_mhz_min": min(clocks),
                  "event_reasons": dict(reasons.most_common(6)),
                  "note": "telemetry started 20:13, ~5.5 min after validation start (first ~3 paragraphs not covered); "
                          "event reason 0x20 = software thermal slowdown"}
    out["thermal"] = th
    out["thermal"]["interruptions"] = ("Windows hibernated/rebooted the machine at 20:40:29 (thermal); the run "
                                        "stopped after 87/90 arm-runs and was resumed for the last paragraph at 20:43")

    # ── qwen2.5:3b (B+C) extraction-model comparison, if present ─────────────
    QD = VD.parent / "extraction_validation_qwen2.5_3b"
    if (QD / "results.jsonl").exists():
        qrows = {json.loads(l)["chunk_id"]: json.loads(l)
                 for l in (QD / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        qc = [c for c in cids if c in qrows]
        qcfg = json.loads((QD / "config.json").read_text(encoding="utf-8"))
        w = [qrows[c]["wall_s"] for c in qc]; t = [qrows[c]["eval_count"] for c in qc]
        q = {"model": qcfg.get("model"), "digest": qcfg.get("digest"), "parameter_size": qcfg.get("parameter_size"),
             "quantization": qcfg.get("quantization"), "n_paragraphs": len(qc),
             "wall_s_mean": round(st.fmean(w), 2), "wall_s_median": round(st.median(w), 2), "wall_s_total": round(sum(w), 1),
             "output_tokens_mean": round(st.fmean(t), 1), "output_tokens_total": sum(t),
             "decode_tok_per_s": round(sum(t) / max(sum(qrows[c]["eval_s"] for c in qc), 1e-9), 1),
             "parse_failures": sum(not qrows[c]["parse_ok"] for c in qc),
             "paragraphs_with_retries": sum(qrows[c]["attempts"] > 1 for c in qc),
             "entities_total": sum(len(qrows[c]["entities"]) for c in qc),
             "relationships_total": sum(len(qrows[c]["relationships"]) for c in qc)}
        for ref in ("baseline", "BC"):
            ref_total = sum(by[(ref, c)]["wall_s"] for c in qc)
            ref_tok = sum(by[(ref, c)]["eval_count"] for c in qc)
            q[f"time_reduction_vs_mistral_{ref}_pct"] = round(100 * (1 - q["wall_s_total"] / ref_total), 1)
            q[f"token_reduction_vs_mistral_{ref}_pct"] = round(100 * (1 - q["output_tokens_total"] / ref_tok), 1)
            ent, rel, rel_t = [], [], []
            lost = 0
            for c in qc:
                B, V = by[(ref, c)], qrows[c]
                if not (B["parse_ok"] and V["parse_ok"]):
                    continue
                be = {_norm(e["name"]) for e in B["entities"]}; ve = {_norm(e["name"]) for e in V["entities"]}
                br = {(_norm(r["source"]), _norm(r["target"])): str(r.get("type", "")).upper() for r in B["relationships"]}
                vr = {(_norm(r["source"]), _norm(r["target"])): str(r.get("type", "")).upper() for r in V["relationships"]}
                ent.append(prf(be, ve)); rel.append(prf(set(br), set(vr)))
                rel_t.append(prf({k + (v,) for k, v in br.items()}, {k + (v,) for k, v in vr.items()}))
                lost += bool(br) and len(set(br) & set(vr)) / len(br) < 0.5
            q[f"agreement_vs_mistral_{ref}"] = {"entities_micro": micro(ent), "relationships_micro": micro(rel),
                                               "relationships_typed_micro": micro(rel_t),
                                               "paragraphs_losing_over_half_of_mistral_relationships": lost}
        ne = ng = nr = nrg = ns = nsv = title_hit = 0
        for c in qc:
            r = qrows[c]; txt = paras[c]["text"]; norm_txt = " ".join(txt.split())
            for e in r["entities"]:
                ne += 1; ng += grounded(e["name"], txt)
            for rl in r["relationships"]:
                nr += 1; nrg += grounded(rl["source"], txt) and grounded(rl["target"], txt)
            for it in r["entities"] + r["relationships"]:
                if it.get("source_sentence"):
                    ns += 1; nsv += " ".join(it["source_sentence"].split()) in norm_txt
            title_hit += any(_norm(e["name"]) == _norm(paras[c]["title"]) or _norm(paras[c]["title"]).startswith(_norm(e["name"]))
                             for e in r["entities"])
        q["grounding"] = {"entity_name_in_text": round(ng / ne, 4) if ne else None,
                          "relationship_endpoints_in_text": round(nrg / nr, 4) if nr else None,
                          "source_sentence_verbatim": round(nsv / ns, 4) if ns else None,
                          "paragraph_title_entity_extracted": f"{title_hit}/{len(qc)}"}
        tf2 = QD / "gpu_telemetry.csv"
        if tf2.exists():
            temps, clocks = [], []
            for row in csv.reader(tf2.read_text(encoding="utf-8", errors="ignore").splitlines()):
                try:
                    temps.append(float(row[1])); clocks.append(float(row[2].split()[0]))
                except (ValueError, IndexError):
                    continue
            if temps:
                q["thermal"] = {"samples_5s": len(temps), "temp_c_mean": round(st.fmean(temps), 1), "temp_c_max": max(temps),
                                "share_samples_ge_87c": round(sum(x >= 87 for x in temps) / len(temps), 3),
                                "sm_clock_mhz_median": st.median(clocks)}
        q["caveat"] = ("qwen2.5:3b ran after the mistral arms (not interleaved), so its timing reflects a different "
                       "thermal window; compare decode tok/s and output tokens as well as wall time.")
        out["qwen2.5_3b_BC"] = q

    (VD / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("quality_vs_baseline",)}, indent=1, ensure_ascii=False)[:6000])
    for a in ("C", "BC"):
        q = out["quality_vs_baseline"][a]
        print(a, {k: v for k, v in q.items() if k != "differences"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
