"""
provider_pilot.py -- SMALL PROVIDER TIMING AND EXTRACTION-QUALITY PILOT (2 paragraphs). Not a benchmark.

Runs the same two HotpotQA dev-distractor paragraphs, with the same original MindMesh extraction
prompt and schema, through:
  * local Ollama qwen2.5:3b       (existing ExtractionCache path, own cache namespace)
  * OpenRouter google/gemini-2.5-flash-lite (provider_extraction, own cache namespace, no fallback)
and a separate fallback self-test (OpenRouter pointed at an unreachable URL, so the local fallback
path is exercised and recorded; its timings are NOT provider timings).

Paragraphs: from the 30 paragraphs of the earlier extraction-variant study (pilot corpus, not the
frozen internal test set), the one with median length and the one at the 90th length percentile,
chosen by rule before any extraction. Answer quality is NOT evaluated.

Output: data/benchmarks/hotpotqa/provider_pilot_20261120/
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from graphrag_lite.graph.builder import _norm  # noqa: E402
from graphrag_lite.retrieval import provider_extraction as PE  # noqa: E402
from graphrag_lite.retrieval.extraction_cache import ExtractionCache  # noqa: E402
from mindmesh.benchmarks.hotpotqa.runtime import GpuLock, resolve_local_model  # noqa: E402

CORPUS = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42/corpus.jsonl"
E3 = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/extraction_validation/results.jsonl"
OUT = ROOT / "data/benchmarks/hotpotqa/provider_pilot_20261120"
LOCAL_MODEL = "qwen2.5:3b"


def prf(ref: set, var: set) -> dict:
    tp = len(ref & var)
    p = tp / len(var) if var else 0.0
    r = tp / len(ref) if ref else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "tp": tp, "ref_items": len(ref), "var_items": len(var)}


def ent_set(rec):
    return {_norm(e["name"]) for e in rec["entities"]}


def rel_set(rec):
    return {(_norm(x["source"]), _norm(x["target"])) for x in rec["relationships"]}


def power_state() -> dict:
    import subprocess
    def q(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc}"
    return {"battery_status": q(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Battery).BatteryStatus"]),
            "battery_status_meaning": "1 = discharging (on battery), 2 = on AC",
            "gpu": q(["nvidia-smi", "--query-gpu=name,temperature.gpu,pstate,clocks.max.sm", "--format=csv,noheader"])}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="sub-folder for a re-test; empty = the original pilot folder")
    ap.add_argument("--no-selftest", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.run:
        OUT = OUT / "runs" / args.run
    OUT.mkdir(parents=True, exist_ok=True)
    power_before = power_state()
    print("power:", power_before)
    paras = {p["chunk_id"]: p for p in map(json.loads, CORPUS.read_text(encoding="utf-8").splitlines()) if p}
    e3 = [json.loads(l) for l in E3.read_text(encoding="utf-8").splitlines() if l.strip()]
    mistral = {r["chunk_id"]: r for r in e3 if r["arm"] == "baseline"}
    pool = sorted(mistral, key=lambda c: (len(paras[c]["text"]), c))
    picks = [pool[len(pool) // 2], pool[int(0.9 * (len(pool) - 1))]]
    sel = [paras[c] for c in picks]
    print("paragraphs:", [(p["title"], len(p["text"])) for p in sel])

    settings = PE.load_provider_settings()
    settings = replace(settings, provider="openrouter")          # explicit selection for this pilot
    s = settings.openrouter
    if not PE.api_key_available(s):
        sys.exit(f"{s.api_key_env} not found in the environment or .env -- OpenRouter pilot not run.")
    support = PE.endpoint_support(s)
    (OUT / "endpoint_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
    if not support["any_endpoint_supports_all"]:
        sys.exit("No OpenRouter endpoint supports strict structured outputs + all sent parameters. Not run.")

    # 1) OpenRouter (cloud timings come only from these entries; fallback recorded separately)
    orc = PE.OpenRouterExtractionCache(OUT / "cloud", s)
    cloud = {}
    for p in sel:
        e = orc.extract(p, "provider_pilot")
        cloud[p["chunk_id"]] = e
        print(f"[openrouter] {p['title'][:30]:30} {e['status']:12} {e['request_wall_s']:.2f}s "
              f"tok {e['prompt_tokens']}/{e['completion_tokens']} cost {e['cost_usd']} "
              f"model {e['actual_model']} via {e['upstream_provider']} ent {len(e['entities'])} rel {len(e['relationships'])}")

    # 2) local qwen2.5:3b, same prompt (existing local path)
    tag, info = resolve_local_model(LOCAL_MODEL)
    local = {}
    with GpuLock("provider_pilot local qwen2.5:3b"):
        lc = ExtractionCache(OUT / "local" if not args.run else OUT / "cloud" / "ollama", tag, info.get("digest", ""))
        for p in sel:
            e = lc.get(p["text"]) or lc.extract(p, "provider_pilot")
            local[p["chunk_id"]] = e
            print(f"[ollama]     {p['title'][:30]:30} {e['status']:12} {e['extraction_wall_s']:.2f}s "
                  f"tok {e['prompt_tokens']}/{e['eval_tokens']} ent {len(e['entities'])} rel {len(e['relationships'])}")

        # 2b) real fallback for any paragraph OpenRouter did not extract (shares the local cache namespace,
        #     so a paragraph already extracted locally above is reused, not re-extracted)
        fallback_used = []
        if args.run:
            px = PE.ProviderExtractor(OUT / "cloud", settings)
            for p in sel:
                if cloud[p["chunk_id"]]["status"] != "ok":
                    r = px.extract(p, "provider_pilot")
                    fallback_used.append({"chunk_id": p["chunk_id"], "title": p["title"], "used_provider": r["used_provider"],
                                          "fallback_status": (r["fallback"] or {}).get("status"),
                                          "openrouter_error": r["primary"]["error"][:200]})
            print("[fallback]", fallback_used)

        # 3) fallback self-test: unreachable primary -> local fallback (mechanism check only)
        fallback_selftest = None
        if not args.no_selftest:
            bad = replace(settings, openrouter=replace(s, base_url="http://127.0.0.1:9/api/v1", max_retries=0))
            fx = PE.ProviderExtractor(OUT / "fallback_selftest", bad)
            ft = fx.extract(sel[0], "fallback_selftest")
            fallback_selftest = {"primary_status": ft["primary"]["status"], "primary_error": ft["primary"]["error"][:160],
                                 "used_provider": ft["used_provider"], "fallback_model": (ft["fallback"] or {}).get("model"),
                                 "fallback_status": (ft["fallback"] or {}).get("status"),
                                 "fallback_wall_s": (ft["fallback"] or {}).get("extraction_wall_s"),
                                 "note": "mechanism check only; primary URL deliberately unreachable; not a timing result"}
            print("[fallback self-test]", fallback_selftest)

    # 4) comparison
    rows = []
    for p in sel:
        c, l, m = cloud[p["chunk_id"]], local[p["chunk_id"]], mistral.get(p["chunk_id"])
        both = c["status"] == "ok" and l["status"] == "ok"
        rows.append({
            "chunk_id": p["chunk_id"], "title": p["title"], "chars": len(p["text"]),
            "openrouter": {"status": c["status"], "json_valid": c["json_valid"], "wall_s": c["request_wall_s"],
                           "prompt_tokens": c["prompt_tokens"], "output_tokens": c["completion_tokens"],
                           "reasoning_tokens": c["reasoning_tokens"], "cost_usd": c["cost_usd"],
                           "retries": c["retries"], "actual_model": c["actual_model"],
                           "upstream_provider": c["upstream_provider"],
                           "entities": len(c["entities"]), "relationships": len(c["relationships"])},
            "ollama_qwen2.5_3b": {"status": l["status"], "wall_s": l["extraction_wall_s"],
                                  "prompt_tokens": l["prompt_tokens"], "output_tokens": l["eval_tokens"],
                                  "attempts": l["attempts"], "model_digest": l["model_digest"],
                                  "entities": len(l["entities"]), "relationships": len(l["relationships"])},
            "agreement_gemini_vs_qwen": {"entities": prf(ent_set(l), ent_set(c)),
                                         "relationships": prf(rel_set(l), rel_set(c))} if both else None,
            "agreement_vs_mistral_baseline_reference": {
                "gemini": {"entities": prf(ent_set(m), ent_set(c)), "relationships": prf(rel_set(m), rel_set(c))}
                if m and m["parse_ok"] and c["status"] == "ok" else None,
                "qwen2.5:3b": {"entities": prf(ent_set(m), ent_set(l)), "relationships": prf(rel_set(m), rel_set(l))}
                if m and m["parse_ok"] and l["status"] == "ok" else None},
        })
    ok_c = [cloud[c] for c in picks if cloud[c]["status"] == "ok"]
    ok_l = [local[c] for c in picks if local[c]["status"] == "ok"]
    report = {
        "label": "SMALL PROVIDER TIMING AND EXTRACTION-QUALITY PILOT (n=2 paragraphs). Not a benchmark conclusion. "
                 "No answer-quality comparison. Agreement is not correctness.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paragraph_selection": "median and 90th-percentile length among the 30 E3 paragraphs (pilot corpus)",
        "prompt_template_sha256": orc.prompt_hash, "schema_sha256": PE.SCHEMA_SHA256,
        "openrouter": {"requested_model": s.model, "base_url": s.base_url, "endpoint": s.endpoint,
                       "cache_version": orc.version, "config": orc.config,
                       "valid_json_rate": f"{sum(cloud[c]['json_valid'] for c in picks)}/{len(picks)}",
                       "wall_s_mean": round(st.fmean(e["request_wall_s"] for e in ok_c), 3) if ok_c else None,
                       "total_cost_usd": round(sum(cloud[c]["cost_usd"] or 0 for c in picks), 8),
                       "fallbacks": len(fallback_used) if args.run else 0, "fallback_detail": fallback_used if args.run else []},
        "ollama": {"model": tag, "digest": info.get("digest"), "cache_version": lc.version, "config": lc.config,
                   "valid_rate": f"{len(ok_l)}/{len(picks)}",
                   "wall_s_mean": round(st.fmean(e["extraction_wall_s"] for e in ok_l), 3) if ok_l else None},
        "paragraphs": rows,
        "fallback_selftest": fallback_selftest,
        "power_before": power_before, "power_after": power_state(),
    }
    (OUT / "pilot_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("openrouter", "ollama")}, indent=1))
    for r in rows:
        print(r["title"], json.dumps(r["agreement_gemini_vs_qwen"]))


if __name__ == "__main__":
    main()
