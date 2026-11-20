"""
local_extraction_comparison.py -- EXPERIMENT B: local Ollama extraction-model comparison (small pilot).

Same two paragraphs as the provider pilot (median and 90th-percentile length of the 30 E3 paragraphs:
'New Mosque (Istanbul)', 'Rango (2011 film)'), same original MindMesh prompt, same production extraction
path and settings (ExtractionCache: temperature 0, seed 42, num_predict 4096, num_ctx 8192, frequency
penalty 0.25 with the [0.5, 1.0] retry ladder). One model-specific cache per model under
data/benchmarks/hotpotqa/local_extraction_comparison_20261120/<model>/. Nothing here feeds Experiment A.

Models requested: qwen2.5:1.5b, qwen2.5:3b, qwen2.5:4b, qwen2.5:7b. Models not installed are marked
unavailable; nothing is downloaded.

Agreement is reported pairwise against qwen2.5:3b, and against two references for the same paragraphs:
the earlier Mistral baseline (E3) and Gemini 2.5 Flash Lite (provider pilot). Agreement is not correctness.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from graphrag_lite.graph.builder import _norm  # noqa: E402
from graphrag_lite.retrieval.extraction_cache import ExtractionCache  # noqa: E402
from mindmesh.benchmarks.hotpotqa.runtime import (  # noqa: E402
    GpuLock, ModelNotAvailable, gpu_info, installed_models, resolve_local_model, unload_model)

MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:4b", "qwen2.5:7b"]
CORPUS = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42/corpus.jsonl"
E3 = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/extraction_validation/results.jsonl"
GEMINI = ROOT / "data/benchmarks/hotpotqa/provider_pilot_20261120/runs/retest_ac_20261120/cloud/openrouter"
OUT = ROOT / "data/benchmarks/hotpotqa/local_extraction_comparison_20261120"
PICKS = ["New Mosque (Istanbul)", "Rango (2011 film)"]


def prf(ref: set, var: set) -> dict:
    tp = len(ref & var); p = tp / len(var) if var else 0.0; r = tp / len(ref) if ref else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "tp": tp, "ref": len(ref), "var": len(var)}


def es(rec): return {_norm(e["name"]) for e in rec["entities"]}
def rs(rec): return {(_norm(x["source"]), _norm(x["target"])) for x in rec["relationships"]}


def provenance(rec, text: str) -> dict:
    items = rec["entities"] + rec["relationships"]
    nt = " ".join(text.split())
    has = sum(bool((it.get("source_sentence") or "").strip()) for it in items)
    verb = sum(" ".join((it.get("source_sentence") or "").split()) in nt for it in items
               if (it.get("source_sentence") or "").strip())
    return {"items": len(items), "with_source_sentence": round(has / len(items), 3) if items else None,
            "source_sentence_verbatim": round(verb / len(items), 3) if items else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paras = {p["title"]: p for p in map(json.loads, CORPUS.read_text(encoding="utf-8").splitlines()) if p}
    sel = [paras[t] for t in PICKS]
    mistral = {r["chunk_id"]: r for r in map(json.loads, E3.read_text(encoding="utf-8").splitlines())
               if r["arm"] == "baseline" and r["parse_ok"]}
    gem = {}
    for f in GEMINI.glob("*/cache.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            e = json.loads(l)
            if e["status"] == "ok":
                gem[e["chunk_id"]] = e
    inst = installed_models()
    results, availability = {}, {}
    with GpuLock("experiment B local extraction comparison"):
        for m in MODELS:
            try:
                tag, info = resolve_local_model(m)
            except ModelNotAvailable:
                availability[m] = "unavailable (not installed locally; not downloaded)"
                print(f"[skip] {m}: not installed"); continue
            availability[m] = f"installed, digest {info.get('digest')}"
            cache = ExtractionCache(OUT / m.replace(":", "_"), tag, info.get("digest", ""))
            rows = {}
            for p in sel:
                e = cache.get(p["text"]) or cache.extract(p, "experiment_b")
                rows[p["chunk_id"]] = e
                print(f"[{m}] {p['title'][:26]:26} {e['status']:12} {e['extraction_wall_s']:6.1f}s "
                      f"out {e['eval_tokens']} att {e['attempts']} ent {len(e['entities'])} rel {len(e['relationships'])}")
            results[m] = {"tag": tag, "digest": info.get("digest"), "quantization": info.get("quantization"),
                          "cache_version": cache.version, "config": cache.config, "rows": rows}
            unload_model(tag)
    ref = results.get("qwen2.5:3b", {}).get("rows", {})
    table = {}
    for m, R in results.items():
        per = []
        for p in sel:
            e = R["rows"][p["chunk_id"]]; ok = e["status"] == "ok"; c = p["chunk_id"]
            per.append({
                "title": p["title"], "status": e["status"], "valid_json": ok, "wall_s": e["extraction_wall_s"],
                "decode_tok_s": round(e["eval_tokens"] / e["eval_s"], 1) if e["eval_s"] else None,
                "input_tokens": e["prompt_tokens"], "output_tokens": e["eval_tokens"], "attempts": e["attempts"],
                "truncation_suspected": e["eval_tokens"] >= 4096, "error": e["error"][:200],
                "entities": len(e["entities"]), "relationships": len(e["relationships"]),
                "provenance": provenance(e, p["text"]) if ok else None,
                "agreement_vs_qwen2.5_3b": ({"entities": prf(es(ref[c]), es(e)), "relationships": prf(rs(ref[c]), rs(e))}
                                            if ok and m != "qwen2.5:3b" and ref.get(c, {}).get("status") == "ok" else None),
                "agreement_vs_mistral_baseline": ({"entities": prf(es(mistral[c]), es(e)), "relationships": prf(rs(mistral[c]), rs(e))}
                                                  if ok and c in mistral else None),
                "agreement_vs_gemini_pilot": ({"entities": prf(es(gem[c]), es(e)), "relationships": prf(rs(gem[c]), rs(e))}
                                              if ok and c in gem else None),
            })
        table[m] = {"digest": R["digest"], "cache_version": R["cache_version"],
                    "valid_json_rate": f"{sum(r['valid_json'] for r in per)}/{len(per)}", "paragraphs": per}
    report = {"label": "EXPERIMENT B: small local extraction-model comparison (n=2 paragraphs). Not a benchmark. "
                       "Agreement is not correctness. Not used in Experiment A.",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "availability": availability,
              "paragraphs": [{"chunk_id": p["chunk_id"], "title": p["title"], "chars": len(p["text"])} for p in sel],
              "settings": next(iter(results.values()))["config"] if results else None,
              "gpu": gpu_info(), "results": table}
    (OUT / "experiment_b_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"availability": availability}, indent=1))


if __name__ == "__main__":
    main()
