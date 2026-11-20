"""
local_9b_timing.py -- small isolated local extraction timing experiment (n = 2 paragraphs).

Models: qwen2.5:3b (baseline) and mistral:latest (7B; installed). A 9B model (gemma2:9b) could not be
downloaded: drive C: had under 2 GB free. No download was made for this run.
Reference: Gemini 2.5 Flash-Lite results saved by the provider pilot re-test (same paragraphs, same prompt,
same schema fields; Gemini used strict JSON schema + temperature 0 + seed 42 + max_tokens 4096).

Same two paragraphs as Experiment B ('New Mosque (Istanbul)', 'Rango (2011 film)'), same original MindMesh
prompt and production local settings (ExtractionCache). One fresh cache per model under
data/benchmarks/hotpotqa/local_9b_timing_20261121/<model>/ -- nothing is shared with any other experiment.

Per model: cold-load time is measured with a 1-token request after unloading (Ollama load_duration),
then each paragraph is extracted. GPU (temperature, power, memory, utilisation), CPU and RAM are sampled
every second for the whole run. Projections for the 985-paragraph corpus and the observed C5 workload
(516 paragraphs) are computed from measured seconds per paragraph character and labelled as projections.
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

import ollama
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval" / "hotpotqa"))

from graphrag_lite.retrieval.extraction_cache import ExtractionCache  # noqa: E402
from local_extraction_comparison import es, prf, provenance, rs  # noqa: E402
from mindmesh.benchmarks.hotpotqa.runtime import GpuLock, ollama_ps, resolve_local_model, unload_model  # noqa: E402

MODELS = ["qwen2.5:3b", "mistral:latest"]
PICKS = ["New Mosque (Istanbul)", "Rango (2011 film)"]
PILOT_CORPUS = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42/corpus.jsonl"
CORPUS_985 = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/dev_distractor_100q_seed20261119/corpus.jsonl"
C5_CACHE = ROOT / "data/benchmarks/hotpotqa/main5_gemini_20261120/c5/extraction_cache/openrouter"
GEM_BUILD = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/dev_distractor_100q_seed20261119/index/openrouter__google_gemini-2.5-flash-lite/offline_build_report.json"
GEMINI = ROOT / "data/benchmarks/hotpotqa/provider_pilot_20261120/runs/retest_ac_20261120/cloud/openrouter"
E3 = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/extraction_validation/results.jsonl"
OUT = ROOT / "data/benchmarks/hotpotqa/local_7b_timing_20261121"


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.rows, self.label = threading.Event(), [], "idle"

    def run(self):
        psutil.cpu_percent(None)
        while not self.stop.is_set():
            g = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(", ")
            vm = psutil.virtual_memory()
            try:
                self.rows.append({"t": time.time(), "label": self.label, "gpu_temp_c": float(g[0]), "gpu_power_w": float(g[1]),
                                  "gpu_mem_mib": float(g[2]), "gpu_util": float(g[3]), "sm_mhz": float(g[4]),
                                  "cpu_pct": psutil.cpu_percent(None), "ram_used_gb": round(vm.used / 2**30, 2)})
            except (ValueError, IndexError):
                pass
            time.sleep(1.0)

    def summary(self, label):
        r = [x for x in self.rows if x["label"] == label]
        if not r:
            return None
        f = lambda k: {"mean": round(st.fmean(x[k] for x in r), 1), "max": max(x[k] for x in r)}  # noqa: E731
        return {"samples": len(r), **{k: f(k) for k in ("gpu_temp_c", "gpu_power_w", "gpu_mem_mib", "gpu_util",
                                                       "sm_mhz", "cpu_pct", "ram_used_gb")}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paras = {p["title"]: p for p in map(json.loads, PILOT_CORPUS.read_text(encoding="utf-8").splitlines()) if p}
    sel = [paras[t] for t in PICKS]
    mistral = {r["chunk_id"]: r for r in map(json.loads, E3.read_text(encoding="utf-8").splitlines())
               if r["arm"] == "baseline" and r["parse_ok"]}
    gem = {}
    for f in GEMINI.glob("*/cache.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            e = json.loads(l); gem[e["chunk_id"]] = e
    sampler = Sampler(); sampler.start()
    res = {}
    with GpuLock("local 9B timing experiment"):
        for m in MODELS:
            tag, info = resolve_local_model(m)
            unload_model(tag); time.sleep(3)
            sampler.label = f"{m}:load"
            t0 = time.monotonic()
            r = ollama.Client(timeout=600).generate(model=tag, prompt="ok", options={"num_predict": 1, "num_ctx": 8192})
            load = {"wall_s": round(time.monotonic() - t0, 2), "ollama_load_duration_s": round((r.load_duration or 0) / 1e9, 2)}
            ps = ollama_ps()
            cache = ExtractionCache(OUT / m.replace(":", "_"), tag, info.get("digest", ""))
            rows = {}
            for p in sel:
                sampler.label = f"{m}:extract"
                e = cache.get(p["text"]) or cache.extract(p, "local_9b_timing")
                rows[p["chunk_id"]] = e
                print(f"[{m}] {p['title'][:24]:24} {e['status']:12} {e['extraction_wall_s']:6.1f}s out {e['eval_tokens']} "
                      f"att {e['attempts']} ent {len(e['entities'])} rel {len(e['relationships'])}", flush=True)
            sampler.label = "idle"
            unload_model(tag)
            res[m] = {"tag": tag, "digest": info.get("digest"), "details": {k: info.get(k) for k in ("parameter_size", "quantization", "size")},
                      "load": load, "ollama_ps_after_load": ps, "cache_version": cache.version, "config": cache.config, "rows": rows}
    sampler.stop.set(); time.sleep(1.2)

    ref = res["qwen2.5:3b"]["rows"]
    table = {}
    for m, R in res.items():
        per = []
        for p in sel:
            c = p["chunk_id"]; e = R["rows"][c]; ok = e["status"] == "ok"
            agree = lambda other: ({"entities": prf(es(other), es(e)), "relationships": prf(rs(other), rs(e))}  # noqa: E731
                                   if ok and other else None)
            per.append({"title": p["title"], "chars": len(p["text"]), "status": e["status"], "valid_json": ok,
                        "wall_s": e["extraction_wall_s"], "attempts": e["attempts"],
                        "input_tokens": e["prompt_tokens"], "output_tokens": e["eval_tokens"],
                        "decode_tok_s": round(e["eval_tokens"] / e["eval_s"], 1) if e["eval_s"] else None,
                        "truncation_suspected": e["eval_tokens"] >= 4096, "error": e["error"][:200],
                        "entities": len(e["entities"]), "relationships": len(e["relationships"]),
                        "provenance": provenance(e, p["text"]) if ok else None,
                        "agreement_vs_qwen2.5_3b": agree(ref[c] if ref[c]["status"] == "ok" else None) if m != "qwen2.5:3b" else None,
                        "agreement_vs_gemini": agree(gem.get(c) if gem.get(c, {}).get("status") == "ok" else None),
                        "agreement_vs_mistral_baseline": agree(mistral.get(c))})
        table[m] = {"digest": R["digest"], "details": R["details"], "load": R["load"], "ollama_ps_after_load": R["ollama_ps_after_load"],
                    "cache_version": R["cache_version"], "hardware_during_extraction": sampler.summary(f"{m}:extract"),
                    "hardware_during_load": sampler.summary(f"{m}:load"), "paragraphs": per}
    gem_rows = [{"title": p["title"], "status": gem[p["chunk_id"]]["status"], "wall_s": gem[p["chunk_id"]]["request_wall_s"],
                 "input_tokens": gem[p["chunk_id"]]["prompt_tokens"], "output_tokens": gem[p["chunk_id"]]["completion_tokens"],
                 "cost_usd": gem[p["chunk_id"]]["cost_usd"], "entities": len(gem[p["chunk_id"]]["entities"]),
                 "relationships": len(gem[p["chunk_id"]]["relationships"]),
                 "provenance": provenance(gem[p["chunk_id"]], p["text"]) if gem[p["chunk_id"]]["status"] == "ok" else None}
                for p in sel if p["chunk_id"] in gem]

    # ── projections (labelled) ──
    lens985 = [len(json.loads(l)["text"]) for l in CORPUS_985.read_text(encoding="utf-8").splitlines() if l.strip()]
    c5_lens = [len(json.loads(l)["text"]) for f in C5_CACHE.glob("*/cache.jsonl") for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    proj = {"basis": "PROJECTION, not measured: total = (sum of paragraph characters) x (measured wall seconds per character, "
                     "mean over the 2 paragraphs, including retries); sequential, one model on one GPU; excludes thermal "
                     "throttling, failures beyond those observed, and model load (added once).",
            "corpus_985": {"paragraphs": len(lens985), "chars_total": sum(lens985), "chars_mean": round(st.fmean(lens985))},
            "c5_workload_observed": {"paragraphs": len(c5_lens), "chars_total": sum(c5_lens), "source": "main 100q C5 run cache (Gemini)"}}
    for m in MODELS:
        per = table[m]["paragraphs"]
        spc = sum(r["wall_s"] for r in per) / sum(r["chars"] for r in per)
        load_s = table[m]["load"]["wall_s"]
        proj[m] = {"s_per_char": round(spc, 5),
                   "full_985_hours": round((spc * sum(lens985) + load_s) / 3600, 2),
                   "c5_516_hours": round((spc * sum(c5_lens) + load_s) / 3600, 2)}
    gb = json.loads(GEM_BUILD.read_text(encoding="utf-8"))
    proj["gemini_measured_reference"] = {"full_985": "about 10 min wall clock with 8 parallel requests; $0.703 (MEASURED)",
                                         "c5_516": "2,195 s sequential = 0.61 h; $0.331 (MEASURED)",
                                         "sum_of_request_times_s": gb["extraction"]["time_s_total"]}
    report = {"label": "Small isolated local extraction timing experiment (n = 2 paragraphs). Not a benchmark. "
                       "Agreement is not correctness. Projections are estimates.",
              "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "paragraphs": [{"title": p["title"], "chars": len(p["text"])} for p in sel],
              "settings": res["qwen2.5:3b"]["config"], "results": table, "gemini_reference": gem_rows, "projections": proj,
              "electricity": "not estimated: only GPU board power from nvidia-smi is available (no CPU/system meter), and "
                             "it gave an implausible 501 W reading earlier in this session; GPU power is reported as indicative only."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "hardware_samples.jsonl", "w", encoding="utf-8") as fh:
        for r in sampler.rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({m: {k: table[m][k] for k in ("load", "hardware_during_extraction")} for m in MODELS}, indent=1))
    print(json.dumps(proj, indent=1))


if __name__ == "__main__":
    main()
