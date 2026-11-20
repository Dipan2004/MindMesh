"""
eval/hotpotqa/validate_extraction_variants.py -- measure extraction speed-up options before adoption.

    python eval/hotpotqa/validate_extraction_variants.py [--n 30]

Arms (same model, same generation options as eval/hotpotqa/build_index.py):
  baseline  current MindMesh extraction prompt, unconstrained output (as in production)
  C         same prompt text, Ollama `format` = extraction JSON schema
  BC        sentence-index variant: the paragraph's sentences are numbered and the
            model returns `sentence_id` instead of re-typing `source_sentence`;
            code restores the exact sentence; plus the JSON schema of C

Paragraphs: the first N pilot paragraphs ordered by chunk_id (fixed, not chosen).
Arms are interleaved per paragraph (baseline, C, BC) so GPU heat affects them equally.
All arms use the retry ladder of the production extractor on parse failure.
Post-processing for every arm is the production `_extract_json` + `_parse_extraction`.

This script does not modify the extractor, the indexes or any experiment; it writes
only to data/benchmarks/hotpotqa/expanded_20261119/extraction_validation/.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import ollama  # noqa: E402

from graphrag_lite.config import get_config  # noqa: E402
from graphrag_lite.graph.builder import _norm  # noqa: E402
from graphrag_lite.ingestion import extractor as X  # noqa: E402
from graphrag_lite.llm.ollama_client import OllamaClient  # noqa: E402
from mindmesh.benchmarks.hotpotqa.runtime import GpuLock, ollama_ps, resolve_local_model  # noqa: E402
from mindmesh.model_config import think_for  # noqa: E402

PILOT = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42"
OUT = ROOT / "data/benchmarks/hotpotqa/expanded_20261119/extraction_validation"
MODEL = "mistral:latest"

_ENT = {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"},
                                          "description": {"type": "string"}}, "required": ["name", "type", "description"]}
_REL = {"type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"},
                                          "type": {"type": "string"}, "description": {"type": "string"}},
        "required": ["source", "target", "type", "description"]}


def schema(field: str, ftype: str) -> dict:
    ent = json.loads(json.dumps(_ENT)); rel = json.loads(json.dumps(_REL))
    for s in (ent, rel):
        s["properties"][field] = {"type": ftype}
        s["required"].append(field)
    return {"type": "object", "properties": {"entities": {"type": "array", "items": ent},
                                             "relationships": {"type": "array", "items": rel}},
            "required": ["entities", "relationships"]}


SCHEMA_C = schema("source_sentence", "string")
SCHEMA_BC = schema("sentence_id", "integer")

# Variant B prompt: identical wording to the production template except that the
# evidence field is the sentence number instead of the verbatim sentence.
PROMPT_BC = (X._EXTRACTION_PROMPT_TEMPLATE
             .replace("and the EXACT sentence from the text where it appears.",
                      "and the sentence_id: the [number] of the sentence in the text where it appears.")
             .replace("3. source_sentence must be copied verbatim from the TEXT — do not paraphrase.",
                      "3. sentence_id must be the number shown in [brackets] before that sentence.")
             .replace('"source_sentence": "..."', '"sentence_id": 0'))


def numbered_text(title: str, sentences: list[str]) -> str:
    return title + "\n" + "\n".join(f"[{i}] {s.strip()}" for i, s in enumerate(sentences))


def restore_sentences(raw: dict, sentences: list[str]) -> dict:
    for k in ("entities", "relationships"):
        for item in raw.get(k, []) or []:
            sid = item.pop("sentence_id", None)
            try:
                item["source_sentence"] = sentences[int(sid)].strip()
            except Exception:
                item["source_sentence"] = ""
    return raw


def call(client_kwargs: dict, fmt) -> tuple[str, dict]:
    kw = dict(client_kwargs)
    if fmt is not None:
        kw["format"] = fmt
    r = ollama.Client(timeout=600).generate(**kw)
    return r.response or "", {"eval_count": r.eval_count or 0, "eval_s": (r.eval_duration or 0) / 1e9,
                              "prompt_count": r.prompt_eval_count or 0, "prompt_s": (r.prompt_eval_duration or 0) / 1e9}


def run_arm(arm: str, para: dict, base_client: OllamaClient, cfg) -> dict:
    if arm == "BC":
        prompt = PROMPT_BC.format(chunk_text=numbered_text(para["title"], para["sentences"]))
        fmt = SCHEMA_BC
    else:
        prompt = X._EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=para["text"])
        fmt = SCHEMA_C if arm == "C" else None
    ladder = [None] + [p for p in cfg.LLM_RETRY_FREQUENCY_PENALTIES if float(p) != float(base_client.frequency_penalty)]
    t0 = time.monotonic()
    stats = {"eval_count": 0, "eval_s": 0.0, "prompt_count": 0, "prompt_s": 0.0}
    out = {"arm": arm, "chunk_id": para["chunk_id"], "title": para["title"], "attempts": 0}
    for pen in ladder:
        client = base_client if pen is None else base_client.with_overrides(frequency_penalty=pen)
        raw, s = call(client._kwargs(prompt), fmt)
        out["attempts"] += 1
        for k in stats:
            stats[k] += s[k]
        try:
            j = X._extract_json(raw)
            if arm == "BC":
                j = restore_sentences(j, para["sentences"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = X._parse_extraction(j, para["chunk_id"], para["doc_id"], chunk_text=para["text"],
                                          allow_stub_endpoints=cfg.EXTRACTION_STUB_ENDPOINTS)
            out.update(entities=res["entities"], relationships=res["relationships"], parse_ok=True)
            break
        except Exception as exc:
            out.update(entities=[], relationships=[], parse_ok=False, error=str(exc)[:200])
    out["wall_s"] = round(time.monotonic() - t0, 3)
    out.update({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    return out


def agreement(base: dict, other: dict) -> dict:
    be = {_norm(e["name"]) for e in base["entities"]}; oe = {_norm(e["name"]) for e in other["entities"]}
    br = {(_norm(r["source"]), _norm(r["target"])) for r in base["relationships"]}
    orr = {(_norm(r["source"]), _norm(r["target"])) for r in other["relationships"]}
    return {"ent_base": len(be), "ent_other": len(oe), "ent_overlap": len(be & oe),
            "rel_base": len(br), "rel_other": len(orr), "rel_overlap": len(br & orr)}


def verbatim_rate(rec: dict, text: str) -> tuple[int, int]:
    norm_text = " ".join(text.split())
    items = rec["entities"] + rec["relationships"]
    ok = sum(1 for it in items if it.get("source_sentence") and " ".join(it["source_sentence"].split()) in norm_text)
    return ok, len(items)


def main(argv=None) -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default=MODEL, help="exact local Ollama tag (extraction-model comparison)")
    ap.add_argument("--arms", default="baseline,C,BC", help="comma-separated subset of baseline,C,BC")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default extraction_validation/)")
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.out:
        OUT = args.out if args.out.is_absolute() else ROOT / args.out
    cfg = get_config()
    model, info = resolve_local_model(args.model)
    base_client = OllamaClient(model=model, temperature=cfg.EXTRACTION_TEMPERATURE, seed=cfg.EXTRACTION_SEED,
                               timeout=cfg.LLM_TIMEOUT_S, num_predict=cfg.LLM_NUM_PREDICT, num_ctx=cfg.LLM_NUM_CTX,
                               think=think_for(model), frequency_penalty=cfg.LLM_FREQUENCY_PENALTY)
    paras = sorted((json.loads(l) for l in (PILOT / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()),
                   key=lambda p: p["chunk_id"])[: args.n]
    pilot_ext = {json.loads(l)["chunk_id"]: json.loads(l)
                 for l in (PILOT / "index/mistral_latest/extractions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = {(json.loads(l)["arm"], json.loads(l)["chunk_id"]) for l in res_path.read_text(encoding="utf-8").splitlines()
            if l.strip()} if res_path.exists() else set()
    (OUT / "config.json").write_text(json.dumps({
        "model": model, "digest": info.get("digest"), "parameter_size": info.get("parameter_size"),
        "quantization": info.get("quantization"), "n_paragraphs": len(paras),
        "paragraph_selection": "first N pilot paragraphs by chunk_id", "arms": arms,
        "client_kwargs": base_client._kwargs(None), "retry_penalties": cfg.LLM_RETRY_FREQUENCY_PENALTIES,
        "schema_C": SCHEMA_C, "schema_BC": SCHEMA_BC, "prompt_BC_template": PROMPT_BC,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")

    with GpuLock("extraction variant validation"):
        with open(res_path, "a", encoding="utf-8") as fh:
            for i, p in enumerate(paras, 1):
                for arm in arms:
                    if (arm, p["chunk_id"]) in done:
                        continue
                    r = run_arm(arm, p, base_client, cfg)
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
                    tps = r["eval_count"] / r["eval_s"] if r["eval_s"] else 0
                    print(f"[{i}/{len(paras)}] {arm:8s} {p['title'][:28]:28s} {r['wall_s']:6.1f}s out={r['eval_count']:5d} "
                          f"{tps:4.1f} tok/s att={r['attempts']} ok={r['parse_ok']} ent={len(r['entities'])} rel={len(r['relationships'])}", flush=True)
        ps = ollama_ps()

    if sorted(arms) != ["BC", "C", "baseline"]:
        (OUT / "ollama_ps_at_end.json").write_text(json.dumps(ps, indent=2), encoding="utf-8")
        print("single-model comparison arm(s) done; analysed by analyze_extraction_validation.py")
        return 0
    rows =[json.loads(l) for l in res_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["arm"], r["chunk_id"]): r for r in rows}
    cids = [p["chunk_id"] for p in paras]
    text = {p["chunk_id"]: p["text"] for p in paras}
    summ = {"n_paragraphs": len(cids), "ollama_ps": ps, "arms": {}}
    for arm in ("baseline", "C", "BC"):
        rs = [by[(arm, c)] for c in cids]
        ag_pilot = [agreement({"entities": pilot_ext[c]["entities"], "relationships": pilot_ext[c]["relationships"]}, by[(arm, c)])
                    for c in cids if "_parse_error" not in pilot_ext[c]]
        ag_fresh = [agreement(by[("baseline", c)], by[(arm, c)]) for c in cids if by[("baseline", c)]["parse_ok"]]
        vb = [verbatim_rate(r, text[r["chunk_id"]]) for r in rs]

        def rec(ag, kind):
            base = sum(a[f"{kind}_base"] for a in ag)
            return round(sum(a[f"{kind}_overlap"] for a in ag) / base, 4) if base else None

        def prec(ag, kind):
            oth = sum(a[f"{kind}_other"] for a in ag)
            return round(sum(a[f"{kind}_overlap"] for a in ag) / oth, 4) if oth else None
        summ["arms"][arm] = {
            "wall_s_mean": round(st.fmean(r["wall_s"] for r in rs), 2),
            "wall_s_median": round(st.median(r["wall_s"] for r in rs), 2),
            "wall_s_total": round(sum(r["wall_s"] for r in rs), 1),
            "output_tokens_mean": round(st.fmean(r["eval_count"] for r in rs), 1),
            "decode_tok_per_s": round(sum(r["eval_count"] for r in rs) / max(sum(r["eval_s"] for r in rs), 1e-9), 1),
            "parse_failures": sum(not r["parse_ok"] for r in rs),
            "paragraphs_needing_retry": sum(r["attempts"] > 1 for r in rs),
            "entities_total": sum(len(r["entities"]) for r in rs),
            "relationships_total": sum(len(r["relationships"]) for r in rs),
            "entity_recall_vs_pilot": rec(ag_pilot, "ent"), "entity_precision_vs_pilot": prec(ag_pilot, "ent"),
            "relationship_recall_vs_pilot": rec(ag_pilot, "rel"), "relationship_precision_vs_pilot": prec(ag_pilot, "rel"),
            "entity_recall_vs_fresh_baseline": rec(ag_fresh, "ent"),
            "relationship_recall_vs_fresh_baseline": rec(ag_fresh, "rel"),
            "source_sentence_verbatim_rate": round(sum(a for a, _ in vb) / max(sum(b for _, b in vb), 1), 4),
        }
    b = summ["arms"]["baseline"]
    for arm in ("C", "BC"):
        a = summ["arms"][arm]
        a["time_reduction_vs_baseline"] = round(1 - a["wall_s_total"] / b["wall_s_total"], 4)
        a["gate"] = {
            "time_reduction_ge_25pct": a["time_reduction_vs_baseline"] >= 0.25,
            "entity_recall_vs_baseline_ge_0.90": (a["entity_recall_vs_fresh_baseline"] or 0) >= 0.90,
            "relationship_recall_vs_baseline_ge_0.85": (a["relationship_recall_vs_fresh_baseline"] or 0) >= 0.85,
            "parse_failures_not_higher": a["parse_failures"] <= b["parse_failures"],
            "verbatim_rate_not_lower": a["source_sentence_verbatim_rate"] >= b["source_sentence_verbatim_rate"],
        }
        a["passes_gate"] = all(a["gate"].values())
    summ["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in v if kk not in ("gate",)} for k, v in summ["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
