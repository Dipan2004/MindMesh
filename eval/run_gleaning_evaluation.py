"""
run_gleaning_evaluation.py — Build graph + collections + evaluate all 6 pipelines
after genuine gleaning extraction completes.

Run AFTER extract_with_gleaning.py has written all 42 chunks:
    python eval/extract_with_gleaning.py --status   # verify 42/42
    python eval/run_gleaning_evaluation.py

Outputs (all in data/experiments/qwen2_5_1_5b_genuine_gleaning_on/):
    graph.graphml
    graph_metrics.json
    genuine_gleaning_metrics.json
    genuine_gleaning_evaluation.json
    genuine_gleaning_comparison.json
    genuine_gleaning_report.md

Never touches:
    data/cache          (canonical)
    data/experiments/qwen2_5_1_5b_validate   (Phase 1)
    data/experiments/qwen2_5_1_5b_v2_*       (stub-filter arms)
    eval/gold_test_set_frozen.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

EXP_DIR    = ROOT / "data" / "experiments" / "qwen2_5_1_5b_genuine_gleaning_on"
JSONL_PATH = EXP_DIR / "extractions.jsonl"
DEV_SET    = ROOT / "eval" / "gold_dev_set.json"
PHASE1_JSON = ROOT / "data" / "experiments" / "qwen2_5_1_5b_validate" / "validation_20261109_223804.json"
V2OFF_JSON  = ROOT / "data" / "experiments" / "qwen2_5_1_5b_v2_off" / "evaluation.json"
V2ON_JSON   = ROOT / "data" / "experiments" / "qwen2_5_1_5b_v2_on" / "evaluation.json"
TOP_K      = 6
MODEL      = "qwen2.5:1.5b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gleaning_eval")


def check_extraction_complete() -> tuple[list[dict], dict]:
    """Load and verify the gleaning extraction. Returns (records, summary)."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Extraction JSONL not found: {JSONL_PATH}")

    records = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    parse_errors  = sum(1 for r in records if r.get("_parse_error"))
    gleaning_ok   = sum(1 for r in records if r.get("gleaning_ok"))
    extract_calls = sum(r.get("extract_calls", 0) for r in records)
    glean_calls   = sum(r.get("gleaning_calls", 0) for r in records)
    total_e       = sum(r.get("entities_final", len(r.get("entities", []))) for r in records)
    total_r       = sum(r.get("rels_final", len(r.get("relationships", []))) for r in records)
    glean_e       = sum(r.get("glean_added_entities", 0) for r in records)
    glean_r       = sum(r.get("glean_added_rels", 0) for r in records)

    summary = {
        "total_chunks": len(records),
        "parse_errors": parse_errors,
        "gleaning_ok": gleaning_ok,
        "total_entities": total_e,
        "total_rels": total_r,
        "glean_added_entities": glean_e,
        "glean_added_rels": glean_r,
        "total_extract_calls": extract_calls,
        "total_gleaning_calls": glean_calls,
    }

    log.info("Extraction: %d chunks, %d errors, %d gleaning_ok",
             len(records), parse_errors, gleaning_ok)
    log.info("  Entities: %d total (+%d from gleaning)", total_e, glean_e)
    log.info("  Rels: %d total (+%d from gleaning)", total_r, glean_r)
    log.info("  LLM calls: %d extract + %d glean = %d total",
             extract_calls, glean_calls, extract_calls + glean_calls)

    if len(records) < 42:
        log.warning("Only %d/42 chunks complete. Proceeding with partial data.", len(records))

    return records, summary


def attach_texts(records: list[dict]) -> None:
    """Attach chunk text from canonical store for graph building."""
    import zvec
    col = zvec.open(str(ROOT / "data" / "cache" / "chunks_v1"))
    try:
        res = col.query(
            queries=[zvec.Query("text_vec", vector=[0.0]*768,
                                param=zvec.HnswQueryParam(ef=300))],
            topk=100,
        )
        text_map = {doc.id: (doc.fields or {}).get("text", "") for doc in res}
    finally:
        col.close()

    for r in records:
        if not r.get("text"):
            r["text"] = text_map.get(r.get("chunk_id", ""), "")


def build_graph_metrics(G) -> dict:
    import collections, networkx as nx

    comps   = list(nx.weakly_connected_components(G))
    largest = max((len(c) for c in comps), default=0)
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)

    cross_nodes = sum(1 for n, d in G.nodes(data=True)
                      if len(d.get("source_docs", set())) >= 2)
    cross_edges = 0
    for u, v, d in G.edges(data=True):
        ud = G.nodes[u].get("source_docs", set())
        vd = G.nodes[v].get("source_docs", set())
        if ud and vd and not ud.intersection(vd):
            cross_edges += 1

    usable: set = set()
    for u, v, d in G.edges(data=True):
        for ss in d.get("source_sentences", []):
            if ss.get("chunk_id"):
                usable.add(ss["chunk_id"])

    type_counts = collections.Counter()
    for u, v, d in G.edges(data=True):
        for t in d.get("types", []):
            type_counts[t] += 1

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(comps),
        "largest_component": largest,
        "isolated_nodes": isolated,
        "cross_doc_nodes": cross_nodes,
        "cross_doc_edges": cross_edges,
        "usable_chunks": len(usable),
        "usable_chunk_rate": round(len(usable) / 42, 4),
        "edge_type_top10": dict(type_counts.most_common(10)),
    }


def evaluate_all_pipelines(G, pr: dict, exp_dir: Path) -> dict:
    from graphrag_lite.retrieval.lightrag import (
        retrieve_local, retrieve_global, retrieve_hybrid, retrieve_mix)
    from graphrag_lite.retrieval.pipeline_a import retrieve_flat
    from graphrag_lite.retrieval.pipeline_b import retrieve_structural
    import graphrag_lite.retrieval.pipeline_b as pb

    orig_G, orig_PR, orig_PR_MAX = pb._G, pb._PR, pb._PR_MAX
    pb._G, pb._PR, pb._PR_MAX = G, pr, max(pr.values()) if pr else 1.0

    with open(DEV_SET, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]

    r_acc, m_acc, l_acc = ({p: [] for p in PIPELINES} for _ in range(3))
    errs = {p: 0 for p in PIPELINES}
    ug   = {p: set() for p in PIPELINES}
    pq   = []

    log.info("Evaluating %d queries across 6 pipelines...", len(queries))

    try:
        for q in queries:
            qid  = q["query_id"]
            text = q["query"]
            gold = q.get("gold_chunk_ids", [])
            row  = {"query_id": qid, "type": q.get("type"), "gold_chunk_ids": gold}

            for pl in PIPELINES:
                t0 = time.monotonic()
                ids, err = [], None
                try:
                    if pl == "pipeline_a":
                        ev = retrieve_flat(text); ids = [e.chunk_id for e in ev[:TOP_K]]
                    elif pl == "pipeline_b":
                        ev = retrieve_structural(text, top_k=TOP_K); ids = [e.chunk_id for e in ev]
                    elif pl == "lightrag_local":
                        ev = retrieve_local(text, G, pr, exp_dir, top_k=TOP_K)
                        ids = [e["chunk_id"] for e in ev]
                    elif pl == "lightrag_global":
                        ev = retrieve_global(text, G, exp_dir, top_k=TOP_K)
                        ids = [e["chunk_id"] for e in ev]
                    elif pl == "lightrag_hybrid":
                        ev = retrieve_hybrid(text, G, pr, exp_dir, top_k=TOP_K)
                        ids = [e["chunk_id"] for e in ev]
                    elif pl == "lightrag_mix":
                        ev = retrieve_mix(text, G, pr, exp_dir, top_k=TOP_K)
                        ids = [e["chunk_id"] for e in ev]
                except Exception as exc:
                    err = str(exc); errs[pl] += 1
                    log.warning("  %s %s ERROR: %s", qid, pl, err[:80])

                lat = round(time.monotonic() - t0, 3)
                rec = sum(1 for g in gold if g in ids) / len(gold) if gold else 1.0
                m   = next((1.0/i for i, c in enumerate(ids, 1) if c in set(gold)), 0.0)
                r_acc[pl].append(rec); m_acc[pl].append(m); l_acc[pl].append(lat)
                for g in gold:
                    if g in ids: ug[pl].add(g)
                row[pl] = {"ids": ids, "recall": round(rec, 4), "mrr": round(m, 4),
                           "latency_s": lat, "error": err}
            pq.append(row)
    finally:
        pb._G, pb._PR, pb._PR_MAX = orig_G, orig_PR, orig_PR_MAX

    n = len(queries)
    agg = {}
    for pl in PIPELINES:
        agg[pl] = {
            "avg_recall_at_k":       round(sum(r_acc[pl]) / n, 4),
            "mrr":                   round(sum(m_acc[pl]) / n, 4),
            "unique_gold_recovered": len(ug[pl]),
            "avg_latency_s":         round(sum(l_acc[pl]) / n, 4),
            "errors":                errs[pl],
        }

    return {"per_question": pq, "aggregates": agg}


def build_comparison_and_report(
    graph_metrics: dict,
    extraction_summary: dict,
    eval_result: dict,
) -> str:
    """Build 4-way comparison table and return the markdown report."""

    def load_agg(path: Path) -> dict:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return d.get("aggregates", {})
        return {}

    p1_agg = {}
    if PHASE1_JSON.exists():
        with open(PHASE1_JSON, encoding="utf-8") as f:
            p1_agg = json.load(f).get("aggregates", {})
    v2off_agg = load_agg(V2OFF_JSON)
    v2on_agg  = load_agg(V2ON_JSON)
    gl_agg    = eval_result["aggregates"]

    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]

    # Comparison JSON
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Genuine gleaning-ON (MAX_GLEANINGS=1) vs Phase 1, v2-OFF and v2-ON (stub filter).",
        "extraction_summary": extraction_summary,
        "graph_metrics": graph_metrics,
        "aggregates": {
            "phase1_gleaning_off": p1_agg,
            "v2_stub_off": v2off_agg,
            "v2_stub_on":  v2on_agg,
            "genuine_gleaning_on": gl_agg,
        },
    }
    with open(EXP_DIR / "genuine_gleaning_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Markdown report
    lines = [
        "# Genuine Gleaning-ON Experiment Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Model: `{MODEL}`  MAX_GLEANINGS=1  ",
        "Source: `data/experiments/qwen2_5_1_5b_genuine_gleaning_on/`",
        "",
        "---",
        "",
        "## Extraction summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Chunks completed | {extraction_summary['total_chunks']}/42 |",
        f"| Parse errors | {extraction_summary['parse_errors']} |",
        f"| Gleaning OK | {extraction_summary['gleaning_ok']} |",
        f"| Total entities | {extraction_summary['total_entities']} |",
        f"| Entities added by gleaning | {extraction_summary['glean_added_entities']} |",
        f"| Total relationships | {extraction_summary['total_rels']} |",
        f"| Rels added by gleaning | {extraction_summary['glean_added_rels']} |",
        f"| Extraction LLM calls | {extraction_summary['total_extract_calls']} |",
        f"| Gleaning LLM calls | {extraction_summary['total_gleaning_calls']} |",
        "",
        "---",
        "",
        "## Graph metrics",
        "",
        "| Metric | Phase 1 (0.5b) | v2-OFF (1.5b) | v2-ON (stub) | Genuine Gleaning |",
        "|---|---|---|---|---|",
        f"| Nodes | 72 | 209 | 224 | {graph_metrics['nodes']} |",
        f"| Edges | 30 | 112 | 157 | {graph_metrics['edges']} |",
        f"| Components | 45 | 103 | 75 | {graph_metrics['components']} |",
        f"| Largest component | 7 | 15 | 29 | {graph_metrics['largest_component']} |",
        f"| Isolated nodes | 29 | 75 | 43 | {graph_metrics['isolated_nodes']} |",
        f"| Cross-doc nodes | 0 | 9 | 9 | {graph_metrics['cross_doc_nodes']} |",
        f"| **Cross-doc edges** | **0** | **0** | **0** | **{graph_metrics['cross_doc_edges']}** |",
        f"| Usable chunks | 17/41 | 36/42 | 40/42 | {graph_metrics['usable_chunks']}/42 |",
        "",
        "---",
        "",
        "## 4-way retrieval comparison (Recall@6, top_k=6, 20 dev questions)",
        "",
        "| Pipeline | Phase 1 | v2-OFF | v2-ON | **Gleaning-ON** | Best |",
        "|---|---|---|---|---|---|",
    ]

    def r(agg, pl, key="avg_recall_at_k"):
        a = agg.get(pl, {})
        v = a.get(key)
        return f"{v:.4f}" if v is not None else "N/A"

    for pl in PIPELINES:
        v1  = r(p1_agg, pl)
        vOF = r(v2off_agg, pl)
        vON = r(v2on_agg, pl)
        vGL = r(gl_agg, pl)
        vals = [float(x) for x in [v1, vOF, vON, vGL] if x != "N/A"]
        best = max(vals) if vals else None
        best_s = f"**{best:.4f}**" if best is not None else "N/A"
        lines.append(f"| {pl} | {v1} | {vOF} | {vON} | {vGL} | {best_s} |")

    lines += [
        "",
        "### MRR",
        "",
        "| Pipeline | Phase 1 | v2-OFF | v2-ON | **Gleaning-ON** |",
        "|---|---|---|---|---|",
    ]
    for pl in PIPELINES:
        v1  = r(p1_agg, pl, "mrr")
        vOF = r(v2off_agg, pl, "mrr")
        vON = r(v2on_agg, pl, "mrr")
        vGL = r(gl_agg, pl, "mrr")
        lines.append(f"| {pl} | {v1} | {vOF} | {vON} | {vGL} |")

    lines += [
        "",
        "---",
        "",
        "## D18 diagnostic (cross-doc hard query)",
        "",
        "Gold chunk: `895ee8b439ed7d7b`",
        "",
    ]

    pq = eval_result["per_question"]
    d18 = next((r for r in pq if r["query_id"] == "D18"), None)
    if d18:
        lines.append("| Pipeline | Gold retrieved | Recall | Top-3 IDs |")
        lines.append("|---|---|---|---|")
        for pl in PIPELINES:
            r_ = d18.get(pl, {})
            ids = r_.get("ids", [])
            hit = "895ee8b439ed7d7b" in ids
            lines.append(f"| {pl} | {hit} | {r_.get('recall', 0):.2f} | {', '.join(ids[:3])} |")
    else:
        lines.append("D18 not found in results.")

    lines += [
        "",
        "---",
        "",
        "## D03, D17 special attention",
        "",
    ]

    for qid in ["D03", "D17"]:
        qrow = next((r for r in pq if r["query_id"] == qid), None)
        if qrow:
            lines.append(f"### {qid}")
            lines.append(f"Gold: {qrow['gold_chunk_ids']}")
            lines.append("")
            lines.append("| Pipeline | Recall | MRR |")
            lines.append("|---|---|---|")
            for pl in PIPELINES:
                r_ = qrow.get(pl, {})
                lines.append(f"| {pl} | {r_.get('recall', 0):.2f} | {r_.get('mrr', 0):.2f} |")
            lines.append("")

    lines += [
        "---",
        "",
        "## Naming clarification",
        "",
        "- **v2-OFF** and **v2-ON** in earlier reports refer to the **stub-endpoint filter**",
        "  (EXTRACTION_STUB_ENDPOINTS), not gleaning. Both arms used identical LLM responses.",
        "- **Genuine Gleaning-ON** = `MAX_GLEANINGS=1`, meaning a second LLM call per chunk",
        "  asking the model what it missed. This is the first true gleaning experiment.",
        "",
        "---",
        "",
        "## Known limitations",
        "",
        f"- Cross-doc edges: {graph_metrics['cross_doc_edges']} (expected 0 until a model extracts",
        "  entities shared across documents consistently).",
        "- D18 cross-doc paraphrase will not resolve until cross-doc edges exist.",
        "- qwen2.5:1.5b entity naming is inconsistent (e.g. 'LLM' vs 'Large Language Model')",
        "  which prevents alias merging across documents.",
    ]

    report = "\n".join(lines)
    with open(EXP_DIR / "genuine_gleaning_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collections", action="store_true")
    args = parser.parse_args()

    fh = logging.FileHandler(str(EXP_DIR / "evaluation.log"), encoding="utf-8",
                              mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=== run_gleaning_evaluation.py ===")
    t_total = time.monotonic()

    # 1. Check extraction
    records, extraction_summary = check_extraction_complete()
    attach_texts(records)

    # 2. Build graph
    import io, contextlib
    from graphrag_lite.graph.builder import build_graph
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(records)
    for line in buf.getvalue().splitlines():
        if line.strip(): log.info("  [builder] %s", line)

    graph_metrics = build_graph_metrics(G)
    with open(EXP_DIR / "genuine_gleaning_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**extraction_summary, **graph_metrics}, f, indent=2)
    log.info("Graph: %d nodes, %d edges, largest=%d, cross_doc_edges=%d",
             graph_metrics["nodes"], graph_metrics["edges"],
             graph_metrics["largest_component"], graph_metrics["cross_doc_edges"])

    # 3. Build collections
    from graphrag_lite.graph.ranking import rank_pagerank
    pr = rank_pagerank(G)

    if not args.skip_collections or not (EXP_DIR / "entities").exists():
        from run_model_experiment import build_experiment_collections
        log.info("Building Zvec collections...")
        build_experiment_collections(G, records, EXP_DIR, model=MODEL)
    else:
        log.info("Reusing existing collections")

    # 4. Evaluate
    eval_result = evaluate_all_pipelines(G, pr, EXP_DIR)
    with open(EXP_DIR / "genuine_gleaning_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    # 5. Comparison + report
    report = build_comparison_and_report(graph_metrics, extraction_summary, eval_result)

    # Print aggregate table
    agg = eval_result["aggregates"]
    print("\n" + "="*80)
    print("GENUINE GLEANING-ON RESULTS — qwen2.5:1.5b, top_k=6, 20 dev questions")
    print("="*80)
    fmt = "{:<22} {:>10} {:>8} {:>10} {:>12} {:>6}"
    print(fmt.format("Pipeline","Recall@K","MRR","UniqueGold","AvgLat(s)","Errors"))
    print("-"*70)
    for pl, a in agg.items():
        print(fmt.format(pl, f"{a['avg_recall_at_k']:.4f}", f"{a['mrr']:.4f}",
                         str(a["unique_gold_recovered"]), f"{a['avg_latency_s']:.3f}",
                         str(a["errors"])))

    total_s = round(time.monotonic() - t_total, 1)
    log.info("Complete. Total: %.1fs. Results in %s", total_s, EXP_DIR)
    print(f"\nReport: {EXP_DIR / 'genuine_gleaning_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
