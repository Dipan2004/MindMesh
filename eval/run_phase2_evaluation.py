"""
run_phase2_evaluation.py — Phase 2 evaluation using existing v2 extraction data.

Uses data from the October 2026 v2 build:
  - extractions_v2_off.jsonl: qwen2.5:1.5b, gleaning OFF, stub filter OFF (245 ent / 118 rel)
  - extractions_v2_on.jsonl:  qwen2.5:1.5b, gleaning OFF, stub filter ON  (263 ent / 163 rel)

NOTE on naming: The "_on" arm enables the conditional STUB ENDPOINT FILTER
(EXTRACTION_STUB_ENDPOINTS=True), not gleaning. Both arms use the same LLM
responses (no new calls; this was measured in DOCUMENT.md Item 13).

The actual gleaning-ON extraction has not been run yet (it requires ~2-4 hours
of LLM inference for 42 chunks x 2 calls each on qwen2.5:1.5b).

This script:
  1. Builds isolated graphs from both v2 arms
  2. Builds isolated Zvec collections for both
  3. Evaluates all 6 pipelines on both
  4. Reports a 3-way comparison: Phase 1 (canonical) vs v2-OFF vs v2-ON

Usage:
    python eval/run_phase2_evaluation.py
    python eval/run_phase2_evaluation.py --skip-collections  # reuse if already built
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

EXP_BASE   = ROOT / "data" / "experiments"
PHASE1_JSON = EXP_BASE / "qwen2_5_1_5b_validate" / "validation_20261109_223804.json"
DEV_SET    = ROOT / "eval" / "gold_dev_set.json"
TOP_K      = 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase2_eval")


def load_extractions(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "_parse_error" not in r:
                    records.append(r)
    log.info("Loaded %d clean records from %s", len(records), jsonl_path.name)
    return records


def attach_texts(records: list[dict], chunks: list[dict]) -> None:
    """Attach chunk text from canonical store to extraction records."""
    text_map = {c["chunk_id"]: c["text"] for c in chunks}
    for r in records:
        r["text"] = text_map.get(r.get("chunk_id", ""), "")


def read_canonical_chunks() -> list[dict]:
    import zvec
    log.info("Reading 42 canonical chunks (read-only)...")
    col = zvec.open(str(ROOT / "data" / "cache" / "chunks_v1"))
    try:
        res = col.query(
            queries=[zvec.Query("text_vec", vector=[0.0]*768,
                                param=zvec.HnswQueryParam(ef=300))],
            topk=100,
        )
        chunks = []
        for doc in res:
            f = doc.fields or {}
            src = f.get("source_doc", "")
            chunks.append({"chunk_id": doc.id, "doc_id": src, "source_doc": src,
                           "chunk_index": int(f.get("chunk_index", 0)),
                           "text": f.get("text", "")})
    finally:
        col.close()
    chunks.sort(key=lambda c: (c["doc_id"], c["chunk_index"]))
    return chunks


def build_graph_from_records(records: list[dict]):
    import io, contextlib, collections, networkx as nx
    from graphrag_lite.graph.builder import build_graph

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(records)
    for line in buf.getvalue().splitlines():
        if line.strip():
            log.info("  [builder] %s", line)

    comps     = list(nx.weakly_connected_components(G))
    largest   = max((len(c) for c in comps), default=0)
    isolated  = sum(1 for n in G.nodes() if G.degree(n) == 0)
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

    metrics = {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "components": len(comps), "largest_component": largest,
        "isolated_nodes": isolated, "cross_doc_nodes": cross_nodes,
        "cross_doc_edges": cross_edges,
        "usable_chunks": len(usable),
        "usable_chunk_rate": round(len(usable) / 42, 4),
        "edge_type_top5": dict(type_counts.most_common(5)),
    }
    log.info("Graph: %d nodes, %d edges, %d components, largest=%d, cross_doc_edges=%d",
             metrics["nodes"], metrics["edges"], metrics["components"],
             metrics["largest_component"], cross_edges)
    return G, metrics


def build_collections_for_arm(G, records: list[dict], exp_dir: Path) -> dict:
    from run_model_experiment import build_experiment_collections
    log.info("Building collections in %s...", exp_dir)
    meta = build_experiment_collections(G, records, exp_dir, model="qwen2.5:1.5b")
    log.info("  entities=%d rels=%d chunks=%d",
             meta["entity_count"], meta["relationship_count"], meta["chunk_count"])
    return meta


def _recall(ids, gold):
    if not gold: return 1.0
    return sum(1 for g in gold if g in ids) / len(gold)

def _mrr(ids, gold):
    gs = set(gold)
    for i, c in enumerate(ids, 1):
        if c in gs: return 1.0 / i
    return 0.0

def _prec(ids, gold):
    if not ids: return 0.0
    return sum(1 for c in ids if c in gold) / len(ids)


def evaluate_arm(G, pr: dict, exp_dir: Path, arm_name: str) -> dict:
    from graphrag_lite.retrieval.lightrag import (
        retrieve_local, retrieve_global, retrieve_hybrid, retrieve_mix)
    from graphrag_lite.retrieval.pipeline_a import retrieve_flat
    from graphrag_lite.retrieval.pipeline_b import retrieve_structural

    # Reset pipeline_b graph singleton so it uses the right graph for this arm
    import graphrag_lite.retrieval.pipeline_b as pb
    orig_G, orig_PR, orig_PR_MAX = pb._G, pb._PR, pb._PR_MAX
    pb._G, pb._PR, pb._PR_MAX = G, pr, max(pr.values()) if pr else 1.0

    with open(DEV_SET, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]

    r_acc = {p: [] for p in PIPELINES}
    m_acc = {p: [] for p in PIPELINES}
    l_acc = {p: [] for p in PIPELINES}
    errs  = {p: 0  for p in PIPELINES}
    ug    = {p: set() for p in PIPELINES}
    pq    = []

    log.info("Evaluating arm=%s on %d queries...", arm_name, len(queries))

    try:
        for q in queries:
            qid   = q["query_id"]
            text  = q["query"]
            gold  = q.get("gold_chunk_ids", [])
            row   = {"query_id": qid, "type": q.get("type"), "gold_chunk_ids": gold}

            for pl in PIPELINES:
                t0  = time.monotonic()
                ids = []
                err = None
                try:
                    if pl == "pipeline_a":
                        ev = retrieve_flat(text)
                        ids = [e.chunk_id for e in ev[:TOP_K]]
                    elif pl == "pipeline_b":
                        ev = retrieve_structural(text, top_k=TOP_K)
                        ids = [e.chunk_id for e in ev]
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
                rec = _recall(ids, gold)
                m   = _mrr(ids, gold)
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
    return {"arm": arm_name, "per_question": pq, "aggregates": agg}


def print_arm_table(arm_name: str, agg: dict) -> None:
    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]
    print(f"\n{'='*80}")
    print(f"ARM: {arm_name}  (top_k={TOP_K}, 20 dev questions)")
    print(f"{'='*80}")
    fmt = "{:<22} {:>10} {:>8} {:>10} {:>12} {:>6}"
    print(fmt.format("Pipeline","Recall@K","MRR","UniqueGold","AvgLat(s)","Errors"))
    print("-"*70)
    for pl in PIPELINES:
        a = agg.get(pl, {})
        print(fmt.format(pl, f"{a.get('avg_recall_at_k',0):.4f}",
                         f"{a.get('mrr',0):.4f}",
                         str(a.get("unique_gold_recovered","?")),
                         f"{a.get('avg_latency_s',0):.3f}",
                         str(a.get("errors",0))))


def print_comparison(p1_agg: dict, off_agg: dict, on_agg: dict) -> None:
    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]
    print(f"\n{'='*110}")
    print("3-WAY COMPARISON: Phase 1 canonical (v1) vs v2-OFF (stub filter off) vs v2-ON (stub filter on)")
    print("All: qwen2.5:1.5b extraction, top_k=6, 20 dev questions")
    print(f"{'='*110}")
    fmt = "{:<22} {:>9} {:>9} {:>9}  {:>7} {:>7} {:>7}  {:>10} {:>10} {:>10}"
    print(fmt.format("Pipeline",
                     "R@K(v1)","R@K(OFF)","R@K(ON)",
                     "MRR(v1)","MRR(OFF)","MRR(ON)",
                     "Gold(v1)","Gold(OFF)","Gold(ON)"))
    print("-"*110)
    for pl in PIPELINES:
        a1  = p1_agg.get(pl, {})
        aOF = off_agg.get(pl, {})
        aON = on_agg.get(pl, {})
        r1  = a1.get("avg_recall_at_k", a1.get("recall", 0))
        rOF = aOF.get("avg_recall_at_k", 0)
        rON = aON.get("avg_recall_at_k", 0)
        m1  = a1.get("mrr", 0)
        mOF = aOF.get("mrr", 0)
        mON = aON.get("mrr", 0)
        g1  = a1.get("unique_gold_chunks_recovered", a1.get("unique_gold_recovered", "?"))
        gOF = aOF.get("unique_gold_recovered", "?")
        gON = aON.get("unique_gold_recovered", "?")
        print(fmt.format(pl,
                         f"{r1:.4f}", f"{rOF:.4f}", f"{rON:.4f}",
                         f"{m1:.4f}", f"{mOF:.4f}", f"{mON:.4f}",
                         str(g1), str(gOF), str(gON)))
    print("="*110)


def print_per_question(off_result: dict, on_result: dict) -> None:
    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]
    off_pq = {r["query_id"]: r for r in off_result["per_question"]}
    on_pq  = {r["query_id"]: r for r in on_result["per_question"]}

    print("\nPer-question recall — v2-OFF / v2-ON (A / B / Local / Global / Hybrid / Mix):")
    qids = [r["query_id"] for r in off_result["per_question"]]
    for qid in qids:
        ro = off_pq.get(qid, {})
        rn = on_pq.get(qid, {})
        gold = ro.get("gold_chunk_ids", [])
        off_vals = " / ".join(f"{ro.get(pl,{}).get('recall',0):.2f}" for pl in PIPELINES)
        on_vals  = " / ".join(f"{rn.get(pl,{}).get('recall',0):.2f}" for pl in PIPELINES)
        print(f"  {qid}  gold={len(gold)}")
        print(f"    OFF: {off_vals}")
        print(f"    ON:  {on_vals}")


def d18_diagnostic(off_result: dict, on_result: dict, p1_result: dict = None) -> None:
    PIPELINES = ["pipeline_a", "pipeline_b",
                 "lightrag_local", "lightrag_global",
                 "lightrag_hybrid", "lightrag_mix"]
    GOLD_D18 = "895ee8b439ed7d7b"
    print("\nD18 diagnostic — gold: 895ee8b439ed7d7b")

    for label, result in [("v2-OFF", off_result), ("v2-ON", on_result)]:
        pq = {r["query_id"]: r for r in result["per_question"]}
        d18 = pq.get("D18", {})
        print(f"  {label}:")
        for pl in PIPELINES:
            r = d18.get(pl, {})
            hit = GOLD_D18 in r.get("ids", [])
            print(f"    {pl:<22}  hit={hit}  recall={r.get('recall',0):.2f}  ids={r.get('ids',[])[:3]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-collections", action="store_true")
    args = parser.parse_args()

    t_total = time.monotonic()

    # Directories for the two arms
    exp_off = EXP_BASE / "qwen2_5_1_5b_v2_off"
    exp_on  = EXP_BASE / "qwen2_5_1_5b_v2_on"
    exp_off.mkdir(parents=True, exist_ok=True)
    exp_on.mkdir(parents=True, exist_ok=True)

    # Setup logging
    fh = logging.FileHandler(str(EXP_BASE / "phase2_eval.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=== PHASE 2 EVALUATION ===")
    log.info("v2-OFF: data/cache_v2/extractions_v2_off.jsonl (245 ent / 118 rel, stub filter OFF)")
    log.info("v2-ON:  data/cache_v2/extractions_v2_on.jsonl  (263 ent / 163 rel, stub filter ON)")
    log.info("NOTE: These are SAME LLM responses, different post-processing (not gleaning)")

    # Read canonical chunks for text attachment
    canonical_chunks = read_canonical_chunks()

    # Load both arms
    records_off = load_extractions(ROOT / "data" / "cache_v2" / "extractions_v2_off.jsonl")
    records_on  = load_extractions(ROOT / "data" / "cache_v2" / "extractions_v2_on.jsonl")
    attach_texts(records_off, canonical_chunks)
    attach_texts(records_on,  canonical_chunks)

    # Build graphs
    log.info("Building v2-OFF graph...")
    G_off, metrics_off = build_graph_from_records(records_off)
    log.info("Building v2-ON graph...")
    G_on,  metrics_on  = build_graph_from_records(records_on)

    from graphrag_lite.graph.ranking import rank_pagerank
    pr_off = rank_pagerank(G_off)
    pr_on  = rank_pagerank(G_on)

    # Save graph metrics
    with open(exp_off / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_off, f, indent=2)
    with open(exp_on / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_on, f, indent=2)

    # Build collections
    if args.skip_collections and (exp_off / "entities").exists() and (exp_on / "entities").exists():
        log.info("Reusing existing collections")
    else:
        log.info("Building v2-OFF collections...")
        build_collections_for_arm(G_off, records_off, exp_off)
        log.info("Building v2-ON collections...")
        build_collections_for_arm(G_on, records_on, exp_on)

    # Evaluate both arms
    log.info("Evaluating v2-OFF arm...")
    off_result = evaluate_arm(G_off, pr_off, exp_off, "v2-OFF")
    with open(exp_off / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(off_result, f, indent=2, ensure_ascii=False)

    log.info("Evaluating v2-ON arm...")
    on_result = evaluate_arm(G_on, pr_on, exp_on, "v2-ON")
    with open(exp_on / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(on_result, f, indent=2, ensure_ascii=False)

    # Load Phase 1 aggregates
    p1_agg = {}
    if PHASE1_JSON.exists():
        with open(PHASE1_JSON, encoding="utf-8") as f:
            p1_data = json.load(f)
        p1_agg = p1_data.get("aggregates", {})

    # Print tables
    print_arm_table("v2-OFF (stub filter OFF, 209 nodes, 112 edges)", off_result["aggregates"])
    print_arm_table("v2-ON  (stub filter ON,  224 nodes, 157 edges)", on_result["aggregates"])
    print_comparison(p1_agg, off_result["aggregates"], on_result["aggregates"])
    print_per_question(off_result, on_result)
    d18_diagnostic(off_result, on_result)

    # Graph metrics summary
    print(f"\n{'='*80}")
    print("GRAPH METRICS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<28} {'Phase 1 (0.5b)':<18} {'v2-OFF (1.5b)':<18} {'v2-ON (1.5b+stub)':<18}")
    print("-"*82)
    m_p1 = {"nodes":72,"edges":30,"components":45,"largest_component":7,
             "isolated_nodes":29,"cross_doc_nodes":0,"cross_doc_edges":0,
             "usable_chunks":17,"usable_chunk_rate":0.415}
    for key in ["nodes","edges","components","largest_component",
                "isolated_nodes","cross_doc_nodes","cross_doc_edges",
                "usable_chunks","usable_chunk_rate"]:
        v1 = m_p1.get(key, "?")
        vO = metrics_off.get(key, "?")
        vN = metrics_on.get(key, "?")
        print(f"  {key:<26} {str(v1):<18} {str(vO):<18} {str(vN):<18}")

    # Save comparison report
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "v2-OFF = stub filter OFF; v2-ON = stub filter ON. Same LLM responses, no new gleaning.",
        "graph_metrics": {"v2_off": metrics_off, "v2_on": metrics_on},
        "aggregates": {"v2_off": off_result["aggregates"], "v2_on": on_result["aggregates"],
                       "phase1": p1_agg},
        "total_s": round(time.monotonic() - t_total, 1),
    }
    out_path = EXP_BASE / f"phase2_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    log.info("Total: %.1fs. Report: %s", comparison["total_s"], out_path)
    print(f"\nComparison saved to {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
