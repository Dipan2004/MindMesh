"""
eval/hotpotqa/simulate_query_focused_graph.py -- offline simulation of query-focused graph construction.

    python eval/hotpotqa/simulate_query_focused_graph.py [--bench-dir <20q pilot dir>] [--run-id <run>]

Extraction is per paragraph and context-free (the prompt holds only that paragraph),
so the graph a query-focused system would build from a paragraph set S equals the
full graph's records restricted to S. This script therefore uses an existing full
extraction (no LLM calls, no GPU) plus the stored Dense top-10 lists of a finished run.

Selections compared per question:
  A        all corpus paragraphs (current offline design)
  B_k      Dense top-k paragraphs (k = 5, 10)
  C_k      B_k + paragraphs whose TITLE matches (a) a phrase in the question or
           (b) an entity extracted from the B_k paragraphs  (one expansion hop)

Measured per selection: paragraphs to extract, unique paragraphs across all questions
with caching, entity/relationship volume, gold-paragraph coverage, and whether the two
gold paragraphs are connected in the induced graph (cross-document path within <= 3 hops).
Writes <bench-dir>/../query_focused_simulation_<tag>.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import networkx as nx  # noqa: E402

from graphrag_lite.graph.builder import _norm, build_graph  # noqa: E402

DEF_BENCH = ROOT / "data/benchmarks/hotpotqa/dev_distractor_20q_seed42"
DEF_RUN = "gen-mistral_latest__idx-mistral_latest__a3b14137"


def induced_graph(records: list[dict]):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_graph(records)


def gold_connected(G, recs_by_chunk: dict, gold: list[str], max_hops: int = 3) -> bool | None:
    """True if an entity node from gold paragraph 1 reaches one from gold paragraph 2 within max_hops
    (undirected), or the two share an entity node. None if a gold paragraph is not in the graph."""
    if len(gold) < 2:
        return None
    nodes = []
    for c in gold[:2]:
        names = set()
        for n, d in G.nodes(data=True):
            if any(m.get("chunk_id") == c for m in d.get("source_sentences", []) + d.get("descriptions", [])):
                names.add(n)
        if not names:
            return False
        nodes.append(names)
    if nodes[0] & nodes[1]:
        return True
    U = G.to_undirected(as_view=True)
    for s in nodes[0]:
        lengths = nx.single_source_shortest_path_length(U, s, cutoff=max_hops)
        if any(t in lengths for t in nodes[1]):
            return True
    return False


def title_matches(text_norm: str, titles: dict[str, str]) -> set[str]:
    out = set()
    for cid, t in titles.items():
        base = _norm(t.split(" (")[0])
        if len(base) >= 4 and base in text_norm:
            out.add(cid)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=DEF_BENCH)
    ap.add_argument("--run-id", default=DEF_RUN)
    ap.add_argument("--index", default="mistral_latest")
    ap.add_argument("--tag", default="pilot20_mistral")
    args = ap.parse_args(argv)
    B = args.bench_dir
    paras = {json.loads(l)["chunk_id"]: json.loads(l) for l in (B / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    recs = {json.loads(l)["chunk_id"]: json.loads(l) for l in (B / "index" / args.index / "extractions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    clean = {c: r for c, r in recs.items() if "_parse_error" not in r}
    qs = json.loads((B / "questions.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (B / "runs" / args.run_id / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dense = {r["qid"]: r["retrieved_chunk_ids"] for r in rows if r["pipeline"] == "pipeline_a"}
    titles = {c: p["title"] for c, p in paras.items()}

    G_full = induced_graph(list(clean.values()))
    sel_names = ["A_full", "B_5", "B_10", "C_5", "C_10"]
    per = {s: [] for s in sel_names}
    for q in qs:
        gold = q["gold_chunk_ids"]
        qn = _norm(q["question"])
        for s in sel_names:
            if s == "A_full":
                S = set(paras)
            else:
                k = int(s.split("_")[1])
                S = set(dense[q["_id"]][:k])
                if s.startswith("C"):
                    S |= title_matches(qn, titles)                       # question phrase -> title
                    ents = {_norm(e["name"]) for c in list(S) if c in clean for e in clean[c]["entities"]}
                    for cid, t in titles.items():                         # extracted entity -> title (1 hop)
                        if _norm(t) in ents or _norm(t.split(" (")[0]) in ents:
                            S.add(cid)
            R = [clean[c] for c in S if c in clean]
            G = G_full if s == "A_full" else induced_graph(R)
            per[s].append({
                "qid": q["_id"], "type": q["type"], "n_paragraphs": len(S), "selected": sorted(S),
                "entities": sum(len(r["entities"]) for r in R), "relationships": sum(len(r["relationships"]) for r in R),
                "gold_coverage": sum(c in S for c in gold) / len(gold), "all_gold": all(c in S for c in gold),
                "gold_connected": gold_connected(G, clean, [c for c in gold if c in S]) if all(c in S for c in gold) else False,
            })
    out = {"bench_dir": str(B), "n_questions": len(qs), "corpus_paragraphs": len(paras),
           "extraction_records_clean": len(clean), "note": "measured on existing extractions; no LLM calls", "selections": {}}
    for s in sel_names:
        P = per[s]
        uniq = set().union(*[set(p["selected"]) for p in P])
        conn_full = [p for p in per["A_full"]]
        out["selections"][s] = {
            "paragraphs_per_question_mean": round(st.fmean(p["n_paragraphs"] for p in P), 2),
            "unique_paragraphs_all_questions_with_cache": len(uniq),
            "share_of_corpus_extracted": round(len(uniq) / len(paras), 3),
            "entities_per_question_mean": round(st.fmean(p["entities"] for p in P), 1),
            "relationships_per_question_mean": round(st.fmean(p["relationships"] for p in P), 1),
            "gold_paragraph_coverage": round(st.fmean(p["gold_coverage"] for p in P), 4),
            "all_gold_paragraphs_selected": sum(p["all_gold"] for p in P),
            "gold_pair_connected_in_graph": sum(bool(p["gold_connected"]) for p in P),
            "by_type": {t: {"n": sum(p["type"] == t for p in P),
                            "gold_coverage": round(st.fmean(p["gold_coverage"] for p in P if p["type"] == t), 4),
                            "gold_pair_connected": sum(bool(p["gold_connected"]) for p in P if p["type"] == t)}
                        for t in ("bridge", "comparison")},
        }
    dst = B.parent / f"query_focused_simulation_{args.tag}.json"
    dst.write_text(json.dumps({**out, "per_question": per}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
