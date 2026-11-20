"""
eval/hotpotqa/prepare_sample.py -- build the HotpotQA 20-question shared candidate corpus.

    python eval/hotpotqa/prepare_sample.py            # 10 bridge + 10 comparison, seed 42

Reads the official dev-distractor split from data/benchmarks/hotpotqa/raw/
(official JSON, or the HotpotQA organisation's HF parquet mirror) and writes
an isolated benchmark directory:

    data/benchmarks/hotpotqa/dev_distractor_20q_seed42/
        benchmark_config.json   sample_ids.json   questions.json
        corpus.jsonl            supporting_fact_map.json   corpus_stats.json

Refuses to overwrite an existing benchmark directory whose sample differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mindmesh.benchmarks.hotpotqa.dataset import (  # noqa: E402
    build_corpus, load_dev_distractor, map_supporting_facts, sample_questions,
    sample_questions_nested, write_benchmark,
)

RAW_DIR = ROOT / "data" / "benchmarks" / "hotpotqa" / "raw"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bridge", type=int, default=10)
    ap.add_argument("--n-comparison", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--draw-per-type", type=int, default=None,
                    help="nested sampling: draw this many per type once, keep the first n of each "
                         "(smaller samples with the same seed are prefix-subsets of larger ones)")
    args = ap.parse_args(argv)

    n = args.n_bridge + args.n_comparison
    out = args.out or ROOT / "data" / "benchmarks" / "hotpotqa" / f"dev_distractor_{n}q_seed{args.seed}"

    records, prov = load_dev_distractor(RAW_DIR)
    if args.draw_per_type:
        sample = sample_questions_nested(records, args.draw_per_type, args.n_bridge,
                                         args.n_comparison, args.seed)
        sampling_desc = (f"nested: random.Random(seed).sample(pool, {args.draw_per_type}) per type "
                         f"(bridge drawn first, then comparison, one RNG; pools sorted by _id); "
                         f"keep the first n_bridge / n_comparison of each draw")
    else:
        sample = sample_questions(records, args.n_bridge, args.n_comparison, args.seed)
        sampling_desc = "random.Random(seed).sample over records of each type sorted by _id"
    paras, stats = build_corpus(sample)
    sp_map = map_supporting_facts(sample, paras)

    unresolved = [(qid, f) for qid, fs in sp_map.items() for f in fs if not f["chunk_id"] or f["sentence"] is None]
    stats["supporting_facts_total"] = sum(len(fs) for fs in sp_map.values())
    stats["supporting_facts_unresolved"] = len(unresolved)

    config = {
        "benchmark": "HotpotQA",
        "split": "dev",
        "setting": "distractor",
        "corpus_scope": f"{n}-question shared candidate corpus (union of the sampled questions' "
                        f"supplied gold + distractor paragraphs). NOT the official FullWiki setting.",
        "n_questions": n,
        "n_bridge": args.n_bridge,
        "n_comparison": args.n_comparison,
        "seed": args.seed,
        "sampling": sampling_desc,
        "draw_per_type": args.draw_per_type,
        "dataset_provenance": prov,
        "id_scheme": {
            "doc_id": "hp_ + sha1(json([title, sentences]))[:16]",
            "chunk_id": "sha1(doc_id + ':0')[:16]  (one chunk per paragraph)",
        },
    }

    ids_path = out / "sample_ids.json"
    if ids_path.exists():
        prev = json.loads(ids_path.read_text(encoding="utf-8"))
        if prev["ids"] != [q["_id"] for q in sample]:
            print(f"[REFUSED] {out} already holds a different sample. Use --out.", file=sys.stderr)
            return 1
        print(f"[OK] {out} already holds this exact sample; rewriting derived files only.")

    write_benchmark(out, sample, paras, sp_map, stats, config)
    print(f"Benchmark written to {out}")
    print(json.dumps(stats, indent=2))
    types = {}
    for q in sample:
        types[q["type"]] = types.get(q["type"], 0) + 1
    print("Question types:", types)
    if unresolved:
        print(f"[ERROR] {len(unresolved)} supporting facts could not be mapped: {unresolved[:3]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
