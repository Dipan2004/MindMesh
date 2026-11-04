"""
embedding_ablation.py — Task 8.3: Embedding model ablation (complete).

Compares three embedding models over Pipeline B on the 20-question dev set:
  1. nomic-embed-text   (768-dim, canonical — Ollama)
  2. qwen3-embedding:0.6b  (1024-dim — Ollama)
  3. all-minilm:latest     (384-dim — Ollama)

Each model gets its OWN Zvec collection built from the SAME source data
(extractions.jsonl + graph). The canonical collections (entities_v1,
chunks_v1) are NEVER touched. Temporary collections live under
data/cache_embed_ablation/ and are built fresh each run.

TASKS.md spec requires re-embedding + re-indexing between runs. This script
does exactly that.

Usage:
    python eval/ablations/embedding_ablation.py

DoD: paste the 3-way comparison table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

EMBED_ABLATION_ROOT = _REPO_ROOT / "data" / "cache_embed_ablation"
EXTRACTIONS_PATH = _REPO_ROOT / "data" / "cache" / "extractions.jsonl"

# Ollama tag → collection-name suffix → expected dim
MODELS = [
    ("nomic-embed-text",      "nomic",  768),
    ("qwen3-embedding:0.6b",  "qwen3",  1024),
    ("all-minilm:latest",     "minilm", 384),
]


# ── per-model index builder ───────────────────────────────────────────────────

def _build_index_for_model(model_tag: str, collection_suffix: str, dim: int) -> Path:
    """
    Build entities + chunks Zvec collections for one embedding model.
    Returns the root directory containing the two collections.
    """
    import zvec
    from graphrag_lite.config import get_config
    from graphrag_lite.embeddings.embedder import embed_batch
    from graphrag_lite.graph.builder import build_graph, load_extractions
    from graphrag_lite.graph.ranking import rank_pagerank
    from graphrag_lite.vectorstore.zvec_client import (
        _to_doc_id, _entities_schema, _chunks_schema, reset_collection,
    )

    root = EMBED_ABLATION_ROOT / collection_suffix
    root.mkdir(parents=True, exist_ok=True)

    ents_path = str(root / "entities_v1")
    chunks_path = str(root / "chunks_v1")

    print(f"    [build_index] model={model_tag} dim={dim} root={root}")

    # Remove any stale RocksDB LOCK files before resetting collections.
    # A MemoryError or crash on a previous run can leave these behind,
    # preventing zvec.open() from succeeding.
    for col_path in [ents_path, chunks_path]:
        for lock in Path(col_path).rglob("LOCK"):
            lock.unlink(missing_ok=True)

    # Reset if exists so we rebuild fresh
    for p in [ents_path, chunks_path]:
        if Path(p).exists():
            reset_collection(p)

    # ── Load graph ─────────────────────────────────────────────────────────────
    records = load_extractions(str(EXTRACTIONS_PATH))
    G = build_graph(records)
    pr = rank_pagerank(G)

    # ── Prepare entity records ─────────────────────────────────────────────────
    entity_dicts = []
    for node, attrs in G.nodes(data=True):
        entity_dicts.append({
            "name": node,
            "type": attrs.get("type", "OTHER"),
            "descriptions": attrs.get("descriptions", []),
            "source_docs": attrs.get("source_docs", set()),
        })

    # ── Prepare chunk records (from extractions) ───────────────────────────────
    chunk_dicts = []
    seen_chunk_ids: set[str] = set()
    for rec in records:
        if "_parse_error" in rec:
            continue
        cid = rec.get("chunk_id", "")
        if not cid or cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        chunk_dicts.append({
            "chunk_id": cid,
            "doc_id": rec.get("doc_id", ""),
            "chunk_index": rec.get("chunk_index", 0),
            "text": rec.get("text", ""),
        })

    # ── Embed entities ─────────────────────────────────────────────────────────
    entity_texts = [
        (d.get("descriptions") or [{}])[0].get("text") or d["name"]
        for d in entity_dicts
    ]
    print(f"    [embed] {len(entity_texts)} entities ...", end="", flush=True)
    t0 = time.monotonic()
    entity_vecs = embed_batch(entity_texts, model=model_tag)
    print(f" {time.monotonic()-t0:.1f}s")

    # ── Embed chunks ───────────────────────────────────────────────────────────
    chunk_texts = [c["text"] for c in chunk_dicts]
    print(f"    [embed] {len(chunk_texts)} chunks ...", end="", flush=True)
    t0 = time.monotonic()
    chunk_vecs = embed_batch(chunk_texts, model=model_tag)
    print(f" {time.monotonic()-t0:.1f}s")

    # ── Create Zvec collections ────────────────────────────────────────────────
    # Override EMBEDDING_DIM so _open_or_create can create the collection.
    cfg = get_config()
    object.__setattr__(cfg, "EMBEDDING_DIM", dim)

    # entities
    ents_col = zvec.create_and_open(ents_path, _entities_schema(dim))
    (Path(ents_path) / "chunking_strategy.txt").write_text(cfg.CHUNKING_STRATEGY)
    import hashlib
    ent_docs = []
    used_ids: set[str] = set()
    for ent, vec in zip(entity_dicts, entity_vecs):
        name = ent["name"]
        descs = ent.get("descriptions", [])
        first_desc = descs[0]["text"] if descs else ""
        src_docs = ent.get("source_docs", set())
        primary = next(iter(src_docs)) if src_docs else ""
        doc_id = _to_doc_id(name)
        if doc_id in used_ids:
            digest = hashlib.sha1(name.encode()).hexdigest()[:8]
            doc_id = f"{doc_id}_{digest}"
        used_ids.add(doc_id)
        ent_docs.append(zvec.Doc(
            id=doc_id,
            vectors={"text_vec": list(vec)},
            fields={
                "name": name,
                "entity_type": ent.get("type", "OTHER"),
                "source_doc": primary,
                "pagerank_score": float(pr.get(name, 0.0)),
                "description": first_desc,
            },
        ))
    ents_col.insert(ent_docs)
    ents_col.flush()
    ents_col.close()
    print(f"    [index] entities: {len(ent_docs)} docs at {ents_path}")

    # chunks
    chunks_col = zvec.create_and_open(chunks_path, _chunks_schema(dim))
    (Path(chunks_path) / "chunking_strategy.txt").write_text(cfg.CHUNKING_STRATEGY)
    chunk_docs = [
        zvec.Doc(
            id=c["chunk_id"],
            vectors={"text_vec": list(vec)},
            fields={
                "chunk_id": c["chunk_id"],
                "source_doc": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        )
        for c, vec in zip(chunk_dicts, chunk_vecs)
    ]
    chunks_col.insert(chunk_docs)
    chunks_col.flush()
    chunks_col.close()
    print(f"    [index] chunks: {len(chunk_docs)} docs at {chunks_path}")

    return root


# ── per-model eval run ────────────────────────────────────────────────────────

def _eval_with_model(model_tag: str, collection_root: Path, out_path: str) -> dict:
    """
    Run the Task 8.2 harness against the isolated collection for this model.
    Sets ZVEC_ROOT + EMBEDDING_MODEL env vars, then calls run_eval.
    Resets pipeline_b singleton between queries to avoid RocksDB lock contention.
    """
    from run_accuracy_eval import main as run_eval

    # Point ZVEC_ROOT at the ablation collection and set the embedding model.
    os.environ["ZVEC_ROOT"] = str(collection_root)
    os.environ["EMBEDDING_MODEL"] = model_tag

    # Reset the config and pipeline_b singletons so they pick up the new env.
    from graphrag_lite.config import get_config
    import graphrag_lite.retrieval.pipeline_b as pb
    import graphrag_lite.retrieval.pipeline_a  # warm import
    get_config.cache_clear()
    pb._G = None; pb._PR = None; pb._PR_MAX = None

    # Also reset EMBEDDING_DIM in config so the new model's dim is picked up
    try:
        cfg = get_config()
        object.__setattr__(cfg, "EMBEDDING_DIM", 0)
    except Exception:
        pass

    try:
        run_eval(["--pipeline", "b", "--out", out_path])
    finally:
        # Restore canonical env vars and reset singletons
        os.environ.pop("ZVEC_ROOT", None)
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
        get_config.cache_clear()
        pb._G = None; pb._PR = None; pb._PR_MAX = None
        # Restore canonical EMBEDDING_DIM
        try:
            cfg2 = get_config()
            object.__setattr__(cfg2, "EMBEDDING_DIM", 0)
        except Exception:
            pass

    with open(out_path) as f:
        return json.load(f)


# ── main ──────────────────────────────────────────────────────────────────────

def run_ablation() -> None:
    from run_accuracy_eval import RESULTS_DIR

    print("=" * 80)
    print("EMBEDDING MODEL ABLATION (Task 8.3)")
    print("=" * 80)
    print(f"Models: {', '.join(m[0] for m in MODELS)}")
    print(f"Canonical collections: UNTOUCHED (read-only)")
    print(f"Ablation collections: {EMBED_ABLATION_ROOT}\n")

    EMBED_ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    results: dict[str, dict] = {}
    build_times: dict[str, float] = {}

    for model_tag, suffix, dim in MODELS:
        print(f"\n{'='*60}")
        print(f"  Model: {model_tag}  (dim={dim})")
        print(f"{'='*60}")

        # Build index
        t_build = time.monotonic()
        col_root = _build_index_for_model(model_tag, suffix, dim)
        build_times[model_tag] = round(time.monotonic() - t_build, 1)
        print(f"  Index built in {build_times[model_tag]}s")

        # Run eval
        out_path = str(RESULTS_DIR / f"embedding_{suffix}_{ts}.json")
        print(f"  Running eval ...")
        data = _eval_with_model(model_tag, col_root, out_path)
        results[model_tag] = data
        print(f"  Saved to {out_path}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("EMBEDDING ABLATION COMPARISON TABLE (Pipeline B, dev set)")
    print("=" * 90)
    model_tags = [m[0] for m in MODELS]
    fmt = "{:<32}" + " {:>14}" * len(model_tags)
    labels = [f"{t}\n(dim={d})" for t, _, d in MODELS]
    print(fmt.format("Metric", *[f"{t} ({d}d)" for t, _, d in MODELS]))
    print("-" * (32 + 14 * len(model_tags)))

    metrics = [
        ("Recall@K", "avg_recall_at_k"),
        ("MRR", "mrr"),
        ("% Grounded", "pct_grounded"),
        ("Term coverage (avg)", "avg_term_coverage"),
        ("% No-answer correct", "pct_no_answer_correct"),
        ("Avg evidence tokens", "avg_evidence_tokens"),
        ("Avg retrieval latency (s)", "avg_retrieval_latency_s"),
        ("Avg generation latency (s)", "avg_generation_latency_s"),
        ("Avg total latency (s)", "avg_total_latency_s"),
    ]
    for label, key in metrics:
        vals = []
        for model_tag, _, _ in MODELS:
            agg = results.get(model_tag, {}).get("aggregates", {}).get("b", {})
            v = agg.get(key, float("nan"))
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print(fmt.format(label, *vals))

    print("-" * (32 + 14 * len(model_tags)))
    build_vals = [f"{build_times.get(t, 0):.1f}s" for t, _, _ in MODELS]
    print(fmt.format("Index build time", *build_vals))
    print("=" * 90)

    # Save combined results
    ablation_path = RESULTS_DIR / f"embedding_ablation_{ts}.json"
    with open(ablation_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "8.3",
            "models": [{"tag": t, "suffix": s, "dim": d} for t, s, d in MODELS],
            "build_times": build_times,
            "results": results,
            "aggregates": {
                tag: results.get(tag, {}).get("aggregates", {}).get("b", {})
                for tag, _, _ in MODELS
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to {ablation_path}")


if __name__ == "__main__":
    run_ablation()
