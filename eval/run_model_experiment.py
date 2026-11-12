"""
run_model_experiment.py — Universal multi-model extraction and evaluation experiment.

One script handles ALL models. Only --model changes between runs.

Usage:
    python eval/run_model_experiment.py --model qwen3:4b
    python eval/run_model_experiment.py --model qwen3.5:9b
    python eval/run_model_experiment.py --model qwen2.5:7b
    python eval/run_model_experiment.py --model qwen2.5:3b

Design guarantees (per research spec 2026-11-09):
  - Stage A: Preflight verification — model must exist locally; never downloads.
  - Stage B: Read existing 42 canonical chunks; never re-chunks.
  - Stage C: Fresh extraction with model-private cache; full gleaning + retry ladder.
  - Stage D: Build an isolated, model-private NetworkX graph.
  - Stage E: Build model-private Zvec collections for entities, relationships, chunks.
  - Stage F: LightRAG-style retrieval (local, global, hybrid, mix) + existing pipelines.
  - Stage G: Evaluate all pipelines against the 20 dev questions.
  - Stage H: Write per-experiment output directory; never overwrite previous runs.

Safety:
  - Never touches data/cache (canonical collection).
  - Never touches eval/gold_test_set_frozen.json.
  - Stops without downloading if model is unavailable.
  - Each experiment lives in data/experiments/<model_slug>/.
  - Re-running with the same model APPENDS a timestamp suffix rather than overwriting.

Model slug derivation:
  "qwen2.5:3b"  -> "qwen2_5_3b"
  "qwen3:4b"    -> "qwen3_4b"
  "qwen3.5:9b"  -> "qwen3_5_9b"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Logging (file + console) ──────────────────────────────────────────────────
_log = logging.getLogger("experiment")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


# ── Slug ──────────────────────────────────────────────────────────────────────

def model_to_slug(model: str) -> str:
    """Convert a model tag to a filesystem-safe slug.

    Examples:
        qwen2.5:3b   -> qwen2_5_3b
        qwen3:4b     -> qwen3_4b
        qwen3.5:9b   -> qwen3_5_9b
        nomic-embed-text -> nomic_embed_text
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", model).strip("_")


# ── Stage A — Preflight ───────────────────────────────────────────────────────

def preflight(model: str, exp_dir: Path) -> dict:
    """
    Verify all prerequisites before touching any data.

    Returns a metadata dict that is saved to exp_dir/model_metadata.json.
    Raises SystemExit if any check fails.
    """
    _log.info("=== Stage A: Preflight ===")

    metadata: dict[str, Any] = {
        "requested_model": model,
        "preflight_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # 1. Ollama availability
    _log.info("Checking Ollama availability...")
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            _log.error("ollama list failed: %s", r.stderr[:200])
            sys.exit(1)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        _log.error("Ollama not available: %s", exc)
        sys.exit(1)
    _log.info("Ollama available.")

    # 2. Model existence — parse from 'ollama list'
    # Format: NAME  ID  SIZE  MODIFIED
    available_models = []
    for line in r.stdout.strip().splitlines()[1:]:  # skip header
        parts = line.split()
        if parts:
            available_models.append(parts[0])
    _log.info("Installed models: %s", available_models)

    if model not in available_models:
        _log.error(
            "Model %r is NOT installed locally. Available: %s. "
            "Do NOT pull automatically. Install it manually and retry.",
            model, available_models,
        )
        sys.exit(1)
    _log.info("Model %r is available.", model)
    metadata["model_available"] = True
    metadata["all_installed_models"] = available_models

    # 3. Model metadata (digest, quantization, parameters)
    _log.info("Collecting model metadata for %r...", model)
    try:
        r2 = subprocess.run(["ollama", "show", model], capture_output=True, text=True, timeout=30)
        metadata["ollama_show"] = r2.stdout
        # Parse key fields from ollama show output
        for line in r2.stdout.splitlines():
            line = line.strip()
            if "parameters" in line.lower() and ":" in line:
                k, _, v = line.partition(":")
                metadata[f"model_{k.strip().replace(' ', '_').lower()}"] = v.strip()
            elif "quantization" in line.lower() and ":" in line:
                k, _, v = line.partition(":")
                metadata[f"model_{k.strip().replace(' ', '_').lower()}"] = v.strip()
            elif "context length" in line.lower() and ":" in line:
                k, _, v = line.partition(":")
                metadata[f"model_{k.strip().replace(' ', '_').lower()}"] = v.strip()
            elif "architecture" in line.lower() and ":" in line:
                k, _, v = line.partition(":")
                metadata[f"model_{k.strip().replace(' ', '_').lower()}"] = v.strip()
    except Exception as exc:
        _log.warning("Could not collect model metadata: %s", exc)

    # 4. System resources
    _log.info("Checking system resources...")
    try:
        import shutil
        disk = shutil.disk_usage(str(ROOT))
        metadata["disk_free_gb"] = round(disk.free / 1_073_741_824, 2)
        _log.info("Disk free: %.1f GB", metadata["disk_free_gb"])
    except Exception as exc:
        _log.warning("Could not check disk: %s", exc)

    # 5. Embedding model availability
    _log.info("Checking embedding model availability...")
    from graphrag_lite.config import get_config
    cfg = get_config()
    embedding_model = cfg.EMBEDDING_MODEL
    if not any(embedding_model.split(":")[0] in m for m in available_models):
        _log.error(
            "Embedding model %r not found in ollama list. Cannot proceed.", embedding_model
        )
        sys.exit(1)
    _log.info("Embedding model %r available.", embedding_model)
    metadata["embedding_model"] = embedding_model

    # 6. Canonical chunks available
    canonical_chunks_path = ROOT / "data" / "cache" / "chunks_v1"
    if not canonical_chunks_path.exists():
        _log.error("Canonical chunks_v1 not found at %s", canonical_chunks_path)
        sys.exit(1)
    _log.info("Canonical chunks_v1 found at %s", canonical_chunks_path)
    metadata["canonical_chunks_path"] = str(canonical_chunks_path)

    # 7. Dev evaluation set available
    dev_set_path = ROOT / "eval" / "gold_dev_set.json"
    if not dev_set_path.exists():
        _log.error("Development eval set not found at %s", dev_set_path)
        sys.exit(1)
    with open(dev_set_path, encoding="utf-8") as f:
        dev_data = json.load(f)
    n_dev = len(dev_data.get("queries", []))
    if n_dev != 20:
        _log.warning("Expected 20 dev questions, found %d", n_dev)
    _log.info("Dev evaluation set: %d questions", n_dev)
    metadata["dev_set_questions"] = n_dev

    # 8. Frozen test set must NOT be read
    frozen_path = ROOT / "eval" / "gold_test_set_frozen.json"
    metadata["frozen_test_set_exists"] = frozen_path.exists()
    _log.info("Frozen test set present (not accessed): %s", frozen_path.exists())

    _log.info("Stage A complete — all preflight checks passed.")
    metadata["preflight_passed"] = True
    return metadata


# ── Stage B — Read canonical chunks ──────────────────────────────────────────

def read_canonical_chunks() -> list[dict]:
    """
    Read all 42 canonical chunks from chunks_v1. Read-only.
    Returns list of dicts with chunk_id, source_doc, chunk_index, text.
    """
    _log.info("=== Stage B: Read canonical chunks ===")
    import zvec

    canonical_path = str(ROOT / "data" / "cache" / "chunks_v1")
    _log.info("Opening chunks_v1 at %s (read-only)", canonical_path)

    col = zvec.open(canonical_path)
    try:
        res = col.query(
            queries=[zvec.Query(
                "text_vec",
                vector=[0.0] * 768,
                param=zvec.HnswQueryParam(ef=300),
            )],
            topk=100,
        )
        chunks = []
        for doc in res:
            f = doc.fields or {}
            chunks.append({
                "chunk_id":    doc.id,
                "source_doc":  f.get("source_doc", ""),
                "chunk_index": int(f.get("chunk_index", 0)),
                "text":        f.get("text", ""),
            })
    finally:
        col.close()

    chunks.sort(key=lambda c: (c["source_doc"], c["chunk_index"]))
    unique_ids = {c["chunk_id"] for c in chunks}

    _log.info("Read %d chunks, %d unique IDs", len(chunks), len(unique_ids))

    if len(unique_ids) != 42:
        _log.warning(
            "Expected 42 unique canonical chunk IDs, found %d. "
            "Proceeding with what is available.",
            len(unique_ids),
        )
    else:
        _log.info("Verified: all 42 canonical chunk IDs present.")

    return chunks


# ── Stage C — Fresh extraction ────────────────────────────────────────────────

def run_extraction(
    model: str,
    chunks: list[dict],
    cache_dir: str,
    cfg_think: bool | None,
) -> list[dict]:
    """
    Extract entities/relationships from all chunks using the given model.

    Uses model-private cache (cache_dir) so no results bleed between models.
    Returns list of extraction records (same format as extractions.jsonl).
    """
    _log.info("=== Stage C: Fresh extraction with %r ===", model)

    import diskcache
    import graphrag_lite.cache.llm_cache as lc_mod
    from graphrag_lite.config import get_config
    from graphrag_lite.llm.ollama_client import OllamaClient
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    from graphrag_lite.ingestion.gleaning import glean

    cfg = get_config()

    # Build model-private cache — never touches data/cache
    os.makedirs(cache_dir, exist_ok=True)
    _log.info("Using private LLM cache at %s", cache_dir)
    old_cache = lc_mod._cache
    old_hits, old_misses = lc_mod._hits, lc_mod._misses
    lc_mod._cache = diskcache.Cache(cache_dir)
    lc_mod._hits = 0
    lc_mod._misses = 0

    # Build client with explicit model (NOT from cfg singleton)
    # For qwen3-family: think=False to suppress chain-of-thought consuming budget
    think = cfg_think
    if think is None:
        think = False if model.startswith("qwen3") else cfg.effective_think()

    client = OllamaClient(
        model=model,
        temperature=cfg.EXTRACTION_TEMPERATURE,
        seed=cfg.EXTRACTION_SEED,
        timeout=cfg.LLM_TIMEOUT_S,
        num_predict=cfg.LLM_NUM_PREDICT,
        num_ctx=cfg.LLM_NUM_CTX,
        think=think,
        frequency_penalty=cfg.LLM_FREQUENCY_PENALTY,
    )

    _log.info(
        "Extraction settings: temperature=%s, seed=%s, num_predict=%s, "
        "num_ctx=%s, frequency_penalty=%s, think=%s, max_gleanings=%s",
        cfg.EXTRACTION_TEMPERATURE, cfg.EXTRACTION_SEED,
        cfg.LLM_NUM_PREDICT, cfg.LLM_NUM_CTX,
        cfg.LLM_FREQUENCY_PENALTY, think, cfg.MAX_GLEANINGS,
    )
    _log.info("Using model: %r (client.model=%r)", model, client.model)

    results = []
    total_start = time.monotonic()
    parse_failures = 0
    retry_events = 0

    try:
        for i, chunk in enumerate(chunks):
            cid = chunk["chunk_id"]
            doc = chunk["source_doc"]
            t0 = time.monotonic()
            _log.info(
                "[%02d/%02d] chunk=%s doc=%s",
                i + 1, len(chunks), cid[:12], doc[:20],
            )

            # Capture warnings for parse failures and retries
            attempt_warnings: list[str] = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    extraction = extract_entities_relationships(chunk, client)
                    for w in caught:
                        attempt_warnings.append(str(w.message))
                        if "retrying" in str(w.message).lower():
                            retry_events += 1
                            _log.info("  RETRY: %s", str(w.message)[:120])

                # Gleaning pass
                if cfg.MAX_GLEANINGS > 0:
                    with warnings.catch_warnings(record=True) as caught_g:
                        warnings.simplefilter("always")
                        extraction = glean(chunk, extraction, client,
                                           max_gleanings=cfg.MAX_GLEANINGS)
                        for w in caught_g:
                            attempt_warnings.append(str(w.message))

                n_ent = len(extraction.get("entities", []))
                n_rel = len(extraction.get("relationships", []))
                dt = time.monotonic() - t0
                _log.info(
                    "  -> %d entities, %d relationships (%.1fs) cache: hits=%d misses=%d",
                    n_ent, n_rel, dt, lc_mod._hits, lc_mod._misses,
                )

                results.append({
                    "chunk_id":      cid,
                    "doc_id":        doc,
                    "chunk_index":   chunk["chunk_index"],
                    "text":          chunk["text"],
                    "entities":      extraction.get("entities", []),
                    "relationships": extraction.get("relationships", []),
                    "extraction_model": client.model,
                    "latency_s":     round(dt, 3),
                    "warnings":      attempt_warnings,
                })

            except Exception as exc:
                dt = time.monotonic() - t0
                parse_failures += 1
                _log.error(
                    "  PARSE ERROR chunk=%s: %s (%.1fs)",
                    cid[:12], str(exc)[:120], dt,
                )
                results.append({
                    "chunk_id":      cid,
                    "doc_id":        doc,
                    "chunk_index":   chunk["chunk_index"],
                    "text":          chunk["text"],
                    "entities":      [],
                    "relationships": [],
                    "extraction_model": client.model,
                    "latency_s":     round(dt, 3),
                    "_parse_error":  str(exc),
                    "warnings":      attempt_warnings,
                })

        total_dt = time.monotonic() - total_start
        final_hits = lc_mod._hits
        final_misses = lc_mod._misses
        _log.info(
            "Stage C complete: %d chunks, %d parse failures, %d retries, "
            "%.1fs total (%.1fs/chunk avg). Cache: %d hits, %d misses.",
            len(chunks), parse_failures, retry_events, total_dt,
            total_dt / max(len(chunks), 1), final_hits, final_misses,
        )
        return results

    finally:
        # Restore original cache state (do not close old_cache — it may be None
        # or may belong to a parallel component; just restore the reference)
        lc_mod._cache = old_cache
        lc_mod._hits = old_hits
        lc_mod._misses = old_misses


# ── Stage D — Build isolated graph ────────────────────────────────────────────

def build_experiment_graph(extractions: list[dict]) -> tuple[Any, dict]:
    """
    Build a NetworkX graph from extraction records and compute metrics.
    Returns (G, metrics_dict).
    """
    _log.info("=== Stage D: Build isolated graph ===")

    import contextlib
    import io
    import collections
    import networkx as nx
    from graphrag_lite.graph.builder import build_graph

    # Suppress builder's print output while capturing it to the log
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G = build_graph(extractions)
    for line in buf.getvalue().splitlines():
        if line.strip():
            _log.info("  [builder] %s", line)

    # Compute metrics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    components = list(nx.weakly_connected_components(G))
    n_components = len(components)
    largest_component = max((len(c) for c in components), default=0)
    isolated_nodes = sum(1 for n in G.nodes() if G.degree(n) == 0)

    # Cross-document nodes and edges
    cross_doc_nodes = 0
    for node, attrs in G.nodes(data=True):
        src_docs = attrs.get("source_docs", set())
        if len(src_docs) >= 2:
            cross_doc_nodes += 1

    cross_doc_edges = 0
    for u, v, data in G.edges(data=True):
        u_docs = G.nodes[u].get("source_docs", set())
        v_docs = G.nodes[v].get("source_docs", set())
        if u_docs and v_docs and not u_docs.intersection(v_docs):
            cross_doc_edges += 1

    # Stub nodes
    stub_nodes = sum(1 for n, d in G.nodes(data=True) if d.get("stub", False))

    # Duplicate detection: entities that appear in >1 source chunk
    multi_chunk_nodes = sum(
        1 for n, d in G.nodes(data=True)
        if len(d.get("descriptions", [])) > 1
    )

    # Provenance coverage: fraction of chunks that contributed ≥1 relationship
    source_chunks_with_rels: set[str] = set()
    for u, v, d in G.edges(data=True):
        for ss in d.get("source_sentences", []):
            if "chunk_id" in ss:
                source_chunks_with_rels.add(ss["chunk_id"])
    total_chunks = len({
        ss.get("chunk_id")
        for n, d in G.nodes(data=True)
        for ss in d.get("source_sentences", [])
        if ss.get("chunk_id")
    })

    # Edge type distribution
    type_counts: dict[str, int] = collections.Counter()
    for u, v, d in G.edges(data=True):
        for t in d.get("types", []):
            type_counts[t] += 1

    metrics = {
        "nodes": n_nodes,
        "edges": n_edges,
        "components": n_components,
        "largest_component": largest_component,
        "isolated_nodes": isolated_nodes,
        "cross_doc_nodes": cross_doc_nodes,
        "cross_doc_edges": cross_doc_edges,
        "stub_nodes": stub_nodes,
        "multi_chunk_nodes": multi_chunk_nodes,
        "chunks_with_relationships": len(source_chunks_with_rels),
        "total_source_chunks": total_chunks,
        "provenance_coverage": (
            round(len(source_chunks_with_rels) / total_chunks, 4)
            if total_chunks > 0 else 0.0
        ),
        "edge_type_distribution": dict(type_counts.most_common(20)),
    }

    _log.info(
        "Graph: nodes=%d edges=%d components=%d largest=%d isolated=%d "
        "cross_doc_nodes=%d cross_doc_edges=%d",
        n_nodes, n_edges, n_components, largest_component,
        isolated_nodes, cross_doc_nodes, cross_doc_edges,
    )

    return G, metrics


# ── Stage E — Build experimental Zvec collections ────────────────────────────

def build_experiment_collections(
    G: Any,
    extractions: list[dict],
    exp_dir: Path,
    model: str,
) -> dict:
    """
    Build model-private Zvec collections:
      - entities collection (from graph nodes)
      - relationships collection (from graph edges, LightRAG-style embedding)
      - chunks collection (mirrored from canonical, but in private storage)

    Returns a metadata dict with collection paths and counts.
    """
    _log.info("=== Stage E: Build experimental Zvec collections ===")

    import zvec
    import networkx as nx
    from graphrag_lite.config import get_config
    from graphrag_lite.embeddings.embedder import embed_batch
    from graphrag_lite.graph.ranking import rank_pagerank

    cfg = get_config()
    entities_dir = str(exp_dir / "entities")
    rels_dir     = str(exp_dir / "relationships")
    chunks_dir   = str(exp_dir / "chunks")

    # Compute PageRank for experimental graph
    pr = rank_pagerank(G)

    # ── Embed + insert entities ───────────────────────────────────────────────
    _log.info("Embedding and inserting entities...")
    entity_records = []
    for node_name, attrs in G.nodes(data=True):
        descs = attrs.get("descriptions", [])
        first_desc = descs[0]["text"] if descs and isinstance(descs[0], dict) else ""
        src_docs = attrs.get("source_docs", set())
        primary_doc = next(iter(sorted(src_docs)), "")
        # Text to embed: "name (TYPE): description" — same format as Pipeline B verbalization
        embed_text = f"{node_name} ({attrs.get('type', 'OTHER')}): {first_desc}"
        entity_records.append({
            "name": node_name,
            "type": attrs.get("type", "OTHER"),
            "source_doc": primary_doc,
            "source_docs": sorted(src_docs),
            "description": first_desc,
            "descriptions": descs,
            "source_sentences": attrs.get("source_sentences", []),
            "pagerank": float(pr.get(node_name, 0.0)),
            "_embed_text": embed_text,
        })

    ent_texts = [r["_embed_text"] for r in entity_records]
    _log.info("Embedding %d entities...", len(ent_texts))
    ent_vecs = embed_batch(ent_texts) if ent_texts else []
    dim = len(ent_vecs[0]) if ent_vecs else 768

    # Create entities collection
    import re as _re
    _DOC_ID_UNSAFE = _re.compile(r"[^A-Za-z0-9_.\-]")

    def to_doc_id(name: str, used: set) -> str:
        import hashlib
        doc_id = _DOC_ID_UNSAFE.sub("_", name.replace(" ", "_").replace("/", "_"))
        if len(doc_id) > 64:
            digest = hashlib.sha1(name.encode()).hexdigest()[:8]
            doc_id = doc_id[:55] + "_" + digest
        if doc_id in used:
            digest = hashlib.sha1(name.encode()).hexdigest()[:8]
            doc_id = doc_id[:55] + "_" + digest
        used.add(doc_id)
        return doc_id

    ent_schema = zvec.CollectionSchema(
        name="entities_exp",
        fields=[
            zvec.FieldSchema("name",          zvec.DataType.STRING),
            zvec.FieldSchema("entity_type",   zvec.DataType.STRING),
            zvec.FieldSchema("source_doc",    zvec.DataType.STRING),
            zvec.FieldSchema("pagerank_score",zvec.DataType.FLOAT),
            zvec.FieldSchema("description",   zvec.DataType.STRING,
                             index_param=zvec.FtsIndexParam()),
        ],
        vectors=[
            zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32, dimension=dim,
                              index_param=zvec.HnswIndexParam()),
        ],
    )

    import shutil as _shutil
    if Path(entities_dir).exists():
        _shutil.rmtree(entities_dir)
    ent_col = zvec.create_and_open(entities_dir, ent_schema)
    used_ids: set = set()
    ent_docs = [
        zvec.Doc(
            id=to_doc_id(r["name"], used_ids),
            vectors={"text_vec": vec},
            fields={
                "name": r["name"],
                "entity_type": r["type"],
                "source_doc": r["source_doc"],
                "pagerank_score": r["pagerank"],
                "description": r["description"],
            },
        )
        for r, vec in zip(entity_records, ent_vecs)
    ]
    if ent_docs:
        ent_col.insert(ent_docs)
        ent_col.flush()
    ent_col.close()
    _log.info("Entities collection: %d documents at %s", len(ent_docs), entities_dir)

    # ── Build + embed relationships (LightRAG-style) ──────────────────────────
    # LightRAG's relationship embedding text:
    #   combined_keywords + source entity + target entity + description
    # MindMesh adaptation: we don't have separate keyword extraction at this stage,
    # so we use: relationship_type + source + target + description
    # This follows the verified LightRAG design principle: embed the semantic
    # content (what connects what, and how) rather than just the description alone.
    _log.info("Building relationship records with LightRAG-style embedding text...")

    rel_records = []
    rel_id_counter = 0
    for u, v, edge_data in G.edges(data=True):
        types = edge_data.get("types", ["RELATED_TO"])
        rel_type = types[0] if types else "RELATED_TO"
        description = edge_data.get("description", "")
        source_sentences = edge_data.get("source_sentences", [])
        source_doc = edge_data.get("source_doc", "")

        # Collect all chunk IDs and doc IDs from source_sentences
        chunk_ids = list({ss.get("chunk_id", "") for ss in source_sentences if ss.get("chunk_id")})
        doc_ids   = list({ss.get("doc_id", "")   for ss in source_sentences if ss.get("doc_id")})

        # LightRAG-style embedding text: type + source + target + description
        # The relationship type encodes the "combined keywords" role in LightRAG,
        # since we have structured type labels rather than free keywords.
        embed_text = f"{rel_type} {u} {v} {description}".strip()

        rel_id = f"rel_{rel_id_counter:05d}"
        rel_id_counter += 1

        rel_records.append({
            "rel_id":       rel_id,
            "source_entity": u,
            "target_entity": v,
            "rel_type":     rel_type,
            "all_types":    types,
            "description":  description,
            "weight":       len(source_sentences),  # occurrence count as weight
            "chunk_ids":    chunk_ids,
            "doc_ids":      doc_ids,
            "source_doc":   source_doc,
            "_embed_text":  embed_text,
        })

    rel_texts = [r["_embed_text"] for r in rel_records]
    _log.info("Embedding %d relationships...", len(rel_texts))
    rel_vecs = embed_batch(rel_texts) if rel_texts else []

    rel_schema = zvec.CollectionSchema(
        name="rels_exp",
        fields=[
            zvec.FieldSchema("source_entity", zvec.DataType.STRING),
            zvec.FieldSchema("target_entity", zvec.DataType.STRING),
            zvec.FieldSchema("rel_type",      zvec.DataType.STRING),
            zvec.FieldSchema("description",   zvec.DataType.STRING,
                             index_param=zvec.FtsIndexParam()),
            zvec.FieldSchema("source_doc",    zvec.DataType.STRING),
            zvec.FieldSchema("weight",        zvec.DataType.INT32),
        ],
        vectors=[
            zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32, dimension=dim,
                              index_param=zvec.HnswIndexParam()),
        ],
    )

    if Path(rels_dir).exists():
        _shutil.rmtree(rels_dir)
    rel_col = zvec.create_and_open(rels_dir, rel_schema)
    rel_docs = [
        zvec.Doc(
            id=r["rel_id"],
            vectors={"text_vec": vec},
            fields={
                "source_entity": r["source_entity"],
                "target_entity": r["target_entity"],
                "rel_type":      r["rel_type"],
                "description":   r["description"],
                "source_doc":    r["source_doc"],
                "weight":        r["weight"],
            },
        )
        for r, vec in zip(rel_records, rel_vecs)
    ]
    if rel_docs:
        rel_col.insert(rel_docs)
        rel_col.flush()
    rel_col.close()
    _log.info("Relationships collection: %d documents at %s", len(rel_docs), rels_dir)

    # ── Mirror canonical chunks into private collection ───────────────────────
    # We re-embed so the collection is self-contained, but do NOT re-chunk.
    _log.info("Mirroring canonical chunks into private collection...")
    canonical_chunks: list[dict] = []
    for rec in extractions:
        canonical_chunks.append({
            "chunk_id":    rec["chunk_id"],
            "source_doc":  rec.get("doc_id", rec.get("source_doc", "")),
            "chunk_index": rec.get("chunk_index", 0),
            "text":        rec.get("text", ""),
        })
    # Deduplicate by chunk_id
    seen_cids: set = set()
    deduped_chunks = []
    for c in canonical_chunks:
        if c["chunk_id"] not in seen_cids:
            seen_cids.add(c["chunk_id"])
            deduped_chunks.append(c)

    chunk_texts = [c["text"] for c in deduped_chunks]
    _log.info("Embedding %d chunks...", len(chunk_texts))
    chunk_vecs = embed_batch(chunk_texts) if chunk_texts else []

    chunk_schema = zvec.CollectionSchema(
        name="chunks_exp",
        fields=[
            zvec.FieldSchema("chunk_id",    zvec.DataType.STRING),
            zvec.FieldSchema("source_doc",  zvec.DataType.STRING),
            zvec.FieldSchema("chunk_index", zvec.DataType.INT32),
            zvec.FieldSchema("text",        zvec.DataType.STRING,
                             index_param=zvec.FtsIndexParam()),
        ],
        vectors=[
            zvec.VectorSchema("text_vec", zvec.DataType.VECTOR_FP32, dimension=dim,
                              index_param=zvec.HnswIndexParam()),
        ],
    )

    if Path(chunks_dir).exists():
        _shutil.rmtree(chunks_dir)
    chunk_col = zvec.create_and_open(chunks_dir, chunk_schema)
    chunk_docs = [
        zvec.Doc(
            id=c["chunk_id"],
            vectors={"text_vec": vec},
            fields={
                "chunk_id":    c["chunk_id"],
                "source_doc":  c["source_doc"],
                "chunk_index": c["chunk_index"],
                "text":        c["text"],
            },
        )
        for c, vec in zip(deduped_chunks, chunk_vecs)
    ]
    if chunk_docs:
        chunk_col.insert(chunk_docs)
        chunk_col.flush()
    chunk_col.close()
    _log.info("Chunks collection: %d documents at %s", len(chunk_docs), chunks_dir)

    # Save relationship metadata for downstream use
    rels_meta_path = exp_dir / "relationship_records.json"
    rels_meta = [{k: v for k, v in r.items() if k != "_embed_text"} for r in rel_records]
    with open(rels_meta_path, "w", encoding="utf-8") as f:
        json.dump(rels_meta, f, indent=2, ensure_ascii=False, default=list)
    _log.info("Relationship metadata saved to %s", rels_meta_path)

    return {
        "entities_dir": entities_dir,
        "relationships_dir": rels_dir,
        "chunks_dir": chunks_dir,
        "entity_count": len(ent_docs),
        "relationship_count": len(rel_docs),
        "chunk_count": len(chunk_docs),
        "embedding_dim": dim,
        "relationship_embed_format": "rel_type + source + target + description",
    }


# ── Stage F — LightRAG-style retrieval ───────────────────────────────────────

def retrieve_all_pipelines(
    query: str,
    G: Any,
    pr: dict,
    exp_dir: Path,
    top_k: int = 6,
) -> dict:
    """
    Run all retrieval pipelines for a single query:
      - pipeline_a: canonical flat vector search (baseline, unchanged)
      - pipeline_b: canonical graph retrieval (baseline, unchanged)
      - lightrag_local:  entity → one-hop edge → provenance
      - lightrag_global: relationship vector → endpoint expansion → provenance
      - lightrag_hybrid: merge local + global
      - lightrag_mix:    hybrid + direct chunk search

    Returns dict of pipeline_name -> list of Evidence-like dicts.
    """
    from graphrag_lite.embeddings.embedder import embed_batch
    from graphrag_lite.retrieval.lightrag import (
        retrieve_local,
        retrieve_global,
        retrieve_hybrid,
        retrieve_mix,
    )

    results = {}

    # Pipeline A — canonical, read-only
    try:
        from graphrag_lite.retrieval.pipeline_a import retrieve_flat
        t0 = time.monotonic()
        ev_a = retrieve_flat(query)
        results["pipeline_a"] = {
            "evidence": [_ev_to_dict(e) for e in ev_a[:top_k]],
            "latency_s": round(time.monotonic() - t0, 3),
            "source": "pipeline_a",
        }
    except Exception as exc:
        _log.warning("Pipeline A failed for query: %s", exc)
        results["pipeline_a"] = {"evidence": [], "latency_s": 0.0, "error": str(exc)}

    # Pipeline B — canonical, read-only
    try:
        from graphrag_lite.retrieval.pipeline_b import retrieve_structural
        t0 = time.monotonic()
        ev_b = retrieve_structural(query, top_k=top_k)
        results["pipeline_b"] = {
            "evidence": [_ev_to_dict(e) for e in ev_b],
            "latency_s": round(time.monotonic() - t0, 3),
            "source": "pipeline_b",
        }
    except Exception as exc:
        _log.warning("Pipeline B failed for query: %s", exc)
        results["pipeline_b"] = {"evidence": [], "latency_s": 0.0, "error": str(exc)}

    # LightRAG Local
    try:
        t0 = time.monotonic()
        ev_local = retrieve_local(query, G, pr, exp_dir, top_k=top_k)
        results["lightrag_local"] = {
            "evidence": ev_local,
            "latency_s": round(time.monotonic() - t0, 3),
            "source": "lightrag_local",
        }
    except Exception as exc:
        _log.warning("LightRAG Local failed: %s", exc)
        results["lightrag_local"] = {"evidence": [], "latency_s": 0.0, "error": str(exc)}

    # LightRAG Global
    try:
        t0 = time.monotonic()
        ev_global = retrieve_global(query, G, exp_dir, top_k=top_k)
        results["lightrag_global"] = {
            "evidence": ev_global,
            "latency_s": round(time.monotonic() - t0, 3),
            "source": "lightrag_global",
        }
    except Exception as exc:
        _log.warning("LightRAG Global failed: %s", exc)
        results["lightrag_global"] = {"evidence": [], "latency_s": 0.0, "error": str(exc)}

    # LightRAG Hybrid
    try:
        t0 = time.monotonic()
        ev_hybrid = retrieve_hybrid(query, G, pr, exp_dir, top_k=top_k)
        results["lightrag_hybrid"] = {
            "evidence": ev_hybrid,
            "latency_s": round(time.monotonic() - t0, 3),
            "source": "lightrag_hybrid",
        }
    except Exception as exc:
        _log.warning("LightRAG Hybrid failed: %s", exc)
        results["lightrag_hybrid"] = {"evidence": [], "latency_s": 0.0, "error": str(exc)}

    # LightRAG Mix
    try:
        t0 = time.monotonic()
        ev_mix = retrieve_mix(query, G, pr, exp_dir, top_k=top_k)
        results["lightrag_mix"] = {
            "evidence": ev_mix,
            "latency_s": round(time.monotonic() - t0, 3),
            "source": "lightrag_mix",
        }
    except Exception as exc:
        _log.warning("LightRAG Mix failed: %s", exc)
        results["lightrag_mix"] = {"evidence": [], "latency_s": 0.0, "error": str(exc)}

    return results


def _ev_to_dict(e) -> dict:
    """Convert an Evidence dataclass to a plain dict."""
    return {
        "chunk_id":    e.chunk_id,
        "source_doc":  e.source_doc,
        "chunk_index": e.chunk_index,
        "text":        e.text,
        "score":       e.score,
        "source":      e.source,
    }


# ── Stage G — Evaluation ──────────────────────────────────────────────────────

def evaluate_all(
    G: Any,
    pr: dict,
    exp_dir: Path,
    model: str,
) -> dict:
    """
    Run the 20 dev questions through all pipelines and compute metrics.
    Returns a full evaluation dict.
    """
    _log.info("=== Stage G: Universal evaluation ===")

    dev_path = ROOT / "eval" / "gold_dev_set.json"
    with open(dev_path, encoding="utf-8") as f:
        dev_data = json.load(f)
    queries = dev_data["queries"]

    pipelines = [
        "pipeline_a",
        "pipeline_b",
        "lightrag_local",
        "lightrag_global",
        "lightrag_hybrid",
        "lightrag_mix",
    ]
    top_k = 6

    all_results: dict[str, list] = {pl: [] for pl in pipelines}
    runtime_by_pipeline: dict[str, list] = {pl: [] for pl in pipelines}

    for i, q in enumerate(queries, 1):
        qid = q["query_id"]
        query_text = q["query"]
        gold_ids = q.get("gold_chunk_ids", [])
        q_type = q.get("type", "single-chunk")

        _log.info("[%02d/%02d] %s (%s)", i, len(queries), qid, q_type)

        # Get retrieval results for all pipelines
        retrieval = retrieve_all_pipelines(query_text, G, pr, exp_dir, top_k=top_k)

        for pl in pipelines:
            pl_result = retrieval.get(pl, {})
            evidence = pl_result.get("evidence", [])
            latency = pl_result.get("latency_s", 0.0)
            error = pl_result.get("error")
            runtime_by_pipeline[pl].append(latency)

            ev_ids = [e.get("chunk_id", e.get("id", "")) for e in evidence]

            # Recall@K
            recall = _recall_at_k(ev_ids, gold_ids)
            # MRR
            gold_rank = _gold_rank(ev_ids, gold_ids)
            # Unique gold chunks recovered
            gold_hit_ids = [g for g in gold_ids if g in ev_ids]
            # Cross-document: evidence from >1 doc
            ev_docs = {e.get("source_doc", "") for e in evidence if e.get("source_doc")}
            cross_doc = len(ev_docs) > 1

            all_results[pl].append({
                "query_id":        qid,
                "type":            q_type,
                "pipeline":        pl,
                "evidence_ids":    ev_ids,
                "gold_chunk_ids":  gold_ids,
                "gold_hit_ids":    gold_hit_ids,
                "recall_at_k":     round(recall, 4),
                "gold_rank":       gold_rank,
                "mrr":             round(1.0 / gold_rank if gold_rank else 0.0, 4),
                "retrieved_count": len(evidence),
                "cross_doc_evidence": cross_doc,
                "retrieval_latency_s": latency,
                "error":           error,
            })

    # Compute aggregates per pipeline
    aggregates = {}
    for pl in pipelines:
        results = all_results[pl]
        n = len(results)
        if n == 0:
            aggregates[pl] = {}
            continue

        mrr_vals = [r["mrr"] for r in results]
        mrr = round(sum(mrr_vals) / n, 4)
        avg_recall = round(sum(r["recall_at_k"] for r in results) / n, 4)
        avg_latency = round(sum(r["retrieval_latency_s"] for r in results) / n, 4)
        unique_gold = len({cid for r in results for cid in r["gold_hit_ids"]})
        pct_cross_doc = round(sum(1 for r in results if r["cross_doc_evidence"]) / n, 4)

        aggregates[pl] = {
            "n":                n,
            "avg_recall_at_k":  avg_recall,
            "mrr":              mrr,
            "unique_gold_chunks_recovered": unique_gold,
            "pct_cross_doc_evidence": pct_cross_doc,
            "avg_retrieval_latency_s": avg_latency,
        }
        _log.info(
            "  %s: recall=%.3f, mrr=%.3f, unique_gold=%d, latency=%.3fs",
            pl, avg_recall, mrr, unique_gold, avg_latency,
        )

    # D18 diagnostic
    d18 = next((q for q in queries if q["query_id"] == "D18"), None)
    d18_result = None
    if d18:
        d18_retrieval = retrieve_all_pipelines(d18["query"], G, pr, exp_dir, top_k=top_k)
        d18_gold = d18.get("gold_chunk_ids", [])
        d18_result = {
            "query": d18["query"],
            "gold_chunk_ids": d18_gold,
        }
        for pl in pipelines:
            ev = d18_retrieval.get(pl, {}).get("evidence", [])
            ev_ids = [e.get("chunk_id", e.get("id", "")) for e in ev]
            d18_result[pl] = {
                "evidence_ids": ev_ids,
                "gold_retrieved": any(g in ev_ids for g in d18_gold),
                "latency_s": d18_retrieval.get(pl, {}).get("latency_s", 0.0),
            }
        _log.info("D18 diagnostic: %s", {pl: d18_result[pl]["gold_retrieved"] for pl in pipelines})

    return {
        "model": model,
        "per_question": all_results,
        "aggregates": aggregates,
        "d18_diagnostic": d18_result,
    }


def _recall_at_k(ev_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0
    hit = sum(1 for g in gold_ids if g in ev_ids)
    return hit / len(gold_ids)


def _gold_rank(ev_ids: list[str], gold_ids: list[str]) -> int | None:
    gold_set = set(gold_ids)
    for i, cid in enumerate(ev_ids, 1):
        if cid in gold_set:
            return i
    return None


# ── Stage H — Save outputs ────────────────────────────────────────────────────

def save_outputs(
    exp_dir: Path,
    model: str,
    config_snapshot: dict,
    model_metadata: dict,
    extractions: list[dict],
    G: Any,
    graph_metrics: dict,
    collection_meta: dict,
    evaluation: dict,
    runtime_metrics: dict,
) -> None:
    """Save all experiment outputs to exp_dir."""
    _log.info("=== Stage H: Save outputs to %s ===", exp_dir)
    import networkx as nx

    # config.json
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # model_metadata.json
    with open(exp_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)

    # extractions.jsonl
    ext_path = exp_dir / "extractions.jsonl"
    with open(ext_path, "w", encoding="utf-8") as f:
        for rec in extractions:
            # Don't save full text in JSONL (chunks are in the canonical store)
            out = {k: v for k, v in rec.items() if k != "text"}
            f.write(json.dumps(out, ensure_ascii=False, default=list) + "\n")
    _log.info("Saved %d extraction records to %s", len(extractions), ext_path)

    # graph.graphml
    graphml_path = exp_dir / "graph.graphml"
    try:
        # Convert sets to lists for GraphML serialization
        G_copy = G.copy()
        for node, attrs in G_copy.nodes(data=True):
            if "source_docs" in attrs and isinstance(attrs["source_docs"], set):
                G_copy.nodes[node]["source_docs"] = sorted(attrs["source_docs"])
            if "descriptions" in attrs:
                G_copy.nodes[node]["descriptions"] = json.dumps(attrs["descriptions"])
            if "source_sentences" in attrs:
                G_copy.nodes[node]["source_sentences"] = json.dumps(attrs["source_sentences"])
        nx.write_graphml(G_copy, str(graphml_path))
        _log.info("Saved graph to %s", graphml_path)
    except Exception as exc:
        _log.warning("Could not save GraphML: %s", exc)

    # graph_metrics.json
    with open(exp_dir / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump(graph_metrics, f, indent=2, ensure_ascii=False)

    # evaluation.json
    with open(exp_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    # runtime_metrics.json
    with open(exp_dir / "runtime_metrics.json", "w", encoding="utf-8") as f:
        json.dump(runtime_metrics, f, indent=2, ensure_ascii=False)

    _log.info("All outputs saved to %s", exp_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Universal multi-model extraction and evaluation experiment"
    )
    parser.add_argument(
        "--model", required=True,
        help="Ollama model tag (must be installed locally). E.g. qwen3:4b",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip extraction and load existing extractions.jsonl from exp_dir.",
    )
    parser.add_argument(
        "--skip-collections", action="store_true",
        help="Skip building Zvec collections (requires them to already exist).",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation (useful when only extraction is needed).",
    )
    parser.add_argument(
        "--think", default=None, choices=["on", "off", "auto"],
        help="Override think mode. Default: 'off' for qwen3-family, else from config.",
    )
    args = parser.parse_args(argv)

    model = args.model
    slug  = model_to_slug(model)

    # Determine experiment directory
    base_dir = ROOT / "data" / "experiments"
    exp_dir  = base_dir / slug
    # Never overwrite: if directory exists, append a timestamp suffix
    if exp_dir.exists() and not args.skip_extraction:
        ts = time.strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"{slug}_{ts}"
        print(f"[experiment] Previous run exists; using {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(exp_dir / "experiment.log")
    _log.info("Experiment: model=%r  slug=%r  dir=%s", model, slug, exp_dir)
    _log.info("Script: %s", __file__)

    # think override
    think_override: bool | None = None
    if args.think == "off":
        think_override = False
    elif args.think == "on":
        think_override = True
    # else None -> determined per model family in run_extraction

    # ── Config snapshot ───────────────────────────────────────────────────────
    from graphrag_lite.config import get_config
    cfg = get_config()
    config_snapshot = {
        "model": model,
        "extraction_temperature": cfg.EXTRACTION_TEMPERATURE,
        "extraction_seed":        cfg.EXTRACTION_SEED,
        "llm_num_predict":        cfg.LLM_NUM_PREDICT,
        "llm_num_ctx":            cfg.LLM_NUM_CTX,
        "llm_frequency_penalty":  cfg.LLM_FREQUENCY_PENALTY,
        "llm_retry_penalties":    cfg.LLM_RETRY_FREQUENCY_PENALTIES,
        "max_gleanings":          cfg.MAX_GLEANINGS,
        "embedding_model":        cfg.EMBEDDING_MODEL,
        "bfs_hop_depth":          cfg.BFS_HOP_DEPTH,
        "ranking_method":         cfg.RANKING_METHOD,
        "top_n_candidates":       cfg.TOP_N_CANDIDATES_AFTER_RANKING,
        "top_n_final_evidence":   cfg.TOP_N_FINAL_EVIDENCE,
        "pipeline_b_sim_weight":  cfg.PIPELINE_B_SIM_WEIGHT,
        "pipeline_b_pr_weight":   cfg.PIPELINE_B_PR_WEIGHT,
        "exp_dir":                str(exp_dir),
        "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
        "think_override":         str(think_override),
    }

    t_total_start = time.monotonic()
    runtime_metrics: dict[str, Any] = {
        "model": model,
        "exp_dir": str(exp_dir),
        "timestamp": config_snapshot["timestamp"],
    }

    # ── Stage A ───────────────────────────────────────────────────────────────
    model_metadata = preflight(model, exp_dir)

    # ── Stage B ───────────────────────────────────────────────────────────────
    t_b = time.monotonic()
    canonical_chunks = read_canonical_chunks()
    runtime_metrics["stage_b_read_chunks_s"] = round(time.monotonic() - t_b, 3)
    runtime_metrics["canonical_chunk_count"] = len(canonical_chunks)

    # ── Stage C ───────────────────────────────────────────────────────────────
    extraction_jsonl = exp_dir / "extractions.jsonl"

    if args.skip_extraction and extraction_jsonl.exists():
        _log.info("Skipping extraction — loading from %s", extraction_jsonl)
        extractions = []
        with open(extraction_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    # Re-attach text from canonical chunks
                    for c in canonical_chunks:
                        if c["chunk_id"] == rec.get("chunk_id"):
                            rec["text"] = c["text"]
                            break
                    extractions.append(rec)
        _log.info("Loaded %d extraction records", len(extractions))
        runtime_metrics["stage_c_skipped"] = True
    else:
        private_cache_dir = str(exp_dir / "llm_cache")
        t_c = time.monotonic()
        extractions = run_extraction(
            model, canonical_chunks, private_cache_dir, think_override
        )
        runtime_metrics["stage_c_extraction_s"] = round(time.monotonic() - t_c, 3)
        runtime_metrics["extraction_parse_failures"] = sum(
            1 for r in extractions if "_parse_error" in r
        )
        runtime_metrics["extraction_total_entities"] = sum(
            len(r.get("entities", [])) for r in extractions
        )
        runtime_metrics["extraction_total_relationships"] = sum(
            len(r.get("relationships", [])) for r in extractions
        )

    # ── Stage D ───────────────────────────────────────────────────────────────
    t_d = time.monotonic()
    G, graph_metrics = build_experiment_graph(extractions)
    runtime_metrics["stage_d_graph_build_s"] = round(time.monotonic() - t_d, 3)

    from graphrag_lite.graph.ranking import rank_pagerank
    pr = rank_pagerank(G)
    pr_max = max(pr.values()) if pr else 1.0

    # ── Stage E ───────────────────────────────────────────────────────────────
    if args.skip_collections and (exp_dir / "entities").exists():
        _log.info("Skipping collection build — using existing collections.")
        collection_meta = {
            "entities_dir": str(exp_dir / "entities"),
            "relationships_dir": str(exp_dir / "relationships"),
            "chunks_dir": str(exp_dir / "chunks"),
            "skipped": True,
        }
    else:
        t_e = time.monotonic()
        collection_meta = build_experiment_collections(G, extractions, exp_dir, model)
        runtime_metrics["stage_e_collections_s"] = round(time.monotonic() - t_e, 3)

    # ── Stage F + G ───────────────────────────────────────────────────────────
    if args.skip_eval:
        _log.info("Skipping evaluation (--skip-eval).")
        evaluation: dict = {"model": model, "skipped": True}
    else:
        t_g = time.monotonic()
        evaluation = evaluate_all(G, pr, exp_dir, model)
        runtime_metrics["stage_g_eval_s"] = round(time.monotonic() - t_g, 3)

        # Print summary table
        _print_eval_summary(evaluation)

    # ── Stage H ───────────────────────────────────────────────────────────────
    runtime_metrics["total_s"] = round(time.monotonic() - t_total_start, 3)
    save_outputs(
        exp_dir=exp_dir,
        model=model,
        config_snapshot=config_snapshot,
        model_metadata=model_metadata,
        extractions=extractions,
        G=G,
        graph_metrics=graph_metrics,
        collection_meta=collection_meta,
        evaluation=evaluation,
        runtime_metrics=runtime_metrics,
    )

    _log.info("=== Experiment complete: %s ===", model)
    _log.info("Output directory: %s", exp_dir)
    _log.info("Total wall time: %.1fs", runtime_metrics["total_s"])
    return 0


def _print_eval_summary(evaluation: dict) -> None:
    agg = evaluation.get("aggregates", {})
    print()
    print("=" * 90)
    print(f"EVALUATION SUMMARY — model={evaluation.get('model', '?')}")
    print("=" * 90)
    fmt = "{:<20} {:>10} {:>10} {:>14} {:>12}"
    print(fmt.format("Pipeline", "Recall@K", "MRR", "UniqueGold", "AvgLatency(s)"))
    print("-" * 70)
    for pl, a in agg.items():
        if not a:
            continue
        print(fmt.format(
            pl,
            f"{a.get('avg_recall_at_k', 0):.4f}",
            f"{a.get('mrr', 0):.4f}",
            str(a.get("unique_gold_chunks_recovered", "-")),
            f"{a.get('avg_retrieval_latency_s', 0):.3f}",
        ))
    print("=" * 90)

    d18 = evaluation.get("d18_diagnostic")
    if d18:
        print("\nD18 Diagnostic:")
        for pl, res in d18.items():
            if isinstance(res, dict) and "gold_retrieved" in res:
                print(f"  {pl:<20} gold_retrieved={res['gold_retrieved']}  latency={res.get('latency_s', 0):.3f}s")


if __name__ == "__main__":
    sys.exit(main())
