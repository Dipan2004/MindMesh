"""
builder.py — Build a NetworkX DiGraph from extraction results.

Entities become nodes; relationships become directed edges. Duplicate entity
names (case-insensitive, whitespace-normalized) are merged into a single
node.

SELF-LOOP POLICY
----------------
Self-loops (source == target after merge) are DROPPED and counted. Rationale:
  1. Self-loops carry no graph information — they connect nothing.
  2. NetworkX PageRank inflates scores for self-looping nodes.
  3. Every self-loop in this corpus is extractor noise (qwen2.5:0.5b
     hallucinating 'X --> X' directly) — none are meaningful KB facts.
  4. Verified: 0 post-merge self-loops exist in the canonical corpus,
     so dropping them never loses real cross-entity relationships.

PARALLEL EDGE POLICY
--------------------
When two or more relationships share the same (source, target) pair (whether
same type or different types), they are MERGED into one edge with combined
provenance. This is consistent with how node descriptions are merged, and
avoids DiGraph silently overwriting earlier relationships with later ones.
Combined edge attributes:
  - ``types``: list of all relationship types across all merged pairs.
  - ``source_sentences``: list of all {text, chunk_id, doc_id, type} entries.
  - ``description``: first non-empty description seen.
  - ``source_doc``: first source doc seen.

ATTRIBUTE MERGE POLICY FOR NODES
---------------------------------
  - ``type``: first non-OTHER type wins.
  - ``descriptions``: list of all, with (chunk_id, doc_id) provenance.
  - ``source_sentences``: same list structure.
  - ``source_docs``: set of all doc_ids.

Usage:
    from graphrag_lite.graph.builder import build_graph, load_extractions
    records = load_extractions("data/cache/extractions.jsonl")
    G = build_graph(records)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """
    Case-insensitive, whitespace-normalised key for entity deduplication.
    Matches normalization in extractor.py and gleaning.py — do not change
    independently.
    """
    return re.sub(r"\s+", " ", name.strip().lower())


# ── JSONL loader ──────────────────────────────────────────────────────────────

def load_extractions(jsonl_path: str) -> list[dict]:
    """
    Load clean extraction records from a JSONL file.

    Records with ``_parse_error`` are skipped with a visible [SKIP] log line.

    Returns
    -------
    list[dict]  — clean records only.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Extractions file not found: {path}")

    clean: list[dict] = []
    skipped = 0

    with path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"  [SKIP] line {lineno}: invalid JSON — {exc}")
                skipped += 1
                continue

            if "_parse_error" in record:
                print(
                    f"  [SKIP] chunk_id={record.get('chunk_id', '?')} "
                    f"doc={record.get('doc_id', '?')} — "
                    f"_parse_error: {record['_parse_error'][:120]}"
                )
                skipped += 1
                continue

            clean.append(record)

    print(f"[load_extractions] {len(clean)} clean records, {skipped} skipped")
    return clean


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(extractions: list[dict]) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from extraction records.

    See module docstring for full merge and self-loop policies.

    Parameters
    ----------
    extractions:
        List of clean ExtractionResult dicts (no ``_parse_error`` field).

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    # norm_key → canonical_name (first occurrence wins)
    canonical: dict[str, str] = {}

    # ── Pass 1: build nodes ───────────────────────────────────────────────────
    for record in extractions:
        chunk_id = record.get("chunk_id", "")
        doc_id = record.get("doc_id", "")

        for entity in record.get("entities", []):
            name: str = entity.get("name", "").strip()
            if not name:
                continue

            key = _norm(name)
            etype = entity.get("type", "OTHER").strip().upper() or "OTHER"
            desc = entity.get("description", "").strip()
            ssent = entity.get("source_sentence", "").strip()

            if key not in canonical:
                canonical[key] = name
                G.add_node(
                    name,
                    type=etype if etype != "OTHER" else "OTHER",
                    descriptions=[{"text": desc,
                                   "chunk_id": chunk_id,
                                   "doc_id": doc_id}],
                    source_sentences=[{"text": ssent,
                                       "chunk_id": chunk_id,
                                       "doc_id": doc_id}],
                    source_docs={doc_id},
                )
            else:
                canon = canonical[key]
                nd: dict[str, Any] = G.nodes[canon]
                if nd["type"] == "OTHER" and etype != "OTHER":
                    nd["type"] = etype
                nd["descriptions"].append(
                    {"text": desc, "chunk_id": chunk_id, "doc_id": doc_id}
                )
                nd["source_sentences"].append(
                    {"text": ssent, "chunk_id": chunk_id, "doc_id": doc_id}
                )
                nd["source_docs"].add(doc_id)

    # ── Pass 2: add edges ─────────────────────────────────────────────────────
    self_loops_dropped = 0
    parallel_merged = 0
    dangling_dropped = 0
    edges_added = 0

    for record in extractions:
        chunk_id = record.get("chunk_id", "")
        doc_id = record.get("doc_id", "")

        for rel in record.get("relationships", []):
            src_raw = rel.get("source", "").strip()
            tgt_raw = rel.get("target", "").strip()
            if not src_raw or not tgt_raw:
                continue

            rel_type = rel.get("type", "RELATED_TO").strip().upper()
            desc = rel.get("description", "").strip()
            ssent = rel.get("source_sentence", "").strip()

            src_canon = canonical.get(_norm(src_raw))
            tgt_canon = canonical.get(_norm(tgt_raw))

            # Dangling: endpoint has no node
            if src_canon is None or tgt_canon is None:
                missing = []
                if src_canon is None:
                    missing.append(f"source={src_raw!r}")
                if tgt_canon is None:
                    missing.append(f"target={tgt_raw!r}")
                print(
                    f"  [WARN] dropping dangling edge "
                    f"({src_raw!r} --{rel_type}--> {tgt_raw!r}): "
                    f"missing: {', '.join(missing)}"
                )
                dangling_dropped += 1
                continue

            # Self-loop: same canonical node on both ends — drop
            if src_canon == tgt_canon:
                print(
                    f"  [DROP-LOOP] ({src_raw!r} --{rel_type}--> {tgt_raw!r}) "
                    f"→ both resolve to {src_canon!r} [{doc_id}/{chunk_id[:8]}]"
                )
                self_loops_dropped += 1
                continue

            # Parallel edge: (src_canon, tgt_canon) pair already exists — merge
            if G.has_edge(src_canon, tgt_canon):
                ed = G[src_canon][tgt_canon]
                ed["types"].append(rel_type)
                ed["source_sentences"].append(
                    {"text": ssent, "chunk_id": chunk_id,
                     "doc_id": doc_id, "type": rel_type}
                )
                parallel_merged += 1
                continue

            # New edge
            G.add_edge(
                src_canon,
                tgt_canon,
                types=[rel_type],
                description=desc,
                source_sentences=[
                    {"text": ssent, "chunk_id": chunk_id,
                     "doc_id": doc_id, "type": rel_type}
                ],
                source_doc=doc_id,
            )
            edges_added += 1

    print(
        f"[build_graph] nodes={G.number_of_nodes()}  edges={G.number_of_edges()}\n"
        f"  self_loops_dropped={self_loops_dropped}  "
        f"parallel_merged={parallel_merged}  "
        f"dangling_dropped={dangling_dropped}"
    )
    return G
