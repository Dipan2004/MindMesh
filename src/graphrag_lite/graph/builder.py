"""
builder.py -- Build a NetworkX DiGraph from extraction results.

Entities become nodes; relationships become directed edges. Duplicate entity
names (case-insensitive, whitespace-normalized) are merged into a single node.

A second alias-merge pass collapses plural/capitalisation variants
(e.g. 'Knowledge Graph'/'Knowledge graphs', 'LLM'/'LLMs') using a conservative
normalisation: casefold, strip punctuation, strip trailing 's'.

SELF-LOOP POLICY: dropped and logged. Self-loops carry no graph information;
PageRank inflates scores for self-looping nodes; every instance in this corpus
is extractor noise. 0 post-merge self-loops confirmed.

PARALLEL EDGE POLICY: merged. Same (src, tgt) pair accumulates all types and
source_sentences on one edge. Consistent with node description merging.

NODE ATTRIBUTE MERGE: type=first non-OTHER wins; descriptions/source_sentences
= lists with (chunk_id, doc_id) provenance; source_docs = set.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


# ---- Normalisation ----------------------------------------------------------

def _norm(name: str) -> str:
    """Primary dedup key: casefold + whitespace collapse.
    Matches extractor.py and gleaning.py -- do not change independently."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _norm_alias(name: str) -> str:
    """Secondary alias-merge key: casefold, strip punctuation, strip trailing s.
    More aggressive than _norm; kept separate so the relationship-endpoint
    validator (_norm) is not affected."""
    s = name.casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1] if s.endswith("s") else s


# ---- JSONL loader -----------------------------------------------------------

def load_extractions(jsonl_path: str) -> list[dict]:
    """Load clean records from extractions.jsonl. Skips _parse_error records."""
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
                print(f"  [SKIP] line {lineno}: invalid JSON -- {exc}")
                skipped += 1
                continue

            if "_parse_error" in record:
                print(
                    f"  [SKIP] chunk_id={record.get('chunk_id', '?')} "
                    f"doc={record.get('doc_id', '?')} -- "
                    f"_parse_error: {record['_parse_error'][:120]}"
                )
                skipped += 1
                continue

            clean.append(record)

    print(f"[load_extractions] {len(clean)} clean records, {skipped} skipped")
    return clean


# ---- Graph builder ----------------------------------------------------------

def build_graph(extractions: list[dict]) -> nx.DiGraph:
    """Build a NetworkX DiGraph from extraction records."""
    G = nx.DiGraph()
    canonical: dict[str, str] = {}  # norm_key -> canonical_name

    # ---- Pass 1: nodes ------------------------------------------------------
    for record in extractions:
        chunk_id = record.get("chunk_id", "")
        doc_id   = record.get("doc_id", "")

        for entity in record.get("entities", []):
            name: str = entity.get("name", "").strip()
            if not name:
                continue
            key   = _norm(name)
            etype = entity.get("type", "OTHER").strip().upper() or "OTHER"
            desc  = entity.get("description", "").strip()
            ssent = entity.get("source_sentence", "").strip()

            if key not in canonical:
                canonical[key] = name
                G.add_node(
                    name,
                    type=etype if etype != "OTHER" else "OTHER",
                    descriptions=[{"text": desc, "chunk_id": chunk_id, "doc_id": doc_id}],
                    source_sentences=[{"text": ssent, "chunk_id": chunk_id, "doc_id": doc_id}],
                    source_docs={doc_id},
                )
            else:
                canon = canonical[key]
                nd: dict[str, Any] = G.nodes[canon]
                if nd["type"] == "OTHER" and etype != "OTHER":
                    nd["type"] = etype
                nd["descriptions"].append({"text": desc, "chunk_id": chunk_id, "doc_id": doc_id})
                nd["source_sentences"].append({"text": ssent, "chunk_id": chunk_id, "doc_id": doc_id})
                nd["source_docs"].add(doc_id)

    # ---- Pass 2: edges ------------------------------------------------------
    self_loops_dropped = parallel_merged = dangling_dropped = edges_added = 0

    for record in extractions:
        chunk_id = record.get("chunk_id", "")
        doc_id   = record.get("doc_id", "")

        for rel in record.get("relationships", []):
            src_raw  = rel.get("source", "").strip()
            tgt_raw  = rel.get("target", "").strip()
            if not src_raw or not tgt_raw:
                continue
            rel_type = rel.get("type", "RELATED_TO").strip().upper()
            desc     = rel.get("description", "").strip()
            ssent    = rel.get("source_sentence", "").strip()

            src_canon = canonical.get(_norm(src_raw))
            tgt_canon = canonical.get(_norm(tgt_raw))

            if src_canon is None or tgt_canon is None:
                missing = []
                if src_canon is None: missing.append(f"source={src_raw!r}")
                if tgt_canon is None: missing.append(f"target={tgt_raw!r}")
                print(f"  [WARN] dropping dangling edge ({src_raw!r} --{rel_type}--> {tgt_raw!r}): {', '.join(missing)}")
                dangling_dropped += 1
                continue

            if src_canon == tgt_canon:
                print(f"  [DROP-LOOP] ({src_raw!r} --{rel_type}--> {tgt_raw!r}) -> both resolve to {src_canon!r} [{doc_id}/{chunk_id[:8]}]")
                self_loops_dropped += 1
                continue

            if G.has_edge(src_canon, tgt_canon):
                ed = G[src_canon][tgt_canon]
                ed["types"].append(rel_type)
                ed["source_sentences"].append({"text": ssent, "chunk_id": chunk_id, "doc_id": doc_id, "type": rel_type})
                parallel_merged += 1
                continue

            G.add_edge(src_canon, tgt_canon,
                       types=[rel_type], description=desc,
                       source_sentences=[{"text": ssent, "chunk_id": chunk_id, "doc_id": doc_id, "type": rel_type}],
                       source_doc=doc_id)
            edges_added += 1

    print(
        f"[build_graph] nodes={G.number_of_nodes()}  edges={G.number_of_edges()}\n"
        f"  self_loops_dropped={self_loops_dropped}  parallel_merged={parallel_merged}  "
        f"dangling_dropped={dangling_dropped}"
    )

    # ---- Alias-merge pass ---------------------------------------------------
    alias_groups: dict[str, list[str]] = defaultdict(list)
    for node in list(G.nodes()):
        alias_groups[_norm_alias(node)].append(node)

    alias_merged = 0
    for key, nodes in alias_groups.items():
        if len(nodes) < 2:
            continue
        canon = sorted(nodes)[0]
        for old_name in nodes:
            if old_name == canon or old_name not in G:
                continue
            od, cd = G.nodes[old_name], G.nodes[canon]
            cd["descriptions"].extend(od.get("descriptions", []))
            cd["source_sentences"].extend(od.get("source_sentences", []))
            cd["source_docs"].update(od.get("source_docs", set()))
            if cd.get("type") == "OTHER" and od.get("type") != "OTHER":
                cd["type"] = od["type"]
            for _, tgt, edata in list(G.out_edges(old_name, data=True)):
                rt = tgt if tgt != old_name else canon
                if rt != canon and not G.has_edge(canon, rt):
                    G.add_edge(canon, rt, **edata)
                elif rt != canon:
                    G[canon][rt]["source_sentences"].extend(edata.get("source_sentences", []))
            for src, _, edata in list(G.in_edges(old_name, data=True)):
                rs = src if src != old_name else canon
                if rs != canon and not G.has_edge(rs, canon):
                    G.add_edge(rs, canon, **edata)
                elif rs != canon:
                    G[rs][canon]["source_sentences"].extend(edata.get("source_sentences", []))
            G.remove_node(old_name)
            alias_merged += 1

    if alias_merged:
        print(f"[build_graph] alias_merged={alias_merged} -> now {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G
