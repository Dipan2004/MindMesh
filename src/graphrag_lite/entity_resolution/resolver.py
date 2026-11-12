"""
resolver.py — EntityResolver: orchestrates the full resolution pipeline.

Runs on an ISOLATED COPY of the graph (never on the canonical graph directly).
Applies the conservative matching policy:

  AUTOMATIC: exact name match + compatible type -> auto-merge
  REVIEW:    alias match + compatible type       -> merge with annotation
  EMBEDDING: strong cosine + compatible type     -> candidate only (not auto)
  KEEP:      everything else                     -> no change

Creates three graph configurations for comparison:
  baseline         — no merges
  config_a_exact   — AUTOMATIC merges only
  config_b_semantic — AUTOMATIC + REVIEW + strong embedding candidates

Usage:
    resolver = EntityResolver.from_extractions("data/cache/extractions.jsonl")
    results = resolver.run_all_configs()
    resolver.report(results)
"""
from __future__ import annotations
import copy
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from graphrag_lite.entity_resolution.normalization import normalize, norm_alias
from graphrag_lite.entity_resolution.canonical_ids import compatible_types
from graphrag_lite.entity_resolution.aliases import AliasRegistry
from graphrag_lite.entity_resolution.candidate_matching import (
    MergeCandidate, find_candidates
)
from graphrag_lite.entity_resolution.merge import safe_merge_node

log = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    name: str
    G: nx.DiGraph
    merges: list[dict] = field(default_factory=list)
    candidates: list[MergeCandidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def metrics(self) -> dict:
        comps = list(nx.weakly_connected_components(self.G))
        largest = max((len(c) for c in comps), default=0)
        isolated = sum(1 for n in self.G.nodes() if self.G.degree(n) == 0)
        cross_doc_nodes = sum(
            1 for n, d in self.G.nodes(data=True)
            if len(d.get("source_docs", set())) >= 2
        )
        cross_doc_edges = sum(
            1 for u, v in self.G.edges()
            if self.G.nodes[u].get("source_docs", set())
            and self.G.nodes[v].get("source_docs", set())
            and not self.G.nodes[u].get("source_docs", set())
               .intersection(self.G.nodes[v].get("source_docs", set()))
        )
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "components": len(comps),
            "largest_component": largest,
            "isolated_nodes": isolated,
            "cross_doc_nodes": cross_doc_nodes,
            "cross_doc_edges": cross_doc_edges,
            "merges_applied": len(self.merges),
            "merge_errors": len(self.errors),
        }


class EntityResolver:
    """
    Orchestrates entity resolution on an isolated copy of the graph.

    Never modifies the original graph directly. All changes happen on
    deep copies stored as GraphConfig instances.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        registry: AliasRegistry | None = None,
        embeddings: dict[str, list[float]] | None = None,
        embedding_threshold: float = 0.92,
    ) -> None:
        self._G_original = G
        self._registry = registry or AliasRegistry.with_builtins()
        self._embeddings = embeddings or {}
        self._embedding_threshold = embedding_threshold

    @classmethod
    def from_extractions(
        cls,
        jsonl_path: str | Path,
        registry: AliasRegistry | None = None,
    ) -> "EntityResolver":
        """Build a resolver from an extractions JSONL file."""
        import io, contextlib
        from graphrag_lite.graph.builder import build_graph, load_extractions
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            records = load_extractions(str(jsonl_path))
            G = build_graph(records)
        return cls(G, registry=registry)

    def _isolated_copy(self) -> nx.DiGraph:
        """Return a deep copy of the original graph."""
        G = copy.deepcopy(self._G_original)
        G.graph["merge_history"] = []
        return G

    def run_baseline(self) -> GraphConfig:
        """Return an unmodified copy as the baseline."""
        return GraphConfig(name="baseline", G=self._isolated_copy())

    def run_config_a(self) -> GraphConfig:
        """
        Configuration A: exact normalization + alias resolution only.
        No embedding-based candidates. Only AUTOMATIC classifications.
        """
        G = self._isolated_copy()
        reg = AliasRegistry.with_builtins()
        candidates = find_candidates(G, registry=reg, embeddings=None)
        merges, errors = [], []

        auto = [c for c in candidates if c.classification == "AUTOMATIC"]
        for cand in auto:
            if cand.node_a not in G or cand.node_b not in G:
                continue  # already merged in a prior step
            try:
                record = safe_merge_node(G, cand.node_a, cand.node_b, alias_registry=reg)
                merges.append(record)
            except Exception as exc:
                err = f"Merge {cand.node_a!r}+{cand.node_b!r}: {exc}"
                log.warning(err)
                errors.append(err)

        return GraphConfig(name="config_a_exact", G=G,
                           merges=merges, candidates=auto, errors=errors)

    def run_config_b(self) -> GraphConfig:
        """
        Configuration B: exact + alias + conservative embedding candidates.
        Applies AUTOMATIC and REVIEW merges.
        EMBEDDING candidates are recorded but NOT applied automatically.
        """
        G = self._isolated_copy()
        reg = AliasRegistry.with_builtins()
        candidates = find_candidates(
            G, registry=reg,
            embeddings=self._embeddings or None,
            embedding_threshold=self._embedding_threshold,
        )
        merges, errors = [], []

        # Apply AUTOMATIC + REVIEW (not EMBEDDING — too risky without human review)
        apply_classes = {"AUTOMATIC", "REVIEW"}
        for cand in [c for c in candidates if c.classification in apply_classes]:
            if cand.node_a not in G or cand.node_b not in G:
                continue
            try:
                record = safe_merge_node(G, cand.node_a, cand.node_b, alias_registry=reg)
                merges.append(record)
            except Exception as exc:
                err = f"Merge {cand.node_a!r}+{cand.node_b!r}: {exc}"
                log.warning(err)
                errors.append(err)

        return GraphConfig(name="config_b_semantic", G=G,
                           merges=merges, candidates=candidates, errors=errors)

    def run_all_configs(self) -> dict[str, GraphConfig]:
        """Run all three configurations and return them."""
        return {
            "baseline":          self.run_baseline(),
            "config_a_exact":    self.run_config_a(),
            "config_b_semantic": self.run_config_b(),
        }

    def report(self, configs: dict[str, GraphConfig]) -> dict:
        """Print and return a comparison report."""
        print("\n" + "="*75)
        print("  ENTITY RESOLUTION COMPARISON REPORT")
        print("="*75)
        fmt = "{:<22} {:>6} {:>6} {:>6} {:>8} {:>12} {:>7}"
        print(fmt.format("Config", "Nodes", "Edges", "Comps",
                         "LargestC", "XdocEdges", "Merges"))
        print("-"*75)
        rows = {}
        for name, cfg in configs.items():
            m = cfg.metrics()
            rows[name] = m
            print(fmt.format(
                name, m["nodes"], m["edges"], m["components"],
                m["largest_component"], m["cross_doc_edges"],
                m["merges_applied"],
            ))
        print("="*75)
        # Cross-doc edges breakdown
        print("\nCross-document edges per configuration:")
        for name, m in rows.items():
            indicator = " <-- IMPROVEMENT" if m["cross_doc_edges"] > 0 else ""
            print(f"  {name:<22}  {m['cross_doc_edges']}{indicator}")
        return rows

    def save_isolated_graph(
        self,
        config: GraphConfig,
        out_dir: Path,
    ) -> Path:
        """Save the resolved graph to a GraphML file."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{config.name}_graph.graphml"
        G_copy = config.G.copy()
        for n, d in G_copy.nodes(data=True):
            if "source_docs" in d and isinstance(d["source_docs"], set):
                G_copy.nodes[n]["source_docs"] = sorted(d["source_docs"])
            if "descriptions" in d:
                import json as _json
                G_copy.nodes[n]["descriptions"] = _json.dumps(d["descriptions"])
            if "source_sentences" in d:
                import json as _json
                G_copy.nodes[n]["source_sentences"] = _json.dumps(d["source_sentences"])
        nx.write_graphml(G_copy, str(path))
        return path
