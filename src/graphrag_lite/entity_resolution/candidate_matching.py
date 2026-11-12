"""
candidate_matching.py — Find entity merge candidates from a NetworkX graph.

Implements conservative matching policy:
  AUTOMATIC  — exact normalized name + compatible type
  REVIEW     — alias-normalized name match + compatible type
  EMBEDDING  — strong cosine similarity + compatible type + not ambiguous
  KEEP       — everything else

NEVER merge based on embedding similarity alone.
Ambiguous terms (Apple, Java, Python) are never merged automatically.

Adapted from Zeno's GraphStore.merge_duplicate_entities() candidate-finding
logic, but separated from the merge operation for safety.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from graphrag_lite.entity_resolution.normalization import (
    normalize, norm_alias, is_ambiguous, AMBIGUOUS_TERMS
)
from graphrag_lite.entity_resolution.canonical_ids import (
    canonical_entity_id, compatible_types, canonicalize_type
)
from graphrag_lite.entity_resolution.aliases import AliasRegistry


@dataclass
class MergeCandidate:
    """A proposed entity merge candidate."""
    node_a: str
    node_b: str
    type_a: str
    type_b: str
    docs_a: set
    docs_b: set
    # Matching signals
    exact_match: bool = False
    alias_match: bool = False
    embedding_sim: float = 0.0
    # Decision
    classification: str = "KEEP_SEPARATE"  # AUTOMATIC | REVIEW | EMBEDDING | KEEP_SEPARATE
    reason: str = ""

    @property
    def is_cross_doc(self) -> bool:
        return bool(self.docs_a and self.docs_b and
                    not self.docs_a.intersection(self.docs_b))


def find_candidates(
    G: nx.DiGraph,
    registry: AliasRegistry | None = None,
    embeddings: dict[str, list[float]] | None = None,
    embedding_threshold: float = 0.92,
    cross_doc_only: bool = False,
) -> list[MergeCandidate]:
    """
    Find entity merge candidates in the graph.

    Parameters
    ----------
    G : nx.DiGraph
        The MindMesh knowledge graph.
    registry : AliasRegistry | None
        Alias registry for known alias mappings.
    embeddings : dict[str, list[float]] | None
        Pre-computed embeddings keyed by node name. If provided, enables
        embedding-based candidate discovery.
    embedding_threshold : float
        Minimum cosine similarity for embedding-based candidates.
        Conservative default: 0.92. Never auto-merge on embedding alone.
    cross_doc_only : bool
        If True, only return candidates from different source documents.

    Returns
    -------
    list[MergeCandidate]
        Sorted by classification priority (AUTOMATIC first).
    """
    nodes = list(G.nodes(data=True))
    seen_pairs: set[frozenset] = set()
    candidates: list[MergeCandidate] = []

    for i, (n1, d1) in enumerate(nodes):
        t1 = d1.get("type", "OTHER")
        docs1 = d1.get("source_docs", set())
        norm1 = normalize(n1)
        alias1 = norm_alias(n1)

        for n2, d2 in nodes[i + 1:]:
            pair_key = frozenset({n1, n2})
            if pair_key in seen_pairs:
                continue
            t2 = d2.get("type", "OTHER")
            docs2 = d2.get("source_docs", set())

            # Skip if same node
            if n1 == n2:
                continue

            # Skip if cross_doc_only and not cross-doc
            if cross_doc_only and docs1.intersection(docs2):
                continue

            # Type compatibility check first (cheap)
            if not compatible_types(t1, t2):
                continue

            # Skip ambiguous base terms
            if is_ambiguous(n1) and is_ambiguous(n2):
                continue

            norm2 = normalize(n2)
            alias2 = norm_alias(n2)

            # Signal 1: exact normalized match
            exact = norm1 == norm2

            # Signal 2: alias-normalized match
            alias_match = (alias1 == alias2) and not exact

            # Signal 3: alias registry lookup
            registry_match = False
            if registry and not exact:
                canon1 = registry.resolve(n1)
                canon2 = registry.resolve(n2)
                if canon1 and canon1 == canon2:
                    registry_match = True
                elif canon1 and normalize(canon1) == norm2:
                    registry_match = True
                elif canon2 and normalize(canon2) == norm1:
                    registry_match = True

            # Signal 4: canonical ID match (Zeno-style deterministic)
            canonical_match = (canonical_entity_id(n1, t1) ==
                               canonical_entity_id(n2, t2))

            # Signal 5: embedding similarity (optional)
            emb_sim = 0.0
            if embeddings and n1 in embeddings and n2 in embeddings:
                v1, v2 = embeddings[n1], embeddings[n2]
                dot = sum(a * b for a, b in zip(v1, v2))
                n1m = sum(a * a for a in v1) ** 0.5
                n2m = sum(a * a for a in v2) ** 0.5
                emb_sim = dot / (n1m * n2m) if n1m * n2m > 0 else 0.0

            # Classification
            if not (exact or alias_match or registry_match or canonical_match or
                    emb_sim >= embedding_threshold):
                continue

            seen_pairs.add(pair_key)

            if (exact or canonical_match or registry_match) and not (
                is_ambiguous(n1) or is_ambiguous(n2)
            ):
                cls = "AUTOMATIC"
                reason = "exact/canonical/registry match"
            elif alias_match and not (is_ambiguous(n1) or is_ambiguous(n2)):
                cls = "REVIEW"
                reason = "alias-normalized match"
            elif emb_sim >= embedding_threshold and not (
                is_ambiguous(n1) or is_ambiguous(n2)
            ):
                cls = "EMBEDDING"
                reason = f"embedding sim={emb_sim:.4f}"
            else:
                cls = "KEEP_SEPARATE"
                reason = "ambiguous or insufficient signal"

            candidates.append(MergeCandidate(
                node_a=n1, node_b=n2,
                type_a=t1, type_b=t2,
                docs_a=set(docs1), docs_b=set(docs2),
                exact_match=exact or canonical_match,
                alias_match=alias_match or registry_match,
                embedding_sim=round(emb_sim, 4),
                classification=cls,
                reason=reason,
            ))

    # Sort: AUTOMATIC first, then REVIEW, then EMBEDDING, then KEEP
    priority = {"AUTOMATIC": 0, "REVIEW": 1, "EMBEDDING": 2, "KEEP_SEPARATE": 3}
    candidates.sort(key=lambda c: (priority.get(c.classification, 9), c.node_a))
    return candidates
