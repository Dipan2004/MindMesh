"""
entity_resolution — Zeno-inspired safe entity resolution for MindMesh.

Adapted from Zeno's memory system (zeno/memory/core/ids.py,
zeno/memory/storage/graph_store.py). Key adaptations:
  - Uses MindMesh's NetworkX DiGraph, not Zeno's SQLite graph_store
  - Conservative matching policy (never merge on embedding alone)
  - Full provenance preservation (source_docs, chunk_ids, descriptions)
  - Explicit merge history and rollback support
  - False-positive safeguards (Apple, Java disambiguation)

What was NOT copied from Zeno:
  - Zeno's merge_duplicate_entities() deletes the loser entity without
    migrating fact_entity associations — unsafe for MindMesh
  - Zeno's GraphConsolidator uses Gemini LLM — excluded entirely
  - Zeno's community detection — not needed for MindMesh
"""
from graphrag_lite.entity_resolution.normalization import normalize, norm_alias
from graphrag_lite.entity_resolution.canonical_ids import canonical_entity_id
from graphrag_lite.entity_resolution.aliases import AliasRegistry
from graphrag_lite.entity_resolution.candidate_matching import find_candidates
from graphrag_lite.entity_resolution.merge import safe_merge_node
from graphrag_lite.entity_resolution.resolver import EntityResolver

__all__ = [
    "normalize",
    "norm_alias",
    "canonical_entity_id",
    "AliasRegistry",
    "find_candidates",
    "safe_merge_node",
    "EntityResolver",
]
