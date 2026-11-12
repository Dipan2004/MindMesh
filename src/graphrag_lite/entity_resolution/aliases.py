"""
aliases.py — Alias registry for entity resolution.

Allows different names to resolve to the same canonical entity.
Adapted from Zeno's GraphStore.add_alias() / resolve_entity_id(),
but implemented as a pure Python dict (no SQLite dependency) so it
works directly with MindMesh's NetworkX graph.

Usage:
    registry = AliasRegistry()
    registry.register("AI", "Artificial Intelligence", "CONCEPT")
    registry.register("GPT4", "GPT-4", "TECHNOLOGY")

    # Lookup
    canon = registry.resolve("AI")  # -> "Artificial Intelligence"
"""
from __future__ import annotations
import json
from pathlib import Path
from graphrag_lite.entity_resolution.normalization import normalize


class AliasRegistry:
    """
    Maps alias normalized-labels to canonical entity labels.

    Thread-safety: not thread-safe. Use a lock in concurrent contexts.
    Persistence: optional JSON file for durability across runs.
    """

    def __init__(self, path: Path | None = None) -> None:
        # alias_norm -> canonical_label
        self._aliases: dict[str, str] = {}
        self._path = path
        if path and path.exists():
            self._load(path)

    # ── Built-in domain aliases for the MindMesh corpus ──────────────────
    # These are KNOWN safe mappings in the AI/ML/RAG domain.
    # Do NOT add general-purpose mappings here; use register() for
    # corpus-specific aliases.

    _BUILTIN: list[tuple[str, str]] = [
        ("ai",                "Artificial Intelligence"),
        ("artificial intelligence (ai)", "Artificial Intelligence"),
        ("artificial intelligence(ai)", "Artificial Intelligence"),
        ("llm",               "Large Language Models"),
        ("large language model", "Large Language Models"),
        ("llms",              "Large Language Models"),
        ("gpt4",              "GPT-4"),
        ("gpt 4",             "GPT-4"),
        ("rag",               "Retrieval-Augmented Generation"),
        ("retrieval augmented generation", "Retrieval-Augmented Generation"),
        ("retrieval-augmented generation", "Retrieval-Augmented Generation"),
        ("bm25",              "BM25"),
        ("bm25 retrieval",    "BM25"),
        ("hnsw",              "HNSW"),
        ("hierarchical navigable small world", "HNSW"),
        ("knowledge graph",   "Knowledge Graph"),
        ("knowledge graphs",  "Knowledge Graph"),
        ("kg",                "Knowledge Graph"),
        ("transformer",       "Transformer"),
        ("transformer architecture", "Transformer"),
        ("bert",              "BERT"),
        ("gpt",               "GPT"),
        ("faiss",             "FAISS"),
        ("facebook ai similarity search", "FAISS"),
        ("ann",               "Approximate Nearest Neighbor Search"),
        ("approximate nearest neighbor", "Approximate Nearest Neighbor Search"),
        ("approximate nearest-neighbor", "Approximate Nearest Neighbor Search"),
        ("vector database",   "Vector Databases"),
        ("vector databases",  "Vector Databases"),
        ("vector db",         "Vector Databases"),
        ("rrf",               "Reciprocal Rank Fusion (RRF)"),
        ("reciprocal rank fusion", "Reciprocal Rank Fusion (RRF)"),
        ("hybrid search",     "Hybrid Search"),
    ]

    @classmethod
    def with_builtins(cls, path: Path | None = None) -> "AliasRegistry":
        """Create a registry pre-loaded with domain aliases."""
        reg = cls(path=path)
        for alias_norm, canonical in cls._BUILTIN:
            # Store directly (already normalized)
            reg._aliases[alias_norm] = canonical
        return reg

    def register(self, alias: str, canonical_label: str, _type: str | None = None) -> None:
        """
        Register an alias mapping.

        Parameters
        ----------
        alias : str
            The alias to map (e.g. "AI")
        canonical_label : str
            The canonical entity label to map to (e.g. "Artificial Intelligence")
        _type : str | None
            Unused; for API compatibility with Zeno-style callers.
        """
        self._aliases[normalize(alias)] = canonical_label
        if self._path:
            self._save()

    def resolve(self, label: str) -> str | None:
        """
        Return the canonical label for an alias, or None if not registered.

        Case-insensitive, unicode-aware lookup.
        """
        return self._aliases.get(normalize(label))

    def resolve_or_keep(self, label: str) -> str:
        """Return the canonical label, or the original label if no alias exists."""
        return self._aliases.get(normalize(label), label)

    def all_aliases(self) -> dict[str, str]:
        """Return a copy of all alias mappings."""
        return dict(self._aliases)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._aliases.update(data.get("aliases", {}))
        except Exception:
            pass

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"aliases": self._aliases}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def __len__(self) -> int:
        return len(self._aliases)

    def __repr__(self) -> str:
        return f"AliasRegistry({len(self._aliases)} aliases)"
