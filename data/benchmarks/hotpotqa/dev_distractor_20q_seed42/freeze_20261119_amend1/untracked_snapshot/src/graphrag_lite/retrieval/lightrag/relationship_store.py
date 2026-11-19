"""relationship_store.py -- cached index over <exp_dir>/relationship_records.json."""

from __future__ import annotations

import json
from pathlib import Path

from graphrag_lite.retrieval.lightrag.context import collection_path, resolve_exp_dir

_CACHE: dict[str, "RelationshipStore"] = {}


def _records_path(exp_dir: Path | str | None) -> Path:
    """relationship_records.json next to the relationship collection actually used."""
    rel = collection_path(exp_dir, "relationships")
    base = rel.parent if rel is not None else resolve_exp_dir(exp_dir)
    return base / "relationship_records.json"


class RelationshipStore:
    """rel_id -> record, plus (source, target) -> rel_ids.  Loaded once per path."""

    def __init__(self, exp_dir: Path | str | None = None) -> None:
        self.path = _records_path(exp_dir)
        self.by_id: dict[str, dict] = {}
        self.by_pair: dict[tuple[str, str], list[str]] = {}
        if self.path.exists():
            records = json.loads(self.path.read_text(encoding="utf-8"))
            for r in records:
                rid = r.get("rel_id")
                if not rid:
                    continue
                self.by_id[rid] = r
                key = (r.get("source_entity", ""), r.get("target_entity", ""))
                self.by_pair.setdefault(key, []).append(rid)

    def __len__(self) -> int:
        return len(self.by_id)

    def get(self, rel_id: str) -> dict | None:
        return self.by_id.get(rel_id)


def get_relationship_store(exp_dir: Path | str | None = None) -> RelationshipStore:
    key = str(_records_path(exp_dir))
    st = _CACHE.get(key)
    if st is None:
        st = RelationshipStore(exp_dir)
        _CACHE[key] = st
    return st
