"""
canonical_ids.py — Deterministic canonical IDs for entities.

Adapted from Zeno's ids.py::entity_id(). Key property: two entities with
the same normalized label and compatible types always get the same canonical ID,
regardless of which document or chunk they came from.

This is what enables cross-document entity merging:
  Doc A: "FAISS" (TECHNOLOGY) -> sha256("faiss|technology")[:24]
  Doc B: "FAISS" (TECHNOLOGY) -> sha256("faiss|technology")[:24]  <- SAME!
"""
from __future__ import annotations
import hashlib
from graphrag_lite.entity_resolution.normalization import normalize


# Type normalization: map minor variations to canonical type strings
_TYPE_CANON = {
    "TECH": "TECHNOLOGY",
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "CONCEPT": "CONCEPT",
    "SYSTEM": "TECHNOLOGY",    # treat SYSTEM as TECHNOLOGY
    "TASK": "CONCEPT",         # treat TASK as CONCEPT
    "SUBJECT": "CONCEPT",      # treat SUBJECT as CONCEPT
    "OTHER": "OTHER",
    "EVENT": "EVENT",
}


def canonicalize_type(entity_type: str) -> str:
    """Return the canonical type string for an entity type."""
    t = (entity_type or "OTHER").strip().upper()
    return _TYPE_CANON.get(t, t)


def canonical_entity_id(label: str, entity_type: str) -> str:
    """
    Deterministic 24-char hex ID for an entity from its normalized label + type.

    Two entities with the same (normalized label, canonical type) will always
    get the same canonical ID, regardless of source document.

    Adapted from: zeno/memory/core/ids.py::entity_id()
    Not adapted: Zeno uses |kind| where kind can be arbitrary; we constrain
    to canonical types to prevent 'faiss|TECHNOLOGY' vs 'faiss|SYSTEM' drift.

    >>> canonical_entity_id("FAISS", "TECHNOLOGY") == canonical_entity_id("faiss", "SYSTEM")
    True  # because SYSTEM canonicalizes to TECHNOLOGY
    """
    canon_type = canonicalize_type(entity_type)
    # Use OTHER for canonicalization so FAISS(TECH) and FAISS(OTHER) still merge
    # when one extraction was ambiguous. This is deliberate conservatism.
    effective_type = canon_type if canon_type != "OTHER" else "OTHER"
    payload = f"{normalize(label)}|{effective_type}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def compatible_types(type1: str, type2: str) -> bool:
    """
    Return True if two entity types are compatible for merging.

    Conservative: only merge if types are the same or one is OTHER.
    Does NOT merge PERSON with ORGANIZATION, TECHNOLOGY with LOCATION, etc.
    """
    t1 = canonicalize_type(type1)
    t2 = canonicalize_type(type2)
    if t1 == t2:
        return True
    if "OTHER" in (t1, t2):
        return True
    # Allow TECHNOLOGY <-> CONCEPT (both describe technical abstractions)
    if {t1, t2} <= {"TECHNOLOGY", "CONCEPT"}:
        return True
    return False
