"""
extractor.py — LLM-based entity and relationship extraction (single pass).

Sends one chunk of text to the LLM with a structured extraction prompt,
parses the JSON response, and returns typed Entity and Relationship objects.
Every call goes through cached_call (Task 0.2) so re-runs don't re-pay.

The prompt forces the LLM to return an exact ``source_sentence`` for every
entity and relationship — giving sentence-level attribution per PRD §4.

Usage:
    from graphrag_lite.ingestion.extractor import extract_entities_relationships
    result = extract_entities_relationships(chunk, llm_client)
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Protocol, TypedDict

from graphrag_lite.cache.llm_cache import cached_call
from graphrag_lite.config import get_config
from graphrag_lite.ingestion.chunker import Chunk


# ------------------------------------------------------------------ #
# Output types
# ------------------------------------------------------------------ #

class Entity(TypedDict):
    name: str            # canonical entity name
    type: str            # e.g. PERSON, ORGANIZATION, CONCEPT, LOCATION
    description: str     # one-sentence description
    source_sentence: str # exact sentence from the chunk where this entity appears


class Relationship(TypedDict):
    source: str          # name of the source entity
    target: str          # name of the target entity
    type: str            # relationship label, e.g. "FOUNDED_BY", "WORKS_AT"
    description: str     # one-sentence description of this relationship
    source_sentence: str # exact sentence from the chunk where this relationship appears


class ExtractionResult(TypedDict):
    entities: list[Entity]
    relationships: list[Relationship]
    chunk_id: str
    doc_id: str


# ------------------------------------------------------------------ #
# LLM client protocol — any object with a .generate(prompt) -> str method
# ------------------------------------------------------------------ #

class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


# ------------------------------------------------------------------ #
# Prompt template
# ------------------------------------------------------------------ #

_EXTRACTION_PROMPT_TEMPLATE = """You are an expert knowledge-graph builder.
Extract all entities and relationships from the TEXT below.

Rules:
1. For every entity, return: name, type (PERSON/ORGANIZATION/LOCATION/CONCEPT/TECHNOLOGY/EVENT/OTHER), a one-sentence description, and the EXACT sentence from the text where it appears.
2. For every relationship, return: source entity name, target entity name, relationship type (an uppercase label like FOUNDED_BY / DEVELOPED_BY / PART_OF / USED_FOR / INTRODUCED_BY / SUCCESSOR_OF / LOCATED_IN / WORKS_AT / ACQUIRED_BY / other), a one-sentence description, and the EXACT sentence from the text where it appears.
3. source_sentence must be copied verbatim from the TEXT — do not paraphrase.
4. Return ONLY valid JSON matching this exact schema — no markdown fences, no commentary:

{{
  "entities": [
    {{"name": "...", "type": "...", "description": "...", "source_sentence": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "...", "source_sentence": "..."}}
  ]
}}

TEXT:
{chunk_text}
"""


# ------------------------------------------------------------------ #
# JSON extraction helper
# ------------------------------------------------------------------ #

def _extract_json(raw: str) -> dict:
    """
    Pull the first {...} JSON object out of the LLM's raw response.

    Handles common LLM failures:
    - Leading/trailing prose around the JSON
    - Markdown code fences (```json ... ```)
    - Truncated responses (raises ValueError so caller can see the failure)
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()

    # Find the outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found in LLM response. Raw response:\n{raw[:500]}"
        )

    json_str = cleaned[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON parse error: {exc}\nExtracted string:\n{json_str[:500]}"
        ) from exc


def _parse_extraction(raw_json: dict, chunk_id: str, doc_id: str) -> ExtractionResult:
    """Validate and normalise the parsed JSON into typed ExtractionResult."""
    entities: list[Entity] = []
    for e in raw_json.get("entities") or []:
        if not isinstance(e, dict):
            continue
        entity: Entity = {
            "name": str(e.get("name") or "").strip(),
            "type": str(e.get("type") or "OTHER").strip().upper(),
            "description": str(e.get("description") or "").strip(),
            "source_sentence": str(e.get("source_sentence") or "").strip(),
        }
        if entity["name"]:          # skip nameless entries silently
            entities.append(entity)

    # Build a normalised set of entity names for relationship validation
    entity_names: set[str] = {e["name"].lower() for e in entities}

    relationships: list[Relationship] = []
    for r in raw_json.get("relationships") or []:
        if not isinstance(r, dict):
            continue
        rel: Relationship = {
            "source": str(r.get("source") or "").strip(),
            "target": str(r.get("target") or "").strip(),
            "type": str(r.get("type") or "RELATED_TO").strip().upper(),
            "description": str(r.get("description") or "").strip(),
            "source_sentence": str(r.get("source_sentence") or "").strip(),
        }
        if not rel["source"] or not rel["target"]:
            continue   # skip structurally incomplete relationships

        # Option (a): validate that both endpoints appear in the entities list.
        # If not, warn and drop — do not silently pass through a dangling edge.
        src_known = rel["source"].lower() in entity_names
        tgt_known = rel["target"].lower() in entity_names
        if not src_known or not tgt_known:
            missing = []
            if not src_known:
                missing.append(f"source={rel['source']!r}")
            if not tgt_known:
                missing.append(f"target={rel['target']!r}")
            warnings.warn(
                f"[extractor] chunk={chunk_id} doc={doc_id}: dropping relationship "
                f"({rel['source']!r} --{rel['type']}--> {rel['target']!r}) — "
                f"endpoint(s) not in extracted entities list: {', '.join(missing)}",
                UserWarning,
                stacklevel=2,
            )
            continue

        relationships.append(rel)

    return ExtractionResult(
        entities=entities,
        relationships=relationships,
        chunk_id=chunk_id,
        doc_id=doc_id,
    )


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def extract_entities_relationships(
    chunk: Chunk,
    llm_client: LLMClient,
) -> ExtractionResult:
    """
    Extract entities and relationships from a single Chunk using the LLM.

    The call is routed through ``cached_call`` so identical (chunk_text,
    model) pairs are served from disk cache without a live LLM call.

    Parameters
    ----------
    chunk:
        A Chunk as returned by ``chunk_document``.
    llm_client:
        Any object with a ``generate(prompt: str) -> str`` method.

    Returns
    -------
    ExtractionResult
        Typed dict with ``entities``, ``relationships``, ``chunk_id``,
        ``doc_id``.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed as valid JSON — propagated so
        the caller can decide to retry or skip. Never silently swallowed.
    """
    cfg = get_config()
    model = cfg.effective_extraction_model()

    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])

    raw_response = cached_call(
        prompt=prompt,
        model=model,
        fn=lambda p: llm_client.generate(p),
    )

    raw_json = _extract_json(raw_response)
    return _parse_extraction(raw_json, chunk["chunk_id"], chunk["doc_id"])
