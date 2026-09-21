"""
gleaning.py — Re-prompt the LLM to catch missed entities/relationships.

After the first extraction pass (Task 1.3), this module re-prompts the LLM
up to ``max_gleanings`` times asking "did you miss anything?" and merges
new findings into the result. Deduplication is by normalised entity name /
(source, target, type) triple.

This mirrors Microsoft GraphRAG's gleaning loop in
``index/operations/extract_graph/graph_extractor.py``.

Usage:
    from graphrag_lite.ingestion.gleaning import glean
    final = glean(chunk, first_pass_result, llm_client, max_gleanings=1)
"""

from __future__ import annotations

import json

from graphrag_lite.cache.llm_cache import cached_call
from graphrag_lite.config import get_config
from graphrag_lite.ingestion.chunker import Chunk
from graphrag_lite.ingestion.extractor import (
    Entity,
    ExtractionResult,
    LLMClient,
    Relationship,
    _extract_json,
    _parse_extraction,
)


# ------------------------------------------------------------------ #
# Gleaning prompt
# ------------------------------------------------------------------ #

_GLEANING_PROMPT_TEMPLATE = """You previously extracted entities and relationships from the TEXT below.
Here is what you found so far:

{existing_json}

Review the TEXT again carefully. Are there any entities or relationships you missed?
Return ONLY the NEW ones you missed — do not repeat what is already listed above.
If you found nothing new, return: {{"entities": [], "relationships": []}}

Return ONLY valid JSON with this exact schema — no markdown fences, no commentary:
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
# Deduplication helpers
# ------------------------------------------------------------------ #

def _entity_key(e: Entity) -> str:
    """Normalise entity name for deduplication (lowercase, stripped)."""
    return e["name"].strip().lower()


def _relationship_key(r: Relationship) -> tuple[str, str, str]:
    """Normalise relationship triple for deduplication."""
    return (
        r["source"].strip().lower(),
        r["target"].strip().lower(),
        r["type"].strip().upper(),
    )


def _merge(
    base: ExtractionResult,
    new_entities: list[Entity],
    new_relationships: list[Relationship],
) -> ExtractionResult:
    """
    Return a new ExtractionResult with new_entities/relationships merged in,
    deduplicating by normalised key.
    """
    existing_entity_keys = {_entity_key(e) for e in base["entities"]}
    existing_rel_keys = {_relationship_key(r) for r in base["relationships"]}

    merged_entities = list(base["entities"])
    for e in new_entities:
        if _entity_key(e) not in existing_entity_keys:
            merged_entities.append(e)
            existing_entity_keys.add(_entity_key(e))

    merged_rels = list(base["relationships"])
    for r in new_relationships:
        if _relationship_key(r) not in existing_rel_keys:
            merged_rels.append(r)
            existing_rel_keys.add(_relationship_key(r))

    return ExtractionResult(
        entities=merged_entities,
        relationships=merged_rels,
        chunk_id=base["chunk_id"],
        doc_id=base["doc_id"],
    )


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def glean(
    chunk: Chunk,
    first_pass: ExtractionResult,
    llm_client: LLMClient,
    max_gleanings: int | None = None,
) -> ExtractionResult:
    """
    Re-prompt the LLM up to ``max_gleanings`` times to catch missed items.

    Parameters
    ----------
    chunk:
        The original Chunk that was extracted.
    first_pass:
        The ExtractionResult from the first extraction pass (Task 1.3).
    llm_client:
        Any object with a ``generate(prompt: str) -> str`` method.
    max_gleanings:
        How many additional passes to run. Defaults to ``MAX_GLEANINGS``
        from config. Pass 0 to skip gleaning entirely.

    Returns
    -------
    ExtractionResult
        The merged result after all gleaning passes. If no new items are
        found in any pass, returns first_pass unchanged (no copies made
        unnecessarily beyond the first merge).

    Raises
    ------
    ValueError
        If the LLM returns unparseable JSON during a gleaning pass.
        Propagated — not swallowed.
    """
    cfg = get_config()
    model = cfg.effective_extraction_model()
    if max_gleanings is None:
        max_gleanings = cfg.MAX_GLEANINGS

    result = first_pass

    for pass_num in range(max_gleanings):
        existing_json = json.dumps(
            {"entities": result["entities"], "relationships": result["relationships"]},
            indent=2,
        )
        prompt = _GLEANING_PROMPT_TEMPLATE.format(
            existing_json=existing_json,
            chunk_text=chunk["text"],
        )

        # Cache key includes pass_num so each gleaning pass is cached independently
        cache_prompt = f"GLEANING_PASS_{pass_num}::{prompt}"

        raw_response = cached_call(
            prompt=cache_prompt,
            model=model,
            fn=lambda p, _client=llm_client, _prompt=prompt: _client.generate(_prompt),
        )

        try:
            raw_json = _extract_json(raw_response)
            new_result = _parse_extraction(raw_json, chunk["chunk_id"], chunk["doc_id"])
        except ValueError:
            # Unparseable gleaning response — treat as "nothing new found"
            # and stop gleaning rather than crashing the whole pipeline.
            # Re-raise so the caller can log/decide — we don't swallow silently.
            raise

        added_entities = len(new_result["entities"])
        added_rels = len(new_result["relationships"])

        if added_entities == 0 and added_rels == 0:
            # LLM found nothing new — no point doing further passes
            break

        result = _merge(result, new_result["entities"], new_result["relationships"])

    return result
