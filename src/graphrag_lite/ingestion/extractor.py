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

class _EntityExtra(TypedDict, total=False):
    """
    Optional keys on an Entity.

    Python 3.10 (the project's floor) has no ``typing.NotRequired``, so the
    optional key is declared in a ``total=False`` base and the required
    keys in the subclass.  Totality is per-key: a key is required only if
    it is declared in a class whose ``total`` is True.
    """
    stub: bool           # True for an entity recovered by the endpoint filter


class Entity(_EntityExtra):
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
# Endpoint-stub recovery helpers
# ------------------------------------------------------------------ #

def _normalize_ws(text: str) -> str:
    """Collapse every run of whitespace to a single space."""
    return " ".join(text.split())


def _name_occurs_verbatim(name: str, text: str) -> bool:
    """
    True if ``name`` occurs in ``text`` as a whole word or whole phrase.

    The comparison is case-insensitive (casefold) and whitespace-tolerant:
    the chunk text may wrap a name across a newline, so internal runs of
    whitespace are collapsed on both sides before the match.  The match is
    bounded by word characters, so "AI" does not match "AIDS" and "Graph"
    does not match "GraphQL".

    A name that ends in punctuation (e.g. "U.S.") is bounded the same way:
    the character after it must not be a word character, so "U.S." matches
    "the U.S. economy" but not "the U.S.A economy".

    This is the test the endpoint filter uses to decide whether a
    relationship endpoint the model named but did not list as an entity is
    a real thing mentioned in the chunk, or a hallucination.
    """
    if not name or not text:
        return False
    norm_name = _normalize_ws(name).casefold()
    if not norm_name:
        return False
    norm_text = _normalize_ws(text).casefold()
    # The name must not be glued to a neighbouring word character on either
    # side. Lookarounds rather than \b, because \b is defined against
    # \w and would misbehave for a name that starts or ends in punctuation.
    pattern = r"(?<!\w)" + re.escape(norm_name) + r"(?!\w)"
    return re.search(pattern, norm_text) is not None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _find_sentence_with(text: str, name: str) -> str:
    """
    Return the first sentence of ``text`` that contains ``name``.

    Used to give a recovered stub entity a real ``source_sentence`` — the
    same provenance contract every other entity has.  Returns "" when no
    sentence contains the name (which cannot happen for a name that
    passed ``_name_occurs_verbatim``, except when the name straddles a
    sentence boundary).
    """
    target = _normalize_ws(name).casefold()
    if not target:
        return ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if target in _normalize_ws(sentence).casefold():
            return sentence.strip()
    return ""


# ------------------------------------------------------------------ #
# JSON extraction helper
# ------------------------------------------------------------------ #

def _extract_json(raw: str) -> dict:
    """
    Pull the first {...} JSON object out of the LLM's raw response.

    Handles common LLM failures:
    - Leading/trailing prose around the JSON
    - Markdown code fences (```json ... ```)
    - Trailing commas before a closing ] or } (repaired and re-parsed)
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
        # Retry once with trailing commas removed.
        #
        # Small models very often emit a trailing comma before the closing
        # ] or }, which strict json.loads rejects. Measured case (2026-10-22,
        # qwen2.5:1.5b, chunk a4a679c4, num_predict=4096): the response was a
        # COMPLETE 20-entity / 20-relationship extraction whose only defect
        # was one trailing comma -- strict parsing threw the whole thing away
        # and the chunk was recorded as a parse failure.
        #
        # The repair is a strict improvement: it can only turn a failure into
        # a success, and it never changes a response that already parsed.
        repaired = re.sub(r",(\s*[}\]])", r"\1", json_str)
        if repaired != json_str:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"JSON parse error: {exc}\nExtracted string:\n{json_str[:500]}"
        ) from exc


def _parse_extraction(raw_json: dict, chunk_id: str, doc_id: str,
                      prior_entity_names: set[str] | None = None,
                      chunk_text: str | None = None,
                      allow_stub_endpoints: bool = False) -> ExtractionResult:
    """Validate and normalise the parsed JSON into typed ExtractionResult.

    Parameters
    ----------
    prior_entity_names:
        Optional set of lowercased entity names already known for this chunk
        (e.g. from the first extraction pass). When provided, relationship
        endpoint validation uses (current_entities | prior_entity_names) so
        gleaning relationships that reference first-pass entities are not
        incorrectly dropped just because the gleaning response didn't
        re-declare those entities.
    chunk_text:
        The source text of the chunk. Required when ``allow_stub_endpoints``
        is True — it is the text the endpoint name is checked against.
    allow_stub_endpoints:
        When True, a relationship endpoint that is NOT in the entity list is
        **recovered** instead of dropped, provided the endpoint's name occurs
        verbatim in ``chunk_text``.  The recovered endpoint is added to the
        entity list with type ``OTHER`` and the marker attribute
        ``stub=True``; the relationship is then kept.  When the name does
        not occur in the text the relationship is dropped exactly as before.

        This is OFF by default (``EXTRACTION_STUB_ENDPOINTS``).  It exists
        because the endpoint-consistency filter drops 31% of the
        relationships the model emits, and most of those are not
        hallucinations: the model simply listed the entity in the
        relationship but not in the entity array.  The verbatim check is
        the guard that keeps genuinely fabricated endpoints out.
    """
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
        if entity["name"]:
            entities.append(entity)

    # Build a normalised set of entity names for relationship validation.
    # Merge with prior_entity_names when provided (gleaning fix).
    entity_names: set[str] = {e["name"].lower() for e in entities}
    if prior_entity_names:
        entity_names = entity_names | prior_entity_names

    # Stub entities recovered by the endpoint filter, keyed by the
    # lowercased endpoint name so the same missing endpoint is only
    # recovered once per chunk.
    stubs: dict[str, Entity] = {}

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
            # ---- Conditional recovery (EXTRACTION_STUB_ENDPOINTS) ---------
            # An endpoint the model named in a relationship but did not list
            # as an entity is recovered ONLY if the name occurs verbatim in
            # the chunk text. Otherwise it is dropped, exactly as before.
            if allow_stub_endpoints and chunk_text is not None:
                recoverable: list[str] = []
                for endpoint in (rel["source"], rel["target"]):
                    key = endpoint.lower()
                    if key in entity_names:
                        continue
                    if key in stubs:
                        continue          # already recovered in this chunk
                    if _name_occurs_verbatim(endpoint, chunk_text):
                        recoverable.append(endpoint)
                    else:
                        recoverable = []   # one bad endpoint dooms the pair
                        break
                if recoverable:
                    for endpoint in recoverable:
                        key = endpoint.lower()
                        sentence = _find_sentence_with(chunk_text, endpoint)
                        stub: Entity = {
                            "name": endpoint,
                            "type": "OTHER",
                            "description": (
                                "Recovered as a relationship endpoint: the name "
                                "occurs in the chunk text but was not listed as "
                                "an extracted entity."
                            ),
                            "source_sentence": sentence or rel["source_sentence"],
                            "stub": True,
                        }
                        stubs[key] = stub
                        entities.append(stub)
                        entity_names.add(key)
                    warnings.warn(
                        f"[extractor] chunk={chunk_id} doc={doc_id}: RECOVERED "
                        f"relationship ({rel['source']!r} --{rel['type']}--> "
                        f"{rel['target']!r}) by adding stub entity/entities "
                        f"{[e for e in recoverable]!r} (present in chunk text)",
                        UserWarning,
                        stacklevel=2,
                    )
                    relationships.append(rel)
                    continue
            # ----------------------------------------------------------------
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
    model, settings) triples are served from disk cache without a live LLM
    call.

    RETRY WITH ESCALATION (2026-10-22). A parse failure is not always a
    hopeless chunk: on a small model the dominant failure mode is a
    *degenerate repetition loop*, and the model is **bistable** — it
    settles into one of several stable attractors depending on the
    server's numeric state, so the same chunk and the same settings can
    parse in one run and loop in the next. A retry that changes the
    generation settings lands in a different attractor, and therefore has
    a good chance of succeeding.

    The ladder is ``LLM_RETRY_FREQUENCY_PENALTIES``: after the first
    attempt fails to parse, retry with each listed ``frequency_penalty``
    in order. Each attempt is a distinct decode and therefore a distinct
    cache entry, so a retry is never served the first attempt's failed
    response.

    Parameters
    ----------
    chunk:
        A Chunk as returned by ``chunk_document``.
    llm_client:
        Any object with a ``generate(prompt: str) -> str`` method.
        Objects that also provide ``with_overrides(**kw)`` and
        ``cache_tag`` participate in the retry ladder; for any other
        client only the first attempt is made.

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

    # Attempt 0 uses the client as configured; each later attempt
    # escalates the frequency penalty. ``with_overrides`` is optional --
    # a client without it simply gets one attempt.
    #
    # A rung whose value EQUALS the client's current penalty is skipped:
    # it would be the same decode under the same cache key, so the "retry"
    # would be handed the first attempt's failed response from the cache
    # and change nothing.
    can_escalate = hasattr(llm_client, "with_overrides")
    current_penalty = getattr(llm_client, "frequency_penalty", None)
    ladder: list[float | None] = [None]
    if can_escalate:
        for value in cfg.LLM_RETRY_FREQUENCY_PENALTIES:
            if current_penalty is not None and float(value) == float(current_penalty):
                continue
            ladder.append(float(value))

    last_error: Exception | None = None

    for attempt, penalty in enumerate(ladder):
        client = llm_client
        if penalty is not None:
            client = llm_client.with_overrides(frequency_penalty=penalty)

        # The cache key must distinguish this attempt's settings from the
        # previous attempt's, or the retry returns the failed response it
        # is supposed to replace.
        cache_tag = getattr(client, "cache_tag", "")

        raw_response = cached_call(
            prompt=prompt,
            model=model,
            fn=lambda p, _client=client: _client.generate(p),
            cache_tag=cache_tag,
        )

        try:
            raw_json = _extract_json(raw_response)
            return _parse_extraction(
                raw_json,
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk_text=chunk["text"],
                allow_stub_endpoints=cfg.EXTRACTION_STUB_ENDPOINTS,
            )
        except ValueError as exc:
            last_error = exc
            if attempt + 1 < len(ladder):
                warnings.warn(
                    f"[extractor] chunk {chunk['chunk_id']} failed to parse "
                    f"(attempt {attempt + 1}/{len(ladder)}, "
                    f"frequency_penalty="
                    f"{getattr(client, 'frequency_penalty', 'n/a')}): "
                    f"{str(exc)[:120]} — retrying with a higher penalty",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            break

    # The ladder is exhausted. Re-raise the LAST error so the caller sees
    # the most recent failure (the highest penalty, which is the attempt
    # most likely to have produced a different response). Not an assert:
    # `python -O` strips asserts, and `raise None` would be a TypeError
    # rather than the parse failure the caller needs to see.
    if last_error is None:      # unreachable: the ladder always has >= 1 rung
        raise ValueError(
            f"[extractor] chunk {chunk['chunk_id']}: no attempt was made"
        )
    raise last_error
