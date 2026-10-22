"""
test_stub_endpoints.py — the conditional endpoint filter (step 2).

The extractor's endpoint-consistency filter drops 31% of the relationships
the model emits (53 of 171 on the ladder arm), 45 of them because the
model named an entity in a relationship that it did not also list as an
entity.  Most of those are not hallucinations — the name IS in the chunk
text — so the filter can be relaxed **conditionally**:

    if a relationship endpoint is not in the chunk's entity list BUT its
    name occurs verbatim in the chunk text (casefold, word boundaries),
    add it as an entity with type OTHER and marker attribute stub=True;
    otherwise drop as before.

The flag is OFF by default (`EXTRACTION_STUB_ENDPOINTS`).  These tests
cover the OFF behaviour (unchanged), the ON behaviour, the word-boundary
rule, and the "one bad endpoint dooms the pair" rule.

No network, no LLM: `_parse_extraction` is a pure function over a parsed
JSON dict.

Run:  python -m pytest tests/test_stub_endpoints.py -q
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag_lite.ingestion.extractor import (  # noqa: E402
    _name_occurs_verbatim,
    _parse_extraction,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

CHUNK_TEXT = (
    "Google Knowledge Graph is a knowledge graph produced by Google. "
    "The Wikidata project is run by the Wikimedia Foundation, and the "
    "Wikipedia article describes it. Google acquired DeepMind in 2014."
)

RAW_JSON = {
    "entities": [
        {"name": "Google Knowledge Graph", "type": "CONCEPT",
         "description": "A knowledge graph produced by Google.",
         "source_sentence": "Google Knowledge Graph is a knowledge graph produced by Google."},
        {"name": "Google", "type": "ORGANIZATION",
         "description": "The search company.",
         "source_sentence": "Google Knowledge Graph is a knowledge graph produced by Google."},
    ],
    "relationships": [
        # Both endpoints ARE entities -> kept, always.
        {"source": "Google Knowledge Graph", "target": "Google",
         "type": "PRODUCED_BY", "description": "Produced by Google.",
         "source_sentence": "Google Knowledge Graph is a knowledge graph produced by Google."},
        # "DeepMind" is NOT in the entity list, but IS in the chunk text
        # -> recovered as a stub when the flag is on.
        {"source": "Google", "target": "DeepMind",
         "type": "ACQUIRED_BY", "description": "Google acquired DeepMind in 2014.",
         "source_sentence": "Google acquired DeepMind in 2014."},
        # "Nonexistent Corp" is NOT in the entity list and NOT in the
        # chunk text -> dropped either way.
        {"source": "Google", "target": "Nonexistent Corp",
         "type": "PARTNERS_WITH", "description": "Invented.",
         "source_sentence": "Google partners with Nonexistent Corp."},
    ],
}


def _parse(raw: dict, *, chunk_text: str | None = CHUNK_TEXT,
           allow_stub: bool = False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _parse_extraction(
            raw, "chunk1", "doc1",
            chunk_text=chunk_text,
            allow_stub_endpoints=allow_stub,
        )
    return result, [str(w.message) for w in caught]


# ------------------------------------------------------------------ #
# 1. Default OFF — behaviour must be unchanged
# ------------------------------------------------------------------ #

def test_flag_defaults_to_off():
    """`EXTRACTION_STUB_ENDPOINTS` must default to False."""
    from graphrag_lite.config import Config

    assert Config().EXTRACTION_STUB_ENDPOINTS is False


def test_off_drops_the_dangling_relationship():
    """With the flag off, the endpoint filter behaves exactly as before."""
    result, messages = _parse(RAW_JSON, allow_stub=False)

    assert [r["target"] for r in result["relationships"]] == ["Google"]
    assert any("DeepMind" in m and "dropping relationship" in m for m in messages)
    assert not any(e.get("stub") for e in result["entities"])
    assert "DeepMind" not in {e["name"] for e in result["entities"]}


def test_off_still_drops_the_hallucinated_endpoint():
    result, messages = _parse(RAW_JSON, allow_stub=False)
    assert "Nonexistent Corp" not in {r["target"] for r in result["relationships"]}
    assert any("Nonexistent Corp" in m for m in messages)


# ------------------------------------------------------------------ #
# 2. Flag ON — the conditional recovery
# ------------------------------------------------------------------ #

def test_on_recovers_the_endpoint_present_in_the_chunk_text():
    """The point of the flag: DeepMind IS in the text, so it is kept."""
    result, messages = _parse(RAW_JSON, allow_stub=True)

    targets = [r["target"] for r in result["relationships"]]
    assert "DeepMind" in targets
    assert len(result["relationships"]) == 2          # +1 vs the off case
    assert any("RECOVERED" in m for m in messages)


def test_on_adds_the_stub_entity_with_type_other_and_the_marker():
    result, _ = _parse(RAW_JSON, allow_stub=True)

    stubs = [e for e in result["entities"] if e.get("stub")]
    assert len(stubs) == 1
    stub = stubs[0]
    assert stub["name"] == "DeepMind"
    assert stub["type"] == "OTHER"
    assert stub["stub"] is True
    # A stub still carries provenance: the sentence it was found in.
    assert "DeepMind" in stub["source_sentence"]


def test_on_still_drops_the_hallucinated_endpoint():
    """The verbatim check is the guard: a name NOT in the text is dropped."""
    result, messages = _parse(RAW_JSON, allow_stub=True)

    assert "Nonexistent Corp" not in {r["target"] for r in result["relationships"]}
    assert any("Nonexistent Corp" in m and "dropping relationship" in m
               for m in messages)
    assert not any(e["name"] == "Nonexistent Corp" for e in result["entities"])


def test_on_adds_the_stub_entity_to_the_entity_list():
    """The recovered endpoint must be a real entity in the result, so the
    graph builder can create a node for it."""
    result, _ = _parse(RAW_JSON, allow_stub=True)
    names = {e["name"] for e in result["entities"]}
    assert "DeepMind" in names
    # The original entities are untouched.
    assert "Google" in names and "Google Knowledge Graph" in names
    assert len(result["entities"]) == 3


def test_stub_is_added_only_once_per_chunk():
    """Two relationships naming the same missing endpoint -> one stub."""
    raw = {
        "entities": [{"name": "Google", "type": "ORGANIZATION",
                      "description": "d", "source_sentence": "s"}],
        "relationships": [
            {"source": "Google", "target": "DeepMind", "type": "ACQUIRED_BY",
             "description": "d", "source_sentence": "s"},
            {"source": "Google", "target": "DeepMind", "type": "INVESTED_IN",
             "description": "d", "source_sentence": "s"},
        ],
    }
    result, _ = _parse(raw, allow_stub=True)
    stubs = [e for e in result["entities"] if e.get("stub")]
    assert len(stubs) == 1
    assert len(result["relationships"]) == 2


def test_one_bad_endpoint_doomsthe_whole_relationship():
    """A relationship with one recoverable and one hallucinated endpoint is
    dropped — a half-known edge is still a dangling edge."""
    raw = {
        "entities": [{"name": "Google", "type": "ORGANIZATION",
                      "description": "d", "source_sentence": "s"}],
        "relationships": [
            {"source": "DeepMind", "target": "Imaginary Labs",
             "type": "PARTNERS_WITH", "description": "d",
             "source_sentence": "s"},
        ],
    }
    result, messages = _parse(raw, allow_stub=True)
    assert result["relationships"] == []
    # No stub was created for the dropped relationship.
    assert not any(e.get("stub") for e in result["entities"])
    assert any("dropping relationship" in m for m in messages)


# ------------------------------------------------------------------ #
# 3. The verbatim / word-boundary rule
# ------------------------------------------------------------------ #

def test_word_boundaries_are_respected():
    assert _name_occurs_verbatim("AI", "The AI field grew.") is True
    assert _name_occurs_verbatim("AI", "AIDS was studied.") is False
    assert _name_occurs_verbatim("Graph", "A Graph database.") is True
    assert _name_occurs_verbatim("Graph", "GraphQL is a query language.") is False


def test_the_match_is_case_insensitive():
    assert _name_occurs_verbatim("google", "GOOGLE is a company.") is True
    assert _name_occurs_verbatim("Google", "google is a company.") is True


def test_the_match_tolerates_a_wrapped_name():
    """A name broken across a newline still matches."""
    assert _name_occurs_verbatim(
        "Google Knowledge Graph",
        "The Google\n   Knowledge  Graph is a thing.",
    ) is True


def test_a_name_absent_from_the_text_does_not_match():
    assert _name_occurs_verbatim("DeepMind", "Google acquired DeepMind in 2014.") is True
    assert _name_occurs_verbatim("DeepMind", "Google acquired something in 2014.") is False
    assert _name_occurs_verbatim("DeepMind", "") is False


def test_a_multi_word_name_needs_the_whole_phrase():
    assert _name_occurs_verbatim(
        "Wikimedia Foundation", "run by the Wikimedia Foundation.") is True
    assert _name_occurs_verbatim(
        "Wikimedia Foundation", "run by the Wikimedia office.") is False


def test_punctuation_at_the_edge_of_a_name():
    """A name ending in '.' is bounded only where a word boundary makes
    sense — 'U.S.' must match, 'U.S.A' must not count as 'U.S.'."""
    assert _name_occurs_verbatim("U.S.", "The U.S. economy grew.") is True
    assert _name_occurs_verbatim("U.S.", "The U.S.A economy grew.") is False


# ------------------------------------------------------------------ #
# 4. Interaction with gleaning's prior_entity_names
# ------------------------------------------------------------------ #

def test_prior_entity_names_still_rescue_endpoints():
    """Gleaning passes the first pass's entity names in; those endpoints are
    already known, so the relationship is kept WITHOUT a stub being made."""
    raw = {
        "entities": [],
        "relationships": [
            {"source": "Google", "target": "DeepMind", "type": "ACQUIRED_BY",
             "description": "d", "source_sentence": "s"},
        ],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _parse_extraction(
            raw, "chunk1", "doc1",
            prior_entity_names={"google", "deepmind"},
            chunk_text=CHUNK_TEXT,
            allow_stub_endpoints=True,
        )
    assert len(result["relationships"]) == 1
    assert not any(e.get("stub") for e in result["entities"])
    assert not any("dropping relationship" in str(w.message) for w in caught)
    assert not any("RECOVERED" in str(w.message) for w in caught)


def test_without_prior_names_the_same_relationship_becomes_a_stub():
    """The control for the test above: the same relationship, with no prior
    names, is recovered as a stub because the name is in the chunk text."""
    raw = {
        "entities": [],
        "relationships": [
            {"source": "Google", "target": "DeepMind", "type": "ACQUIRED_BY",
             "description": "d", "source_sentence": "s"},
        ],
    }
    result, _ = _parse(raw, allow_stub=True)
    assert len(result["relationships"]) == 1
    stubs = [e for e in result["entities"] if e.get("stub")]
    # Neither endpoint was listed as an entity, and BOTH names occur in the
    # chunk text, so both are recovered as stubs.
    assert sorted(e["name"] for e in stubs) == ["DeepMind", "Google"]
