"""
test_serialize.py -- the evidence serializer (Task 4.4).

The contract: to_toon and to_json are LOSSLESS.  A serializer that quietly
drops a field is worse than no serializer, because the LLM never sees what
was lost -- so the round trip is the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag_lite.retrieval.pipeline_a import Evidence  # noqa: E402
from graphrag_lite.retrieval.serialize import (  # noqa: E402
    count_tokens,
    from_json,
    from_toon,
    to_json,
    to_toon,
)

SAMPLES = [
    Evidence(chunk_id="6d878b7f56a9b0c1", source_doc="knowledge_graphs",
             chunk_index=0, text="# Knowledge Graphs\n\nA knowledge graph is...",
             score=0.0327890123, source="chunk"),
    Evidence(chunk_id="33d32892fc640b3f", source_doc="retrieval_augmented_generation",
             chunk_index=1, text='He said "hello", then left.',
             score=0.0318, source="structural"),
    Evidence(chunk_id="abc", source_doc="d", chunk_index=0, text="",
             score=0.0, source="chunk"),
    Evidence(chunk_id="with,comma", source_doc="doc,with,commas",
             chunk_index=12, text="multi\nline\ttext with \"quotes\" and, commas",
             score=1.5e-05, source="chunk"),
]


def _same(a: Evidence, b: Evidence) -> bool:
    return (a.chunk_id == b.chunk_id and a.source_doc == b.source_doc
            and a.chunk_index == b.chunk_index and a.text == b.text
            and a.score == b.score and a.source == b.source)


def test_toon_round_trip_is_lossless():
    out = from_toon(to_toon(SAMPLES))
    assert len(out) == len(SAMPLES)
    for a, b in zip(SAMPLES, out):
        assert _same(a, b), f"{a} != {b}"


def test_json_round_trip_is_lossless():
    out = from_json(to_json(SAMPLES))
    assert len(out) == len(SAMPLES)
    for a, b in zip(SAMPLES, out):
        assert _same(a, b), f"{a} != {b}"


def test_toon_header_names_the_columns_once():
    text = to_toon(SAMPLES)
    assert text.startswith(
        "evidence[4]{chunk_id,source_doc,chunk_index,score,source,text}:")
    assert text.count("chunk_id") == 1        # the key appears only in the header
    assert text.count("source_doc") == 1
    assert "{" not in text.split("\n", 1)[1]  # no per-row braces


def test_metadata_only_variant_drops_the_text_column():
    text = to_toon(SAMPLES, include_text=False)
    assert "text}" not in text.split("\n")[0]
    out = from_toon(text)
    assert len(out) == len(SAMPLES)
    for a, b in zip(SAMPLES, out):
        assert a.chunk_id == b.chunk_id
        assert b.text == ""            # the column is gone, so the field is empty


def test_empty_evidence_round_trips():
    assert from_toon(to_toon([])) == []
    assert from_json(to_json([])) == []


def test_token_counting_uses_the_real_tokenizer():
    text = to_toon(SAMPLES)
    n = count_tokens(text)
    assert n > 0
    assert n < len(text)          # more tokens than words, fewer than chars
    assert count_tokens("") == 0


def test_toon_is_not_longer_than_json_on_the_metadata():
    toon = to_toon(SAMPLES, include_text=False)
    js = to_json(SAMPLES, include_text=False)
    assert count_tokens(toon) < count_tokens(js)
