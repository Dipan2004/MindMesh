"""
test_zvec_doc_id.py -- the Zvec doc-ID sanitiser.

The native Zvec insert rejects a whole batch when a doc ID contains a
character it does not accept, or is longer than 64 characters.  The
accepted set was measured against the installed build by
tests/probe_zvec_docid.py:

    accepted : letters, digits, "_", "-", "."
    rejected : ":", ",", "(", ")", "/", " "
    length   : <= 64 characters

These are offline assertions on the mapping; no collection is created.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag_lite.vectorstore.zvec_client import _to_doc_id  # noqa: E402

ACCEPTED = re.compile(r"^[A-Za-z0-9_.\-]+$")
MAX_LEN = 64

NAMES = [
    "Simple",
    "with space",
    "with/slash",
    "Resource Description Framework (RDF)",
    "Subjective quality judgments beyond what automated benchmarks measure",
    "Retrieval-Augmented Generation (RAG)",
    "Qwen2.5:1.5b",
    "A,B;C",
    "MiXeD_Case-1.2",
    "x" * 200,
    "Vaswani et al.",
    "GPT-3",
    "1leading_digit",
]


def test_all_ids_are_accepted_by_the_native_rules():
    for name in NAMES:
        doc_id = _to_doc_id(name)
        assert ACCEPTED.match(doc_id), f"{name!r} -> {doc_id!r} has rejected characters"
        assert len(doc_id) <= MAX_LEN, f"{name!r} -> {doc_id!r} is {len(doc_id)} chars"


def test_id_is_never_empty():
    for name in NAMES:
        assert _to_doc_id(name)


def test_spaces_and_slashes_become_underscores():
    assert _to_doc_id("with space") == "with_space"
    assert _to_doc_id("with/slash") == "with_slash"


def test_rejected_characters_become_underscores():
    assert _to_doc_id("A,B;C") == "A_B_C"
    assert _to_doc_id("Qwen2.5:1.5b") == "Qwen2.5_1.5b"
    assert _to_doc_id("RAG (v2)") == "RAG__v2_"


def test_long_names_are_truncated_but_stay_unique():
    a = "a" * 100
    b = "b" * 100
    id_a, id_b = _to_doc_id(a), _to_doc_id(b)
    assert len(id_a) == MAX_LEN
    assert len(id_b) == MAX_LEN
    assert id_a != id_b


def test_canonical_names_keep_their_old_ids():
    # The old rule replaced only " " and "/".  Every canonical entity name
    # must produce the same ID under the old and the new rule, so the
    # canonical collection is untouched by the change.
    def old(name: str) -> str:
        return name.replace(" ", "_").replace("/", "_")

    for name in ["Vaswani et al.", "GPT-3", "Retrieval-Augmented Generation",
                 "Qdrant", "Pinecone", "Large Language Models"]:
        assert _to_doc_id(name) == old(name)
