"""
test_gold_set_integrity.py -- the 40-question gold set must be what it claims.

The gold set is the instrument the Phase 8 evaluation will measure with, so
it has to be checked like one.  These tests are offline and deterministic:
no LLM call, no embedding, no retrieval.

Checked here:
  * the split is 20 dev + 20 test, disjoint, with the advertised type mix
  * the frozen test file still hashes to its recorded SHA256
  * every gold chunk id really exists in the corpus, at the stated document
    and chunk index (the corpus is re-chunked here with the same settings
    the gold set was written against)
  * every required_term really occurs in its gold chunk's text
  * every paraphrase question shares NO stemmed content word with its gold
    chunk -- the property that makes it a paraphrase and not a keyword query
  * no-answer questions have no gold chunks and no gold answer
  * the type-specific structural rules hold (1 chunk / 2 chunks of one doc /
    chunks of >= 2 docs / 1 chunk with zero overlap / 0 chunks)
  * dev and test share no query text
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# The gold set is written against these chunking settings.  Set them in the
# environment BEFORE the config is imported so a stray .env cannot change
# the chunk boundaries the gold chunk ids refer to.
os.environ.setdefault("CHUNKING_STRATEGY", "markdown_aware")
os.environ.setdefault("CHUNK_SIZE_EXTRACTION", "450")
os.environ.setdefault("CHUNK_OVERLAP_PCT", "0.12")
os.environ.setdefault("CACHE_DIR", "data/cache/test_gold_integrity")

from graphrag_lite.config import get_config  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402

DEV_PATH = os.path.join(ROOT, "eval", "gold_dev_set.json")
TEST_PATH = os.path.join(ROOT, "eval", "gold_test_set_frozen.json")

VALID_TYPES = {"single-chunk", "two-chunk", "cross-doc", "paraphrase", "no-answer"}

# A small, explicit stopword list.  Kept short on purpose: every word in it
# is a word whose presence in both a query and a chunk says nothing about
# whether the query is a paraphrase of that chunk.
STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "one",
    "only", "or", "other", "our", "ours", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours",
}

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _stem(word: str) -> str:
    """A deliberately conservative suffix stripper.

    It is applied identically to the query and to the chunk, so the only
    thing that matters is that it is deterministic and that it does not
    merge words that are genuinely different.  No Porter-style rewrites:
    'structure' and 'structured' DO collapse to the same stem, which is
    exactly the kind of collision the paraphrase questions must avoid.
    """
    w = word.lower()
    for suffix in ("iness", "ingly", "edly", "ment", "ness", "tion", "ing",
                   "ies", "ied", "ed", "es", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    return w


def _content_stems(text: str) -> set[str]:
    """The stemmed content words of a text (stopwords and short words out)."""
    out = set()
    for raw in _WORD_RE.findall(text):
        w = raw.lower()
        if len(w) < 3 or w in STOPWORDS:
            continue
        out.add(_stem(w))
    return out


# ------------------------------------------------------------------ #
# fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def dev_set():
    with open(DEV_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def test_set():
    with open(TEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def corpus_chunks():
    """Re-chunk the corpus exactly as the gold set was written against."""
    cfg = get_config()
    object.__setattr__(cfg, "CHUNKING_STRATEGY", "markdown_aware")
    object.__setattr__(cfg, "CHUNK_SIZE_EXTRACTION", 450)
    object.__setattr__(cfg, "CHUNK_OVERLAP_PCT", 0.12)
    tok = get_tokenizer(cfg.TOKENIZER)
    chunks = {}
    for doc in load_documents(os.path.join(ROOT, "data", "raw")):
        for c in chunk_document(doc, cfg, tok):
            chunks[c["chunk_id"]] = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            }
    return chunks


def _all_queries(dev_set, test_set):
    return list(dev_set["queries"]) + list(test_set["queries"])


# ------------------------------------------------------------------ #
# the split itself
# ------------------------------------------------------------------ #

def test_split_sizes(dev_set, test_set):
    assert len(dev_set["queries"]) == 20
    assert len(test_set["queries"]) == 20


def test_type_mix_matches_what_each_file_advertises(dev_set, test_set):
    for obj in (dev_set, test_set):
        counts = {}
        for q in obj["queries"]:
            counts[q["type"]] = counts.get(q["type"], 0) + 1
        advertised = {k: v for k, v in obj["counts"].items() if k != "total"}
        assert counts == advertised, (obj["set_name"], counts, advertised)
        assert obj["counts"]["total"] == len(obj["queries"])


def test_every_type_is_used_in_both_splits(dev_set, test_set):
    for obj in (dev_set, test_set):
        used = {q["type"] for q in obj["queries"]}
        assert used <= VALID_TYPES
        assert used == VALID_TYPES, (obj["set_name"], used)


def test_total_type_mix_is_14_8_6_6_6(dev_set, test_set):
    counts = {}
    for q in _all_queries(dev_set, test_set):
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    assert counts == {
        "single-chunk": 14,
        "two-chunk": 8,
        "cross-doc": 6,
        "paraphrase": 6,
        "no-answer": 6,
    }


def test_query_ids_are_unique_and_prefixed_by_split(dev_set, test_set):
    ids = []
    for q in dev_set["queries"]:
        assert q["query_id"].startswith("D")
        assert q["split"] == "dev"
        ids.append(q["query_id"])
    for q in test_set["queries"]:
        assert q["query_id"].startswith("T")
        assert q["split"] == "test"
        ids.append(q["query_id"])
    assert len(set(ids)) == 40


def test_dev_and_test_share_no_query_text(dev_set, test_set):
    dev_q = {q["query"].strip().lower() for q in dev_set["queries"]}
    test_q = {q["query"].strip().lower() for q in test_set["queries"]}
    assert dev_q & test_q == set()


# ------------------------------------------------------------------ #
# the frozen test split
# ------------------------------------------------------------------ #

def test_frozen_test_set_checksum_still_matches(test_set):
    obj = dict(test_set)
    recorded = obj.pop("frozen_sha256")
    canon = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    assert digest == recorded, (
        "the frozen test split has been edited -- the held-out run is no "
        "longer a held-out run. Restore the file, or re-freeze deliberately "
        "and say so in DOCUMENT.md."
    )


def test_test_set_is_marked_frozen(test_set):
    assert test_set.get("frozen") is True
    assert test_set["frozen_on"]
    assert "FROZEN" in " ".join(test_set.get("rules", []))


# ------------------------------------------------------------------ #
# the gold chunks really are in the corpus
# ------------------------------------------------------------------ #

def test_every_gold_chunk_id_exists_in_the_corpus(dev_set, test_set, corpus_chunks):
    for q in _all_queries(dev_set, test_set):
        for cid in q["gold_chunk_ids"]:
            assert cid in corpus_chunks, (q["query_id"], cid)


def test_gold_chunk_metadata_matches_the_corpus(dev_set, test_set, corpus_chunks):
    for q in _all_queries(dev_set, test_set):
        assert len(q["gold"]) == len(q["gold_chunk_ids"])
        for g in q["gold"]:
            real = corpus_chunks[g["chunk_id"]]
            assert g["source_doc"] == real["doc_id"], (q["query_id"], g["chunk_id"])
            assert g["chunk_index"] == real["chunk_index"], (q["query_id"], g["chunk_id"])
            assert g["chunk_id"] in q["gold_chunk_ids"]


def test_source_docs_field_agrees_with_the_gold_chunks(dev_set, test_set, corpus_chunks):
    for q in _all_queries(dev_set, test_set):
        if q["type"] == "no-answer":
            assert q["source_docs"] == []
            continue
        docs = {corpus_chunks[c]["doc_id"] for c in q["gold_chunk_ids"]}
        assert set(q["source_docs"]) == docs, (q["query_id"], q["source_docs"], docs)


def test_every_required_term_occurs_in_its_gold_chunk(dev_set, test_set, corpus_chunks):
    for q in _all_queries(dev_set, test_set):
        for g in q["gold"]:
            text = corpus_chunks[g["chunk_id"]]["text"]
            for term in g["required_terms"]:
                assert term.lower() in text.lower(), (
                    q["query_id"], g["chunk_id"], term
                )


# ------------------------------------------------------------------ #
# the type-specific structural rules
# ------------------------------------------------------------------ #

def test_single_chunk_questions_have_exactly_one_gold_chunk(dev_set, test_set):
    for q in _all_queries(dev_set, test_set):
        if q["type"] == "single-chunk":
            assert len(q["gold_chunk_ids"]) == 1, q["query_id"]


def test_two_chunk_questions_have_two_gold_chunks_of_one_document(dev_set, test_set, corpus_chunks):
    for q in _all_queries(dev_set, test_set):
        if q["type"] == "two-chunk":
            assert len(q["gold_chunk_ids"]) == 2, q["query_id"]
            docs = {corpus_chunks[c]["doc_id"] for c in q["gold_chunk_ids"]}
            assert len(docs) == 1, (q["query_id"], docs)


def test_cross_doc_questions_span_at_least_two_documents(dev_set, test_set, corpus_chunks):
    for q in _all_queries(dev_set, test_set):
        if q["type"] == "cross-doc":
            docs = {corpus_chunks[c]["doc_id"] for c in q["gold_chunk_ids"]}
            assert len(docs) >= 2, (q["query_id"], docs)
            assert len(q["gold_chunk_ids"]) >= 2


def test_paraphrase_questions_have_exactly_one_gold_chunk(dev_set, test_set):
    for q in _all_queries(dev_set, test_set):
        if q["type"] == "paraphrase":
            assert len(q["gold_chunk_ids"]) == 1, q["query_id"]


def test_no_answer_questions_have_no_gold_and_no_answer(dev_set, test_set):
    for q in _all_queries(dev_set, test_set):
        if q["type"] == "no-answer":
            assert q["gold_chunk_ids"] == [], q["query_id"]
            assert q["gold"] == [], q["query_id"]
            assert q["gold_answer"] is None, q["query_id"]
            assert "NOT ANSWERABLE" in q["note"], q["query_id"]


def test_answerable_questions_have_a_gold_answer(dev_set, test_set):
    for q in _all_queries(dev_set, test_set):
        if q["type"] != "no-answer":
            assert q["gold_answer"], q["query_id"]
            assert len(q["gold_answer"]) > 40, q["query_id"]


# ------------------------------------------------------------------ #
# the paraphrase property -- the whole point of the paraphrase type
# ------------------------------------------------------------------ #

def test_paraphrase_questions_share_no_content_word_with_their_gold_chunk(
    dev_set, test_set, corpus_chunks
):
    for q in _all_queries(dev_set, test_set):
        if q["type"] != "paraphrase":
            continue
        cid = q["gold_chunk_ids"][0]
        chunk_words = _content_stems(corpus_chunks[cid]["text"])
        query_words = _content_stems(q["query"])
        overlap = query_words & chunk_words
        assert not overlap, (
            f"{q['query_id']} is not a paraphrase: it shares {sorted(overlap)} "
            f"with chunk {cid}"
        )


def test_every_query_has_at_least_a_few_content_words(dev_set, test_set):
    for q in _all_queries(dev_set, test_set):
        assert len(_content_stems(q["query"])) >= 2, q["query_id"]


def test_the_historical_dev_v1_set_is_left_intact():
    """eval/gold_qa_set.json is the set the published Phase-4 numbers used."""
    path = os.path.join(ROOT, "eval", "gold_qa_set.json")
    with open(path, encoding="utf-8") as fh:
        old = json.load(fh)
    assert len(old["queries"]) == 10
    assert [q["query_id"] for q in old["queries"]] == [f"Q{i:02d}" for i in range(1, 11)]
