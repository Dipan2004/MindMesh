"""
exp_gold_set.py -- Write the gold set BEFORE either pipeline is run.

This script performs NO retrieval. It only:
  1. chunks the corpus,
  2. resolves every gold chunk_id prefix to a real chunk,
  3. prints that chunk's source doc, chunk index, heading and the
     answer-bearing text, so the gold can be checked against the source,
  4. writes eval/gold_qa_set.json with full chunk ids.

Running this to completion is the precondition for exp_pipeline_b_eval.py,
which reads the file this script writes. No pipeline is invoked here, so the
gold ids cannot have been tuned against retrieval output.

Isolation:
  * own CACHE_DIR  (data/cache/exp_gold_set) -- never the canonical cache
  * ZVEC_ROOT      data/cache (canonical collections; not used here)
  * own output file: eval/gold_qa_set.json
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_gold_set.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_gold_set"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"

from graphrag_lite.config import get_config        # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache   # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer   # noqa: E402
from graphrag_lite.ingestion.loader import load_documents   # noqa: E402

cfg = get_config()
cache = _get_cache()
tokenizer = get_tokenizer(cfg.TOKENIZER)

# ── The 10 queries and their gold chunks ───────────────────────────────────
# Gold ids were chosen by reading the source documents under data/raw/ and
# locating the passage that answers the query -- NOT by inspecting retrieval
# output. Each entry:
#   (query_id, query, {gold chunk_id prefixes}, note,
#    {chunk_id_prefix: [required substrings that must appear in that chunk]})
# The required substrings are a self-check: a gold chunk that does not
# contain them is wrong, and the script reports it as such.
#
# NOTE ON CORRECTIONS: the gold ids inherited from
# tests/test_pipeline_b_task42.py were wrong for Q02, Q04 and Q10 --
# they pointed at chunks that do not contain the answer (Q04's "Wikidata"
# gold was the "History and Origins" chunk; Q10's "DPO" gold was the
# "Fine-tuning" chunk; Q02's "Encoder-Decoder" gold was the Overview and
# Self-Attention chunks). Those three are corrected below.
GOLD = [
    ("Q01", "How does the self-attention mechanism compute attention scores?",
     {"c81c2a06"}, "transformer_architecture -- Self-Attention Mechanism section",
     {"c81c2a06": ["Self-Attention Mechanism", "Query (Q)", "attention score"]}),

    ("Q02", "What is the Encoder-Decoder structure of the original Transformer?",
     {"a885be6b"}, "transformer_architecture -- Encoder-Decoder Structure section",
     {"a885be6b": ["Encoder-Decoder Structure", "encoder", "decoder"]}),

    ("Q03", "How are entities and relationships extracted from text in GraphRAG?",
     {"6d878b7f"}, "knowledge_graphs -- Automated Extraction section",
     {"6d878b7f": ["Automated Extraction", "named entity recognition"]}),

    ("Q04", "What is Wikidata and who maintains it?",
     {"3256abe3"}, "knowledge_graphs -- Wikidata subsection (Wikimedia Foundation)",
     {"3256abe3": ["Wikidata", "Wikimedia Foundation"]}),

    ("Q05", "What are knowledge graph embedding methods like TransE and RotatE?",
     {"4e802a79"}, "knowledge_graphs -- Knowledge Graph Embeddings section",
     {"4e802a79": ["TransE", "RotatE"]}),

    ("Q06", "What makes Zvec different from FAISS for hybrid search?",
     {"895ee8b4", "5a5f4b8a", "d49b99f9"},
     "vector_databases -- Hybrid Search, Zvec section, FAISS section",
     {"895ee8b4": ["Hybrid Search"],
      "5a5f4b8a": ["Zvec"],
      "d49b99f9": ["FAISS"]}),

    ("Q07", "What is Retrieval-Augmented Generation and why does it matter?",
     {"33d32892"}, "retrieval_augmented_generation -- RAG overview",
     {"33d32892": ["Retrieval-Augmented Generation (RAG) is"]}),

    ("Q08", "How does HyDE improve retrieval quality in RAG systems?",
     {"b744eeeb"}, "retrieval_augmented_generation -- Advanced RAG (HyDE)",
     {"b744eeeb": ["HyDE"]}),

    ("Q09", "What is HNSW and how does it speed up approximate nearest neighbor search?",
     {"417905a7"}, "vector_databases -- Approximate Nearest Neighbor Search section",
     {"417905a7": ["HNSW"]}),

    ("Q10", "What is Direct Preference Optimization and how does it differ from RLHF?",
     {"5c56e059"}, "large_language_models -- Direct Preference Optimization section",
     {"5c56e059": ["Direct Preference Optimization", "RLHF"]}),
]

print("=" * 72)
print("EXP: write the gold set (no retrieval performed)")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print(f"ZVEC_ROOT              : {cfg.ZVEC_ROOT}")
print("=" * 72)
print()

docs = load_documents("data/raw")
chunks = []
for d in sorted(docs, key=lambda x: x["doc_id"]):
    chunks.extend(chunk_document(d, cfg, tokenizer))
by_prefix = {}
for c in chunks:
    by_prefix.setdefault(c["chunk_id"][:8], []).append(c)

print(f"Corpus chunks: {len(chunks)}")
print()


def heading_of(text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s
    return "(no heading)"


def snippet_of(text, n=180):
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
    body = " ".join(body.split())
    return body[:n]


out_queries = []
missing = []
bad_terms = []
for qid, query, prefixes, note, required in GOLD:
    print("-" * 72)
    print(f"{qid}: {query}")
    print(f"  note: {note}")
    gold_entries = []
    for pfx in sorted(prefixes):
        matches = by_prefix.get(pfx, [])
        if not matches:
            missing.append((qid, pfx))
            print(f"  !! {pfx}  -- NOT FOUND in corpus")
            continue
        for c in matches:
            # self-check: the gold chunk must actually contain the answer terms
            need = required.get(pfx, [])
            absent = [t for t in need if t.lower() not in c["text"].lower()]
            ok = not absent
            if absent:
                bad_terms.append((qid, pfx, absent))
            print(f"  gold {pfx}  -> {c['chunk_id']}   "
                  f"contains_answer_terms={ok}")
            print(f"      doc={c['doc_id']}  chunk_index={c['chunk_index']}  "
                  f"tokens={c['token_count']}")
            print(f"      heading: {heading_of(c['text'])}")
            if absent:
                print(f"      !! MISSING terms: {absent}")
            print(f"      text   : {snippet_of(c['text'])}")
            gold_entries.append({
                "chunk_id": c["chunk_id"],
                "chunk_id_prefix": pfx,
                "source_doc": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "heading": heading_of(c["text"]),
                "required_terms": need,
                "required_terms_present": ok,
                "evidence_snippet": snippet_of(c["text"], 300),
            })
    out_queries.append({
        "query_id": qid,
        "query": query,
        "note": note,
        "gold_chunk_ids": [g["chunk_id"] for g in gold_entries],
        "gold": gold_entries,
    })
    print()

print("=" * 72)
if missing:
    print(f"MISSING GOLD PREFIXES: {missing}")
else:
    print("All gold prefixes resolved to real corpus chunks.")
if bad_terms:
    print(f"GOLD CHUNKS NOT CONTAINING THE ANSWER TERMS: {bad_terms}")
else:
    print("Every gold chunk contains the required answer terms.")
print("=" * 72)

os.makedirs("eval", exist_ok=True)
OUT = "eval/gold_qa_set.json"
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({
        "created_by": "tests/exp_gold_set.py",
        "method": (
            "Gold chunk ids chosen by reading the source documents under "
            "data/raw/ and locating the passage that answers the query. "
            "Written before Pipeline A and Pipeline B were run. No retrieval "
            "output was consulted."
        ),
        "chunking_strategy": cfg.CHUNKING_STRATEGY,
        "chunk_size_extraction": cfg.CHUNK_SIZE_EXTRACTION,
        "queries": out_queries,
    }, fh, indent=2, ensure_ascii=False)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {OUT} ({len(out_queries)} queries, "
      f"{sum(len(q['gold_chunk_ids']) for q in out_queries)} gold chunks)")
print("EXP gold-set complete.")
