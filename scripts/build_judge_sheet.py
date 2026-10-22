"""
build_judge_sheet.py -- build eval/judge_sheet.csv (step 3 of the 2026-10-22
work order).

The sheet holds 40 relationships sampled from the LADDER ARM
(data/cache/exp_reextract_ladder), stratified:

    A  "kept"      15  relationships the endpoint filter keeps today
                       (flag OFF) -- drawn from chunks the filter never
                       touches, i.e. chunks with no dangling endpoints.
    B  "recovered" 15  relationships the conditional endpoint filter
                       (step 2, EXTRACTION_STUB_ENDPOINTS=1) recovers
                       because their missing endpoint occurs verbatim in
                       the chunk text.
    C  "rescued"   10  relationships that were ALREADY kept inside the
                       12 chunks the conditional filter rescues -- i.e. the
                       pre-existing edges of the chunks that needed
                       rescuing.  (The "rescued chunks" stratum.  The two
                       chunks rescued by the RETRY LADDER -- 24b17cf3 and
                       4af3dc8c5 -- contribute only 1 usable relationship
                       between them, so they cannot fill a stratum of 10;
                       the stratum is therefore defined as "chunks the
                       step-2 filter rescues", which is the chunk-level
                       counterpart of stratum B's relationship-level view.)

Labels use the vocabulary already established in DOCUMENT.md
("Item 4 -- Relationship quality: 15 sampled relationships, judged"):

    REAL   true and supported by the chunk text
    WEAK   true in substance, but the cited sentence does not support the
           edge as stated (usually the relation type is a mis-typing:
           membership written as DEVELOPED_BY, "based on" written as
           DEVELOPED_BY, ...)
    NOISE  unsupported, inverted, or hallucinated

The `human_label` column is left BLANK for the annotator.  The
`fill_first` column marks the 20 rows sampled for the human pass:
random.Random(20261022), stratified 7 / 7 / 6 over A / B / C, so the
annotated subsample is representative of the sheet (and therefore of the
strata) instead of being biased towards rows the agent is unsure about.

READ-ONLY against the ladder arm's cache.  Writes only eval/judge_sheet.csv.

Run:  python scripts/build_judge_sheet.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE_DIR = "data/cache/exp_reextract_ladder"
RECORDS = "data/cache/exp_reextract_ladder.jsonl"

os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"
os.environ["CHUNKING_STRATEGY"] = "markdown_aware"
os.environ["CHUNK_SIZE_EXTRACTION"] = "450"
os.environ["EXTRACTION_MODEL"] = "qwen2.5:1.5b"
os.environ["LLM_NUM_PREDICT"] = "4096"
os.environ["LLM_NUM_CTX"] = "8192"
os.environ["LLM_THINK"] = "off"
os.environ["EXTRACTION_STUB_ENDPOINTS"] = "0"

import diskcache  # noqa: E402

from graphrag_lite.config import get_config  # noqa: E402

get_config.cache_clear()

from graphrag_lite.cache.llm_cache import _make_key  # noqa: E402
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer  # noqa: E402
from graphrag_lite.ingestion.extractor import (  # noqa: E402
    _EXTRACTION_PROMPT_TEMPLATE,
    _extract_json,
    _parse_extraction,
)
from graphrag_lite.ingestion.loader import load_documents  # noqa: E402

PENALTIES = (0.25, 0.5, 1.0)
SEED = 20261022

# ------------------------------------------------------------------ #
# The 40 selected relationships.
# (stratum, chunk prefix, source, target, type, agent label, why)
# ------------------------------------------------------------------ #
SELECTION: list[tuple[str, str, str, str, str, str, str]] = [
    # ---------------- A: kept by the filter today ------------------- #
    ("A", "b76780cc", "Freebase", "Google", "ACQUIRED_BY", "REAL",
     "text: 'acquired by Google in 2010'"),
    ("A", "b76780cc", "Wikidata", "Google", "MIGRATED_TO", "NOISE",
     "text: Google migrated its data TO Wikidata; the edge is inverted"),
    ("A", "a04b3d90", "Cyc",
     "Microelectronics and Computer Technology Corporation (MCC)",
     "FOUNDED_BY", "REAL",
     "text: 'started in 1984 by Doug Lenat at ... MCC'"),
    ("A", "a04b3d90", "OpenCyc",
     "Microelectronics and Computer Technology Corporation (MCC)",
     "ACQUIRED_BY", "NOISE",
     "the chunk says nothing about an acquisition of OpenCyc"),
    ("A", "c3893b32", "Anthropic", "Claude Series", "DEVELOPED_BY", "REAL",
     "text: 'Anthropic developed the Claude series of LLMs'"),
    ("A", "c3893b32", "Google DeepMind", "Gemini", "DEVELOPED_BY", "REAL",
     "text: 'Google DeepMind developed Gemini'"),
    ("A", "32c85d7e", "PPO", "LLM", "USED_FOR", "REAL",
     "text: 'The LLM is then fine-tuned using PPO'"),
    ("A", "32c85d7e", "LLM", "RLHF", "INTRODUCED_BY", "NOISE",
     "RLHF aligns an LLM that already exists; it did not introduce it"),
    ("A", "0e82efa9", "Claude 3.5", "context window", "SUPPORTS", "REAL",
     "text: 'Claude 3.5 supports 200,000 tokens'"),
    ("A", "0e82efa9", "GPT-3", "context window", "DETERMINES", "NOISE",
     "text: GPT-3 was LIMITED TO 2048-token context windows; inverted"),
    ("A", "ab2692f1", "Knowledge Graph", "Birth Date", "USED_FOR", "REAL",
     "text: facts '(birth date, ...) drawn directly from the knowledge graph'"),
    ("A", "ab2692f1", "Knowledge Graph", "Nationality", "USED_FOR", "REAL",
     "text: facts '(birth date, nationality, ...) drawn ... from the knowledge graph'"),
    ("A", "4dcff615", "generator", "BART", "USED_FOR", "NOISE",
     "text: 'BART was used as the generator'; the edge is inverted"),
    ("A", "efaf36ff", "LLM", "Transformer architecture", "DEVELOPED_BY", "WEAK",
     "text: 'LLMs are based on the Transformer architecture' -- 'based on' "
     "is not 'developed by'"),
    ("A", "025a4d30", "Weaviate", "Pinecone", "SUCCESSOR_OF", "NOISE",
     "the chunk describes the two products side by side; no succession"),
    # ---------------- B: recovered by the step-2 filter ------------ #
    ("B", "3256abe3", "Wikidata", "Wikipedia", "PART_OF", "NOISE",
     "text: Wikidata serves as a storage hub across Wikimedia projects, "
     "'including Wikipedia' -- siblings, not part of"),
    ("B", "e9c66a7a", "LLMs", "next-token prediction", "DEVELOPED_BY", "NOISE",
     "next-token prediction is an objective, not a developer"),
    ("B", "5c56e059", "GPT Series (OpenAI)", "OpenAI", "DEVELOPED_BY", "REAL",
     "text: 'OpenAI developed the GPT series of models'"),
    ("B", "5c56e059", "GPT-1", "GPT Series (OpenAI)", "DEVELOPED_BY", "WEAK",
     "GPT-1 is a member of the series; the cited sentence does not say it "
     "was developed BY the series"),
    ("B", "5c56e059", "ChatGPT", "GPT Series (OpenAI)", "BUILT_ON", "REAL",
     "text: 'ChatGPT, built on the GPT-3.5 and GPT-4 families'"),
    ("B", "73e97519", "Qwen", "Alibaba Cloud", "DEVELOPED_BY", "REAL",
     "text: 'Alibaba Cloud developed the Qwen series of LLMs'"),
    ("B", "73e97519", "Qwen", "Qwen3 family", "INTRODUCED", "WEAK",
     "membership / succession, not an introduction by the Qwen line"),
    ("B", "310582c4", "MMLU", "academic subjects", "TESTS", "REAL",
     "text: 'MMLU ... tests knowledge across 57 academic subjects'"),
    ("B", "310582c4", "HumanEval", "code generation accuracy", "MEASURES", "REAL",
     "text: 'HumanEval and MBPP measure code generation accuracy'"),
    ("B", "310582c4", "TruthfulQA", "factually incorrect statements",
     "MEASURES", "REAL",
     "text: 'TruthfulQA measures the tendency to generate factually "
     "incorrect statements'"),
    ("B", "e146728c", "FAISS", "Vector store", "ACQUIRED_BY", "NOISE",
     "text lists FAISS as a choice of vector store; it acquires nothing"),
    ("B", "b744eeeb", "Graph RAG", "Microsoft's GraphRAG system", "INTEGRATED",
     "WEAK",
     "text: Microsoft's GraphRAG is 'a prominent example' of Graph RAG -- "
     "an instance, not an integration"),
    ("B", "a4a679c4", "Patrick Lewis", "Microsoft Research", "ACQUIRED_BY",
     "NOISE",
     "text: the RAG paper was authored at Facebook AI Research; no "
     "acquisition by Microsoft Research"),
    ("B", "a4a679c4", "Darren Edge", "Microsoft Research", "CONTRIBUTED_TO",
     "REAL",
     "text: 'Microsoft Research developed GraphRAG, with contributions from "
     "Darren Edge, ...'"),
    ("B", "5a5f4b8a", "Zvec", "Microsoft GraphRAG", "SUCCESSOR_OF", "NOISE",
     "the chunk only says Microsoft GraphRAG uses LanceDB; Zvec is a "
     "separate project"),
    # ---------------- C: pre-existing edges of rescued chunks ------- #
    ("C", "3256abe3", "Schema.org", "Wikidata", "USES", "NOISE",
     "text: Schema.org is used for web markup; it does not use Wikidata"),
    ("C", "3256abe3", "DBpedia", "Wikidata", "EXTRACTED_FROM", "NOISE",
     "text: DBpedia is 'extracted from Wikipedia infoboxes' -- wrong target"),
    ("C", "3256abe3", "SNOMED CT", "Wikidata", "USED_FOR", "NOISE",
     "text: SNOMED CT is a healthcare ontology; it is not used for Wikidata"),
    ("C", "e9c66a7a", "BERT", "masked language modeling", "DEVELOPED_BY",
     "WEAK",
     "text: BERT USES a masked language modeling objective; 'developed by' "
     "is a mis-typing"),
    ("C", "310582c4", "LLMs", "MMLU", "DEVELOPED_BY", "NOISE",
     "MMLU evaluates LLMs; it does not develop them"),
    ("C", "e146728c", "Dense Retriever", "Dense Passage Retriever (DPR)",
     "DEVELOPED_BY", "NOISE",
     "DPR IS the dense retriever; it was developed by Facebook AI Research"),
    ("C", "b744eeeb", "Advanced RAG", "Graph RAG", "INTEGRATED", "NOISE",
     "the two are sibling techniques; no integration is stated"),
    ("C", "d6426773", "Vector Database", "Relational Database", "INTRODUCED_BY",
     "NOISE",
     "text: 'Unlike traditional relational databases ...' -- a contrast, "
     "not a derivation"),
    ("C", "417905a7", "embedding", "encoder model", "GENERATED_BY", "REAL",
     "text: 'Embeddings are generated by passing text ... through a trained "
     "encoder model'"),
    ("C", "5a5f4b8a", "LanceDB", "Zvec", "LOCATED_IN", "NOISE",
     "the chunk has two sibling sections; LanceDB is not located in Zvec"),
]

COLUMNS = [
    "chunk_id", "source", "target", "type", "source_sentence",
    "agent_label", "human_label",
    "doc_id", "stratum", "fill_first", "agent_note", "chunk_text",
]


def tag(penalty: float) -> str:
    return f"t=0.0;seed=42;np=4096;ctx=8192;fp={penalty};think=False"


def main() -> None:
    warnings.simplefilter("ignore", UserWarning)

    cfg = get_config()
    model = cfg.effective_extraction_model()
    cache = diskcache.Cache(CACHE_DIR)

    tokenizer = get_tokenizer(cfg.TOKENIZER)
    chunks: list[dict] = []
    for doc in load_documents("data/raw"):
        chunks.extend(chunk_document(doc, cfg, tokenizer))
    by_id = {c["chunk_id"]: c for c in chunks}

    recs: dict[str, dict] = {}
    with (ROOT / RECORDS).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["chunk_id"]] = r

    # Parse every chunk twice (flag OFF / flag ON) and index the results.
    kept: dict[tuple, dict] = {}
    recovered: dict[tuple, dict] = {}
    for cid, rec in recs.items():
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        raw = None
        for pen in PENALTIES:
            candidate = cache.get(_make_key(prompt, model, tag(pen)), default=None)
            if candidate is None:
                continue
            try:
                _extract_json(candidate)
            except Exception:
                continue
            raw = candidate
            break
        if raw is None:
            continue
        obj = _extract_json(raw)
        off = _parse_extraction(obj, cid, rec["doc_id"],
                                chunk_text=chunk["text"],
                                allow_stub_endpoints=False)
        on = _parse_extraction(obj, cid, rec["doc_id"],
                               chunk_text=chunk["text"],
                               allow_stub_endpoints=True)
        off_keys = {(r["source"].lower(), r["target"].lower(), r["type"])
                    for r in off["relationships"]}
        for r in off["relationships"]:
            kept[(cid, r["source"].lower(), r["target"].lower(), r["type"])] = r
        for r in on["relationships"]:
            k = (cid, r["source"].lower(), r["target"].lower(), r["type"])
            if k not in off_keys:
                recovered[k] = r

    # ---- resolve the 40 selected rows ------------------------------------ #
    rows: list[dict] = []
    missing: list[str] = []
    for stratum, prefix, source, target, rtype, label, note in SELECTION:
        cid = next((c for c in by_id if c.startswith(prefix)), None)
        if cid is None:
            missing.append(f"{prefix} {source}->{target} (no chunk)")
            continue
        key = (cid, source.lower(), target.lower(), rtype)
        rel = kept.get(key) if stratum != "B" else recovered.get(key)
        if rel is None:
            missing.append(f"{stratum} {prefix} {source} --{rtype}--> {target}")
            continue
        rows.append({
            "chunk_id": cid,
            "source": rel["source"],
            "target": rel["target"],
            "type": rel["type"],
            "source_sentence": " ".join(str(rel["source_sentence"]).split()),
            "agent_label": label,
            "human_label": "",
            "doc_id": recs[cid]["doc_id"],
            "stratum": stratum,
            "fill_first": "",
            "agent_note": note,
            "chunk_text": " ".join(by_id[cid]["text"].split()),
        })

    if missing:
        print("[WARN] could not resolve:")
        for m in missing:
            print("   ", m)

    # ---- which 20 rows the human fills first ----------------------------- #
    rng = random.Random(SEED)
    quota = {"A": 7, "B": 7, "C": 6}
    for stratum, n in quota.items():
        pool = [i for i, r in enumerate(rows) if r["stratum"] == stratum]
        for i in rng.sample(pool, min(n, len(pool))):
            rows[i]["fill_first"] = "y"

    out = ROOT / "eval" / "judge_sheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"[judge_sheet] wrote {len(rows)} rows -> {out}")
    print("  strata      :", dict(Counter(r["stratum"] for r in rows)))
    print("  agent labels:", dict(Counter(r["agent_label"] for r in rows)))
    print("  fill_first  :", sum(1 for r in rows if r["fill_first"] == "y"))
    for stratum in ("A", "B", "C"):
        sub = [r for r in rows if r["stratum"] == stratum]
        print(f"  {stratum}: {dict(Counter(r['agent_label'] for r in sub))}")


if __name__ == "__main__":
    main()
