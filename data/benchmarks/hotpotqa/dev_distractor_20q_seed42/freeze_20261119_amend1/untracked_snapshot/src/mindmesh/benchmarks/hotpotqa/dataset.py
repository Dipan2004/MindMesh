"""
dataset.py -- HotpotQA (dev, distractor setting) ingestion for MindMesh.

Responsibilities
  * load the official dev-distractor split (official JSON or the HotpotQA
    organisation's Hugging Face parquet mirror -- identical content)
  * draw a seeded, type-stratified sample (default 10 bridge + 10 comparison)
  * build ONE shared candidate corpus from the sampled questions' supplied
    paragraphs, deduplicating identical paragraphs
  * assign stable document / chunk IDs (content hashes, independent of order)
  * keep paragraph titles and sentence boundaries
  * map each supporting fact (title, sentence_index) -> chunk_id

This is a "20-question shared candidate corpus", NOT the official FullWiki
setting: retrieval is over the union of the sampled questions' gold +
distractor paragraphs (~10 per question), not over all of Wikipedia.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

OFFICIAL_JSON_NAME = "hotpot_dev_distractor_v1.json"
HF_PARQUET_NAME = "hotpot_qa_distractor_validation.parquet"
OFFICIAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HF_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
          "distractor/validation-00000-of-00001.parquet")
EXPECTED_DEV_SIZE = 7405


# ── Loading ──────────────────────────────────────────────────────────────────

def _hf_row_to_official(row: dict) -> dict:
    """Convert a HF parquet row to the official HotpotQA JSON record shape."""
    sf = row["supporting_facts"]
    ctx = row["context"]
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": [[t, int(s)] for t, s in zip(sf["title"], sf["sent_id"])],
        "context": [[t, list(sents)] for t, sents in zip(ctx["title"], ctx["sentences"])],
    }


def load_dev_distractor(raw_dir: Path) -> tuple[list[dict], dict]:
    """
    Load the full dev-distractor split in official JSON record format.
    Returns (records, provenance).
    """
    raw_dir = Path(raw_dir)
    js = raw_dir / OFFICIAL_JSON_NAME
    pq_path = raw_dir / HF_PARQUET_NAME
    if js.exists():
        records = json.loads(js.read_text(encoding="utf-8"))
        prov = {"source": "official_json", "url": OFFICIAL_URL, "file": js.name}
        path = js
    elif pq_path.exists():
        import pyarrow.parquet as pq
        rows = pq.read_table(pq_path).to_pylist()
        records = [_hf_row_to_official(r) for r in rows]
        prov = {"source": "hf_parquet_mirror", "url": HF_URL, "file": pq_path.name}
        path = pq_path
    else:
        raise FileNotFoundError(
            f"HotpotQA dev distractor not found in {raw_dir}. Download either\n"
            f"  {OFFICIAL_URL}\n  -> {js}\nor\n  {HF_URL}\n  -> {pq_path}"
        )
    prov["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    prov["n_records"] = len(records)
    if len(records) != EXPECTED_DEV_SIZE:
        raise ValueError(f"Expected {EXPECTED_DEV_SIZE} dev records, got {len(records)}")
    return records, prov


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_questions(records: list[dict], n_bridge: int, n_comparison: int, seed: int) -> list[dict]:
    """Seeded stratified sample. Sorted by _id before sampling for stability."""
    rng = random.Random(seed)
    out = []
    for qtype, n in (("bridge", n_bridge), ("comparison", n_comparison)):
        pool = sorted((r for r in records if r["type"] == qtype), key=lambda r: r["_id"])
        out.extend(rng.sample(pool, n))
    return out


# ── Corpus ───────────────────────────────────────────────────────────────────

@dataclass
class Paragraph:
    doc_id: str
    chunk_id: str
    title: str
    sentences: list[str]
    text: str                       # title + body; what is embedded / extracted
    sentence_spans: list[list[int]] # char [start, end) of each sentence in `text`
    question_ids: list[str] = field(default_factory=list)


def _paragraph_key(title: str, sentences: list[str]) -> str:
    return hashlib.sha1(json.dumps([title, sentences], ensure_ascii=False).encode("utf-8")).hexdigest()


def make_paragraph(title: str, sentences: list[str]) -> Paragraph:
    key = _paragraph_key(title, sentences)
    doc_id = f"hp_{key[:16]}"
    chunk_id = hashlib.sha1(f"{doc_id}:0".encode()).hexdigest()[:16]
    header = f"{title}\n"
    text = header
    spans = []
    for s in sentences:
        start = len(text)
        text += s
        spans.append([start, len(text)])
    return Paragraph(doc_id=doc_id, chunk_id=chunk_id, title=title,
                     sentences=list(sentences), text=text, sentence_spans=spans)


def build_corpus(sample: list[dict]) -> tuple[list[Paragraph], dict]:
    """
    Union of all supplied paragraphs of the sampled questions, deduplicated by
    (title, sentences).  Returns (paragraphs sorted by doc_id, stats).
    """
    by_key: dict[str, Paragraph] = {}
    total = 0
    for q in sample:
        for title, sents in q["context"]:
            total += 1
            p = make_paragraph(title, sents)
            if p.doc_id not in by_key:
                by_key[p.doc_id] = p
            if q["_id"] not in by_key[p.doc_id].question_ids:
                by_key[p.doc_id].question_ids.append(q["_id"])
    paras = sorted(by_key.values(), key=lambda p: p.doc_id)
    titles = [p.title for p in paras]
    stats = {
        "paragraphs_supplied": total,
        "paragraphs_unique": len(paras),
        "duplicates_removed": total - len(paras),
        "unique_titles": len(set(titles)),
        "titles_with_multiple_versions": len(titles) - len(set(titles)),
        "sentences": sum(len(p.sentences) for p in paras),
        "documents": len(paras),
        "chunks": len(paras),
        "chunking": "one chunk per Wikipedia paragraph (HotpotQA paragraphs are short)",
    }
    return paras, stats


def map_supporting_facts(sample: list[dict], paras: list[Paragraph]) -> dict[str, list[dict]]:
    """
    For every question: [{title, sent_idx, chunk_id, doc_id, sentence}] for each
    supporting fact, resolved against that question's own context paragraph
    (titles are unique within one question's context).
    """
    by_doc = {p.doc_id: p for p in paras}
    out: dict[str, list[dict]] = {}
    for q in sample:
        ctx_doc = {}
        for title, sents in q["context"]:
            ctx_doc[title] = make_paragraph(title, sents).doc_id
        facts = []
        for title, sid in q["supporting_facts"]:
            doc_id = ctx_doc.get(title)
            p = by_doc.get(doc_id) if doc_id else None
            facts.append({
                "title": title,
                "sent_idx": int(sid),
                "doc_id": doc_id,
                "chunk_id": p.chunk_id if p else None,
                "sentence": p.sentences[sid] if p and 0 <= sid < len(p.sentences) else None,
            })
        out[q["_id"]] = facts
    return out


# ── Materialisation ──────────────────────────────────────────────────────────

def write_benchmark(bench_dir: Path, sample: list[dict], paras: list[Paragraph],
                    sp_map: dict, stats: dict, config: dict) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    questions = []
    for q in sample:
        facts = sp_map[q["_id"]]
        questions.append({
            "_id": q["_id"], "question": q["question"], "answer": q["answer"],
            "type": q["type"], "level": q["level"],
            "supporting_facts": q["supporting_facts"],
            "gold_chunk_ids": sorted({f["chunk_id"] for f in facts if f["chunk_id"]}),
            "context_chunk_ids": [make_paragraph(t, s).chunk_id for t, s in q["context"]],
        })
    (bench_dir / "questions.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    (bench_dir / "sample_ids.json").write_text(json.dumps(
        {"seed": config["seed"], "ids": [q["_id"] for q in sample],
         "types": {q["_id"]: q["type"] for q in sample}}, indent=2), encoding="utf-8")
    with open(bench_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for p in paras:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    (bench_dir / "supporting_fact_map.json").write_text(json.dumps(sp_map, indent=2, ensure_ascii=False), encoding="utf-8")
    (bench_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (bench_dir / "benchmark_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_benchmark(bench_dir: Path) -> dict[str, Any]:
    bench_dir = Path(bench_dir)
    paras = [Paragraph(**json.loads(l)) for l in (bench_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "questions": json.loads((bench_dir / "questions.json").read_text(encoding="utf-8")),
        "paragraphs": paras,
        "sp_map": json.loads((bench_dir / "supporting_fact_map.json").read_text(encoding="utf-8")),
        "config": json.loads((bench_dir / "benchmark_config.json").read_text(encoding="utf-8")),
        "stats": json.loads((bench_dir / "corpus_stats.json").read_text(encoding="utf-8")),
    }
