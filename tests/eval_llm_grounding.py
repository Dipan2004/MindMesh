"""
eval_llm_grounding.py -- Real-run verification of the AnswerGenerator.

Tests 6 dev questions from the gold set:
- 2 single-chunk
- 1 cross-doc
- 1 paraphrase
- 2 no-answer
"""
from __future__ import annotations

import time
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config
from graphrag_lite.llm.client_factory import get_llm_client
from graphrag_lite.llm.generator import AnswerGenerator
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.ingestion.loader import load_documents
from graphrag_lite.ingestion.chunker import chunk_document, get_tokenizer

def get_corpus_chunks():
    """Re-chunk the corpus to get a lookup of chunk_id -> text."""
    cfg = get_config()
    tok = get_tokenizer(cfg.TOKENIZER)
    chunks = {}
    for doc in load_documents("data/raw"):
        for c in chunk_document(doc, cfg, tok):
            chunks[c["chunk_id"]] = c["text"]
    return chunks

def run_eval():
    cfg = get_config()

    # Task 5.2 Requirements: NP=512, T=0, Seed=42
    ans_client = OllamaClient(
        model=cfg.SLM_MODEL,
        temperature=0.0,
        seed=42,
        num_predict=512,
    )

    gen = AnswerGenerator(ans_client)
    corpus = get_corpus_chunks()

    with open("eval/gold_dev_set.json", "r", encoding="utf-8") as f:
        dev_set = json.load(f)

    # Select 6 questions
    queries_to_run = []
    queries_to_run.extend([q for q in dev_set["queries"] if q["type"] == "single-chunk"][:2])
    queries_to_run.extend([q for q in dev_set["queries"] if q["type"] == "cross-doc"][:1])
    queries_to_run.extend([q for q in dev_set["queries"] if q["type"] == "paraphrase"][:1])
    queries_to_run.extend([q for q in dev_set["queries"] if q["type"] == "no-answer"][:2])

    print(f"Running LLM Grounding Eval on {len(queries_to_run)} questions...\n")
    print(f"Model: {ans_client.model} | NP: {ans_client.num_predict} | T: {ans_client.temperature}\n")

    for q in queries_to_run:
        # Prepare evidence chunks (real texts from corpus)
        evidence_chunks = []
        for cid in q["gold_chunk_ids"]:
            if cid in corpus:
                evidence_chunks.append({"chunk_id": cid, "text": corpus[cid]})

        # Prompt size check
        prompt = gen._assemble_prompt(q["query"], [c["text"] for c in evidence_chunks])
        prompt_size = len(prompt)

        start_time = time.time()
        response = gen.generate_answer(q["query"], evidence_chunks)
        duration = time.time() - start_time

        # Citation Validity Check (Format check only)
        # Valid if every cited ID in response was actually in the provided evidence
        cited_ids = response.cited_evidence_ids
        valid_ids = [c["chunk_id"] for c in evidence_chunks]
        is_valid = all(cid in valid_ids for cid in cited_ids)

        # Refusal check
        refused = "I cannot answer this question" in response.text

        print("-" * 40)
        print(f"Query: {q['query']}")
        print(f"Type: {q['type']}")
        print(f"Prompt Size: {prompt_size} chars")
        print(f"Answer: {response.text}")
        print(f"Citations: {cited_ids}")
        print(f"Citation Validity: {'PASS' if is_valid else 'FAIL'}")
        print(f"Refused: {refused}")
        print(f"Latency: {duration:.2f}s")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    run_eval()
