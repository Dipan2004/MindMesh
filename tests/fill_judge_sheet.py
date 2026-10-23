"""
fill_judge_sheet.py -- Runs the full GraphRAG-lite pipeline on the dev set
and populates the model_answer column in the blind judge sheet.
"""
from __future__ import annotations

import json
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag_lite.config import get_config
from graphrag_lite.llm.client_factory import get_llm_client
from graphrag_lite.llm.generator import AnswerGenerator
from graphrag_lite.retrieval.pipeline_a import retrieve_flat

def run_full_pipeline():
    cfg = get_config()

    # Setup LLM clients
    llm_client = get_llm_client("local")
    gen = AnswerGenerator(llm_client)

    with open("eval/gold_dev_set.json", "r", encoding="utf-8") as f:
        dev_set = json.load(f)

    queries = dev_set["queries"]
    results = {}

    print(f"Running full pipeline for {len(queries)} queries...")

    for q in queries:
        query_text = q["query"]
        print(f"Processing {q['query_id']}: {query_text[:50]}...")

        # 1. Retrieval
        # PipelineA.retrieve returns a list of Evidence objects
        evidence_objs = retrieve_flat(query_text)

        # Convert Evidence objects to the dict format AnswerGenerator expects
        evidence_chunks = [
            {"chunk_id": e.chunk_id, "text": e.text}
            for e in evidence_objs
        ]

        # 2. Generation
        answer_obj = gen.generate_answer(query_text, evidence_chunks)
        results[q["query_id"]] = answer_obj.text

    # 3. Update the CSV
    sheet_path = "eval/judge_sheet_blind.csv"
    with open(sheet_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        qid = row["query_id"]
        if qid in results:
            row["model_answer"] = results[qid]

    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully populated {sheet_path}")

if __name__ == "__main__":
    run_full_pipeline()
