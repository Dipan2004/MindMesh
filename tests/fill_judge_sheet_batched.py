"""
fill_judge_sheet_batched.py -- Runs the full GraphRAG-lite pipeline in batches
to avoid timeouts and ensure progress is saved.
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

def process_batch(queries, results_map, gen):
    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]
        print(f"Processing {qid}: {query_text[:50]}...")
        try:
            # 1. Retrieval
            evidence_objs = retrieve_flat(query_text)
            evidence_chunks = [{"chunk_id": e.chunk_id, "text": e.text} for e in evidence_objs]

            # 2. Generation
            answer_obj = gen.generate_answer(query_text, evidence_chunks)
            results_map[qid] = answer_obj.text
        except Exception as e:
            print(f"Error processing {qid}: {e}")
            results_map[qid] = f"ERROR: {str(e)}"

def run_batched():
    cfg = get_config()
    llm_client = get_llm_client("local")
    gen = AnswerGenerator(llm_client)

    with open("eval/gold_dev_set.json", "r", encoding="utf-8") as f:
        dev_set = json.load(f)

    all_queries = dev_set["queries"]
    sheet_path = "eval/judge_sheet_blind.csv"

    # Batch size
    BATCH_SIZE = 5

    for i in range(0, len(all_queries), BATCH_SIZE):
        batch = all_queries[i : i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({i} to {i+len(batch)}) ---")

        batch_results = {}
        process_batch(batch, batch_results, gen)

        # Update CSV immediately after each batch
        with open(sheet_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        for row in rows:
            qid = row["query_id"]
            if qid in batch_results:
                row["model_answer"] = batch_results[qid]

        with open(sheet_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Batch {i//BATCH_SIZE + 1} saved to {sheet_path}")

if __name__ == "__main__":
    run_batched()
