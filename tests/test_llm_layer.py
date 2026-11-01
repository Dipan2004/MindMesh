"""
test_llm_layer.py -- Integration tests for the grounded AnswerGenerator.

This test requires a running Ollama instance with qwen2.5:1.5b.
It verifies that the generator adheres to the grounding and citation rules.
"""
from __future__ import annotations

import pytest
from graphrag_lite.config import get_config
from graphrag_lite.llm.ollama_client import OllamaClient
from graphrag_lite.llm.generator import AnswerGenerator

def test_grounded_answer_with_citations():
    """The model should use provided evidence and cite it using [index]."""
    cfg = get_config()
    client = OllamaClient(model=cfg.SLM_MODEL)
    gen = AnswerGenerator(client)

    query = "What is the relationship between Alpha and Beta?"
    evidence = [
        "Alpha is a high-performance computing cluster located in Zurich.",
        "Beta is the primary software layer that manages Alpha's workload.",
    ]

    response = gen.generate_answer(query, evidence)
    answer = response.answer

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")

    # Assertions
    assert "Alpha" in answer
    assert "Beta" in answer
    # Check for citation markers [0] or [1]
    assert any(marker in answer for marker in ["[0]", "[1]"]), \
        f"Answer should contain citations. Got: {answer}"

def test_refusal_on_irrelevant_evidence():
    """The model should refuse to answer if the evidence is irrelevant."""
    cfg = get_config()
    client = OllamaClient(model=cfg.SLM_MODEL)
    gen = AnswerGenerator(client)

    query = "What is the capital of France?"
    evidence = [
        "The internal combustion engine was patented by Nikolaus Otto.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    ]

    response = gen.generate_answer(query, evidence)
    answer = response.answer

    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")

    # It should use the exact refusal phrase from the system prompt
    expected_refusal = "I cannot answer this question based on the provided evidence."
    assert expected_refusal.lower() in answer.lower(), \
        f"Expected refusal phrase not found. Got: {answer}"

def test_empty_evidence_handling():
    """The generator should handle empty evidence without crashing."""
    cfg = get_config()
    client = OllamaClient(model=cfg.SLM_MODEL)
    gen = AnswerGenerator(client)

    query = "Any question"
    evidence = []

    response = gen.generate_answer(query, evidence)

    assert "no relevant evidence was retrieved" in response.answer
