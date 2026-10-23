"""
generator.py -- Grounded answer generation using an LLM.

This module implements the 'Answer Layer' of the GraphRAG-lite pipeline.
It takes a user query and a set of retrieved evidence chunks, and
produces a grounded response where every claim is cited back to
the source evidence.
"""
from __future__ import annotations

import re
import time
from typing import Any, Protocol, Sequence
from dataclasses import dataclass

# Define the client protocol to keep the generator decoupled from Ollama
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...
    @property
    def model(self) -> str: ...

@dataclass
class Answer:
    """The final output of the LLM layer per PRD Task 5.2."""
    text: str
    cited_evidence_ids: list[str]
    confidence: float | None = None

class AnswerGenerator:
    """
    Grounded Answer Generator.

    Transforms retrieved evidence and a query into a cited response.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a precise knowledge assistant. Your goal is to answer the "
        "user's question using ONLY the provided evidence. Do not use "
        "external knowledge or your own training data.\n\n"
        "OUTPUT FORMAT:\n"
        "Every single fact must be on a new line starting with the evidence index in brackets.\n"
        "Format: [index] fact\n\n"
        "EXAMPLES:\n"
        "Query: What is the capital of France and its landmark?\n"
        "Answer:\n"
        "[0] The capital of France is Paris.\n"
        "[1] Paris is known for the Eiffel Tower.\n\n"
        "Query: Who founded the company?\n"
        "Answer:\n"
        "[2] The company was founded by Jane Doe.\n\n"
        "RULES:\n"
        "1. Every fact must start with [index].\n"
        "2. If the evidence does not contain enough information, state exactly: 'I cannot answer this question based on the provided evidence.'\n"
        "3. Do not include any introductory text, conversational filler, or conclusions."
    )

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def _assemble_prompt(self, query: str, evidence: Sequence[str]) -> str:
        """
        Assemble the final prompt for the LLM.
        """
        evidence_block = "\n\n".join(
            f"[{i}] {text}" for i, text in enumerate(evidence)
        )

        prompt = (
            f"SYSTEM: {self.system_prompt}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            f"USER QUERY: {query}\n\n"
            f"ANSWER:"
        )
        return prompt

    def _extract_citations(self, text: str, evidence_ids: Sequence[str]) -> list[str]:
        """
        Extract indices like [0], [1] from text and map them to real chunk_ids.
        """
        # Find all patterns like [0], [12]
        matches = re.findall(r"\[(\d+)\]", text)
        cited_ids = []
        for m in matches:
            idx = int(m)
            if 0 <= idx < len(evidence_ids):
                cited_ids.append(evidence_ids[idx])
        return list(dict.fromkeys(cited_ids)) # dedup preserve order

    def generate_answer(self, query: str, evidence_chunks: Sequence[dict]) -> Answer:
        """
        Generate a grounded answer.

        Args:
            query: The user's question.
            evidence_chunks: List of chunk dicts containing 'text' and 'chunk_id'.
        """
        if not evidence_chunks:
            return Answer(
                text="I cannot answer this question because no relevant evidence was retrieved.",
                cited_evidence_ids=[],
                confidence=None
            )

        # Extract texts and IDs for the prompt and the mapping
        evidence_texts = [c["text"] for c in evidence_chunks]
        evidence_ids = [c["chunk_id"] for c in evidence_chunks]

        prompt = self._assemble_prompt(query, evidence_texts)

        # The LLM client is configured with the Answer-layer settings (T=0, Seed=42, NP=512)
        raw_out = self.client.generate(prompt)

        answer_text = raw_out.strip()
        if answer_text.lower().startswith("answer:"):
            answer_text = answer_text[7:].strip()

        # Simple format recovery: "[0] fact" -> "fact [0]"
        lines = answer_text.splitlines()
        processed_lines = []
        for line in lines:
            line = line.strip()
            if not line: continue
            match = re.match(r"^\[(\d+)\]\s*(.*)", line)
            if match:
                idx = match.group(1)
                text = match.group(2)
                processed_lines.append(f"{text} [{idx}]")
            else:
                processed_lines.append(line)

        final_text = " ".join(processed_lines).strip()
        cited_ids = self._extract_citations(final_text, evidence_ids)

        return Answer(
            text=final_text,
            cited_evidence_ids=cited_ids,
            confidence=None
        )
