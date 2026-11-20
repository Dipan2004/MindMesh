"""
model_config.py -- MindMesh model assignments (one place, not hardcoded across the code).

    from mindmesh.model_config import load_model_assignments, think_for

`config/models.json` holds the default model for each role: extraction, generation, embedding.
Scripts use these only when the user does not pass an explicit model flag, so existing
commands keep their exact behaviour.

`think_for(model)` returns the Ollama `think` setting for an extraction model. Qwen3-family
models are reasoning models whose thinking text consumes the output budget, so extraction
runs them with think=False. Every other model keeps the configured default
(cfg.effective_think()), which leaves existing extraction caches and configurations unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = ROOT / "config" / "models.json"
_THINKING_FAMILIES = ("qwen3", "qwen3.5")


@dataclass(frozen=True)
class ModelAssignments:
    extraction: str
    generation: str
    embedding: str
    source: str


def load_model_assignments(path: Path | None = None) -> ModelAssignments:
    p = path or MODELS_FILE
    data = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in ("extraction", "generation", "embedding") if not data.get(k)]
    if missing:
        raise ValueError(f"{p}: missing model assignment(s): {', '.join(missing)}")
    return ModelAssignments(extraction=data["extraction"], generation=data["generation"],
                            embedding=data["embedding"], source=str(p))


def think_for(model: str):
    """Ollama `think` value for extraction with `model`."""
    if model.lower().startswith(_THINKING_FAMILIES):
        return False
    from graphrag_lite.config import get_config
    return get_config().effective_think()
