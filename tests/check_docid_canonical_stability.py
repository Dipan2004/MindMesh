"""
check_docid_canonical_stability.py -- prove the doc-ID sanitiser change did
not alter the IDs of any canonical entity.

The old rule replaced spaces and "/" with "_".  The new rule replaces every
character zvec rejects and truncates at 64 characters.  If the two agree on
every canonical entity name, the canonical collection is untouched by the
change.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag_lite.graph.builder import build_graph, load_extractions  # noqa: E402
from graphrag_lite.vectorstore.zvec_client import _to_doc_id  # noqa: E402


def old_to_doc_id(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def main() -> None:
    recs = load_extractions(str(ROOT / "data" / "cache" / "extractions.jsonl"))
    G = build_graph(recs)
    names = sorted(G.nodes())
    changed = []
    for n in names:
        old, new = old_to_doc_id(n), _to_doc_id(n)
        if old != new:
            changed.append((n, old, new))
    print(f"canonical entities: {len(names)}")
    print(f"doc IDs changed by the sanitiser: {len(changed)}")
    for n, o, w in changed:
        print(f"  {n!r}\n     old={o!r}\n     new={w!r}")
    print("CANONICAL IDS UNCHANGED" if not changed else "CANONICAL IDS WOULD CHANGE")


if __name__ == "__main__":
    main()
