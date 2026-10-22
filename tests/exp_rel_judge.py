"""
exp_rel_judge.py -- Judge 15 random relationships: real vs noise, with type shares.

Reads the pilot extraction JSONL (the new exp_pilot_15b_glean1.jsonl when
present, otherwise the original pilot files), samples 15 relationships with
a fixed seed, and labels each REAL or NOISE.

Verdict rules (each NOISE case names its reason):
  * an endpoint is missing
  * the relationship is a self-loop (source == target)
  * an endpoint is a generic taxonomy term (Person, Organization, ...)
  * the relationship has no usable source_sentence (< 10 chars)
  * the type is the WORKS_AT fallback AND the source sentence does not
    mention either endpoint -- i.e. the type carries no support from the text
  * otherwise REAL

Note on WORKS_AT: it is NOT treated as noise on its own. A genuine
"Rafael Rafailov WORKS_AT Stanford University" is a real relationship even
though WORKS_AT is the type the extractor reaches for most often. It is only
judged noise when the source sentence does not actually support it.

Type shares are reported twice: over every relationship in the corpus, and
over the 15 sampled ones, split by verdict.

Isolation:
  * own CACHE_DIR  (data/cache/exp_rel_judge) -- never the canonical cache
  * own output file: data/cache/exp_rel_judge.json
  * prints CACHE_DIR and cache entry count at start

Run:  python tests/exp_rel_judge.py
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE_DIR = "data/cache/exp_rel_judge"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["CACHE_DIR"] = CACHE_DIR
os.environ["ZVEC_ROOT"] = "data/cache"

from graphrag_lite.config import get_config        # noqa: E402
get_config.cache_clear()
from graphrag_lite.cache.llm_cache import _get_cache   # noqa: E402

cfg = get_config()
cache = _get_cache()

print("=" * 72)
print("EXP: judge 15 random relationships (real vs noise) + type shares")
print("=" * 72)
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at start : {len(cache)}")
print("=" * 72)
print()

CANDIDATES = [
    "data/cache/exp_pilot_15b_glean1.jsonl",
    "data/cache/exp_pilot_15b_glean0.jsonl",
    "data/cache/extractions_pilot_1.5b_glean1.jsonl",
    "data/cache/extractions_pilot_1.5b_glean0.jsonl",
    "data/cache/extractions_pilot_1.5b.jsonl",
]
JSONL = next((p for p in CANDIDATES if os.path.exists(p)), None)
if JSONL is None:
    print("No pilot JSONL found -- run exp_pilot_15b.py first.")
    sys.exit(0)
print(f"Reading: {JSONL}")

records = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
clean = [r for r in records if "_parse_error" not in r]
print(f"Records: {len(records)}  clean: {len(clean)}")

all_rels = []
for r in clean:
    for rel in r.get("relationships", []):
        all_rels.append({**rel, "_doc": r["doc_id"], "_chunk": r["chunk_id"][:8]})

print(f"Total relationships: {len(all_rels)}")
print()
print("Type shares (ALL relationships):")
all_type_counts = Counter(r.get("type", "?") for r in all_rels)
for t, c in all_type_counts.most_common():
    print(f"  {t:<20} {c:>4}  ({100 * c / len(all_rels):.1f}%)")
print()

SCHEMA_TERMS = {
    "person", "organization", "location", "entity", "other", "technology",
    "event", "concept", "node", "edge", "relationship", "type", "class",
    "object", "thing",
}


def screen(rel):
    """
    Mechanical screen. Only flags what is objectively broken.

    It deliberately does NOT try to decide whether a relation is semantically
    true -- that is what the manual adjudication below is for. A mechanical
    rule that says "the sentence names neither endpoint" catches the
    obviously-unsupported extractions, but it cannot catch a relation whose
    endpoints are both named and whose *direction or meaning* is wrong
    (e.g. 'Dense Retriever' DEVELOPED_BY 'Dense Passage Retriever (DPR)').
    """
    src = (rel.get("source") or "").strip()
    tgt = (rel.get("target") or "").strip()
    rtype = (rel.get("type") or "").strip().upper()
    sent = (rel.get("source_sentence") or "").strip()
    sent_low = sent.lower()

    if not src or not tgt:
        return "NOISE", "missing endpoint"
    if src.lower() == tgt.lower():
        return "NOISE", "self-loop"
    if src.lower() in SCHEMA_TERMS or tgt.lower() in SCHEMA_TERMS:
        return "NOISE", "schema endpoint"
    if len(sent) < 10:
        return "NOISE", "no source sentence"
    if sent.lstrip().startswith("#"):
        return "WEAK", "source sentence is a markdown heading, not a claim"
    if src.lower() not in sent_low and tgt.lower() not in sent_low:
        return "NOISE", "source sentence names neither endpoint"
    return "REAL", ""


# Manual adjudication of the seeded 15-sample. Keyed by
# (source, target, type). The sample is drawn with random.seed(42) from a
# fixed source file, so this table is stable across runs and the script
# re-checks that every sampled relationship is either in this table or was
# screened mechanically.
#
# WEAK = the claim is true, but the source sentence cannot support it
# (e.g. the provenance is a markdown heading).
MANUAL = {
    ("Vector Index", "Chroma", "LOCATED_IN"): (
        "NOISE", "sentence is about vector-store latency and names neither "
                 "endpoint; direction is also backwards (the index lives in "
                 "the store, not the store in the index)"),
    ("Vector Index", "LanceDB", "LOCATED_IN"): (
        "NOISE", "sentence is about vector-store latency and names neither "
                 "endpoint; 'Vector Index' is a schema label, not an entity"),
    ("Vector Index", "FAISS", "LOCATED_IN"): (
        "NOISE", "sentence is about vector-store latency and names neither "
                 "endpoint"),
    ("RAG", "knowledge management", "WORKS_AT"): (
        "NOISE", "WORKS_AT fallback -- RAG is a technique, not a tenant; "
                 "sentence is about citations"),
    ("RAG", "knowledge management", "SUCCESSOR_OF"): (
        "NOISE", "unsupported -- RAG is not a successor of knowledge "
                 "management; sentence is about citations"),
    ("Transformer", "Sequence-to-sequence tasks", "APPLIES_TO"): (
        "WEAK", "claim is true (the original Transformer was built for "
                "seq2seq) but the only source sentence is the heading "
                "'# The Transformer Architecture'"),
    ("Multi-Head Attention", "Transformer", "USES"): (
        "WEAK", "claim is true (multi-head attention is a Transformer "
                "component) but the only source sentence is the heading "
                "'### Multi-Head Attention'"),
    ("Self-Attention Mechanism", "Softmax", "CALCULATED"): (
        "REAL", "attention weights do come from a softmax over the "
                "similarity scores; the source sentence supports it"),
    ("Self-Attention Mechanism", "Sequence", "OPERATES_ON"): (
        "REAL", "correct and supported by the quoted sentence"),
    ("Self-Attention Mechanism", "Token", "OPERATES_ON"): (
        "REAL", "correct and supported by the quoted sentence"),
    ("Dense Retriever", "Dense Passage Retriever (DPR)", "DEVELOPED_BY"): (
        "NOISE", "relation is inverted -- DPR *is* the dense retriever, it "
                 "was not developed by itself (it was developed by Facebook "
                 "AI Research)"),
    ("context compression", "LLM", "WORKS_AT"): (
        "NOISE", "WORKS_AT fallback -- context compression is a strategy, "
                 "not something that works at an LLM"),
    ("LMM", "LLM", "WORKS_AT"): (
        "NOISE", "WORKS_AT fallback; 'LMM' is not in the source sentence at "
                 "all and looks like a hallucinated variant of 'LLM'"),
    ("RAGAS", "custom LLM-as-judge", "USED_FOR"): (
        "NOISE", "the sentence lists RAGAS, TruLens and custom LLM-as-judge "
                 "as separate alternatives -- RAGAS is not 'used for' a "
                 "custom LLM-as-judge"),
    ("Chunking Strategy", "RAG", "DEVELOPED_BY"): (
        "NOISE", "unsupported -- the sentence is about decomposing the RAG "
                 "pipeline into modules, not about who developed the "
                 "chunking strategy"),
}


def judge(rel):
    """Final verdict: manual adjudication wins, mechanical screen otherwise."""
    key = (rel.get("source", ""), rel.get("target", ""),
           (rel.get("type") or "").strip().upper())
    if key in MANUAL:
        return MANUAL[key]
    return screen(rel)


random.seed(42)
sample = random.sample(all_rels, min(15, len(all_rels)))

print(f"15 sampled relationships (seed=42):")
print("-" * 72)
sample_verdicts = []
for i, r in enumerate(sample, 1):
    src = r.get("source", "?")
    tgt = r.get("target", "?")
    rtype = r.get("type", "?")
    sent = r.get("source_sentence", "")
    verdict, reason = judge(r)
    scr_verdict, scr_reason = screen(r)
    sample_verdicts.append((rtype, verdict))
    manual = (src, tgt, rtype) in MANUAL
    print(f"{i:2d}. [{r['_doc'][:26]}/{r['_chunk']}] {src!r} --{rtype}--> {tgt!r}")
    print(f"    sent: {sent[:110]!r}")
    print(f"    => {verdict}" + (f" ({reason})" if reason else "")
          + (f"   [screen said {scr_verdict}]" if manual else ""))
print()

real = sum(1 for _, v in sample_verdicts if v == "REAL")
weak = sum(1 for _, v in sample_verdicts if v == "WEAK")
noise = sum(1 for _, v in sample_verdicts if v == "NOISE")
screen_real = sum(1 for r in sample if screen(r)[0] == "REAL")

print("=" * 72)
print("VERDICT ON THE 15-CHUNK SAMPLE")
print("=" * 72)
print(f"  REAL  : {real}/15  ({100 * real / 15:.1f}%)  -- true and supported by the text")
print(f"  WEAK  : {weak}/15  ({100 * weak / 15:.1f}%)  -- true, but the source sentence cannot support it")
print(f"  NOISE : {noise}/15  ({100 * noise / 15:.1f}%)  -- unsupported, inverted, or hallucinated")
print(f"  (mechanical screen alone would have said REAL for {screen_real}/15)")
print()

print("Type shares (SAMPLE, split by verdict):")
for verdict in ("REAL", "WEAK", "NOISE"):
    sub = Counter(t for t, v in sample_verdicts if v == verdict)
    label = f"  {verdict}:"
    if not sub:
        print(f"{label} (none)")
        continue
    for t, c in sub.most_common():
        print(f"{label:<9}{t:<18}{c:>3}")
    print()

print("Type shares (SAMPLE, all):")
sample_types = Counter(t for t, _ in sample_verdicts)
for t, c in sample_types.most_common():
    print(f"  {t:<20}{c:>3}")

print()
print("Type shares (CORPUS, all relationships) with the sample's noise rate for that type:")
sample_by_type = {}
for t, v in sample_verdicts:
    s = sample_by_type.setdefault(t, Counter())
    s[v] += 1
for t, c in all_type_counts.most_common():
    s = sample_by_type.get(t)
    n_txt = ""
    if s:
        bad = s.get("NOISE", 0) + s.get("WEAK", 0)
        n_txt = f"   sample: {bad}/{sum(s.values())} not fully sound"
    print(f"  {t:<20}{c:>4} ({100 * c / len(all_rels):>5.1f}%){n_txt}")

out = "data/cache/exp_rel_judge.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump({
        "source_jsonl": JSONL,
        "total_relationships": len(all_rels),
        "type_shares_all": dict(all_type_counts),
        "sample": [
            {"source": r.get("source"), "target": r.get("target"),
             "type": r.get("type"), "source_sentence": r.get("source_sentence"),
             "doc_id": r["_doc"], "chunk_id": r["_chunk"],
             "verdict": judge(r)[0], "reason": judge(r)[1],
             "screen_verdict": screen(r)[0], "screen_reason": screen(r)[1],
             "manually_adjudicated": (r.get("source", ""), r.get("target", ""),
                                      (r.get("type") or "").strip().upper()) in MANUAL}
            for r in sample
        ],
        "real": real,
        "weak": weak,
        "noise": noise,
        "mechanical_screen_real": screen_real,
        "sample_type_shares": dict(sample_types),
        "sample_type_by_verdict": {
            v: dict(Counter(t for t, vv in sample_verdicts if vv == v))
            for v in ("REAL", "WEAK", "NOISE")
        },
    }, fh, indent=2, ensure_ascii=False)

print()
print(f"CACHE_DIR              : {cfg.CACHE_DIR}")
print(f"Cache entries at end   : {len(cache)}")
print(f"Wrote {out}")
print("EXP rel-judge complete.")
