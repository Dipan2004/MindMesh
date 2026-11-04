"""Read-only research pipeline audit. No modifications."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── 1. Parse-error chunk: which one, retrieved by canonical pipelines? ────────
with open('data/cache/extractions.jsonl') as f:
    recs = [json.loads(l) for l in f if l.strip()]
parse_errs = [r for r in recs if '_parse_error' in r]
clean = [r for r in recs if '_parse_error' not in r]
print(f"Total records: {len(recs)} | clean: {len(clean)} | parse_error: {len(parse_errs)}")
for r in parse_errs:
    cid = r.get('chunk_id', '?')
    doc = r.get('doc_id', '?')
    print(f"  parse_error chunk: {cid} from {doc}")

missing_cid = parse_errs[0]['chunk_id'] if parse_errs else None

with open('eval/results/eval_20261103_015802.json') as f:
    ev = json.load(f)

print("\nDoes canonical eval retrieve the parse-error chunk?")
for pl in ('a', 'b'):
    for q in ev['results'][pl]:
        if missing_cid and missing_cid in q.get('evidence_ids', []):
            print(f"  Pipeline {pl.upper()} Q={q['query_id']} retrieved {missing_cid}")

# ── 2. Is D16 the only FTS error query? Check D16 text ───────────────────────
with open('eval/gold_dev_set.json') as f:
    dev = json.load(f)
print("\nD16 query text and gold:")
for q in dev['queries']:
    if q['query_id'] == 'D16':
        print(f"  query: {q['query']!r}")
        print(f"  gold_chunk_ids: {q['gold_chunk_ids']}")
        print(f"  type: {q['type']}")

# ── 3. Pipeline A vs B: are retrieved sets identical for each query? ──────────
print("\nPipeline A vs B retrieved sets per query:")
print(f"{'QID':<6} {'A_ids (8)':<70} {'B_ids (6)':<60} same_top6?")
for qa, qb in zip(ev['results']['a'], ev['results']['b']):
    assert qa['query_id'] == qb['query_id']
    a_ids = qa['evidence_ids']
    b_ids = qb['evidence_ids']
    top6_a = set(a_ids[:6])
    top6_b = set(b_ids)
    same = top6_a == top6_b
    diff = top6_a.symmetric_difference(top6_b)
    print(f"  {qa['query_id']}: A@8={a_ids[:2]}... B@6={b_ids[:2]}... same_top6={same} diff={sorted(diff)}")

# ── 4. For cross-doc questions (D13-D15): are both gold chunks retrieved? ─────
print("\nCross-doc recall detail:")
for pl in ('a', 'b'):
    for q in ev['results'][pl]:
        if q.get('type') == 'cross-doc':
            gold = q.get('gold_chunk_ids', [])
            retr = q.get('evidence_ids', [])
            hits = [g for g in gold if g in retr]
            miss = [g for g in gold if g not in retr]
            print(f"  [{pl.upper()}] {q['query_id']}: gold={gold} hits={hits} miss={miss}")

# ── 5. Embedding ablation: are the 41-chunk indexes missing the parse-error? ──
print("\nEmbedding ablation vs canonical chunk counts:")
import zvec
for label, path in [('canonical', 'data/cache/chunks_v1'),
                     ('ablation_nomic', 'data/cache_embed_ablation/nomic/chunks_v1'),
                     ('ablation_qwen3', 'data/cache_embed_ablation/qwen3/chunks_v1'),
                     ('ablation_minilm', 'data/cache_embed_ablation/minilm/chunks_v1')]:
    try:
        c = zvec.open(path)
        n = c.stats.doc_count
        c.close()
        print(f"  {label}: {n} docs")
    except Exception as e:
        print(f"  {label}: ERROR {e}")

# ── 6. TOON paradox: answer token counts compared ─────────────────────────────
with open('eval/results/toon_vs_json_ablation_20261103_021434.json') as f:
    toon_json = json.load(f)

toon_agg = toon_json['toon']['aggregates']['b']
json_agg  = toon_json['json']['aggregates']['b']
print("\nTOON vs JSON aggregate comparison:")
print(f"  Evidence tokens: TOON={toon_agg['avg_evidence_tokens']:.0f}  JSON={json_agg['avg_evidence_tokens']:.0f}")
print(f"  Answer tokens:   TOON={toon_agg['avg_answer_tokens']:.1f}     JSON={json_agg['avg_answer_tokens']:.1f}")
print(f"  Gen latency:     TOON={toon_agg['avg_generation_latency_s']:.2f}s  JSON={json_agg['avg_generation_latency_s']:.2f}s")
print(f"  Term coverage:   TOON={toon_agg['avg_term_coverage']:.4f}      JSON={json_agg['avg_term_coverage']:.4f}")
print(f"  Recall@K:        TOON={toon_agg['avg_recall_at_k']:.4f}         JSON={json_agg['avg_recall_at_k']:.4f}")

# ── 7. Per-query: where do A and B actually differ? ───────────────────────────
print("\nQueries where A and B give different answers (term_coverage diff > 0.05):")
for qa, qb in zip(ev['results']['a'], ev['results']['b']):
    diff = abs(qa.get('term_coverage', 0) - qb.get('term_coverage', 0))
    if diff > 0.05:
        print(f"  {qa['query_id']} ({qa['type']}): A_term={qa['term_coverage']:.2f} B_term={qb['term_coverage']:.2f} A_rank={qa.get('gold_rank')} B_rank={qb.get('gold_rank')}")

print("\nQueries where no_answer_correct differs between A and B:")
for qa, qb in zip(ev['results']['a'], ev['results']['b']):
    if qa.get('no_answer_correct') != qb.get('no_answer_correct'):
        print(f"  {qa['query_id']} ({qa['type']}): A={qa['no_answer_correct']} B={qb['no_answer_correct']}")
