"""Temporary read-only consistency check script. Safe to delete after use."""
import re

with open('TASKS.md', 'r', encoding='utf-8') as f:
    tasks = f.read()
with open('DOCUMENT.md', 'r', encoding='utf-8') as f:
    doc = f.read()

# Task 8.4 block in TASKS.md
tasks_84_idx = tasks.find('[!] PARTIAL')
tasks_84_block = tasks[tasks_84_idx:tasks_84_idx + 800] if tasks_84_idx >= 0 else ''

# Deviation notice block in DOCUMENT.md
doc_dev_idx = doc.find('Hardware-limitation deviation notice')
doc_dev_block = doc[doc_dev_idx:doc_dev_idx + 2000] if doc_dev_idx >= 0 else ''

print('=== DOCUMENT.md deviation notice contains ===')
checks_doc = {
    'gold_dev_set.json mentioned':      'gold_dev_set.json' in doc_dev_block,
    'frozen SHA256 present':            '5d9bcbbf48ff45c2' in doc_dev_block,
    'PIPELINE_B_SCOPE=docs':            'PIPELINE_B_SCOPE=docs' in doc_dev_block,
    'SIM_WEIGHT=0.85':                  'SIM_WEIGHT=0.85' in doc_dev_block,
    'think=False':                      'think=False' in doc_dev_block,
    'slm_ablation.py script command':   'slm_ablation.py' in doc_dev_block,
    'git commit 7158756':               '7158756' in doc_dev_block,
    'professor system mentioned':       "professor" in doc_dev_block,
    'qwen3:4b deferred named':          'qwen3:4b' in doc_dev_block,
    'phi4-mini deferred named':         'phi4-mini' in doc_dev_block,
}
for k, v in checks_doc.items():
    status = 'YES' if v else 'MISSING'
    print(f'  {k}: {status}')

print()
print('=== TASKS.md Task 8.4 deviation notice contains ===')
checks_tasks = {
    '[!] PARTIAL marker present':       '[!] PARTIAL' in tasks_84_block,
    'No [x] done marker':               '[x]' not in tasks_84_block,
    'qwen3:4b and phi4-mini named':     ('qwen3:4b' in tasks_84_block and 'phi4-mini' in tasks_84_block),
    'OOM reason stated':                'OOM' in tasks_84_block,
    'professor system mentioned':       'professor' in tasks_84_block,
    'Do not mark [x] instruction':      'not mark' in tasks_84_block.lower() or 'Do not mark' in tasks_84_block,
}
for k, v in checks_tasks.items():
    status = 'YES' if v else 'MISSING'
    print(f'  {k}: {status}')

# ── DOCUMENT.md Task 8.4 header check ──────────────────────────────────────
print()
print('=== DOCUMENT.md Task 8.4 heading ===')
m84 = re.search(r'### Task 8\.4[^\n]*', doc)
if m84:
    print(f'  Heading: {m84.group(0)!r}')
    has_complete = 'complete' in m84.group(0).lower() and 'partial' not in m84.group(0).lower()
    print(f'  Falsely claims complete (no partial caveat): {has_complete}')
else:
    print('  NOT FOUND')

# ── Total accounting ─────────────────────────────────────────────────────────
print()
print('=== Running Summary arithmetic check ===')
rows = re.findall(
    r'\| ([^|]+?) \| (\d+) \| (\d+) \| (\d+) \|',
    doc
)
# Only rows that look like phase rows (phase name contains a number or **)
phase_rows = [(a.strip(), int(b), int(c), int(d)) for a, b, c, d in rows if any(ch.isdigit() for ch in a) or '**' in a]
for name, total, done, blocked in phase_rows:
    not_started = total - done - blocked
    check = 'OK' if done + blocked + not_started == total else 'MISMATCH'
    if '**' not in name:
        print(f'  {name}: total={total} done={done} blocked={blocked} not_started={not_started} => {check}')
    else:
        sum_total = sum(r[1] for r in phase_rows if '**' not in r[0])
        sum_done  = sum(r[2] for r in phase_rows if '**' not in r[0])
        sum_block = sum(r[3] for r in phase_rows if '**' not in r[0])
        match_total = '(sum=%d, stated=%d %s)' % (sum_total, total, 'OK' if sum_total==total else 'MISMATCH')
        match_done  = '(sum=%d, stated=%d %s)' % (sum_done,  done,  'OK' if sum_done==done   else 'MISMATCH')
        match_block = '(sum=%d, stated=%d %s)' % (sum_block, blocked, 'OK' if sum_block==blocked else 'MISMATCH')
        print(f'  **Total**: total={total} {match_total}')
        print(f'             done={done}  {match_done}')
        print(f'             blocked={blocked}  {match_block}')
