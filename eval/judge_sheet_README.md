# judge_sheet.csv — how to label

40 relationships sampled from the **ladder arm** (`data/cache/exp_reextract_ladder`),
built by `scripts/build_judge_sheet.py`.

## What to do

Open `eval/judge_sheet.csv` and fill the `human_label` column for the **20 rows
whose `fill_first` is `y`**. Leave the other 20 blank — they are the audit copy.

The 20 rows were drawn with `random.Random(20261022)`, stratified 7 / 7 / 6 over
the three strata, so the annotated subsample is representative of the sheet.
Labelling a different 20 is fine but then the numbers no longer compare with
the ones quoted in DOCUMENT.md.

## The labels

| label | meaning |
|---|---|
| `REAL` | The relationship is true **and supported by the chunk text**. The chunk shows that the source and the target stand in the stated relation. |
| `WEAK` | True in substance, but the edge as written is not what the text says — usually the relation *type* is a mis-typing (membership written as `DEVELOPED_BY`, "based on" written as `DEVELOPED_BY`, an instance written as `INTEGRATED`). You can see what was meant; the edge is still sloppy. |
| `NOISE` | Unsupported, inverted, or hallucinated. The text does not say this, or says the opposite, or the two endpoints have nothing to do with each other. |

Use `WEAK` when you are genuinely torn. Do not use it to mean "I don't know" —
if you cannot tell from the chunk, that is `NOISE` (the chunk does not support
it).

## The columns

| column | what it is |
|---|---|
| `chunk_id` | the chunk the relationship came from |
| `source`, `target`, `type` | the relationship as the model emitted it |
| `source_sentence` | the sentence the model cited. **It is often wrong or a bare heading** — judge against `chunk_text`, not against this column. |
| `agent_label` | my label, filled in before you saw the sheet. Do not change it. |
| `human_label` | **your label — fill this in** |
| `doc_id` | source document |
| `stratum` | `A` kept normally · `B` recovered by the step-2 conditional filter · `C` pre-existing edge of a chunk the step-2 filter rescues |
| `fill_first` | `y` = please label this one |
| `agent_note` | why I labelled it the way I did (for the disagreement discussion) |
| `chunk_text` | the full chunk the relationship came from — **this is the evidence** |

## After labelling

```
python tests/eval_judge_sheet.py
```

prints the agent-vs-human agreement, the human noise rate with a 95% interval,
the agent's noise rate on the same rows, the per-stratum breakdown, and every
row the two of us disagree on.
