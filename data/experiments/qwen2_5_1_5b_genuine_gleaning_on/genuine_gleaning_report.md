# Genuine Gleaning-ON Experiment Report

Generated: 2026-11-12 00:40:28  
Model: `qwen2.5:1.5b`  MAX_GLEANINGS=1  
Source: `data/experiments/qwen2_5_1_5b_genuine_gleaning_on/`

---

## Extraction summary

| Metric | Value |
|---|---|
| Chunks completed | 42/42 |
| Parse errors | 4 |
| Gleaning OK | 37 |
| Total entities | 232 |
| Entities added by gleaning | 34 |
| Total relationships | 134 |
| Rels added by gleaning | 26 |
| Extraction LLM calls | 40 |
| Gleaning LLM calls | 37 |

---

## Graph metrics

| Metric | Phase 1 (0.5b) | v2-OFF (1.5b) | v2-ON (stub) | Genuine Gleaning |
|---|---|---|---|---|
| Nodes | 72 | 209 | 224 | 202 |
| Edges | 30 | 112 | 157 | 124 |
| Components | 45 | 103 | 75 | 86 |
| Largest component | 7 | 15 | 29 | 22 |
| Isolated nodes | 29 | 75 | 43 | 57 |
| Cross-doc nodes | 0 | 9 | 9 | 4 |
| **Cross-doc edges** | **0** | **0** | **0** | **0** |
| Usable chunks | 17/41 | 36/42 | 40/42 | 35/42 |

---

## 4-way retrieval comparison (Recall@6, top_k=6, 20 dev questions)

| Pipeline | Phase 1 | v2-OFF | v2-ON | **Gleaning-ON** | Best |
|---|---|---|---|---|---|
| pipeline_a | 0.9250 | 0.9250 | 0.9250 | 0.9250 | **0.9250** |
| pipeline_b | 0.8750 | 0.8250 | 0.8750 | 0.8250 | **0.8750** |
| lightrag_local | 0.7250 | 0.7250 | 0.7250 | 0.7750 | **0.7750** |
| lightrag_global | 0.5750 | 0.5750 | 0.6000 | 0.6500 | **0.6500** |
| lightrag_hybrid | 0.7500 | 0.7500 | 0.7500 | 0.7750 | **0.7750** |
| lightrag_mix | 0.7500 | 0.7500 | 0.7500 | 0.7750 | **0.7750** |

### MRR

| Pipeline | Phase 1 | v2-OFF | v2-ON | **Gleaning-ON** |
|---|---|---|---|---|
| pipeline_a | 0.7450 | 0.7450 | 0.7450 | 0.7450 |
| pipeline_b | 0.7350 | 0.6850 | 0.7350 | 0.6850 |
| lightrag_local | 0.4167 | 0.4167 | 0.4583 | 0.5083 |
| lightrag_global | 0.3667 | 0.3667 | 0.4183 | 0.4667 |
| lightrag_hybrid | 0.4642 | 0.4642 | 0.4667 | 0.5250 |
| lightrag_mix | 0.4642 | 0.4642 | 0.4667 | 0.5250 |

---

## D18 diagnostic (cross-doc hard query)

Gold chunk: `895ee8b439ed7d7b`

| Pipeline | Gold retrieved | Recall | Top-3 IDs |
|---|---|---|---|
| pipeline_a | False | 0.00 | a5123f467123650e, c81c2a06e4ee9667, 0e82efa9409ff33c |
| pipeline_b | False | 0.00 | a5123f467123650e, c81c2a06e4ee9667, a885be6b749e39a1 |
| lightrag_local | False | 0.00 | f08417a79e9ed8e6, 32c85d7ec02b6b19, 4a4ea014c5ca9c45 |
| lightrag_global | False | 0.00 | 24b17cf355abdce3, a885be6b749e39a1, 0e82efa9409ff33c |
| lightrag_hybrid | False | 0.00 | f08417a79e9ed8e6, 32c85d7ec02b6b19, 4a4ea014c5ca9c45 |
| lightrag_mix | False | 0.00 | f08417a79e9ed8e6, 32c85d7ec02b6b19, 4a4ea014c5ca9c45 |

---

## D03, D17 special attention

### D03
Gold: ['6d878b7f56a99aa2']

| Pipeline | Recall | MRR |
|---|---|---|
| pipeline_a | 1.00 | 1.00 |
| pipeline_b | 1.00 | 1.00 |
| lightrag_local | 0.00 | 0.00 |
| lightrag_global | 0.00 | 0.00 |
| lightrag_hybrid | 0.00 | 0.00 |
| lightrag_mix | 0.00 | 0.00 |

### D17
Gold: ['4af3dc8c5da84fad']

| Pipeline | Recall | MRR |
|---|---|---|
| pipeline_a | 1.00 | 1.00 |
| pipeline_b | 1.00 | 1.00 |
| lightrag_local | 0.00 | 0.00 |
| lightrag_global | 0.00 | 0.00 |
| lightrag_hybrid | 0.00 | 0.00 |
| lightrag_mix | 0.00 | 0.00 |

---

## Naming clarification

- **v2-OFF** and **v2-ON** in earlier reports refer to the **stub-endpoint filter**
  (EXTRACTION_STUB_ENDPOINTS), not gleaning. Both arms used identical LLM responses.
- **Genuine Gleaning-ON** = `MAX_GLEANINGS=1`, meaning a second LLM call per chunk
  asking the model what it missed. This is the first true gleaning experiment.

---

## Known limitations

- Cross-doc edges: 0 (expected 0 until a model extracts
  entities shared across documents consistently).
- D18 cross-doc paraphrase will not resolve until cross-doc edges exist.
- qwen2.5:1.5b entity naming is inconsistent (e.g. 'LLM' vs 'Large Language Model')
  which prevents alias merging across documents.