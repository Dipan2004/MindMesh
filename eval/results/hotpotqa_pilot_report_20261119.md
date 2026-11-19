# HotpotQA pilot: four MindMesh pipelines (2026-11-19)

> **Pilot / exploratory evaluation, not definitive evidence.** 20 questions, one run, one model,
> one machine. Differences of one or two questions are within noise. Kept separate from the
> internal 20-question dev results (`internal_dev_4pipelines_qwen2.5_1.5b_20261119.md`).

## 1. Setup

| Item | Value |
|---|---|
| Dataset | HotpotQA **dev, distractor setting** (7,405 questions), HotpotQA organisation's Hugging Face mirror `hotpotqa/hotpot_qa`, `distractor/validation`. The official CMU URL did not resolve (DNS failure); the mirror holds the same split. SHA-256 recorded in `benchmark_config.json`. |
| Sample | 20 questions: **10 bridge + 10 comparison**, seed 42 (`random.Random(42).sample` over each type sorted by `_id`) |
| Corpus | **20-question shared candidate corpus, NOT FullWiki**: the union of the sampled questions' supplied gold and distractor paragraphs |
| Paragraphs | 200 supplied, 200 unique (0 duplicates across these 20 questions), 889 sentences, 1 chunk per paragraph |
| IDs | `doc_id = hp_ + sha1(json([title, sentences]))[:16]`, `chunk_id = sha1(doc_id + ':0')[:16]` (content-stable) |
| Supporting facts | 42, all mapped `(title, sentence_index) -> chunk_id`, title and sentence text verified |
| Storage | Isolated: `data/benchmarks/hotpotqa/dev_distractor_20q_seed42/` (canonical `data/cache` untouched) |
| Extraction / generation model | `mistral:latest`, 7.2B, Q4_K_M, digest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf` |
| Embedding model | `nomic-embed-text:latest`, 768-d, F16, digest `0a109f422b47…` |
| Retrieval | top-10 for every pipeline; top-5 retrieved paragraphs passed to the generator |
| Generation | temperature 0, seed 42, `num_ctx` 8192, `num_predict` 256, JSON-schema output (answer + supporting-fact pairs), identical for all pipelines |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU 8 GB (driver 592.82, CUDA 13.1), Intel Core i7-13700HX, 15.6 GB RAM, Windows 11 (26200), Ollama 0.40.16, Python 3.11.9 |
| GPU offload (`ollama ps` during the run) | mistral:latest 5.5 GB **100% GPU** (ctx 8192); nomic-embed-text 336 MB **100% GPU** |

### Sample IDs

Bridge: `5ae31d5a5542990afbd1e0b1`, `5a7b89a3554299294a54a9e3`, `5a7344e95542991f9a20c6ce`,
`5a8ba59a554299240d9c2058`, `5a89007355429938390d4026`, `5a86edcc55429960ec39b6da`,
`5a7e1a5455429965cec5ea5f`, `5a7af32e55429931da12c99c`, `5ae5b99a5542993aec5ec1ce`,
`5adcfc135542990d50227d87`.
Comparison: `5a77720255429967ab10517f`, `5adc69ef554299438c868daf`, `5ab6be2f55429954757d335c`,
`5a734db15542994cef4bc523`, `5a7345ed5542992359bc32ce`, `5a77afcd5542997042120adf`,
`5a82accd55429966c78a6a78`, `5a8434445542996488c2e517`, `5abe76c255429965af743f15`,
`5add11935542994734353827`.

## 2. Index (shared by all four pipelines)

| Item | Value |
|---|---|
| Chunks indexed (Zvec) | 200 |
| Extraction | 200/200 chunks processed, **4 parse failures** (all distractors, no supporting paragraph): '2-D (character)', 'Emilio Baglioni', 'Pride and Glory (album)', 'Best of Decade I: 1995–2005'. Their text is still indexed for dense search; they add nothing to the graph. Gleaning off. |
| Entities / relationships extracted | 1,598 / 1,117 (before graph merge) |
| Graph (NetworkX) | **1,311 nodes, 1,093 edges**, 326 weak components (largest 129), 232 isolated nodes |
| Cross-document structure | 133 entities shared by 2+ documents; 146/200 documents linked through a shared entity; 12 edges whose provenance crosses documents |
| Zvec collections | entities 1,311 · relationships 1,093 · chunks 200 |
| Extraction time | 7,251 s of model time (mean 36.3 s/chunk, median 26.7 s, max 259 s), spread over ~3 h wall time because of thermal shutdowns (see section 7) |
| Graph build / collection build | 0.02 s / 59.5 s |

Validation before the run: 11/11 checks passed (`validation_report.json`).

## 3. Results (n = 20 per pipeline; 0 errors, 0 output-parse failures)

Run folder: `runs/gen-mistral_latest__idx-mistral_latest__a3b14137/`

**Retrieval** (gold = paragraphs holding supporting facts; this measures evidence retrieval only, not answer correctness):

| Pipeline | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| MindMesh Dense | 0.425 | 0.750 | 0.875 | **0.975** | **0.912** |
| MindMesh Graph | 0.375 | 0.725 | 0.775 | 0.825 | 0.825 |
| MindMesh Mix | 0.425 | 0.675 | 0.775 | 0.925 | 0.900 |
| MindMesh Mix Optimized | 0.425 | 0.750 | 0.875 | **0.975** | **0.912** |

(R@1 is at most 0.5 for HotpotQA, because each question has two supporting paragraphs.)

**Answers and supporting facts** (official HotpotQA definitions):

| Pipeline | Ans EM | Ans F1 | SP P | SP R | SP F1 | Joint EM | Joint F1 |
|---|---|---|---|---|---|---|---|
| MindMesh Dense | 0.450 | 0.570 | 0.300 | 0.308 | **0.300** | 0.100 | **0.182** |
| MindMesh Graph | 0.450 | 0.570 | 0.217 | 0.217 | 0.208 | 0.050 | 0.121 |
| MindMesh Mix | 0.450 | 0.568 | 0.108 | 0.125 | 0.115 | 0.050 | 0.075 |
| MindMesh Mix Optimized | 0.450 | 0.570 | 0.275 | 0.283 | 0.275 | 0.100 | 0.158 |

**Latency** (mean seconds per question):

| Pipeline | Retrieval | Generation | Total |
|---|---|---|---|
| MindMesh Dense | 0.57 | 1.11 | 1.68 |
| MindMesh Graph | 1.35 | 1.18 | 2.53 |
| MindMesh Mix | 1.34 | 1.38 | 2.72 |
| MindMesh Mix Optimized | 0.73 | 1.10 | 1.82 |

Generation throughput: 52 tokens/s mean (46–55) on the RTX 4060.

**By question type** (R@10 / Ans F1 / SP F1):

| Pipeline | Bridge (10) | Comparison (10) |
|---|---|---|
| MindMesh Dense | 0.95 / 0.507 / 0.13 | 1.00 / 0.633 / 0.47 |
| MindMesh Graph | 0.70 / 0.507 / 0.09 | 0.95 / 0.633 / 0.33 |
| MindMesh Mix | 0.90 / 0.502 / 0.00 | 0.95 / 0.633 / 0.23 |
| MindMesh Mix Optimized | 0.95 / 0.507 / 0.13 | 1.00 / 0.633 / 0.42 |

## 4. What the pilot shows (and does not)

- **Graph retrieval did not improve performance on this pilot.** No graph-based pipeline found a
  supporting paragraph that Dense missed on any question. Graph retrieved fewer gold paragraphs than
  Dense on 5 questions, and Mix on 2. Graph's drop is largest on bridge questions (R@10 0.70 vs 0.95),
  the type graph retrieval is meant to help.
- **Answer accuracy is identical across pipelines** (EM 0.45). With 20 questions and top-5 context,
  all four usually put both supporting paragraphs in front of the model; answer errors come from
  the generator (for example answering the actor instead of the character on Q7, or "Musicians"
  instead of the gold phrase on Q13), not from retrieval.
- **Mix Optimized behaved like Dense.** Its dense-anchored design kept Dense's top-8 and the 2 graph
  slots never added a gold paragraph; 18/20 generation contexts were identical and all 20 answers
  matched Dense. Its small SP-F1 difference (0.275 vs 0.300) comes from the two differing contexts.
- **Supporting-fact scores are low for every pipeline** (SP F1 ≤ 0.30). mistral:latest usually cites
  one sentence where HotpotQA marks two, and often the wrong sentence index.
- These are retrieval and string-match metrics. They do not establish semantic correctness: several
  "partial" answers are arguably right ("Ozzy Osbourne", "Film director").

## 5. Per-question results

Cell = R@10 · answer (EM exact, part partial F1 > 0, no) · predicted answer (truncated).
Full records, including retrieved chunk IDs, raw model output and supporting-fact predictions, are in
`results.jsonl`.

| # | ID | Type | Gold | Dense | Graph | Mix | Mix Optimized |
|---|---|---|---|---|---|---|---|
| 1 | `5ae31d5a5542990afbd1e0b1` | bridge | Wolfgang Amadeus Mozart | 1.0 · no · Salzburg | 1.0 · EM · Wolfgang Amadeus Mozart | 1.0 · EM · Wolfgang Amadeus Mozart | 1.0 · no · Salzburg |
| 2 | `5a7b89a3554299294a54a9e3` | bridge | 25 million copies | 1.0 · no · Unknown | 0.0 · no · 17 | 0.5 · no · unknown | 1.0 · no · Unknown |
| 3 | `5a7344e95542991f9a20c6ce` | bridge | Rudebox | 0.5 · no · Daddy Don't You Walk So Fa | 0.0 · no · I Want to Be Loved Like Th | 0.5 · no · No specific song informati | 0.5 · no · Daddy Don't You Walk So Fa |
| 4 | `5a8ba59a554299240d9c2058` | bridge | 1999 | 1.0 · EM · 1999 | 1.0 · no · 1929 | 1.0 · no · 1929 | 1.0 · EM · 1999 |
| 5 | `5a89007355429938390d4026` | bridge | Talib Kweli | 1.0 · EM · Talib Kweli | 0.5 · EM · Talib Kweli | 1.0 · EM · Talib Kweli | 1.0 · EM · Talib Kweli |
| 6 | `5a86edcc55429960ec39b6da` | bridge | Jennifer Aniston | 1.0 · EM · Jennifer Aniston | 1.0 · EM · Jennifer Aniston | 1.0 · EM · Jennifer Aniston | 1.0 · EM · Jennifer Aniston |
| 7 | `5a7e1a5455429965cec5ea5f` | bridge | Steve McGarrett | 1.0 · no · Alex O'Loughlin | 1.0 · no · Alex O'Loughlin | 1.0 · no · Alex O'Loughlin | 1.0 · no · Alex O'Loughlin |
| 8 | `5a7af32e55429931da12c99c` | bridge | Brooklyn, New York | 1.0 · part · Brooklyn | 0.5 · part · Brooklyn | 1.0 · part · Biohazard (formed in New Y | 1.0 · part · Brooklyn |
| 9 | `5ae5b99a5542993aec5ec1ce` | bridge | Drowning Pool | 1.0 · EM · Drowning Pool | 1.0 · EM · Drowning Pool | 1.0 · EM · Drowning Pool | 1.0 · EM · Drowning Pool |
| 10 | `5adcfc135542990d50227d87` | bridge | Magic Band | 1.0 · part · Captain Beefheart & His Ma | 1.0 · part · Captain Beefheart & His Ma | 1.0 · part · Captain Beefheart & His Ma | 1.0 · part · Captain Beefheart & His Ma |
| 11 | `5a77720255429967ab10517f` | comparison | 1985 | 1.0 · EM · 1985 | 1.0 · EM · 1985 | 1.0 · EM · 1985 | 1.0 · EM · 1985 |
| 12 | `5adc69ef554299438c868daf` | comparison | yes | 1.0 · EM · yes | 1.0 · EM · yes | 1.0 · EM · yes | 1.0 · EM · yes |
| 13 | `5ab6be2f55429954757d335c` | comparison | singer, songwriter, and multi- | 1.0 · no · Musicians | 1.0 · no · Musicians | 1.0 · no · Musicians | 1.0 · no · Musicians |
| 14 | `5a734db15542994cef4bc523` | comparison | Lili Chookasian | 1.0 · EM · Lili Chookasian | 1.0 · EM · Lili Chookasian | 1.0 · EM · Lili Chookasian | 1.0 · EM · Lili Chookasian |
| 15 | `5a7345ed5542992359bc32ce` | comparison | He was a mathematician | 1.0 · no · Mathematics | 1.0 · no · Mathematics | 1.0 · no · Mathematics | 1.0 · no · Mathematics |
| 16 | `5a77afcd5542997042120adf` | comparison | Carrie Akre | 1.0 · EM · Carrie Akre | 1.0 · EM · Carrie Akre | 1.0 · EM · Carrie Akre | 1.0 · EM · Carrie Akre |
| 17 | `5a82accd55429966c78a6a78` | comparison | no | 1.0 · EM · no | 0.5 · EM · no | 1.0 · EM · no | 1.0 · EM · no |
| 18 | `5a8434445542996488c2e517` | comparison | John Michael "Ozzy" Osbourne | 1.0 · part · Ozzy Osbourne | 1.0 · part · Ozzy Osbourne | 1.0 · part · Ozzy Osbourne | 1.0 · part · Ozzy Osbourne |
| 19 | `5abe76c255429965af743f15` | comparison | director | 1.0 · part · Film director | 1.0 · part · film director | 1.0 · part · Film director | 1.0 · part · Film director |
| 20 | `5add11935542994734353827` | comparison | no | 1.0 · no · yes | 1.0 · no · yes | 0.5 · no · yes | 1.0 · no · yes |

## 6. Reproducibility record

- **Freeze:** `freeze_20261119/` (code hashes, git HEAD `68b3475`, patch, snapshot of untracked code),
  plus three approved amendments made **before** the final run, each in its own folder with
  `AMENDMENT.md`, before/after hashes and a passing `verify_report.json`:
  1. `amend1`: Zvec insert batching (1,311 entities exceeded Zvec's 1,024-docs-per-call limit).
     Write path only.
  2. `amend2`: supporting-fact parser also accepts a flat `[p, s]` list (mistral's format).
     Scoring only.
  3. `amend3`: Ollama gets an explicit JSON schema instead of generic `format="json"`. Prompt text
     unchanged.
- **No retrieval code or retrieval/fusion setting was changed after the freeze.** Nothing was tuned on HotpotQA results.
- **Preserved, not used for results:**
  - `runs/smoke_2q_diagnostic/`: pre-amend2 smoke test.
  - `runs/smoke_2q_diagnostic_after_amend2/`
  - `runs/gen-mistral_latest__idx-mistral_latest__72597386/`: first full run. 44/80 generations
    aborted by Ollama ("token repeat limit reached"), which led to amend3. Its answer metrics are invalid.
- Provenance: `freeze_20261119/runtime_provenance.json`, `run_config.json` (hardware, `ollama ps`),
  `index/mistral_latest/index_meta.json`.

## 7. Limitations

- **Small pilot:** 20 questions, a single run and a single model. No significance testing. Not FullWiki.
- **Mix / Mix Optimized are reconstructions.** The original `graphrag_lite/retrieval/lightrag/`
  package was never committed (hidden by a `.gitignore` rule). It was rebuilt from the documented
  design, and its Mix fusion was redesigned after a per-list diagnosis on the internal dev set, before
  HotpotQA was run.
- **Extraction quality:** 4/200 paragraphs failed extraction. Only 12 graph edges cross documents.
  Gleaning was off. A 7B Q4 model extracted the graph.
- **Thermal shutdowns:** the laptop repeatedly hit its critical thermal point under sustained GPU
  load. There were 40+ thermal hibernations and 6 hard power-offs during extraction. Extraction resumed
  from saved progress each time with no duplicates (verified), but per-chunk extraction latencies include
  throttled periods. The final 80-question run completed without interruption.
- **No retries:** generation is single-shot at temperature 0.
- **Pipeline 3** (upstream LightRAG) was not part of this comparison and could not be tested (see the test summary).
