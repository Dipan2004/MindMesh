# HotpotQA run `smoke_4pipe_2q`

- Benchmark: 100-question shared candidate corpus (union of the sampled questions' supplied gold + distractor paragraphs). NOT the official FullWiki setting.
- Questions: 2 ({'bridge': 2})
- Generator: `qwen2.5:3b`; index/extraction: `google/gemini-2.5-flash-lite`; embedding: `nomic-embed-text`
- Retrieval depth 10, generation context top-5, options {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256}
- Hardware: [{'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'memory_total_mib': 8188, 'memory_used_mib': 451, 'driver': '592.82'}] | CPU Intel64 Family 6 Model 191 Stepping 2, GenuineIntel

| Pipeline | n | R@1 | R@3 | R@5 | R@10 | MRR | EM | F1 | SP-P | SP-R | SP-F1 | J-EM | J-F1 | Ret s | Gen s | Tot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MindMesh Dense (`pipeline_a`) | 2 | 0.500 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.250 | 0.333 | 0.000 | 0.333 | 0.328 | 1.688 | 2.015 |
| MindMesh Graph (`pipeline_b`) | 2 | 0.500 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.813 | 0.976 | 4.789 |
| MindMesh Mix (`lightrag_mix`) | 2 | 0.500 | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 1.094 | 0.711 | 1.805 |
| MindMesh Mix Optimized (`lightrag_mix_opt`) | 2 | 0.500 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.594 | 0.610 | 1.204 |

Metrics: answer EM/F1, SP and joint per the official HotpotQA script; Recall@k/MRR over the supporting-fact paragraphs (MindMesh addition). Retrieval recall and citation validity do not establish semantic correctness.
