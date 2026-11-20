# HotpotQA run `main5_gemini_4pipe_100q`

- Benchmark: 100-question shared candidate corpus (union of the sampled questions' supplied gold + distractor paragraphs). NOT the official FullWiki setting.
- Questions: 100 ({'bridge': 50, 'comparison': 50})
- Generator: `qwen2.5:3b`; index/extraction: `google/gemini-2.5-flash-lite`; embedding: `nomic-embed-text`
- Retrieval depth 10, generation context top-5, options {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256}
- Hardware: [{'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'memory_total_mib': 8188, 'memory_used_mib': 451, 'driver': '592.82'}] | CPU Intel64 Family 6 Model 191 Stepping 2, GenuineIntel

| Pipeline | n | R@1 | R@3 | R@5 | R@10 | MRR | EM | F1 | SP-P | SP-R | SP-F1 | J-EM | J-F1 | Ret s | Gen s | Tot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MindMesh Dense (`pipeline_a`) | 100 | 0.420 | 0.825 | 0.925 | 0.990 | 0.907 | 0.510 | 0.569 | 0.381 | 0.260 | 0.286 | 0.030 | 0.185 | 0.323 | 0.784 | 1.107 |
| MindMesh Graph (`pipeline_b`) | 100 | 0.400 | 0.735 | 0.800 | 0.825 | 0.872 | 0.500 | 0.541 | 0.317 | 0.221 | 0.240 | 0.010 | 0.142 | 1.919 | 0.826 | 2.746 |
| MindMesh Mix (`lightrag_mix`) | 100 | 0.420 | 0.790 | 0.885 | 0.980 | 0.905 | 0.530 | 0.569 | 0.402 | 0.280 | 0.308 | 0.020 | 0.185 | 0.804 | 0.680 | 1.484 |
| MindMesh Mix Optimized (`lightrag_mix_opt`) | 100 | 0.420 | 0.805 | 0.925 | 0.985 | 0.907 | 0.520 | 0.579 | 0.408 | 0.292 | 0.316 | 0.040 | 0.198 | 0.857 | 0.687 | 1.544 |

Metrics: answer EM/F1, SP and joint per the official HotpotQA script; Recall@k/MRR over the supporting-fact paragraphs (MindMesh addition). Retrieval recall and citation validity do not establish semantic correctness.
