# HotpotQA run `gen-mistral_latest__idx-mistral_latest__72597386`

- Benchmark: 20-question shared candidate corpus (union of the sampled questions' supplied gold + distractor paragraphs). NOT the official FullWiki setting.
- Questions: 20 ({'bridge': 10, 'comparison': 10})
- Generator: `mistral:latest`; index/extraction: `mistral:latest`; embedding: `nomic-embed-text`
- Retrieval depth 10, generation context top-5, options {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256}
- Hardware: [{'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'memory_total_mib': 8188, 'memory_used_mib': 449, 'driver': '592.82'}] | CPU Intel64 Family 6 Model 191 Stepping 2, GenuineIntel

| Pipeline | n | R@1 | R@3 | R@5 | R@10 | MRR | EM | F1 | SP-P | SP-R | SP-F1 | J-EM | J-F1 | Ret s | Gen s | Tot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MindMesh Dense (`pipeline_a`) | 20 | 0.425 | 0.750 | 0.875 | 0.975 | 0.912 | 0.150 | 0.204 | 0.150 | 0.075 | 0.100 | 0.000 | 0.070 | 0.444 | 0.581 | 1.025 |
| MindMesh Graph (`pipeline_b`) | 20 | 0.375 | 0.725 | 0.775 | 0.825 | 0.825 | 0.200 | 0.254 | 0.200 | 0.100 | 0.133 | 0.000 | 0.103 | 1.558 | 0.607 | 2.165 |
| MindMesh Mix (`lightrag_mix`) | 20 | 0.425 | 0.675 | 0.775 | 0.925 | 0.900 | 0.250 | 0.334 | 0.200 | 0.100 | 0.133 | 0.000 | 0.089 | 1.094 | 0.759 | 1.852 |
| MindMesh Mix Optimized (`lightrag_mix_opt`) | 20 | 0.425 | 0.750 | 0.875 | 0.975 | 0.912 | 0.150 | 0.204 | 0.150 | 0.075 | 0.100 | 0.000 | 0.070 | 0.898 | 0.491 | 1.389 |

Metrics: answer EM/F1, SP and joint per the official HotpotQA script; Recall@k/MRR over the supporting-fact paragraphs (MindMesh addition). Retrieval recall and citation validity do not establish semantic correctness.
