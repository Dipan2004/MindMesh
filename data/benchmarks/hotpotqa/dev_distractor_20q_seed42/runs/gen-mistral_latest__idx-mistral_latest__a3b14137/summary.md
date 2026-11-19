# HotpotQA run `gen-mistral_latest__idx-mistral_latest__a3b14137`

- Benchmark: 20-question shared candidate corpus (union of the sampled questions' supplied gold + distractor paragraphs). NOT the official FullWiki setting.
- Questions: 20 ({'bridge': 10, 'comparison': 10})
- Generator: `mistral:latest`; index/extraction: `mistral:latest`; embedding: `nomic-embed-text`
- Retrieval depth 10, generation context top-5, options {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256}
- Hardware: [{'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'memory_total_mib': 8188, 'memory_used_mib': 5373, 'driver': '592.82'}] | CPU Intel64 Family 6 Model 191 Stepping 2, GenuineIntel

| Pipeline | n | R@1 | R@3 | R@5 | R@10 | MRR | EM | F1 | SP-P | SP-R | SP-F1 | J-EM | J-F1 | Ret s | Gen s | Tot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MindMesh Dense (`pipeline_a`) | 20 | 0.425 | 0.750 | 0.875 | 0.975 | 0.912 | 0.450 | 0.570 | 0.300 | 0.308 | 0.300 | 0.100 | 0.182 | 0.573 | 1.109 | 1.682 |
| MindMesh Graph (`pipeline_b`) | 20 | 0.375 | 0.725 | 0.775 | 0.825 | 0.825 | 0.450 | 0.570 | 0.217 | 0.217 | 0.208 | 0.050 | 0.121 | 1.347 | 1.183 | 2.530 |
| MindMesh Mix (`lightrag_mix`) | 20 | 0.425 | 0.675 | 0.775 | 0.925 | 0.900 | 0.450 | 0.568 | 0.108 | 0.125 | 0.115 | 0.050 | 0.075 | 1.338 | 1.380 | 2.718 |
| MindMesh Mix Optimized (`lightrag_mix_opt`) | 20 | 0.425 | 0.750 | 0.875 | 0.975 | 0.912 | 0.450 | 0.570 | 0.275 | 0.283 | 0.275 | 0.100 | 0.158 | 0.730 | 1.095 | 1.824 |

Metrics: answer EM/F1, SP and joint per the official HotpotQA script; Recall@k/MRR over the supporting-fact paragraphs (MindMesh addition). Retrieval recall and citation validity do not establish semantic correctness.
