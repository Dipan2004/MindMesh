# HotpotQA run `smoke_2q_diagnostic_after_amend2`

- Benchmark: 20-question shared candidate corpus (union of the sampled questions' supplied gold + distractor paragraphs). NOT the official FullWiki setting.
- Questions: 2 ({'bridge': 2})
- Generator: `mistral:latest`; index/extraction: `mistral:latest`; embedding: `nomic-embed-text`
- Retrieval depth 10, generation context top-5, options {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 256}
- Hardware: [{'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'memory_total_mib': 8188, 'memory_used_mib': 449, 'driver': '592.82'}] | CPU Intel64 Family 6 Model 191 Stepping 2, GenuineIntel

| Pipeline | n | R@1 | R@3 | R@5 | R@10 | MRR | EM | F1 | SP-P | SP-R | SP-F1 | J-EM | J-F1 | Ret s | Gen s | Tot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MindMesh Dense (`pipeline_a`) | 2 | 0.500 | 0.750 | 0.750 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.289 | 3.906 | 4.196 |
| MindMesh Graph (`pipeline_b`) | 2 | 0.250 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.149 | 1.203 | 2.352 |
| MindMesh Mix (`lightrag_mix`) | 2 | 0.500 | 0.750 | 0.750 | 0.750 | 1.000 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.587 | 1.078 | 1.665 |
| MindMesh Mix Optimized (`lightrag_mix_opt`) | 2 | 0.500 | 0.750 | 0.750 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.359 | 0.898 | 1.258 |

Metrics: answer EM/F1, SP and joint per the official HotpotQA script; Recall@k/MRR over the supporting-fact paragraphs (MindMesh addition). Retrieval recall and citation validity do not establish semantic correctness.
