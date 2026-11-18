# Third-Party Notices and Acknowledgements

MindMesh incorporates and adapts open-source software under permissive licenses. This document acknowledges upstream authors, projects, and components utilized within this repository.

---

### 1. LightRAG (HKUDS)

- **Upstream Project:** [LightRAG](https://github.com/HKUDS/LightRAG)
- **Organization / Authors:** HKUDS (Hong Kong University of Science and Technology) and Zirui Guo
- **License:** MIT License
- **Reused & Adapted Components:**
  - **Web UI:** The React/TypeScript interface under `ui/lightrag_ui/` was adapted from LightRAG's Web UI to provide MindMesh with interactive graph exploration, document management, and chat capabilities.
  - **Graph Retrieval Principles:** The algorithmic mechanisms for local entity neighborhood search, global relationship retrieval, hybrid fusion, and mixed graph-dense retrieval were adapted in `src/graphrag_lite/retrieval/lightrag/` to operate natively over MindMesh's NetworkX graphs and Zvec vector storage collections.

---

### 2. Core Python Ecosystem Libraries

- **NetworkX:** High-productivity software for complex networks, licensed under the 3-Clause BSD License.
- **FastAPI / Starlette / Pydantic:** Modern, fast web framework and data validation stack, licensed under the MIT License.
- **Zvec / Nano-VectorDB:** Lightweight embedded vector databases, licensed under Apache 2.0 / MIT.
- **Ollama Python SDK:** Client library for local Ollama language model inference, licensed under the MIT License.
