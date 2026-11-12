# LightRAG Integration — Source, Version, License, Architecture

## Source

| Field | Value |
|---|---|
| Project | LightRAG |
| Organization | HKUDS (Hong Kong University of Science and Technology) |
| Repository | https://github.com/HKUDS/LightRAG |
| Package name | `lightrag-hku` |
| Version in this repo | v1.5.8 (git: `793f929c0`, tag: `temp-128-g793f929c0`) |
| License | **MIT License** |
| Location in repo | `LightRAG/` (vendored with its own `.git`) |
| Verification | `python -c "import sys; sys.path.insert(0,'LightRAG'); import lightrag; print(lightrag.__version__)"` → `1.5.8` |

## License

LightRAG is licensed under the **MIT License**. The full license text is at `LightRAG/LICENSE`.

```
MIT License
Copyright (c) 2024 Zirui Guo
...
```

MindMesh's own code is separate and does not inherit or modify the LightRAG license. The two packages coexist in this repository.

## Architecture

### Three distinct layers

```
MindMesh Repository
│
├── MindMesh-native code (src/graphrag_lite/, src/mindmesh/)
│   ├── Ingestion: loader, chunker, extractor, gleaning
│   ├── Graph: NetworkX DiGraph, builder, traversal, ranking
│   ├── Vector store: Zvec collections
│   ├── Retrieval:
│   │   ├── Pipeline A — flat vector (canonical, unchanged)
│   │   ├── Pipeline B — structural graph filter (canonical, unchanged)
│   │   └── lightrag/  — MindMesh-NATIVE adaptations (NOT upstream LightRAG)
│   │       ├── local.py, global_.py, hybrid.py, keywords.py, context.py
│   │       └── These adapted the paper algorithms; use Zvec + NetworkX
│   └── API: FastAPI (src/graphrag_lite/api/)
│
├── Genuine upstream LightRAG (LightRAG/ — vendored, MIT)
│   ├── lightrag/lightrag.py — LightRAG class, full pipeline
│   ├── lightrag/operate.py  — extraction, querying, kg_query, naive_query
│   ├── lightrag/api/        — LightRAG's own FastAPI server
│   └── lightrag_webui/      — React/TypeScript SPA for LightRAG
│
└── Integration layer (src/mindmesh/, run_pipeline.py, ui/)
    ├── Pipeline registry — maps names to adapter classes
    ├── Pipeline 3 adapter — wraps upstream LightRAG.query() 
    ├── Pipeline A/B adapters — thin wrappers preserving originals
    ├── LightRAG experimental adapters — wrap lightrag/ native code
    ├── LightRAG-compat API routes — connect UI to MindMesh backend
    └── ui/lightrag_ui/ — copied from LightRAG/lightrag_webui/
```

### Naming clarification (critical)

| Name | What it is | Location |
|---|---|---|
| `lightrag_local`, `lightrag_global`, `lightrag_hybrid`, `lightrag_mix` | **MindMesh-native** adaptations of LightRAG paper algorithms | `src/graphrag_lite/retrieval/lightrag/` |
| `pipeline_3_lightrag` | **Genuine upstream** LightRAG v1.5.8 | `src/mindmesh/pipelines/pipeline3_lightrag/` |
| LightRAG source | Vendored upstream package | `LightRAG/lightrag/` |
| LightRAG UI | Vendored React SPA | `LightRAG/lightrag_webui/` → copied to `ui/lightrag_ui/` |

The MindMesh-native `lightrag/` modules were written from scratch adapting the algorithm descriptions in arXiv:2410.05779. They use Zvec and NetworkX instead of LightRAG's NanoVectorDB and JSON KV stores. They are independent research implementations, NOT the upstream package.

## Pipeline 3 — Genuine LightRAG Integration

### How it works

`Pipeline3LightRAGAdapter` in `src/mindmesh/pipelines/pipeline3_lightrag/adapter.py`:

1. Adds `LightRAG/` to `sys.path` for import.
2. Creates an `OllamaClient`-compatible async LLM function (`_make_ollama_llm_func`).
3. Creates an Ollama-compatible async embed function (`_make_ollama_embed_func`).
4. Instantiates `lightrag.LightRAG(working_dir=data/pipeline3_workdir/, ...)`.
5. Calls `rag.aquery(question, QueryParam(mode=mode, top_k=top_k))`.

### Supported modes

`local`, `global`, `hybrid`, `naive`, `mix`, `bypass` — directly maps to LightRAG's `QueryParam.mode`.

### Storage isolation

Pipeline 3 writes to `data/pipeline3_workdir/` only. It does NOT use or modify:
- `data/cache/` (MindMesh canonical)
- `data/cache_v2/`
- Any experiment directory
- Any Zvec collection

### Dependencies added

Two lightweight packages were added to the Python environment:
- `json-repair` — JSON repair utilities (used by LightRAG internally)
- `nano-vectordb` — in-memory vector storage (LightRAG's default vector store)

These are listed as LightRAG's optional dependencies. They are NOT added to MindMesh's `pyproject.toml` to avoid making LightRAG a hard MindMesh dependency.

### GPU requirement (G7)

Before Pipeline 3 can answer queries, it must index documents:
```bash
python run_pipeline.py --pipeline pipeline_3_lightrag --model qwen2.5:1.5b --setup-only
```
This calls `rag.ainsert(docs)` which uses the LLM for entity/relationship extraction — a GPU-intensive operation tagged as **G7** in `GPU_DEFERRED_TASKS.md`.

## UI Integration

### Source

The LightRAG WebUI (`LightRAG/lightrag_webui/`) is a React/TypeScript SPA built with Vite, Bun, Tailwind CSS v4, shadcn/ui, and sigma.js for graph visualization.

The original source is preserved at `LightRAG/lightrag_webui/`. A copy was made to `ui/lightrag_ui/` for MindMesh integration.

### Integration approach

The LightRAG UI expects its backend to provide:
- `GET /health` — extended status
- `GET /documents`, `POST /documents/upload` — document management
- `POST /query` — query execution (all modes)
- `GET /graphs`, `GET /graph/label/list` — graph visualization

These are implemented in `src/graphrag_lite/api/lightrag_compat.py` backed by MindMesh's actual infrastructure (not faked).

### API backend mapping

| LightRAG UI call | MindMesh backend | Notes |
|---|---|---|
| `GET /health` | `/health` (extended) | Returns MindMesh-specific fields |
| `GET /documents` | Reads `data/raw/` | Lists source .md/.txt files |
| `POST /documents/upload` | Writes to `data/raw/` | Saves file; indexing is separate |
| `POST /query` mode=naive | `pipeline_a` | Flat vector baseline |
| `POST /query` mode=local | `lightrag_local` | MindMesh-native local |
| `POST /query` mode=global | `lightrag_global` | MindMesh-native global |
| `POST /query` mode=hybrid | `lightrag_hybrid` | MindMesh-native hybrid |
| `POST /query` mode=mix | `lightrag_mix` | MindMesh-native mix |
| `POST /query` mode=bypass | `pipeline_b` | Structural graph |
| `GET /graphs` | Canonical NetworkX graph | Built from `extractions.jsonl` |
| `GET /auth-status`, `POST /login` | No-auth stub | MindMesh has no auth |

### Configuration for development

The UI's `VITE_BACKEND_URL` defaults to `http://localhost:9621` (LightRAG's port). For MindMesh:

```bash
# Copy the MindMesh environment config
cp ui/lightrag_ui/.env.mindmesh ui/lightrag_ui/.env.development
# Then run the dev server
cd ui/lightrag_ui
bun run dev
```

The API runs at port 8000 by default:
```bash
uvicorn graphrag_lite.api.main:app --reload --port 8000
```

### Building the UI

```bash
# 1. Update vite.config.ts outDir to point to ui/lightrag_ui_dist
# (Change: outDir: path.resolve(import.meta.dirname, '../lightrag/api/webui')
#  To:     outDir: path.resolve(import.meta.dirname, '../lightrag_ui_dist'))
# 2. Build
cd ui/lightrag_ui
bun install
bun run build
# 3. The app will serve it at /webui when it finds ui/lightrag_ui_dist/
```

## Limitations

| Feature | Status | Notes |
|---|---|---|
| Pipeline A (flat) | ✅ Working | Via MindMesh canonical Zvec |
| Pipeline B (structural) | ✅ Working | Via MindMesh canonical graph |
| LightRAG Local/Global/Hybrid/Mix | ✅ Working | MindMesh-native, requires experiment collections |
| Pipeline 3 import | ✅ Working | LightRAG v1.5.8 fully importable |
| Pipeline 3 queries | ⏳ GPU task G7 | Requires document indexing first |
| UI dev server | ✅ Working | `bun run dev` after `cp .env.mindmesh .env.development` |
| UI production build | 🔧 Config needed | Update vite.config.ts outDir, then `bun run build` |
| Graph visualization | ✅ Working | Reads from MindMesh canonical graph |
| Document upload | ✅ Working | Writes to `data/raw/` |
| Document indexing via UI | ❌ Not supported | Use `python scripts/build_index.py` |
| Auth/workspaces | ❌ Not supported | MindMesh is single-user, no auth |
| Cross-document edges | ❌ Graph limitation | No cross-doc edges in any current graph |
| D18 retrieval | ❌ Graph limitation | Requires cross-doc edges (GPU task G1 may help) |
