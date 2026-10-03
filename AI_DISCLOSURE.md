# AI Disclosure

This project used an AI assistant (Anthropic Claude, through the Claude desktop app) at specific stages. This file describes what the assistant did and what the author did.

## Where AI was used

**1. Research**

- Surveying published approaches to code retrieval: CoIR and MTEB benchmarks, embedding models such as EmbeddingGemma and Qwen3-Embedding, rerankers, query preprocessing, and incremental indexing techniques.
- Summarising papers, model cards and library documentation into a research note.

**2. Writing code to the author's specification**

After the design decisions below were made, the assistant wrote the implementation:

- The `prism` Python package: chunking, versioned SQLite index, embedder, search engine, function-lineage tracking, MTEB adapter (`PrePostPipelineEncoder`), local API server, desktop window and UI.
- Training and evaluation scripts, the PowerShell helper scripts, unit tests, the Dockerfile, and the README and architecture diagrams.
- Fixes for bugs the author found while running and testing the application.

## What the author did

- **Pipeline design:** decided the overall retrieval flow: an embedding model with a CPU-first path, an optional GPU path, query preprocessing, and version-aware indexing that reuses unchanged code.
- **Model experiments:** ran the baseline evaluations (EmbeddingGemma-300m and Qwen3-Embedding-0.6B) and the fine-tuning experiment on a local RTX 4060 laptop. Compared the results on a held-out dev set and on the test split, and chose the best configuration: base EmbeddingGemma-300m. The fine-tuned model scored lower and was not used.
- **Application architecture:** specified a desktop application that takes a local folder or a GitHub URL, a user-selectable CPU/GPU mode, versioned search across releases, and the interface design: layout, black-and-white theme, overlay sidebars and macOS-style window controls.
- **Testing and verification:** installed and ran every step on local hardware, checked all reported numbers, tested the desktop application end to end, and reported issues for fixing.
- **Final decisions:** reviewed the code and results, and decided what was kept, changed or dropped.

## Results integrity

All benchmark numbers in this repository (for example 84.03 NDCG@10 on AppsRetrieval, CPU) come from runs executed on the author's machine. The raw MTEB outputs are in `results/`. No results were produced or edited by the AI assistant.
