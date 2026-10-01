# CodeRetriever

Version-aware text-to-code retrieval for the Samsung PRISM Gen AI Hackathon, Theme 1 (Agentic Code Intelligence).

Given a natural-language query, CodeRetriever ranks code snippets by relevance, on any version of a codebase or across all of its versions.

## Status

Work in progress. Development happens on `dev`; `main` is updated at each milestone.

| Goal | What it covers | Status |
|---|---|---|
| P0 | Retrieval accuracy on MTEB `AppsRetrieval` (NDCG@10, MRR) | in progress |
| P1 | Retrieval on any version, with fast incremental re-indexing | planned |
| Bonus | Retrieval across all versions | planned |

## Requirements

- Python 3.10–3.13
- Inference runs on CPU; an NVIDIA GPU is optional (used for training and the GPU profile)

Setup and run instructions will be added as each milestone lands.
