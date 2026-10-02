# CodeRetriever

Version-aware text-to-code retrieval for the Samsung PRISM Gen AI Hackathon, Theme 1 (Agentic Code Intelligence).

Given a natural-language query, CodeRetriever ranks code snippets by relevance, on any version of a codebase or across all of its versions.

## Status

Development happens on `dev`; `main` is updated at each milestone.

| Goal | What it covers | Status |
|---|---|---|
| P0 | Retrieval accuracy on MTEB `AppsRetrieval` (NDCG@10, MRR) | Baselines measured; fine-tuning in progress |
| P1 | Retrieval on any version, with fast incremental re-indexing | Planned |
| Bonus | Retrieval across all versions | Planned |

## Findings so far

### Baseline embedding models on AppsRetrieval (test split)

Each model was run through MTEB's own wrapper for that model, so the numbers are comparable to the MTEB leaderboard.

| Model | Params | NDCG@10 | MRR@10 | Recall@10 | Recall@100 | Encode time* | Peak VRAM |
|---|---|---|---|---|---|---|---|
| **google/embeddinggemma-300m** | 308M | **84.05** | **80.73** | 94.3% | 99.5% | 3.7 min | 1.5 GB |
| Qwen/Qwen3-Embedding-0.6B | 596M | 75.92 | 71.75 | 89.0% | 98.2% | 7.0 min | 2.2 GB |

\*Full evaluation (8,765 solutions + 3,765 queries) on an RTX 4060 Laptop GPU, bf16, inputs capped at 2,048 tokens.

- Both scores match the published MTEB results (84.39 and 75.34), which validates the evaluation setup.
- EmbeddingGemma is the stronger and smaller model, so it is the base for fine-tuning.
- The correct solution is in the top 100 for 99.5% of queries but ranked first only about 75% of the time, so the remaining gains are in ranking, not recall.
- MTEB evaluates EmbeddingGemma with its generic retrieval prompt (`task: search result | query: `). The model also ships a code-specific prompt (`task: code retrieval | query: `), which we compare on a held-out dev set.

### Dataset notes

- `CoIR-Retrieval/apps`: 5,000 train and 3,765 test queries over one shared corpus of 8,765 Python solutions, with exactly one correct solution per query.
- Queries are long contest problem statements with Input/Output specifications and worked examples; solutions are uncommented stdin scripts.
- Training uses only the train split. Negatives are mined from train-split solutions only, so no test data enters training.

## Setup

Requires Python 3.10–3.13. Inference runs on CPU; an NVIDIA GPU is optional and used for training.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install -r requirements.txt
```

Use the PyTorch index that matches your driver (`cu130`, `cu128`, `cu126`), or plain `pip install torch` for CPU only. EmbeddingGemma is gated on Hugging Face: accept its licence and log in with `hf auth login` first.

## Reproducing the baselines

```bash
python scripts/prefetch.py --models google/embeddinggemma-300m Qwen/Qwen3-Embedding-0.6B
python scripts/run_mteb.py --model google/embeddinggemma-300m --device cuda --batch-size 8 --max-length 2048 --dtype bf16 --out-dir results --tag embeddinggemma-300m
python scripts/run_mteb.py --model Qwen/Qwen3-Embedding-0.6B --device cuda --batch-size 4 --max-length 2048 --dtype bf16 --out-dir results --tag qwen3-embedding-0.6b
```

Each run writes the MTEB result JSON and a summary with NDCG@10, MRR@10, recall, time and peak VRAM.
