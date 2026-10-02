import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    mine_hard_negatives,
)
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

DATASET = "CoIR-Retrieval/apps"
DATASET_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"
DOC_PROMPT = "title: none | text: "
QUERY_PROMPTS = {
    "search": "task: search result | query: ",
    "code": "task: code retrieval | query: ",
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def starter_code(row):
    return (row.get("meta_information") or {}).get("starter_code") or ""


def load_apps(seed, dev_size):
    queries = load_dataset(DATASET, "queries", split="queries", revision=DATASET_REVISION)
    corpus = load_dataset(DATASET, "corpus", split="corpus", revision=DATASET_REVISION)
    qrels = load_dataset(DATASET, "default", split="train", revision=DATASET_REVISION)
    query_text = {r["_id"]: r["text"] for r in queries if r["partition"] == "train" and not starter_code(r)}
    doc_text = {r["_id"]: r["text"] for r in corpus}
    train_doc_ids = sorted(r["_id"] for r in corpus if r["partition"] == "train")
    train_doc_set = set(train_doc_ids)
    pairs = [
        (r["query-id"], r["corpus-id"])
        for r in qrels
        if r["query-id"] in query_text and r["corpus-id"] in train_doc_set
    ]
    random.Random(seed).shuffle(pairs)
    return query_text, doc_text, train_doc_ids, pairs[:dev_size], pairs[dev_size:]


def dev_evaluator(query_text, doc_text, dev_pairs, query_prompt, name):
    return InformationRetrievalEvaluator(
        queries={qid: query_text[qid] for qid, _ in dev_pairs},
        corpus=doc_text,
        relevant_docs={qid: {did} for qid, did in dev_pairs},
        query_prompt=query_prompt,
        corpus_prompt=DOC_PROMPT,
        name=name,
        batch_size=16,
        show_progress_bar=True,
        write_csv=False,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[10, 100],
        map_at_k=[100],
    )


def run_eval(model, evaluator, max_length):
    model.max_seq_length = max_length
    model.eval()
    with torch.no_grad():
        metrics = evaluator(model)
    return {k: round(float(v), 5) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EmbeddingGemma on the APPS train split")
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--revision", default="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--train-max-length", type=int, default=1024)
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--query-prompt", choices=["auto", *QUERY_PROMPTS], default="auto")
    parser.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    results = {"args": vars(args)}

    log("loading APPS")
    query_text, doc_text, train_doc_ids, dev_pairs, train_pairs = load_apps(args.seed, args.dev_size)
    results["data"] = {
        "test_like_train_queries": len(dev_pairs) + len(train_pairs),
        "dev_queries": len(dev_pairs),
        "train_pairs": len(train_pairs),
        "train_docs": len(train_doc_ids),
        "eval_corpus_docs": len(doc_text),
    }
    log(f"data {results['data']}")

    log(f"loading {args.model}")
    model = SentenceTransformer(args.model, revision=args.revision, device="cuda")

    base_scores = {}
    for key, prompt in QUERY_PROMPTS.items():
        if args.query_prompt not in ("auto", key):
            continue
        log(f"dev eval of base model with the {key} prompt")
        base_scores[key] = run_eval(model, dev_evaluator(query_text, doc_text, dev_pairs, prompt, f"dev-base-{key}"), args.eval_max_length)
        log(f"base {key}: {base_scores[key]}")
    results["base_dev"] = base_scores

    def ndcg(scores):
        return next(v for k, v in scores.items() if k.endswith("ndcg@10"))

    prompt_key = max(base_scores, key=lambda k: ndcg(base_scores[k]))
    query_prompt = QUERY_PROMPTS[prompt_key]
    results["query_prompt"] = {"key": prompt_key, "text": query_prompt}
    log(f"training with the {prompt_key} prompt")

    log("mining hard negatives from train-split solutions")
    model.max_seq_length = args.train_max_length
    pairs_ds = Dataset.from_dict({
        "anchor": [query_text[q] for q, _ in train_pairs],
        "positive": [doc_text[d] for _, d in train_pairs],
    })
    mined = mine_hard_negatives(
        pairs_ds,
        model,
        corpus=[doc_text[d] for d in train_doc_ids],
        relative_margin=0.05,
        num_negatives=args.negatives,
        sampling_strategy="top",
        query_prompt=query_prompt,
        corpus_prompt=DOC_PROMPT,
        output_format="n-tuple",
        batch_size=16,
    )
    mined.save_to_disk(str(out_dir / "mined"))
    results["mined_rows"] = len(mined)
    log(f"mined {len(mined)} rows with columns {mined.column_names}")

    if args.freeze_embeddings:
        for parameter in model[0].auto_model.get_input_embeddings().parameters():
            parameter.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["trainable_parameters"] = trainable
    log(f"trainable parameters: {trainable:,}")

    model.train()
    model.max_seq_length = args.train_max_length
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    prompts = {column: (query_prompt if column == "anchor" else DOC_PROMPT) for column in mined.column_names}
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        bf16=True,
        batch_sampler="no_duplicates",
        prompts=prompts,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=mined, loss=loss)
    start = time.time()
    trainer.train()
    results["train_minutes"] = round((time.time() - start) / 60, 1)
    results["train_log"] = trainer.state.log_history
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    log(f"training done in {results['train_minutes']} min, peak VRAM {results['peak_vram_gb']} GB")

    final_dir = out_dir / "final"
    model.save(str(final_dir))
    with open(out_dir / "prompts.json", "w", encoding="utf-8") as f:
        json.dump({"query": query_prompt, "document": DOC_PROMPT}, f, indent=2)

    log("dev eval of the fine-tuned model")
    results["finetuned_dev"] = run_eval(model, dev_evaluator(query_text, doc_text, dev_pairs, query_prompt, "dev-finetuned"), args.eval_max_length)
    log(f"fine-tuned: {results['finetuned_dev']}")
    results["dev_ndcg_at_10"] = {"base": ndcg(base_scores[prompt_key]), "finetuned": ndcg(results["finetuned_dev"])}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("SUMMARY", json.dumps({"query_prompt": prompt_key, **results["dev_ndcg_at_10"], "train_minutes": results["train_minutes"], "peak_vram_gb": results["peak_vram_gb"]}))


if __name__ == "__main__":
    main()
