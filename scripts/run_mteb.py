import argparse
import json
import time
from pathlib import Path

import mteb
import torch
from sentence_transformers import SentenceTransformer


def find_scores(node):
    if isinstance(node, dict):
        if "ndcg_at_10" in node:
            return node
        for value in node.values():
            found = find_scores(value)
            if found:
                return found
    if isinstance(node, list):
        for value in node:
            found = find_scores(value)
            if found:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate an MTEB-registered model on AppsRetrieval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompts-file", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model = mteb.get_model(args.model, device=args.device)
    if args.checkpoint:
        model.model = SentenceTransformer(args.checkpoint, device=args.device)
    inner = getattr(model, "model", None)
    if inner is not None and args.max_length and hasattr(inner, "max_seq_length"):
        inner.max_seq_length = args.max_length
    if inner is not None and args.dtype == "bf16":
        inner.to(torch.bfloat16)
    task = mteb.get_task("AppsRetrieval")
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = json.load(f)
        model.model_prompts = dict(model.model_prompts or {})
        model.model_prompts[f"{task.metadata.name}-query"] = prompts["query"]
        model.model_prompts[f"{task.metadata.name}-document"] = prompts["document"]
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": args.batch_size}, cache=None)
    elapsed = time.time() - start

    task_result = list(result.task_results)[0]
    data = task_result.to_dict()
    with open(out_dir / f"{args.tag}_appsretrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    scores = find_scores(data) or {}
    summary = {
        "tag": args.tag,
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "checkpoint": args.checkpoint,
        "prompts_file": args.prompts_file,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
        "seconds": round(elapsed, 1),
        "ndcg_at_10": scores.get("ndcg_at_10"),
        "mrr_at_10": scores.get("mrr_at_10"),
        "recall_at_10": scores.get("recall_at_10"),
        "recall_at_100": scores.get("recall_at_100"),
    }
    with open(out_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
