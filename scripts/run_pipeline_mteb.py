import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mteb
import torch

from prism.config import Config
from prism.execution import ExampleVerifier
from prism.mteb_adapter import PipelineSearch, PrePostPipelineEncoder, load_base


def parse_views(text: str | None, default: dict) -> dict:
    if not text:
        return default
    return {k: float(v) for k, v in (part.split("=") for part in text.split(","))}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the CodeRetriever pipeline on AppsRetrieval and write the submission JSON")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--views", default=None, help="e.g. full=1,core=0.5,io=0.25")
    parser.add_argument("--mode", choices=["encoder", "search"], default="encoder")
    parser.add_argument("--execute-examples", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", default="pipeline")
    args = parser.parse_args()

    config = Config.load(args.config)
    choice = config.model_choice()
    checkpoint = args.checkpoint or (choice.model if choice.fine_tuned else None)
    dtype = args.dtype or config.profile(args.device)["dtype"]
    prompts = {"query": choice.query_prompt, "document": choice.document_prompt.replace("{title}", "none")}
    views = parse_views(args.views, config["query_views"])
    print(f"model: {checkpoint or config['embedder']['base_model']} | device {args.device} {dtype} | views {views} | mode {args.mode}", flush=True)

    start = time.time()
    base = load_base(config["embedder"]["base_model"], args.device, checkpoint, dtype, config["embedder"]["max_length"])
    encoder = PrePostPipelineEncoder(base, views, prompts)
    model = encoder
    if args.mode == "search":
        exec_cfg = config["execution_rerank"]
        verifier = ExampleVerifier(exec_cfg["timeout_seconds"], exec_cfg["workers"]) if args.execute_examples else None
        model = PipelineSearch(encoder, verifier, exec_cfg["top_k"])
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": args.batch_size}, cache=None)
    task_result = list(result.task_results)[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = task_result.to_dict()
    with open(out_dir / f"{args.tag}_appsretrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    scores = data["scores"]["test"][0]
    summary = {
        "tag": args.tag,
        "checkpoint": checkpoint,
        "device": args.device,
        "dtype": dtype,
        "views": views,
        "mode": args.mode,
        "execute_examples": args.execute_examples,
        "minutes": round((time.time() - start) / 60, 1),
        "ndcg_at_10": scores.get("ndcg_at_10"),
        "mrr_at_10": scores.get("mrr_at_10"),
        "recall_at_100": scores.get("recall_at_100"),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
    }
    with open(out_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
