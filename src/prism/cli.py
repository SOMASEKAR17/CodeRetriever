import argparse
import json
import os
import sys
import time
from pathlib import Path

from .embedder import DEFAULT_DOCUMENT_PROMPT, DEFAULT_QUERY_PROMPT, HashEmbedder, SentenceTransformerEmbedder
from .engine import Engine

DEFAULT_MODEL = "google/embeddinggemma-300m"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_engine(args) -> Engine:
    if args.embedder == "hash":
        embedder = HashEmbedder()
    else:
        device = resolve_device(args.device)
        dtype = args.dtype if args.dtype != "auto" else ("bf16" if device == "cuda" else "fp32")
        query_prompt, document_prompt = DEFAULT_QUERY_PROMPT, DEFAULT_DOCUMENT_PROMPT
        if args.prompts_file:
            with open(args.prompts_file, encoding="utf-8") as f:
                prompts = json.load(f)
            query_prompt = prompts["query"]
            document_prompt = prompts["document"].replace("title: none", "title: {title}")
        model = args.checkpoint or args.model
        space = args.space or Path(model.rstrip("/\\")).name
        embedder = SentenceTransformerEmbedder(
            model,
            space=space,
            device=device,
            dtype=dtype,
            max_length=args.max_length,
            batch_size=args.batch_size,
            query_prompt=query_prompt,
            document_prompt=document_prompt,
        )
    return Engine(args.index_dir, embedder)


def cmd_add(args) -> None:
    engine = build_engine(args)
    if args.corpus:
        source_id = engine.add_corpus(args.corpus, args.id or "corpus")
    elif args.location.startswith(("http://", "https://", "git@")) or (Path(args.location) / ".git").exists():
        source_id = engine.add_git(args.location, args.id, partial=args.partial)
    else:
        source_id = engine.add_folder(args.location, args.id)
    print(f"added source {source_id}")


def cmd_versions(args) -> None:
    engine = build_engine(args)
    indexed = {v["version_id"]: v for v in engine.versions(args.source)}
    print("indexed:")
    for v in indexed.values():
        print(f"  {v['label']:<30} {v['version_id']}")
    if args.available:
        print("available:")
        for v in engine.available_versions(args.source):
            print(f"  {v.label:<30} {v.ref[:12]}")


def cmd_index(args) -> None:
    engine = build_engine(args)
    stats = engine.index(args.source, args.refs or None)
    if args.json:
        print(json.dumps([s.as_dict() for s in stats], indent=2))


def cmd_search(args) -> None:
    engine = build_engine(args)
    if args.version == "all" and not args.flat:
        cmd_search_evolution(engine, args)
        return
    started = time.time()
    hits = engine.search(args.query, args.source, args.version, args.k)
    elapsed = time.time() - started
    if args.json:
        print(json.dumps({"seconds": round(elapsed, 3), "hits": [h.__dict__ for h in hits]}, indent=2))
        return
    print(f"{len(hits)} results in {elapsed * 1000:.0f} ms")
    for rank, hit in enumerate(hits, 1):
        places = sorted({(o["version_id"].split("@")[-1], o["path"], o["start_line"], o["end_line"]) for o in hit.occurrences})
        first = places[0] if places else ("?", "?", 0, 0)
        print(f"\n#{rank}  score {hit.score:.4f}  {hit.symbol}  {first[1]}:{first[2]}-{first[3]}")
        if len(places) > 1:
            print(f"    also in {len(places) - 1} other place(s)/version(s)")
        preview = "\n".join(hit.code.splitlines()[: args.lines])
        print("    " + preview.replace("\n", "\n    "))


def cmd_search_evolution(engine, args) -> None:
    started = time.time()
    intent, hits = engine.search_evolution(args.query, args.source, args.k)
    elapsed = time.time() - started
    if args.json:
        print(json.dumps({"seconds": round(elapsed, 3), "intent": intent.__dict__, "hits": [h.__dict__ for h in hits]}, indent=2))
        return
    mode = f"version {intent.matched}" if intent.kind == "version" else intent.kind
    print(f"{len(hits)} results in {elapsed * 1000:.0f} ms (mode: {mode})")
    for rank, hit in enumerate(hits, 1):
        print(f"\n#{rank}  score {hit.score:.4f}  {hit.symbol}  {hit.path}:{hit.start_line}-{hit.end_line}  [{hit.label}]")
        if hit.timeline:
            marks = " -> ".join(f"{t['label']}{'*' if t['changed'] else ''}" for t in hit.timeline)
            print(f"    versions: {marks}   (* = changed)")
        preview = "\n".join(hit.code.splitlines()[: args.lines])
        print("    " + preview.replace("\n", "\n    "))
        if hit.diff and args.diff:
            print("    diff:\n    " + hit.diff.replace("\n", "\n    "))


def cmd_bench(args) -> None:
    engine = build_engine(args)
    rows = [s.as_dict() for s in engine.index(args.source, args.refs)]
    print(json.dumps(rows, indent=2))
    total_new = sum(r["chunks_embedded"] for r in rows)
    total_seen = sum(r["chunks_in_version"] for r in rows)
    if total_seen:
        print(f"embedded {total_new} chunks for {total_seen} chunk occurrences across versions ({100 * (1 - total_new / total_seen):.1f}% reused)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="prism", description="Version-aware code retrieval")
    parser.add_argument("--index-dir", default=os.environ.get("PRISM_INDEX_DIR", str(Path.home() / ".prism")))
    parser.add_argument("--embedder", choices=["model", "hash"], default="model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--space", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="register a git repo (path or URL), a folder, or JSONL snippet files")
    p.add_argument("location", nargs="?", default="")
    p.add_argument("--id")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--corpus", nargs="+")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("versions", help="list indexed (and available) versions")
    p.add_argument("source")
    p.add_argument("--available", action="store_true")
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("index", help="index versions of a source incrementally")
    p.add_argument("source")
    p.add_argument("--refs", nargs="*")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search one version or all versions")
    p.add_argument("query")
    p.add_argument("--source", required=True)
    p.add_argument("--version", default="all")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--lines", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.add_argument("--flat", action="store_true", help="return raw chunks instead of one result per lineage")
    p.add_argument("--diff", action="store_true", help="show the diff to the previous version")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("bench-versions", help="index versions in order and report reuse and timings")
    p.add_argument("source")
    p.add_argument("--refs", nargs="+", required=True)
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
