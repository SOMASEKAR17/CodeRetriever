import argparse
import json
import sys
import time
from pathlib import Path

from .chunker import Chunker
from .config import Config, resolve_device
from .embedder import HashEmbedder, SentenceTransformerEmbedder
from .engine import Engine


def build_engine(args, device: str | None = None) -> Engine:
    config = Config.load(args.config)
    index_dir = args.index_dir or str(config.index_dir)
    chunk_cfg = config["chunking"]
    chunker = Chunker(max_lines=chunk_cfg["max_lines"], window_overlap=chunk_cfg["window_overlap"])
    if args.embedder == "hash":
        return Engine(index_dir, HashEmbedder(), chunker)
    device = device or resolve_device(args.device or config["device"])
    profile = config.profile(device)
    choice = config.model_choice()
    model = args.checkpoint or choice.model
    space = args.space or (choice.space if not args.checkpoint else model.rstrip("/\\").split("/")[-1].split("\\")[-1])
    if not choice.fine_tuned and not args.checkpoint:
        print(f"note: fine-tuned checkpoint not found, using base model {choice.model}", file=sys.stderr)
    emb_cfg = config["embedder"]
    embedder = SentenceTransformerEmbedder(
        model,
        space=space,
        device=device,
        dtype=args.dtype if args.dtype != "auto" else profile["dtype"],
        max_length=emb_cfg["max_length"],
        batch_size=emb_cfg["batch_size"],
        query_prompt=choice.query_prompt,
        document_prompt=choice.document_prompt,
        revision=None if args.checkpoint else choice.revision,
    )
    reranker = None
    if profile.get("reranker") and not args.no_rerank:
        from .reranker import load_reranker

        reranker = load_reranker(profile["reranker"], device)
    return Engine(index_dir, embedder, chunker, reranker=reranker, rerank_top_k=profile.get("rerank_top_k") or 0)


def cmd_add(args) -> None:
    engine = build_engine(args)
    if args.corpus:
        source_id = engine.add_corpus(args.corpus, args.id or "corpus")
    elif args.location.startswith(("http://", "https://", "git@")) or (Path(args.location) / ".git").exists():
        source_id = engine.add_git(args.location, args.id, partial=args.partial)
    else:
        source_id = engine.add_folder(args.location, args.id)
    print(f"added source {source_id}")


def cmd_remove(args) -> None:
    engine = build_engine(args)
    result = engine.remove_source(args.source, delete_clone=not args.keep_clone)
    print(f"removed {result['source']}: {result['versions_removed']} indexed version(s)" + (", deleted its clone" if result["clone_removed"] else ""))


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


def _app(args):
    from .server import App

    config = Config.load(args.config)
    device = "cpu" if args.embedder == "hash" else resolve_device(args.device or config["device"])
    return App(lambda d: build_engine(args, d), device), config


def cmd_serve(args) -> None:
    from .server import serve

    app, config = _app(args)
    serve(app, args.host or config["server"]["host"], args.port or config["server"]["port"])


def cmd_desktop(args) -> None:
    from .desktop import launch

    app, config = _app(args)
    launch(app, config["server"]["host"], args.port or config["server"]["port"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="prism", description="Version-aware code retrieval")
    parser.add_argument("--config", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--embedder", choices=["model", "hash"], default="model")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--space", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--no-rerank", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="register a git repo (path or URL), a folder, or JSONL snippet files")
    p.add_argument("location", nargs="?", default="")
    p.add_argument("--id")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--corpus", nargs="+")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("remove", help="remove a source, its indexed versions and (for cloned URLs) its clone")
    p.add_argument("source")
    p.add_argument("--keep-clone", action="store_true", help="keep the cloned repository on disk")
    p.set_defaults(func=cmd_remove)

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

    p = sub.add_parser("serve", help="run the local web app in your browser")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("desktop", help="run the desktop app window")
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_desktop)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
