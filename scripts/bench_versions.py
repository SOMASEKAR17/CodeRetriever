import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.cli import build_engine


def main():
    parser = argparse.ArgumentParser(description="Benchmark incremental indexing across versions of a Git repository")
    parser.add_argument("--repo", default="https://github.com/psf/requests.git")
    parser.add_argument("--refs", nargs="+", default=["v2.30.0", "v2.31.0", "v2.32.0", "v2.32.3"])
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--queries", nargs="*", default=["how are redirects followed and limited", "where are proxy settings read from the environment"])
    parser.add_argument("--embedder", choices=["model", "hash"], default="model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    engine_args = argparse.Namespace(config=None, index_dir=args.index_dir, embedder=args.embedder, checkpoint=None, space=None, device=args.device, dtype="auto", no_rerank=True)
    engine = build_engine(engine_args)
    source = engine.add_git(args.repo, args.source_id)
    rows = []
    for ref in args.refs:
        rows.extend(s.as_dict() for s in engine.index(source, [ref]))
    searches = []
    for query in args.queries:
        started = time.time()
        intent, hits = engine.search_evolution(query, source, k=5)
        searches.append({"query": query, "ms": round((time.time() - started) * 1000, 1), "top": [f"{h.symbol} @ {h.path} [{h.label}]" for h in hits[:3]]})

    lines = ["| Version | Files | Chunks | Newly embedded | Reused | Time (s) |", "|---|---|---|---|---|---|"]
    for r in rows:
        reused = 100 * (1 - r["chunks_embedded"] / r["chunks_in_version"]) if r["chunks_in_version"] else 0.0
        lines.append(f"| {r['label']} | {r['files']} | {r['chunks_in_version']} | {r['chunks_embedded']} | {max(0.0, reused):.1f}% | {r['seconds_total']} |")
    first, rest = rows[0], rows[1:]
    if rest:
        speedup = first["seconds_total"] / max(0.01, sum(r["seconds_total"] for r in rest) / len(rest))
        lines.append("")
        lines.append(f"Average incremental re-index is {speedup:.1f}x faster than the first full index.")
    report = "\n".join(lines)
    print(report)
    for s in searches:
        print(f"\n{s['query']} ({s['ms']} ms)\n  " + "\n  ".join(s["top"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "searches": searches, "markdown": report}, f, indent=2)


if __name__ == "__main__":
    main()
