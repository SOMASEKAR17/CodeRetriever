import hashlib
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .preprocess import parse_examples


def _tokens_match(actual: str, expected: str) -> bool:
    got, want = actual.split(), expected.split()
    if got == want:
        return True
    if len(got) != len(want):
        return False
    for a, b in zip(got, want):
        if a == b:
            continue
        try:
            if abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(b))):
                return False
        except ValueError:
            return False
    return True


def run_program(code: str, stdin: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="prism-exec-") as workdir:
        script = Path(workdir) / "solution.py"
        script.write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONIOENCODING": "utf-8"}
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                input=stdin.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                cwd=workdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, ""
        return result.returncode == 0, result.stdout.decode("utf-8", errors="replace")


class ExampleVerifier:
    def __init__(self, timeout: float = 2.0, workers: int = 4):
        self.timeout = timeout
        self.workers = workers
        self.cache: dict[tuple[str, str], float] = {}

    def score(self, code: str, examples: list[tuple[str, str]]) -> float:
        key = (hashlib.sha1(code.encode()).hexdigest(), hashlib.sha1(repr(examples).encode()).hexdigest())
        if key not in self.cache:
            passed = 0
            for stdin, expected in examples:
                ok, out = run_program(code, stdin, self.timeout)
                if ok and _tokens_match(out, expected):
                    passed += 1
                else:
                    break
            self.cache[key] = passed / len(examples)
        return self.cache[key]

    def rerank(self, query: str, candidates: list[tuple[str, str, float]], top_k: int = 20) -> tuple[list[tuple[str, float]], dict]:
        examples = parse_examples(query)
        ranked = [(doc_id, score) for doc_id, _, score in candidates]
        if not examples:
            return ranked, {"examples": 0, "verified": 0}
        head, tail = candidates[:top_k], candidates[top_k:]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            fractions = list(pool.map(lambda c: self.score(c[1], examples), head))
        order = sorted(range(len(head)), key=lambda i: (-(fractions[i] == 1.0), -fractions[i], i))
        top_score = max((s for _, _, s in candidates), default=0.0)
        reranked = [(head[i][0], top_score + 1.0 + fractions[i] - 0.001 * rank) if fractions[i] == 1.0 else (head[i][0], head[i][2]) for rank, i in enumerate(order)]
        passing = [r for r in reranked if r[1] > top_score]
        others = sorted([r for r in reranked if r[1] <= top_score], key=lambda r: -r[1])
        return passing + others + [(doc_id, score) for doc_id, _, score in tail], {"examples": len(examples), "verified": len(passing)}
