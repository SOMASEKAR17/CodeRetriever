import ast
from dataclasses import dataclass

from .hashing import chunk_hash, normalize_code

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rs": "rust",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
}


@dataclass
class Chunk:
    symbol: str
    kind: str
    start_line: int
    end_line: int
    code: str
    language: str

    @property
    def hash(self) -> str:
        return chunk_hash(self.symbol, self.code)


def language_of(path: str) -> str | None:
    lower = path.lower()
    for suffix, language in LANGUAGE_BY_SUFFIX.items():
        if lower.endswith(suffix):
            return language
    return None


class Chunker:
    def __init__(self, max_lines: int = 200, window_overlap: int = 20, whole_file_lines: int = 0):
        self.max_lines = max_lines
        self.window_overlap = window_overlap
        self.whole_file_lines = whole_file_lines

    def split(self, path: str, text: str) -> list[Chunk]:
        language = language_of(path) or "text"
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        if not normalize_code(text):
            return []
        if self.whole_file_lines and len(lines) <= self.whole_file_lines:
            return [Chunk("<file>", "file", 1, len(lines), text, language)]
        if language == "python":
            chunks = self._split_python(text, lines)
            if chunks is not None:
                return chunks
        return self._windows("<file>", lines, 1, len(lines), "window", language)

    def _split_python(self, text: str, lines: list[str]) -> list[Chunk] | None:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return None
        chunks: list[Chunk] = []
        covered: set[int] = set()
        self._visit(tree.body, "", lines, chunks, covered)
        block = 0
        run: list[int] = []
        for line_no in list(range(1, len(lines) + 1)) + [None]:
            if line_no is not None and line_no not in covered:
                run.append(line_no)
                continue
            if run and any(lines[i - 1].strip() for i in run):
                first = next(i for i in run if lines[i - 1].strip())
                last = max(i for i in run if lines[i - 1].strip())
                symbol = "<module>" if block == 0 else f"<module>:{block}"
                chunks.extend(self._windows(symbol, lines, first, last, "module", "python"))
                block += 1
            run = []
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    def _visit(self, nodes, prefix: str, lines: list[str], chunks: list[Chunk], covered: set[int]) -> None:
        for node in nodes:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            end = node.end_lineno or node.lineno
            name = f"{prefix}{node.name}"
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if isinstance(node, ast.ClassDef):
                members = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                if members and end - start + 1 > self.max_lines:
                    first_member = min(min([d.lineno for d in m.decorator_list] + [m.lineno]) for m in members)
                    header_end = max(start, first_member - 1)
                    chunks.extend(self._windows(name, lines, start, header_end, "class", "python"))
                    covered.update(range(start, header_end + 1))
                    self._visit(node.body, f"{name}.", lines, chunks, covered)
                    covered.update(range(start, end + 1))
                    continue
            chunks.extend(self._windows(name, lines, start, end, kind, "python"))
            covered.update(range(start, end + 1))

    def _windows(self, symbol: str, lines: list[str], start: int, end: int, kind: str, language: str) -> list[Chunk]:
        out: list[Chunk] = []
        step = max(1, self.max_lines - self.window_overlap)
        position = start
        part = 0
        while position <= end:
            stop = min(end, position + self.max_lines - 1)
            code = "\n".join(lines[position - 1:stop])
            if normalize_code(code):
                label = symbol if (position == start and stop == end) else f"{symbol}#{part}"
                out.append(Chunk(label, kind, position, stop, code, language))
            part += 1
            if stop == end:
                break
            position += step
        return out
