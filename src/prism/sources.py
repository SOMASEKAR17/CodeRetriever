import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .chunker import language_of
from .hashing import git_blob_sha, merkle_root

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "dist", "build", "__pycache__", ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
MAX_FILE_BYTES = 1_000_000


@dataclass
class Version:
    label: str
    ref: str
    ordinal: int


def _is_wanted(path: str, size: int | None = None) -> bool:
    parts = path.split("/")
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return False
    if size is not None and size > MAX_FILE_BYTES:
        return False
    return language_of(path) is not None


class GitSource:
    kind = "git"

    def __init__(self, repo_path: str):
        self.repo = str(Path(repo_path).resolve())
        self._catfile = None

    def _git(self, *args: str) -> str:
        result = subprocess.run(["git", "-C", self.repo, *args], capture_output=True, check=True)
        return result.stdout.decode("utf-8", errors="replace")

    @staticmethod
    def clone(url: str, dest: str, partial: bool = False) -> "GitSource":
        command = ["git", "clone"]
        if partial:
            command += ["--filter=blob:none"]
        subprocess.run(command + [url, dest], check=True)
        return GitSource(dest)

    def resolve(self, ref: str) -> str:
        return self._git("rev-parse", f"{ref}^{{commit}}").strip()

    def list_versions(self, include_branches: bool = True) -> list[Version]:
        versions: list[Version] = []
        out = self._git("for-each-ref", "--sort=creatordate", "--format=%(refname:short)\t%(objectname)\t%(*objectname)\t%(creatordate:unix)", "refs/tags")
        for line in out.splitlines():
            if not line.strip():
                continue
            name, obj, peeled, _ = (line.split("\t") + ["", "", ""])[:4]
            commit = peeled or obj
            versions.append(Version(name, commit, self.depth(commit)))
        if include_branches:
            out = self._git("for-each-ref", "--sort=committerdate", "--format=%(refname:short)\t%(objectname)\t%(committerdate:unix)", "refs/heads")
            for line in out.splitlines():
                if line.strip():
                    name, obj, _ = (line.split("\t") + ["", ""])[:3]
                    versions.append(Version(name, obj, self.depth(obj)))
        return versions

    def depth(self, commit: str) -> int:
        return int(self._git("rev-list", "--count", commit).strip() or 0)

    def version(self, ref: str) -> Version:
        commit = self.resolve(ref)
        return Version(ref, commit, self.depth(commit))

    def list_files(self, version: Version) -> list[tuple[str, str]]:
        out = subprocess.run(["git", "-C", self.repo, "ls-tree", "-r", "-l", "-z", version.ref], capture_output=True, check=True).stdout
        files = []
        for entry in out.split(b"\0"):
            if not entry:
                continue
            meta, path = entry.split(b"\t", 1)
            mode, kind, sha, size = meta.split()
            path_text = path.decode("utf-8", errors="replace")
            if kind != b"blob" or mode in (b"120000", b"160000"):
                continue
            if _is_wanted(path_text, int(size) if size != b"-" else None):
                files.append((path_text, sha.decode()))
        return files

    def read_blob(self, blob_sha: str, path: str) -> str:
        if self._catfile is None or self._catfile.poll() is not None:
            self._catfile = subprocess.Popen(["git", "-C", self.repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._catfile.stdin.write(f"{blob_sha}\n".encode())
        self._catfile.stdin.flush()
        header = self._catfile.stdout.readline().split()
        if len(header) < 3 or header[1] != b"blob":
            raise KeyError(f"blob {blob_sha} not found")
        data = self._catfile.stdout.read(int(header[2]))
        self._catfile.stdout.read(1)
        return data.decode("utf-8", errors="replace")

    def changed_paths(self, old: Version, new: Version) -> list[tuple[str, str, str]]:
        out = self._git("diff", "-M", "--name-status", old.ref, new.ref)
        changes = []
        for line in out.splitlines():
            parts = line.split("\t")
            if parts[0].startswith("R") and len(parts) == 3:
                changes.append(("R", parts[1], parts[2]))
            elif len(parts) == 2:
                changes.append((parts[0][0], parts[1], parts[1]))
        return changes

    def close(self) -> None:
        if self._catfile is not None and self._catfile.poll() is None:
            self._catfile.stdin.close()
            self._catfile.wait(timeout=10)
            self._catfile.stdout.close()
        self._catfile = None


class FolderSource:
    kind = "folder"

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self._paths_by_blob: dict[str, Path] = {}

    def _walk(self) -> list[tuple[str, str]]:
        files = []
        for directory, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                full = Path(directory) / name
                rel = full.relative_to(self.root).as_posix()
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                if not _is_wanted(rel, size):
                    continue
                sha = git_blob_sha(full.read_bytes())
                self._paths_by_blob[sha] = full
                files.append((rel, sha))
        return sorted(files)

    def snapshot(self, label: str | None = None) -> tuple[Version, list[tuple[str, str]]]:
        files = self._walk()
        root = merkle_root(files)
        return Version(label or f"snapshot-{root[:12]}", root, int(time.time())), files

    def list_files(self, version: Version) -> list[tuple[str, str]]:
        files = self._walk()
        if merkle_root(files) != version.ref:
            raise ValueError("folder changed since this snapshot; take a new snapshot")
        return files

    def read_blob(self, blob_sha: str, path: str) -> str:
        full = self._paths_by_blob.get(blob_sha) or (self.root / path)
        data = full.read_bytes()
        if git_blob_sha(data) != blob_sha:
            raise ValueError(f"{path} changed while indexing")
        return data.decode("utf-8", errors="replace")


class CorpusSource:
    kind = "corpus"

    def __init__(self, files: list[str], id_field: str = "_id", text_field: str = "text", suffix: str = ".py"):
        self.files = [str(Path(f).resolve()) for f in files]
        self.id_field = id_field
        self.text_field = text_field
        self.suffix = suffix
        self._texts: dict[str, str] = {}

    def list_versions(self) -> list[Version]:
        return [Version(Path(f).stem, f, i) for i, f in enumerate(self.files)]

    def list_files(self, version: Version) -> list[tuple[str, str]]:
        files = []
        with open(version.ref, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record[self.text_field]
                sha = git_blob_sha(text.encode("utf-8"))
                self._texts[sha] = text
                files.append((f"{record[self.id_field]}{self.suffix}", sha))
        return files

    def read_blob(self, blob_sha: str, path: str) -> str:
        return self._texts[blob_sha]
