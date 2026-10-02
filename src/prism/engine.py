import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunker import Chunker, language_of
from .sources import CorpusSource, FolderSource, GitSource, Version
from .store import Store
from .vector_index import VectorIndex


@dataclass
class IndexStats:
    version_id: str
    label: str
    files: int
    blobs_new: int
    blobs_reused: int
    chunks_in_version: int
    chunks_new: int
    chunks_embedded: int
    seconds_parse: float
    seconds_embed: float
    seconds_total: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk_hash: str
    score: float
    symbol: str
    language: str
    code: str
    occurrences: list[dict] = field(default_factory=list)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "source"


class Engine:
    def __init__(self, index_dir: str, embedder, chunker: Chunker | None = None, reranker=None, rerank_top_k: int = 0):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = chunker or Chunker()
        config = {"max_lines": self.chunker.max_lines, "window_overlap": self.chunker.window_overlap, "whole_file_lines": self.chunker.whole_file_lines}
        self.store = Store(str(self.index_dir / "index.sqlite"), chunker_config=config)
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.vectors = VectorIndex(embedder.dim)
        hashes, matrix = self.store.embeddings(embedder.space)
        if hashes:
            self.vectors.add(hashes, matrix)
        self._sources: dict[str, object] = {}

    def add_git(self, location: str, source_id: str | None = None, partial: bool = False) -> str:
        is_url = location.startswith(("http://", "https://", "git@"))
        source_id = source_id or _slug(Path(location.rstrip("/")).name.removesuffix(".git"))
        if is_url:
            dest = self.index_dir / "repos" / source_id
            if not dest.exists():
                GitSource.clone(location, str(dest), partial=partial)
            location = str(dest)
        self.store.add_source(source_id, "git", str(Path(location).resolve()))
        return source_id

    def add_folder(self, location: str, source_id: str | None = None) -> str:
        source_id = source_id or _slug(Path(location).resolve().name)
        self.store.add_source(source_id, "folder", str(Path(location).resolve()))
        return source_id

    def add_corpus(self, files: list[str], source_id: str) -> str:
        self.store.add_source(source_id, "corpus", json.dumps([str(Path(f).resolve()) for f in files]))
        return source_id

    def source(self, source_id: str):
        if source_id in self._sources:
            return self._sources[source_id]
        row = self.store.source(source_id)
        if row is None:
            raise KeyError(f"unknown source {source_id}")
        _, kind, location = row
        if kind == "git":
            source = GitSource(location)
        elif kind == "folder":
            source = FolderSource(location)
        else:
            source = CorpusSource(json.loads(location))
        self._sources[source_id] = source
        return source

    def available_versions(self, source_id: str) -> list[Version]:
        source = self.source(source_id)
        if isinstance(source, FolderSource):
            return [source.snapshot()[0]]
        return source.list_versions()

    def index(self, source_id: str, refs: list[str] | None = None, progress=print) -> list[IndexStats]:
        source = self.source(source_id)
        stats = []
        if isinstance(source, FolderSource):
            version, files = source.snapshot()
            stats.append(self._index_version(source_id, source, version, files, progress))
        elif isinstance(source, GitSource):
            versions = [source.version(r) for r in refs] if refs else source.list_versions(include_branches=False)
            for version in versions:
                stats.append(self._index_version(source_id, source, version, source.list_files(version), progress))
        else:
            versions = source.list_versions()
            if refs:
                versions = [v for v in versions if v.label in refs]
            for version in versions:
                stats.append(self._index_version(source_id, source, version, source.list_files(version), progress))
        from .evolution import rebuild_lineages

        rebuild_lineages(self, source_id)
        return stats

    def _version_id(self, source_id: str, version: Version) -> str:
        if isinstance(self.source(source_id), CorpusSource):
            return f"{source_id}@{version.label}"
        return f"{source_id}@{version.ref[:12]}"

    def _index_version(self, source_id: str, source, version: Version, files: list[tuple[str, str]], progress) -> IndexStats:
        started = time.time()
        version_id = self._version_id(source_id, version)
        blobs_new = blobs_reused = chunks_new = 0
        seen: set[str] = set()
        for path, blob_sha in files:
            if blob_sha in seen:
                continue
            seen.add(blob_sha)
            if self.store.has_blob(blob_sha):
                blobs_reused += 1
                continue
            text = source.read_blob(blob_sha, path)
            chunks = self.chunker.split(path, text)
            chunks_new += self.store.add_blob_chunks(blob_sha, language_of(path) or "text", chunks)
            blobs_new += 1
        self.store.commit()
        parsed = time.time()
        missing = self.store.chunks_without_embeddings(self.embedder.space)
        batch = 256
        for start in range(0, len(missing), batch):
            part = missing[start:start + batch]
            vectors = self.embedder.embed_documents([r[2] for r in part], titles=["none"] * len(part))
            hashes = [r[0] for r in part]
            self.store.add_embeddings(self.embedder.space, hashes, vectors)
            self.vectors.add(hashes, vectors)
            progress(f"  embedded {min(start + batch, len(missing))}/{len(missing)} new chunks")
        embedded = time.time()
        self.store.add_version(version_id, source_id, version.label, version.ref, version.ordinal, files)
        self.vectors.clear_masks()
        in_version = len(self.store.version_chunk_hashes(version_id))
        result = IndexStats(
            version_id=version_id,
            label=version.label,
            files=len(files),
            blobs_new=blobs_new,
            blobs_reused=blobs_reused,
            chunks_in_version=in_version,
            chunks_new=chunks_new,
            chunks_embedded=len(missing),
            seconds_parse=round(parsed - started, 2),
            seconds_embed=round(embedded - parsed, 2),
            seconds_total=round(time.time() - started, 2),
        )
        progress(f"indexed {version.label} ({version_id}): {json.dumps(result.as_dict())}")
        return result

    def candidates(self, query: str, mask, k: int) -> list[tuple[str, float]]:
        query_vec = self.embedder.embed_queries([query])[0]
        depth = max(k, self.rerank_top_k) if self.reranker else k
        rows, scores = self.vectors.search(query_vec, mask, depth)
        ranked = [(self.vectors.hashes[r], float(s)) for r, s in zip(rows, scores)]
        if self.reranker and ranked:
            head = ranked[: self.rerank_top_k]
            codes = [self.store.chunk(h)[2] for h, _ in head]
            rerank_scores = self.reranker.score(query, codes)
            head = sorted(((h, 1.0 + r) for (h, _), r in zip(head, rerank_scores)), key=lambda x: -x[1])
            ranked = head + ranked[self.rerank_top_k:]
        return ranked[:k]

    def search_evolution(self, query: str, source_id: str, k: int = 10):
        from .evolution import search_lineages

        return search_lineages(self, query, source_id, k)

    def versions(self, source_id: str) -> list[dict]:
        return [{"version_id": v, "label": l, "ref": r, "ordinal": o} for v, l, r, o in self.store.versions(source_id)]

    def resolve_version(self, source_id: str, version: str) -> str:
        for item in self.versions(source_id):
            if version in (item["version_id"], item["label"]) or item["ref"].startswith(version):
                return item["version_id"]
        raise KeyError(f"version {version} of {source_id} is not indexed")

    def search(self, query: str, source_id: str, version: str | None = None, k: int = 10) -> list[Hit]:
        if version and version != "all":
            version_ids = [self.resolve_version(source_id, version)]
        else:
            version_ids = [v["version_id"] for v in self.versions(source_id)]
        key = "|".join(sorted(version_ids))
        hashes: set[str] = set()
        for version_id in version_ids:
            hashes.update(self.store.version_chunk_hashes(version_id))
        mask = self.vectors.mask(key, list(hashes))
        hits = []
        for chunk_hash, score in self.candidates(query, mask, k):
            language, symbol, code = self.store.chunk(chunk_hash)
            occurrences = [
                {"version_id": v, "path": p, "symbol": s, "start_line": a, "end_line": b}
                for v, p, s, a, b in self.store.occurrences(chunk_hash, version_ids)
            ]
            hits.append(Hit(chunk_hash, float(score), symbol, language, code, occurrences))
        return hits
