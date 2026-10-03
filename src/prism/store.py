import json
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sources (source_id TEXT PRIMARY KEY, kind TEXT, location TEXT);
CREATE TABLE IF NOT EXISTS versions (version_id TEXT PRIMARY KEY, source_id TEXT, label TEXT, ref TEXT, ordinal INTEGER, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS version_files (version_id TEXT, path TEXT, blob_sha TEXT, PRIMARY KEY (version_id, path));
CREATE TABLE IF NOT EXISTS blobs (blob_sha TEXT PRIMARY KEY, language TEXT, n_chunks INTEGER);
CREATE TABLE IF NOT EXISTS blob_chunks (blob_sha TEXT, ordinal INTEGER, chunk_hash TEXT, symbol TEXT, kind TEXT, start_line INTEGER, end_line INTEGER, PRIMARY KEY (blob_sha, ordinal));
CREATE TABLE IF NOT EXISTS chunks (chunk_hash TEXT PRIMARY KEY, language TEXT, symbol TEXT, code TEXT);
CREATE TABLE IF NOT EXISTS embeddings (chunk_hash TEXT, space TEXT, vector BLOB, PRIMARY KEY (chunk_hash, space));
CREATE INDEX IF NOT EXISTS idx_version_files_blob ON version_files (blob_sha);
CREATE INDEX IF NOT EXISTS idx_blob_chunks_chunk ON blob_chunks (chunk_hash);
CREATE INDEX IF NOT EXISTS idx_versions_source ON versions (source_id);
"""


class Store:
    def __init__(self, path: str, chunker_config: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        if chunker_config is not None:
            self._check_meta("chunker", json.dumps(chunker_config, sort_keys=True))

    def _check_meta(self, key: str, value: str) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO meta VALUES (?, ?)", (key, value))
            self.db.commit()
        elif row[0] != value:
            raise ValueError(f"index was built with a different {key} config: {row[0]}")

    def add_source(self, source_id: str, kind: str, location: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?)", (source_id, kind, location))
        self.db.commit()

    def remove_source(self, source_id: str) -> int:
        db = self.db
        versions = [r[0] for r in db.execute("SELECT version_id FROM versions WHERE source_id = ?", (source_id,))]
        for version_id in versions:
            db.execute("DELETE FROM version_files WHERE version_id = ?", (version_id,))
        db.execute("DELETE FROM versions WHERE source_id = ?", (source_id,))
        if db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'lineage_members'").fetchone():
            db.execute("DELETE FROM lineage_members WHERE source_id = ?", (source_id,))
        db.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
        db.execute("DELETE FROM blob_chunks WHERE blob_sha NOT IN (SELECT DISTINCT blob_sha FROM version_files)")
        db.execute("DELETE FROM blobs WHERE blob_sha NOT IN (SELECT DISTINCT blob_sha FROM version_files)")
        db.execute("DELETE FROM chunks WHERE chunk_hash NOT IN (SELECT DISTINCT chunk_hash FROM blob_chunks)")
        db.execute("DELETE FROM embeddings WHERE chunk_hash NOT IN (SELECT chunk_hash FROM chunks)")
        db.commit()
        return len(versions)

    def sources(self) -> list[tuple[str, str, str]]:
        return self.db.execute("SELECT source_id, kind, location FROM sources ORDER BY source_id").fetchall()

    def source(self, source_id: str) -> tuple[str, str, str] | None:
        return self.db.execute("SELECT source_id, kind, location FROM sources WHERE source_id = ?", (source_id,)).fetchone()

    def has_blob(self, blob_sha: str) -> bool:
        return self.db.execute("SELECT 1 FROM blobs WHERE blob_sha = ?", (blob_sha,)).fetchone() is not None

    def add_blob_chunks(self, blob_sha: str, language: str, chunks) -> int:
        new_chunks = 0
        for ordinal, chunk in enumerate(chunks):
            digest = chunk.hash
            self.db.execute(
                "INSERT OR REPLACE INTO blob_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (blob_sha, ordinal, digest, chunk.symbol, chunk.kind, chunk.start_line, chunk.end_line),
            )
            cursor = self.db.execute("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?)", (digest, chunk.language, chunk.symbol, chunk.code))
            new_chunks += cursor.rowcount
        self.db.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)", (blob_sha, language, len(chunks)))
        return new_chunks

    def chunks_without_embeddings(self, space: str) -> list[tuple[str, str, str]]:
        return self.db.execute(
            "SELECT c.chunk_hash, c.symbol, c.code FROM chunks c LEFT JOIN embeddings e ON e.chunk_hash = c.chunk_hash AND e.space = ? WHERE e.chunk_hash IS NULL ORDER BY c.chunk_hash",
            (space,),
        ).fetchall()

    def add_embeddings(self, space: str, hashes: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self.db.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
            [(h, space, v.tobytes()) for h, v in zip(hashes, vectors)],
        )
        self.db.commit()

    def embeddings(self, space: str) -> tuple[list[str], np.ndarray | None]:
        rows = self.db.execute("SELECT chunk_hash, vector FROM embeddings WHERE space = ? ORDER BY rowid", (space,)).fetchall()
        if not rows:
            return [], None
        return [r[0] for r in rows], np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    def add_version(self, version_id: str, source_id: str, label: str, ref: str, ordinal: int, files: list[tuple[str, str]]) -> None:
        self.db.execute("DELETE FROM version_files WHERE version_id = ?", (version_id,))
        self.db.executemany("INSERT INTO version_files VALUES (?, ?, ?)", [(version_id, p, s) for p, s in files])
        self.db.execute(
            "INSERT OR REPLACE INTO versions VALUES (?, ?, ?, ?, ?, ?)",
            (version_id, source_id, label, ref, ordinal, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.db.commit()

    def versions(self, source_id: str) -> list[tuple[str, str, str, int]]:
        return self.db.execute("SELECT version_id, label, ref, ordinal FROM versions WHERE source_id = ? ORDER BY ordinal, indexed_at", (source_id,)).fetchall()

    def version_chunk_hashes(self, version_id: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT bc.chunk_hash FROM version_files vf JOIN blob_chunks bc ON bc.blob_sha = vf.blob_sha WHERE vf.version_id = ?",
            (version_id,),
        )]

    def occurrences(self, chunk_hash: str, version_ids: list[str] | None = None) -> list[tuple[str, str, str, int, int]]:
        sql = (
            "SELECT vf.version_id, vf.path, bc.symbol, bc.start_line, bc.end_line FROM blob_chunks bc "
            "JOIN version_files vf ON vf.blob_sha = bc.blob_sha WHERE bc.chunk_hash = ?"
        )
        params: list = [chunk_hash]
        if version_ids:
            sql += f" AND vf.version_id IN ({','.join('?' * len(version_ids))})"
            params += version_ids
        return self.db.execute(sql, params).fetchall()

    def chunk(self, chunk_hash: str) -> tuple[str, str, str] | None:
        return self.db.execute("SELECT language, symbol, code FROM chunks WHERE chunk_hash = ?", (chunk_hash,)).fetchone()

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
