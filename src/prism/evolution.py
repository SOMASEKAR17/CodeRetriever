import difflib
import re
from dataclasses import dataclass, field

import numpy as np

LINEAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS lineage_members (source_id TEXT, version_id TEXT, lineage_id INTEGER, path TEXT, symbol TEXT, chunk_hash TEXT, start_line INTEGER, end_line INTEGER);
CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage_members (source_id, chunk_hash);
"""

CHANGE_PATTERN = re.compile(
    r"\b(change[sd]?|changing|added|removed|deleted|fixed|modified|history|evolv\w*|before|since|diff|differen\w*|previous|older|introduced|when (?:was|did|were)|refactor\w*)\b",
    re.IGNORECASE,
)
RENAME_SIMILARITY = 0.9


@dataclass
class Intent:
    kind: str
    version_id: str | None = None
    matched: str | None = None


@dataclass
class LineageHit:
    lineage_id: int
    score: float
    symbol: str
    path: str
    version_id: str
    label: str
    code: str
    start_line: int
    end_line: int
    timeline: list[dict] = field(default_factory=list)
    diff: str | None = None


def ensure_schema(store) -> None:
    store.db.executescript(LINEAGE_SCHEMA)


def detect_intent(query: str, versions: list[dict]) -> Intent:
    lowered = query.lower()
    for item in sorted(versions, key=lambda v: -len(v["label"])):
        label = item["label"].lower()
        if label and re.search(rf"(?<![\w.]){re.escape(label)}(?![\w.])", lowered):
            return Intent("version", item["version_id"], item["label"])
    for token in re.findall(r"\b[0-9a-f]{7,40}\b", lowered):
        for item in versions:
            if item["ref"].lower().startswith(token):
                return Intent("version", item["version_id"], token)
    if CHANGE_PATTERN.search(query):
        return Intent("change")
    return Intent("latest")


def _occurrences(store, version_id: str) -> list[tuple[str, str, str, int, int]]:
    return store.db.execute(
        "SELECT vf.path, bc.symbol, bc.chunk_hash, bc.start_line, bc.end_line FROM version_files vf "
        "JOIN blob_chunks bc ON bc.blob_sha = vf.blob_sha WHERE vf.version_id = ? ORDER BY vf.path, bc.ordinal",
        (version_id,),
    ).fetchall()


def rebuild_lineages(engine, source_id: str) -> int:
    store = engine.store
    ensure_schema(store)
    store.db.execute("DELETE FROM lineage_members WHERE source_id = ?", (source_id,))
    versions = sorted(engine.versions(source_id), key=lambda v: (v["ordinal"], v["version_id"]))
    source = engine.source(source_id)
    next_id = 0
    previous = None
    previous_by_key: dict[tuple[str, str], int] = {}
    previous_by_hash: dict[str, int] = {}
    previous_hash_of: dict[int, str] = {}
    rows = []
    for item in versions:
        renames: dict[str, str] = {}
        if previous is not None and hasattr(source, "changed_paths"):
            try:
                from .sources import Version

                old = Version(previous["label"], previous["ref"], previous["ordinal"])
                new = Version(item["label"], item["ref"], item["ordinal"])
                renames = {new_path: old_path for status, old_path, new_path in source.changed_paths(old, new) if status == "R"}
            except Exception:
                renames = {}
        current_by_key: dict[tuple[str, str], int] = {}
        current_by_hash: dict[str, int] = {}
        current_hash_of: dict[int, str] = {}
        unmatched = []
        used: set[int] = set()
        for path, symbol, chunk_hash, start, end in _occurrences(store, item["version_id"]):
            key = (path, symbol)
            lineage = previous_by_key.get(key)
            if lineage is None:
                lineage = previous_by_key.get((renames.get(path, path), symbol))
            if lineage is None:
                lineage = previous_by_hash.get(chunk_hash)
            if lineage is None or lineage in used:
                unmatched.append((path, symbol, chunk_hash, start, end))
                continue
            used.add(lineage)
            current_by_key[key] = lineage
            current_by_hash.setdefault(chunk_hash, lineage)
            current_hash_of[lineage] = chunk_hash
            rows.append((source_id, item["version_id"], lineage, path, symbol, chunk_hash, start, end))
        vanished = [lid for lid in previous_hash_of if lid not in used]
        for path, symbol, chunk_hash, start, end in unmatched:
            lineage = _closest_vanished(engine, chunk_hash, vanished, previous_hash_of)
            if lineage is None:
                lineage = next_id
                next_id += 1
            else:
                vanished.remove(lineage)
            current_by_key[(path, symbol)] = lineage
            current_by_hash.setdefault(chunk_hash, lineage)
            current_hash_of[lineage] = chunk_hash
            rows.append((source_id, item["version_id"], lineage, path, symbol, chunk_hash, start, end))
        next_id = max([next_id] + [r[2] + 1 for r in rows])
        previous = item
        previous_by_key, previous_by_hash, previous_hash_of = current_by_key, current_by_hash, current_hash_of
    store.db.executemany("INSERT INTO lineage_members VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    store.commit()
    return next_id


def _closest_vanished(engine, chunk_hash: str, vanished: list[int], hash_of: dict[int, str]) -> int | None:
    if not vanished or chunk_hash not in engine.vectors.row_of:
        return None
    vector = engine.vectors.matrix[engine.vectors.row_of[chunk_hash]]
    best, best_score = None, RENAME_SIMILARITY
    for lineage in vanished:
        other = hash_of.get(lineage)
        if other not in engine.vectors.row_of:
            continue
        score = float(np.dot(vector, engine.vectors.matrix[engine.vectors.row_of[other]]))
        if score >= best_score:
            best, best_score = lineage, score
    return best


def search_lineages(engine, query: str, source_id: str, k: int = 10, candidates: int = 200, intent: Intent | None = None) -> tuple[Intent, list[LineageHit]]:
    store = engine.store
    ensure_schema(store)
    versions = sorted(engine.versions(source_id), key=lambda v: (v["ordinal"], v["version_id"]))
    order = {v["version_id"]: i for i, v in enumerate(versions)}
    labels = {v["version_id"]: v["label"] for v in versions}
    intent = intent or detect_intent(query, versions)
    if intent.kind == "version":
        hits = engine.search(query, source_id, intent.version_id, k)
        out = []
        for hit in hits:
            occurrence = next((o for o in hit.occurrences if o["version_id"] == intent.version_id), hit.occurrences[0])
            out.append(LineageHit(-1, hit.score, hit.symbol, occurrence["path"], intent.version_id, labels.get(intent.version_id, ""), hit.code, occurrence["start_line"], occurrence["end_line"]))
        return intent, out
    hashes: set[str] = set()
    for item in versions:
        hashes.update(store.version_chunk_hashes(item["version_id"]))
    mask = engine.vectors.mask("all:" + source_id + ":" + str(len(versions)), list(hashes))
    query_vec = engine.embedder.embed_queries([query])[0]
    rows, scores = engine.vectors.search(query_vec, mask, candidates)
    score_of = {engine.vectors.hashes[r]: float(s) for r, s in zip(rows, scores)}
    if not score_of:
        return intent, []
    placeholders = ",".join("?" * len(score_of))
    members = store.db.execute(
        f"SELECT lineage_id, version_id, path, symbol, chunk_hash, start_line, end_line FROM lineage_members WHERE source_id = ? AND lineage_id IN "
        f"(SELECT DISTINCT lineage_id FROM lineage_members WHERE source_id = ? AND chunk_hash IN ({placeholders}))",
        [source_id, source_id, *score_of],
    ).fetchall()
    by_lineage: dict[int, list[tuple]] = {}
    for row in members:
        by_lineage.setdefault(row[0], []).append(row)
    ranked = []
    for lineage, items in by_lineage.items():
        items.sort(key=lambda r: order.get(r[1], 0))
        best = max(score_of.get(r[4], -1.0) for r in items)
        ranked.append((best, order.get(items[-1][1], 0), lineage, items))
    ranked.sort(key=lambda x: (-x[0], -x[1]))
    out = []
    for best, _, lineage, items in ranked[:k]:
        timeline = []
        previous_hash = None
        for row in items:
            timeline.append({
                "version_id": row[1],
                "label": labels.get(row[1], ""),
                "path": row[2],
                "chunk_hash": row[4],
                "changed": previous_hash is not None and row[4] != previous_hash,
                "score": round(score_of.get(row[4], float("nan")), 4) if row[4] in score_of else None,
            })
            previous_hash = row[4]
        latest = items[-1]
        if intent.kind == "change":
            changed = [i for i, t in enumerate(timeline) if t["changed"]]
            pick = items[changed[-1]] if changed else latest
        else:
            pick = latest
        _, symbol, code = store.chunk(pick[4])
        diff = None
        position = items.index(pick)
        if position > 0 and items[position - 1][4] != pick[4]:
            old_code = store.chunk(items[position - 1][4])[2]
            diff = "\n".join(difflib.unified_diff(
                old_code.splitlines(), code.splitlines(),
                fromfile=f"{labels.get(items[position - 1][1], '')}:{items[position - 1][2]}",
                tofile=f"{labels.get(pick[1], '')}:{pick[2]}", lineterm="",
            ))
        out.append(LineageHit(lineage, best, pick[3], pick[2], pick[1], labels.get(pick[1], ""), code, pick[5], pick[6], timeline, diff))
    return intent, out
