import hashlib


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def normalize_code(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def chunk_hash(symbol: str, code: str) -> str:
    payload = f"{symbol}\n{normalize_code(code)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def merkle_root(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha1()
    for path, blob_sha in sorted(entries):
        digest.update(f"{path}\0{blob_sha}\n".encode("utf-8"))
    return digest.hexdigest()
