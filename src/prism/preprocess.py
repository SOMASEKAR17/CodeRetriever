import re

SECTION = re.compile(r"^\s*-{3,}\s*([A-Za-z][A-Za-z ]*?)\s*-{3,}\s*$", re.MULTILINE)
SAMPLE_HEADER = re.compile(r"^\s*(?:sample\s+)?(input|output)(?:\s*\d+)?\s*:?\s*$", re.IGNORECASE)


def split_sections(text: str) -> dict[str, str]:
    matches = list(SECTION.finditer(text))
    if not matches:
        return {"statement": text.strip()}
    sections = {"statement": text[: matches[0].start()].strip()}
    for i, match in enumerate(matches):
        name = match.group(1).strip().lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if name.startswith("example"):
            name = "examples"
        sections[name] = (sections.get(name, "") + "\n\n" + body).strip()
    return sections


def query_views(text: str) -> dict[str, str]:
    sections = split_sections(text)
    statement = sections.get("statement") or text
    io_parts = [sections[k] for k in ("input", "output") if sections.get(k)]
    return {
        "full": text,
        "core": statement.strip() or text,
        "io": "\n\n".join(io_parts) if io_parts else text,
    }


def parse_examples(text: str) -> list[tuple[str, str]]:
    sections = split_sections(text)
    block = sections.get("examples")
    if not block:
        return []
    pairs: list[tuple[str, str]] = []
    mode = None
    current = {"input": [], "output": []}
    for line in block.split("\n"):
        header = SAMPLE_HEADER.match(line)
        if header:
            kind = header.group(1).lower()
            if kind == "input" and mode == "output":
                pairs.append(("\n".join(current["input"]).strip("\n"), "\n".join(current["output"]).strip("\n")))
                current = {"input": [], "output": []}
            mode = kind
            continue
        if mode:
            current[mode].append(line)
    if mode == "output":
        pairs.append(("\n".join(current["input"]).strip("\n"), "\n".join(current["output"]).strip("\n")))
    return [(i + "\n", o) for i, o in pairs if o.strip()]


def weighted_views(text: str, weights: dict[str, float]) -> list[tuple[str, float]]:
    views = query_views(text)
    return [(views[name], float(weight)) for name, weight in weights.items() if weight and name in views]
