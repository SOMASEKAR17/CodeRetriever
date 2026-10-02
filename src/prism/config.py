import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

FALLBACK = {
    "index_dir": "~/.prism",
    "device": "auto",
    "space": "auto",
    "embedder": {
        "base_model": "google/embeddinggemma-300m",
        "base_revision": None,
        "checkpoint": None,
        "prompts_file": None,
        "query_prompt": "task: search result | query: ",
        "document_prompt": "title: {title} | text: ",
        "max_length": 2048,
        "batch_size": 8,
    },
    "profiles": {
        "cpu": {"dtype": "fp32", "reranker": None, "rerank_top_k": 0},
        "gpu": {"dtype": "bf16", "reranker": None, "rerank_top_k": 0},
    },
    "query_views": {"full": 1.0, "core": 0.0, "io": 0.0},
    "execution_rerank": {"top_k": 20, "timeout_seconds": 2.0, "workers": 4},
    "chunking": {"max_lines": 200, "window_overlap": 20},
    "server": {"host": "127.0.0.1", "port": 8765},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(os.path.expanduser(str(path_value)))
    return path if path.is_absolute() else (base_dir / path)


@dataclass
class ModelChoice:
    model: str
    revision: str | None
    space: str
    query_prompt: str
    document_prompt: str
    fine_tuned: bool


class Config:
    def __init__(self, data: dict, base_dir: Path):
        self.data = data
        self.base_dir = base_dir

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = Path(path or os.environ.get("PRISM_CONFIG") or DEFAULT_CONFIG)
        raw = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        data = _merge(FALLBACK, raw)
        if os.environ.get("PRISM_CHECKPOINT"):
            data["embedder"]["checkpoint"] = os.environ["PRISM_CHECKPOINT"]
            data["embedder"]["prompts_file"] = str(Path(os.environ["PRISM_CHECKPOINT"]) / "prompts.json")
        if os.environ.get("PRISM_DEVICE"):
            data["device"] = os.environ["PRISM_DEVICE"]
        if os.environ.get("PRISM_INDEX_DIR"):
            data["index_dir"] = os.environ["PRISM_INDEX_DIR"]
        return cls(data, path.parent if path.exists() else REPO_ROOT)

    def __getitem__(self, key):
        return self.data[key]

    @property
    def index_dir(self) -> Path:
        return _resolve(self.data["index_dir"], self.base_dir)

    def profile(self, device: str) -> dict:
        return self.data["profiles"]["gpu" if device == "cuda" else "cpu"]

    def model_choice(self) -> ModelChoice:
        import json

        emb = self.data["embedder"]
        checkpoint = _resolve(emb.get("checkpoint"), self.base_dir)
        prompts_file = _resolve(emb.get("prompts_file"), self.base_dir)
        query_prompt, document_prompt = emb["query_prompt"], emb["document_prompt"]
        fine_tuned = bool(checkpoint and (checkpoint / "modules.json").exists())
        if fine_tuned and prompts_file and prompts_file.exists():
            with open(prompts_file, encoding="utf-8") as f:
                prompts = json.load(f)
            query_prompt = prompts.get("query", query_prompt)
            document_prompt = prompts.get("document", document_prompt).replace("title: none", "title: {title}")
        model = str(checkpoint) if fine_tuned else emb["base_model"]
        revision = None if fine_tuned else emb.get("base_revision")
        space = self.data.get("space") or "auto"
        if space == "auto":
            space = checkpoint.name if fine_tuned else emb["base_model"].split("/")[-1]
        return ModelChoice(model, revision, space, query_prompt, document_prompt, fine_tuned)


def resolve_device(device: str) -> str:
    if device not in ("auto", None):
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
