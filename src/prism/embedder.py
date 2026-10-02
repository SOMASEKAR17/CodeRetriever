import hashlib
import re

import numpy as np

DEFAULT_QUERY_PROMPT = "task: search result | query: "
DEFAULT_DOCUMENT_PROMPT = "title: {title} | text: "


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model: str,
        space: str,
        device: str = "cpu",
        dtype: str = "fp32",
        max_length: int = 2048,
        batch_size: int = 8,
        query_prompt: str = DEFAULT_QUERY_PROMPT,
        document_prompt: str = DEFAULT_DOCUMENT_PROMPT,
        revision: str | None = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.space = space
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.model = SentenceTransformer(model, device=device, revision=revision)
        self.model.max_seq_length = max_length
        if dtype == "bf16":
            self.model.to(torch.bfloat16)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prompt: str) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prompt)

    def embed_documents(self, codes: list[str], titles: list[str] | None = None) -> np.ndarray:
        titles = titles or ["none"] * len(codes)
        groups: dict[str, list[int]] = {}
        for i, title in enumerate(titles):
            groups.setdefault(self.document_prompt.format(title=title or "none"), []).append(i)
        out = np.zeros((len(codes), self.dim), dtype=np.float32)
        for prompt, indices in groups.items():
            out[indices] = self._encode([codes[i] for i in indices], prompt)
        return out


class HashEmbedder:
    def __init__(self, dim: int = 512, space: str = "hash-512"):
        self.dim = dim
        self.space = space

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % self.dim
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_documents(self, codes: list[str], titles: list[str] | None = None) -> np.ndarray:
        return self.embed_queries(codes)
