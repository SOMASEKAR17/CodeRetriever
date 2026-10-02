import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.hashes: list[str] = []
        self.row_of: dict[str, int] = {}
        self._matrix = np.zeros((1024, dim), dtype=np.float32)
        self._masks: dict[str, np.ndarray] = {}

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix[: len(self.hashes)]

    def add(self, hashes: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        fresh = [(h, v) for h, v in zip(hashes, vectors) if h not in self.row_of]
        if not fresh:
            return
        needed = len(self.hashes) + len(fresh)
        if needed > self._matrix.shape[0]:
            grown = np.zeros((max(needed, 2 * self._matrix.shape[0]), self.dim), dtype=np.float32)
            grown[: len(self.hashes)] = self.matrix
            self._matrix = grown
        for h, v in fresh:
            self.row_of[h] = len(self.hashes)
            self._matrix[len(self.hashes)] = v
            self.hashes.append(h)
        self._masks.clear()

    def clear_masks(self) -> None:
        self._masks.clear()

    def mask(self, key: str, chunk_hashes: list[str]) -> np.ndarray:
        cached = self._masks.get(key)
        if cached is not None and cached.shape[0] == len(self.hashes):
            return cached
        mask = np.zeros(len(self.hashes), dtype=bool)
        rows = [self.row_of[h] for h in chunk_hashes if h in self.row_of]
        mask[rows] = True
        self._masks[key] = mask
        return mask

    def search(self, query_vec: np.ndarray, mask: np.ndarray | None = None, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if not self.hashes:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        scores = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
        k = min(k, len(scores))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]
