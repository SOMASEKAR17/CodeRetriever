import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.types import PromptType

from .execution import ExampleVerifier
from .preprocess import query_views


def load_base(model_name: str, device: str, checkpoint: str | None = None, dtype: str = "fp32", max_length: int = 2048):
    import mteb
    import torch
    from sentence_transformers import SentenceTransformer

    base = mteb.get_model(model_name, device=device)
    if checkpoint:
        base.model = SentenceTransformer(checkpoint, device=device)
    base.model.max_seq_length = max_length
    if dtype == "bf16":
        base.model.to(torch.bfloat16)
    return base


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, base, view_weights: dict[str, float] | None = None, prompts: dict | None = None, name: str = "prism/code-retriever", revision: str = "local"):
        self.base = base
        self.view_weights = {k: float(v) for k, v in (view_weights or {"full": 1.0}).items() if v}
        self.mteb_model_meta = base.mteb_model_meta.model_copy(update={"name": name, "revision": revision})
        self.model_prompts = dict(base.model_prompts or {})
        if prompts:
            self.model_prompts["AppsRetrieval-query"] = prompts["query"]
            self.model_prompts["AppsRetrieval-document"] = prompts["document"]
        self.base.model_prompts = self.model_prompts

    def similarity(self, embeddings1, embeddings2):
        return self.base.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1, embeddings2):
        return self.base.similarity_pairwise(embeddings1, embeddings2)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        if prompt_type != PromptType.query or set(self.view_weights) == {"full"}:
            return self.base.encode(inputs, task_metadata=task_metadata, hf_split=hf_split, hf_subset=hf_subset, prompt_type=prompt_type, **kwargs)
        texts = [text for batch in inputs for text in batch["text"]]
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        prompt = self.model_prompts.get(prompt_name, "") if prompt_name else ""
        batch_size = kwargs.get("batch_size", 8)
        total = None
        for view, weight in self.view_weights.items():
            view_texts = [query_views(t)[view] for t in texts]
            vectors = self.base.model.encode(view_texts, prompt=prompt, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            total = weight * vectors if total is None else total + weight * vectors
        return total / np.linalg.norm(total, axis=1, keepdims=True)


class PipelineSearch:
    def __init__(self, encoder: PrePostPipelineEncoder, verifier: ExampleVerifier | None = None, rerank_depth: int = 20):
        self.encoder = encoder
        self.inner = SearchEncoderWrapper(encoder)
        self.verifier = verifier
        self.rerank_depth = rerank_depth
        self.doc_text: dict[str, str] = {}

    @property
    def mteb_model_meta(self):
        return self.encoder.mteb_model_meta

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None):
        self.doc_text = dict(zip(corpus["id"], corpus["text"]))
        return self.inner.index(corpus, task_metadata=task_metadata, hf_split=hf_split, hf_subset=hf_subset, encode_kwargs=encode_kwargs, num_proc=num_proc)

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs, top_ranked=None, num_proc=None):
        depth = max(top_k, self.rerank_depth)
        results = self.inner.search(queries, task_metadata=task_metadata, hf_split=hf_split, hf_subset=hf_subset, top_k=depth, encode_kwargs=encode_kwargs, top_ranked=top_ranked, num_proc=num_proc)
        if not self.verifier:
            return results
        query_text = dict(zip(queries["id"], queries["text"]))
        reranked = {}
        for qid, scores in results.items():
            ordered = sorted(scores.items(), key=lambda kv: -kv[1])
            candidates = [(doc_id, self.doc_text.get(doc_id, ""), score) for doc_id, score in ordered]
            ranked, _ = self.verifier.rerank(query_text.get(qid, ""), candidates, self.rerank_depth)
            reranked[qid] = dict(ranked[:top_k])
        return reranked
