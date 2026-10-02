SYSTEM = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a description of a programming task or code behaviour, judge whether the code snippet implements it"


class QwenReranker:
    def __init__(self, model: str, device: str = "cuda", max_length: int = 4096, batch_size: int = 4, instruction: str = INSTRUCTION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(SYSTEM, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def _batch(self, query: str, documents: list[str]):
        texts = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        encoded = self.tokenizer(texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=budget, add_special_tokens=False)
        encoded["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]]
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(documents), self.batch_size):
                batch = self._batch(query, documents[start:start + self.batch_size])
                logits = self.model(**batch).logits[:, -1, :]
                pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
                scores.extend(self.torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist())
        return scores


def load_reranker(name: str | None, device: str):
    if not name or device != "cuda":
        return None
    return QwenReranker(name, device=device)
