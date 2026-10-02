# Model weights

Fine-tuned weights are not stored in git. Put the fine-tuned EmbeddingGemma checkpoint here:

```text
models/embeddinggemma-apps-ft/
├── config.json, model.safetensors, tokenizer files, modules.json, ...
└── prompts.json
```

`config.yaml` points `embedder.checkpoint` at this folder. While the folder is missing, CodeRetriever falls back to the base model `google/embeddinggemma-300m`, so everything still runs.

Produce the checkpoint with `scripts/finetune.py`, then copy its `final/` folder and `prompts.json` here.
