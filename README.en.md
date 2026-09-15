# DKU-reranker-v1

[한국어](README.md) | **English**

A reranker LoRA adapter trained on **gold-evidence labels** from Korean public-sector PDF documents.
The backbone weights of [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
are **frozen**; only the LoRA adapter is trained.

`DKU` stands for Dankook University.

## What it does

Given a question and a set of retrieved chunks, it **moves the chunks that actually contain the
answer to the front.** Small language models extract values reliably when the supporting evidence
appears early in the input, so this ordering changes answer accuracy.

## What it is not

- It is not a new generative model.
- It is not a reranker trained from scratch.
- It is **the public BGE reranker plus a LoRA adapter trained on Korean data.**

## Install and use

```bash
pip install torch transformers peft
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # attach the DKU adapter
model.eval()

question = "..."
chunks = ["...chunk 1...", "...chunk 2...", "...chunk 3..."]

enc = tok([question] * len(chunks), chunks,
          padding=True, truncation=True, max_length=1024, return_tensors="pt")
with torch.no_grad():
    scores = model(**enc).logits.view(-1).float()

order = scores.argsort(descending=True)
ranked = [chunks[i] for i in order]      # highest score first
```

Note: on some devices a very large batch changes the scores. Scoring in **batches of 8** is recommended.

## Training

| Item | Value |
|---|---|
| Base model | `BAAI/bge-reranker-v2-m3` (backbone frozen) |
| Trained part | LoRA adapter (r=8, alpha=16, dropout=0.05) |
| Target modules | `key`, `query`, `value` |
| Training signal | Human-verified **gold-evidence labels** |
| Loss | A term raising the score of gold evidence, plus a term ordering gold evidence above distractors (weight 1:1.0) |
| Training items | 264 (D 98, F 166) |
| Epochs / learning rate / seed | 5 / 0.0001 / 20260915 |
| Max length / steps | 1024 / 980 |
| Hardware | NVIDIA L4 (AWS g6.xlarge) |

### Metrics before and after training

| Metric | Before | After (epoch 5) |
|---|---|---|
| Share of items where every gold chunk outranks every distractor | 0.4796 | **0.7959** |
| Pairwise order-violation rate (lower is better) | 0.2866 | **0.1311** |

The rule for choosing which epoch's adapter to keep was **fixed before training began**:
(1) drop any epoch whose order-violation rate exceeds the pre-training baseline;
(2) among the rest, take the epoch with the highest share above;
(3) break ties toward the earlier epoch.

Both metrics are **fit on the training set**, not downstream task performance. They are used only to
select the adapter. Full records are in `docs/train_record.json`.

## Code

```
src/build_c2_trainset.py    build the training set
src/make_doc_folds_v2.py    document-level splits
src/train_c2_targeted.py    LoRA training
```

The training items and source documents are **not** included in this repository.

## Affiliation

**Graduate School of Information Convergence Technology and Startup**, Dankook University
<https://cms.dankook.ac.kr/web/gict>

This adapter was trained and released as part of a master's thesis project at the school above.
`DKU` stands for Dankook University.

## License

This adapter is a derivative of `BAAI/bge-reranker-v2-m3` and is released under the same
**Apache License 2.0**. The original model's license and attribution must be preserved.

## Citation

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Kim, Jinseok},
  year   = {2026},
  school = {Graduate School of Information Convergence Technology and Startup, Dankook University},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```
