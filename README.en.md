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

## Purpose

Built to make **Korean PDF documents usable by small language models (SLMs)**.

Small models have short contexts and tend to answer with whatever sits near the front of
the input. Korean public-sector reports are table-heavy, and one report repeats similar
tables and figures many times. So even when retrieval returns relevant chunks, **if the
chunk holding the answer is not near the front**, the model copies a figure from some
other chunk instead.

This adapter targets exactly that — it moves the chunk that holds the answer to the front.

### Where it fits

- Question answering over Korean public-sector and government reports
- Tasks that require reading a value out of a table
- RAG with small language models (a few billion parameters), where evidence position matters most

### Where it does not fit

- Documents in other languages. The base model is multilingual, so it will run, but this adapter was
  trained on Korean only
- Prose documents without tables. The training data is table-centric
- Replacing retrieval. This adapter only **reorders candidates that retrieval already returned**

## Training data

| Item | Detail |
|---|---|
| Type | **Korean public-sector PDF reports** |
| Sources | National Assembly Budget Office (nabo), Korea Rural Economic Institute (krei), Korea Transport Institute (koti) |
| How they were obtained | Downloaded directly from each institute's own public website; all are openly published reports |
| Nature | Budget and settlement analyses, industry statistics, transport surveys — **table-heavy policy reports** |
| Labels | For each question, the chunks containing the supporting evidence were **marked by a human** |
| Training items | 264 |
| Splits | **Document-level** — chunks from one document never appear in both train and validation |

The training items and source documents are **not** included in this repository.

## Install and use

```bash
pip install torch transformers peft
```

### 1) Attach the adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # the model/ directory in this repo
model.eval()
```

### 2) Score and reorder

```python
def rerank(question, chunks, batch_size=8, max_length=1024):
    """Return the chunks sorted by score, highest first."""
    scores = []
    for i in range(0, len(chunks), batch_size):
        part = chunks[i:i + batch_size]
        enc = tok([question] * len(part), part,
                  padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            scores.extend(model(**enc).logits.view(-1).float().tolist())
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [(chunks[i], scores[i]) for i in order]
```

### 3) In a RAG pipeline

```python
ranked = rerank(question, retrieved_chunks)
top_k  = [text for text, _ in ranked[:28]]
prompt = "Question: " + question + "\n\nEvidence:\n" + "\n\n".join(top_k)
```

### Practical notes

| | |
|---|---|
| **Fix the batch size** | Changing the batch changes the padding length within a batch, which can perturb scores slightly. Keep it fixed for reproducibility |
| **max_length 1024** | This is the value used during training. Changing it far from this changes the conditions |
| **Pair order** | Feed the tokenizer `(question, chunk)` in that order. Swapping them changes the scores |
| **Score semantics** | The absolute value carries no meaning. Use only the **relative ranking within one question** |
| **Reordering only** | The adapter never deletes or rewrites chunks. Truncation is the caller's decision |

### Dropping it into an existing BGE setup

If you already use `bge-reranker-v2-m3`, **one line is all you add.**

```python
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # <- this line

model = model.unload()    # detach to return to the stock public model
```

## Evaluation — tested with Gemma 4 E2B

The generator is **`google/gemma-4-E2B-it` (2.3B effective, GGUF Q8_0)**, served with
`llama-server` at context 10240, `temperature 0`, 99 GPU layers, 4 threads, on an
AWS g6.xlarge (NVIDIA L4).

The evaluation set is **81 questions**. Each one is a case where E2B answered wrongly on
the plain-RAG input **and the wrong value came from another chunk in that same input** —
one that was not the labelled evidence. The answer was present; the model read the wrong
chunk.
The baseline scores **0 / 81** by construction — questions it already answered
correctly were excluded from the set. So each row's score is also its net gain over the
baseline.

| Reranker | Correct | Recovery |
|---|---|---|
| none (plain RAG) | 0 / 81 | 0.0% |
| public `BAAI/bge-reranker-v2-m3` | 17 / 81 | 21.0% |
| **this adapter (DKU)** | **58 / 81** | **71.6%** |

Same 81 questions, same candidate chunks, same generation settings — **only the reranker
changed.** That is **41 more questions (+50.6pp)** than the public model
(McNemar exact, two-sided, `p = 2.46e-10`).

### Why it rises

What differed is how often the labelled evidence was placed **first** in the input.

| Reranker | Evidence ranked first |
|---|---|
| none | 1 / 81 |
| public BGE | 18 / 81 |
| **this adapter** | **79 / 81** |

On the **61 questions whose evidence rank changed**, accuracy went from 6/61 (10%) to
47/61 (77%). On the **18 whose rank did not change**, it was 11/18 (61%) either way.
⇒ The cause is the ranking, not question difficulty.

⛔ Keeping only the top 28 chunks, and placing the question ahead of the evidence, showed
**no detectable difference** on these 81 questions (p = 1 each). The one confirmed effect
is the trained reranker.

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

| | |
|---|---|
| Author | **YANG SEONG HUN (양성훈)** |
| Department | **Department of IT Convergence** (IT컨버젼스학과) — <https://cms.dankook.ac.kr/web/gict/it-2> |
| Graduate school | Graduate School of Information Convergence Technology and Entrepreneurship, **Dankook University** |
| Homepage | <https://cms.dankook.ac.kr/web/gict> |

This adapter was trained and released as part of a master's thesis project in that
department. `DKU` stands for Dankook University.

## License

This adapter is a derivative of `BAAI/bge-reranker-v2-m3` and is released under the same
**Apache License 2.0**. The original model's license and attribution must be preserved.

## Citation

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  school = {Dankook University, Dept. of IT Convergence},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```
