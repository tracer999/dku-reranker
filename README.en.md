# DKU-reranker-v1

[한국어](README.md) | **English** | [日本語](README.ja.md) | [中文](README.zh.md)

**A reranker for Retrieval-Augmented Generation (RAG) over Korean PDF documents.**
It reorders the retrieved evidence chunks fed to a small language model (SLM) so that the chunk
holding the answer comes first.

The backbone weights of [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
are **frozen**; only a LoRA adapter is trained, on **gold-evidence labels** from Korean
public-sector PDF reports. The adapter file is 8.9MB; the base model is downloaded from its own
public repository.

| Item | Detail |
|---|---|
| Where it sits | In a RAG pipeline, **after retrieval and before generation** |
| What it does | Scores each (question, chunk) pair and **reorders** the chunks |
| Base model | `BAAI/bge-reranker-v2-m3` (frozen backbone, LoRA adapter only) |
| Training data | 264 items from Korean public-sector PDF reports, four institutes |
| Evaluation | `google/gemma-4-E2B-it` on 81 items — **58** correct vs. 17 for the public BGE |
| Size | 8.9MB adapter (`r=8`) |
| License | Apache-2.0 |

`DKU` stands for Dankook University.

---

## 1. What it does

Among the chunks returned by retrieval, it **moves the ones that actually contain the answer to
the front.**

```
question + [chunk1, chunk2, …, chunkN]
           ↓  one score per (question, chunk) pair
           ↓  sort by score, descending
question + [gold evidence, …, …]      ← feed the generator in this order
```

Small language models extract values reliably when the supporting evidence appears early in the
input. So **the order matters as much as the selection.**

## 2. Why it is needed

Korean public-sector reports are table-heavy, and a single report repeats **similar-looking
tables and figures many times** — year-by-year budget tables, region-by-region statistics.

Retrieval returns several relevant chunks. But if **the chunk holding the answer does not come
first**, a small model copies a figure from whichever table sits at the front. The answer was
present in the input, and the model still got it wrong. This adapter targets that one failure.

A general-purpose reranker measures **how related a chunk is to the question**. This adapter is
trained further, on human-marked labels of **which chunk actually supports the answer**. The two
are similar but not the same: several chunks on the same topic are all "related", yet usually
only one of them contains the answer.

## 3. What it is not

| Not this | Explanation |
|---|---|
| Not a generative model | It produces no answers; it only reorders chunks |
| Not a reranker trained from scratch | The public BGE backbone is **frozen**; only the LoRA adapter is trained |
| Not a retriever | It reorders existing candidates; it does not fetch new documents |
| Not a context compressor | It never deletes, summarizes, or rewrites chunk text |
| Not a general multilingual model | It was trained only on Korean public-sector reports |

## 4. Purpose and fit

Built to make **Korean PDF documents usable by small language models (SLMs)**.

### 4.1 Where it helps

| Setting | Why |
|---|---|
| QA over Korean public-sector and government reports | Same document type as the training data |
| Tasks that look up a value in a table | The training data is table-centric |
| RAG with a small (few-B) generator | Evidence position matters most there |
| Tight context budgets | When only a few chunks fit, which ones you keep matters |

### 4.2 Where it does not fit

| Setting | Why |
|---|---|
| Documents in English or other languages | The backbone is multilingual, so it runs, but this adapter saw only Korean |
| Prose documents with few tables | The training data is table-centric |
| As a replacement for retrieval | A chunk absent from the candidates cannot be promoted |
| Tasks needing edits inside a chunk | This adapter does not alter chunk text |

## 5. Install and use

### 5.1 Requirements

```bash
pip install torch transformers peft
```

| Item | Value |
|---|---|
| `peft` | Trained with **0.20.0**. Older versions may not read every field of `adapter_config.json` |
| Device | Runs on CPU. With a GPU, call `model.to("cuda")` |
| Base model | The first run downloads `BAAI/bge-reranker-v2-m3`, about 2.2GB |

### 5.2 Attach the adapter

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

### 5.3 Score and reorder

```python
def rerank(question, chunks, batch_size=8, max_length=1024):
    # Return the chunks sorted by score, highest first.
    scores = []
    for i in range(0, len(chunks), batch_size):
        part = chunks[i:i + batch_size]
        enc = tok([question] * len(part), part,
                  padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            scores.extend(model(**enc).logits.view(-1).float().tolist())
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [(chunks[i], scores[i]) for i in order]
```

### 5.4 In a RAG pipeline

```python
ranked = rerank(question, retrieved_chunks)
top_k  = [text for text, _ in ranked[:28]]
prompt = "Question: " + question + "\n\nEvidence:\n" + "\n\n".join(top_k)
```

### 5.5 Dropping it into an existing BGE setup

**One line is all you add.**

```python
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # <- this line

model = model.unload()    # detach to return to the stock public model
```

### 5.6 Practical notes

| Rule | Reason |
|---|---|
| **Keep the batch size fixed** | A different batch changes the padding length inside it, which can perturb scores slightly. Fix one value for reproducibility |
| **max_length 1024** | The value used during training. Moving far from it changes the conditions |
| **Pair order** | Feed the tokenizer `(question, chunk)` in that order. Swapping them changes the scores |
| **Score semantics** | The absolute value carries no meaning. Use only the **relative ranking within one question** |
| **Truncation is the caller's choice** | The adapter only sorts. How many chunks to keep is a context-budget decision |
| **Chunk boundaries** | Training used whole tables and whole sections as chunks. Splitting into sentences departs from that |

## 6. Training data

| Item | Detail |
|---|---|
| Type | **Korean public-sector PDF reports** |
| How they were obtained | Downloaded **directly from each institute's own public website**; all are openly published |
| Nature | Budget and settlement analyses, agricultural and industry statistics, transport surveys, ICT issue briefs — **table-heavy policy reports** |
| Labels | For each question, the chunks containing the supporting evidence were **marked by a human** |
| Training items | 264 (98 for score separation, 166 for pairwise order) |
| Splits | **Document-level** — chunks from one document never appear in both train and validation |

The per-institute breakdown, chosen so that no single source dominates:

| Source institute | Training items | Documents |
|---|---|---|
| National Assembly Budget Office (nabo) | 84 | 10 |
| Korea Rural Economic Institute (krei) | 82 | 13 |
| Korea Transport Institute (koti) | 49 | 9 |
| National Information Society Agency (nia) | 49 | 12 |
| **Total** | **264** | **44** |

The training items and source PDFs are not included in this repository.

## 7. Evaluation — tested with Gemma 4 E2B

### 7.1 How it was measured

| Item | Value |
|---|---|
| Generator | `google/gemma-4-E2B-it` (2.3B effective, GGUF Q8_0) |
| Runtime | `llama-server`, context 10240, `temperature 0`, 99 GPU layers, 4 threads |
| Machine | AWS g6.xlarge (NVIDIA L4) |
| Evaluation items | **81** |
| What changed | **Only the reranker.** Items, candidate chunks and generation settings were identical |

The items were selected as follows: on the plain-RAG input, E2B answered **wrongly**, and the
**wrong value came from another chunk in that same input**. The answer was present; the model
read the wrong chunk. By construction the baseline scores **0 / 81**.

### 7.2 Results

| Reranker | Correct | Recovery |
|---|---|---|
| none (plain RAG) | 0 / 81 | 0.0% |
| public `BAAI/bge-reranker-v2-m3` | 17 / 81 | 21.0% |
| **this adapter (DKU)** | **58 / 81** | **71.6%** |

That is **41 more items (+50.6pp)** than the public model (McNemar exact, two-sided, `p = 2.46e-10`).

### 7.3 Why it rises

What differed is how often the labelled evidence was placed **first** in the input.

| Reranker | Evidence ranked first |
|---|---|
| none | 1 / 81 |
| public BGE | 18 / 81 |
| **this adapter** | **79 / 81** |

Splitting the same 81 items into two groups makes the cause plain.

| Group | Items | Public BGE | This adapter |
|---|---|---|---|
| Evidence rank **changed** | 61 | 6 / 61 (10%) | **47 / 61 (77%)** |
| Evidence rank **unchanged** | 18 | 11 / 18 (61%) | 11 / 18 (61%) |

Where the rank did not change, the two models are **identical**. The gain comes from the
ranking, not from the items being easier.

### 7.4 What was not established

| Item | Status |
|---|---|
| Keeping only the top 28 chunks | **No detectable difference** on these 81 items (`p = 1`) |
| Placing the question ahead of the evidence | **No detectable difference** on these 81 items (`p = 1`) |
| Other generators | Not tested. Measured with E2B only |
| Other languages or document types | Not tested |

The one confirmed effect is **the trained reranker**. For the rest, this sample showed no direction.

## 8. Training

| Item | Value |
|---|---|
| Base model | `BAAI/bge-reranker-v2-m3` (frozen backbone) |
| Trained part | LoRA adapter — `r=8`, `alpha=16`, `dropout=0.05` |
| Target modules | `key`, `query`, `value` |
| Task type | `SEQ_CLS` (the classification head is saved alongside) |
| Supervision | Human-verified **gold-evidence labels** |
| Loss | A term raising gold-evidence scores + a term ordering gold above non-gold (weighted 1 : 1.0) |
| Training items | 264 (98 + 166) |
| epochs · lr · seed | 5 · 0.0001 · 20260915 |
| max_length · steps | 1024 · 980 |
| Wall-clock | 82.9 min (5.07 s/step) |
| Device | NVIDIA L4 (AWS g6.xlarge) |

### 8.1 Which epoch to ship — **fixed as a rule before training**

Choosing after seeing the numbers is itself a selection bias, so the rule was written down before
a single training step ran.

```
1. Discard any epoch whose order-violation rate exceeds the pre-training baseline (epoch 0)
2. Among the rest, take the epoch with the highest gold-separation metric
3. On a tie, take the earlier epoch
```

| Metric | Before training | Selected epoch 5 |
|---|---|---|
| Share of items where every gold chunk outranks every non-gold chunk | 0.4796 | **0.7959** |
| Order-violation rate (lower is better) | 0.2866 | **0.1311** |

Both metrics are **fit on the training set**, not downstream performance. They were used only to
select the adapter; performance is reported separately in section 7.

## 9. Repository layout

| Path | Contents |
|---|---|
| `model/adapter_model.safetensors` | LoRA adapter (8.9MB) |
| `model/adapter_config.json` | Adapter configuration |
| `src/build_c2_trainset.py` | Training-set construction |
| `src/make_doc_folds_v2.py` | Document-level splits |
| `src/train_c2_targeted.py` | LoRA training |
| `docs/train_record.json` | Training record — settings, losses, epoch selection |

The training items and source PDFs are not included.

## 10. Limitations

| Limitation | Detail |
|---|---|
| Data scope | Trained on 264 items drawn from 44 reports across four Korean institutes |
| Evaluation scope | Verified with one generator (E2B) on 81 items |
| Nature of the eval set | Only items the plain RAG setup got **wrong**. Not a general question distribution |
| One targeted failure | It addresses "the answer is in the input but another chunk is read". It cannot fix retrieval misses |
| Comparison | Compared against one baseline, the public `bge-reranker-v2-m3` |

## 11. FAQ

**Is the base model bundled?** No. `BAAI/bge-reranker-v2-m3` comes from its own public
repository; this repository holds only the adapter that sits on top.

**Can I use it on English documents?** The backbone is multilingual, so it runs. But this adapter
saw only Korean, and no gain in English has been verified.

**How many chunks should I keep?** The evaluation used the top 28. That selection step itself
showed no measurable effect (7.4), so set it by your context budget.

**Can I threshold the scores?** Not recommended. The absolute scale shifts from question to
question, which moves any fixed threshold. Use the **relative order within one question**.

**Is the released file the one used in the paper?** Yes — byte-for-byte identical.

## 12. Contact

This adapter was trained and released as part of a master's thesis project.
For enquiries, write to <sh.yang@dankook.ac.kr>.

## 13. License

This adapter is a derivative of `BAAI/bge-reranker-v2-m3` and is released under the same
**Apache License 2.0**. The original model's license and attribution requirements apply as well.
The full text is in [LICENSE](LICENSE).

## 14. Citation

Cite it in this form:

> S. H. Yang, "DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking," GitHub repository, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

Or copy the BibTeX entry below.

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
