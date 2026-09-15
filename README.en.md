# DKU-reranker-v1

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)
[![Language](https://img.shields.io/badge/language-Korean--only-red.svg)](#)

[한국어](README.md) | **English** | [日本語](README.ja.md) | [中文](README.zh.md)

**This is the public reranker `BAAI/bge-reranker-v2-m3`, specialised for the writing style of
Korean public-sector documents.** In Retrieval-Augmented Generation (RAG) over Korean PDFs, it
fixes the order of the evidence chunks handed to a small language model (SLM).

The base model is multilingual, covering many languages at once. That breadth is its strength,
but it does not reach down into how documents in any one language are actually written. This
adapter leaves those backbone weights frozen and adds a sense of ranking for **one narrow slice
only: Korean public-sector PDF reports.** The supervision comes from gold-evidence labels a human
marked in those very reports.

**So this adapter is Korean-only.** It was trained on Korean documents and verified in Korean, so
how it behaves in other languages is unknown. If you work with Korean public documents you can
expect better ordering than the base model gives; anywhere else, the base model is the safer
choice.

Small language models read values accurately when the evidence sits near the front of the input.
Korean public-sector reports, however, repeat similar tables and figures many times within one
document. So even when retrieval does bring back the right chunk, if that chunk is not near the
front, the model copies a number from whichever table happens to be there instead. This adapter
targets that one failure.

## Why we built it

If you try answering questions over Korean PDFs with a small language model, one failure shows up
again and again: **the answer is right there in the input, and the model still gets it wrong.**
Retrieval had done its job, the chunk containing the answer was included — and the model still
copied a number from a different table sitting closer to the front.

Tracing it through the raw logs, the cause turned out to be **ranking**. A public-sector report
repeats similar-looking tables year by year and region by region. Retrieval and reranking decide
which of them goes first, but a public reranker looks at only one thing: **how related is this
chunk to the question?** Tables on the same topic all look "related", while usually only one of
them actually holds the answer. Relatedness alone cannot single that one out and lift it to the top.

So we changed the approach slightly. Using labels where a human marked **which chunk actually
supports the answer**, we nudged the public model a little further in that one direction.
Retraining the whole backbone would be costly and would risk losing the general ability the public
model already has, so we froze the backbone and attached only a small LoRA adapter. That is why
the adapter is just 8.9MB.

Whether this actually helps, we measured directly. With the same questions and the same candidate
chunks, swapping in this adapter raised the number of questions whose gold evidence ranked first
from 18 to 79, and correct answers rose from 17 to 58. The details are in
[Results](#results) below. For anyone who wants to check or verify this under the same conditions,
we publish the exact adapter file used in the measurement.

## Results

Evaluated on 81 questions with Gemma 4 E2B. Questions, candidate chunks and generation settings
were identical; **only the reranker changed.** The evaluation was carried out entirely on
**Korean questions over Korean documents.**

| Reranker | Correct | Recovery | Gold evidence ranked first |
|---|---|---|---|
| none (plain RAG) | 0 / 81 | 0.0% | 1 / 81 |
| `BAAI/bge-reranker-v2-m3` (public) | 17 / 81 | 21.0% | 18 / 81 |
| **DKU-reranker-v1 (this adapter)** | **58 / 81** | **71.6%** | **79 / 81** |

That is 41 more questions than the public model (+50.6pp, McNemar exact two-sided `p = 2.46e-10`).
Full conditions and the breakdown are in [Evaluation](#evaluation).

## Model details

| Item | Value |
|---|---|
| Base model | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) — frozen backbone |
| Trained part | LoRA adapter (`r=8`, `alpha=16`, `dropout=0.05`, target modules `key`, `query`, `value`) |
| Task type | `SEQ_CLS` |
| Input | One (question, chunk) pair, max length 1024 |
| Output | A single real-valued score — use it only as a relative rank within one question |
| Size | 8.9MB (`model/adapter_model.safetensors`) |
| Language | Korean |
| License | Apache-2.0 (derivative of the base model) |

## Who it is for

We are releasing this so that anyone in Korea adopting SLM-based RAG can use it. Individual
developers, companies, public institutions, schools and research institutes were all in mind.

| Who | Where it helps |
|---|---|
| Individual developers | While building a Korean-document RAG, you can swap it in for the public reranker as is |
| Companies | For QA over internal documents and reports where a large model is impractical |
| Public institutions | Where data must stay inside the organisation and be served on-premises |
| Schools and research institutes | As a baseline or starting point for Korean RAG research |

At 8.9MB, if you already use `bge-reranker-v2-m3` you can try it by adding a single line.

## Intended use

### Where it fits

- QA over Korean public-sector and government reports — the training data is exactly this kind of document
- Tasks that look up a value in a table — it was trained on situations where similar tables recur
- RAG with a small (few-B) generator — the more evidence position matters, the more you can expect
- Tight context budgets where only the top few chunks fit — what goes first matters most there

### Out of scope

The following are either unverified or simply not what this adapter was built to do.

- Documents not in Korean. The backbone is multilingual so it runs, but this adapter saw only Korean, and no gain in other languages has been verified
- Prose-heavy documents with few tables. The training data is table-centric, so results there are not something we can promise
- Replacing retrieval itself. This adapter only reorders; a chunk that never made it into the candidates cannot be pulled in
- Editing chunk content. It never deletes, summarizes or rewrites a chunk — it only changes the order

## Installation

```bash
pip install torch transformers peft
```

| Item | Value |
|---|---|
| `peft` | Trained with 0.20.0. Older versions may fail to read some fields of `adapter_config.json` |
| Device | Runs on CPU. With a GPU, call `model.to("cuda")` |
| Download | The first run fetches the base model, about 2.2GB |

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # the model/ directory in this repo
model.eval()


def rerank(question, chunks, batch_size=8, max_length=1024):
    """Return the chunks sorted by score, highest first."""
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


ranked = rerank(question, retrieved_chunks)
top_k = [text for text, _ in ranked[:28]]
prompt = "Question: " + question + "\n\nEvidence:\n" + "\n\n".join(top_k)
```

If you already use `bge-reranker-v2-m3`, adding the single `PeftModel.from_pretrained` line after
loading the model is all it takes. Calling `model.unload()` detaches the adapter and returns you
to the stock public model.

### Practical notes

A few things worth knowing when you actually run it.

| Item | What works well | Why |
|---|---|---|
| Batch size | Please keep it fixed | A different batch changes the padding length inside it, which nudges scores slightly |
| Max length | 1024 as provided is fine | It is the value used in training, so changing it also shifts the scale of the scores |
| Pair order | Please keep `(question, chunk)` | Swapping them changes the scores |
| Using scores | Please use relative rank, not absolute values | The absolute scale differs per question, so a fixed threshold drifts from question to question |
| Chunk boundaries | Feed whole tables and whole sections | Scores are most stable at the chunk granularity used in training |
| Truncation (how many to keep) | Please decide this outside the adapter | It only sorts; how many chunks you use is up to you |

## Training data

Korean public-sector PDF reports. Each report was downloaded directly from the publishing
institute's own public website. They are table-heavy policy documents: budget and settlement
analyses, agricultural and industry statistics, transport surveys, ICT issue briefs. For every
question, a human checked and marked the chunks that support the answer.

Public-sector documents were chosen because they **use standard Korean precisely and keep a
consistent writing style even across institutes.** When style varies, that variation leaks into
the training signal.

| Source institute | Training items | Documents |
|---|---|---|
| National Assembly Budget Office (nabo) | 84 | 10 |
| Korea Rural Economic Institute (krei) | 82 | 13 |
| Korea Transport Institute (koti) | 49 | 9 |
| National Information Society Agency (nia) | 49 | 12 |
| **Total** | **264** | **44** |

Splits are **document-level**: chunks from one document never appear in both train and validation.
The training items and the source PDFs are not included in this repository.

## Training

```bash
python src/build_c2_trainset.py      # build the training set
python src/make_doc_folds_v2.py      # document-level splits
python src/train_c2_targeted.py      # LoRA training
```

| Hyperparameter | Value |
|---|---|
| epochs | 5 |
| learning rate | 0.0001 |
| seed | 20260915 |
| max length | 1024 |
| steps | 980 |
| loss | A term raising gold-evidence scores + a term ordering gold above non-gold (weighted 1 : 1.0) |
| training items | 264 (98 for score separation, 166 for pairwise order) |
| device | NVIDIA L4 (AWS g6.xlarge), 82.9 min |

Which epoch to ship was fixed as a rule **before a single training step ran**: (1) discard any
epoch whose order-violation rate exceeds the pre-training baseline, (2) among the rest take the
highest gold-separation metric, (3) on a tie take the earlier epoch. The rule selected epoch 5.

| Metric (on the training set) | Before training | epoch 5 |
|---|---|---|
| Share of items where every gold chunk outranks every non-gold chunk | 0.4796 | **0.7959** |
| Order-violation rate (lower is better) | 0.2866 | **0.1311** |

Both are fit measured on the training set, not downstream performance; they were used only to
select the adapter. The full record is in [`docs/train_record.json`](docs/train_record.json).

## Evaluation

### Setup

| Item | Value |
|---|---|
| Generator | `google/gemma-4-E2B-it` (2.3B effective, GGUF Q8_0) |
| Runtime | `llama-server`, context 10240, `temperature 0`, 99 GPU layers, 4 threads |
| Machine | AWS g6.xlarge (NVIDIA L4) |
| Questions | 81 |
| Control | Identical questions, candidate chunks and generation settings; only the reranker was swapped |

To keep the comparison fair, the three conditions receive exactly the same input once the reranker
is set aside. Same questions, same candidate chunks, same generation settings. The only thing that
varies is the reranker, so the difference in correct answers can be attributed to it.

The questions are those where E2B answered wrongly on the plain-RAG input **and the wrong value
came from another chunk inside that same input**. The answer was present; the model read a
different chunk. By that criterion the baseline is 0 / 81, and each model's correct count is also
its net gain over the baseline.

### Results

See the [Results](#results) table above. The gain comes from gold evidence being placed first far
more often — 18 → 79. Splitting the same 81 questions by whether the rank changed makes it clearer.

| Group | Questions | Public BGE | DKU |
|---|---|---|---|
| Gold evidence rank changed | 61 | 6 / 61 (10%) | **47 / 61 (77%)** |
| Gold evidence rank unchanged | 18 | 11 / 18 (61%) | 11 / 18 (61%) |

Where the rank did not change, the two models are identical. The improvement comes from the
ranking, not from the questions being easier.

### What we could not establish

| Item | Outcome |
|---|---|
| Keeping only the top 28 chunks | No difference detected on these 81 questions (`p = 1`) |
| Placing the question before the evidence | No difference detected on these 81 questions (`p = 1`) |
| Other generators | Not measured |
| Other languages or document types | Not measured |

The one confirmed effect is the trained reranker.

## Limitations

- Trained on 264 items built from 44 reports across four Korean institutes
- Verified with one generator (E2B) on 81 questions only
- The evaluation set collects only questions the plain RAG setup got wrong; it is not a general question distribution
- It addresses one failure — "the answer is in the input but another chunk is read". It cannot fix cases retrieval never returned
- The comparison baseline is a single model, the public `bge-reranker-v2-m3`

## Repository layout

```
model/adapter_model.safetensors   LoRA adapter (8.9MB)
model/adapter_config.json         adapter configuration
src/build_c2_trainset.py          training-set construction
src/make_doc_folds_v2.py          document-level splits
src/train_c2_targeted.py          LoRA training
docs/train_record.json            training record (settings, losses, epoch selection)
```

## Citation

> S. H. Yang, "DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub repository, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  school = {Dankook University, Graduate School of Information Convergence Technology and Entrepreneurship},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```

## License

A derivative of `BAAI/bge-reranker-v2-m3`, released under the same Apache License 2.0. The
original model's license and attribution requirements apply as well. The full text is in
[LICENSE](LICENSE).

## Affiliation

Trained and released at the **Graduate School of Information Convergence Technology and
Entrepreneurship (정보융합기술·창업대학원), Dankook University**.
<https://cms.dankook.ac.kr/web/gict>

## Contact

For questions or suggestions, please use
[Issues](https://github.com/tracer999/dku-reranker/issues) or write to
<sh.yang@dankook.ac.kr>.
