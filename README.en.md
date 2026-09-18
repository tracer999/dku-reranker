# DKU-reranker

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)
[![Language](https://img.shields.io/badge/language-Korean--only-red.svg)](#)

[한국어](README.md) | **English** | [日本語](README.ja.md) | [中文](README.zh.md)

**This is the public reranker `BAAI/bge-reranker-v2-m3`, specialised for the writing style of
Korean public-sector documents.** In Retrieval-Augmented Generation (RAG) over Korean PDFs, it
fixes the order of the evidence chunks handed to a small language model (SLM).

The base model is multilingual — broad, but it does not reach into how documents in any one
language are written. This adapter keeps those backbone weights frozen and trains only a LoRA
adapter, on gold-evidence labels a human marked in Korean public-sector PDF reports.
**It is Korean-only; its behaviour in other languages has not been verified.**

## Why we built it

It targets one recurring failure in SLM-based RAG: **the supporting evidence is in the input, yet
the model answers with a value taken from a different chunk.**

The cause is ranking. Public-sector reports repeat tables of identical layout year by year and
region by region. A public reranker scores only relatedness to the question, so every table on the
topic scores high and the single one holding the answer is not separated out.

So we used **whether a chunk is the supporting evidence** as the training signal instead of
relatedness. The backbone stays frozen and only a LoRA adapter is trained, preserving the public
model's general performance and keeping deployment cost unchanged.

We measured the effect directly. Swapping only the reranker raised the number of questions whose
gold evidence ranked first and the number of correct answers both rose. The adapter file used
in that measurement is released as is.

## Results

Evaluated on **documents not used for training**. We collected 46 fresh reports sharing no document
with the 44 used in training, and measured on 80 questions drawn from them. Questions, candidate
chunks and generation settings are identical; **only the reranker is swapped**. Generator is Gemma 4 E2B.

| Reranker | Correct | Gold evidence ranked 1st | Other tables ahead of gold |
|---|---|---|---|
| None (plain RAG) | 0 / 80 | — | — |
| `BAAI/bge-reranker-v2-m3` (public) | 69 / 80 | 64 / 80 | 4 |
| Previous release | 60 / 80 | 27 / 80 | 22 |
| **DKU-reranker (this adapter)** | **71 / 80** | **65 / 80** | **3** |

That is **11 more questions than the previous release** (McNemar exact two-sided `p = 0.0034`).
The reason is visible: questions with another table placed ahead of the gold evidence fell from
**22 to 3**.

★ These are numbers **on documents never seen in training**. Measuring on training documents gives
higher values, but those do not predict behaviour on new documents.

⛔ The gap to the public model (69 → 71) is **not** statistically significant at this sample size
(`p = 0.754`). We do **not** claim to beat the public model. What this table shows is that
*training corrects the ranking*, at a magnitude confirmed against the previous release at `p = 0.0034`.

⛔ Values depend on the evaluation setup. The design is in [Evaluation](#evaluation).

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

- QA over **documents written in standard Korean** — the training data is written in precise standard Korean
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
| **Total** | **44** |

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
| steps | 1,870 |
| loss | A term raising gold-evidence scores + a term ordering gold above non-gold (weighted 1 : 1.0) |
| training items | 353 (98 evidence separation, 166 order preservation, 89 table targeting) |
| device | NVIDIA L4 (AWS g6.xlarge), 82.9 min |

Which epoch to ship was fixed as a rule **before a single training step ran**: (1) discard any
epoch whose order-violation rate exceeds the pre-training baseline, (2) among the rest take the
highest gold-separation metric, (3) on a tie take the earlier epoch. The rule selected epoch 5.

| Metric (on the training set) | Before training | epoch 5 |
|---|---|---|
| Share of items where every gold chunk outranks every non-gold chunk | 0.6738 | **0.8877** |
| Order-violation rate (lower is better) | 0.2866 | **0.082** |

Both are fit measured on the training set, not downstream performance; they were used only to
select the adapter. The full record is in [`docs/train_record.json`](docs/train_record.json).

## Evaluation

### How to measure

| Item | Value |
|---|---|
| Generator | `google/gemma-4-E2B-it` (2.3B effective, GGUF Q8_0) |
| Runtime | `llama-server`, context 10240, `temperature 0`, GPU layers 99, threads 4 |
| Control | Questions, candidate chunks and generation settings identical. **Only the reranker is swapped** |

For the comparison to be fair, everything but the reranker must be held equal. Same questions,
same candidate chunks, same generation settings — only then can a difference in the number of
correct answers be attributed to the reranker.

### Which questions to use

The failure this adapter targets is narrow: **the gold evidence is in the input, yet the model
takes its value from a different chunk in that same input.** Your evaluation set should be built
to that condition.

1. Take questions the generator gets **wrong** on the plain RAG input (no reranker).
2. Check that the wrong value **actually appears in another chunk of the same input**.
3. Drop cases where the model invented the value — those cannot be attributed to chunk order.

Built this way, the baseline scores 0, and each model's score is its own net gain.

### What to report alongside

The number of correct answers alone does not say why it moved. Measure these two as well.

| What | Why |
|---|---|
| **Questions where gold evidence is ranked first** | This is what the adapter directly changes |
| **Other tables placed ahead of the gold evidence** | This is where the failure happens; it must fall for accuracy to rise |

Also split the questions into those whose ranking **changed** and those that **did not**. On the
unchanged ones the two models should agree. If they differ there, something other than ranking
has crept in.

> ⛔ **Why no numbers** — results shift substantially with document type, question phrasing and
> candidate-chunk composition. Publishing one set as a headline invites mismatched expectations.
> Please measure with the design above.

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

> S. H. Yang, "DKU-reranker: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub repository, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker,
  title  = {DKU-reranker: A LoRA Adapter for Korean Public-Document Evidence Reranking},
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

| Item | Detail |
|---|---|
| University | Dankook University · <https://www.dankook.ac.kr> |
| Graduate school | Graduate School of Information Convergence Technology and Entrepreneurship · <https://cms.dankook.ac.kr/web/gict> |

## Contact

For questions or suggestions, please use
[Issues](https://github.com/tracer999/dku-reranker/issues) or write to
<sh.yang@dankook.ac.kr>.
