# DKU-reranker-v1

**한국어** | [English](README.en.md)

한국어 공공기관 PDF 문서의 **정답 근거 라벨**로 학습한 재순위(reranker) LoRA 어댑터다.
공개 모델 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) 의
본체 가중치를 **동결**하고 LoRA 어댑터만 학습하였다.

`DKU` 는 단국대학교(Dankook University)를 가리킨다.

## 무엇을 하는가

검색으로 가져온 조각(chunk)들 가운데 **정답의 근거가 되는 조각을 맨 앞으로 올린다.**
소형 언어모델은 근거가 입력의 앞쪽에 있을 때 값을 정확히 뽑아내므로, 이 순위가 답의 정확도를 바꾼다.

## 무엇이 아닌가

- 새로 만든 생성 모델이 아니다.
- 처음부터 학습한 재순위 모델이 아니다.
- **공개 BGE 재순위 모델 + 한국어 자료로 학습한 LoRA 어댑터**가 이 모델의 실체다.

## 목적

**한국어 PDF 문서를 소형 언어모델(SLM)에 적용하는 것**을 목표로 만들었다.

소형 언어모델은 컨텍스트가 짧고, 입력 앞쪽에 놓인 내용을 답으로 옮겨 적는 성향이 있다.
한국어 공공기관 보고서는 표가 많고 한 문서 안에 비슷한 표와 숫자가 여러 번 나오므로,
검색이 관련 있는 조각을 여럿 가져와도 **정답을 담은 조각이 앞에 오지 않으면**
모델은 앞에 놓인 다른 조각의 숫자를 답으로 적는다.

이 어댑터는 그 지점을 겨냥한다 — **정답을 담은 조각을 맨 앞으로 올린다.**

### 어떤 자리에 쓰면 좋은가

- 한국어 공공기관·정부기관 보고서 질의응답
- 표에서 값을 찾아 답하는 과제
- 소형 언어모델(수 B 규모)을 쓰는 RAG — 근거 위치의 영향이 큰 환경

### 어떤 자리에는 맞지 않는가

- 영어 등 다른 언어 문서. 기반 모델이 다국어이므로 동작은 하지만 이 어댑터는 한국어로만 학습하였다
- 표가 없는 서술형 문서. 학습 자료가 표 중심이다
- 검색 자체를 대신하는 용도. 이 어댑터는 **이미 검색된 후보의 순위만** 바꾼다

## 학습에 쓴 자료

| 항목 | 내용 |
|---|---|
| 종류 | **한국어 공공기관 PDF 보고서** |
| 출처 기관 | 국회예산정책처(nabo) · 한국농촌경제연구원(krei) · 한국교통연구원(koti) |
| 확보 경로 | 각 기관이 **누구나 받을 수 있게 공개한 보고서**를 해당 기관 누리집에서 직접 내려받았다 |
| 문서 성격 | 예산·결산 분석, 산업 통계, 교통 조사 — **표가 많은 정책 보고서** |
| 라벨 | 질문마다 **정답의 근거가 되는 조각을 사람이 확인해 표시**하였다 |
| 학습 문항 | 264개 |
| 분할 | **문서 단위**로 나누었다 — 같은 문서의 조각이 학습과 검증에 함께 들어가지 않는다 |

⛔ 학습에 쓴 문항과 원문 문서는 이 저장소에 포함하지 않는다.

## 설치와 사용

```bash
pip install torch transformers peft
```

### 1) 어댑터 결합

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # 이 저장소의 model/ 을 가리킨다
model.eval()
```

### 2) 점수 매기고 순위 바꾸기

```python
def rerank(question, chunks, batch_size=8, max_length=1024):
    """질문과 조각 목록을 받아 점수가 높은 순으로 돌려준다."""
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

### 3) 실제로 쓰는 꼴

```python
# ⛔ 아래는 형태를 보이려는 예시다. 실제 보고서의 값이 아니다.
question = "○○ 사업의 2024년 집행률은 몇 퍼센트인가?"

chunks = [
    "Ⅰ. 사업 개요 … 추진 배경과 근거 법령 …",
    "Ⅱ. 예산 집행 현황 … 2024년 집행률 …",        # ← 정답 근거가 든 조각
    "Ⅲ. 향후 계획 … 연차별 추진 일정 …",
]

ranked = rerank(question, chunks)
for text, score in ranked:
    print(round(score, 4), text[:40])

# 상위 k 개만 생성 모델에 넣는다
top_k = [text for text, _ in ranked[:28]]
prompt = "질문: " + question + "\n\n근거:\n" + "\n\n".join(top_k)
```

### 쓸 때 지킬 것

| | |
|---|---|
| **배치를 고정한다** | 배치가 달라지면 묶음 안의 패딩 길이가 달라져 점수가 미세하게 흔들릴 수 있다. 재현하려면 한 값으로 고정한다 |
| **최대 길이 1024** | 학습할 때 쓴 값이다. 크게 바꾸면 학습 때와 다른 조건이 된다 |
| **질문–조각 순서** | 토크나이저에 `(질문, 조각)` 차례로 넣는다. 뒤바꾸면 점수가 달라진다 |
| **점수의 뜻** | 절댓값에 의미를 두지 않는다. **같은 질문 안에서의 상대 순위**만 쓴다 |
| **정렬만 한다** | 이 어댑터는 조각을 지우거나 고치지 않는다. 자르는 것은 쓰는 쪽이 정한다 |

### 기존 BGE 와 바꿔 끼우기

이미 `bge-reranker-v2-m3` 을 쓰고 있다면 **모델을 불러온 뒤 한 줄만 더하면 된다.**

```python
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # ← 이 줄만 추가
```

어댑터를 떼면 원래 공개 모델로 돌아간다.

```python
model = model.unload()      # 어댑터 제거
```

## 검증 — Gemma 4 E2B 로 시험하였다

생성 모델은 **`google/gemma-4-E2B-it` (유효 2.3B · GGUF Q8_0)** 이다.
`llama-server` 로 실행하였고 컨텍스트 10240 · `temperature 0` · GPU 층 99 · 스레드 4 ·
기계는 AWS g6.xlarge (NVIDIA L4) 다.

평가 문항은 **81개**다. 재순위 없는 기본 RAG 입력에서 E2B 가 틀렸고, **그 틀린 값이
입력에 함께 들어간 다른 조각(정답 근거가 아닌 조각)에서 온** 문항만 골랐다.
즉 답은 입력 안에 있었는데 모델이 다른 조각을 읽은 경우다.
기준선이 **0 / 81** 인 것은 선정 기준에서 따라온다 — 기준선이 맞힌 문항은 집합에서 뺐다.
그러므로 각 조건이 맞힌 수가 곧 기준선 대비 순증이다.

| 재순위 | 맞은 문항 | 회복률 |
|---|---|---|
| 없음 (기본 RAG) | 0 / 81 | 0.0% |
| 공개 `BAAI/bge-reranker-v2-m3` | 17 / 81 | 21.0% |
| **이 어댑터 (DKU)** | **58 / 81** | **71.6%** |

같은 81문항 · 같은 후보 조각 · 같은 생성 설정이며 **재순위 모델만 바꾸었다.**
공개 모델 대비 **41문항(+50.6%p)** 을 더 맞혔다 (McNemar 양측 정확검정 `p = 2.46e-10`).

### 왜 올랐는가

정답 근거를 **입력의 1순위**에 놓은 문항 수가 갈랐다.

| 재순위 | 정답 근거가 1순위인 문항 |
|---|---|
| 없음 | 1 / 81 |
| 공개 BGE | 18 / 81 |
| **이 어댑터** | **79 / 81** |

순위가 **바뀐 61문항**에서 정답률이 6/61(10%) → 47/61(77%) 로 올랐고,
순위가 **그대로인 18문항**에서는 11/18(61%) ↔ 11/18(61%) 로 같았다.
⇒ 문항 난이도가 아니라 **순위가 원인**이다.

⛔ 상위 28개만 남기는 선별과 질문을 근거보다 앞에 두는 배치는 이 81문항에서
**차이를 확인하지 못했다**(각각 p = 1). 확인된 효과는 **재순위 모델 학습 하나**다.

## 학습

| 항목 | 값 |
|---|---|
| 기반 모델 | `BAAI/bge-reranker-v2-m3` (본체 동결) |
| 학습 대상 | LoRA 어댑터 (r=8 · alpha=16 · dropout=0.05) |
| 대상 모듈 | `key`, `query`, `value` |
| 학습 신호 | 사람이 확인한 **정답 근거 라벨** |
| 손실 | 정답 근거의 점수를 올리는 항 + 정답 근거가 비정답 근거보다 앞서게 하는 항 (가중 1:1.0) |
| 학습 문항 | 264 (D 98 · F 166) |
| epoch · 학습률 · seed | 5 · 0.0001 · 20260915 |
| 최대 길이 · step | 1024 · 980 |
| 장치 | NVIDIA L4 (AWS g6.xlarge) |

### 학습 전후 지표

| 지표 | 학습 전 | 학습 후 (epoch 5) |
|---|---|---|
| 정답 근거가 비정답 근거보다 모두 앞서는 문항 비율 | 0.4796 | **0.7959** |
| 순서 위반율 (낮을수록 좋다) | 0.2866 | **0.1311** |

어느 epoch 의 어댑터를 쓸지는 **학습을 시작하기 전에** 규칙으로 고정하였다.
① 순서 위반율이 학습 전을 넘는 epoch 은 제외한다 ② 남은 것 중 위 비율이 가장 높은 것을 고른다
③ 동률이면 더 이른 epoch 을 고른다.

재현에 필요한 기록은 `docs/train_record.json` 에 있다.

## 코드

```
src/build_c2_trainset.py    학습셋 구성
src/make_doc_folds_v2.py    문서 단위 분할
src/train_c2_targeted.py    LoRA 학습
```

⛔ 학습에 사용한 문항과 원문은 이 저장소에 포함하지 않는다.

## 소속

| | |
|---|---|
| 작성자 | **양성훈 (YANG SEONG HUN)** |
| 학과 | **IT컨버젼스학과** — <https://cms.dankook.ac.kr/web/gict/it-2> |
| 대학원 | 단국대학교 **정보융합기술창업대학원** |
| 홈페이지 | <https://cms.dankook.ac.kr/web/gict> |

본 어댑터는 해당 대학원 IT컨버젼스학과의 석사 학위논문 연구 과정에서 학습·공개되었다.
`DKU` 는 단국대학교(Dankook University)를 가리킨다.

## 라이선스

이 어댑터는 `BAAI/bge-reranker-v2-m3` 의 파생물이며 원본과 같은 **Apache License 2.0** 을 따른다.
원본 모델의 라이선스와 저작자 표시를 함께 지켜야 한다.

## 인용

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  school = {단국대학교 정보융합기술창업대학원 IT컨버젼스학과},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```
