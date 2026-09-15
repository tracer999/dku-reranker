# DKU-reranker-v1

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)

**한국어** | [English](README.en.md) | [日本語](README.ja.md) | [中文](README.zh.md)

한국어 PDF 문서를 대상으로 하는 검색증강생성(RAG)에서, 소형 언어모델(SLM)에 넘길 근거 조각의
순위를 바로잡는 재순위 모델(reranker)입니다. 공개 모델 `BAAI/bge-reranker-v2-m3` 의 본체 가중치를
동결하고, 한국어 공공기관 PDF 보고서의 정답 근거 라벨로 LoRA 어댑터만 학습했습니다.

소형 언어모델은 근거가 입력 앞쪽에 있을 때 값을 정확히 뽑아냅니다. 그런데 공공기관 보고서는
비슷한 표와 숫자가 한 문서에 여러 번 나오기 때문에, 검색이 정답 조각을 가져와도 그것이 앞에
오지 않으면 모델은 앞에 놓인 다른 표의 숫자를 답으로 옮겨 적습니다. 이 어댑터는 그 지점
하나를 겨냥합니다.

## 왜 만들었나

소형 언어모델로 한국어 PDF 문서에 답하게 하려다 한 가지 실패가 반복해서 눈에 띄었습니다.
**답이 분명히 입력 안에 들어 있는데도 모델이 틀리는 것**입니다. 검색은 제 몫을 했고 정답이 담긴
조각도 함께 들어갔는데, 모델은 앞쪽에 놓인 다른 표의 숫자를 답으로 옮겨 적었습니다.

원인을 원자료에서 따라가 보니 **순위**였습니다. 공공기관 보고서는 한 문서 안에 연도별·지역별로
비슷하게 생긴 표가 여러 번 나옵니다. 검색과 재순위가 그중 어느 것을 앞에 둘지 정하는데,
공개 재순위 모델은 **질문과 얼마나 관련 있는가**를 잽니다. 그런데 같은 주제를 다루는 표 여럿이
모두 "관련 있음"이고, 정작 답이 든 것은 그중 하나뿐인 경우가 많습니다. 관련성만으로는 그 하나를
앞으로 올리지 못합니다.

그래서 **어느 조각이 실제로 정답의 근거인가**를 사람이 표시한 라벨로 공개 모델을 조금 더
학습시켜 보기로 했습니다. 본체를 새로 학습하면 비용도 크고 공개 모델의 일반성도 잃기 때문에,
본체는 동결하고 LoRA 어댑터만 얹었습니다. 어댑터가 8.9MB 인 이유입니다.

효과가 있는지 확인한 결과는 아래에 있습니다. 같은 문항과 같은 조각에서 재순위 모델만 바꿨을 때
정답 근거를 입력 1순위에 놓은 문항이 18 → 79 로 늘었고, 정답 수가 17 → 58 로 올랐습니다.
쓰거나 검증하려는 분이 같은 조건을 재현할 수 있도록, 측정에 사용한 어댑터 파일을 그대로
공개합니다.

## 주요 결과

Gemma 4 E2B 로 81문항을 평가했습니다. 문항·후보 조각·생성 설정은 모두 같고 **재순위 모델만**
바꿨습니다.

| 재순위 모델 | 정답 | 회복률 | 정답 근거가 1순위 |
|---|---|---|---|
| 없음 (기본 RAG) | 0 / 81 | 0.0% | 1 / 81 |
| `BAAI/bge-reranker-v2-m3` (공개) | 17 / 81 | 21.0% | 18 / 81 |
| **DKU-reranker-v1 (본 어댑터)** | **58 / 81** | **71.6%** | **79 / 81** |

공개 모델 대비 41문항을 더 맞혔습니다 (+50.6%p · McNemar 양측 정확검정 `p = 2.46e-10`).
자세한 조건과 분해는 [평가](#평가)에 있습니다.

## 모델 정보

| 항목 | 값 |
|---|---|
| 기반 모델 | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) — 본체 동결 |
| 학습 부위 | LoRA 어댑터 (`r=8`, `alpha=16`, `dropout=0.05`, 대상 모듈 `key`·`query`·`value`) |
| 과업 유형 | `SEQ_CLS` |
| 입력 | (질문, 조각) 한 쌍 · 최대 길이 1024 |
| 출력 | 실수 점수 하나 — 같은 질문 안에서의 상대 순위로만 씁니다 |
| 크기 | 8.9MB (`model/adapter_model.safetensors`) |
| 언어 | 한국어 |
| 라이선스 | Apache-2.0 (기반 모델의 파생물) |

## 사용 목적

### 적합한 용도

- 한국어 공공기관·정부 보고서에 대한 질의응답
- 표에서 값을 찾아 답하는 과제
- 소형 언어모델(수 B 규모)을 쓰는 RAG — 근거 위치의 영향이 큰 환경
- 컨텍스트 예산이 빠듯해 상위 몇 개만 넣어야 하는 환경

### 범위 밖 용도

- 한국어가 아닌 문서. 기반 모델이 다국어라 동작은 하지만, 이 어댑터는 한국어로만 학습했고 다른 언어에서의 이득은 확인된 바 없습니다
- 표가 거의 없는 서술 중심 문서. 학습 자료가 표 중심입니다
- 검색 자체의 대체. 후보에 없는 조각은 순위를 올릴 수 없습니다
- 조각 본문의 가공. 이 어댑터는 조각을 지우거나 요약하거나 다시 쓰지 않습니다

## 설치

```bash
pip install torch transformers peft
```

| 항목 | 값 |
|---|---|
| `peft` | 학습에 쓴 판은 0.20.0 입니다. 더 낮으면 `adapter_config.json` 의 일부 칸을 못 읽을 수 있습니다 |
| 장치 | CPU 에서도 동작합니다. GPU 가 있으면 `model.to("cuda")` 를 부릅니다 |
| 내려받기 | 첫 실행 때 기반 모델 약 2.2GB 를 받습니다 |

## 사용법

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # 이 저장소의 model/
model.eval()


def rerank(question, chunks, batch_size=8, max_length=1024):
    """질문과 조각 목록을 받아 점수가 높은 순으로 돌려줍니다."""
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
prompt = "질문: " + question + "\n\n근거:\n" + "\n\n".join(top_k)
```

이미 `bge-reranker-v2-m3` 을 쓰고 있다면 모델을 불러온 뒤 `PeftModel.from_pretrained` 한 줄만
더하면 됩니다. `model.unload()` 로 어댑터를 떼면 원래 공개 모델로 돌아갑니다.

### 쓸 때 지킬 것

| 항목 | 권장 | 까닭 |
|---|---|---|
| 배치 크기 | 하나로 고정 | 배치가 달라지면 묶음 안 패딩 길이가 달라져 점수가 미세하게 흔들립니다 |
| 최대 길이 | 1024 | 학습할 때 쓴 값입니다 |
| 입력 차례 | `(질문, 조각)` | 뒤바꾸면 점수가 달라집니다 |
| 점수 사용 | 상대 순위만 | 절댓값은 질문마다 분포가 달라 고정 임계값이 옮겨 갑니다 |
| 조각 경계 | 표는 표째, 절은 절째 | 학습 때의 조각 단위입니다 |
| 자르기 | 쓰는 쪽이 결정 | 이 어댑터는 정렬만 합니다 |

## 학습 자료

한국어 공공기관 PDF 보고서입니다. 각 기관이 공개한 보고서를 해당 기관 누리집에서 직접
내려받았습니다. 예산·결산 분석, 농업·산업 통계, 교통 조사, 정보화 이슈분석 등 표가 많은
정책 보고서입니다. 질문마다 정답의 근거가 되는 조각을 사람이 확인해 표시했습니다.

| 출처 기관 | 학습 문항 | 문서 |
|---|---|---|
| 국회예산정책처 (nabo) | 84 | 10 |
| 한국농촌경제연구원 (krei) | 82 | 13 |
| 한국교통연구원 (koti) | 49 | 9 |
| 한국지능정보사회진흥원 (nia) | 49 | 12 |
| **합계** | **264** | **44** |

분할은 **문서 단위**입니다. 같은 문서의 조각이 학습과 검증에 함께 들어가지 않습니다.
학습에 쓴 문항과 원문 PDF 는 이 저장소에 포함하지 않았습니다.

## 학습

```bash
python src/build_c2_trainset.py      # 학습셋 구성
python src/make_doc_folds_v2.py      # 문서 단위 분할
python src/train_c2_targeted.py      # LoRA 학습
```

| 하이퍼파라미터 | 값 |
|---|---|
| epoch | 5 |
| 학습률 | 0.0001 |
| seed | 20260915 |
| 최대 길이 | 1024 |
| step | 980 |
| 손실 | 정답 근거 점수를 올리는 항 + 정답 근거를 비정답 근거보다 앞세우는 항 (가중 1 : 1.0) |
| 학습 문항 | 264 (점수 분리용 98 · 순위 쌍 위반용 166) |
| 장치 | NVIDIA L4 (AWS g6.xlarge) · 82.9분 |

어느 epoch 의 어댑터를 쓸지는 **학습을 한 step 도 돌리기 전에** 규칙으로 고정했습니다.
① 순서 위반율이 학습 전을 넘는 epoch 은 제외 ② 남은 것 중 정답 근거 분리 지표가 가장 높은 것
③ 동률이면 더 이른 epoch. 규칙에 따라 epoch 5 를 골랐습니다.

| 지표 (학습셋) | 학습 전 | epoch 5 |
|---|---|---|
| 정답 근거가 비정답 근거보다 모두 앞서는 문항 비율 | 0.4796 | **0.7959** |
| 순서 위반율 (낮을수록 좋음) | 0.2866 | **0.1311** |

두 지표는 학습셋에서 잰 적합도이며 하류 과제 성능이 아닙니다. 어댑터 선택에만 썼습니다.
전체 기록은 [`docs/train_record.json`](docs/train_record.json) 에 있습니다.

## 평가

### 설정

| 항목 | 값 |
|---|---|
| 생성 모델 | `google/gemma-4-E2B-it` (유효 2.3B · GGUF Q8_0) |
| 실행 | `llama-server` · 컨텍스트 10240 · `temperature 0` · GPU 층 99 · 스레드 4 |
| 기계 | AWS g6.xlarge (NVIDIA L4) |
| 문항 | 81 |
| 통제 | 문항·후보 조각·생성 설정 동일. 재순위 모델만 교체 |

평가 문항은 재순위 없는 기본 RAG 입력에서 E2B 가 틀렸고, **그 틀린 값이 같은 입력 안의 다른
조각에서 온** 문항만 모은 것입니다. 답은 입력에 있었는데 모델이 다른 조각을 읽은 경우입니다.
이 기준에 따라 기준선은 0 / 81 이며, 각 모델이 맞힌 수가 곧 기준선 대비 순증입니다.

### 결과

위 [주요 결과](#주요-결과) 표와 같습니다. 오른 이유는 정답 근거를 입력 1순위에 놓은 문항이
18 → 79 로 늘었기 때문입니다. 같은 81문항을 순위 변화 여부로 가르면 더 분명합니다.

| 문항 묶음 | 문항 | 공개 BGE | DKU |
|---|---|---|---|
| 정답 근거의 순위가 바뀐 문항 | 61 | 6 / 61 (10%) | **47 / 61 (77%)** |
| 정답 근거의 순위가 그대로인 문항 | 18 | 11 / 18 (61%) | 11 / 18 (61%) |

순위가 그대로인 문항에서는 두 모델이 같습니다. 성능 향상은 문항 난이도가 아니라 순위 변화에서
옵니다.

### 확인하지 못한 것

| 항목 | 결과 |
|---|---|
| 상위 28개 선별 | 이 81문항에서 차이를 확인하지 못했습니다 (`p = 1`) |
| 질문 선행 배치 | 이 81문항에서 차이를 확인하지 못했습니다 (`p = 1`) |
| 다른 생성 모델 | 측정하지 않았습니다 |
| 다른 언어·문서 종류 | 측정하지 않았습니다 |

확인된 효과는 재순위 모델 학습 하나입니다.

## 한계

- 한국 네 기관의 보고서 44편에서 만든 264문항으로 학습했습니다
- 생성 모델 하나(E2B), 문항 81개에서만 검증했습니다
- 평가 집합은 기본 RAG 가 틀린 문항만 모은 것으로, 일반적인 질문 분포가 아닙니다
- 「답이 입력에 있는데 다른 조각을 읽는」 실패 하나만 다룹니다. 검색이 가져오지 못한 경우는 고치지 못합니다
- 비교 대상은 공개 `bge-reranker-v2-m3` 하나입니다

## 저장소 구성

```
model/adapter_model.safetensors   LoRA 어댑터 (8.9MB)
model/adapter_config.json         어댑터 설정
src/build_c2_trainset.py          학습셋 구성
src/make_doc_folds_v2.py          문서 단위 분할
src/train_c2_targeted.py          LoRA 학습
docs/train_record.json            학습 기록 (설정 · 손실 · epoch 선택)
```

## 인용

> 양성훈, "DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub 저장소, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```

## 라이선스

`BAAI/bge-reranker-v2-m3` 의 파생물이며 원본과 같은 Apache License 2.0 을 따릅니다.
원본 모델의 라이선스와 저작자 표시를 함께 지켜야 합니다. 전문은 [LICENSE](LICENSE) 에 있습니다.

## 문의

석사 학위논문 연구 과정에서 학습·공개한 어댑터입니다. 문의는 <sh.yang@dankook.ac.kr> 로 주십시오.
