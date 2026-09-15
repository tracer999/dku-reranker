# DKU-reranker-v1

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)
[![Language](https://img.shields.io/badge/language-Korean--only-red.svg)](#)

**한국어** | [English](README.en.md) | [日本語](README.ja.md) | [中文](README.zh.md)

**공개 재순위 모델 `BAAI/bge-reranker-v2-m3` 를 한국어 공공문서의 문체에 특화시킨 판입니다.**
한국어 PDF 문서를 대상으로 하는 검색증강생성(RAG)에서, 소형 언어모델(SLM)에 넘길 근거 조각의
순위를 바로잡습니다.

기반 모델은 다국어 모델이라 폭은 넓지만 한 언어의 문체까지 다루지는 않습니다. 이 어댑터는 본체
가중치를 동결한 채, 한국어 공공기관 PDF 보고서에서 사람이 표시한 정답 근거 라벨로 LoRA 어댑터만
학습했습니다. **한국어 전용이며, 다른 언어에서의 동작은 확인하지 않았습니다.**

## 왜 만들었나

소형 언어모델 기반 RAG 에서 반복되는 실패 하나를 겨냥했습니다. **정답 근거가 입력에 들어 있는데도
모델이 다른 조각의 값을 답으로 내놓는 경우**입니다.

원인은 순위였습니다. 공공기관 보고서에는 연도별·지역별로 형식이 같은 표가 반복됩니다. 공개 재순위
모델은 질문과의 관련도만 평가하므로 같은 주제의 표가 모두 높은 점수를 받고, 그중 정답이 담긴 하나를
가려내지 못합니다.

그래서 관련도 대신 **정답 근거 여부**를 학습 신호로 삼았습니다. 본체는 동결하고 LoRA 어댑터만
학습해, 공개 모델의 일반적인 성능과 배포 비용을 그대로 유지했습니다.

효과는 직접 측정했습니다. 재순위 모델만 교체했을 때 정답 근거가 1순위인 문항이 18 → 79 로,
정답 수가 17 → 58 로 늘었습니다. 측정에 사용한 어댑터 파일을 그대로 공개합니다.

## 주요 결과

Gemma 4 E2B 로 81문항을 평가했습니다. 문항·후보 조각·생성 설정은 모두 같고 **재순위 모델만**
바꿨습니다. 평가는 **모두 한국어 문항과 한국어 문서**를 대상으로 했습니다.

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

## 누구를 위한 것인가

한국에서 소형 언어모델 기반 RAG를 도입하려는 곳이라면 어디든 쓸 수 있도록 공개합니다. 개인
개발자, 기업, 공공기관, 학교와 연구기관을 모두 염두에 두고 만들었습니다.

| 쓰는 곳 | 어떤 자리에서 |
|---|---|
| 개인 개발자 | 한국어 문서로 RAG를 만들어 보는 단계에서, 공개 재순위 모델을 그대로 바꿔 끼울 수 있습니다 |
| 기업 | 사내 문서·보고서 질의응답에 큰 모델을 쓰기 어려운 환경에서 |
| 공공기관 | 자료를 기관 밖으로 내보내지 않고 기관 안에서 직접 운영해야 하는 환경에서 |
| 학교·연구기관 | 한국어 RAG 연구의 비교 기준이나 출발점으로 |

어댑터 크기가 8.9MB로 작아서, 이미 `bge-reranker-v2-m3`을 쓰고 계시다면 한 줄만 더하면 바로
써 보실 수 있습니다.

## 사용 목적

### 적합한 용도

- **표준 한국어로 작성된 문서**에 대한 질의응답 — 학습 자료가 표준어를 정확하게 쓰는 문서입니다
- 표에서 값을 찾아 답하는 과제 — 비슷한 표가 여러 번 나오는 상황을 겨냥해 학습했습니다
- 소형 언어모델(수 B 규모)을 쓰는 RAG — 근거 위치의 영향이 큰 환경일수록 효과를 기대할 수 있습니다
- 컨텍스트 예산이 빠듯해 상위 몇 개만 넣어야 하는 환경 — 무엇을 앞에 둘지가 특히 중요해집니다

### 범위 밖 용도

아래는 이 어댑터가 검증되지 않았거나, 애초에 하도록 만들지 않은 용도입니다.

- 한국어가 아닌 문서. 기반 모델이 다국어라 동작은 하지만, 이 어댑터는 한국어로만 학습했고 다른 언어에서의 이득은 확인된 바 없습니다
- 표가 거의 없는 서술 중심 문서. 학습 자료가 표 중심이라 그런 문서에서는 효과를 장담하기 어렵습니다
- 검색 자체의 대체. 이 어댑터는 순위만 다시 매길 뿐이라, 애초에 후보에 없는 조각은 끌어올 수 없습니다
- 조각 본문의 가공. 이 어댑터는 조각을 지우거나 요약하거나 다시 쓰지 않고, 순서만 바꿉니다

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

실제로 써 보실 때 아래를 참고하시면 도움이 됩니다.

| 항목 | 이렇게 하면 좋습니다 | 까닭 |
|---|---|---|
| 배치 크기 | 하나로 고정해 주세요 | 배치가 달라지면 묶음 안 패딩 길이가 달라져 점수가 미세하게 흔들립니다 |
| 최대 길이 | 1024를 그대로 쓰시면 됩니다 | 학습할 때 쓴 값이라, 바꾸면 점수의 기준도 함께 바뀝니다 |
| 입력 차례 | `(질문, 조각)` 순서를 지켜 주세요 | 뒤바꾸면 점수가 달라집니다 |
| 점수 사용 | 절댓값이 아니라 상대 순위로만 써 주세요 | 절댓값은 질문마다 분포가 달라, 고정 임계값을 두면 그 기준이 질문마다 옮겨 갑니다 |
| 조각 경계 | 표는 표째로, 절은 절째로 넣어 주세요 | 학습할 때 쓴 조각 단위와 같아야 점수가 안정적입니다 |
| 자르기(상위 몇 개까지 쓸지) | 이 어댑터 바깥, 쓰시는 쪽에서 정해 주세요 | 이 어댑터는 순서를 정렬만 할 뿐, 몇 개를 쓸지는 관여하지 않습니다 |

## 학습 자료

한국어 공공기관 PDF 보고서입니다. 각 기관이 공개한 보고서를 해당 기관 누리집에서 직접
내려받았습니다. 예산·결산 분석, 농업·산업 통계, 교통 조사, 정보화 이슈분석 등 표가 많은
정책 보고서입니다. 질문마다 정답의 근거가 되는 조각을 사람이 확인해 표시했습니다.

공공기관 문서를 고른 까닭은 **표준어를 정확하게 쓰고 문체가 기관 사이에도 일정하기** 때문입니다.
문체가 흔들리면 학습 신호에 그 흔들림이 섞여 들어갑니다.

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

비교를 공정하게 만들려고, 재순위 모델을 빼면 세 조건이 완전히 같은 입력을 받도록 맞췄습니다.
문항도 같고 후보 조각도 같고 생성 설정도 같습니다. 바뀌는 것은 재순위 모델 하나뿐이라, 정답
수의 차이는 재순위 모델이 만든 차이로 볼 수 있습니다.

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
  school = {단국대학교 정보융합기술·창업대학원},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```

## 라이선스

`BAAI/bge-reranker-v2-m3` 의 파생물이며 원본과 같은 Apache License 2.0 을 따릅니다.
원본 모델의 라이선스와 저작자 표시를 함께 지켜야 합니다. 전문은 [LICENSE](LICENSE) 에 있습니다.

## 소속

**단국대학교 정보융합기술·창업대학원** 에서 학습·공개하였습니다.

| 항목 | 내용 |
|---|---|
| 대학교 | 단국대학교 · <https://www.dankook.ac.kr> |
| 대학원 | 정보융합기술·창업대학원 · <https://cms.dankook.ac.kr/web/gict> |

## 문의

질문이나 제안은 [Issues](https://github.com/tracer999/dku-reranker/issues) 또는
<sh.yang@dankook.ac.kr> 로 주십시오.
