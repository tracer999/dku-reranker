# DKU-reranker

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

효과는 직접 측정했으며, 재순위 모델만 교체했을 때 정답 근거를 입력 1순위에 놓는 비율과 정답 수가
함께 올랐습니다. 측정에 사용한 어댑터 파일을 그대로 공개합니다.

## 주요 결과

**학습에 쓰지 않은 문서**에서 평가했습니다. 학습에 쓴 44편과 문서가 한 편도 겹치지 않는 새 보고서
46편을 따로 모아, 그중 80문항으로 측정했습니다. 문항·후보 조각·생성 설정은 모두 같고 **재순위
모델만** 바꿨습니다. 생성 모델은 Gemma 4 E2B 입니다.

| 재순위 모델 | 정답 | 정답 근거가 1순위 | 정답 근거보다 앞에 놓인 다른 표 |
|---|---|---|---|
| 없음 (기본 RAG) | 0 / 80 | — | — |
| `BAAI/bge-reranker-v2-m3` (공개) | 69 / 80 | 64 / 80 | 4 |
| 이전 판 | 60 / 80 | 27 / 80 | 22 |
| **DKU-reranker (본 어댑터)** | **71 / 80** | **65 / 80** | **3** |

이전 판보다 **11문항을 더 맞혔습니다** (McNemar 양측 정확검정 `p = 0.0034`). 오른 까닭은
분명합니다 — 정답 근거보다 앞에 다른 표가 놓인 문항이 **22 → 3** 으로 줄었습니다.

★ **학습에 쓰지 않은 문서에서 잰 값**입니다. 학습 문서에서 재면 값이 더 높게 나오지만, 그것은
새 문서에 적용했을 때의 기대치가 되지 못합니다.

⛔ 공개 모델과의 차이(69 → 71)는 이 문항 수에서 통계적으로 유의하지 않습니다(`p = 0.754`).
**「공개 모델보다 낫다」고는 말하지 않습니다.** 이 표가 보이는 것은 *학습이 순위를 바로잡는다*는
것이며, 그 크기가 이전 판 대비 `p = 0.0034` 로 확인되었다는 것입니다.

⛔ 값은 평가 조건에 따라 달라집니다. 평가 설계는 [평가](#평가)에 적었습니다.

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

| 출처 기관 | 문서 |
|---|---|
| 국회예산정책처 (nabo) | 10 |
| 한국농촌경제연구원 (krei) | 13 |
| 한국교통연구원 (koti) | 9 |
| 한국지능정보사회진흥원 (nia) | 12 |
| **합계** | **44** |

학습 문항은 **353개**입니다. 세 갈래를 함께 씁니다.

| 갈래 | 문항 | 무엇을 가르치나 |
|---|---|---|
| 정답 근거 분리 | 98 | 정답 근거를 비정답 근거보다 위로 |
| 순서 보존 | 166 | 원래 앞서던 차례를 뒤집지 않도록 |
| **표 지목** | **89** | 「표 N-N 에서 …」처럼 **표 번호로 지목하는 질문**에서 그 표를 고르도록 |

표 지목 갈래는 같은 머리행을 가진 표가 여러 개일 때 제목이 없는 표를 고르는 실패를 겨냥합니다.

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
| step | 1,870 |
| 손실 | 정답 근거 점수를 올리는 항 + 정답 근거를 비정답 근거보다 앞세우는 항 (가중 1 : 1.0) |
| 학습 문항 | 353 (정답 근거 분리 98 · 순서 보존 166 · 표 지목 89) |
| 장치 | NVIDIA L4 (AWS g6.xlarge) · 198분 |

어느 epoch 의 어댑터를 쓸지는 **학습을 한 step 도 돌리기 전에** 규칙으로 고정했습니다.
① 순서 위반율이 학습 전을 넘는 epoch 은 제외 ② 남은 것 중 정답 근거 분리 지표가 가장 높은 것
③ 동률이면 더 이른 epoch. 규칙에 따라 epoch 5 를 골랐습니다.

| 지표 (학습셋) | 학습 전 | epoch 5 |
|---|---|---|
| 정답 근거가 비정답 근거보다 모두 앞서는 문항 비율 | 0.6738 | **0.8877** |
| 순서 위반율 (낮을수록 좋음) | 0.2866 | **0.082** |

두 지표는 학습셋에서 잰 적합도이며 하류 과제 성능이 아닙니다. 어댑터 선택에만 썼습니다.
전체 기록은 [`docs/train_record.json`](docs/train_record.json) 에 있습니다.

## 평가

### 어떻게 재야 하나

| 항목 | 값 |
|---|---|
| 생성 모델 | `google/gemma-4-E2B-it` (유효 2.3B · GGUF Q8_0) |
| 실행 | `llama-server` · 컨텍스트 10240 · `temperature 0` · GPU 층 99 · 스레드 4 |
| 통제 | 문항·후보 조각·생성 설정 동일. **재순위 모델만 교체** |

비교를 공정하게 만들려면 재순위 모델을 빼고 나머지를 전부 같게 맞춰야 합니다. 문항도 같고 후보
조각도 같고 생성 설정도 같아야, 정답 수의 차이를 재순위 모델이 만든 차이로 볼 수 있습니다.

### 어떤 문항으로 재야 하나

이 어댑터가 겨냥하는 실패는 좁습니다. **정답 근거가 입력에 들어 있는데 모델이 같은 입력 안의 다른
조각에서 값을 가져오는 경우**입니다. 그래서 평가 문항도 그 조건으로 모아야 합니다.

1. 재순위 없는 기본 RAG 입력에서 생성 모델이 **틀린** 문항을 고릅니다.
2. 그 틀린 값이 **같은 입력 안의 다른 조각에 실제로 있는지** 확인합니다.
3. 값을 지어낸 경우는 뺍니다 — 원인을 조각 순위로 돌릴 수 없습니다.

이렇게 모으면 기준선(재순위 없음)의 정답이 0 이 되고, 각 모델이 맞힌 수가 곧 순증이 됩니다.

### 무엇을 함께 봐야 하나

정답 수만 보면 왜 올랐는지 알 수 없습니다. 아래 둘을 함께 재면 원인이 드러납니다.

| 무엇 | 왜 |
|---|---|
| **정답 근거가 입력 1순위인 문항 수** | 이 어댑터가 직접 바꾸는 값입니다 |
| **정답 근거보다 앞에 놓인 다른 표의 수** | 실패가 나는 자리입니다. 줄어야 정답이 늡니다 |

그리고 문항을 **순위가 바뀐 것**과 **그대로인 것**으로 갈라 보십시오. 순위가 그대로인 문항에서는
두 모델이 같아야 합니다. 거기서도 차이가 나면 순위가 아닌 다른 것이 섞인 것입니다.

> ⛔ **위 [주요 결과](#주요-결과)의 값은 한 벌의 측정입니다.** 평가 조건(문서 종류·질문 형식·후보
> 조각 구성)에 따라 값이 달라집니다. 위 설계로 각자의 문서에서도 재어 보시기를 권합니다.

## 저장소 구성

```
model/adapter_model.safetensors   LoRA 어댑터 (8.9MB)
model/adapter_config.json         어댑터 설정
src/build_c2_trainset.py          학습셋 구성
src/make_doc_folds_v2.py          문서 단위 분할
src/train_c2_targeted.py          LoRA 학습
src/make_G_지목문항.py            표 지목 문항 생성 (학습 문서에서)
src/add_G_후보조각.py             그 문항에 후보 조각 30개를 붙인다
docs/train_record.json            학습 기록 (설정 · 손실 · epoch 선택)
```

## 인용

> 양성훈, "DKU-reranker: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub 저장소, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker,
  title  = {DKU-reranker: A LoRA Adapter for Korean Public-Document Evidence Reranking},
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
