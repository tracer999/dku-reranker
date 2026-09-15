# DKU-reranker-v1

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

## 설치와 사용

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
model = PeftModel.from_pretrained(model, "./model")   # DKU 어댑터 결합
model.eval()

question = "2025회계연도 국가통합 순자산변동표에서 조정항목의 적립금 및 잉여금은 몇 억원인가?"
chunks = ["...조각 1...", "...조각 2...", "...조각 3..."]

enc = tok([question] * len(chunks), chunks,
          padding=True, truncation=True, max_length=1024, return_tensors="pt")
with torch.no_grad():
    scores = model(**enc).logits.view(-1).float()

order = scores.argsort(descending=True)
ranked = [chunks[i] for i in order]      # 점수가 높은 순
```

⛔ 배치를 너무 크게 잡으면 일부 장치에서 값이 달라진다. **배치 8** 로 나누어 계산하기를 권한다.

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

## 라이선스

이 어댑터는 `BAAI/bge-reranker-v2-m3` 의 파생물이며 원본과 같은 **Apache License 2.0** 을 따른다.
원본 모델의 라이선스와 저작자 표시를 함께 지켜야 한다.

## 인용

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Kim, Jinseok},
  year   = {2026},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```
