r"""C′ 표적 학습 — **D 는 gold 손실 · F 는 A 순서 보존 손실**. 3-fold · 이어 학습.

```
    ~/venvs/thesis-v92/bin/python code/train/train_c2_targeted.py \
        --학습셋 results/v92/C2학습셋_262.jsonl \
        --adapter자리 results/l4_run/out_canonical \
        --out results/v92/c2_adapter \
        --epochs 5 --lr 1e-4 --dfratio 1.0 --wgold 1.0 --worder 1.0
```

## ⛔⛔ 두 손실은 **갈라져 있다** — 섞이면 안 된다

```
D (`집합=="D"`)  →  `손실_gold()`   listwise grouped softmax CE + **1/|P_q|**
                   「이 근거가 답을 받친다」
F (`집합=="F"`)  →  `손실_order()`  **A 순서 보존** — A 에서 i 가 j 보다 앞이면 s_i > s_j
                   ⛔ **gold 를 올리라고 하지 않는다**
★★ `한_문항_손실()` 이 `집합` 으로 갈라 **둘 중 하나만** 부른다. 더하지 않는다
```

★★ 왜 F 에 gold 손실을 안 주나 — **어젯밤 진단이 말한다**
```
손상 21 중 **9 건이 「gold 가 전부 앞으로 갔는데도 틀렸다」**
⇒ ⇒ gold 를 올리는 것 자체가 손상을 만든다 ⇒ F 에 가르치면 **B 를 따라 한다**
```

## ⛔ held-out — **fold k 를 학습에 쓰지 않는다**

```
fold k 의 adapter 를 만들 때 **`fold != k` 인 문항만** 쓴다
⇒ ★ 그래야 C 와 같은 잣대(OOF)가 되고 **C ↔ C′ 가 나란히 선다**
⛔ 문서가 두 fold 에 걸치면 held-out 이 깨진다 — **문항의 `fold` 는 문서 단위로 배정됐다**
```

## ⛔ 하지 않는 것

```
⛔ **from scratch 로 학습하지 않는다** — `adapter_fold{k}_best` 에서 **이어** 한다
⛔ **문항마다 가중치를 주지 않는다** — 사후에 고른 문항을 성공이라 부르는 자리가 된다
⛔ **생성 모델(E2B)을 건드리지 않는다** — 재순위 모델만이다 (절대 규칙 1)
```
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, random, sys, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAXLEN = 1024


def 점수내기(m, tok, q, texts, dev):
    """조각마다 점수 하나. ⛔ micro-batch 는 조각 수만큼이다(문항 단위 손실이라 함께 넣는다)."""
    import torch
    enc = tok([q] * len(texts), texts, truncation=True, max_length=MAXLEN,
              padding=True, return_tensors="pt").to(dev)
    return m(**enc).logits.float().view(-1)


def 손실_gold(s, gold_idx, n):
    """**D 용** — 각 양성 대 **모든 non-gold** 의 pairwise softplus 평균.

        L = mean_{i in P} mean_{j in N} softplus(s_j - s_i)

    ★ 양성끼리 비교가 **아예 없다.** 그래서 서로 경쟁하지 않는다.
    우리 목적이 「**요구 근거를 전부 보존**」이므로 이 꼴이라야 한다.

    ## ⛔⛔ 2026-09-15 — **앞의 구현이 틀렸다** (amy 가 찾고 alex 가 확인 · leo 가 실측)

    ```
    ⛔ 앞 구현   lse = logsumexp(s)                    ← **전체**에 대한 lse
                -(sum(s[i] - lse for i in P) / |P|)
    ⛔ 무엇이 틀렸나  분모에 **다른 양성이 들어 있다** ⇒ 양성 하나를 올리면
                  다른 양성의 분모도 커진다 ⇒ ★ **양성끼리 경쟁한다**
    ⛔ docstring 이 「양성끼리 서로 밀어내지 않게」라고 적고 있었다 — **거짓이었다**
    ```

    ★★ **돌려서 gradient 부호를 봤다** (`torch.autograd` · 양성 2 · 음성 4):

    ```
    s=[6.0, 0.5, 0, 0, 0, 0]   앞 구현 양성 grad [+0.486, -0.496] ⇒ **높은 양성에 「내려라」**
    s=[5.0, 4.5, 0, 0, 0, 0]   앞 구현 양성 grad [+0.112, -0.129] ⇒ **같은 꼴**
    ⇒ ⇒ ★ **gold 가 이미 최상위인 문항을 망가뜨리는 방향**이었다
    이 구현   위 둘 다 [-0.001, -0.189] · [-0.003, -0.006] ⇒ **둘 다 올려라**
    ```

    ⛔ 다른 안(`-logsumexp(pos) + logsumexp(all)`)은 쓰지 않는다 —
    **양성 하나만 올려도 만족**해서 「전부 보존」과 어긋난다.
    """
    import torch
    import torch.nn.functional as F
    P = list(gold_idx)
    N = [j for j in range(n) if j not in set(P)]
    if not P or not N:
        # ⛔ 견줄 것이 없으면 **0 을 돌려주되 그래프는 잇는다** — 빈 값으로 끊지 않는다
        return s.sum() * 0.0
    return torch.stack([F.softplus(s[j] - s[i]) for i in P for j in N]).mean()


def 규칙읽기(p: str) -> dict:
    """⛔ **선택 규칙은 수를 보기 전에 파일로 고정한다.** 없으면 멈춘다."""
    if not p:
        raise SystemExit("⛔⛔ `--선택규칙` 이 없다 — **어느 epoch 을 쓸지 수를 보기 전에 정해야 한다**")
    q = Path(p)
    if not q.exists():
        raise SystemExit(f"⛔⛔ 선택 규칙 파일이 없다: {p} — ⛔ **없는 것을 「기본값」으로 읽지 않는다**")
    return json.loads(q.read_text(encoding="utf-8"))


def 규칙적용(epoch기록: list, 규칙: dict):
    """규칙대로 epoch 하나를 고른다. ⛔ 규칙 밖의 판단을 넣지 않는다."""
    기준선 = next((e for e in epoch기록 if e.get("★ 학습 전 기준선")), None)
    후보 = [e for e in epoch기록 if not e.get("★ 학습 전 기준선")]
    if 기준선 and 규칙.get("① 보존 제약") and 기준선.get("F 순서 위반율") is not None:
        여유 = float(규칙.get("보존 여유", 0.0))
        후보 = [e for e in 후보 if e.get("F 순서 위반율") is not None
                and e["F 순서 위반율"] <= 기준선["F 순서 위반율"] + 여유]
    if not 후보:
        return {"고른 epoch": None, "⛔ 까닭": "보존 제약을 통과한 epoch 이 없다 — **사람이 본다**"}
    best = max(후보, key=lambda e: ((e.get("D 구제 proxy") or 0.0), -e["epoch"]))
    return {"고른 epoch": best["epoch"], "adapter": best.get("adapter"),
            "D 구제 proxy": best.get("D 구제 proxy"), "F 순서 위반율": best.get("F 순서 위반율"),
            "★ 규칙": 규칙, "통과한 epoch": [e["epoch"] for e in 후보]}


def 손실_order(s):
    """**F 용** — A 순서 보존. 입력 차례가 곧 A 의 차례다.

    ⛔ **gold 를 올리라고 하지 않는다.** 앞에 있던 것이 앞에 있게만 한다.
    앞 i < 뒤 j 마다 `softplus(s_j - s_i)` — s_i 가 더 크면 0 에 가깝다.
    """
    import torch
    n = s.shape[0]
    if n < 2:
        return s.sum() * 0.0
    i, j = torch.triu_indices(n, n, offset=1)
    return torch.nn.functional.softplus(s[j] - s[i]).mean()


def 한_문항_손실(r, m, tok, dev, wgold, worder):
    """⛔⛔ **집합으로 갈라 둘 중 하나만** 부른다. 더하지 않는다."""
    s = 점수내기(m, tok, r["question"], r["chunk_texts"], dev)
    if r["집합"] == "D":
        gi = [r["chunk_ids"].index(c) for c in r["gold_ids"]]
        return wgold * 손실_gold(s, gi, len(r["chunk_ids"])), "gold"
    else:                      # F — ⛔ gold 를 쓰지 않는다
        return worder * 손실_order(s), "order"


def 그룹표집(D, rng, epoch=0):
    """⛔⛔ **epoch 당 D 호출을 원래 행 수로 고정하고** 그룹을 round-robin 으로 채운다.

    ⛔ 옛 판(2026-09-15 11:2x)은 **가장 큰 그룹 크기(32)에 맞춰 29 그룹을 전부 oversample** 했다.
      ⇒ D 98행이 한 epoch 에 **928회** 불렸다 —
        ① 1행 그룹이 같은 행을 **32번** 되풀이한다(과적합) · ② epoch 길이가 **9.47배**가 된다
      ★ 그룹 균형은 섰지만 **그 값을 과표집으로 샀다.** amy 가 짚었다

    ✅ 새 판 — **슬롯을 나눠 준다**
      ㉠ epoch 당 D 호출 = **len(D)** 로 고정 (98)
      ㉡ 그룹을 **섞어** round-robin 으로 98 슬롯을 나눈다 ⇒ 그룹마다 3 또는 4 (max−min ≤ 1)
      ㉢ 그룹 **안에서는 섞고 돌려 쓴다** — `epoch` 을 오프셋으로 주어 **다음 epoch 이 다음 행**을 본다
         ⇒ 큰 그룹의 행이 영영 안 뽑히는 일이 없다
    """
    from collections import defaultdict
    g = defaultdict(list)
    for r in D:
        g[str(r.get("group_key") or "?")].append(r)
    키 = sorted(g)
    rng.shuffle(키)                      # ㉡ 그룹 차례를 섞는다 — ⛔ 앞 그룹이 늘 4를 받지 않게
    n = len(D)
    몫 = {k: n // len(키) for k in 키}
    for k in 키[: n % len(키)]:
        몫[k] += 1                        # 나머지를 앞에서부터 하나씩 ⇒ max−min ≤ 1
    뽑힌 = []
    for k in 키:
        v = g[k][:]
        rng.shuffle(v)                   # ㉢ 그룹 안 섞기
        off = epoch % max(len(v), 1)     # ㉢ epoch 마다 시작점을 옮긴다
        for i in range(몫[k]):
            뽑힌.append(v[(off + i) % len(v)])
    rng.shuffle(뽑힌)
    return 뽑힌, {k: 몫[k] for k in 키}, {k: len(v) for k, v in g.items()}


def 균형표집(D, F, ratio, rng, epoch=0):
    """배치마다 **D 와 F 를 `ratio` 로** 뽑는다. ⛔ 그냥 섞으면 F 가 배치를 지배한다.

    ★ D 안은 **그룹 round-robin** 이다 — 동일 가중도 아니고 oversample 도 아니다.
    """
    d, 몫, 그룹크기 = 그룹표집(D, rng, epoch)
    f = F[:]
    rng.shuffle(f)
    한바퀴 = []
    nF = int(round(len(d) * ratio))
    fi = 0
    붙일 = max(int(round(nF / max(len(d), 1))), 1) if F else 0
    for x in d:
        한바퀴.append(x)
        for _ in range(붙일):
            if not f:
                break
            한바퀴.append(f[fi % len(f)]); fi += 1
    return 한바퀴


def main() -> int:
    ap = argparse.ArgumentParser(description="C′ 표적 학습")
    ap.add_argument("--학습셋", required=True)
    ap.add_argument("--adapter자리", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dfratio", type=float, default=1.0, help="배치의 F:D 비율. 1.0 = 1:1")
    ap.add_argument("--wgold", type=float, default=1.0)
    ap.add_argument("--worder", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--선택규칙", default="",
                    help="⛔ **어느 epoch 의 adapter 를 쓸 것인가**를 적은 json. "
                         "**수를 보기 전에** 만들어 커밋한다. ⛔ 없으면 멈춘다")
    ap.add_argument("--모드", choices=("full", "oof"), default="oof",
                    help="`full` = 표적 전부로 한 adapter (⇒ **주 재시험의 C**) · "
                         "`oof` = fold 별 adapter (⇒ **전이 진단 · 별도 표**). "
                         "⛔ 둘은 **목적이 다르다.** 섞어 쓰지 않는다")
    ap.add_argument("--F검증출처", default="results/v92/C2학습셋_262.jsonl",
                    help="지금 F172 가 든 학습셋. ⛔ 옛 F166 을 주지 마라 — 대칭차 검사의 기준이다")
    ap.add_argument("--기대그룹수", type=int, default=None,
                    help="D 안 그룹 수의 기대값. ⛔ 자료에서 자동으로 받지 않는다 — 그러면 순환이다")
    ap.add_argument("--기대그룹수출처", default="results/v92/학습정본_100_검사.json",
                    help="`--기대그룹수` 를 안 주면 이 파일의 `★ 샘플링 그룹.그룹 수` 를 쓴다")
    ap.add_argument("--preflight-only", action="store_true", help="⛔ 학습하지 않고 preflight 만 낸다")
    # ⛔⛔ **측정용** — 기계가 바뀌면 **초/step 을 먼저 재고** 전체를 환산한다
    #   2026-09-15 맥북에서 77분에 50 step 을 못 돌아 정지했는데, 그때 **초/step 을 몰랐다**.
    #   `n % 50` 마다 찍는 첫 신호가 50 step 뒤라 그 전에 죽으면 수가 남지 않는다
    #   ⇒ 이 인자로 몇 step 만 돌아 **그 기계의 초/step 을 먼저 안다.** ⛔ adapter 를 안 남긴다
    ap.add_argument("--최대step", type=int, default=0,
                    help="⛔ 측정용 — 이만큼 돌고 멈춘다(adapter 를 안 남긴다). 0 이면 끝까지")
    ap.add_argument("--preflight-out", default="results/v92/DKU_C프라임_preflight.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.학습셋, encoding="utf-8") if l.strip()]
    folds = sorted({r["fold"] for r in rows})
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    # ⛔⛔ **CUDA 를 먼저 본다** (2026-09-15 leo) — 전에는 이 줄에 `cuda` 가 없어서
    #   GPU 가 있는 기계에서도 **조용히 CPU 로 떨어졌다.** 맥북에서는 값이 바뀌지 않는다
    #   (cuda 가 없으므로 그대로 mps 다) ⇒ 지금까지의 학습과 같은 장치를 쓴다
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    print("★ 학습 장치 %s" % dev, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    ad자리 = Path(a.adapter자리)
    rng = random.Random(a.seed)

    # ── ⛔ preflight — **학습을 걸기 전에 넷을 확인한다** ──────────────────
    pf = {
      "무엇": "C′ 표적 학습 **preflight** — ⛔ **착수 전에** 넷이 코드에 실제로 구현됐나",
      "언제": datetime.datetime.now().isoformat(timespec="seconds"),
      "⛔ 왜": "amy 요구 — 「「구현했다」가 아니라 **「코드의 어느 줄이 그것을 하나」**를 적어라」",
      "학습셋": {"경로": a.학습셋, "줄": len(rows),
              "sha256": hashlib.sha256(Path(a.학습셋).read_bytes()).hexdigest()[:16],
              "집합": dict(Counter(r["집합"] for r in rows))},
      "고정값 (⛔ 학습 전에 박는다)": {"epochs": a.epochs, "lr": a.lr,
              "D:F 비율": f"1:{a.dfratio}", "가중": {"gold": a.wgold, "order": a.worder},
              "seed": a.seed, "max_length": MAXLEN, "device": dev,
              "⛔ 왜 1:1 인가": "**근거 없이 기울이지 않는다.** c 를 겨누는 목표는 철회됐다(alex·amy)"},
    }

    # ㉠ 두 손실이 갈라져 있나
    import inspect
    src = inspect.getsource(한_문항_손실)
    pf["㉠ 두 손실이 갈라져 있나"] = {
      "판정": "✅" if ('r["집합"] == "D"' in src and "손실_gold" in src and "손실_order" in src) else "⛔",
      "어느 줄이": "`한_문항_손실()` — `if r[\"집합\"] == \"D\": … 손실_gold(...)` / "
              "`else: … 손실_order(...)`",
      "★ 더하지 않는다": "`return` 이 **둘 중 하나**다. 두 손실을 합산하는 줄이 **없다**",
      "⛔ F 에 gold 가 섞이나": "⛔ **아니다** — `else` 가지는 `r[\"gold_ids\"]` 를 **참조하지 않는다**",
      "★ 코드": src,
    }
    # ㉡ 균형 표집이 실제로 도나 — 배치 하나를 꺼내 센다
    D0 = [r for r in rows if r["집합"] == "D"]
    F0 = [r for r in rows if r["집합"] == "F"]
    표본 = 균형표집(D0, F0, a.dfratio, random.Random(a.seed))[:40]
    c = Counter(x["집합"] for x in 표본)
    # ⛔⛔ **㉣ 첫 epoch 의 그룹별 표집 질량을 실측한다** (2026-09-15 · amy)
    #   ⛔ 가중치 파일이 있는 것만으로 통과시키지 않는다 — **실제로 몇 번 불렸나**를 센다
    한바퀴 = 균형표집(D0, F0, a.dfratio, random.Random(a.seed))
    D한바퀴 = [x for x in 한바퀴 if x["집합"] == "D"]
    그룹호출 = Counter(str(x.get("group_key") or "?") for x in D한바퀴)
    그룹행수 = Counter(str(x.get("group_key") or "?") for x in D0)
    질량 = {k: 그룹호출[k] / max(len(D한바퀴), 1) for k in 그룹호출}
    고름 = (max(질량.values()) - min(질량.values())) if 질량 else 0.0
    pf["㉡b D 안 그룹 균형"] = {
      "판정": "⛔ 채우는 중",
      "어느 줄이": "`그룹표집()` 과 `균형표집()` — 그룹을 섞어 **round-robin 으로 슬롯을 나눈다**",
    }
    # ⛔⛔ **관문 다섯을 실측한다** (2026-09-15 11:4x · amy)
    #   ① D 호출수 == 원행수 · ② F 호출수 == 원행수(ratio 1) ·
    #   ③ 그룹별 호출 max−min ≤ 1 · ④ 그룹 누락 0 · ⑤ **seed 재현**
    한바퀴 = 균형표집(D0, F0, a.dfratio, random.Random(a.seed))
    D한바퀴 = [x for x in 한바퀴 if x["집합"] == "D"]
    F한바퀴 = [x for x in 한바퀴 if x["집합"] == "F"]
    그룹호출 = Counter(str(x.get("group_key") or "?") for x in D한바퀴)
    그룹행수 = Counter(str(x.get("group_key") or "?") for x in D0)
    누락 = sorted(set(그룹행수) - set(그룹호출))
    벌림 = (max(그룹호출.values()) - min(그룹호출.values())) if 그룹호출 else 0
    다시 = [x["qid"] for x in 균형표집(D0, F0, a.dfratio, random.Random(a.seed))]
    재현 = (다시 == [x["qid"] for x in 한바퀴])
    # ⛔⛔ **그룹이 하나면 「최대−최소」는 언제나 0 이다** (2026-09-15 11:5x · leo 가 시험해서 잡았다)
    #   ⛔ `group_key` 를 **통째로 빼도** ㉡b 가 ✅ 로 통과했다 — 그룹 수 1 · 최대−최소 0.0
    #   ⇒ ⇒ 「막는다고 적혀 있는데 안 막는다」의 오늘 **여섯째** 얼굴이다
    #   ✅ 그래서 둘을 더한다 —
    #     ⓐ **그룹 수 == 기대값** (모르면 ⛔ · 자료에서 자동으로 받지 않는다 — 그러면 순환이다)
    #     ⓑ **`group_key` 가 빠진 D 행 == 0**
    #   ⛔ leo 가 권한 「D 한바퀴 > D 행 수」는 **못 쓴다** — 새 표집판은 D 호출을 원행 수로 **고정**한다
    빈키 = [x["qid"] for x in D0 if not str(x.get("group_key") or "").strip()]
    기대 = a.기대그룹수
    if 기대 is None:
        try:
            기대 = json.loads(Path(a.기대그룹수출처).read_text(encoding="utf-8"))["★ 샘플링 그룹"]["그룹 수"]
        except Exception:
            기대 = None
    ok_a = (기대 is not None and len(그룹행수) == 기대)
    ok5 = (ok_a and not 빈키
           and len(D한바퀴) == len(D0)
           and (not F0 or len(F한바퀴) == len(D0))
           and 벌림 <= 1 and not 누락 and 재현)
    pf["㉡b D 안 그룹 균형"] = {
      "판정": "✅" if ok5 else "⛔",
      "어느 줄이": "`그룹표집()` 과 `균형표집()` — 그룹을 섞어 **round-robin 으로 슬롯을 나눈다**",
      "① D 호출수 == 원행수": {"D 원행": len(D0), "D 호출": len(D한바퀴),
                           "판정": "✅" if len(D한바퀴) == len(D0) else "⛔"},
      "② F 호출수": {"F 원행": len(F0), "F 호출": len(F한바퀴),
                   "판정": "✅" if (not F0 or len(F한바퀴) == len(D0)) else "⛔",
                   "⛔ ratio": a.dfratio},
      "③ 그룹별 호출 max−min": {"값": 벌림, "판정": "✅" if 벌림 <= 1 else "⛔",
                            "그룹 수": len(그룹행수),
                            "호출 분포": dict(Counter(그룹호출.values()))},
      "④ 그룹 누락": {"값": 누락, "판정": "✅" if not 누락 else "⛔"},
      "ⓐ 그룹 수 == 기대값": {"기대": 기대, "실제": len(그룹행수),
                          "판정": "✅" if ok_a else "⛔",
                          "⛔ 왜 필요한가": ("**그룹이 하나면 최대−최소는 언제나 0** 이다. "
                                        "`group_key` 를 통째로 빼도 ③ 이 통과한다 — leo 가 시험해서 잡았다"),
                          "출처": a.기대그룹수출처 if a.기대그룹수 is None else "--기대그룹수"},
      "ⓑ group_key 가 빠진 D 행": {"값": 빈키[:5], "수": len(빈키),
                                "판정": "✅" if not 빈키 else "⛔"},
      "⑤ seed 재현": {"판정": "✅" if 재현 else "⛔",
                    "⛔ 무엇을 쟀나": "같은 seed 로 두 번 뽑아 **qid 차례가 글자까지 같은가**"},
      "★ 한 epoch 의 뜻": {
        "D 호출/epoch": len(D한바퀴), "F 호출/epoch": len(F한바퀴),
        "총 step/epoch": len(한바퀴),
        "⛔ 옛 판과의 차이": ("옛 판은 가장 큰 그룹 크기(%d)만큼 29 그룹을 **oversample** 해 "
                        "D 를 928회 불렀다 — 1행 그룹이 같은 행을 32번 되풀이하고 "
                        "epoch 길이가 9.47배가 됐다. ⇒ 지금은 **원행 수로 고정**한다"
                        % (그룹행수.most_common(1)[0][1] if 그룹행수 else 0)),
        "★ 큰 그룹의 행은 어떻게 도나": "`epoch` 을 오프셋으로 주어 **다음 epoch 이 다음 행**을 본다",
      },
    }
    # ⛔⛔ **F 집합 대칭차 검사** (2026-09-15 · amy) — 낡은 F 로 학습하는 것을 막는다
    #   ⛔ 옛 F166 은 **판정 갱신 전 구성**이었다 — 지금 F172 에 없는 `ko_nabo_9346_05` 가 있고
    #     `ko_nia_39485_28916_05` 가 빠져 있었다
    #   ⇒ ★ **지금 F172 − 확정결함6** 과 **글자까지 같아야** 한다
    _결함6 = ("ko_krei_0254c2e0_08", "ko_krei_fd5a8d6f_06", "ko_nabo_9390_01",
             "ko_nia_39485_24794_05", "ko_krei_cab7ed38_01", "ko_krei_9a4e1535_06")
    _qF = {r["qid"] for r in F0}
    _대칭차, _기대수 = None, None
    if a.F검증출처 and os.path.exists(a.F검증출처):
        _원 = [json.loads(l) for l in open(a.F검증출처, encoding="utf-8") if l.strip()]
        _q172 = {x["qid"] for x in _원 if x.get("집합") == "F"}
        _기대 = _q172 - set(_결함6)
        _대칭차 = sorted(_qF ^ _기대)
        _기대수 = len(_기대)
    pf["㉥ F 집합 대칭차"] = {
      "판정": "✅" if (_대칭차 == []) else "⛔",
      "어느 줄이": "지금 F172 에서 **확정 결함 6 만** 뺀 집합과 대조한다",
      "출처": a.F검증출처,
      "기대 F 수": _기대수, "실제 F 수": len(_qF),
      "⛔ 대칭차": _대칭차 if _대칭차 is not None else "⛔ 출처를 못 읽었다 — **검사하지 못했다**",
      "㉠ 28916_05 포함": ("ko_nia_39485_28916_05" in _qF),
      "㉡ 9346_05 제외": ("ko_nabo_9346_05" not in _qF),
      "⛔ 왜 필요한가": "옛 F166 은 **판정 갱신 전 구성**이다. 되살리면 낡은 F 로 학습한다",
    }
    pf["㉡ 균형 표집이 실제로 도나"] = {
      "판정": "✅" if c["D"] and c["F"] else "⛔",
      "어느 줄이": "`균형표집()` — D 하나마다 F 를 `ratio` 만큼 끼워 넣는다",
      "★ 앞 40 을 세어 본 실제 비율": dict(c),
      "⛔ 그냥 섞으면": f"D {len(D0)} ↔ F {len(F0)} 이라 **F 가 배치를 지배**한다",
    }
    # ㉢ fold 별 warm-start 와 held-out
    ho = {}
    for k in folds:
        학습 = [r for r in rows if r["fold"] != k]
        검증 = [r for r in rows if r["fold"] == k]
        ad = ad자리 / f"adapter_fold{k}_best"
        ho[f"fold{k}"] = {
          "warm-start": str(ad), "adapter 있나": (ad / "adapter_model.safetensors").exists(),
          "adapter sha": hashlib.sha256((ad / "adapter_model.safetensors").read_bytes()).hexdigest()[:16]
                         if (ad / "adapter_model.safetensors").exists() else "⛔ 없다",
          "학습 문항": len(학습), "⛔ held-out(학습에 안 씀)": len(검증),
          "⛔ 겹침": len({r["qid"] for r in 학습} & {r["qid"] for r in 검증}),
        }
    # ⛔⛔ `full` 모드에는 fold 가 없다 — **fold adapter 를 찾는 검사 자체가 뜻이 없다**
    #   ⇒ 2026-09-15 에 이것이 실제로 걸렸다. `full` 로 걸려는데 ㉢ 이 ⛔ 라 통과가 불가능했다
    if a.모드 == "full":
        ad = ad자리 / "adapter_model.safetensors"
        후보 = [ad] + [ad자리 / f"adapter_fold{k}_best" / "adapter_model.safetensors" for k in folds]
        있는것 = ad if ad.exists() else None
        pf["㉢ warm-start (모드 full)"] = {
          "판정": "✅" if 있는것 else "⛔",
          "모드": "full — **표적 전부로 한 adapter**. ⇒ 주 재시험의 C 가 이것을 쓴다",
          "warm-start": str(ad), "있나": bool(있는것),
          "sha": hashlib.sha256(ad.read_bytes()).hexdigest()[:16] if 있는것 else "⛔ 없다",
          "학습 문항": len(rows), "⛔ held-out": 0,
          "⛔⛔ 그래서 무엇을 조심하나": "**학습에 쓴 문항이 평가 분모에 들어간다.** "
                "⇒ 주 재시험의 격을 **「동일 문항 학습 적합도·교정 성능」**으로 명시하고, "
                "**전이는 `oof` 표가 따로 본다**. ⛔ 둘을 한 표에 섞지 않는다",
          "★ 찾아본 자리": [str(x) for x in 후보],
        }
        # ⛔⛔ **`㉡b` 가 검사키에 빠져 있었다** (2026-09-15 11:3x · amy preflight 감사)
        #   ⇒ ㉡b 가 ⛔ 여도 「다섯 다 통과」가 **참**이 되어 학습이 시작된다
        #   ★ 「막는다고 적혀 있는데 안 막는다」의 오늘 다섯째 얼굴이다 — **양쪽 모드에 다 넣는다**
        검사키 = ("㉠ 두 손실이 갈라져 있나", "㉡ 균형 표집이 실제로 도나",
                "㉡b D 안 그룹 균형", "㉥ F 집합 대칭차",
                "㉢ warm-start (모드 full)", "㉣ 문서-fold 배치", "㉤ gold 손실의 gradient 부호")
    else:
        검사키 = ("㉠ 두 손실이 갈라져 있나", "㉡ 균형 표집이 실제로 도나",
                "㉡b D 안 그룹 균형", "㉥ F 집합 대칭차",
                "㉢ fold 별 warm-start 와 held-out", "㉣ 문서-fold 배치", "㉤ gold 손실의 gradient 부호")
    pf["㉢ fold 별 warm-start 와 held-out"] = {
      "판정": ("✅" if all(v["adapter 있나"] and v["⛔ 겹침"] == 0 for v in ho.values()) else "⛔")
              if a.모드 == "oof" else "— (모드 full 에서는 보지 않는다)",
      "어느 줄이": "`for k in folds:` 안에서 `학습 = [r for r in rows if r[\"fold\"] != k]` · "
              "`PeftModel.from_pretrained(base, adapter_fold{k}_best, is_trainable=True)`",
      "★ 이것이 깨지면": "**C ↔ C′ 이 나란히 안 선다** — C 도 같은 fold adapter 로 채점했다",
      "fold 별": ho,
    }
    # ㉣ 새 문항의 문서-fold 배치
    문서fold = {}
    걸친문서 = []
    for r in rows:
        d = r["document_id"]
        문서fold.setdefault(d, set()).add(r["fold"])
    걸친문서 = [d for d, v in 문서fold.items() if len(v) > 1]
    pf["㉣ 문서-fold 배치"] = {
      "판정": "✅" if not 걸친문서 else "⛔",
      "문서 수": len(문서fold),
      "⛔ 두 fold 에 걸친 문서": len(걸친문서), "목록": 걸친문서[:5],
      "★ 왜 보나": "같은 문서가 두 fold 에 걸치면 **held-out 이 깨진다**",
      "⛔ 증설 문항은 아직 없다": "오면 **그 문서의 fold** 로 넣어야 한다. "
              "⇒ 이 검사를 **증설 병합 뒤 다시 돌린다**",
    }
    # ㉤ ⛔⛔ **gold 손실을 실제로 돌려 gradient 부호를 본다**
    #   ★ 「구현했다」가 아니라 **「돌려서 부호를 봤다」**여야 한다 (amy 지시 · 2026-09-15)
    #   ⇒ 목적은 「**요구 근거를 전부 보존**」이다 ⇒ **양성이 둘이면 둘 다 ↑ 여야 한다**
    사례 = {"① 두 양성이 대칭": [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
           "② 한 양성만 매우 높다": [6.0, 0.5, 0.0, 0.0, 0.0, 0.0],
           "③ 두 양성이 이미 최상위": [5.0, 4.5, 0.0, 0.0, 0.0, 0.0]}
    P시험, n시험 = [0, 1], 6
    grad기록, 모두올림 = {}, True
    for 이름, v in 사례.items():
        t = torch.tensor(v, requires_grad=True)
        L = 손실_gold(t, P시험, n시험)
        L.backward()
        g = t.grad.tolist()
        올림 = [g[i] < -1e-9 for i in P시험]
        모두올림 = 모두올림 and all(올림)
        grad기록[이름] = {
          "s": v, "손실": round(L.item(), 6),
          "양성 grad": [round(g[i], 6) for i in P시험],
          "⇒ 방향": ["↑올려라" if g[i] < -1e-9 else ("↓내려라" if g[i] > 1e-9 else "·유지") for i in P시험],
          "음성 grad(둘)": [round(g[j], 6) for j in range(n시험) if j not in P시험][:2],
        }
    pf["㉤ gold 손실의 gradient 부호"] = {
      "판정": "✅" if 모두올림 else "⛔",
      "무엇을 보나": "**양성 둘을 함께 올리는 방향인가.** ⛔ 하나에라도 「↓내려라」가 나오면 실격",
      "★ 왜 보나": "앞 구현은 `logsumexp(전체)` 를 분모로 써서 **양성끼리 경쟁**했다. "
              "⇒ ②③ 에서 **높은 양성에 「내려라」**가 나왔고, 그것은 "
              "**gold 가 이미 최상위인 문항을 망가뜨리는 방향**이다",
      "⛔ 이것은 실측이다": "`torch.autograd` 로 실제로 backward 를 돌린 값이다",
      "사례별": grad기록,
    }
    pf["⛔ 모드"] = a.모드
    pf["⇒ 다섯 다 통과했나"] = all(pf[k]["판정"] == "✅" for k in 검사키)
    pf["⇒ 넷 다 통과했나"] = pf["⇒ 다섯 다 통과했나"]   # ★ 옛 키를 남긴다 — 읽는 쪽이 있다
    pp = Path(a.preflight_out); pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(pf, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk != "★ 코드"})
                      for k, v in pf.items()}, ensure_ascii=False, indent=1))
    if a.preflight_only:
        print("⛔ `--preflight-only` — **학습하지 않고 멈춘다**")
        return 0 if pf["⇒ 다섯 다 통과했나"] else 1
    if not pf["⇒ 다섯 다 통과했나"]:
        raise SystemExit("⛔⛔ preflight 가 통과하지 못했다 — **학습하지 않는다**")

    # ── 학습 ────────────────────────────────────────────────────────
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    기록 = {"고정값": pf["고정값 (⛔ 학습 전에 박는다)"], "모드": a.모드,
          "⛔ epoch 선택 규칙": 규칙읽기(a.선택규칙), "fold": {}}

    def 지표(m, 표본):
        """**epoch 마다 재는 둘** — ⛔ 학습셋으로 재는 학습 적합도다. held-out 이 아니다.

        ㉠ **relevance 구제 proxy** — D 에서 `min(gold) > max(non-gold)` 인 문항의 비율
        ㉡ **F 순서 보존 위반율** — F 에서 A 차례가 깨진 쌍의 비율
        """
        m.eval()
        구제, Dn, 위반, 쌍 = 0, 0, 0, 0
        with torch.no_grad():
            for r in 표본:
                sc = 점수내기(m, tok, r["question"], r["chunk_texts"], dev)
                if r["집합"] == "D":
                    gi = [r["chunk_ids"].index(c) for c in r["gold_ids"]]
                    ni = [i for i in range(len(r["chunk_ids"])) if i not in set(gi)]
                    Dn += 1
                    if gi and ni and float(min(sc[i] for i in gi)) > float(max(sc[i] for i in ni)):
                        구제 += 1
                else:
                    for i in range(len(sc)):
                        for j in range(i + 1, len(sc)):
                            쌍 += 1
                            if float(sc[i]) <= float(sc[j]): 위반 += 1
        m.train()
        return {"D 구제 proxy": (round(구제 / Dn, 4) if Dn else None), "D 분모": Dn,
                "F 순서 위반율": (round(위반 / 쌍, 4) if 쌍 else None), "F 쌍": 쌍}

    작업 = ([("full", rows, ad자리)] if a.모드 == "full"
           else [(f"fold{k}", [r for r in rows if r["fold"] != k],
                  ad자리 / f"adapter_fold{k}_best") for k in folds])
    for 이름, 학습, warm in 작업:
        D = [r for r in 학습 if r["집합"] == "D"]
        F = [r for r in 학습 if r["집합"] == "F"]
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True, num_labels=1, dtype=torch.float32)
        m = PeftModel.from_pretrained(base, str(warm), is_trainable=True).to(dev)
        # ⛔⛔ **MPS OOM** (2026-09-15 12:07 · 학습 step 1 의 forward 에서 터졌다)
        #   ⛔ D 한 행이 조각 **41~53 개 × 1024 토큰**을 **한 배치**로 올린다(F 는 중앙 10).
        #     기준선은 `no_grad` 라 통과했고 **backward 가 붙는 순간** 활성값이 20 GiB 를 넘었다
        #   ✅ gradient checkpointing — **손실 계산은 한 글자도 안 바뀐다.** 활성값만 다시 계산한다
        #     ⛔ 배치를 쪼개면 listwise 손실이 바뀐다. ⛔ MAXLEN 을 줄이면 **자료**가 바뀐다
        #     ⇒ ★ 둘 다 안 한다. **수학을 그대로 두고 메모리만** 줄이는 길이 이것뿐이다
        try:
            base.config.use_cache = False
            m.gradient_checkpointing_enable()
            if hasattr(m, "enable_input_require_grads"):
                m.enable_input_require_grads()   # ⛔ LoRA + checkpointing 은 이것이 있어야 grad 가 흐른다
            _ckpt = True
        except Exception as _e:
            _ckpt = False
            print("⛔ gradient checkpointing 을 못 켰다: %s" % _e)
        print("★ gradient checkpointing %s" % ("켰다" if _ckpt else "⛔ 못 켰다"))
        m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr)
        n, t0, 손실합 = 0, time.time(), Counter()
        첫20, epoch기록 = [], []
        # ★ epoch 0 — **학습 전** 값을 먼저 잰다. ⛔ 이것이 「보존이 나빠졌나」의 기준선이다
        # ⛔ 측정 모드에서는 건너뛴다 — 2026-09-15 맥북에서 이 한 줄이 **56분**을 썼다
        #   (12:27 시작 → 13:23 에 기준선). 초/step 을 재려는데 56분을 기다릴 수 없다
        if a.최대step:
            print(f"  ⛔ `--최대step` 이라 epoch0 기준선을 건너뛴다 (맥북에서 56분 걸린 자리다)",
                  flush=True)
        else:
            e0 = 지표(m, 학습)
            epoch기록.append({"epoch": 0, "★ 학습 전 기준선": True, **e0})
            print(f"  [{이름} epoch0 기준선] {e0}", flush=True)
        멈춤 = False
        for ep in range(1, a.epochs + 1):
            if 멈춤:
                break
            # ★ `ep` 를 넘긴다 — 그룹 안 시작점이 epoch 마다 옮겨져 큰 그룹의 행이 돌아간다
            차례 = 균형표집(D, F, a.dfratio, rng, ep)
            for r in 차례:
                ts = time.time()
                loss, 갈래 = 한_문항_손실(r, m, tok, dev, a.wgold, a.worder)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                n += 1; 손실합[갈래] += float(loss)
                if n <= 20:
                    걸 = round(time.time() - ts, 3)
                    첫20.append(걸)
                    # ⛔ **조각 수를 함께 찍는다** — 느린 까닭이 배치 크기인지 여기서 갈린다
                    #   (D 행 조각 중앙 41 · F 중앙 10 ⇒ 한 배치가 네 배다)
                    print(f"  [{이름} {n} step] {걸:.2f}초 · {r['집합']} · 조각 {len(r['chunk_texts'])}",
                          flush=True)
                if a.최대step and n >= a.최대step:
                    멈춤 = True
                    break
                if n % 50 == 0:
                    print(f"  [{이름} {n} step] {time.time()-t0:.1f}초 · "
                          f"{(time.time()-t0)/n:.3f}초/step", flush=True)
            if 멈춤:
                # ⛔ 측정용으로 끊은 것이다 — **adapter 를 남기지 않는다.**
                #   반쯤 돈 것을 저장하면 뒤에 오는 사람이 그것을 학습 결과로 읽는다
                print(f"  ⛔ `--최대step {a.최대step}` 으로 멈췄다 — adapter 를 남기지 않는다",
                      flush=True)
                break
            # ⛔ **epoch 마다 저장한다** — 나중에 규칙대로 고를 수 있어야 한다
            ep자리 = out / f"adapter_{이름}_ep{ep}"
            m.save_pretrained(str(ep자리))
            g = 지표(m, 학습)
            epoch기록.append({"epoch": ep, "adapter": str(ep자리), **g})
            print(f"  ★ [{이름} epoch{ep}] {g} → {ep자리.name}", flush=True)
        기록["fold"][이름] = {
          "학습 문항": len(학습), "D": len(D), "F": len(F), "step": n,
          "초": round(time.time() - t0, 1), "초/step": round((time.time() - t0) / max(n, 1), 3),
          "★ 첫 20 step 초": 첫20,
          "손실 합": {k2: round(v, 4) for k2, v in 손실합.items()},
          "★ epoch 별": epoch기록,
          "⇒ 규칙이 고른 epoch": 규칙적용(epoch기록, 기록["⛔ epoch 선택 규칙"]),
        }
        print(f"★ {이름} 끝 — {n} step · {기록['fold'][이름]['초']}초 · "
              f"규칙이 고른 epoch {기록['fold'][이름]['⇒ 규칙이 고른 epoch']}", flush=True)
        del m, base
    (out / "학습기록.json").write_text(json.dumps(기록, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(기록, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
