r"""문서 단위 3겹 분할 **제2안** — 계열(prefix)을 **층화 제약**으로 둔다. ⛔ 모델 호출 0.

⛔ **제1안을 덮어쓰지 않는다.** 제1안은 `fc15581` · `DKU_문서3겹_manifest_v10.json` 그대로 둔다.
⛔ **전체 학습을 시작하지 않는다.** 제안일 뿐이다.
⛔ **미개봉 74 의 질문·답·근거·라벨·본문을 열지 않는다.**

## ★ 왜 제2안인가 (amy 2026-09-14)

```
제1안의 계열 분포   krei 9/3/9 · nabo 3/9/5
★ 기관·자료 계열 편중이 커서 **겹별 검증 차이가 모델이 아니라 출처 구성으로 흔들릴 수 있다**
★ 그리고 이것은 **결과를 보고 바꾸는 것이 아니다** —
  모델 학습 · 성능 · 시험 결과가 **아직 0** 이고, 관측한 것은 **공변량 균형**이다.
  ⇒ 학습 전 공변량 보정은 **사후 성능 최적화가 아니다**
```

## ★★★ 알고리즘 — **실행 이전에 고정한다** (⛔ 이 파일을 먼저 커밋하고 실행한다)

```
⛔ **난수를 쓰지 않는다. seed 가 없다** — 완전 결정적이다 (제1안과 같다)

### ⛔ hard constraint 둘
  ① **계열별 문서 수의 겹 간 차이 ≤ 1**
     ⇒ 합계가 정해져 있으므로 multiset 이 **유일하게** 결정된다 —
       krei 21 → 7/7/7 · nia 23 → 8/8/7 · nabo 17 → 6/6/5 · koti 13 → 5/4/4
  ② **겹 문서 정원 25 / 25 / 24**

### 최소화 목적함수 — **사전순**
  1차 **qid 수 최대차** · 2차 **양성 수 최대차** · 3차 **강한방해 수 최대차**

### 세 단계 · 전부 결정적
  ㉠ ①②를 **동시에** 만족하는 계열 배치를 **전부 열거**한다 (5가지)
  ㉡ 배치마다 **그리디** — 문서를 (qid↓ · 양성↓ · 강한방해↓ · `document_id`↑) 로 집어,
     **그 계열의 할당량이 남은 겹 중** (qid · 양성 · 강한방해 · 겹 번호) 최소인 곳에 넣는다
  ㉢ 배치마다 **쌍 교환 개선** — **같은 계열**의 문서 둘을 다른 겹 사이에서 맞바꾼다
     (계열 할당량과 정원이 보존된다). 모든 쌍을 `document_id` 오름차순으로 훑어
     **목적함수를 낮추는 첫 교환**을 적용하고, 더 낮출 수 없을 때까지 반복한다
     ⇒ 목적함수가 단조 감소하므로 **반드시 끝난다**
  ⇒ 다섯 결과 중 **(목적함수, 배치 인덱스)** 사전순 최소를 고른다

최종 tie-break — **`document_id` 오름차순** · **겹 번호 오름차순**
⛔ **결과를 본 뒤 손으로 옮기지 않는다** — ㉢ 은 알고리즘의 일부이고 여기 미리 적혀 있다
```

## ⛔ 검증 — 제1안과 같은 다섯 + 계열 제약

```
① 겹 사이 문서 교집합 0 · qid 교집합 0
② 항등식 합 = 74문서 · 488 qid · 800 양성 · 13,840 음성 · 9,679 강한방해
③ 겹마다 문서 수가 정확히 25 · 25 · 24
④ 미개봉 시험 문서 11편과 교집합 0
⑤ ★ **계열별 문서 수의 겹 간 차이 ≤ 1**
```

## ⛔ 채택 판단은 내가 하지 않는다

```
amy 지시 — 「제2안이 계열 균형은 개선하지만 **qid/양성 불균형이 실질적으로 과도해지면
           채택하지 말고 다시 결정을 요청해라**」
⇒ 제1안과 제2안의 최대차를 **나란히 보고**한다
```
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations, product
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "results/v92/DKU학습_마스터_488_v10.jsonl"
제1안 = ROOT / "results/smoke/DKU_문서3겹_manifest_v10.json"
OUT = ROOT / "results/smoke/DKU_문서3겹_manifest_제2안_v10.json"
정원 = (25, 25, 24)

# ⛔ `DECISIONS.md` 13.1 「시험 문서 11편」 — **이름만**. 내용은 열지 않았다
시험문서11 = [
    "koti_BFCKvxKUvZge", "koti_XZP40PLUS40U", "krei_0a664282", "krei_1e50eec0",
    "krei_5f2a2053", "nabo_9278", "nabo_9289", "nabo_9307", "nabo_9339",
    "nia_39485_24113", "nia_39485_24294",
]


def 멈춘다(까닭: str):
    print(f"⛔⛔ {까닭}")
    sys.exit(2)


rows = [json.loads(l) for l in MASTER.open(encoding="utf-8") if l.strip()]
문서: dict[str, dict] = defaultdict(lambda: {"qid": set(), "양성": 0, "음성": 0, "강": 0})
문항양성: Counter = Counter()
for r in rows:
    d = 문서[r["document_id"]]
    d["qid"].add(r["qid"])
    d["양성" if r["is_gold"] else "음성"] += 1
    d["강"] += bool(r["강한방해"])
    문항양성[r["qid"]] += bool(r["is_gold"])
for k in 문서:
    문서[k]["n"] = len(문서[k]["qid"])
    문서[k]["pre"] = k.split("_")[0]

전체 = {"문서": len(문서), "qid": len({r["qid"] for r in rows}), "줄": len(rows),
      "양성": sum(d["양성"] for d in 문서.values()),
      "음성": sum(d["음성"] for d in 문서.values()),
      "강한방해": sum(d["강"] for d in 문서.values())}
계열합 = Counter(문서[k]["pre"] for k in 문서)
print(json.dumps(전체, ensure_ascii=False), "·", dict(계열합), flush=True)

# ── ㉠ hard constraint 를 만족하는 계열 배치를 전부 열거 ───────────────
몫: dict[str, tuple] = {}
for p, n in 계열합.items():
    q, r2 = divmod(n, 3)
    몫[p] = tuple(sorted([q + 1] * r2 + [q] * (3 - r2), reverse=True))   # 차이 ≤ 1 ⇒ 유일
키 = sorted(몫)
배치들 = [dict(zip(키, c)) for c in product(*[sorted(set(permutations(몫[p]))) for p in 키])
        if tuple(sum(c[i][j] for i in range(len(키))) for j in range(3)) == 정원]
if not 배치들:
    멈춘다("정원 25/25/24 와 계열 차이 ≤1 을 **동시에** 만족하는 배치가 없다")
print(f"★ 계열 배치 후보 {len(배치들)} 가지", flush=True)

차례 = sorted(문서, key=lambda k: (-문서[k]["n"], -문서[k]["양성"], -문서[k]["강"], k))


def 목적(배정: dict[str, int]) -> tuple:
    """(qid 최대차, 양성 최대차, 강한방해 최대차) — 작을수록 좋다."""
    s = [[0, 0, 0] for _ in range(3)]
    for k, i in 배정.items():
        d = 문서[k]
        s[i][0] += d["n"]; s[i][1] += d["양성"]; s[i][2] += d["강"]
    return tuple(max(x[j] for x in s) - min(x[j] for x in s) for j in range(3))


def 한배치(배치: dict[str, tuple]) -> dict[str, int]:
    # ㉡ 그리디
    남 = {p: list(배치[p]) for p in 배치}
    합 = [[0, 0, 0] for _ in range(3)]
    배정: dict[str, int] = {}
    for k in 차례:
        d = 문서[k]; p = d["pre"]
        후보 = [i for i in range(3) if 남[p][i] > 0]
        if not 후보:
            멈춘다(f"{k}({p}) 를 넣을 겹이 없다")
        i = min(후보, key=lambda j: (합[j][0], 합[j][1], 합[j][2], j))
        배정[k] = i; 남[p][i] -= 1
        합[i][0] += d["n"]; 합[i][1] += d["양성"]; 합[i][2] += d["강"]
    # ㉢ 결정적 쌍 교환 — 같은 계열끼리만 (할당량·정원이 보존된다)
    계열별 = defaultdict(list)
    for k in sorted(문서):
        계열별[문서[k]["pre"]].append(k)
    for _ in range(2000):
        best = None
        cur = 목적(배정)
        for p in sorted(계열별):
            ks = 계열별[p]
            for a in range(len(ks)):
                for b in range(a + 1, len(ks)):
                    x, y = ks[a], ks[b]
                    if 배정[x] == 배정[y]:
                        continue
                    배정[x], 배정[y] = 배정[y], 배정[x]
                    v = 목적(배정)
                    배정[x], 배정[y] = 배정[y], 배정[x]
                    if v < cur:
                        best = (x, y)
                        break
                if best:
                    break
            if best:
                break
        if not best:
            break
        x, y = best
        배정[x], 배정[y] = 배정[y], 배정[x]
    return 배정


후보들 = [(목적(b := 한배치(배치)), n, b) for n, 배치 in enumerate(배치들)]
for 목, n, _ in 후보들:
    print(f"  배치 {n}: qid차 {목[0]} · 양성차 {목[1]} · 강한방해차 {목[2]}", flush=True)
목표, 뽑은n, 배정 = min(후보들, key=lambda t: (t[0], t[1]))
print(f"⇒ 고른 배치 {뽑은n} · 목적함수 {목표}", flush=True)

겹 = [{"문서": [], "qid": 0, "양성": 0, "음성": 0, "강": 0, "계열": Counter()} for _ in range(3)]
for k in sorted(문서):
    i = 배정[k]; d = 문서[k]
    겹[i]["문서"].append(k); 겹[i]["qid"] += d["n"]; 겹[i]["양성"] += d["양성"]
    겹[i]["음성"] += d["음성"]; 겹[i]["강"] += d["강"]; 겹[i]["계열"][d["pre"]] += 1

# ── ⛔ 검증 ─────────────────────────────────────────────────────────
집합 = [set(f["문서"]) for f in 겹]
qid집합 = [set().union(*(문서[k]["qid"] for k in f["문서"])) for f in 겹]
for a in range(3):
    for b in range(a + 1, 3):
        if 집합[a] & 집합[b]:
            멈춘다(f"겹 {a}·{b} 문서 교집합 {len(집합[a] & 집합[b])}")
        if qid집합[a] & qid집합[b]:
            멈춘다(f"겹 {a}·{b} qid 교집합 {len(qid집합[a] & qid집합[b])}")
for i, f in enumerate(겹):
    if len(f["문서"]) != 정원[i]:
        멈춘다(f"겹 {i} 문서 {len(f['문서'])} ≠ 정원 {정원[i]}")
합2 = {"문서": sum(len(f["문서"]) for f in 겹), "qid": sum(f["qid"] for f in 겹),
      "양성": sum(f["양성"] for f in 겹), "음성": sum(f["음성"] for f in 겹),
      "강한방해": sum(f["강"] for f in 겹)}
for k2 in 합2:
    if 합2[k2] != 전체[k2]:
        멈춘다(f"항등식 깨짐 — {k2}: 합 {합2[k2]} ≠ 전체 {전체[k2]}")
계열차 = {p: max(f["계열"][p] for f in 겹) - min(f["계열"][p] for f in 겹) for p in 계열합}
if max(계열차.values()) > 1:
    멈춘다(f"⛔ 계열 차이가 1 을 넘는다: {계열차}")
겹침 = sorted(set(문서) & set(시험문서11))
if 겹침:
    멈춘다(f"⛔⛔ 미개봉 시험 문서가 학습 풀에 있다: {겹침}")

# ── 입력 길이 — ⛔ 토크나이저만. 모델 호출 0 ──────────────────────────
from transformers import AutoTokenizer                    # noqa: E402
tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
길이: dict[tuple, int] = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(0, len(rows), 256):
        b = rows[i:i + 256]
        enc = tok([x["question"] for x in b], [x["chunk_text"] for x in b],
                  truncation=False, padding=False)
        for x, t in zip(b, enc["input_ids"]):
            길이[(x["qid"], x["chunk_id"])] = len(t)


def 분포(xs: list[int]) -> dict:
    s = sorted(xs)
    return {"n": len(s), "p50": s[len(s) // 2], "p95": s[int(len(s) * .95)], "max": s[-1],
            "1024초과": sum(1 for v in s if v > 1024),
            "1024초과 %": round(100 * sum(1 for v in s if v > 1024) / len(s), 2)}


한 = json.loads(제1안.read_text(encoding="utf-8")) if 제1안.exists() else None


def 최대차(fs, key) -> int:
    return max(f[key] for f in fs) - min(f[key] for f in fs)


결과 = {
    "무엇": "DKU 학습용 문서 단위 3겹 분할 **제2안** — 계열(prefix)을 **층화 제약**으로 둔다",
    "⛔ 제안이다": "**전체 학습을 시작하지 않았다.**",
    "⛔ 모델 호출": 0,
    "⛔ 제1안을 덮어쓰지 않았다": str(제1안),
    "언제(KST)": datetime.now().isoformat(timespec="seconds"),
    "★ 변경 사유 (amy 2026-09-14)": {
        "무엇이 문제였나": "제1안의 계열 분포 krei 9/3/9 · nabo 3/9/5 — 기관·자료 계열 편중이 커서 "
                     "겹별 검증 차이가 모델이 아니라 **출처 구성**으로 흔들릴 수 있다",
        "⛔ 사후 최적화가 아니다": "모델 학습·성능·시험 결과가 **아직 0** 이다. "
                          "관측한 것은 결과가 아니라 **공변량 균형**이며, 학습 전 보정은 사후 최적화가 아니다",
    },
    "★ 알고리즘": {
        "⛔ 난수": "**쓰지 않는다. seed 가 없다** — 완전 결정적이다",
        "⛔ hard constraint": ["계열별 문서 수의 겹 간 차이 ≤ 1", "겹 문서 정원 25 / 25 / 24"],
        "계열 multiset(합계에서 유일하게 결정된다)": {p: list(몫[p]) for p in 키},
        "계열 배치 후보 수": len(배치들), "고른 배치": 뽑은n,
        "최소화 목적함수(사전순)": "① qid 수 최대차 ② 양성 수 최대차 ③ 강한방해 수 최대차",
        "단계": "㉠ 배치 전수 열거 → ㉡ 결정적 그리디 → ㉢ 같은 계열 문서의 결정적 쌍 교환 개선",
        "tie-break": "document_id 오름차순 · 겹 번호 오름차순",
        "⛔ 손으로 옮기지 않았다": "㉢ 은 알고리즘의 일부이며 실행 이전에 커밋했다",
        "배치별 목적함수": {str(n): list(목) for 목, n, _ in 후보들},
    },
    "전체": 전체,
    "겹": [],
    "⛔ 검증": {
        "겹 사이 문서 교집합": {f"{a}∩{b}": len(집합[a] & 집합[b])
                        for a in range(3) for b in range(a + 1, 3)},
        "겹 사이 qid 교집합": {f"{a}∩{b}": len(qid집합[a] & qid집합[b])
                        for a in range(3) for b in range(a + 1, 3)},
        "항등식": {"합": 합2, "전체": {k2: 전체[k2] for k2 in 합2},
                "일치": all(합2[k2] == 전체[k2] for k2 in 합2),
                "★ 참고 `줄`": {"합": sum(f["양성"] + f["음성"] for f in 겹), "전체": 전체["줄"]}},
        "★ 계열별 겹 간 차이": 계열차,
        "⛔ 미개봉 시험 문서 11편과 교집합": len(겹침),
    },
}
for i, f in enumerate(겹):
    ls = [길이[(r["qid"], r["chunk_id"])] for r in rows if r["qid"] in qid집합[i]]
    결과["겹"].append({
        "겹": i, "문서": len(f["문서"]), "qid": f["qid"], "양성": f["양성"],
        "음성": f["음성"], "강한방해": f["강"],
        "계열(prefix)": dict(sorted(f["계열"].items())),
        "qid 당 양성 분포": {str(k2): v for k2, v in
                       sorted(Counter(문항양성[q] for q in qid집합[i]).items())},
        "qid 당 양성 평균": round(f["양성"] / f["qid"], 3),
        "입력 길이(무절단)": 분포(ls),
        "document_id": sorted(f["문서"]),
    })

if 한:
    결과["★★ 제1안과 나란히"] = {
        "제1안": {"출처": "fc15581", "qid 최대차": 최대차(한["겹"], "qid"),
                "양성 최대차": 최대차(한["겹"], "양성"), "강한방해 최대차": 최대차(한["겹"], "강한방해"),
                "계열": {f'겹{f["겹"]}': f["계열(prefix)"] for f in 한["겹"]},
                "계열별 겹 간 차이": {p: max(f["계열(prefix)"].get(p, 0) for f in 한["겹"])
                                - min(f["계열(prefix)"].get(p, 0) for f in 한["겹"]) for p in 계열합}},
        "제2안": {"qid 최대차": 최대차(결과["겹"], "qid"),
                "양성 최대차": 최대차(결과["겹"], "양성"), "강한방해 최대차": 최대차(결과["겹"], "강한방해"),
                "계열": {f'겹{f["겹"]}': f["계열(prefix)"] for f in 결과["겹"]},
                "계열별 겹 간 차이": 계열차},
        "⛔ 채택 판단": "**내가 하지 않는다.** amy 지시대로 두 안의 수를 나란히 올린다",
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
t = OUT.with_suffix(OUT.suffix + ".partial")
t.write_text(json.dumps(결과, ensure_ascii=False, indent=1), encoding="utf-8")
t.replace(OUT)
간 = {k: v for k, v in 결과.items() if k != "겹"}
간["겹"] = [{k2: v2 for k2, v2 in f.items() if k2 != "document_id"} for f in 결과["겹"]]
print(json.dumps(간, ensure_ascii=False, indent=1))
print(f"\n✅ {OUT}")
