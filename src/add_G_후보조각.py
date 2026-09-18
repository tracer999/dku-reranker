#!/usr/bin/env python3
r"""집합 **G** 문항에 **후보 조각 30개**를 붙여 학습 대본이 읽는 꼴로 만든다.

    python3 code/train/add_G_후보조각.py

## ⛔ 왜 집합을 **D** 로 넣나 — 손실이 정확히 맞는다

```
D 손실 `손실_gold()`  =  각 양성 대 **모든 비-gold** 의 pairwise softplus 평균
                        L = mean_{i∈P} mean_{j∈N} softplus(s_j − s_i)
G 의 목적             =  **gold 표를 자매표보다 위로**
⇒ ★ 같은 것이다. 그래서 `집합: "D"` 로 넣고 **학습 대본을 한 줄도 고치지 않는다**
⛔ 새 집합 이름을 만들면 대본을 고쳐야 하고 preflight 여덟 검사가 그것을 모른다
```

## 후보 30개를 고르는 규칙 — ⛔ **관측 이전에 정한다**

```
① gold 표 1개                                        ← 양성
② 자매표 **전부** (머리행 절반 이상 겹치고 번호가 다르다) · 최대 9개   ← 강한 음성
③ 같은 문서의 나머지 조각을 **질문과의 낱말 겹침 수** 내림차순으로 채워 **총 30개**
   동률이면 `node_id` 오름차순 ⇒ 재현된다
⛔ 30 은 v10.1 평가 조건과 같은 수다 — 학습과 평가의 조각 수를 맞춘다
⛔ 생성 모델도 재순위 모델도 부르지 않는다. 낱말 겹침은 공백으로 자른 2글자 이상 낱말이다
```
"""
from __future__ import annotations
import argparse, glob, html, io, json, os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
번호 = re.compile(r"표\s*([0-9]+)\s*[-–—.]\s*([0-9]+)")


def 키(s):
    m = 번호.search((s or "").replace("<", "").replace(">", ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def 머리행(t):
    for l in t.splitlines():
        s = l.strip()
        if s.startswith("|") and not s.startswith("|--"):
            return [c.strip() for c in s.split("|")[1:-1]]
    return []


def 낱말(s):
    return {w for w in re.split(r"[\s,()（）:：·ㆍ/\[\]]+", s or "") if len(w) >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", default=os.path.join(ROOT, "results/v10.2/G_지목문항.jsonl"))
    ap.add_argument("--그래프", default=os.path.join(ROOT, "data/graphs/korean_r128"))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--자매상한", type=int, default=9)
    ap.add_argument("--낼곳", default=os.path.join(ROOT, "results/v10.2/G_학습행_89.jsonl"))
    a = ap.parse_args()

    G = [json.loads(l) for l in io.open(a.G, encoding="utf-8") if l.strip()]
    문서 = {}
    for doc in sorted({g["document_id"] for g in G}):
        p = os.path.join(a.그래프, doc + ".json")
        ns = json.load(io.open(p, encoding="utf-8")).get("nodes") or []
        문서[doc] = [(x["id"], html.unescape(x.get("text") or "")) for x in ns if (x.get("text") or "")]

    낸것, 문제 = [], Counter()
    for g in G:
        doc = g["document_id"]; gi = g["gold_ids"][0]
        본 = dict(문서[doc])
        if gi not in 본:
            문제["gold 본문 없음"] += 1; continue
        gh = set(머리행(본[gi])); gk = g["★ 표번호"]
        자매 = sorted(i for i, t in 문서[doc]
                     if i != gi and "#table" in i and 키(t) != gk
                     and len(set(머리행(t)) & gh) >= max(2, len(gh) // 2))[:a.자매상한]
        고름 = [gi] + [x for x in 자매 if x != gi]
        if g["★ 자매표"] not in 고름:          # ⛔ 겨룰 값을 담은 표는 반드시 넣는다
            고름.append(g["★ 자매표"])
        남 = [(i, t) for i, t in 문서[doc] if i not in set(고름)]
        Q = 낱말(g["question"])
        남.sort(key=lambda it: (-len(Q & 낱말(it[1])), it[0]))
        고름 += [i for i, _ in 남[:max(0, a.k - len(고름))]]
        고름 = 고름[:a.k]
        if gi not in 고름:
            문제["gold 가 빠졌다"] += 1; continue
        낸것.append({
            "qid": g["qid"], "fold": None, "집합": "D",
            "question": g["question"], "document_id": doc,
            "chunk_ids": 고름, "chunk_texts": [본[i] for i in 고름],
            "gold_ids": [gi],
            "group_key": f"{gi} → {g['★ 자매표']}",
            "⛔ 어디서 왔나": "v10.2 · 지목 문항(G) — add_G_후보조각.py. ⛔ 집합은 D 로 넣는다(손실이 같다)",
            "★ v10.2 G": {"표번호": gk, "자매표": g["★ 자매표"], "열": g["★ 열"],
                          "행쌍": g["★ 행쌍"], "자매표 수": len(자매)},
        })
    # fold 는 **그 문서의 fold** 를 물려받는다
    학 = [json.loads(l) for l in io.open(os.path.join(ROOT, "results/v10/C2학습셋_D98_F166.jsonl"),
                                        encoding="utf-8") if l.strip()]
    문서fold = {x["document_id"]: x["fold"] for x in 학}
    없 = set()
    for r in 낸것:
        f = 문서fold.get(r["document_id"])
        if f is None: 없.add(r["document_id"])
        r["fold"] = f
    if 없:
        print(f"⛔ fold 를 못 물려받은 문서 {sorted(없)} — 멈춘다"); return 2
    if 문제:
        print(f"⛔ 버린 것 {dict(문제)}")
    os.makedirs(os.path.dirname(a.낼곳), exist_ok=True)
    with io.open(a.낼곳, "w", encoding="utf-8") as f:
        for r in 낸것:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import statistics as st
    크 = [len(r["chunk_ids"]) for r in 낸것]
    유형 = Counter(re.search(r"#([a-z]+)", c).group(1) for r in 낸것 for c in r["chunk_ids"])
    print(f"★ {len(낸것)}행 — 후보 중앙 {st.median(크):.0f} (최소 {min(크)} 최대 {max(크)}) · fold {Counter(r['fold'] for r in 낸것)}")
    print(f"   후보 유형 {dict(유형.most_common())}")
    print(f"   자매표가 후보에 든 문항 {sum(1 for r,g in zip(낸것,G) if g['★ 자매표'] in r['chunk_ids'])}/{len(낸것)}")
    print(f"   낼곳 {os.path.relpath(a.낼곳, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
