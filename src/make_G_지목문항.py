#!/usr/bin/env python3
r"""**집합 G — 「표 N-N 에서 …」 지목 문항**을 학습 문서에서 기계로 만든다.

    python3 code/train/make_G_지목문항.py --낼곳 results/v10.2/G_지목문항.jsonl

★ 왜 만드나 — v10.1 이 찾은 원인(리포트 3.4)

```
v10.1 80문항은 **전부** 「표 N-N 에서 …」로 표를 지목하고 76장은 그 번호가 gold 표 본문에 박혀 있다
그런데 DKU 학습 264문항에는 그런 쌍이 **1건**뿐이다
⇒ DKU 는 그 단서를 못 배웠고, 머리행이 같은 **무제(無題) 자매표**를 gold 앞에 놓는다(22/80)
```

## ⛔ 어디서 뽑나 — **학습 문서 44편뿐이다**

```
✅ 쓴다    results/v10/C2학습셋_D98_F166_재작성.jsonl 의 document_id 44편
⛔ 안 쓴다  v10.1 의 문서 11편 (keei · kiep · kihasa) — 그것을 쓰면 외부 검증이 아니게 된다
⛔ 안 쓴다  시험 분할 74문항의 문서 — 이름도 읽지 않는다
★ 확인     낼 때 문서 목록을 찍고, 44편 밖이 하나라도 있으면 **멈춘다**
```

## 자리가 갖춰야 할 것

```
① gold 표 본문 첫 줄에 「표 N-N」이 있다      ⇒ 질문의 번호가 그 표를 가리킨다
② 같은 문서에 **자매표**가 있다              머리행이 절반 이상 겹치고 번호가 다르거나 없다
③ 행 이름 둘과 열 하나를 골라 값 둘을 읽을 수 있다
④ ⛔ **자매표에도 그 행·열이 있고 두 행 모두 값이 다르다**  ⇒ 겨룰 값이 실재한다. 없으면 버린다
   ⛔ 하나만 다르면 버린다 — 나머지 한 행은 정답=겨룰값이라 함정이 되지 않는다
⑤ 값이 수이고 두 값이 서로 다르다
```

## 낸 뒤에 하는 일

⛔ 이 대본은 **문항만** 만든다. 학습 쌍(양성·음성)은 학습 대본이 만든다.
⛔ 생성 모델을 부르지 않는다. 정답 근거 라벨은 **gold 표 자체**다 — 질문을 그 표에서 만들었기 때문이다.
"""
from __future__ import annotations
import argparse, html, io, json, os, re, sys, random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
번호 = re.compile(r"표\s*([0-9]+)\s*[-–—.]\s*([0-9]+)")
수 = re.compile(r"^-?[0-9][0-9,]*\.?[0-9]*$|^△?-?[0-9][0-9,]*\.?[0-9]*$")


def 키(s: str):
    m = 번호.search((s or "").replace("<", "").replace(">", "").replace("＜", "").replace("＞", ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def 표파싱(t: str):
    """마크다운 표 → (머리행, {행이름: {열이름: 값}})"""
    줄 = [l.strip() for l in t.splitlines() if l.strip().startswith("|") and not l.strip().startswith("|--")]
    if len(줄) < 3:
        return None, None
    칸 = [[c.strip() for c in l.split("|")[1:-1]] for l in 줄]
    머리 = 칸[0]
    if len(머리) < 3:
        return None, None
    표 = {}
    for r in 칸[1:]:
        if len(r) != len(머리) or not r[0]:
            continue
        표[r[0]] = {머리[i]: r[i] for i in range(1, len(머리))}
    return 머리, 표


def 제목(t: str) -> str:
    for l in t.splitlines():
        if l.strip() and not l.strip().startswith("|"):
            return l.strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--학습셋", default=os.path.join(ROOT, "results/v10/C2학습셋_D98_F166.jsonl"),
                    help="문서 44편의 목록을 여기서 얻는다 (문항 내용은 안 쓴다)")
    ap.add_argument("--그래프", default=os.path.join(ROOT, "data/graphs/korean_r128"))
    ap.add_argument("--낼곳", default=os.path.join(ROOT, "results/v10.2/G_지목문항.jsonl"))
    ap.add_argument("--문서당상한", type=int, default=8, help="쏠림을 막는다. v10.1 과 같은 값")
    ap.add_argument("--표당상한", type=int, default=2)
    ap.add_argument("--시드", type=int, default=20260918)
    a = ap.parse_args()

    학문서 = sorted({json.loads(l)["document_id"] for l in io.open(a.학습셋, encoding="utf-8") if l.strip()})
    print(f"학습 문서 {len(학문서)}편 — {Counter(x.split('_')[0] for x in 학문서)}")
    # ⛔ v10.1 계열이 하나라도 섞이면 멈춘다
    막을것 = ("keei_", "kiep_", "kihasa_")
    샌것 = [d for d in 학문서 if d.startswith(막을것)]
    if 샌것:
        print(f"⛔ v10.1 계열이 학습 문서 목록에 있다: {샌것[:3]} — 멈춘다")
        return 2

    rnd = random.Random(a.시드)
    문항, 문서수, 표수 = [], Counter(), Counter()
    자리있음 = 0
    for doc in 학문서:
        p = os.path.join(a.그래프, doc + ".json")
        if not os.path.isfile(p):
            continue
        ns = json.load(io.open(p, encoding="utf-8")).get("nodes") or []
        표들 = [(x["id"], html.unescape(x.get("text") or "")) for x in ns
                if (x.get("type") == "table" or "#table" in x.get("id", "")) and (x.get("text") or "")]
        파싱 = {}
        for i, t in 표들:
            h, tb = 표파싱(t)
            if h and tb:
                파싱[i] = (h, tb, 키(t), 제목(t))
        for gi, (gh, gt, gk, g제목) in 파싱.items():
            if not gk:
                continue                      # ① 번호가 있어야 한다
            자매 = [j for j, (h, tb, k2, _) in 파싱.items()
                    if j != gi and k2 != gk and len(set(h) & set(gh)) >= max(2, len(gh) // 2)]
            if not 자매:
                continue                      # ②
            자리있음 += 1
            if 문서수[doc] >= a.문서당상한:
                continue
            열후보 = [c for c in gh[1:] if any(수.match(v.get(c, "").replace("△", "")) for v in gt.values())]
            행후보 = [r for r in gt if sum(1 for c in 열후보 if 수.match(gt[r].get(c, "").replace("△", ""))) >= 1]
            만든 = 0
            rnd.shuffle(열후보); rnd.shuffle(행후보)
            for 열 in 열후보:
                if 만든 >= a.표당상한 or 문서수[doc] >= a.문서당상한:
                    break
                가능 = [r for r in 행후보 if 수.match(gt[r].get(열, "").replace("△", ""))]
                for i1 in range(len(가능)):
                    if 만든 >= a.표당상한 or 문서수[doc] >= a.문서당상한:
                        break
                    for i2 in range(i1 + 1, len(가능)):
                        r1, r2 = 가능[i1], 가능[i2]
                        v1, v2 = gt[r1][열], gt[r2][열]
                        if v1 == v2:
                            continue          # ⑤ 두 값이 달라야 한다
                        # ④ 자매표에 같은 행·열이 있고 값이 달라야 한다
                        겨룸 = None
                        for j in 자매:
                            h2, tb2, _, _ = 파싱[j]
                            if 열 in h2 and r1 in tb2 and r2 in tb2:
                                w1, w2 = tb2[r1].get(열), tb2[r2].get(열)
                                # ⛔ **두 행 모두** 값이 달라야 한다 — 하나만 다르면
                                #   나머지 한 행은 정답=겨룰값이라 함정이 되지 않는다
                                if w1 and w2 and w1 != v1 and w2 != v2:
                                    겨룸 = (j, w1, w2); break
                        if not 겨룸:
                            continue
                        j, w1, w2 = 겨룸
                        문항.append({
                            "qid": f"g_{doc}_{gi.split('#')[1]}_{re.sub(r'[^0-9A-Za-z가-힣]', '', 열)[:12]}_{len(문항)}",
                            "집합": "G",
                            "document_id": doc,
                            "question": f"표 {gk}에서 {r1}과(와) {r2}의 {열}은(는) 각각 얼마인가?",
                            "gold_ids": [gi],
                            "★ 자매표": j,
                            "★ 표번호": gk,
                            "★ gold 표 제목": g제목,
                            "★ 행쌍": [{"행": r1, "정답": v1, "겨룰값": w1},
                                     {"행": r2, "정답": v2, "겨룰값": w2}],
                            "★ 열": 열,
                            "⛔ 어디서 왔나": "학습 문서 44편의 그래프에서 기계로 만들었다 · make_G_지목문항.py",
                            "⛔ 생성 모델을 안 불렀다": True,
                        })
                        문서수[doc] += 1; 표수[gi] += 1; 만든 += 1
                        break

    print(f"자리(번호+자매표) {자리있음}개 · **만든 문항 {len(문항)}장** · 문서 {len(문서수)}편")
    print(f"   문서당 상위 {문서수.most_common(5)}")
    if not 문항:
        print("⛔ 한 장도 못 만들었다"); return 3
    os.makedirs(os.path.dirname(a.낼곳), exist_ok=True)
    with io.open(a.낼곳, "w", encoding="utf-8") as f:
        for x in 문항:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"★ 냈다 — {os.path.relpath(a.낼곳, ROOT)}")
    print("\n   보기 셋:")
    for x in 문항[:3]:
        print(f"      {x['question'][:80]}")
        print(f"         정답 {[r['정답'] for r in x['★ 행쌍']]} · 겨룰값 {[r['겨룰값'] for r in x['★ 행쌍']]} · gold {x['gold_ids'][0]} ↔ 자매 {x['★ 자매표']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
