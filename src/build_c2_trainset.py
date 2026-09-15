r"""C′ 표적 학습 자료를 만든다 — **D+증설은 gold 손실 · F 는 A 순서 보존**.

```
    python3 code/train/build_c2_trainset.py \
        --조건 results/v92/조건_재진단488A_학습262.jsonl \
        --벡터 results/v92/재진단488_판정벡터_ABC_262.jsonl \
        --세층 results/v92/재진단488A_세층_262.json \
        --oof results/l4_run/out_canonical/oof_원자료_14640.jsonl \
        --out results/v92/C2학습셋_262.jsonl
```

## ⛔⛔ 두 집합은 **손실이 다르다** (17 문서 3.2)

```
**D + 증설**  →  `loss:"gold"`   gold > non-gold relevance
              「이 근거가 답을 받친다」
**F**        →  `loss:"order"`  **A 자연 순서 보존**
              A 에서 i 가 j 보다 앞이면 **s_i > s_j** ⛔ **gold 를 올리라고 하지 않는다**
```

★★ 왜 F 를 gold 손실에 안 넣나 — **어젯밤 진단이 말한다**
```
손상 21 중 **9 건이 「gold 가 전부 앞으로 갔는데도 틀렸다」**
⇒ ⇒ **gold 를 올리는 것 자체가 손상을 만든다**
⇒ ⇒ ⇒ F 에 그것을 가르치면 **B 가 한 것을 그대로 따라 한다**
```

## ⛔ 이 도구가 하지 않는 것

```
⛔ **가중치를 문항마다 주지 않는다** — 「이 8개를 맞히게」가 아니다.
  ⇒ 사후에 고른 문항을 성공이라 부르는 자리가 된다
⛔ 새 문항을 만들지 않는다 — 증설은 제작자가 만들어 따로 온다
⛔ **1024 토큰을 넘는 gold 가 있는 문항은 뺀다** — 잘린 채 「정답 근거다」로 가르치면 틀린 신호다
```
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAXLEN = 1024


def main() -> int:
    ap = argparse.ArgumentParser(description="C′ 표적 학습 자료")
    ap.add_argument("--조건", required=True)
    ap.add_argument("--벡터", required=True)
    ap.add_argument("--세층", required=True)
    ap.add_argument("--oof", required=True, help="`fold` 칸")
    ap.add_argument("--제외", default="", help="한 줄에 qid 하나 — ⛔ **결함이 확인된 문항**을 학습에서 뺀다. "
                    "★ 평가 분모에서는 빼지 않는다(사후 부분집합이 된다). 학습만 뺀다")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    조건 = {r["qid"]: r for r in (json.loads(l) for l in open(a.조건, encoding="utf-8") if l.strip())}
    V = {r["qid"]: r for r in (json.loads(l) for l in open(a.벡터, encoding="utf-8") if l.strip())}
    층4 = set(json.loads(Path(a.세층).read_text(encoding="utf-8"))["qid 목록"]["④ gold 전부 검색·입력"])
    fold = {}
    for l in open(a.oof, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("fold") is not None:
                fold[r["qid"]] = r["fold"]

    제외 = set()
    if a.제외:
        제외 = {l.strip() for l in open(a.제외, encoding="utf-8") if l.strip()}
        모르는 = 제외 - set(V)
        if 모르는:
            raise SystemExit(f"⛔ 제외 목록에 판정벡터에 없는 qid: {sorted(모르는)}")
        print(f"⛔ 학습에서 뺀다(결함 확인) {len(제외)}개 — 평가 분모에서는 빼지 않는다", flush=True)
    D = sorted(q for q in 층4 if q not in 제외 and V[q]["A_판정"] != "○" and V[q]["oracle_판정"] == "○")
    F = sorted(q for q in 층4 if q not in 제외 and V[q]["A_판정"] == "○")
    겹 = set(D) & set(F)
    if 겹:
        raise SystemExit(f"⛔⛔ D 와 F 가 겹친다 {len(겹)}개 — **멈춘다**")
    print(f"D {len(D)} · F {len(F)} · 층④ {len(층4)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)

    줄, 뺀것, 통계 = [], [], Counter()
    for 집합, 이름, loss in ((D, "D", "gold"), (F, "F", "order")):
        for q in 집합:
            r = 조건[q]
            ids, txts = list(r["chunk_ids"]), list(r["evidence_texts"])
            gold = [c for c in ids if c in set(r["gold_evidence_ids"])]
            # ⛔ gold 가 1024 를 넘으면 **그 문항을 뺀다** — 잘린 채 가르치면 틀린 신호다
            긴gold = []
            for c in gold:
                t = len(tok(r["question"], txts[ids.index(c)], truncation=False)["input_ids"])
                if t > MAXLEN:
                    긴gold.append({"chunk_id": c, "토큰": t})
            if 긴gold:
                뺀것.append({"qid": q, "집합": 이름, "까닭": "gold 가 1024 초과", "조각": 긴gold})
                통계[f"⛔ {이름} 에서 뺌"] += 1
                continue
            줄.append({
              "qid": q, "fold": fold[q], "집합": 이름, "loss": loss,
              "question": r["question"], "document_id": r["doc_id"],
              "chunk_ids": ids, "chunk_texts": txts,
              "gold_ids": gold,
              # ★ F 의 목표는 **A 의 차례 그대로**다. 그 차례를 그대로 실어 둔다
              "A_순서": ids,
              "⛔ 무엇을 가르치나": ("gold > non-gold — 「이 근거가 답을 받친다」" if loss == "gold"
                            else "**A 의 차례를 지켜라** — ⛔ gold 를 올리라고 하지 않는다"),
            })
            통계[이름] += 1

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".partial")
    t.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in 줄), encoding="utf-8")
    t.replace(p)
    실제 = sum(1 for l in p.open(encoding="utf-8") if l.strip())

    메타 = {
      "무엇": "C′ 표적 학습 자료 — **D 는 gold 손실 · F 는 A 순서 보존 손실**",
      "언제": datetime.datetime.now().isoformat(timespec="seconds"),
      "★★ 두 집합": {
        "D (gold 손실)": {"수": 통계["D"], "정의": "층④ ∧ A 오답 ∧ oracle ○ — **구제 표적**"},
        "F (A 순서 보존)": {"수": 통계["F"], "정의": "층④ ∧ A 정답 — **보존 표적**",
                     "⛔ gold 를 올리지 않는다": "F 는 **A 순서로 이미 맞은 문항**이다. "
                        "★ 어젯밤 진단 — **손상 21 중 9 건이 「gold 가 전부 앞으로 갔는데도 틀렸다」**"},
      },
      "⛔ 뺀 것": {"수": len(뺀것), "목록": 뺀것,
              "까닭": "**gold 가 1,024 토큰을 넘으면 잘린 채 「정답 근거다」로 가르치게 된다**"},
      "★ 줄 수": 실제,
      "fold 분포": dict(sorted(Counter(x["fold"] for x in 줄).items())),
      "집합 × fold": {f"{s}-fold{k}": sum(1 for x in 줄 if x["집합"] == s and x["fold"] == k)
                   for s in ("D", "F") for k in sorted({x["fold"] for x in 줄})},
      "⛔ 가중치를 주지 않았다": "alex 가 지목한 8 qid(새 손상 6 · 놓친 D 2)는 "
                    "**이미 F·D 안에 전부 있다**(leo 가 확인했다). "
                    "⇒ ★ 따로 넣거나 가중을 주지 않았다 — **사후에 고른 문항을 성공이라 부르는 자리**가 된다",
      "⛔ 결함으로 학습에서 뺀 것": sorted(제외),
      "⛔ 증설은 아직 없다": "제작자 셋이 만드는 중이다. 오면 **D 쪽에 합친다**",
      "산출물 sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
    }
    mp = p.with_name(p.stem + "_메타.json")
    mp.write_text(json.dumps(메타, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(메타, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
