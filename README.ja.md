# DKU-reranker-v1

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)
[![Language](https://img.shields.io/badge/language-Korean--only-red.svg)](#)

[한국어](README.md) | [English](README.en.md) | **日本語** | [中文](README.zh.md)

**公開リランカー `BAAI/bge-reranker-v2-m3` を、韓国語の公的文書の文体に特化させたものです。**
韓国語 PDF 文書を対象とする検索拡張生成（RAG）において、小規模言語モデル（SLM）へ渡す根拠
チャンクの順位を整えます。

ベースモデルは多言語モデルで幅は広いものの、個々の言語の文書がどのような文体で書かれるかまでは
扱いません。本アダプタは本体重みを凍結したまま、韓国語の公的機関 PDF 報告書で人が付けた正解根拠
ラベルのみで LoRA アダプタを学習しました。**韓国語専用であり、他言語での挙動は検証していません。**

## なぜ作ったのか

SLM ベースの RAG で繰り返し現れる失敗を一つ狙っています。**正解の根拠が入力に含まれているのに、
モデルが別のチャンクの値を答えとして出してしまう**ケースです。

原因は順位にあります。公的機関の報告書では、年度別・地域別に同じ書式の表が繰り返されます。公開
リランカーは質問との関連度しか評価しないため、同じ主題の表がすべて高い点を取り、その中で正解を
含む一つを切り分けられません。

そこで関連度の代わりに、**そのチャンクが正解の根拠かどうか**を学習信号としました。本体は凍結し
LoRA アダプタのみを学習することで、公開モデルの汎用的な性能と配布コストをそのまま保っています。

効果は実測しました。リランカーだけを差し替えたところ、正解根拠が 1 位の設問が 18 → 79 に、
正答数が 17 → 58 に増えました。測定に用いたアダプタファイルをそのまま公開します。

## 主な結果

Gemma 4 E2B で 81 問を評価しました。設問・候補チャンク・生成設定はすべて同一で、**リランカー
だけ**を入れ替えています。評価は**すべて韓国語の設問と韓国語文書**を対象に行いました。

| リランカー | 正答 | 回復率 | 正解根拠が 1 位 |
|---|---|---|---|
| なし（基本 RAG） | 0 / 81 | 0.0% | 1 / 81 |
| `BAAI/bge-reranker-v2-m3`（公開） | 17 / 81 | 21.0% | 18 / 81 |
| **DKU-reranker-v1（本アダプタ）** | **58 / 81** | **71.6%** | **79 / 81** |

公開モデルより 41 問多く正答しました（+50.6 ポイント・McNemar 正確検定 両側 `p = 2.46e-10`）。
詳しい条件と内訳は[評価](#評価)にあります。

## モデル情報

| 項目 | 値 |
|---|---|
| ベースモデル | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) — 本体凍結 |
| 学習部位 | LoRA アダプタ（`r=8`, `alpha=16`, `dropout=0.05`、対象モジュール `key`・`query`・`value`） |
| タスク種別 | `SEQ_CLS` |
| 入力 | （質問, チャンク）1 組・最大長 1024 |
| 出力 | 実数スコア 1 つ — 同一質問内の相対順位としてのみ使います |
| サイズ | 8.9MB（`model/adapter_model.safetensors`） |
| 言語 | 韓国語 |
| ライセンス | Apache-2.0（ベースモデルの派生物） |

## 誰のためのものか

韓国で小規模言語モデルによる RAG の導入を考えているところであれば、どこでも使えるように公開
します。個人開発者、企業、公的機関、学校・研究機関のいずれも想定して作りました。

| 使う側 | どのような場面で |
|---|---|
| 個人開発者 | 韓国語文書で RAG を作ってみる段階で、公開リランカーをそのまま差し替えられます |
| 企業 | 社内文書・報告書の質問応答に大規模モデルを使いにくい環境で |
| 公的機関 | 資料を組織の外へ出さず、内部で運用する必要がある環境で |
| 学校・研究機関 | 韓国語 RAG 研究の比較基準や出発点として |

アダプタは 8.9MB と小さいため、すでに `bge-reranker-v2-m3` をお使いであれば 1 行加えるだけで
試していただけます。

## 用途

### 適した用途

- **標準的な韓国語で書かれた文書**に対する質問応答 — 学習データが標準語を正確に用いた文書です
- 表から値を探して答えるタスク — 似た表が何度も現れる状況を狙って学習しています
- 小規模（数 B 規模）生成モデルを使う RAG — 根拠の位置の影響が大きい環境ほど効果が期待できます
- コンテキスト予算が限られ、上位いくつかしか入れられない環境 — 何を前に置くかが特に重要になります

### 範囲外の用途

以下は、検証されていないか、そもそも想定していない使い方です。

- 韓国語以外の文書。ベースが多言語なので動作はしますが、本アダプタは韓国語のみで学習しており、他言語での効果は確認されていません
- 表がほとんどない記述中心の文書。学習データが表中心のため、そうした文書での効果はお約束できません
- 検索そのものの代替。本アダプタは順位を付け直すだけで、そもそも候補に無いチャンクを引き寄せることはできません
- チャンク本文の加工。削除・要約・書き換えは行わず、順序だけを変えます

## 導入

```bash
pip install torch transformers peft
```

| 項目 | 値 |
|---|---|
| `peft` | 学習時のバージョンは 0.20.0 です。これより古いと `adapter_config.json` の一部項目を読めない場合があります |
| デバイス | CPU でも動作します。GPU がある場合は `model.to("cuda")` を呼びます |
| ダウンロード | 初回実行時にベースモデル約 2.2GB を取得します |

## 使い方

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # 本リポジトリの model/
model.eval()


def rerank(question, chunks, batch_size=8, max_length=1024):
    """質問とチャンク一覧を受け取り、スコアの高い順に返します。"""
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
prompt = "質問: " + question + "\n\n根拠:\n" + "\n\n".join(top_k)
```

すでに `bge-reranker-v2-m3` をお使いであれば、モデルを読み込んだあとに
`PeftModel.from_pretrained` の 1 行を加えるだけで済みます。`model.unload()` でアダプタを外せば、
元の公開モデルに戻ります。

### 使用時のポイント

実際に動かすときは、以下を参考にしていただけると安心です。

| 項目 | こうするとよいです | 理由 |
|---|---|---|
| バッチサイズ | 一つに固定してください | バッチが変わると内部のパディング長が変わり、スコアがわずかに揺れます |
| 最大長 | 1024 のままで大丈夫です | 学習時の値なので、変えるとスコアの基準も一緒に変わります |
| 入力の順序 | `(質問, チャンク)` の順を守ってください | 逆にするとスコアが変わります |
| スコアの使い方 | 絶対値ではなく相対順位としてお使いください | 絶対値は質問ごとに分布が異なり、固定しきい値は質問ごとにずれてしまいます |
| チャンクの境界 | 表は表単位、節は節単位で渡してください | 学習時のチャンク単位と揃っているほどスコアが安定します |
| 打ち切り（上位何件まで使うか） | 本アダプタの外、利用側で決めてください | 本アダプタは並べ替えるだけで、何件使うかには関与しません |

## 学習データ

韓国語の公的機関 PDF 報告書です。各機関が公開している報告書を、その機関の公式サイトから直接
取得しました。予算・決算分析、農業・産業統計、交通調査、情報化イシュー分析など、表の多い政策
報告書です。質問ごとに、答えの根拠となるチャンクを人が確認して付与しました。

公的機関の文書を選んだのは、**標準的な韓国語を正確に用い、文体が機関をまたいでも一定している**
ためです。文体が揺れると、その揺れが学習信号にも混じってしまいます。

| 出典機関 | 学習件数 | 文書数 |
|---|---|---|
| 国会予算政策処（nabo） | 84 | 10 |
| 韓国農村経済研究院（krei） | 82 | 13 |
| 韓国交通研究院（koti） | 49 | 9 |
| 韓国知能情報社会振興院（nia） | 49 | 12 |
| **合計** | **264** | **44** |

分割は**文書単位**です。同一文書のチャンクが学習と検証に同時に入ることはありません。
学習に用いた設問と原文 PDF は本リポジトリには含めていません。

## 学習

```bash
python src/build_c2_trainset.py      # 学習セット構築
python src/make_doc_folds_v2.py      # 文書単位の分割
python src/train_c2_targeted.py      # LoRA 学習
```

| ハイパーパラメータ | 値 |
|---|---|
| epoch | 5 |
| 学習率 | 0.0001 |
| seed | 20260915 |
| 最大長 | 1024 |
| step | 980 |
| 損失 | 正解根拠のスコアを上げる項 ＋ 正解根拠を非正解根拠より上位にする項（重み 1 : 1.0） |
| 学習件数 | 264（スコア分離用 98・順位対違反用 166） |
| デバイス | NVIDIA L4（AWS g6.xlarge）・82.9 分 |

どの epoch のアダプタを採用するかは、**学習を 1 step も回す前に**規則として固定しました。
① 順位違反率が学習前を上回る epoch は除外 ② 残りのうち正解根拠の分離指標が最も高いもの
③ 同値ならより早い epoch。規則に従い epoch 5 を選びました。

| 指標（学習セット） | 学習前 | epoch 5 |
|---|---|---|
| 正解根拠がすべて非正解根拠より上位である設問の割合 | 0.4796 | **0.7959** |
| 順位違反率（低いほど良い） | 0.2866 | **0.1311** |

この 2 指標は学習セット上で測った適合度であり、下流タスクの性能ではありません。アダプタの
選択にのみ使いました。全記録は [`docs/train_record.json`](docs/train_record.json) にあります。

## 評価

### 設定

| 項目 | 値 |
|---|---|
| 生成モデル | `google/gemma-4-E2B-it`（実効 2.3B・GGUF Q8_0） |
| 実行 | `llama-server`・コンテキスト 10240・`temperature 0`・GPU 層 99・スレッド 4 |
| マシン | AWS g6.xlarge（NVIDIA L4） |
| 設問数 | 81 |
| 統制 | 設問・候補チャンク・生成設定は同一。リランカーのみ入れ替え |

比較を公平にするため、リランカーを除けば三つの条件が完全に同じ入力を受け取るようにしました。
設問も同じ、候補チャンクも同じ、生成設定も同じです。変わるのはリランカー一つだけなので、
正答数の差はリランカーが生んだ差と見なせます。

評価設問は、リランカーなしの基本 RAG 入力で E2B が誤答し、かつ**その誤った値が同じ入力の中の
別のチャンク由来**である設問のみを集めたものです。答えは入力の中にあったのに、モデルが別の
チャンクを読んだ場合です。この基準によりベースラインは 0 / 81 となり、各モデルの正答数が
そのままベースラインに対する純増になります。

### 結果

上の[主な結果](#主な結果)の表のとおりです。上がった理由は、正解根拠を入力の 1 位に置けた設問が
18 → 79 に増えたことにあります。同じ 81 問を順位が変わったかどうかで分けると、さらに明確です。

| グループ | 設問数 | 公開 BGE | DKU |
|---|---|---|---|
| 正解根拠の順位が変わった設問 | 61 | 6 / 61（10%） | **47 / 61（77%）** |
| 正解根拠の順位が変わらなかった設問 | 18 | 11 / 18（61%） | 11 / 18（61%） |

順位が変わらなかった設問では、両モデルは同じです。性能の向上は設問の難易度ではなく、順位の
変化から来ています。

### 確認できなかったこと

| 項目 | 結果 |
|---|---|
| 上位 28 件の選別 | この 81 問では差を確認できませんでした（`p = 1`） |
| 質問を根拠より前に置く配置 | この 81 問では差を確認できませんでした（`p = 1`） |
| 他の生成モデル | 測定していません |
| 他言語・他の文書種別 | 測定していません |

確認できた効果は、リランカーの学習ひとつです。

## リポジトリ構成

```
model/adapter_model.safetensors   LoRA アダプタ（8.9MB）
model/adapter_config.json         アダプタ設定
src/build_c2_trainset.py          学習セット構築
src/make_doc_folds_v2.py          文書単位の分割
src/train_c2_targeted.py          LoRA 学習
docs/train_record.json            学習記録（設定・損失・epoch 選択）
```

## 引用

> 梁成勲, "DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub リポジトリ, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  school = {Dankook University, Graduate School of Information Convergence Technology and Entrepreneurship},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```

## ライセンス

`BAAI/bge-reranker-v2-m3` の派生物であり、原本と同じ Apache License 2.0 に従います。原モデルの
ライセンスおよび著作者表示の要件も併せて遵守してください。全文は [LICENSE](LICENSE) にあります。

## 所属

檀国大学校 **情報融合技術・創業大学院**（정보융합기술·창업대학원）にて学習・公開しました。

| 項目 | 内容 |
|---|---|
| 大学 | 檀国大学校 · <https://www.dankook.ac.kr> |
| 大学院 | 情報融合技術・創業大学院 · <https://cms.dankook.ac.kr/web/gict> |

## お問い合わせ

ご質問やご提案は [Issues](https://github.com/tracer999/dku-reranker/issues) または
<sh.yang@dankook.ac.kr> までお寄せください。
