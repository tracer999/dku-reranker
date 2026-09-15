# DKU-reranker-v1

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)

[한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | **中文**

这是一个面向韩语 PDF 文档检索增强生成（RAG）的重排序模型（reranker），用于整理送入小型语言模型
（SLM）的证据片段顺序。公开模型 `BAAI/bge-reranker-v2-m3` 的主干权重保持冻结，仅使用韩语公共
机构 PDF 报告的正确证据标注训练了 LoRA 适配器。

**本适配器专门针对韩语。** 它只在韩语文档上训练、只在韩语上验证，因此在其他语言中的表现未知。

小型语言模型在证据位于输入前部时才能准确取值。然而公共机构报告中，相似的表格和数字会在同一份
文档里反复出现。因此即使检索取回了包含答案的片段，只要它没有排在前面，模型就会把前面那张表的
数字当作答案抄下来。本适配器正是针对这一点。

## 为什么做这个

用小型语言模型回答韩语 PDF 文档的问题时，有一类失败格外常见：**答案明明就在输入里，模型却答错
了**。检索已经完成了它的工作，包含答案的片段也确实进入了输入，但模型还是抄了排在前面的另一张
表的数字。

顺着原始日志逐条追查，原因出在**排序**上。公共机构报告的同一份文档中，按年度、按地区的相似表格
会出现很多次。检索与重排要决定其中哪一个排在前面，而公开重排序模型只看一件事：**这个片段与问题
有多相关**。可是讨论同一主题的表格看上去全都"相关"，真正含有答案的往往只有其中一张。仅凭相关度，
无法把那一张挑出来提到前面。

于是我们稍微换了个思路：用人工标注的**哪个片段才是答案依据**，只朝这一个方向对公开模型做少量
追加训练。重新训练整个主干成本高，也容易损失公开模型原有的通用能力，所以主干保持冻结，只叠加
一个很小的 LoRA 适配器。这就是适配器只有 8.9MB 的原因。

它是否真的有用，我们自己做了测量。在相同的问题、相同的候选片段下，仅把重排序模型换成本适配器，
正确证据排在输入首位的条目从 18 增加到 79，正确数也随之从 17 升到 58。详细数字见下方
[主要结果](#主要结果)。为了便于在相同条件下自行确认或验证，我们原样公开了测量时实际使用的
适配器文件。

## 主要结果

使用 Gemma 4 E2B 在 81 条上评测。问题、候选片段与生成设置完全相同，**只更换了重排序模型**。
评测全部基于**韩语**问题与韩语文档。

| 重排序模型 | 正确 | 挽回率 | 正确证据排在首位 |
|---|---|---|---|
| 无（基础 RAG） | 0 / 81 | 0.0% | 1 / 81 |
| `BAAI/bge-reranker-v2-m3`（公开） | 17 / 81 | 21.0% | 18 / 81 |
| **DKU-reranker-v1（本适配器）** | **58 / 81** | **71.6%** | **79 / 81** |

比公开模型多答对 41 条（+50.6 个百分点，McNemar 精确检验双侧 `p = 2.46e-10`）。
完整条件与拆解见[评测](#评测)。

## 模型信息

| 项目 | 取值 |
|---|---|
| 基础模型 | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) —— 主干冻结 |
| 训练部分 | LoRA 适配器（`r=8`、`alpha=16`、`dropout=0.05`，目标模块 `key`、`query`、`value`） |
| 任务类型 | `SEQ_CLS` |
| 输入 | 一组（问题, 片段），最大长度 1024 |
| 输出 | 一个实数分值 —— 仅作为同一问题内部的相对排序使用 |
| 体积 | 8.9MB（`model/adapter_model.safetensors`） |
| 语言 | 韩语 |
| 许可证 | Apache-2.0（基础模型的衍生作品） |

## 面向谁

在韩国考虑引入基于小型语言模型的 RAG 的任何一方，都可以使用。个人开发者、企业、公共机构、
学校与研究机构都在考虑范围之内。

| 使用方 | 适用场景 |
|---|---|
| 个人开发者 | 在用韩语文档搭建 RAG 的阶段，可直接替换现有的公开重排序模型 |
| 企业 | 内部文档与报告问答中难以使用大模型的场合 |
| 公共机构 | 数据不能外传、必须在机构内部自行运行的环境 |
| 学校与研究机构 | 作为韩语 RAG 研究的比较基准或起点 |

适配器仅 8.9MB，如果你已经在用 `bge-reranker-v2-m3`，加一行即可试用。

## 适用范围

### 适合的用途

- 韩语公共机构、政府报告的问答 —— 训练数据正是这一类文档
- 需要从表格中查值作答的任务 —— 训练时就针对相似表格反复出现的情形
- 使用小型（数 B 规模）生成模型的 RAG —— 证据位置影响越大，越能期待效果
- 上下文预算紧张、只能放入前若干片段的环境 —— 把什么放在前面尤为关键

### 范围之外

以下要么未经验证，要么本就不是它的设计用途。

- 非韩语文档。基础模型是多语言的，可以运行，但本适配器只在韩语上训练，其他语言的收益未经验证
- 几乎没有表格的叙述型文档。训练数据以表格为主，此类文档上的效果无法保证
- 替代检索本身。本适配器只重新排序，候选之外的片段无法被拉进来
- 加工片段正文。它不删除、不摘要、不改写，只改变顺序

## 安装

```bash
pip install torch transformers peft
```

| 项目 | 取值 |
|---|---|
| `peft` | 训练时使用 0.20.0。更低版本可能无法读取 `adapter_config.json` 的部分字段 |
| 设备 | CPU 即可运行。若有 GPU，调用 `model.to("cuda")` |
| 下载 | 首次运行会获取基础模型，约 2.2GB |

## 使用方法

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # 本仓库的 model/ 目录
model.eval()


def rerank(question, chunks, batch_size=8, max_length=1024):
    """接收问题与片段列表，按分数由高到低返回。"""
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
prompt = "问题: " + question + "\n\n证据:\n" + "\n\n".join(top_k)
```

如果你已经在用 `bge-reranker-v2-m3`，只需在加载模型后加上 `PeftModel.from_pretrained` 这一行。
调用 `model.unload()` 卸载适配器，即可回到原始的公开模型。

### 使用要点

实际运行时，参考以下几点会更顺利。

| 项目 | 建议这样做 | 原因 |
|---|---|---|
| 批大小 | 请固定为一个值 | 批大小改变会改变批内填充长度，分数会出现细微波动 |
| 最大长度 | 保持 1024 即可 | 这是训练时的取值，改动会连带改变分数的基准 |
| 输入顺序 | 请保持 `(问题, 片段)` | 顺序颠倒会改变分数 |
| 分数的用法 | 请用相对排序而非绝对值 | 绝对值的分布随问题而变，固定阈值会随问题漂移 |
| 片段边界 | 表格整张、小节整节地传入 | 与训练时的片段粒度一致，分数才稳定 |
| 截断（用到前几条） | 请在适配器之外、由调用方决定 | 它只做排序，用多少条不在它的职责之内 |

## 训练数据

韩语公共机构 PDF 报告。每份报告都是从发布机构的官方网站直接下载的。内容包括预算与决算分析、
农业与产业统计、交通调查、信息化议题分析等表格密集的政策报告。每个问题的答案依据片段都由
人工确认并标注。

之所以选择公共机构文档，是因为这类文档使用规范的标准韩语，且文体在机构之间也保持一致，
训练信号不易被文体差异干扰。

| 来源机构 | 训练条数 | 文档数 |
|---|---|---|
| 国会预算政策处（nabo） | 84 | 10 |
| 韩国农村经济研究院（krei） | 82 | 13 |
| 韩国交通研究院（koti） | 49 | 9 |
| 韩国智能信息社会振兴院（nia） | 49 | 12 |
| **合计** | **264** | **44** |

划分按**文档单位**进行，同一文档的片段不会同时出现在训练与验证中。
训练所用的问题与原始 PDF 未包含在本仓库中。

## 训练

```bash
python src/build_c2_trainset.py      # 构建训练集
python src/make_doc_folds_v2.py      # 按文档划分
python src/train_c2_targeted.py      # LoRA 训练
```

| 超参数 | 取值 |
|---|---|
| epoch | 5 |
| 学习率 | 0.0001 |
| seed | 20260915 |
| 最大长度 | 1024 |
| step | 980 |
| 损失 | 提升正确证据分数的项 ＋ 使正确证据排在非证据之前的项（权重 1 : 1.0） |
| 训练条数 | 264（分数分离 98・成对排序违例 166） |
| 设备 | NVIDIA L4（AWS g6.xlarge）・82.9 分钟 |

采用哪个 epoch 的适配器，是**在训练开始前、一个 step 都还没跑时**就以规则固定的：
① 排除排序违例率高于训练前的 epoch ② 在其余中取证据分离指标最高者 ③ 相同则取更早的 epoch。
按此规则选中了 epoch 5。

| 指标（训练集） | 训练前 | epoch 5 |
|---|---|---|
| 所有正确证据都排在非证据之前的条目比例 | 0.4796 | **0.7959** |
| 排序违例率（越低越好） | 0.2866 | **0.1311** |

这两个指标是在训练集上测得的拟合程度，并非下游任务性能，仅用于挑选适配器。
完整记录见 [`docs/train_record.json`](docs/train_record.json)。

## 评测

### 设置

| 项目 | 取值 |
|---|---|
| 生成模型 | `google/gemma-4-E2B-it`（有效 2.3B・GGUF Q8_0） |
| 运行 | `llama-server`・上下文 10240・`temperature 0`・GPU 层 99・线程 4 |
| 机器 | AWS g6.xlarge（NVIDIA L4） |
| 条数 | 81 |
| 控制 | 问题、候选片段与生成设置一致，仅更换重排序模型 |

为使比较公平，除重排序模型之外，三个条件接收的输入完全相同：问题相同、候选片段相同、生成设置
也相同。唯一变化的只有重排序模型，因此正确数的差异可以归因于它。

评测条目的选取标准是：在不使用重排序的基础 RAG 输入下 E2B 答错，且**错误取值来自同一输入中的
另一个片段**。也就是答案本在输入中，模型却读了别的片段。按此标准基线为 0 / 81，各模型的正确数
即相对基线的净增。

### 结果

见上方[主要结果](#主要结果)。提升的原因在于正确证据排在输入首位的条目从 18 增加到 79。
把同样的 81 条按排名是否变化分组，会看得更清楚。

| 分组 | 条数 | 公开 BGE | DKU |
|---|---|---|---|
| 正确证据排名发生变化的条目 | 61 | 6 / 61（10%） | **47 / 61（77%）** |
| 正确证据排名未变化的条目 | 18 | 11 / 18（61%） | 11 / 18（61%） |

在排名未变化的条目上两个模型完全一致。提升来自排名的变化，而非题目更容易。

### 未能确认的部分

| 项目 | 结果 |
|---|---|
| 仅保留前 28 条的筛选 | 在这 81 条上未观察到差异（`p = 1`） |
| 将问题置于证据之前 | 在这 81 条上未观察到差异（`p = 1`） |
| 其他生成模型 | 未测量 |
| 其他语言与文档类型 | 未测量 |

已确认的效果只有重排序模型的训练这一项。

## 局限

- 使用来自韩国四家机构、44 份报告的 264 条数据训练
- 仅用一个生成模型（E2B）、81 条数据验证
- 评测集只收集基础 RAG 答错的条目，并非一般问题分布
- 只处理"答案在输入中却读了别的片段"这一种失败，无法弥补检索未召回的情况
- 比较对象只有公开的 `bge-reranker-v2-m3` 一个

## 仓库结构

```
model/adapter_model.safetensors   LoRA 适配器（8.9MB）
model/adapter_config.json         适配器配置
src/build_c2_trainset.py          训练集构建
src/make_doc_folds_v2.py          按文档划分
src/train_c2_targeted.py          LoRA 训练
docs/train_record.json            训练记录（配置・损失・epoch 选择）
```

## 引用

> 梁成勋, "DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub 仓库, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker_v1,
  title  = {DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  school = {Dankook University, Dept. of IT Convergence},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```

## 许可证

本适配器是 `BAAI/bge-reranker-v2-m3` 的衍生作品，采用与原作相同的 Apache License 2.0。
同时须遵守原模型的许可证与署名要求。全文见 [LICENSE](LICENSE)。

## 所属

在檀国大学 **信息融合技术·创业研究生院 IT 融合学科**（정보융합기술·창업대학원
IT컨버전스학과）训练并公开。
研究生院 <https://cms.dankook.ac.kr/web/gict> · 学科 <https://cms.dankook.ac.kr/web/gict/it-2>

## 联系方式

如有问题或建议，请通过 [Issues](https://github.com/tracer999/dku-reranker/issues) 或
<sh.yang@dankook.ac.kr> 与我们联系。
