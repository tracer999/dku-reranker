# DKU-reranker

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base-bge--reranker--v2--m3-orange.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![Adapter](https://img.shields.io/badge/LoRA-8.9MB-green.svg)](model/)
[![Language](https://img.shields.io/badge/language-Korean--only-red.svg)](#)

[한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | **中文**

**这是把公开重排序模型 `BAAI/bge-reranker-v2-m3` 针对韩语公共文档文体做了专门适配的版本。**
在面向韩语 PDF 文档的检索增强生成（RAG）中，它负责整理送入小型语言模型（SLM）的证据片段顺序。

基础模型是多语言模型，覆盖面广，但并不深入到某一种语言的文档以怎样的文体写成。本适配器在保持
主干权重冻结的前提下，仅用韩语公共机构 PDF 报告中由人工标注的正确证据标签训练 LoRA 适配器。
**仅限韩语，在其他语言中的表现未经验证。**

## 为什么做这个

它针对 SLM 驱动的 RAG 中反复出现的一类失败：**正确证据就在输入里，模型却给出了另一个片段中的值。**

原因在于排序。公共机构报告中，按年度、按地区排版相同的表格会反复出现。公开重排序模型只评估与
问题的相关度，因此同一主题的表格都会拿到高分，其中真正含有答案的那一张无法被区分出来。

因此我们改用**该片段是否为正确证据**作为训练信号。主干保持冻结，仅训练 LoRA 适配器，从而保留
公开模型的通用性能，也不增加部署成本。

效果经过实测。仅更换重排序模型后，正确证据排在首位的条目从 18 增加到 79，正确数从 17 升到 58。
测量时实际使用的适配器文件原样公开。

## 主要结果

在**未用于训练的文档**上评测。我们另外收集了与训练所用 44 篇毫无重叠的 46 篇新报告，并从中取
80 道题目进行测量。题目、候选片段与生成设置完全相同，**只更换重排模型**。生成模型为 Gemma 4 E2B。

| 重排模型 | 正确 | 正确依据排第 1 位 | 排在正确依据之前的其他表格 |
|---|---|---|---|
| 无（原始 RAG） | 0 / 80 | — | — |
| `BAAI/bge-reranker-v2-m3`（公开） | 69 / 80 | 64 / 80 | 4 |
| 旧版 | 60 / 80 | 27 / 80 | 22 |
| **DKU-reranker（本适配器）** | **71 / 80** | **65 / 80** | **3** |

比旧版**多答对 11 条**（McNemar 精确检验双侧 `p = 0.0034`）。原因清楚可见——排在正确依据之前
还有其他表格的题目从 **22 降至 3**。

★ 这是**在训练中从未见过的文档上**测得的数值。在训练文档上测量会得到更高的值，但那无法预测在
新文档上的表现。

⛔ 与公开模型的差距（69 → 71）在此样本量下**不具统计显著性**（`p = 0.754`）。我们**不**声称
优于公开模型。本表所显示的是*训练能够纠正排序*，其幅度相对旧版以 `p = 0.0034` 得到确认。

⛔ 数值随评测条件而变化。设计见 [评测](#评测)。

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

- 针对**以规范标准韩语撰写的文档**的问答 —— 训练数据正是使用规范标准韩语的文档
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
| **合计** | **44** |

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
| step | 1,870 |
| 损失 | 提升正确证据分数的项 ＋ 使正确证据排在非证据之前的项（权重 1 : 1.0） |
| 训练条数 | 353（依据分离 98・顺序保持 166・表格指定 89） |
| 设备 | NVIDIA L4（AWS g6.xlarge）・82.9 分钟 |

采用哪个 epoch 的适配器，是**在训练开始前、一个 step 都还没跑时**就以规则固定的：
① 排除排序违例率高于训练前的 epoch ② 在其余中取证据分离指标最高者 ③ 相同则取更早的 epoch。
按此规则选中了 epoch 5。

| 指标（训练集） | 训练前 | epoch 5 |
|---|---|---|
| 所有正确证据都排在非证据之前的条目比例 | 0.6738 | **0.8877** |
| 排序违例率（越低越好） | 0.2866 | **0.082** |

这两个指标是在训练集上测得的拟合程度，并非下游任务性能，仅用于挑选适配器。
完整记录见 [`docs/train_record.json`](docs/train_record.json)。

## 评测

### 如何测量

| 项目 | 值 |
|---|---|
| 生成模型 | `google/gemma-4-E2B-it`（有效 2.3B・GGUF Q8_0） |
| 运行 | `llama-server`・上下文 10240・`temperature 0`・GPU 层 99・线程 4 |
| 控制 | 题目、候选片段与生成设置相同。**仅更换重排模型** |

要让比较公平，除重排模型外的一切都必须保持一致。题目相同、候选片段相同、生成设置相同，才能把
正确数的差异归因于重排模型。

### 用哪些题目测量

本适配器针对的失败很窄：**正确依据就在输入中，模型却从同一输入的另一个片段取值**。评测题目也应
按这个条件收集。

1. 取生成模型在无重排的原始 RAG 输入上**答错**的题目。
2. 确认那个错误值**确实存在于同一输入的另一个片段中**。
3. 剔除模型凭空编造数值的情况——无法归因于片段排序。

这样收集后基准线为 0，各模型答对的数量即为其净增。

### 还应一并观察什么

只看正确数无法说明为何提升。请一并测量以下两项。

| 什么 | 为什么 |
|---|---|
| **正确依据排在输入第 1 位的题目数** | 这是适配器直接改变的量 |
| **排在正确依据之前的其他表格数** | 失败发生的位置。它下降，正确数才会上升 |

另外请把题目分为**排序发生变化**与**排序未变**两组。未变的题目上两个模型应当一致；若在那里也
出现差异，说明混入了排序以外的因素。

> ⛔ **上面[主要结果](#主要结果)中的数值是一组测量。** 文档类型、题目措辞与候选片段构成会让数值
> 变化。也请按上述设计在您自己的文档上测量。

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

> 梁成勋, "DKU-reranker: A LoRA Adapter for Korean Public-Document Evidence Reranking,"
> GitHub 仓库, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

```bibtex
@misc{dku_reranker,
  title  = {DKU-reranker: A LoRA Adapter for Korean Public-Document Evidence Reranking},
  author = {Yang, Seong Hun},
  year   = {2026},
  school = {Dankook University, Graduate School of Information Convergence Technology and Entrepreneurship},
  note   = {LoRA adapter on BAAI/bge-reranker-v2-m3},
  url    = {https://github.com/tracer999/dku-reranker}
}
```

## 许可证

本适配器是 `BAAI/bge-reranker-v2-m3` 的衍生作品，采用与原作相同的 Apache License 2.0。
同时须遵守原模型的许可证与署名要求。全文见 [LICENSE](LICENSE)。

## 所属

在檀国大学 **信息融合技术·创业研究生院**（정보융합기술·창업대학원）训练并公开。

| 项目 | 内容 |
|---|---|
| 大学 | 檀国大学 · <https://www.dankook.ac.kr> |
| 研究生院 | 信息融合技术·创业研究生院 · <https://cms.dankook.ac.kr/web/gict> |

## 联系方式

如有问题或建议，请通过 [Issues](https://github.com/tracer999/dku-reranker/issues) 或
<sh.yang@dankook.ac.kr> 与我们联系。
