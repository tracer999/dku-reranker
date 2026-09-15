# DKU-reranker-v1

[한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | **中文**

**这是一个面向韩语 PDF 文档检索增强生成（RAG）的重排序模型（reranker）。**
它重新排列送入小型语言模型（SLM）的证据片段，使包含答案的片段排在输入的最前面。

公开模型 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) 的
主干权重保持**冻结**，仅使用韩语公共机构 PDF 报告的**正确证据标注**训练 LoRA 适配器。
适配器文件为 8.9MB，基础模型由使用方从其公开仓库自行下载。

| 项目 | 内容 |
|---|---|
| 使用位置 | RAG 流程中的**检索之后、生成之前** |
| 作用 | 对（问题, 片段）配对打分，**重新排序** |
| 基础模型 | `BAAI/bge-reranker-v2-m3`（主干冻结，仅训练 LoRA 适配器） |
| 训练数据 | 韩语公共机构 PDF 报告 264 条，来自四家机构 |
| 评测 | 使用 `google/gemma-4-E2B-it` 在 81 条上评测 —— 公开 BGE 为 17，本适配器为 **58** |
| 体积 | 适配器 8.9MB（`r=8`） |
| 许可证 | Apache-2.0 |

`DKU` 指檀国大学（Dankook University）。

---

## 1. 它做什么

在检索返回的片段（chunk）中，**把真正包含答案依据的片段提到最前面。**

```
问题 + [片段1, 片段2, …, 片段N]
        ↓  为每个（问题, 片段）配对打一个分
        ↓  按分数从高到低重新排序
问题 + [正确证据, …, …]      ← 按这个顺序送入生成模型
```

小型语言模型在证据位于输入前部时才能准确取值。因此**放什么固然重要，按什么顺序放同样重要。**

## 2. 为什么需要它

韩语公共机构报告表格密集，同一份文档中会**多次出现外形相似的表格和数字**，
例如预算文件的分年度表、统计报告的分地区表。

检索会返回多个相关片段。但如果**包含答案的片段没有排在前面**，小型模型就会把排在前面的
另一张表的数字照抄为答案。答案其实**就在输入里**，模型却答错了。本适配器正是针对这一点。

通用重排序模型衡量的是**片段与问题的相关程度**。本适配器在此基础上，使用人工标注的
**哪个片段才是答案依据**继续训练。两者相似但不同：讨论同一主题的多个片段都"相关"，
而真正含有答案的通常只有一个。

## 3. 它不是什么

| 不是 | 说明 |
|---|---|
| 不是生成模型 | 它不产生答案，只改变片段顺序 |
| 不是从零训练的重排序模型 | 公开 BGE 主干**冻结**，仅训练 LoRA 适配器 |
| 不是检索器 | 只对已有候选重新排序，不会去获取新文档 |
| 不是上下文压缩器 | 不删除、不摘要、不改写片段正文 |
| 不是通用多语言模型 | 仅在韩语公共机构报告上训练 |

## 4. 目的与适用范围

它的目标是**让韩语 PDF 文档能够被小型语言模型有效使用**。

### 4.1 适合的场景

| 场景 | 原因 |
|---|---|
| 韩语公共机构、政府报告的问答 | 与训练数据的文档性质一致 |
| 需要从表格中查值作答的任务 | 训练数据以表格为主 |
| 使用小型（数 B 规模）生成模型的 RAG | 证据位置的影响在此最为明显 |
| 上下文预算紧张的环境 | 只能放入少数片段时，保留哪些就变得关键 |

### 4.2 不适合的场景

| 场景 | 原因 |
|---|---|
| 英语等其他语言的文档 | 基础模型是多语言的，可以运行，但本适配器只见过韩语 |
| 几乎没有表格的叙述型文档 | 训练数据以表格为主 |
| 用来替代检索本身 | 候选之外的片段无法被提升 |
| 需要修改片段内部内容的任务 | 本适配器不改动片段正文 |

## 5. 安装与使用

### 5.1 前置条件

```bash
pip install torch transformers peft
```

| 项目 | 取值 |
|---|---|
| `peft` | 训练时使用 **0.20.0**。更低版本可能无法读取 `adapter_config.json` 的部分字段 |
| 设备 | CPU 即可运行。若有 GPU，调用 `model.to("cuda")` |
| 基础模型 | 首次运行会下载 `BAAI/bge-reranker-v2-m3`，约 2.2GB |

### 5.2 挂载适配器

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "BAAI/bge-reranker-v2-m3"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # 指向本仓库的 model/ 目录
model.eval()
```

### 5.3 打分与重排

```python
def rerank(question, chunks, batch_size=8, max_length=1024):
    # 接收问题与片段列表，按分数由高到低返回
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
```

### 5.4 接入 RAG 流程

```python
ranked = rerank(question, retrieved_chunks)
top_k  = [text for text, _ in ranked[:28]]
prompt = "问题: " + question + "\n\n证据:\n" + "\n\n".join(top_k)
```

### 5.5 如果你已经在用 BGE

**在加载模型之后只需增加一行。**

```python
model = AutoModelForSequenceClassification.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, "./model")   # ← 只加这一行

model = model.unload()      # 卸载适配器即可回到原始公开模型
```

### 5.6 使用注意事项

| 注意事项 | 原因 |
|---|---|
| **固定批大小** | 批大小改变会改变批内的填充长度，分数可能出现细微波动。为便于复现请固定为一个值 |
| **max_length 取 1024** | 这是训练时使用的取值。大幅偏离会改变与训练一致的条件 |
| **问题与片段的顺序** | 传入分词器时按 `(问题, 片段)` 的顺序。顺序颠倒会改变分数 |
| **分数的含义** | 绝对值没有意义，只使用**同一问题内部的相对排序** |
| **截断由调用方决定** | 本适配器只做排序。保留多少条应根据上下文预算决定 |
| **片段边界** | 训练时以整张表、整个小节作为片段。切成句子级别会偏离训练条件 |

## 6. 训练数据

| 项目 | 内容 |
|---|---|
| 类型 | **韩语公共机构 PDF 报告** |
| 获取方式 | 从**各机构官方网站直接下载**其公开发布的报告 |
| 文档性质 | 预算与决算分析、农业与产业统计、交通调查、信息化议题分析 —— **表格密集的政策报告** |
| 标注 | 对每个问题，由**人工确认并标出作为答案依据的片段** |
| 训练条数 | 264 条（用于分数分离 98 条，用于成对排序违例 166 条） |
| 划分 | 按**文档划分**，同一文档的片段不会同时进入训练与验证 |

各机构分布如下，选取时避免集中于单一来源。

| 来源机构 | 训练条数 | 文档数 |
|---|---|---|
| 国会预算政策处（nabo） | 84 | 10 |
| 韩国农村经济研究院（krei） | 82 | 13 |
| 韩国交通研究院（koti） | 49 | 9 |
| 韩国智能信息社会振兴院（nia） | 49 | 12 |
| **合计** | **264** | **44** |

训练所用的问题与原始 PDF 未包含在本仓库中。

## 7. 评测 —— 使用 Gemma 4 E2B

### 7.1 测量条件

| 项目 | 取值 |
|---|---|
| 生成模型 | `google/gemma-4-E2B-it`（有效 2.3B，GGUF Q8_0） |
| 运行环境 | `llama-server`，上下文 10240，`temperature 0`，GPU 层 99，线程 4 |
| 机器 | AWS g6.xlarge（NVIDIA L4） |
| 评测条数 | **81** |
| 唯一变量 | **只有重排序模型不同。** 问题、候选片段与生成设置完全一致 |

评测条目的选取标准如下：在不使用重排序的基础 RAG 输入下 E2B **答错**，且**错误的取值来自
同一输入中的另一个片段**。也就是答案本在输入中，模型却读了别的片段。按此标准，基线为 **0 / 81**。

### 7.2 结果

| 重排序 | 正确数 | 挽回率 |
|---|---|---|
| 无（基础 RAG） | 0 / 81 | 0.0% |
| 公开 `BAAI/bge-reranker-v2-m3` | 17 / 81 | 21.0% |
| **本适配器（DKU）** | **58 / 81** | **71.6%** |

比公开模型多答对 **41 条（+50.6 个百分点）**（McNemar 精确检验，双侧 `p = 2.46e-10`）。

### 7.3 提升从何而来

差别在于把正确证据放到**输入首位**的条目数量。

| 重排序 | 正确证据排在首位的条目 |
|---|---|
| 无 | 1 / 81 |
| 公开 BGE | 18 / 81 |
| **本适配器** | **79 / 81** |

把同样的 81 条分成两组，原因就清楚了。

| 分组 | 条数 | 公开 BGE | 本适配器 |
|---|---|---|---|
| 正确证据排名**发生变化**的条目 | 61 | 6 / 61（10%） | **47 / 61（77%）** |
| 正确证据排名**未变化**的条目 | 18 | 11 / 18（61%） | 11 / 18（61%） |

在排名未变化的条目上，两个模型**完全相同**。可见提升并非因为题目更容易，而是**因为排名变了**。

### 7.4 未能确认的部分

| 项目 | 状态 |
|---|---|
| 仅保留前 28 条的筛选 | 在这 81 条上**未观察到差异**（`p = 1`） |
| 将问题置于证据之前的排布 | 在这 81 条上**未观察到差异**（`p = 1`） |
| 其他生成模型 | 未测试，仅用 E2B 测量 |
| 其他语言与文档类型 | 未测试 |

已确认的效果只有**重排序模型的训练**这一项。其余部分在本样本上未能确认方向。

## 8. 训练方法

| 项目 | 取值 |
|---|---|
| 基础模型 | `BAAI/bge-reranker-v2-m3`（主干冻结） |
| 训练部分 | LoRA 适配器 —— `r=8`、`alpha=16`、`dropout=0.05` |
| 目标模块 | `key`、`query`、`value` |
| 任务类型 | `SEQ_CLS`（同时保存分类头） |
| 监督信号 | 人工确认的**正确证据标注** |
| 损失 | 提升正确证据分数的项 + 使正确证据排在非证据之前的项（权重 1 : 1.0） |
| 训练条数 | 264（98 + 166） |
| epoch、学习率、seed | 5、0.0001、20260915 |
| max_length、step | 1024、980 |
| 耗时 | 82.9 分钟（5.07 秒/step） |
| 设备 | NVIDIA L4（AWS g6.xlarge） |

### 8.1 采用哪个 epoch —— **在训练前就以规则固定**

看到结果后再挑选，本身就构成选择偏差。因此在训练开始前（未跑任何一个 step）就写下了规则。

```
1. 排序违例率高于训练前（epoch 0）的 epoch 一律排除
2. 在其余 epoch 中选择证据分离指标最高者
3. 若指标相同，取更早的 epoch
```

| 指标 | 训练前 | 采用的 epoch 5 |
|---|---|---|
| 所有正确证据都排在非证据之前的条目比例 | 0.4796 | **0.7959** |
| 排序违例率（越低越好） | 0.2866 | **0.1311** |

这两个指标是**在训练集上测得的拟合程度**，不是下游任务性能，仅用于挑选适配器。
性能另见第 7 节。

## 9. 仓库结构

| 路径 | 内容 |
|---|---|
| `model/adapter_model.safetensors` | LoRA 适配器（8.9MB） |
| `model/adapter_config.json` | 适配器配置 |
| `src/build_c2_trainset.py` | 训练集构建 |
| `src/make_doc_folds_v2.py` | 按文档划分 |
| `src/train_c2_targeted.py` | LoRA 训练 |
| `docs/train_record.json` | 训练记录 —— 配置、损失、epoch 选择 |

训练所用的问题与原始 PDF 未包含在内。

## 10. 局限

| 局限 | 内容 |
|---|---|
| 数据范围 | 使用来自韩国四家机构、44 份报告的 264 条数据训练 |
| 评测范围 | 仅在一个生成模型（E2B）、81 条数据上验证 |
| 评测集性质 | 仅收集基础 RAG **答错**的条目，并非一般问题分布 |
| 只针对一种失败 | 只处理"答案在输入中却读了别的片段"的情形，无法弥补检索未召回的情况 |
| 对比对象 | 仅与公开的 `bge-reranker-v2-m3` 作比较 |

## 11. 常见问题

**是否随附基础模型？** 否。`BAAI/bge-reranker-v2-m3` 需从其公开仓库获取，本仓库只包含叠加其上的适配器。

**能用于英文文档吗？** 基础模型是多语言的，可以运行。但本适配器只在韩语上训练，英文上的收益未经验证。

**保留多少条比较合适？** 评测中使用了前 28 条。不过该筛选步骤本身未观察到可测量的效果（见 7.4），
请按你的上下文预算决定。

**可以给分数设阈值吗？** 不建议。分数的绝对尺度随问题而变，固定阈值会漂移。
请使用**同一问题内部的相对排序**。

**公开的文件与论文中使用的是同一个吗？** 是，逐字节一致。

## 12. 联系方式

本适配器是在硕士学位论文研究过程中训练并公开的。
如有咨询，请发送至 <sh.yang@dankook.ac.kr>。

## 13. 许可证

本适配器是 `BAAI/bge-reranker-v2-m3` 的衍生作品，采用与原作相同的 **Apache License 2.0**。
同时须遵守原模型的许可证与署名要求。全文见 [LICENSE](LICENSE)。

## 14. 引用

引用时请使用以下格式。

> 梁成勋, "DKU-reranker-v1: A LoRA Adapter for Korean Public-Document Evidence Reranking," GitHub 仓库, 2026. [Online]. Available: https://github.com/tracer999/dku-reranker

若使用 BibTeX，请直接复制下方条目。

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
