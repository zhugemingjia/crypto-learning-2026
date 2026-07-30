# 🇬🇧→🇫🇷 英译法 Seq2Seq 翻译模型

> 基于 **GRU + Attention** 的 Encoder-Decoder 框架，实现英语 → 法语的机器翻译。**深度学习 NLP 进阶项目**，掌握 Seq2Seq 架构与注意力机制。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 项目目标

使用 **Seq2Seq（Encoder-Decoder）框架**，训练一个英语到法语的翻译模型。输入一句英文（如 "i love you ."），输出对应的法语翻译（如 "je t aime ."）。

这是 NLP 序列到序列建模的经典任务，涵盖了文本预处理、词汇表构建、Teacher Forcing、注意力机制等核心技术。

---

## 🧠 核心概念

### Seq2Seq 框架（N vs M）

- **编码器（Encoder）**：将可变长度的英语句子压缩为固定长度的上下文向量
- **解码器（Decoder）**：根据上下文向量自回归地生成可变长度的法语句子
- **Teacher Forcing**：训练时以一定概率使用真实标签（而非上一步预测结果）作为解码器输入，加速收敛
- **Attention 机制**：解码器每一步动态关注编码器不同时间步的输出，解决长句信息瓶颈问题

---

## 🏗️ 网络结构

```
Encoder                          Decoder (with Attention)
───────                          ───────────────────────
Input [1, seq_len]               Input [1, 1]  ← SOS token
  ↓                                ↓
Embedding(2803 → 256)            Embedding(4345 → 256)
  ↓                                ↓
GRU(256, 256)                    Attention(concat(emb, hidden))
  ↓                                ↓
Output [1, seq_len, 256]         attn_combine + ReLU
  Hidden [1, 1, 256]              ↓
                                 GRU(256, 256)
                                   ↓
                                 Linear(256 → 4345)
                                   ↓
                                 LogSoftmax → 预测词概率
```

**超参数**：

| 项 | 值 |
|---|---|
| 英语词汇表 | 2,803 |
| 法语词汇表 | 4,345 |
| 隐藏层维度 | 256 |
| 最大句子长度 | 10 |
| 优化器 | Adam (lr=5e-4) |
| 损失函数 | NLLLoss |
| 训练轮数 | 20 |
| Teacher Forcing | 0.3 |
| 批大小 | 1（变长序列，逐样本训练）|

---

## 📂 目录结构

```
eng-fra-translator/
├── main.py              # 入口脚本（train / evaluate / test）
├── requirements.txt
├── README.md
├── LICENSE              # MIT
├── .gitignore
│
├── src/                 # 源代码
│   ├── __init__.py
│   ├── dataset.py       # 数据预处理 + 词汇表构建 + Dataset
│   ├── model.py         # EncoderRNN / DecoderRNN / AttnDecoderRNN
│   ├── train.py         # 单批次训练 + 完整训练循环
│   └── evaluate.py      # 推理 & 批量评估
│
├── data/                # 数据
│   └── eng-fra-v2.txt   # 英法平行语料（TSV 格式）
│
├── model/               # 预训练模型权重
│   ├── my_encoder_rnn_1~5.pth
│   └── my_attn_decoder_rnn_1~5.pth
│
└── img/                 # 可视化输出
    └── seq2seq_loss.png  # 损失曲线
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 评估（使用预训练权重翻译）

```bash
python main.py --mode evaluate
```

输出示例：
```
输入（英文句子）: i love you .
参考（法语句子）: je t aime .
输出（预测法语）:   je t aime .
```

### 3. 训练

```bash
# 使用默认 20 轮训练
python main.py --mode train

# 指定训练轮数
python main.py --mode train --epochs 10
```

训练完成后：
- 模型权重保存至 `./model/`（每轮一个 checkpoint）
- 损失曲线保存至 `./img/seq2seq_loss.png`

### 4. 模型结构测试

```bash
python main.py --mode test
```

---

## 📚 学习笔记

本项目对应 PyTorch NLP 学习路线中的 Seq2Seq & Attention：

- **Encoder-Decoder 架构**：编码器和解码器使用独立 GRU，处理 N→M 的序列转换
- **Teacher Forcing**：训练时以概率 `0.3` 使用真实标签，平衡收敛速度与泛化能力
- **Attention 权重计算**：`softmax(Linear(concat(embedded, hidden)))`，然后 `bmm` 加权求和
- **自回归解码**：推理时逐词生成，遇到 EOS 或达到 MAX_LENGTH 时停止
- `batch_first=True` 使输入形状为 `[batch, seq_len, features]`
- NLLLoss 需要配合 LogSoftmax 输出使用
- 变长序列训练时 `batch_size=1` 是最简单的处理方式

**对比：带 Attention vs 不带 Attention**

| 解码器 | 编码器输出使用 | 长句效果 |
|---|---|---|
| DecoderRNN | 仅用最后一步 hidden | 信息瓶颈，长句丢失严重 |
| AttnDecoderRNN | 动态关注所有时间步 | 长距离依赖保持好 |

---

## 📄 License

MIT © 2026 ZREO
