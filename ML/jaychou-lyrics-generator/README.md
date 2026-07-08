# 🎤 Jaychou Lyrics Generator

> 基于 RNN 的周杰伦风格歌词生成器。**深度学习入门第三个完整项目**（前两个：[phone-price-classifier](../phone-price-classifier) / [cifar10-classifier](../cifar10-classifier)）。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 项目目标

用周杰伦歌词训练一个 RNN 模型，**自动续写周杰伦风格的歌词**。这是 NLP 入门最经典的练手项目。

---

## 🏗️ 网络结构

```
Input: 词索引 (batch, seq_len)
  ↓
nn.Embedding(vocab_size, 128)        # 词 → 128 维向量
  ↓
nn.RNN(128, 256, num_layers=1)      # 单层 RNN
  ↓
nn.Linear(256, vocab_size)            # → 词表大小
  ↓
Output: 每个位置下一个词的 logits
```

**超参数**：

| 项 | 值 |
|---|---|
| 优化器 | Adam (lr=1e-3) |
| 损失函数 | CrossEntropyLoss |
| 批大小 | 5 |
| 序列长度 | 32 |
| 训练轮数 | 10 |
| 词表大小 | ~5700 词 |
| 关键技巧 | hidden.detach() + 梯度裁剪 clip_grad_norm_=5 |

---

## 📈 效果示例

> 起始词 "爱"，温度 1.0，生成 30 字

```
爱我连恨都难以下… 将真心抽离写成日… 像是一场默… 你的完美主义 太彻… 分手的话…
```

> 风格是不是有点周杰伦了？训练 10 epoch + 基础款 RNN，能学到这种程度算不错了。

---

## 📂 目录结构

```
jaychou-lyrics-generator/
├── main.py              # 入口（train / generate）
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── dataset.py       # jieba 分词 + 词表 + Dataset
│   ├── model.py         # TextGenerator
│   ├── train.py         # 训练循环（含 vocab 保存）
│   └── generate.py      # 温度采样生成
│
└── data/
    ├── vocab.json       # 词表（自动生成）
    └── lyric_model.pth  # 模型权重
```

> **歌词文件（jaychou_lyrics.txt）不上传**——版权问题，README 写明如何自行准备。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备歌词文件

把周杰伦歌词（UTF-8 编码、**每行一句**）放到 `data/jaychou_lyrics.txt`。

> 数据集来源：网上搜 "jaychou_lyrics.txt" 就能找到，kaggle 上也有。**别传到 GitHub**。

### 3. 训练

```bash
python main.py --mode train --epochs 10
```

### 4. 生成歌词

```bash
# 默认：从"爱"开始 50 字，temperature 1.0
python main.py --mode generate

# 自定义
python main.py --mode generate --start 我 --length 100 --temperature 0.8
```

**温度采样小贴士**：
- `temperature=0.5` → 偏保守，重复多但通顺
- `temperature=1.0` → 平衡（推荐）
- `temperature=1.5` → 偏发散，可能蹦出怪词

---

## 🔑 关键 API 速查

### `nn.RNN`

```python
nn.RNN(input_size, hidden_size, num_layers, batch_first)
```

| 参数 | 作用 | 常用值 |
|---|---|---|
| `input_size` | 输入特征维度（词向量维度）| 128 |
| `hidden_size` | 隐藏状态维度（记忆容量）| 256 |
| `num_layers` | 堆叠层数 | 1（单层）/ 2（多层）|
| `batch_first` | True 则输入 (batch, seq, feature)，False 则 (seq, batch, feature) | False |

### `nn.Embedding`

```python
nn.Embedding(num_embeddings, embedding_dim)
```

把每个词的**整数索引** → `embedding_dim` 维向量。可以理解为"可学习的查表"。

### 关键训练 trick

1. **`hidden.detach()`** —— 每个 batch 训练前把上一 batch 的隐藏状态分离出计算图，**避免跨 batch 反传**
2. **`clip_grad_norm_(max_norm=5)`** —— RNN 极易梯度爆炸，必须裁剪
3. **数据切成定长滑动窗口**（next-token prediction）：`x[0..n] → y[1..n+1]`

---

## 📚 关联笔记

`ml-learn/notes/` 下：
- `02-自动微分.md` → `hidden.detach()` 原理
- `04-神经网络构建.md` → `nn.Module` 子类写法
- `08-优化器.md` → Adam
- `09-学习率调度器.md` → 文本生成可加 CosineAnnealing
- `99-训练循环模板.md` → 训练循环骨架

---

## 🚀 升级方向

| 改动 | 预期效果 | 难度 |
|---|---|---|
| 换 `nn.LSTM` | 长距离依赖学得更好 | 简单 |
| 加 Embedding 预训练（word2vec）| 生成质量提升 | 中等 |
| 加 Attention | 看更长上下文 | 中等 |
| 换 Transformer | 现代化标准做法 | 中等 |
| 字符级（不要 jieba）| 词表小但慢 | 简单 |

---

## 📄 License

MIT © 2026 ZREO

**注意**：训练数据（周杰伦歌词）版权归原作者所有，本项目**不包含**歌词文件。
