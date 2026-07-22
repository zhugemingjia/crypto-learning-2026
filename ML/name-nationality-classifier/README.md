# 🌍 人名国家分类器

> 基于 RNN / LSTM / GRU 的人名国籍识别项目。**深度学习 NLP 入门项目**，掌握循环神经网络的三种经典变体。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 项目目标

根据人名字母序列**预测国籍/语言区域**。输入一个人名（如 "Smith"），输出最可能的 3 个国家及概率。

这是一个经典的 NLP 序列分类入门任务，使用 **one-hot 字符编码 + 循环神经网络** 实现，适合学习 RNN 系列模型的基础用法。

---

## 🧠 为什么用 One-Hot 而不是 Embedding？

本案例是人名分类任务中**少数适合用 one-hot 编码的场景**：

- 人名由字母组成，字母之间**没有语义依赖**（"a" 和 "b" 在名字中是独立的字符单元）
- 字符集很小（57 个），one-hot 维度不高
- 更适合展示 RNN 原生的序列建模能力

如果换成词级别的 NLP 任务（如情感分析），则应使用 Embedding 层。

---

## 🏗️ 网络结构

三种模型结构一致：**RNN/LSTM/GRU → Linear → LogSoftmax**

```
Input [seq_len, 57]          ← one-hot 编码的名字
  ↓
RNN / LSTM / GRU             ← 128 hidden units, 1 layer
  ↓
取最后时间步 output[-1]       ← [1, 128]
  ↓
Linear(128, 18)              ← 全连接
  ↓
LogSoftmax                   ← 对数概率
  ↓
Output [1, 18]               ← 18 个国家类别的 log-prob
```

**超参数**：

| 项 | 值 |
|---|---|
| 输入维度 | 57（字符集大小）|
| 隐藏层 | 128 |
| 输出维度 | 18（国家数）|
| RNN 层数 | 1 |
| 优化器 | Adam (lr=1e-3) |
| 损失函数 | NLLLoss |
| 训练轮数 | 3 |
| 批大小 | 1（变长序列，逐样本训练）|

---

## 📊 三类模型对比

| 模型 | 参数量 | 训练速度 | 特点 |
|---|---|---|---|
| **RNN** | 少 | 快 | 基础循环网络，长序列有梯度消失问题 |
| **LSTM** | 多 | 中 | 引入遗忘门/输入门/输出门，擅长长距离依赖 |
| **GRU** | 中 | 快 | LSTM 简化版，减少一个门，效果接近但更快 |

---

## 🗺️ 18 个国家/语言类别

Italian, English, Arabic, Spanish, Scottish, Irish, Chinese, Vietnamese, Japanese, French, Greek, Dutch, Korean, Polish, Portuguese, Russian, Czech, German

---

## 📂 目录结构

```
name-nationality-classifier/
├── main.py              # 入口脚本（train / predict / predict-all / test）
├── requirements.txt
├── README.md
├── LICENSE              # MIT
├── .gitignore
│
├── src/                 # 源代码
│   ├── __init__.py
│   ├── dataset.py       # 数据加载 + one-hot 编码 + Dataset
│   ├── model.py         # My_RNN / My_LSTM / My_GRU
│   ├── train.py         # 训练循环 + 三模型对比 + 可视化
│   └── predict.py       # 单模型/多模型预测（Top-3）
│
├── data/                # 数据
│   └── name_classfication.txt   # 训练数据（TSV 格式）
│
└── model/               # 训练好的模型权重
    ├── my_rnn_name_prj1.bin
    ├── my_lstm_name_prj1.bin
    └── my_gru_name_prj1.bin
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 预测（使用预训练权重）

```bash
# 三模型对比预测
python main.py --mode predict-all --name "Smith"

# 单模型预测（GRU）
python main.py --mode predict --name "Suzuki"
```

输出示例：
```
[RNN 预测] 待预测人名: Smith
----------------------------------------
  Top 1: English      概率: 0.4521  (log_prob: -0.7937)
  Top 2: Scottish     概率: 0.1834  (log_prob: -1.6955)
  Top 3: Irish        概率: 0.1276  (log_prob: -2.0591)
```

### 3. 训练

```bash
# 完整三模型训练 + 对比图表
python main.py --mode train

# 单独训练某个模型
python main.py --mode train-lstm
```

训练完成后会在 `./img/` 目录生成三张对比图：
- `RNN_LSTM_GRU_loss.png` — 损失曲线
- `RNN_LSTM_GRU_time.png` — 训练耗时
- `RNN_LSTM_GRU_acc.png` — 准确率曲线

### 4. 模型结构测试

```bash
python main.py --mode test
```

---

## 📚 学习笔记

本项目对应 PyTorch NLP 学习路线：

- RNN / LSTM / GRU 的 PyTorch API 使用（三者几乎一致）
- `batch_first=False` 时输入形状为 `[seq_len, batch, features]`
- one-hot 编码适用场景：字符级、小词表、无语义依赖
- NLLLoss 需要配合 LogSoftmax 输出使用
- 变长序列训练时 batch_size=1 是最简单的处理方式

---

## 📄 License

MIT © 2026 ZREO
