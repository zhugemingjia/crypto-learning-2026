# 📱 Phone Price Classifier

> 基于 ANN（人工神经网络）的手机价格区间分类项目。**这是我个人深度学习入门阶段的第一个完整项目。**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 项目目标

给定手机的 **20 维特征**（电池、内存、屏幕、摄像头、重量等），预测它属于 **4 个价格档位**中的哪一档：

| 档位 | 含义 |
|---|---|
| 0 | 低价 |
| 1 | 中低 |
| 2 | 中高 |
| 3 | 高价 |

这是一个**多分类**问题。

---

## 🏗️ 网络结构

```
Input (20)
  ↓
Linear(20, 256) → ReLU → Dropout(0.3)
  ↓
Linear(256, 512) → ReLU → Dropout(0.3)
  ↓
Linear(512, 256) → ReLU → Dropout(0.3)
  ↓
Linear(256, 4)   ← logits (CrossEntropyLoss 内部含 Softmax)
```

**超参数**：

| 项 | 值 |
|---|---|
| 优化器 | Adam |
| 学习率 | 1e-3 |
| 批大小 | 32 |
| 训练轮数 | 100 |
| 损失函数 | CrossEntropyLoss |
| 数据预处理 | StandardScaler |

**测试集准确率：88.25%**（353/400），训练 loss 从 0.99 → 0.029。

---

## 📂 目录结构

```
phone-price-classifier/
├── main.py              # 入口脚本（train / evaluate / predict）
├── requirements.txt     # 依赖列表
├── README.md            # 本文件
├── LICENSE              # MIT 协议
├── .gitignore           # Git 忽略规则
│
├── src/                 # 源代码
│   ├── __init__.py
│   ├── dataset.py       # 数据加载 + 预处理
│   ├── model.py         # 网络结构
│   ├── train.py         # 训练函数
│   └── evaluate.py      # 评估 + 单条预测
│
├── data/
│   └── 手机价格预测.csv  # 训练数据
│
└── model/               # 模型权重
    ├── phone.pth        # PyTorch state_dict
    └── phone.pkl        # StandardScaler（joblib 序列化）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python main.py --mode train --epochs 100
```

### 3. 评估模型

```bash
python main.py --mode evaluate
```

### 4. 单条预测

```bash
python main.py --mode predict --input data/你的样本.csv
```

---

## 📈 迭代记录

| 版本 | 文件 | 优化器 | 网络 | 标准化 | 训练集比例 |
|---|---|---|---|---|---|
| v1（早期实验）| `ann_手机价格_分类.py` | SGD lr=1e-4 | 2 层 128→256 | 无 | 20%（实为 bug）|
| **v2（当前）** | `main.py` | Adam lr=1e-3 | 3 层 256→512→256 + Dropout | StandardScaler | 80% |

> v1 → v2 的关键改进：换成 Adam + 加入 Dropout + 数据标准化 + 修复 train/test 划分比例 + 训练轮数从 50 增到 100。

---

## 🛠️ 依赖

- Python 3.11
- PyTorch 2.0+
- scikit-learn 1.3+
- pandas / numpy

详见 `requirements.txt`。

---

## 📝 学习笔记

本项目对应我在 `ml-learn/notes/` 下整理的 PyTorch 学习笔记：

- `01-张量操作.md` → 数据预处理
- `02-自动微分.md` → `loss.backward()` 原理
- `03-数据加载.md` → `TensorDataset` + `DataLoader`
- `04-神经网络构建.md` → `nn.Module` 子类
- `05-参数初始化.md` → 默认初始化
- `06-激活函数.md` → ReLU + Softmax
- `07-损失函数.md` → `CrossEntropyLoss`
- `08-优化器.md` → Adam
- `10-正则化与归一化.md` → Dropout + StandardScaler
- `99-训练循环模板.md` → 标准训练循环

---

## 📄 License

MIT © 2026 ZREO
