# 🖼️ CIFAR-10 Image Classifier

> 基于 CNN 的 CIFAR-10 图像分类项目。**深度学习入门第二个完整项目**（第一个是 [phone-price-classifier](../phone-price-classifier)）。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 项目目标

把 32×32 像素的彩色图片**分成 10 类**（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车）。

CIFAR-10 是 CV 入门最经典的 benchmark，**简单模型 50%、LeNet 60%、现代 CNN 95%+**。

---

## 🏗️ 网络结构（优化版）

```
Input (3, 32, 32)
  ↓
Conv2d(3, 32, 3, padding=1) → BN → ReLU → MaxPool    # 32x32 → 16x16
  ↓
Conv2d(32, 64, 3, padding=1) → BN → ReLU → MaxPool   # 16x16 → 8x8
  ↓
Conv2d(64, 128, 3, padding=1) → BN → ReLU → MaxPool  # 8x8 → 4x4
  ↓
flatten → Linear(2048, 512) → ReLU → Dropout(0.5)
  ↓
Linear(512, 256) → ReLU → Dropout(0.3)
  ↓
Linear(256, 128) → ReLU
  ↓
Linear(128, 10)   ← logits
```

**超参数**：

| 项 | 值 |
|---|---|
| 优化器 | Adam (lr=1e-3, weight_decay=5e-4) |
| 学习率调度 | StepLR (每 30 epoch × 0.1) |
| 批大小 | 64 |
| 训练轮数 | 50 |
| 损失函数 | CrossEntropyLoss |
| 数据增强 | RandomCrop(4) + RandomHorizontalFlip + Normalize |
| 评估指标 | 测试集 Top-1 Accuracy |

---

## 📈 效果

**测试集准确率：86.23%**（8623/10000）✅

| 版本 | 准确率 | 提升点 |
|---|---|---|
| 基础版（2 层 LeNet）| 60.51% | 起点 |
| **优化版（当前）** | **86.23%** | +25.72% |

**优化关键**（基础版 → 优化版的改动）：

1. **加 BatchNorm**：每个 conv 后 → 加速收敛
2. **加 Dropout**：2 个 fc 后 0.5/0.3 → 防过拟合
3. **加数据增强**：RandomCrop + RandomHorizontalFlip + Normalize → 防过拟合
4. **加学习率调度**：StepLR 30 epoch × 0.1 → 后期精调
5. **加权重衰减**：weight_decay=5e-4 → L2 正则
6. **网络加深**：2 层 → **3 层 conv**（6/16 → 32/64/128）
7. **batch 加大**：8 → 64 → GPU 利用率
8. **epochs 加大**：10 → 50
9. **保存 best**：训练 acc 最高的权重单独存

---

## 📂 目录结构

```
cifar10-classifier/
├── main.py              # 入口脚本（train / evaluate / predict）
├── requirements.txt
├── README.md
├── LICENSE              # MIT
├── .gitignore
│
├── src/                 # 源代码
│   ├── __init__.py
│   ├── dataset.py       # CIFAR-10 加载 + 数据增强
│   ├── model.py         # 3 层 CNN + BN + Dropout
│   ├── train.py         # 训练循环 + 调度器 + best 保存
│   └── evaluate.py      # 评估 + 单张预测
│
└── data/                # 数据 + 模型权重
    ├── img.jpg          # predict 演示样本
    ├── image_model.pth        # 训练最后的权重
    └── image_model_best.pth   # 训练中 acc 最高的权重
```

> **CIFAR-10 数据集（186MB）不上传**。首次训练时 torchvision 会自动从官网下载到 `data/cifar-10-batches-py/`，后续运行直接读本地缓存。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练（首次会下载 CIFAR-10，~186MB）

```bash
python main.py --mode train --epochs 50
```

**GPU 推荐配置**：RTX 3060+ / 8GB 显存 / CUDA 12.x。CPU 也能跑，约 30-60 分钟/50 epoch。

### 3. 评估（用 best 权重跑测试集）

```bash
python main.py --mode evaluate
```

预期输出：
```
==================================================
测试集准确率: 86.23%  (8623/10000)
==================================================
```

### 4. 单张图片预测

```bash
python main.py --mode predict --input data/img.jpg
```

输出示例：
```
预测结果: 类别 0 → airplane
Top-3 概率:
  airplane      0.8521
  bird          0.0723
  ship          0.0431
```

---

## 🛠️ 调参建议

如果想进一步提升准确率（**目标 90%+**），可以试试：

| 改动 | 预期提升 | 难度 |
|---|---|---|
| 加更多数据增强（Cutout / MixUp）| +1~3% | 简单 |
| 换 ResNet18 预训练模型 | +5~8% | 中等 |
| 加 CosineAnnealingLR 替代 StepLR | +1~2% | 简单 |
| 加 Label Smoothing | +0.5~1% | 简单 |
| 跑 200 epoch | +1~2% | 简单（耗时长）|

---

## 📚 学习笔记

本项目对应 `ml-learn/notes/` 下的 PyTorch 速查手册：

- `01-张量操作.md` → 数据预处理
- `03-数据加载.md` → DataLoader + pin_memory + num_workers
- `04-神经网络构建.md` → nn.Module 子类
- `06-激活函数.md` → ReLU
- `07-损失函数.md` → CrossEntropyLoss
- `08-优化器.md` → Adam + weight_decay
- `09-学习率调度器.md` → StepLR
- `10-正则化与归一化.md` → Dropout + BatchNorm + Normalize
- `99-训练循环模板.md` → 标准训练循环

---

## 📄 License

MIT © 2026 ZREO
