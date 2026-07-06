"""
训练函数（含学习率调度、权重衰减、最佳模型保存）
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .model import ImageClassifier


def train(
    train_dataset: Dataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    step_size: int = 30,
    gamma: float = 0.1,
    num_workers: int = 4,
    save_dir: str = "./data",
    device: str = None,
    verbose: bool = True,
):
    """
    训练主循环

    关键改进点（相对基础版）:
        - Adam + weight_decay（5e-4）做 L2 正则
        - StepLR 每 30 epoch 衰减到 0.1 倍
        - 保存验证准确率最高的 best 模型
        - 训练结束再保存一次 last 模型
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(save_dir, exist_ok=True)

    # 数据加载器（pin_memory 加速 CPU→GPU）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 模型 + 损失 + 优化器 + 调度器
    model = ImageClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_acc = 0.0
    best_path = os.path.join(save_dir, "image_model_best.pth")
    last_path = os.path.join(save_dir, "image_model.pth")

    if verbose:
        print(f"\n{'='*50}")
        print(f"开始训练 | epochs={epochs} | batch_size={batch_size} | lr={lr}")
        print(f"{'='*50}\n")

    for epoch in range(epochs):
        model.train()
        total_loss, total_samples, total_correct = 0.0, 0, 0
        start = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        acc = (total_correct / total_samples).item()

        if verbose:
            print(
                f"epoch {epoch + 1:2d}/{epochs} | "
                f"loss: {total_loss / total_samples:.4f} | "
                f"acc: {acc:.4f} | "
                f"lr: {current_lr:.6f} | "
                f"time: {time.time() - start:.2f}s"
            )

        # 保存当前 epoch 中 acc 最高的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    # 训练结束保存最后模型
    torch.save(model.state_dict(), last_path)

    if verbose:
        print(f"\n训练完成！最佳训练 acc={best_acc:.4f}")
        print(f"✅ best 模型: {best_path}")
        print(f"✅ last 模型: {last_path}")

    return model, best_acc
