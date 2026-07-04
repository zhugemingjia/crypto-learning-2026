"""
训练函数
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import PhonePriceClassifier


def train(
    train_dataset: TensorDataset,
    input_dim: int,
    num_classes: int,
    scaler=None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "./model/phone.pth",
    device: str = None,
    verbose: bool = True,
):
    """
    训练主循环

    Args:
        train_dataset: 训练集 TensorDataset
        input_dim: 输入特征数
        num_classes: 类别数
        scaler: StandardScaler 实例，训练时会一并保存到 model 目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        save_path: 模型权重保存路径
        device: 'cpu' / 'cuda'，None 表示自动选
        verbose: 是否打印训练过程
    """
    # 设备选择
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数、优化器
    model = PhonePriceClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss, batch_num = 0.0, 0
        start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # 前向
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

        avg_loss = total_loss / batch_num
        loss_history.append(avg_loss)
        if verbose:
            print(f"epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.4f}  time={time.time() - start:.2f}s")

    # 保存模型 + 标准化器
    import os
    torch.save(model.state_dict(), save_path)

    if scaler is not None:
        from .dataset import save_artifact
        scaler_path = os.path.splitext(save_path)[0] + ".pkl"
        save_artifact(scaler, scaler_path)
        if verbose:
            print(f"\n✅ 模型已保存到: {save_path}")
            print(f"✅ 标准化器已保存到: {scaler_path}")
    else:
        if verbose:
            print(f"\n✅ 模型已保存到: {save_path}")

    return model, loss_history
