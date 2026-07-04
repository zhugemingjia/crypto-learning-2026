"""
评估函数
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import PhonePriceClassifier
from .dataset import load_artifact


def _load_model_and_scaler(model_path: str, device: torch.device):
    """加载模型权重和标准化器（如果有）"""
    model = PhonePriceClassifier  # placeholder, 实际由调用者构造
    scaler = None
    scaler_path = os.path.splitext(model_path)[0] + ".pkl"
    if os.path.exists(scaler_path):
        scaler = load_artifact(scaler_path)
    return scaler


def evaluate(
    test_dataset: TensorDataset,
    input_dim: int,
    num_classes: int,
    model_path: str = "./model/phone.pth",
    device: str = None,
    batch_size: int = 32,
) -> float:
    """
    在测试集上评估模型准确率

    Returns:
        准确率 (0~1)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 加载模型
    model = PhonePriceClassifier(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            total += y.size(0)
            correct += (y_pred == y).sum().item()

    acc = correct / total
    print(f"测试集准确率: {100 * acc:.2f}%  ({correct}/{total})")
    return acc


def predict_single(
    x: torch.Tensor,
    input_dim: int,
    num_classes: int,
    model_path: str = "./model/phone.pth",
    device: str = None,
) -> int:
    """
    对单条样本做预测（**已标准化**的特征）

    Args:
        x: 形状 (input_dim,) 或 (1, input_dim) 的 tensor，必须**已经过 scaler 标准化**
    Returns:
        预测的类别索引
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = PhonePriceClassifier(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    if x.dim() == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x.to(device))
        pred = torch.argmax(logits, dim=1).item()

    return pred


def predict_from_csv(
    csv_path: str,
    input_dim: int,
    num_classes: int,
    model_path: str = "./model/phone.pth",
    device: str = None,
) -> int:
    """
    从 CSV 读样本 + 加载 scaler + 标准化 + 预测（一站式）

    Args:
        csv_path: 包含特征列的 CSV 路径（只取前 input_dim 列）
    """
    import pandas as pd
    sample = pd.read_csv(csv_path).iloc[0, :input_dim].values.astype("float32").reshape(1, -1)

    # 加载 scaler
    scaler_path = os.path.splitext(model_path)[0] + ".pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到标准化器: {scaler_path}，请先跑一次 train 生成。")
    scaler = load_artifact(scaler_path)
    sample = scaler.transform(sample).astype("float32")

    return predict_single(
        x=torch.tensor(sample).squeeze(0),
        input_dim=input_dim,
        num_classes=num_classes,
        model_path=model_path,
        device=device,
    )
