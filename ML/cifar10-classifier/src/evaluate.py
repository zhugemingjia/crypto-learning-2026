"""
评估 / 推理函数
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from .dataset import CLASS_NAMES, get_transforms
from .model import ImageClassifier


def evaluate(
    test_dataset: Dataset,
    model_path: str = "./data/image_model_best.pth",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = None,
) -> float:
    """
    在测试集上评估模型准确率（默认加载 best 权重）

    Returns:
        测试集准确率 (0~1)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = ImageClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_pred = model(x).argmax(dim=-1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"\n{'='*50}")
    print(f"测试集准确率: {100*acc:.2f}%  ({correct}/{total})")
    print(f"{'='*50}")
    return acc


def predict_single(
    image_path: str,
    model_path: str = "./data/image_model_best.pth",
    device: str = None,
) -> tuple:
    """
    对单张图片做预测

    Args:
        image_path: 图片文件路径（jpg/png/...）
    Returns:
        (类别索引, 类别名, 各类别概率字典)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 加载图片 + 预处理
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    transform = get_transforms(train=False)
    x = transform(img).unsqueeze(0)  # (1, 3, 32, 32)

    # 加载模型
    model = ImageClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(probs.argmax().item())
        pred_name = CLASS_NAMES[pred_idx]
        prob_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

    return pred_idx, pred_name, prob_dict
