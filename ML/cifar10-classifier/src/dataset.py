"""
数据加载与预处理

CIFAR-10 通过 torchvision 在线下载到 ./data/cifar-10-batches-py/
训练集加了数据增强（RandomCrop + RandomHorizontalFlip + Normalize）
测试集只做 Normalize
"""

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor


# CIFAR-10 类别名（predict 时用）
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool = True):
    """
    根据是否训练返回不同的 transform 流水线
    - 训练:  数据增强（RandomCrop + RandomHorizontalFlip + Normalize）
    - 测试/推理: 只 Normalize
    """
    if train:
        return Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def build_datasets(data_root: str = "./data", download: bool = True):
    """
    加载 CIFAR-10 训练集和测试集

    第一次运行会自动下载 ~186MB 到 ./data/cifar-10-batches-py/
    之后运行直接读本地缓存

    Returns:
        train_dataset, test_dataset
    """
    train_dataset = CIFAR10(
        root=data_root, train=True, transform=get_transforms(train=True), download=download
    )
    test_dataset = CIFAR10(
        root=data_root, train=False, transform=get_transforms(train=False), download=download
    )
    return train_dataset, test_dataset


def get_class_names():
    """返回 CIFAR-10 10 个类别的英文名"""
    return CLASS_NAMES
