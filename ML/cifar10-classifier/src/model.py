"""
模型定义：3 层 CNN + BatchNorm + Dropout（针对 CIFAR-10 优化版）
"""

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    """
    CIFAR-10 图像分类网络（优化版）

    结构:
        Conv2d(3, 32, 3, padding=1) → BN → ReLU → MaxPool   # 32x32 -> 16x16
        Conv2d(32, 64, 3, padding=1) → BN → ReLU → MaxPool  # 16x16 -> 8x8
        Conv2d(64, 128, 3, padding=1) → BN → ReLU → MaxPool # 8x8 -> 4x4
        flatten → Linear(2048, 512) → ReLU → Dropout(0.5)
        Linear(512, 256) → ReLU → Dropout(0.3)
        Linear(256, 128) → ReLU
        Linear(128, 10)   # logits (CrossEntropyLoss 内部含 Softmax)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 3 个卷积块（Conv + BN + ReLU + Pool）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3 个全连接层 + Dropout
        self.fc1      = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2      = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)

        # 输出层
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        # 卷积块 1
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        # 卷积块 2
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        # 卷积块 3
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        # flatten
        x = x.reshape(x.size(0), -1)

        # 全连接 + Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))

        return self.output(x)
