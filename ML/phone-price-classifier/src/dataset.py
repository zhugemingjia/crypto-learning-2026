"""
数据加载与预处理

流程：CSV → 特征/标签分离 → 标准化 → 划分训练/测试集 → TensorDataset
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """读取原始 CSV"""
    data = pd.read_csv(csv_path)
    return data


def preprocess(data: pd.DataFrame):
    """
    数据预处理：
    1. 分离特征 X 与标签 y
    2. StandardScaler 标准化特征
    3. 标签转为 int64（CrossEntropyLoss 要求）

    Returns:
        x, y (numpy array)
    """
    x = data.iloc[:, :-1].astype(np.float32).values
    y = data.iloc[:, -1].astype(np.int64).values
    return x, y


def build_datasets(csv_path: str, test_size: float = 0.2, random_state: int = 3):
    """
    构造训练集、测试集的 TensorDataset。
    内部 fit 一个 StandardScaler 并作用于 train/test 集。

    Returns:
        train_dataset, test_dataset, input_dim, num_classes, scaler
    """
    data = load_raw_data(csv_path)
    x, y = preprocess(data)

    # 划分训练/测试集（stratify=y 保持类别分布一致）
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 训练集上 fit 标准化器，训练/测试都用训练集的均值方差
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)

    # 转 TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y)), scaler


def save_artifact(scaler, path: str) -> None:
    """保存 StandardScaler 到文件（joblib 序列化）"""
    import joblib
    joblib.dump(scaler, path)


def load_artifact(path: str):
    """从文件加载 StandardScaler"""
    import joblib
    return joblib.load(path)
