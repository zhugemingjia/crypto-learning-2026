"""
数据加载与预处理模块

- 读取人名-国家 TSV 数据
- 人名 → one-hot 编码张量
- PyTorch Dataset / DataLoader 封装
"""

import string
import torch
from torch.utils.data import DataLoader, Dataset

# ============ 全局常量 ============

# 字符集（字母 + 标点）
all_letters = string.ascii_letters + ".;,'"
n_letters = len(all_letters)  # 57

# 18 个国家/语言类别
categorys = [
    'Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish',
    'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
    'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German',
]
category_num = len(categorys)  # 18

# ============ 数据读取 ============


def read_data(file_path):
    """
    读取 TSV 数据文件（每行: 人名\t国家名）

    Args:
        file_path: 数据文件路径

    Returns:
        my_list_x: 人名字符串列表
        my_list_y: 国家名称字符串列表
    """
    my_list_x, my_list_y = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) < 5:
                continue
            x, y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)

    return my_list_x, my_list_y


# ============ Dataset 类 ============


class NameClassDataset(Dataset):
    """人名分类数据集

    把变长人名字符串转为 one-hot 编码序列，国家名转为整数标签。
    """

    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.sample_len = len(my_list_x)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        """返回 (tensor_x, tensor_y)

        tensor_x: [seq_len, n_letters] one-hot
        tensor_y: 整数类别标签
        """
        index = min(max(index, 0), self.sample_len - 1)

        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # 人名 → one-hot
        tensor_x = torch.zeros(len(x), n_letters)
        for li, letter in enumerate(x):
            letter_index = all_letters.find(letter)
            if letter_index != -1:
                tensor_x[li][letter_index] = 1

        # 国家 → 整数标签
        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)

        return tensor_x, tensor_y


# ============ DataLoader 工厂 ============


def get_dataloader(file_path='./data/name_classfication.txt', batch_size=1, shuffle=True):
    """获取数据加载器

    Args:
        file_path: 数据文件路径
        batch_size: 批大小（默认 1，RNN 处理变长序列）
        shuffle: 是否打乱

    Returns:
        DataLoader 实例
    """
    my_list_x, my_list_y = read_data(file_path)
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(name_class_dataset, batch_size=batch_size, shuffle=shuffle)
    return my_dataloader


# ============ 工具函数 ============


def lineToTensor(line):
    """将单个人名字符串转换为 one-hot 张量

    Args:
        line: 人名字符串，如 "Smith"

    Returns:
        tensor: [len(line), n_letters]
    """
    tensor_x = torch.zeros(len(line), n_letters)
    for i, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        if letter_index != -1:
            tensor_x[i][letter_index] = 1
    return tensor_x
