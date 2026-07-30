"""
数据预处理模块：文本清洗、词汇表构建、DataSet / DataLoader 封装
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 全局常量
# ============================================================
SOS_token = 0   # 起始标志
EOS_token = 1   # 结束标志
MAX_LENGTH = 10  # 最大句子长度

# 设备选择（在模块导入时执行一次）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. 字符串规范化
# ============================================================
def normalizeString(s: str) -> str:
    """
    字符串规范化函数
    :param s: 需要处理的字符串
    :return:   处理后的字符串
    """
    # 转化为小写，去除首尾空白
    s = s.lower().strip()
    # 在 .!? 前加一个空格
    s = re.sub(r'([.!?])', r' \1', s)
    # 过滤非标准字符：保留大小写字母和基本标点
    s = re.sub('[^a-zA-Z.!?]+', ' ', s)
    return s


# ============================================================
# 2. 读取数据 & 构建词汇表
# ============================================================
def my_getdata(data_path: str = './data/eng-fra-v2.txt'):
    """
    读取原始数据、清洗文本、构建英语/法语词汇表

    :param data_path: 数据文件路径
    :return: (english_word2index, english_index2word, english_word_n,
              french_word2index, french_index2word, french_word_n, my_pairs)
    """
    # 1. 读取原始文件
    with open(data_path, 'r', encoding='utf-8') as src_f:
        lines = src_f.readlines()

    # 2. 清洗文本并构造双语句子对
    my_pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]

    # 3. 初始化英语词汇表（预留 SOS / EOS）
    english_word2index = {'SOS': SOS_token, 'EOS': EOS_token}
    english_word_n = 2

    # 4. 初始化法语词汇表
    french_word2index = {'SOS': SOS_token, 'EOS': EOS_token}
    french_word_n = 2

    # 5. 遍历所有句子对，构建词汇表
    for pair in my_pairs:
        for word in pair[0].split():
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        for word in pair[1].split():
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 6. 构建反向映射（索引 → 单词）
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}

    return (english_word2index, english_index2word, english_word_n,
            french_word2index, french_index2word, french_word_n, my_pairs)


# ============================================================
# 3. Dataset 类
# ============================================================
class MyPairsDataset(Dataset):
    """英-法 双语句子对 Dataset"""

    def __init__(self, my_pairs, english_word2index, french_word2index, device=device):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)
        self.english_word2index = english_word2index
        self.french_word2index = french_word2index
        self.device = device

    def __len__(self):
        return len(self.my_pairs)

    def __getitem__(self, index):
        # 修正索引范围
        index = min(max(index, 0), self.sample_len - 1)

        x = self.my_pairs[index][0]  # 英语句子
        y = self.my_pairs[index][1]  # 法语句子

        # 英语句子 → 索引序列
        x = [self.english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=self.device)

        # 法语句子 → 索引序列（解码器输入以 SOS 开头，EOS 结尾）
        y = [self.french_word2index[word] for word in y.split()]
        y.insert(0, SOS_token)
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=self.device)

        return tensor_x, tensor_y


# ============================================================
# 4. DataLoader 工厂
# ============================================================
def get_dataloder(my_pairs, english_word2index=None, french_word2index=None,
                  batch_size=1, shuffle=True):
    """
    创建 DataLoader 对象

    :param my_pairs: 双语句子对列表
    :param batch_size: 批次大小（变长序列建议 = 1）
    :param shuffle: 是否打乱
    :return: DataLoader 实例
    """
    my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)
    return my_dataloader
