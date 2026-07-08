"""
数据加载：分词 + 词表 + Dataset
"""

import json
import os

import jieba
import torch
from torch.utils.data import Dataset


def build_vocab(lyrics_path: str, save_path: str = None):
    """
    读取歌词文件，用 jieba 分词，构建词表
    返回: (word_to_index, index_to_word, vocab_size, corpus_idx)

    如果 save_path 不为空，词表会以 json 格式保存
    """
    unique_words, all_words = [], []

    # 1. 读所有行 + jieba 分词
    for line in open(lyrics_path, 'r', encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # 2. 词表
    word_to_index = {w: i for i, w in enumerate(unique_words)}
    index_to_word = {i: w for w, i in word_to_index.items()}

    # 3. 全文索引化（行间用 ' ' 隔开）
    corpus_idx = []
    for words in all_words:
        for word in words:
            corpus_idx.append(word_to_index[word])
        corpus_idx.append(word_to_index[' '])

    # 4. 可选保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word_to_index': word_to_index,
                'index_to_word': index_to_word,
                'vocab_size': len(unique_words),
            }, f, ensure_ascii=False, indent=2)

    return word_to_index, index_to_word, len(unique_words), corpus_idx


def load_vocab(vocab_path: str):
    """从 json 加载词表，返回 (word_to_index, index_to_word, vocab_size)"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # index_to_word 的 key 是 str（json 限制），转回 int
    index_to_word = {int(k): v for k, v in data['index_to_word'].items()}
    return data['word_to_index'], index_to_word, data['vocab_size']


class LyricsDataset(Dataset):
    """
    把长序列切成定长 (num_chars) 的滑动窗口
    每个样本: x=前 num_chars 个词索引, y=向后偏移 1 位（next-token prediction）
    """

    def __init__(self, corpus_idx, num_chars: int = 32):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.total = len(corpus_idx)
        self.number = self.total // num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(idx, 0), self.total - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start + 1:end + 1]
        return torch.tensor(x), torch.tensor(y)
