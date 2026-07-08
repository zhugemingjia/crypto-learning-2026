"""
歌词生成（推理）
"""

import os

import torch

from .dataset import load_vocab
from .model import TextGenerator


def generate(
    start_word: str = "爱",
    length: int = 50,
    temperature: float = 1.0,
    model_path: str = "./data/lyric_model.pth",
    vocab_path: str = "./data/vocab.json",
    device: str = None,
) -> str:
    """
    从给定起始词生成歌词

    Args:
        start_word: 起始词（必须在词表中）
        length: 生成的总长度（含起始词）
        temperature: 采样温度
            - 1.0 = 标准概率分布
            - <1.0 = 更确定（接近 argmax）
            - >1.0 = 更随机（更发散）
    Returns:
        生成的歌词字符串
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 1. 加载词表 + 模型
    word_to_index, index_to_word, vocab_size = load_vocab(vocab_path)
    model = TextGenerator(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if start_word not in word_to_index:
        # 取最接近的词（这里是 fallback 提示）
        raise ValueError(f"起始词 '{start_word}' 不在词表中。常用起始词: 爱/我/你/走/梦")

    # 2. 初始化
    word_idx = word_to_index[start_word]
    hidden = model.init_hidden(1, device=device)
    seq = [start_word]

    # 3. 逐字生成
    for _ in range(length - 1):
        out, hidden = model(torch.tensor([[word_idx]]).to(device), hidden)
        # 温度采样：logits / T → softmax → multinomial
        logits = out.squeeze(0) / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        word_idx = int(torch.multinomial(probs, num_samples=1).item())
        seq.append(index_to_word[word_idx])

    return "".join(seq)
