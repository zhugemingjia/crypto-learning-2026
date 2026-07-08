"""
周杰伦歌词生成器 - 入口脚本

用法:
    # 训练（首次会自动 jieba 分词、构建词表；歌词文件自行准备）
    python main.py --mode train

    # 训练（自定义参数）
    python main.py --mode train --epochs 20 --batch-size 16 --seq-len 64

    # 生成歌词（默认从"爱"开始 50 字）
    python main.py --mode generate

    # 自定义生成
    python main.py --mode generate --start 我 --length 100 --temperature 0.8
"""

import argparse
import os
import sys

# 让脚本能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train
from src.generate import generate


def parse_args():
    parser = argparse.ArgumentParser(description="周杰伦歌词生成器")
    parser.add_argument("--mode", type=str, default="generate",
                        choices=["train", "generate"],
                        help="运行模式")
    parser.add_argument("--data", type=str, default="./data/jaychou_lyrics.txt",
                        help="训练用歌词文件")
    parser.add_argument("--model", type=str, default="./data/lyric_model.pth",
                        help="模型权重路径")
    parser.add_argument("--vocab", type=str, default="./data/vocab.json",
                        help="词表路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=5, help="批大小")
    parser.add_argument("--seq-len", type=int, default=32, help="序列长度")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--start", type=str, default="爱", help="起始词")
    parser.add_argument("--length", type=int, default=50, help="生成长度")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度（<1 更确定，>1 更随机）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        if not os.path.exists(args.data):
            print(f"❌ 歌词文件不存在: {args.data}")
            print("请把周杰伦歌词（UTF-8 编码，每行一句）放到该路径下")
            print("或使用 --data 指定你自己的歌词文件")
            sys.exit(1)

        print(f"📂 加载数据: {args.data}")
        print(f"🚀 开始训练  epochs={args.epochs}  batch_size={args.batch_size}  seq_len={args.seq_len}  lr={args.lr}\n")
        train(
            lyrics_path=args.data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=os.path.dirname(args.model),
        )

    elif args.mode == "generate":
        if not os.path.exists(args.model):
            print(f"❌ 模型权重不存在: {args.model}")
            print("请先跑 --mode train 训练模型")
            sys.exit(1)
        if not os.path.exists(args.vocab):
            print(f"❌ 词表不存在: {args.vocab}")
            sys.exit(1)

        print(f"🎤 起始词: {args.start}  长度: {args.length}  温度: {args.temperature}\n")
        result = generate(
            start_word=args.start,
            length=args.length,
            temperature=args.temperature,
            model_path=args.model,
            vocab_path=args.vocab,
        )
        print(f"\n生成的歌词:\n{result}")


if __name__ == "__main__":
    main()
