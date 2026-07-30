"""
英译法 Seq2Seq 翻译模型 — 入口脚本

基于 GRU + Attention 的 Encoder-Decoder 框架，实现英语 → 法语的机器翻译。

用法:
    # 训练模型
    python main.py --mode train

    # 评估模型（使用预训练权重翻译自定义样本）
    python main.py --mode evaluate

    # 测试模型结构（随机输入验证维度）
    python main.py --mode test
"""

import argparse
import os
import sys

# 让脚本能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import my_getdata, get_dataloder
from src.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from src.train import train_seq2seq
from src.evaluate import seq2seq_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="英译法 Seq2Seq 翻译模型")

    parser.add_argument("--mode", type=str, default="evaluate",
                        choices=["train", "evaluate", "test"],
                        help="运行模式（默认 evaluate）")

    parser.add_argument("--data", type=str, default="./data/eng-fra-v2.txt",
                        help="数据文件路径")

    parser.add_argument("--encoder", type=str,
                        default="./model/my_encoder_rnn_1.pth",
                        help="编码器权重路径 (evaluate 模式)")

    parser.add_argument("--decoder", type=str,
                        default="./model/my_attn_decoder_rnn_1.pth",
                        help="解码器权重路径 (evaluate 模式)")

    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数 (train 模式)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 公共：加载数据 & 构建词汇表 ----
    print(f"[INFO] 当前设备: {__import__('torch').cuda.is_available() and 'cuda' or 'cpu'}")
    print(f"[INFO] 加载数据: {args.data}")

    (english_word2index, english_index2word, english_word_n,
     french_word2index, french_index2word, french_word_n,
     my_pairs) = my_getdata(args.data)

    print(f"  英语词汇表大小: {english_word_n}")
    print(f"  法语词汇表大小: {french_word_n}")
    print(f"  句子对数量:     {len(my_pairs)}")

    # ---- 模式分支 ----
    if args.mode == "train":
        import src.train as train_mod
        train_mod.epochs = args.epochs

        my_dataloader = get_dataloder(my_pairs, english_word2index, french_word2index)
        train_seq2seq(my_dataloader, english_word_n, french_word_n)

    elif args.mode == "evaluate":
        if not os.path.exists(args.encoder):
            print(f"[ERROR] 编码器模型不存在: {args.encoder}")
            sys.exit(1)
        if not os.path.exists(args.decoder):
            print(f"[ERROR] 解码器模型不存在: {args.decoder}")
            sys.exit(1)

        seq2seq_evaluate(
            english_word2index=english_word2index,
            french_index2word=french_index2word,
            english_word_n=english_word_n,
            french_word_n=french_word_n,
            encoder_path=args.encoder,
            decoder_path=args.decoder,
        )

    elif args.mode == "test":
        import torch
        from src.dataset import device

        my_dataloader = get_dataloder(my_pairs, english_word2index, french_word2index)

        # 编码器测试
        encoder = EncoderRNN(english_word_n, 256).to(device)
        print(f"\n编码器结构:\n{encoder}")

        # 解码器（无 Attention）
        decoder_no_attn = DecoderRNN(french_word_n, 256).to(device)
        print(f"基础解码器结构:\n{decoder_no_attn}")

        # 解码器（带 Attention）
        decoder_attn = AttnDecoderRNN(french_word_n, 256).to(device)
        print(f"Attention 解码器结构:\n{decoder_attn}")

        # 用一批数据验证维度
        for i, (x, y) in enumerate(my_dataloader):
            print(f"\n样本维度: x={x.shape}, y={y.shape}")

            # 编码
            h0 = encoder.init_hidden()
            enc_out, enc_hidden = encoder(x, h0)
            print(f"编码器输出: {enc_out.shape}, 隐藏状态: {enc_hidden.shape}")

            # 解码（带 Attention）
            enc_fixed = torch.zeros(10, 256, device=device)
            for idx in range(min(enc_out.shape[1], 10)):
                enc_fixed[idx] = enc_out[0, idx]

            input_token = torch.tensor([[0]], device=device)  # SOS
            out, hidden, attn = decoder_attn(input_token, enc_hidden, enc_fixed)
            print(f"Attention解码器输出: {out.shape}, attn: {attn.shape}")
            break

        print("\n[OK] 所有模型结构验证通过")


if __name__ == "__main__":
    main()
