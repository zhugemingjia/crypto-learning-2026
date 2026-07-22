"""
人名国家分类 - 入口脚本

使用 RNN / LSTM / GRU 根据人名字母序列预测国籍。

用法:
    # 训练全部三个模型并生成对比图表
    python main.py --mode train

    # 单模型预测
    python main.py --mode predict --name "Smith"

    # 三模型对比预测
    python main.py --mode predict-all --name "Suzuki"

    # 模型结构测试
    python main.py --mode test
"""

import argparse
import os
import sys

# 让脚本能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_rnn, train_lstm, train_gru, demo_show_rnn_lstm_gru
from src.predict import predict_all_models, demo_rnn_predict, demo_lstm_predict, demo_gru_predict


def parse_args():
    parser = argparse.ArgumentParser(description="人名国家分类 - RNN/LSTM/GRU")

    parser.add_argument("--mode", type=str, default="predict-all",
                        choices=["train", "train-rnn", "train-lstm", "train-gru",
                                 "predict", "predict-all", "test"],
                        help="运行模式（默认 predict-all）")

    parser.add_argument("--name", type=str, default=None,
                        help="predict / predict-all 模式：输入的人名")

    parser.add_argument("--data", type=str, default="./data/name_classfication.txt",
                        help="数据文件路径")

    parser.add_argument("--rnn-model", type=str, default="./model/my_rnn_name_prj1.bin",
                        help="RNN 模型权重路径")
    parser.add_argument("--lstm-model", type=str, default="./model/my_lstm_name_prj1.bin",
                        help="LSTM 模型权重路径")
    parser.add_argument("--gru-model", type=str, default="./model/my_gru_name_prj1.bin",
                        help="GRU 模型权重路径")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        # 三模型全量训练 + 可视化对比
        if not os.path.exists(args.data):
            print(f"[ERROR] 数据文件不存在: {args.data}")
            sys.exit(1)
        demo_show_rnn_lstm_gru(args.data)

    elif args.mode == "train-rnn":
        if not os.path.exists(args.data):
            print(f"[ERROR] 数据文件不存在: {args.data}")
            sys.exit(1)
        train_rnn(args.data)

    elif args.mode == "train-lstm":
        if not os.path.exists(args.data):
            print(f"[ERROR] 数据文件不存在: {args.data}")
            sys.exit(1)
        train_lstm(args.data)

    elif args.mode == "train-gru":
        if not os.path.exists(args.data):
            print(f"[ERROR] 数据文件不存在: {args.data}")
            sys.exit(1)
        train_gru(args.data)

    elif args.mode == "predict":
        if args.name is None:
            print("[ERROR] predict 模式需要指定 --name <人名>")
            sys.exit(1)
        # 默认用 GRU 预测
        demo_gru_predict(args.name, args.gru_model)

    elif args.mode == "predict-all":
        if args.name is None:
            print("[ERROR] predict-all 模式需要指定 --name <人名>")
            sys.exit(1)
        predict_all_models(args.name,
                           rnn_path=args.rnn_model,
                           lstm_path=args.lstm_model,
                           gru_path=args.gru_model)

    elif args.mode == "test":
        # 模型结构验证（用随机输入跑一遍）
        import torch
        from src.model import My_RNN, My_LSTM, My_GRU
        from src.dataset import read_data, NameClassDataset, n_letters, category_num
        from torch.utils.data import DataLoader

        my_list_x, my_list_y = read_data(args.data)
        dataset = NameClassDataset(my_list_x, my_list_y)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        my_rnn = My_RNN(n_letters, 128, category_num)
        my_lstm = My_LSTM(n_letters, 128, category_num)
        my_gru = My_GRU(n_letters, 128, category_num)

        print(f"RNN 模型结构:\n{my_rnn}\n")
        print(f"LSTM 模型结构:\n{my_lstm}\n")
        print(f"GRU 模型结构:\n{my_gru}\n")

        for i, (x, y) in enumerate(loader):
            print(f"样本形状: x={x.shape}, y={y.shape}")

            h = my_rnn.init_hidden()
            out, h = my_rnn(x[0], h)
            print(f"RNN 输出形状: {out.shape}")

            h, c = my_lstm.init_hidden()
            out, h, c = my_lstm(x[0], h, c)
            print(f"LSTM 输出形状: {out.shape}")

            h = my_gru.init_hidden()
            out, h = my_gru(x[0], h)
            print(f"GRU 输出形状: {out.shape}")
            break

        print("\n[OK] 所有模型结构验证通过")


if __name__ == "__main__":
    main()
