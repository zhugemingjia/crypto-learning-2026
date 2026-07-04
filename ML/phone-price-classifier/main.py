"""
手机价格分类 - 入口脚本

用法:
    # 训练
    python main.py --mode train

    # 训练（自定义参数）
    python main.py --mode train --epochs 50 --batch-size 16 --lr 1e-3

    # 评估
    python main.py --mode evaluate

    # 单条预测
    python main.py --mode predict --input data/test_sample.csv
"""

import argparse
import os
import sys
import torch

# 让脚本能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import build_datasets
from src.train import train
from src.evaluate import evaluate, predict_single, predict_from_csv


def parse_args():
    parser = argparse.ArgumentParser(description="手机价格分类 - ANN 训练/评估/预测")
    parser.add_argument("--mode", type=str, default="evaluate",
                        choices=["train", "evaluate", "predict"],
                        help="运行模式")
    parser.add_argument("--data", type=str, default="./data/手机价格预测.csv",
                        help="训练数据 CSV 路径")
    parser.add_argument("--model", type=str, default="./model/phone.pth",
                        help="模型权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--input", type=str, default=None,
                        help="predict 模式：单条样本 CSV 路径（仅特征列）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ========== 构建数据集 ==========
    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        sys.exit(1)

    print(f"📂 加载数据: {args.data}")
    train_dataset, test_dataset, input_dim, num_classes, _ = build_datasets(args.data)
    print(f"   训练集: {len(train_dataset)} 条  |  测试集: {len(test_dataset)} 条")
    print(f"   输入维度: {input_dim}  |  类别数: {num_classes}\n")

    # ========== 选择模式 ==========
    if args.mode == "train":
        print(f"🚀 开始训练  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}\n")
        train(
            train_dataset=train_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            scaler=_,  # 把 build_datasets 返回的 scaler 传进去一起保存
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.model,
        )
        print("\n📊 训练完成，开始评估...")
        evaluate(
            test_dataset=test_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            model_path=args.model,
        )

    elif args.mode == "evaluate":
        evaluate(
            test_dataset=test_dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            model_path=args.model,
        )

    elif args.mode == "predict":
        if args.input is None:
            print("❌ predict 模式需要指定 --input <csv_path>")
            sys.exit(1)
        pred = predict_from_csv(
            csv_path=args.input,
            input_dim=input_dim,
            num_classes=num_classes,
            model_path=args.model,
        )
        print(f"预测类别: {pred}  (档位 0=低 1=中低 2=中高 3=高)")


if __name__ == "__main__":
    main()
