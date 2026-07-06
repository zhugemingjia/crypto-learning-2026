"""
CIFAR-10 图像分类 - 入口脚本

用法:
    # 训练（首次会从 torchvision 在线下载 CIFAR-10 ~186MB 到 ./data/）
    python main.py --mode train

    # 训练（自定义参数）
    python main.py --mode train --epochs 50 --batch-size 64 --lr 1e-3

    # 评估（用 best 权重跑测试集）
    python main.py --mode evaluate

    # 单张图片预测
    python main.py --mode predict --input data/img.jpg
"""

import argparse
import os
import sys

# 让脚本能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import build_datasets, get_class_names
from src.train import train
from src.evaluate import evaluate, predict_single


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 图像分类 - 训练/评估/预测")
    parser.add_argument("--mode", type=str, default="evaluate",
                        choices=["train", "evaluate", "predict"],
                        help="运行模式")
    parser.add_argument("--data", type=str, default="./data",
                        help="CIFAR-10 数据根目录（首次训练会自动下载到这里）")
    parser.add_argument("--model", type=str, default="./data/image_model_best.pth",
                        help="模型权重路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--input", type=str, default=None,
                        help="predict 模式：单张图片路径")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        print(f"📂 加载 CIFAR-10 数据（首次运行会自动下载 ~186MB）...")
        train_dataset, test_dataset = build_datasets(data_root=args.data, download=True)
        print(f"   训练集: {len(train_dataset)} 张  |  测试集: {len(test_dataset)} 张")
        print(f"   类别: {get_class_names()}\n")

        print(f"🚀 开始训练  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}\n")
        train(
            train_dataset=train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            save_dir=args.data,
        )
        print("\n📊 训练完成，开始评估...")
        evaluate(
            test_dataset=test_dataset,
            model_path=args.model,
            num_workers=args.num_workers,
        )

    elif args.mode == "evaluate":
        print(f"📂 加载 CIFAR-10 测试集...")
        _, test_dataset = build_datasets(data_root=args.data, download=True)
        print(f"   测试集: {len(test_dataset)} 张\n")
        evaluate(
            test_dataset=test_dataset,
            model_path=args.model,
            num_workers=args.num_workers,
        )

    elif args.mode == "predict":
        if args.input is None:
            print("❌ predict 模式需要指定 --input <image_path>")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"❌ 图片不存在: {args.input}")
            sys.exit(1)
        idx, name, probs = predict_single(image_path=args.input, model_path=args.model)
        print(f"\n预测结果: 类别 {idx} → {name}")
        print(f"Top-3 概率:")
        top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        for cls, p in top3:
            print(f"  {cls:<12s} {p:.4f}")


if __name__ == "__main__":
    main()
