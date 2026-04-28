# data/split_dataset.py

"""将增强后图片随机划分为 train/val。"""
from __future__ import annotations
import shutil
import random
import argparse
from pathlib import Path

def split(src: str, dst: str, ratio: float = 0.8, seed: int = 42):
    random.seed(seed)
    src = Path(src)
    dst = Path(dst)
    imgs = list(src.rglob("*.jpg"))
    random.shuffle(imgs)
    split_idx = int(len(imgs) * ratio)
    subsets = [("train", imgs[:split_idx]), ("val", imgs[split_idx:])]
    for name, files in subsets:
        img_dir = dst / name / "images"
        lbl_dir = dst / name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            shutil.copy(img_path, img_dir / img_path.name)
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                shutil.copy(label_path, lbl_dir / label_path.name)
    print(f"划分完毕 → {dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split(args.src, args.dst, args.ratio, args.seed)