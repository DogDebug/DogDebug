# data/split_cls_dataset.py
import shutil
import random
import argparse
from pathlib import Path

def split_cls(src: str, dst: str, ratio: float = 0.8, seed: int = 42):
    random.seed(seed)
    src_dir = Path(src)
    dst_dir = Path(dst)
    
    # 遍历源目录下的各个类别文件夹 (low, normal, high)
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        imgs = list(class_dir.glob("*.jpg"))
        random.shuffle(imgs)
        
        split_idx = int(len(imgs) * ratio)
        subsets = [("train", imgs[:split_idx]), ("val", imgs[split_idx:])]
        
        for split_name, files in subsets:
            # 创建对应的输出目录: dst/train/low/ 等
            out_class_dir = dst_dir / split_name / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in files:
                shutil.copy(img_path, out_class_dir / img_path.name)
                
    print(f"分类数据集划分完毕 → {dst}")

if __name__ == "__main__":
    split_cls("data/raw_frames", "data", ratio=0.8, seed=42)