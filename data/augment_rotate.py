# data/augment_rotate.py

"""对图片和 YOLO 标注框做随机旋转增强。"""
from __future__ import annotations
import cv2
import numpy as np
import random
import argparse
from pathlib import Path
from typing import List, Tuple

Box = Tuple[float, float, float, float]  # xc, yc, w, h 归一化

def rotate_img_and_boxes(img: np.ndarray, boxes: List[Box], angle: float):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), borderValue=(0,0,0))
    new_boxes = []
    for xc, yc, bw, bh in boxes:
        px, py = xc * w, yc * h
        vec = M[:, :2] @ [px, py] + M[:, 2]
        nx, ny = vec[0] / w, vec[1] / h
        nx = np.clip(nx, 0, 1)
        ny = np.clip(ny, 0, 1)
        new_boxes.append((nx, ny, bw, bh))
    return rot, new_boxes

def read_boxes(txt_path: Path) -> List[Box]:
    boxes = []
    if txt_path.exists():
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, w, h = map(float, parts)
                    boxes.append((xc, yc, w, h))
    return boxes

def save_boxes(txt_path: Path, boxes: List[Box], cls_id: int = 0):
    with open(txt_path, 'w') as f:
        for xc, yc, w, h in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def augment(src: str, dst: str, n: int = 5):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for img_path in src.rglob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = read_boxes(img_path.with_suffix('.txt'))
        # 保存原图
        cv2.imwrite(str(dst / f"{img_path.stem}_0.jpg"), img)
        save_boxes(dst / f"{img_path.stem}_0.txt", boxes)
        for i in range(n):
            angle = random.uniform(-180, 180)
            rot, rot_boxes = rotate_img_and_boxes(img, boxes, angle)
            cv2.imwrite(str(dst / f"{img_path.stem}_aug{i}.jpg"), rot)
            save_boxes(dst / f"{img_path.stem}_aug{i}.txt", rot_boxes)
    print(f"增强完成：{dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()
    augment(args.src, args.dst, args.num)