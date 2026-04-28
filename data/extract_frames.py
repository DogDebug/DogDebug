# DogDebug/data/extract_frames.py

"""从视频按时间间隔抽帧。"""
from __future__ import annotations
import cv2
import argparse
from pathlib import Path

def extract(video: str, out_dir: str, interval: float = 0.3) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * interval))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(str(out / f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"共抽取 {saved} 帧 → {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=0.3)
    args = parser.parse_args()
    extract(args.video, args.out, args.interval)