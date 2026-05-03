import os
import cv2
import torch
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# --------- ⚠️ 请填写能找准框的最强权重 ⚠️ ---------
WEIGHT_PATH = "yolov5/runs/train/exp10/weights/best.pt" 
# ---------------------------------------------------

FRAMES_DIR = Path("data/frames")
YOLO_DIR = Path("yolo_dataset")
CLASS_MAPPING = {"gauge_low": 0, "gauge_mid": 1, "gauge_high": 2}

def main():
    # 1. 净化原有的 yolo_dataset
    print("🧹 1/3 正在清理旧的 yolo_dataset...")
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    
    for split in ['train', 'val']:
        (YOLO_DIR / split / 'images').mkdir(parents=True)
        (YOLO_DIR / split / 'labels').mkdir(parents=True)

    # 2. 扫描文件并打乱分发
    print("\n🔍 2/3 正在扫描需要打标的图像...")
    all_files = [] # 格式: (原图路径, target_id, "low_frame_...")
    
    for folder_name, target_id in CLASS_MAPPING.items():
        folder_path = FRAMES_DIR / folder_name
        if not folder_path.exists(): continue
        
        for img_path in folder_path.glob("*.jpg"):
            # 取个好听的新名字
            state_name = folder_name.split("_")[-1] # "low", "mid", "high"
            new_name = f"{state_name}_{img_path.name}"
            all_files.append((img_path, target_id, new_name))
            
    print(f"共发现 {len(all_files)} 张图片，开始按 8:2 切分并利用 AI 自动打标...")
    
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    splits = {"train": all_files[:split_idx], "val": all_files[split_idx:]}

    # 3. 启动模型进行自动打标写入
    print("\n🤖 3/3 启动 YOLO 自动打标机...")
    model = torch.hub.load('yolov5', 'custom', path=WEIGHT_PATH, source='local')
    model.conf = 0.4 # 只要找出框就行
    
    for split_name, files in splits.items():
        for img_path, target_id, new_name in tqdm(files, desc=f"处理 {split_name} 集"):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            H, W = img.shape[:2]
            results = model(img, size=1280)
            detections = results.xyxy[0].cpu().numpy()
            
            if len(detections) > 0:
                # 拿概率最好的那一个框
                x1, y1, x2, y2, conf, _ = detections[0]
                
                # 转换坐标
                w = x2 - x1
                h = y2 - y1
                x_c = x1 + w/2
                y_c = y1 + h/2
                norm_box = (x_c/W, y_c/H, w/W, h/H)
                
                # 写入图像
                shutil.copy(img_path, YOLO_DIR / split_name / "images" / new_name)
                
                # 写入标签（第一位填强制分配的 target_id！！！）
                txt_name = Path(new_name).with_suffix(".txt")
                txt_dest = YOLO_DIR / split_name / "labels" / txt_name
                with open(txt_dest, 'w') as f:
                    f.write(f"{target_id} {norm_box[0]:.6f} {norm_box[1]:.6f} {norm_box[2]:.6f} {norm_box[3]:.6f}\n")

    print("\n✅ 所有工作圆满完成！你可以直接执行 fix_and_train.py 开始大模型训练了！")

if __name__ == "__main__":
    main()