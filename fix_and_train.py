import os
import yaml
from pathlib import Path

# ----------------- 1. 彻底清剿幽灵缓存 -----------------
print("🧹 1/4 正在清洗全盘旧缓存...")
os.system("rm -f yolo_dataset/train/*.cache")
os.system("rm -f yolo_dataset/val/*.cache")
os.system("rm -f yolo_dataset/train/labels/*.cache")
os.system("rm -f yolo_dataset/val/labels/*.cache")

# ----------------- 2. 暴力核实并锁定类别 -----------------
print("🛡️ 2/4 正在锁定类别标签，防御全 0 变异...")
yolo_dir = Path("yolo_dataset")
count_0, count_1, count_2 = 0, 0, 0
for split in ['train', 'val']:
    labels_dir = yolo_dir / split / "labels"
    if not labels_dir.exists(): continue
    for txt in labels_dir.glob("*.txt"):
        # 严格通过前缀决定生死（low -> 0, mid -> 1, high -> 2）
        if txt.name.startswith("low_"):
            target_id = "0"
            count_0 += 1
        elif txt.name.startswith("mid_"):
            target_id = "1"
            count_1 += 1
        elif txt.name.startswith("high_"):
            target_id = "2"
            count_2 += 1
        else:
            continue
            
        with open(txt, 'r') as f: lines = f.readlines()
        with open(txt, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if parts: f.write(f"{target_id} {' '.join(parts[1:])}\n")

print(f"   -> 当前硬盘装弹量: Low={count_0}, Normal={count_1}, High={count_2}")

# ----------------- 3. 定制防混淆超级参数 -----------------
print("⚙️ 3/4 正在配置[无翻转+难例挖掘]超参数...")
hyp_file = "yolov5/data/hyps/hyp.scratch-low.yaml"
# 读取原参数
with open(hyp_file, 'r', encoding='utf-8') as f:
    hyp = yaml.safe_load(f)

# 强制修正参数系列
hyp['fliplr'] = 0.0     # [防颠倒] 绝不允许左右翻转
hyp['flipud'] = 0.0     # [防颠倒] 绝不允许上下翻转
hyp['degrees'] = 5.0    # [抗畸变] 增加非常轻微的旋转容错
hyp['fl_gamma'] = 1.5   # [难例挖掘] 加大模型对界缝照片(high-normal边缘)的专注度！

# 保存为一个专属配置文件
custom_hyp_path = "yolov5/data/hyps/hyp.custom.yaml"
with open(custom_hyp_path, 'w', encoding='utf-8') as f:
    yaml.dump(hyp, f)

# ----------------- 4. 启动完美训练 -----------------
print(f"🚀 4/4 启动特训！指定超参数文件: {custom_hyp_path}")
train_cmd = (
    "python yolov5/train.py "
    "--img 640 "    # 训练可依然用 640 (防止爆显存)，你在推理时用 size=1280 放大看即可
    "--batch 16 "
    "--epochs 50 "
    "--data yolov5/dial.yaml "
    "--weights yolov5m.pt "  # 使用感受野更大、刻度识别更准的 M 级网络
    "--label-smoothing 0.1 " # 让边界判断不那么死板
    "--device 0 "
    f"--hyp {custom_hyp_path}"
)
os.system(train_cmd)