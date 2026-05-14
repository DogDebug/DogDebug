import os
import json
import random
import shutil
from pathlib import Path

# ================= 配置区 =================
# 你存放原始图片和json的文件夹路径
RAW_DATA_DIR = 'data/frames' 
# 生成的YOLO标准数据集的目标文件夹
OUTPUT_DIR = 'dial_dataset' 
# 训练集占比，默认80%训练，20%验证
TRAIN_RATIO = 0.8 

# 标签映射字典 (注意大小写，严格和你标的一致！)
CLASSES = {
    "Low": 0,
    "Mid": 1,
    "High": 2
}
# ==========================================

def create_yolo_dirs():
    """创建YOLO标准目录结构"""
    dirs =[
        f'{OUTPUT_DIR}/images/train', f'{OUTPUT_DIR}/images/val',
        f'{OUTPUT_DIR}/labels/train', f'{OUTPUT_DIR}/labels/val'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ YOLO目录结构已创建在 '{OUTPUT_DIR}' 中。")

def convert_labelme_json(json_file, img_w, img_h):
    """提取JSON中的坐标并转为YOLO格式"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    yolo_labels = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASSES:
            continue
        class_id = CLASSES[label]
        
        # 获取矩形框(或多边形)的外接矩形极值
        points = shape['points']
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # 转为 YOLO 归一化中心点坐标 (x_center, y_center, width, height)
        dw = 1.0 / img_w
        dh = 1.0 / img_h
        x_center = ((xmin + xmax) / 2.0) * dw
        y_center = ((ymin + ymax) / 2.0) * dh
        w = (xmax - xmin) * dw
        h = (ymax - ymin) * dh
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    return yolo_labels

def main():
    create_yolo_dirs()
    
    # 获取所有json文件
    json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    if not json_files:
        print("❌ 在 raw_data 文件夹中没有找到 .json 文件，请检查！")
        return
        
    print(f"🔍 找到 {len(json_files)} 个标注文件，开始处理...")
    
    # 随机打乱以划分数据集
    random.seed(42) # 固定随机种子保证每次划分一致
    random.shuffle(json_files)
    
    train_size = int(len(json_files) * TRAIN_RATIO)
    train_files = json_files[:train_size]
    
    success_count = 0
    for idx, json_name in enumerate(json_files):
        base_name = os.path.splitext(json_name)[0]
        json_path = os.path.join(RAW_DATA_DIR, json_name)
        
        # 寻找对应的图片 (支持常见格式)
        img_path = None
        for ext in['.jpg', '.JPG', '.png', '.PNG', '.jpeg']:
            temp_path = os.path.join(RAW_DATA_DIR, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if not img_path:
            print(f"⚠️ 找不到 {json_name} 对应的图片，已跳过。")
            continue
            
        # 读取JSON获取图片尺寸
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_w = data['imageWidth']
        img_h = data['imageHeight']
        
        # 转换坐标
        yolo_labels = convert_labelme_json(json_path, img_w, img_h)
        
        # 判断放入训练集还是验证集
        is_train = json_name in train_files
        subset = 'train' if is_train else 'val'
        
        # 拷贝图片 (为了安全，使用copy而不是move，保留你的原始数据)
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'images', subset, os.path.basename(img_path)))
        
        # 写入TXT标签
        txt_path = os.path.join(OUTPUT_DIR, 'labels', subset, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.writelines(yolo_labels)
            
        success_count += 1

    print(f"\n🎉 大功告成！成功转换并划分了 {success_count} 组数据。")
    print(f"📁 你的 YOLO 数据集已准备好，路径为: ./dial_dataset")

if __name__ == '__main__':
    main()