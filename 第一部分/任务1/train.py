# train.py
from ultralytics import YOLO

def main():
    # 1. 加载官方预训练的轻量级 Nano 模型
    model = YOLO('yolov8n.pt')
    
    # 2. 开始训练仪表盘数据集 (这里的参数就是你当时在命令行敲的)
    print("开始训练仪表盘模型...")
    results = model.train(
        data='dial.yaml',
        epochs=50,
        imgsz=640,
        degrees=180.0, 
        device='0'   # 使用 GPU
    )
    print("训练完成，权重保存在 runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    main()