import cv2
import torch
import numpy as np
import time

def test_pt_file():
    # --------- ⚠️ 请务必检查这里是不是最新的权重 ⚠️ ---------
    yolo_weight = "yolov5/runs/train/exp10/weights/best.pt" 
    
    print(f"1. 正在加载本地权重：{yolo_weight}")
    yolo_model = torch.hub.load('yolov5', 'custom', path=yolo_weight, source='local')
    
    # 探针 1：让模型自己交代它记得几个类
    print("\n--- 🕵️ 模型内部记忆探测 ---")
    print(f"这个模型以为自己有 {yolo_model.names} 个类。")
    if len(yolo_model.names) == 1:
        print("❌ 破案了！你的最新模型在训练时根本没吃到 dial.yaml 里的 3 个类，它内部只焊死了 1 个类（就是 0）。")
    else:
        print("✅ 模型内部确实有多个类。")
        
    yolo_model.conf = 0.5  
    yolo_model.iou = 0.3

    cap = cv2.VideoCapture(0)
    print("\n2. 开启摄像头... 请把表盘拿过来")

    last_print_time = 0  # 记录上次打印的时间

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame, size=1280)
        
        # 探针 2：每隔 1 秒才真正在终端打印一次，防止疯狂刷屏
        current_time = time.time()
        detections = results.xyxy[0].cpu().numpy()
        
        if len(detections) > 0 and (current_time - last_print_time > 1.0):
            # 获取画面上最高置信度的目标（通常在第一个）
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # 尝试获取友好的类名，如果没有则用 ID
                class_name = yolo_model.names[int(cls)] if isinstance(yolo_model.names, (dict, list)) else f"ID {int(cls)}"
                print(f"👉 [时间:{time.strftime('%H:%M:%S')}] 发现目标！类别: {class_name}，置信度: {conf:.2f}")
            
            # 更新上次打印时间
            last_print_time = current_time

        annotated_frame = results.render()[0].copy()
        cv2.imshow("Debug Vision", annotated_frame)
        
        # 将 60 改为 1，让摄像头画面更流畅
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pt_file()