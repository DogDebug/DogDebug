import cv2
import torch
import numpy as np
import time
import collections
import warnings

# 屏蔽 YOLOv5 内部的 AMP 弃用警告，保持终端干净
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

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
        
    yolo_model.conf = 0.4  
    yolo_model.iou = 0.3

    cap = cv2.VideoCapture(0)
    print("\n2. 开启摄像头... 请把表盘拿过来")

    # ---------- 中文状态映射 ----------
    # 根据你的模型实际输出的类别名修改键
    state_map = {
        'low':    '偏低（异常）',
        'normal': '正常',
        'high':   '偏高（异常）'
    }
    # 如果模型类别是数字，改用数字映射：
    # state_map = {0: '偏低（异常）', 1: '正常', 2: '偏高（异常）'}

    # 防抖缓冲区：保存最近 15 帧的预测类别（None 表示无检测）
    buffer_size = 15
    state_buffer = collections.deque(maxlen=buffer_size)
    last_stable_state = None  # 上一稳定状态

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, size=1600)
        detections = results.xyxy[0].cpu().numpy()

        # 当前帧的预测类别（None 表示没有检测到目标）
        current_class = None

        if len(detections) > 0:
            # 取置信度最高的一个检测结果
            best_det = detections[np.argmax(detections[:, 4])]
            x1, y1, x2, y2, conf, cls = best_det
            class_id = int(cls)

            # 获取类别名
            if isinstance(yolo_model.names, (dict, list)):
                class_name = yolo_model.names[class_id]
            else:
                class_name = str(class_id)

            current_class = class_name

        # 将当前帧结果加入缓冲区
        state_buffer.append(current_class)

        # 投票得出稳定状态
        # 只考虑缓冲区中有效的预测（非 None）
        valid_preds = [p for p in state_buffer if p is not None]
        stable_state = None

        if valid_preds:
            # 找出出现次数最多的类别
            most_common = max(set(valid_preds), key=valid_preds.count)
            # 只有当最常见类别在缓冲区中的占比超过阈值时才确认状态
            if valid_preds.count(most_common) / len(state_buffer) >= 0.6:
                stable_state = most_common
            else:
                # 占比不够，保持上一稳定状态
                stable_state = last_stable_state
        else:
            # 缓冲区全空，表示长时间没有检测到目标
            stable_state = None

        # 状态变化检测与中文输出
        if stable_state != last_stable_state:
            if stable_state is None:
                print("❌ 仪表盘消失！")
            else:
                chinese_status = state_map.get(stable_state, f"未知({stable_state})")
                if last_stable_state is None:
                    print(f"👉 仪表盘状态：{chinese_status}")
                else:
                    old_chinese = state_map.get(last_stable_state, last_stable_state)
                    print(f"⚠️ 状态变化：{old_chinese} → {chinese_status}")
            last_stable_state = stable_state

        # 显示画面（保留 YOLO 渲染框）
        annotated_frame = results.render()[0].copy()
        cv2.imshow("Debug Vision", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pt_file()