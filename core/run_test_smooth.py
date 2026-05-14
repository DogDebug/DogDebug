import cv2
import datetime  # ⭐️ 新增：引入时间库
from ultralytics import YOLO
from collections import deque, Counter

# ================= 配置区 =================
MODEL_PATH = 'runs/detect/train/weights/best.pt' 

# 你的手机 USB 视频流地址
CAMERA_SOURCE = "http://10.184.190.147:8080/video" 

# --- 🚀 核心防抖与输出配置区 ---
BUFFER_SIZE = 10     # 滑动窗口大小：记忆最近 10 帧的预测结果
PRINT_INTERVAL = 5   # 输出频率控制：每隔 5 帧在终端打印一次结果
# ==========================================

# 类别映射字典
CLASS_MAP = {
    0: {"status": "偏低", "result": "异常"},
    1: {"status": "正常", "result": "正常"},
    2: {"status": "偏高", "result": "异常"},
    -1: {"status": "未识别", "result": "失败"}
}

def main():
    print("🚀 正在加载 YOLOv8 模型...")
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功！打开摄像头...")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    # 强制 OpenCV 读取最新帧，拒绝排队延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查网络地址或设备！")
        return

    print("\n" + "="*60)
    print("🎥 实时预测已启动！(开启防抖与时间戳印证) 按 'q' 退出。")
    print("="*60 + "\n")

    pred_buffer = deque(maxlen=BUFFER_SIZE)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法获取画面！")
            break
            
        frame_count += 1

        # 使用 YOLO 进行预测
        results = model(frame, verbose=False)

        current_pred = -1 

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                best_box = max(boxes, key=lambda x: x.conf[0].item())
                if best_box.conf[0].item() > 0.5:
                    current_pred = int(best_box.cls[0].item())
                    
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 核心逻辑 1：压入防抖队列
        pred_buffer.append(current_pred)

        # 核心逻辑 2：少数服从多数投票
        if len(pred_buffer) > 0:
            smoothed_pred = Counter(pred_buffer).most_common(1)[0][0]
        else:
            smoothed_pred = -1

        # ⭐️ 核心逻辑 3：带实时时间戳的终端打印
        if frame_count % PRINT_INTERVAL == 0 and smoothed_pred != -1:
            info = CLASS_MAP.get(smoothed_pred)
            # 获取当前时间，格式为 时:分:秒.毫秒 (保留前3位毫秒)
            now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            print(f"[{now_str}] 👉 识别结果：{info['status']} | 状态：{info['result']} (平滑输出)")

        # 画面显示状态
        if smoothed_pred != -1:
            smooth_info = CLASS_MAP.get(smoothed_pred)
            label_text = f"Status: {smooth_info['status']}"
            cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Dashboard Detection - LIVE", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()