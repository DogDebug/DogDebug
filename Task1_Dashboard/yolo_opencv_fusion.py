import cv2
import numpy as np
import math
import datetime
from ultralytics import YOLO
from collections import Counter, deque

# ================= 1. YOLO 配置区 =================
MODEL_PATH = 'runs/detect/train/weights/best.pt' 
CAMERA_SOURCE = 0 # 手机USB地址，或填 0

# --- 🚀 核心防抖与输出配置区 ---
BUFFER_SIZE = 15     # 滑动窗口：记忆最近 15 帧的结果 (加大记忆容量，彻底吃掉偶尔的错误帧)
PRINT_INTERVAL = 10  # 输出频率：每 10 帧才输出一次 (让终端输出更加从容、沉稳)

# 中文映射
YOLO_CLASS_MAP = {
    0: {"status": "偏低", "result": "异常", "eng": "LOW"},
    1: {"status": "正常", "result": "正常", "eng": "NORMAL"},
    2: {"status": "偏高", "result": "异常", "eng": "HIGH"}
}
OPENCV_STATUS_MAP = {
    'LOW': {"status": "偏低", "result": "异常"},
    'NORMAL': {"status": "正常", "result": "正常"},
    'HIGH': {"status": "偏高", "result": "异常"}
}

# ================= 2. OpenCV 算法配置区 =================
COLOR_RANGES = {
    'green': ([35, 50, 50], [85, 255, 255]),
    'yellow': ([20, 50, 50], [35, 255, 255]),
    'red_low': ([0, 50, 50], [10, 255, 255]),
    'red_high': ([160, 50, 50], [180, 255, 255])
}
HOUGH_DP, HOUGH_MIN_DIST = 1.2, 50
HOUGH_PARAM1, HOUGH_PARAM2 = 100, 40
HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS = 50, 300
LINE_LENGTH_MIN, LINE_GAP_MAX = 50, 10
DIST_TO_CENTER_MAX, MIN_RADIUS_RATIO = 30, 0.5
SAMPLE_R_START, SAMPLE_R_END, SAMPLE_R_NUM = 0.70, 0.95, 8
SAMPLE_PATCH_SIZE, ANGLE_VOTE_OFFSET = 5, 2


# ================= OpenCV 核心函数 =================
def get_color_mask(hsv, color):
    if color == 'red':
        lower1, upper1 = COLOR_RANGES['red_low']
        lower2, upper2 = COLOR_RANGES['red_high']
        return cv2.bitwise_or(cv2.inRange(hsv, np.array(lower1), np.array(upper1)), 
                              cv2.inRange(hsv, np.array(lower2), np.array(upper2)))
    else:
        lower, upper = COLOR_RANGES[color]
        return cv2.inRange(hsv, np.array(lower), np.array(upper))

def detect_dial_center_and_radius(image_gray):
    blurred = cv2.GaussianBlur(image_gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, HOUGH_DP, HOUGH_MIN_DIST,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        idx = np.argmax(circles[:, 2])
        return circles[idx]
    return None

def detect_needle_angle(image_gray, center, radius):
    cx, cy = center
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=LINE_LENGTH_MIN, maxLineGap=LINE_GAP_MAX)
    if lines is None: return None
    best_line, best_score = None, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist1, dist2 = math.hypot(x1 - cx, y1 - cy), math.hypot(x2 - cx, y2 - cy)
        if (dist1 < DIST_TO_CENTER_MAX and dist2 > radius * MIN_RADIUS_RATIO) or \
           (dist2 < DIST_TO_CENTER_MAX and dist1 > radius * MIN_RADIUS_RATIO):
            tip_x, tip_y = (x2, y2) if dist2 > dist1 else (x1, y1)
            score = math.hypot(tip_x - cx, tip_y - cy)
            if score > best_score:
                best_score = score
                best_line = (math.atan2(tip_y - cy, tip_x - cx), (tip_x, tip_y))
    return best_line

def sample_color_at_angle(hsv, center, radius, angle_rad, offset_deg=0):
    cx, cy = center
    angle = angle_rad + math.radians(offset_deg)
    color_votes = {'green': 0, 'yellow': 0, 'red': 0}
    radii = np.linspace(SAMPLE_R_START * radius, SAMPLE_R_END * radius, SAMPLE_R_NUM)
    for r in radii:
        x, y = int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))
        if 0 <= x - SAMPLE_PATCH_SIZE//2 and x + SAMPLE_PATCH_SIZE//2 < hsv.shape[1] and \
           0 <= y - SAMPLE_PATCH_SIZE//2 and y + SAMPLE_PATCH_SIZE//2 < hsv.shape[0]:
            patch = hsv[y-SAMPLE_PATCH_SIZE//2:y+SAMPLE_PATCH_SIZE//2+1, x-SAMPLE_PATCH_SIZE//2:x+SAMPLE_PATCH_SIZE//2+1]
            for color_name in ['green', 'yellow', 'red']:
                if cv2.countNonZero(get_color_mask(patch, color_name)) > 10:
                    color_votes[color_name] += 1
                    break
    if max(color_votes.values()) == 0: return None
    primary = max(color_votes, key=color_votes.get)
    return 'LOW' if primary == 'yellow' else 'NORMAL' if primary == 'green' else 'HIGH'

# ================= 融合主程序 =================
def main():
    print("🚀 正在加载 YOLOv8 模型...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    print("\n" + "="*60)
    print("🎥 YOLO + OpenCV 双轨高精度预测已启动！(开启强力防抖) 按 'q' 退出。")
    print("="*60 + "\n")

    frame_count = 0
    # ⭐️ 初始化防抖缓冲队列
    status_buffer = deque(maxlen=BUFFER_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        results = model(frame, verbose=False)
        detected = False
        raw_status = None
        
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                best_box = max(boxes, key=lambda x: x.conf[0].item())
                if best_box.conf[0].item() > 0.5:
                    detected = True
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    yolo_cls_id = int(best_box.cls[0].item())
                    
                    h, w = frame.shape[:2]
                    pad = 10
                    x1_pad, y1_pad = max(0, x1-pad), max(0, y1-pad)
                    x2_pad, y2_pad = min(w, x2+pad), min(h, y2+pad)
                    
                    roi = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    algorithm_used = "OpenCV"

                    # OpenCV 算法流程
                    dial_info = detect_dial_center_and_radius(gray)
                    if dial_info is not None:
                        center, radius = (dial_info[0], dial_info[1]), dial_info[2]
                        needle = detect_needle_angle(gray, center, radius)
                        
                        if needle is not None:
                            angle_rad, tip = needle
                            status_votes =[]
                            for offset in range(-ANGLE_VOTE_OFFSET, ANGLE_VOTE_OFFSET+1, 2):
                                st = sample_color_at_angle(hsv, center, radius, angle_rad, offset)
                                if st: status_votes.append(st)
                            if status_votes:
                                raw_status = max(set(status_votes), key=status_votes.count)
                            
                            # 绘制 OpenCV 识别标志
                            cv2.circle(roi, center, radius, (0, 255, 0), 2)
                            cv2.line(roi, center, (int(tip[0]), int(tip[1])), (255, 0, 0), 2)

                    # 兜底机制：无缝切换回 YOLO
                    if raw_status is None:
                        raw_status = YOLO_CLASS_MAP[yolo_cls_id]['eng']
                        algorithm_used = "YOLO_Fallback"

                    # 将处理完的 ROI 贴回
                    frame[y1_pad:y2_pad, x1_pad:x2_pad] = roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # ================= 强力防抖与输出控制 =================
        if raw_status is not None:
            # 1. 压入历史队列
            status_buffer.append(raw_status)
            
            # 2. 多数投票：获取最近 15 帧里出现次数最多的状态
            smoothed_status = Counter(status_buffer).most_common(1)[0][0]
            out_info = OPENCV_STATUS_MAP[smoothed_status]

            # 3. 画面显示：显示经过平滑处理后的结果
            cv2.putText(frame, f"Status: {out_info['status']} (Smooth)", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 4. 终端输出：每隔 10 帧才打印一次，保持沉稳
            if frame_count % PRINT_INTERVAL == 0:
                now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{now_str}] 👉 识别结果：{out_info['status']} | 状态：{out_info['result']} | 双轨防抖融合")
        else:
            # 如果画面里彻底没看到表盘，清空一下缓冲池，防止旧记忆干扰下一次
            if len(status_buffer) > 0:
                status_buffer.clear()

        cv2.imshow("Dashboard Detection - Ultimate Edition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()