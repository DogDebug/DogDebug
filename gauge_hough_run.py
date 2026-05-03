import cv2
import torch
import numpy as np
import math
import collections
import warnings

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# ======================== 配置区 ========================
# 1. YOLO模型（设为 None 则用纯霍夫找圆，推荐使用 YOLO 引导）
YOLO_WEIGHT_PATH = "yolov5/runs/train/exp10/weights/best.pt"
# YOLO_WEIGHT_PATH = None

# 2. 视频源（0 表示摄像头）
INPUT_SOURCE = 0

# 3. 摄像头镜像翻转（如果画面左右相反，设为 True）
FLIP_HORIZONTAL = True

# 4. 霍夫圆检测参数（用于 ROI 内或全图）
HOUGH_CIRCLE_DP = 1.2
HOUGH_CIRCLE_MIN_DIST = 50
HOUGH_CIRCLE_PARAM1 = 100
HOUGH_CIRCLE_PARAM2 = 35      # 适当降低以检测更多圆
HOUGH_CIRCLE_MIN_RADIUS = 30
HOUGH_CIRCLE_MAX_RADIUS = 200

# 5. 霍夫直线参数
HOUGH_LINE_THRESHOLD = 50
HOUGH_LINE_MIN_LINE_LENGTH = 30
HOUGH_LINE_MAX_LINE_GAP = 10

# 6. 角度 → 状态多区间映射（请根据实际测量修改）
ANGLE_RANGES = {
    "偏低（异常）": [(143, 219)],
    "正常":        [(219, 324)],
    "偏高（异常）": [(0, 44), (324, 360)]
}

# 7. 防抖参数
BUFFER_SIZE = 15        # 状态投票窗口大小
VOTE_THRESHOLD = 0.6    # 最常见状态占比阈值

# 8. 圆心时序平滑帧数（0 表示不平滑）
SMOOTH_FRAMES = 5
# ========================================================

def load_yolo_model(weight_path):
    if weight_path is None:
        return None
    print(f"加载 YOLO 模型: {weight_path}")
    model = torch.hub.load('yolov5', 'custom', path=weight_path, source='local')
    model.conf = 0.5
    model.iou = 0.3
    return model

def detect_gauge_with_yolo(model, frame):
    results = model(frame, size=1280)
    detections = results.xyxy[0].cpu().numpy()
    if len(detections) == 0:
        return None
    best = detections[np.argmax(detections[:, 4])]
    x1, y1, x2, y2, conf, cls = best
    return int(x1), int(y1), int(x2), int(y2)

def detect_gauge_hough_full(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (9,9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=HOUGH_CIRCLE_DP,
                               minDist=HOUGH_CIRCLE_MIN_DIST,
                               param1=HOUGH_CIRCLE_PARAM1,
                               param2=HOUGH_CIRCLE_PARAM2,
                               minRadius=HOUGH_CIRCLE_MIN_RADIUS,
                               maxRadius=HOUGH_CIRCLE_MAX_RADIUS)
    if circles is None:
        return None
    circles = circles[0]
    if len(circles) == 0:
        return None
    largest = max(circles, key=lambda c: c[2])
    return int(largest[0]), int(largest[1]), int(largest[2])

def compute_angle_from_roi(roi, cx_roi, cy_roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=HOUGH_LINE_THRESHOLD,
                            minLineLength=HOUGH_LINE_MIN_LINE_LENGTH,
                            maxLineGap=HOUGH_LINE_MAX_LINE_GAP)
    if lines is None:
        return None
    best_dist = float('inf')
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dist = np.hypot(mx - cx_roi, my - cy_roi)
        if dist < best_dist:
            best_dist = dist
            best_line = (x1, y1, x2, y2)
    if best_line is None:
        return None
    x1, y1, x2, y2 = best_line
    d1 = (x1 - cx_roi) ** 2 + (y1 - cy_roi) ** 2
    d2 = (x2 - cx_roi) ** 2 + (y2 - cy_roi) ** 2
    tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
    dx = tip_x - cx_roi
    dy = cy_roi - tip_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def angle_to_status(angle):
    for status, ranges in ANGLE_RANGES.items():
        for (low, high) in ranges:
            if low <= angle <= high:
                return status
    return "角度异常"

def main():
    yolo_model = load_yolo_model(YOLO_WEIGHT_PATH)
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        print("无法打开摄像头或视频文件")
        return
    print("开始识别，按 q 退出")

    buffer = collections.deque(maxlen=BUFFER_SIZE)
    last_status = None
    center_history = []          # 用于圆心平滑 (cx_abs, cy_abs, r)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        status = None

        # ========== 定位表盘圆心与半径 ==========
        cx_abs = cy_abs = r_avg = None

        if yolo_model is not None:
            box = detect_gauge_with_yolo(yolo_model, frame)
            if box is not None:
                x1, y1, x2, y2 = box
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray_roi = cv2.equalizeHist(gray_roi)
                    gray_roi = cv2.GaussianBlur(gray_roi, (9,9), 2)
                    circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT,
                                               dp=HOUGH_CIRCLE_DP,
                                               minDist=HOUGH_CIRCLE_MIN_DIST,
                                               param1=HOUGH_CIRCLE_PARAM1,
                                               param2=HOUGH_CIRCLE_PARAM2,
                                               minRadius=HOUGH_CIRCLE_MIN_RADIUS,
                                               maxRadius=HOUGH_CIRCLE_MAX_RADIUS)
                    if circles is not None:
                        circles = circles[0]
                        box_cx = (x2 - x1) / 2
                        box_cy = (y2 - y1) / 2
                        # 选择最接近框中心的圆
                        best_circle = min(circles, key=lambda c: (c[0]-box_cx)**2 + (c[1]-box_cy)**2)
                        cx_r, cy_r, r_cur = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
                    else:
                        # 霍夫圆没找到，用框估算圆心和半径
                        cx_r = (x2 - x1) // 2
                        cy_r = (y2 - y1) // 2
                        r_cur = min(x2 - x1, y2 - y1) // 2

                    cx_abs = x1 + cx_r
                    cy_abs = y1 + cy_r
                    r_avg = r_cur  # 记录当前半径
            else:
                center_history.clear()
        else:
            circle = detect_gauge_hough_full(frame)
            if circle is not None:
                cx_abs, cy_abs, r_cur = circle

        # ---------- 圆心时序平滑 ----------
        if cx_abs is not None and cy_abs is not None and r_avg is not None:
            center_history.append((cx_abs, cy_abs, r_avg))
            if len(center_history) > SMOOTH_FRAMES:
                center_history.pop(0)

        if center_history:
            avg_cx = int(np.mean([p[0] for p in center_history]))
            avg_cy = int(np.mean([p[1] for p in center_history]))
            avg_r = int(np.mean([p[2] for p in center_history]))
        else:
            avg_cx = avg_cy = avg_r = None

        # ========== 计算角度与状态 ==========
        if avg_cx is not None and avg_cy is not None and avg_r is not None:
            # 确定 ROI，用于计算指针角度
            if yolo_model is not None and box is not None:
                # 使用 YOLO 框作为 ROI（保持指针可见）
                x1, y1, x2, y2 = box
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                roi = frame[y1:y2, x1:x2]
                # 圆心在 ROI 内的相对坐标
                roi_cx = avg_cx - x1
                roi_cy = avg_cy - y1
            else:
                # 全图模式：取包含圆心的正方形区域
                r_pad = int(avg_r * 1.2)
                x1 = max(0, avg_cx - r_pad)
                y1 = max(0, avg_cy - r_pad)
                x2 = min(w, avg_cx + r_pad)
                y2 = min(h, avg_cy + r_pad)
                roi = frame[y1:y2, x1:x2]
                roi_cx = avg_cx - x1
                roi_cy = avg_cy - y1

            angle = compute_angle_from_roi(roi, roi_cx, roi_cy)
            if angle is not None:
                print(f"角度：{angle:.1f} 度")
                status = angle_to_status(angle)
                # 绘制可视化
                cv2.circle(frame, (avg_cx, avg_cy), avg_r, (255,0,0), 2)
                cv2.circle(frame, (avg_cx, avg_cy), 3, (0,0,255), -1)
                rad = math.radians(angle)
                tip_x = int(avg_cx + avg_r * 0.9 * math.cos(rad))
                tip_y = int(avg_cy - avg_r * 0.9 * math.sin(rad))
                cv2.line(frame, (avg_cx, avg_cy), (tip_x, tip_y), (0,255,255), 2)
                # 画 YOLO 框（如果有）
                if yolo_model is not None and box is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            else:
                status = None

        # ========== 防抖投票 ==========
        buffer.append(status)
        valid_statuses = [s for s in buffer if s is not None]
        final_status = None
        if valid_statuses:
            most_common = max(set(valid_statuses), key=valid_statuses.count)
            if valid_statuses.count(most_common) / len(buffer) >= VOTE_THRESHOLD:
                final_status = most_common
            else:
                final_status = last_status
        else:
            final_status = None

        # ========== 中文输出 ==========
        if final_status != last_status:
            if final_status is None:
                print("仪表盘识别中...")
            else:
                print(f"仪表盘状态：{final_status}")
            last_status = final_status

        cv2.imshow("Gauge Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()