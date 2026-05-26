import cv2
import numpy as np
import math
from collections import Counter

# ---------------------------- 可调参数 ----------------------------
COLOR_RANGES = {
    'green': ([35, 50, 50], [85, 255, 255]),
    'yellow': ([20, 50, 50], [35, 255, 255]),
    'red_low': ([0, 50, 50], [10, 255, 255]),
    'red_high': ([160, 50, 50], [180, 255, 255])
}

HOUGH_DP = 1.2
HOUGH_MIN_DIST = 50
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 40
HOUGH_MIN_RADIUS = 50
HOUGH_MAX_RADIUS = 300

LINE_LENGTH_MIN = 50
LINE_GAP_MAX = 10
DIST_TO_CENTER_MAX = 30
MIN_RADIUS_RATIO = 0.5

SAMPLE_R_START = 0.70
SAMPLE_R_END = 0.95
SAMPLE_R_NUM = 8
SAMPLE_PATCH_SIZE = 5
ANGLE_VOTE_OFFSET = 2

HISTORY_LEN = 5
MIN_CONSECUTIVE = 3

# ---------------------------- 辅助函数 ----------------------------
def get_color_mask(hsv, color):
    if color == 'red':
        lower1, upper1 = COLOR_RANGES['red_low']
        lower2, upper2 = COLOR_RANGES['red_high']
        mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
        mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        return cv2.bitwise_or(mask1, mask2)
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
        cx, cy, r = circles[idx]
        return (cx, cy, r)
    return None

def detect_needle_angle(image_gray, center, radius):
    cx, cy = center
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=LINE_LENGTH_MIN, maxLineGap=LINE_GAP_MAX)
    if lines is None:
        return None

    best_line = None
    best_score = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist1 = math.hypot(x1 - cx, y1 - cy)
        dist2 = math.hypot(x2 - cx, y2 - cy)

        if (dist1 < DIST_TO_CENTER_MAX and dist2 > radius * MIN_RADIUS_RATIO) or \
           (dist2 < DIST_TO_CENTER_MAX and dist1 > radius * MIN_RADIUS_RATIO):
            if dist2 > dist1:
                tip_x, tip_y = x2, y2
            else:
                tip_x, tip_y = x1, y1
            angle = math.atan2(tip_y - cy, tip_x - cx)
            score = math.hypot(tip_x - cx, tip_y - cy)
            if score > best_score:
                best_score = score
                best_line = (angle, (tip_x, tip_y))

    if best_line is None:
        return None
    angle, tip = best_line
    return angle, tip

def sample_color_at_angle(hsv, center, radius, angle_rad, r_start, r_end, r_num, patch_size, angle_offset_deg=0):
    cx, cy = center
    offset_rad = math.radians(angle_offset_deg)
    angle = angle_rad + offset_rad
    color_votes = {'green': 0, 'yellow': 0, 'red': 0}
    radii = np.linspace(r_start * radius, r_end * radius, r_num)
    
    for r in radii:
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        if x - patch_size//2 < 0 or y - patch_size//2 < 0 or \
           x + patch_size//2 >= hsv.shape[1] or y + patch_size//2 >= hsv.shape[0]:
            continue
        patch = hsv[y-patch_size//2:y+patch_size//2+1, x-patch_size//2:x+patch_size//2+1]
        if patch.size == 0:
            continue
        for color_name in ['green', 'yellow', 'red']:
            mask = get_color_mask(patch, color_name)
            if cv2.countNonZero(mask) > 10:
                color_votes[color_name] += 1
                break
    
    if max(color_votes.values()) == 0:
        return None
    primary_color = max(color_votes, key=color_votes.get)
    if primary_color == 'yellow':
        return 'LOW'
    elif primary_color == 'green':
        return 'NORMAL'
    elif primary_color == 'red':
        return 'HIGH'
    return None

def recognize_frame(frame, prev_dial_info=None):
    if frame is None:
        return None, None, None
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    dial_info = detect_dial_center_and_radius(gray)
    if dial_info is None:
        return None, frame, None
    
    center, radius = (dial_info[0], dial_info[1]), dial_info[2]
    cx, cy = center
    
    needle = detect_needle_angle(gray, center, radius)
    if needle is None:
        return None, frame, dial_info
    
    angle_rad, tip = needle
    
    status_votes = []
    for offset in range(-ANGLE_VOTE_OFFSET, ANGLE_VOTE_OFFSET+1, 2):
        status = sample_color_at_angle(hsv, center, radius, angle_rad,
                                       SAMPLE_R_START, SAMPLE_R_END, SAMPLE_R_NUM,
                                       SAMPLE_PATCH_SIZE, offset)
        if status is not None:
            status_votes.append(status)
    
    if not status_votes:
        final_status = None
    else:
        final_status = max(set(status_votes), key=status_votes.count)
    
    # 可视化
    result = frame.copy()
    cv2.circle(result, center, 3, (0, 0, 255), -1)
    cv2.circle(result, center, radius, (0, 255, 0), 2)
    if tip:
        tip_int = (int(tip[0]), int(tip[1]))
        cv2.line(result, center, tip_int, (255, 0, 0), 2)
    
    sample_r = SAMPLE_R_END * radius
    sample_x = int(cx + sample_r * math.cos(angle_rad))
    sample_y = int(cy + sample_r * math.sin(angle_rad))
    cv2.circle(result, (sample_x, sample_y), SAMPLE_PATCH_SIZE//2, (0, 255, 255), -1)
    
    # 英文显示
    status_display = final_status if final_status else 'UNKNOWN'
    cv2.putText(result, f"Status: {status_display}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return final_status, result, dial_info

def process_video_stream(source=0, save_output=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
    
    status_history = []
    current_stable_status = None
    frame_count = 0
    
    print("Processing video stream, press 'q' to exit...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended")
            break
        
        raw_status, annotated_frame, dial_info = recognize_frame(frame)
        
        if raw_status is not None:
            status_history.append(raw_status)
            if len(status_history) > HISTORY_LEN:
                status_history.pop(0)
            
            if len(status_history) >= MIN_CONSECUTIVE:
                counter = Counter(status_history)
                most_common = counter.most_common(1)[0]
                if most_common[1] >= MIN_CONSECUTIVE:
                    current_stable_status = most_common[0]
        
        if annotated_frame is not None and current_stable_status:
            cv2.putText(annotated_frame, f"Stable: {current_stable_status}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        frame_count += 1
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if annotated_frame is not None:
            cv2.imshow("Gauge Recognition - Video Stream", annotated_frame)
            if out is not None:
                out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Video processing finished")

if __name__ == "__main__":
    # 使用摄像头
    process_video_stream(source=0, save_output="gauge_recording.mp4")
    
    # 或使用视频文件
    # process_video_stream(source="test_video.mp4", save_output="gauge_result.mp4")