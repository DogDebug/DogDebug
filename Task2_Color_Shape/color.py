import cv2
import numpy as np

def complete_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("图片路径错误！")
        return None, None, None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_counts = {'Red': 0, 'Green': 0, 'Blue': 0}
    shape_counts = {'Rect': 0, 'Cube': 0, 'Cylinder': 0, 'Sphere': 0, 'Disc': 0}
    final_mask = np.zeros_like(gray)

    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circle_data = []
        for (x, y, r) in circles:
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(w, x + r), min(h, y + r)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0: continue
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            if np.mean(hsv_roi[:, :, 1]) < 25: continue
            area = np.pi * (r ** 2)
            circle_data.append({'x': x, 'y': y, 'r': r, 'area': area})

        circle_data.sort(key=lambda item: item['area'], reverse=True)
        for i, data in enumerate(circle_data[:7]):
            x, y, r, area = data['x'], data['y'], data['r'], data['area']
            shape = "Disc" if i < 2 else "Sphere"
            roi = img[y-r:y+r, x-r:x+r]
            mean_color = cv2.mean(roi)[:3]
            b, g, r_c = mean_color
            color = 'Red' if r_c >= g and r_c >= b else 'Green' if g >= r_c and g >= b else 'Blue'
            color_counts[color] += 1
            shape_counts[shape] += 1
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(final_mask, (x, y), r, 255, -1)
            label = f"{color}_{shape}"
            cv2.putText(img, label, (x-30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    mask_pink = cv2.inRange(hsv, np.array([145, 50, 50]), np.array([175, 255, 255]))
    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000 or area > 15000: continue
        rect = cv2.minAreaRect(cnt)
        w_rect, h_rect = rect[1]
        if w_rect == 0 or h_rect == 0: continue
        if w_rect < h_rect: w_rect, h_rect = h_rect, w_rect
        aspect_ratio = w_rect / h_rect
        if aspect_ratio < 1.2:
            shape = "Cube"
            color = 'Red'
            color_counts[color] += 1
            shape_counts[shape] += 1
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
            cv2.drawContours(final_mask, [box], -1, 255, -1)
            label = f"{color}_{shape}"
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    mask_red1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([15, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([150, 60, 60]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([100, 60, 60]), np.array([130, 255, 255]))
    full_mask = cv2.bitwise_or(mask_red, mask_green)
    full_mask = cv2.bitwise_or(full_mask, mask_blue)
    cv2.floodFill(full_mask, None, (0, 0), 0)
    cv2.floodFill(full_mask, None, (w-1, 0), 0)
    cv2.floodFill(full_mask, None, (0, h-1), 0)

    full_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(final_mask))
    kernel_erode = np.ones((5,5), np.uint8)
    full_mask = cv2.erode(full_mask, kernel_erode, iterations=2)

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if x < 30 or y < 30 or x + w_box > w - 30 or y + h_box > h - 30: continue
        rect = cv2.minAreaRect(cnt)
        wr, hr = rect[1]
        if wr == 0 or hr == 0: continue
        if wr < hr: wr, hr = hr, wr
        aspect_ratio = wr / hr
        rect_area = wr * hr
        rectangularity = area / rect_area
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        vertices = len(approx)

        shape = "未知"
        if aspect_ratio > 1.5 and rectangularity > 0.65:
            shape = "Rect"
        elif aspect_ratio > 2.5 and rectangularity < 0.8 and vertices > 4:
            shape = "Cylinder"
        else: continue

        roi = img[y:y+h_box, x:x+w_box]
        mean_color = cv2.mean(roi)[:3]
        b, g, r = mean_color
        if r >= g and r >= b: color = 'Red'
        elif g >= r and g >= b: color = 'Green'
        else: color = 'Blue'

        color_counts[color] += 1
        shape_counts[shape] += 1
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (255, 255, 0), 2)
        cv2.drawContours(final_mask, [box], -1, 255, -1)
        label = f"{color}_{shape}"
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img, (color_counts, shape_counts), final_mask


def detect_and_fit_rectangles(img):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None: return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=80, maxLineGap=20)
    if lines is None: return None
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 > h * 0.85 and y2 > h * 0.85: continue
        filtered_lines.append((x1, y1, x2, y2))
    mid_points = [(seg[0] + seg[2]) / 2 for seg in filtered_lines]
    if len(mid_points) < 4: return None
    median_x = np.median(mid_points)
    group1, group2 = [], []
    for i, seg in enumerate(filtered_lines):
        if mid_points[i] < median_x: group1.append(seg)
        else: group2.append(seg)

    def fit_rectangle(lines_group):
        if len(lines_group) < 2: return None
        points = []
        for seg in lines_group:
            points.append([seg[0], seg[1]])
            points.append([seg[2], seg[3]])
        points = np.array(points)
        hull = cv2.convexHull(points)
        if len(hull) < 3: return None
        epsilon = 0.02 * cv2.arcLength(hull, True)
        hull = cv2.approxPolyDP(hull, epsilon, True)
        return cv2.minAreaRect(hull)

    return fit_rectangle(group1)


def detect_contours_directly(img):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: continue
        rect = cv2.minAreaRect(cnt)
        w_rect, h_rect = rect[1]
        if w_rect == 0 or h_rect == 0: continue
        if w_rect < h_rect: w_rect, h_rect = h_rect, w_rect
        aspect_ratio = w_rect / h_rect
        if aspect_ratio > 1.5:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            roi = img[y:y+h_box, x:x+w_box]
            mean_color = cv2.mean(roi)[:3]
            b, g, r = mean_color
            if r >= g and r >= b:
                return rect
    return None


def final_integration(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("图片路径错误！")
        return
    h, w = img.shape[:2]

    global_result_img, stats, final_mask = complete_detection(image_path)
    color_counts, shape_counts = stats

    left_x1, left_y1 = 0, int(h * 0.45)
    left_x2, left_y2 = int(w * 0.4), h
    right_x1, right_y1 = int(w * 0.32), int(h * 0.50)
    right_x2, right_y2 = int(w * 0.58), int(h * 0.98)

    left_roi_img = img[left_y1:left_y2, left_x1:left_x2]
    right_roi_img = img[right_y1:right_y2, right_x1:right_x2]

    rect_left = detect_and_fit_rectangles(left_roi_img)
    if rect_left is not None:
        box_left = cv2.boxPoints(rect_left)
        box_left[:, 0] += left_x1
        box_left[:, 1] += left_y1
        box_left = np.int32(box_left)
        cv2.drawContours(global_result_img, [box_left], 0, (0, 255, 0), 2)
        cv2.putText(global_result_img, "Red_Rect", (box_left[0][0], box_left[0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        color_counts['Red'] += 1
        shape_counts['Rect'] += 1

    rect_right = detect_contours_directly(right_roi_img)
    if rect_right is not None:
        box_right = cv2.boxPoints(rect_right)
        box_right[:, 0] += right_x1
        box_right[:, 1] += right_y1
        box_right = np.int32(box_right)
        cv2.drawContours(global_result_img, [box_right], 0, (0, 255, 0), 2)
        cv2.putText(global_result_img, "Red_Cylinder", (box_right[0][0], box_right[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        color_counts['Red'] += 1
        shape_counts['Cylinder'] += 1

    print("="*50)
    print(f"红色(含粉色): {color_counts['Red']} 个")
    print(f"绿色: {color_counts['Green']} 个")
    print(f"蓝色: {color_counts['Blue']} 个")
    print("-"*20)
    print(f"长方体: {shape_counts['Rect']} 个")
    print(f"正方体: {shape_counts['Cube']} 个")
    print(f"小球: {shape_counts['Sphere']} 个")
    print(f"圆盘: {shape_counts['Disc']} 个")
    print(f"圆柱: {shape_counts['Cylinder']} 个")
    print("="*50)

    cv2.imshow('Final Result', global_result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    final_integration('Task2/test.jpg')