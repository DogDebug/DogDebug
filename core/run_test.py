import cv2
from ultralytics import YOLO

# ================= 配置区 =================
# 填写你刚刚训练好的模型路径
MODEL_PATH = 'runs/detect/train/weights/best.pt' 

# 类别到中文输出的映射字典 (严格按照表5-1要求)
# 0: Low (黄区), 1: Normal (绿区), 2: High (红区)
CLASS_MAP = {
    0: {"status": "偏低", "result": "异常"},
    1: {"status": "正常", "result": "正常"},
    2: {"status": "偏高", "result": "异常"}
}
# ==========================================

def main():
    print("🚀 正在加载 YOLOv8 模型...")
    # 加载你训练好的模型
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功！打开摄像头...")

    # 打开摄像头 (笔记本前置通常是 0)
    # cap = cv2.VideoCapture("http://10.58.230.54:8080/video")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查摄像头是否被占用！")
        return

    print("\n" + "="*50)
    print("🎥 实时预测已启动！按键盘 'q' 键退出程序。")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法获取画面！")
            break
        
        
        # 获取画面中心，裁剪出中间的一块（相当于放大焦距）
        h, w = frame.shape[:2]
        # 裁剪中间 1/2 的区域 (你可以根据实际情况调小比例，比如 w//4)
        crop_frame = frame[h//4 : 3*h//4, w//4 : 3*w//4]
        
        # 把放大的画面喂给 YOLO
        results = model(crop_frame, verbose=False)
        # 注意：画面的显示和画框都要改成 crop_frame
        # 下面的代码里，所有的 frame 都要替换成 crop_frame
        
        # ⭐️ 核心：解决笔记本前置摄像头的镜像问题
        # 左右翻转画面，这样你拿表盘往左偏，画面里也是往左偏
        # frame = cv2.flip(frame, 1)

        # 使用 YOLO 进行预测 (verbose=False 关闭终端里杂乱的自带日志)
        results = model(frame, verbose=False)

        # 标记是否在这个画面中找到了表盘
        detected = False

        # 解析预测结果
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取类别 ID 和 置信度
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                # 如果置信度大于 0.5 (你可以根据实际情况调高或调低)
                if conf > 0.5:
                    detected = True
                    info = CLASS_MAP.get(cls_id)
                    
                    # 💡 比赛核心要求：终端打印中文结果！
                    print(f"👉 识别结果：{info['status']} | 状态：{info['result']} | 置信度: {conf:.2f}")

                    # 在画面上画框和写字 (方便你在屏幕上直观查看)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{info['status']} ({conf:.2f})"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not detected:
             # 为了防刷屏，这里你可以选择注释掉。比赛时如果屏幕里没表盘，可以什么都不输出。
             pass 

        # 实时显示画面
        cv2.imshow("Dashboard Detection - Press 'q' to exit", frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 退出预测程序。")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()