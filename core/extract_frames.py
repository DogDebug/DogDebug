import cv2
import os
import glob

def calculate_blur_laplacian(image):
    """
    计算图像的拉普拉斯方差，值越低代表图片越模糊
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_frames_from_video(video_path, output_dir, prefix, interval_sec=0.2, blur_threshold=100.0):
    """
    从视频中抽取帧
    :param interval_sec: 抽帧间隔(秒)。0.2秒对应1秒抽5帧。
    :param blur_threshold: 模糊过滤阈值，低于该值的直接丢弃。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # 默认30fps兜底
    
    frame_interval = int(fps * interval_sec)
    
    frame_count = 0
    saved_count = 0
    blurry_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 仅在指定的间隔帧进行处理
        if frame_count % frame_interval == 0:
            blur_score = calculate_blur_laplacian(frame)
            
            # 过滤模糊帧
            if blur_score < blur_threshold:
                blurry_count += 1
            else:
                # 命名格式: 类别_视频名_帧序号.jpg
                out_name = f"{prefix}_frame_{frame_count:06d}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, frame)
                saved_count += 1
                
        frame_count += 1
        
    cap.release()
    print(f"✅ 视频 {os.path.basename(video_path)} 处理完成: 保存了 {saved_count} 帧, 丢弃了 {blurry_count} 个模糊帧.")

def main():
    # 定义根目录结构
    base_raw_dir = "data/raw_videos"
    base_out_dir = "data/frames"
    
    # 类别映射字典 (对应上面让您创建的子文件夹名称)
    categories = ["low", "mid", "high"]
    
    # 遍历每个类别
    for category in categories:
        video_dir = os.path.join(base_raw_dir, category)
        output_dir = os.path.join(base_out_dir, "gauge_" + category) # 输出到 gauge_low 等
        
        if not os.path.exists(video_dir):
            print(f"⚠️ 找不到目录 {video_dir}，跳过此类别。")
            continue
            
        video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                      glob.glob(os.path.join(video_dir, "*.avi"))
                      
        print(f"\n--- 开始处理类别: {category} (共 {len(video_files)} 个视频) ---")
        for idx, video_path in enumerate(video_files):
            # 获取视频文件主体名去除后缀，用来做文件命名前缀防止覆盖
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            prefix = f"{category}_{video_basename}"
            
            # 执行抽帧，0.1秒一帧（可防止数据过于冗余），拉普拉斯阈值设为100（可根据实际摄像机画质调整）
            extract_frames_from_video(
                video_path=video_path, 
                output_dir=output_dir, 
                prefix=prefix, 
                interval_sec=0.1,     # 建议: 如果视频很长，设为 0.5秒；如果视频很短，可调小到 0.1秒
                blur_threshold=40.0   # 阈值: 运动频繁可适当降低避免全部丢弃
            )

if __name__ == "__main__":
    main()