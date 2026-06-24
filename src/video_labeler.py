import cv2
import os
import ctypes # [新增] 用于调用 Windows API 处理 DPI

def extract_frames_from_video(video_path: str, output_dir: str = "data/extracted"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0
    
    # ================= 核心修复部分 =================
    # 1. 创建一个允许自由调整大小的窗口，而不是默认的定死大小
    cv2.namedWindow("Video Extractor", cv2.WINDOW_NORMAL)
    # 2. 将窗口初始大小设定为一个绝大多数屏幕都能装下的尺寸（比如 1280x720）
    # 注意：这仅仅是改变了你的“观看窗口”大小，底层保存的 frame 依然是原画质全分辨率！
    cv2.resizeWindow("Video Extractor", 1280, 720) 
    # ===============================================

    print("="*50)
    print(" 视频抽帧工具")
    print(" [D] 或 [→]: 播放下一帧")
    print(" [Space]: 自动播放 / 暂停")
    print(" [S]: 保存当前帧为图片")
    print(" [Q]: 退出")
    print("="*50)

    is_playing = False
    ret, frame = cap.read()  # preload first frame
    if not ret:
        print("Video has no frames.")
        return

    while True:
        if is_playing:
            ret, frame = cap.read()
            if not ret: break
        else:
            pass 
            
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        display = frame.copy()
        cv2.putText(display, f"Frame: {current_frame_pos}/{total_frames} | Saved: {saved_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        cv2.imshow("Video Extractor", display)
        
        wait_time = 30 if is_playing else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '): 
            is_playing = not is_playing
        elif key == ord('d'): 
            is_playing = False 
            ret, frame = cap.read()
            if not ret: break
        elif key == ord('s'): 
            img_path = os.path.join(output_dir, f"extracted_{saved_count:04d}.png")
            # 这里保存的依然是原汁原味、没有被缩小过的原始 frame
            cv2.imwrite(img_path, frame)
            print(f"已保存: {img_path} (来源视频第 {current_frame_pos} 帧)")
            saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("视频处理完毕。")

if __name__ == "__main__":
    import argparse

    # Enable high-DPI awareness so the window renders at native resolution
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Sixshot Video Labeling Tool")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to gameplay video")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Output directory (default: data/raw)")
    args = parser.parse_args()

    extract_frames_from_video(args.video, args.output)