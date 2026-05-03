"""
Capture module — 高速屏幕捕获 (dxcam / Windows Desktop Duplication API)
管线位置: [Capture] -> Preprocess -> Detect -> Control
"""
import dxcam
import numpy as np


class ScreenCapture:
    """区域屏幕捕获器。复用 dxcam 实例以保持内部帧缓存预热。"""

    def __init__(self, region: tuple[int, int, int, int] | None = None):
        """
        Args:
            region: (left, top, right, bottom) 屏幕坐标, None 为全屏。
        """
        self.region = region
        self._cam: dxcam.DXCamera | None = None

    def start(self):
        """初始化 DXCamera 实例（output_color=BGR 直接对接 OpenCV）。"""
        self._cam = dxcam.create(output_color="BGR")

    def grab(self) -> np.ndarray | None:
        if self._cam is None:
            raise RuntimeError("ScreenCapture not started. Call start() before grab().")
        """
        抓取一帧。
        Returns:
            BGR numpy array (H, W, 3), 或 None（帧未就绪）。
        """
        return self._cam.grab(region=self.region)

    def stop(self):
        """释放捕获资源。"""
        if self._cam is not None:
            del self._cam
            self._cam = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


if __name__ == "__main__":
    import cv2
    import time
    import ctypes

    # 1. 开启高 DPI 意识 (物理像素模式)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    # 2. 全屏抓帧后用 NumPy 切片裁剪，绕过 DPI 区域坐标映射问题
    outputs = dxcam.output_info()
    print("[Debug] dxcam 设备列表:", outputs)
    try:
        out0 = outputs[0]["outputs"][0]
        W, H = out0["width"], out0["height"]
    except (IndexError, KeyError, TypeError):
        user32 = ctypes.windll.user32
        W, H = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    SIZE = 600
    left = (W - SIZE) // 2
    top = (H - SIZE) // 2
    # NumPy 切片: array[top:bottom, left:right]
    r1, r2 = top, top + SIZE
    c1, c2 = left, left + SIZE

    print(f"[Debug] 输出分辨率: {W}x{H}, 中心裁剪区域: rows[{r1}:{r2}] cols[{c1}:{c2}]")

    try:
        with ScreenCapture(region=None) as cap:
            print("[Debug] 进入主循环...")

            cv2.namedWindow("Capture Test")
            null_count = 0
            frame_count = 0
            last_time = time.perf_counter()

            while True:
                frame = cap.grab()

                if frame is None:
                    null_count += 1
                    if null_count % 60 == 1:  # 约每秒报一次
                        print(f"[Debug] 连续空帧: {null_count}", end="\r")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n[Debug] 无帧时用户按下 Q，退出。")
                        break
                    continue

                frame_count += 1
                if null_count > 0:
                    print(f"\n[Debug] 恢复抓帧，之前连续空帧: {null_count}")
                null_count = 0

                # NumPy 切片裁剪中心区域: frame[rows, cols]
                frame = frame[r1:r2, c1:c2]

                now = time.perf_counter()
                delta = now - last_time
                fps = 1.0 / delta if delta > 0 else 0.0
                last_time = now

                cv2.imshow("Capture Test", frame)
                title = f"FPS:{int(fps)} | Frame:{frame_count}"
                cv2.setWindowTitle("Capture Test", title)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"\n[Debug] 用户按下 Q，正常退出。(已捕获 {frame_count} 帧)")
                    break

    except Exception as e:
        import traceback
        print(f"\n[Fatal Error] 异常类型: {type(e).__name__}")
        print(f"[Fatal Error] 异常信息: {e}")
        print(f"[Fatal Error] 完整堆栈:\n{traceback.format_exc()}")
    finally:
        cv2.destroyAllWindows()
        print("[Debug] 程序结束。")
        input("按 Enter 关闭此窗口...")