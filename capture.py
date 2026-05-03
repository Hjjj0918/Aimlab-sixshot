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
