"""Test Capture module: show capture region, press Q to quit."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.capture import ScreenCapture

REGION = (640, 360, 1280, 720)  # 示例区域，请根据你的屏幕调整

with ScreenCapture(region=REGION) as cap:
    print(f"[Capture] 区域 {REGION} 已就绪，按 Q 退出...")
    while True:
        frame = cap.grab()
        if frame is None:
            continue
        cv2.imshow("Capture Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
