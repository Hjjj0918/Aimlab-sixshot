"""
Visual Pipeline - Real-time inference and visualization
Pipeline: [Capture] -> [Detect] -> [Visualize]

Usage:
    python src/pipeline.py --checkpoint checkpoints/best_model.pt
"""
import re
import sys
import time
import argparse
import ctypes
from pathlib import Path

import cv2
import dxcam

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture import ScreenCapture
from src.detect import TargetDetector


def main():
    # 1. Enable high-DPI awareness
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Sixshot Real-time Pipeline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                        default=None, help="Capture region (left top right bottom)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Inference device: cuda or cpu")
    args = parser.parse_args()

    # 2. Determine capture region
    if args.region:
        region = tuple(args.region)
    else:
        # Default: center 600x600 of primary display
        info = dxcam.output_info()
        match = re.search(r"Res:\((\d+),\s*(\d+)\)", str(info))
        if match:
            W, H = int(match.group(1)), int(match.group(2))
        else:
            user32 = ctypes.windll.user32
            W, H = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        SIZE = 800
        L = (W - SIZE) // 2
        T = (H - SIZE) // 2
        region = (L, T, L + SIZE, T + SIZE)
        print(f"[Pipeline] Using default center region: {region}")

    # 3. Initialize modules
    print(f"[Pipeline] Loading detector from {args.checkpoint}...")
    detector = TargetDetector(checkpoint_path=args.checkpoint, device=args.device)

    # Fullscreen capture + numpy slicing (avoids dxcam DPI coordinate bugs)
    cap = ScreenCapture(region=None)

    print("=" * 50)
    print(" Pipeline Started!")
    print(" Press 'Q' in the visualization window to quit.")
    print("=" * 50)

    # 4. Main loop
    cap.start()

    # Create window and move it to top-left corner
    # (avoids infinite mirror effect when the display overlaps the capture region)
    window_name = "Sixshot AI Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 10, 10)

    prev_time = time.perf_counter()
    fps_smooth = 0.9
    current_fps = 0.0

    try:
        while True:
            raw = cap.grab()
            if raw is None:
                cv2.waitKey(1)
                continue

            # Crop to target region
            L, T, R, B = region
            frame = raw[T:B, L:R]

            # Run inference
            points = detector.detect(frame)

            # Calculate FPS
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-5)
            current_fps = current_fps * fps_smooth + fps * (1.0 - fps_smooth)
            prev_time = now

            # -- Draw overlay --
            display = frame.copy()

            for px, py in points:
                cv2.circle(display, (px, py), 12, (0, 255, 0), 2)   # green ring
                cv2.circle(display, (px, py), 2, (0, 0, 255), -1)   # red center dot

            cv2.putText(display, f"FPS: {int(current_fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Targets: {len(points)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print("[Pipeline] Stopped.")


if __name__ == "__main__":
    main()
