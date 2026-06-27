"""
Autonomous Bot - Interception driver-level pipeline
Pipeline: [Capture] -> [Detect] -> [Interception Control]

Usage:
    python src/bot.py --checkpoint checkpoints/best_model.pt

Safety:
    G    Toggle auto-shoot ON / OFF
    Esc  Emergency stop
    Q    Quit
"""
import math
import re
import sys
import time
import argparse
import ctypes
from pathlib import Path

import cv2
import numpy as np
import dxcam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture import ScreenCapture
from src.detect import TargetDetector
from src.control import MouseController


def main():
    # 1. High-DPI
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Sixshot Autonomous Bot (Interception)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                        default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--scale", type=float, default=6.0)
    parser.add_argument("--bias-y", type=float, default=-35)
    parser.add_argument("--bias-x", type=float, default=-16)
    args = parser.parse_args()

    # 2. Region
    if args.region:
        region = tuple(args.region)
    else:
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
        print(f"[Bot] Default center region: {region}")

    # 3. Init
    print(f"[Bot] Loading detector from {args.checkpoint}...")
    detector = TargetDetector(checkpoint_path=args.checkpoint, device=args.device,
                              threshold=args.threshold)

    cap = ScreenCapture(region=None)
    mouse = MouseController()

    auto_shoot = False

    print("=" * 55)
    print("  Sixshot Bot (Interception Driver)")
    print("  G  : Toggle auto-shoot")
    print("  Esc: Emergency stop")
    print("  Q  : Quit")
    print("=" * 55)

    cap.start()

    # -- Main loop -----------------------------------------------------------
    win_name = "Sixshot Bot"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_name, 10, 10)

    prev_time = time.perf_counter()
    fps_smooth = 0.9
    current_fps = 0.0
    last_shot_time = 0.0
    shot_cooldown = 0.2   # seconds between shots
    lock_target = None    # (x, y) of locked-on target when >2 detections

    try:
        while True:
            raw = cap.grab()
            if raw is None:
                cv2.waitKey(1)
                continue

            L, T, R, B = region
            frame = raw[T:B, L:R]
            points = detector.detect(frame)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-5)
            current_fps = current_fps * fps_smooth + fps * (1.0 - fps_smooth)
            prev_time = now

            # -- Auto-shoot: pick nearest, lock-on when multiple targets --
            if auto_shoot and points:
                cx = (R - L) / 2
                cy = (B - T) / 2

                if len(points) <= 2 or lock_target is None:
                    # Few targets or no lock: pick nearest freely
                    nearest = min(points, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                    lock_target = nearest if len(points) > 2 else None
                else:
                    # Multiple targets: stick to the locked one
                    nearest = min(points, key=lambda p:
                        (p[0] - lock_target[0]) ** 2 + (p[1] - lock_target[1]) ** 2)
                    # If locked target disappeared, break lock
                    if (nearest[0] - lock_target[0]) ** 2 + (nearest[1] - lock_target[1]) ** 2 > 400:
                        lock_target = None
                        nearest = min(points, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

                dx = nearest[0] - cx + args.bias_x
                dy = nearest[1] - cy + args.bias_y
                dist = math.hypot(dx, dy)

                elapsed = now - last_shot_time
                if dist < 10 and elapsed < shot_cooldown:
                    pass
                else:
                    fired = mouse.shoot(dx, dy, args.scale)
                    if fired:
                        last_shot_time = now
                        lock_target = None  # release lock after shot
                    tag = "FIRE" if fired else "aim "
                    if fired or dist > 50:
                        lock_info = " LOCK" if lock_target else ""
                        print(f"[Bot] {tag}: delta({dx:+6.1f},{dy:+6.1f}) "
                              f"{dist:5.0f}px | {len(points)}d{lock_info}", flush=True)

            # -- Overlay --
            display = frame.copy()
            for px, py in points:
                cv2.circle(display, (px, py), 12, (0, 255, 0), 2)
                cv2.circle(display, (px, py), 2, (0, 0, 255), -1)

            sc = (0, 255, 0) if auto_shoot else (0, 0, 255)
            cv2.putText(display, f"FPS: {int(current_fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Targets: {len(points)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Auto-Shoot: {'ON' if auto_shoot else 'OFF'}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("g"):
                auto_shoot = not auto_shoot
                print(f"[Bot] Auto-shoot: {'ON' if auto_shoot else 'OFF'}")
            elif key == 27:  # Esc
                auto_shoot = False
                print("[Bot] Emergency stop!")

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        mouse.close()
        cv2.destroyAllWindows()
        print("[Bot] Stopped.")


if __name__ == "__main__":
    main()
