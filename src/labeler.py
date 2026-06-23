"""
Labeler - Interactive data labeling tool
Pipeline: [Capture] -> [Labeler]

Usage:
    python src/labeler.py [--region L T R B]

Controls:
    Space      Freeze/unfreeze frame
    Left click  Mark target center on frozen frame
    Right click Undo last marker
    S          Save frame and marker coordinates
    R          Clear all markers on current frame
    Q          Quit
"""
import json
import os
import sys
import argparse
import ctypes
from pathlib import Path

import cv2
import numpy as np
import dxcam

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.capture import ScreenCapture


# -- Color constants ----------------------------------------------------
RED   = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Labeler:
    """Interactive labeler: freeze frame -> click to mark -> save"""

    def __init__(
        self,
        crop_region: tuple[int, int, int, int] | None = None,
        output_dir: str = "data/raw",
    ):
        self.crop_region = crop_region
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 强制 ScreenCapture 全屏抓取，通过 numpy 切片处理区域以绕过 DPI 问题
        self.cap = ScreenCapture(region=None)
        
        self.frame: np.ndarray | None = None             # current live frame
        self.frozen: np.ndarray | None = None            # frozen frame (being labeled)
        self.frozen_display: np.ndarray | None = None    # frozen frame with markers drawn
        self.points: list[tuple[int, int]] = []          # marker coordinates
        self.frozen_frame = False
        self.saved_count = self._count_existing()

        # Mouse callback
        self.window_name = "Labeler | Space:Freeze | Click:Mark | S:Save | Q:Quit"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_click)

    # -- Count existing files ----------------------------------------------
    def _count_existing(self) -> int:
        return len(list(self.output_dir.glob("frame_*.png")))

    # -- Mouse callback ----------------------------------------------------
    def _on_click(self, event, x, y, flags, param):
        if not self.frozen_frame:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self._redraw()

    # -- Redraw markers ----------------------------------------------------
    def _redraw(self):
        """Draw markers onto a copy of the frozen frame."""
        if self.frozen is None:
            return
        self.frozen_display = self.frozen.copy()
        for px, py in self.points:
            cv2.circle(self.frozen_display, (px, py), 6, RED, -1)   # filled circle
            cv2.circle(self.frozen_display, (px, py), 8, RED, 2)    # outline

    # -- HUD overlay -------------------------------------------------------
    def _draw_hud(self, img: np.ndarray) -> np.ndarray:
        """Overlay status bar at the bottom of the frame."""
        h, w = img.shape[:2]

        # Semi-transparent background bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), BLACK, -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        if self.frozen_frame:
            status = f"FROZEN | Targets: {len(self.points)} | [S] Save  [R] Clear  [Space] Resume"
            color = RED
        else:
            status = f"LIVE | Saved: {self.saved_count} | [Space] Freeze  [Q] Quit"
            color = GREEN

        cv2.putText(img, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    # -- Save --------------------------------------------------------------
    def _save(self):
        if self.frozen is None or not self.points:
            print("[Labeler] No targets marked, skip save.")
            return

        idx = self.saved_count
        img_path = self.output_dir / f"frame_{idx:04d}.png"
        json_path = self.output_dir / f"frame_{idx:04d}.json"

        # Save raw frame (without overlay)
        cv2.imwrite(str(img_path), self.frozen)

        # Save marker coordinates
        data = {
            "targets": self.points,
            "region": list(self.crop_region) if self.crop_region else None,
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self.saved_count += 1
        print(f"[Labeler] Saved: {img_path.name} ({len(self.points)} targets)")

    # -- Main loop ---------------------------------------------------------
    def run(self):
        print("=" * 55)
        print("  Sixshot Labeler - Data Labeling Tool")
        print("  Space: Freeze | Left Click: Mark | Right Click: Undo")
        print("  S: Save | R: Clear Markers | Q: Quit")
        print(f"  Output: {self.output_dir.resolve()}")
        if self.crop_region:
            print(f"  Region: {self.crop_region} (L, T, R, B)")
        print("=" * 55)

        self.cap.start()

        try:
            while True:
                # Grab live frame
                if not self.frozen_frame:
                    raw = self.cap.grab()
                    if raw is None:
                        cv2.waitKey(1)
                        continue
                    
                    # 利用 NumPy 切片裁剪目标区域
                    if self.crop_region:
                        L, T, R, B = self.crop_region
                        raw = raw[T:B, L:R]

                    self.frame = raw
                    display = self.frame.copy()
                else:
                    display = self.frozen_display.copy() if self.frozen_display is not None else self.frozen.copy()

                # Overlay HUD
                display = self._draw_hud(display)

                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF

                # -- Key handling --
                if key == ord("q"):
                    print(f"\n[Labeler] Exit. Total saved: {self.saved_count} frames.")
                    break

                elif key == ord(" "):  # Space
                    if self.frozen_frame:
                        # Resume live
                        self.frozen_frame = False
                        self.frozen = None
                        self.frozen_display = None
                        self.points = []
                    else:
                        # Freeze
                        if self.frame is not None:
                            self.frozen_frame = True
                            self.frozen = self.frame.copy()
                            self._redraw()
                            print("[Labeler] Frame frozen. Click to mark target centers...")

                elif key == ord("s") and self.frozen_frame:
                    self._save()
                    # Return to live mode after save
                    self.frozen_frame = False
                    self.frozen = None
                    self.frozen_display = None
                    self.points = []

                elif key == ord("r") and self.frozen_frame:
                    self.points.clear()
                    self._redraw()
                    print("[Labeler] Markers cleared")

        except KeyboardInterrupt:
            print(f"\n[Labeler] Interrupted. Total saved: {self.saved_count} frames.")
        finally:
            self.cap.stop()
            cv2.destroyAllWindows()


# -- CLI ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 开启高 DPI 意识 (物理像素模式)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    parser = argparse.ArgumentParser(description="Sixshot Data Labeling Tool")
    parser.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                        default=None, help="Capture region (left top right bottom)")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Output directory (default: data/raw)")
    args = parser.parse_args()

    # 2. 如果没有通过命令行指定 region，自动计算屏幕中心的 600x600 区域
    if args.region:
        region = tuple(args.region)
    else:
        outputs = dxcam.output_info()
        try:
            out0 = outputs[0]["outputs"][0]
            W, H = out0["width"], out0["height"]
        except (IndexError, KeyError, TypeError):
            user32 = ctypes.windll.user32
            W, H = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            
        SIZE = 600
        L = (W - SIZE) // 2
        T = (H - SIZE) // 2
        region = (L, T, L + SIZE, T + SIZE)
        print(f"[Info] No region provided. Using default center {SIZE}x{SIZE} area.")

    labeler = Labeler(crop_region=region, output_dir=args.output)
    labeler.run()