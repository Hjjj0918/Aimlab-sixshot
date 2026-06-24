"""
Video Labeler - Offline labeling tool using recorded gameplay
Pipeline: Video File -> [Labeler]

Usage:
    python src/video_labeler.py --video gameplay.mp4 [--output data/raw]

Controls:
    Space      Play / Pause
    D / Right  Step forward 1 frame (paused)
    A / Left   Step backward 1 frame (paused)
    Left click  Mark target center (paused)
    Right click Undo last marker
    S          Save frame and marker coordinates
    R          Clear all markers on current frame
    Q          Quit
"""
import json
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

# -- Color constants ----------------------------------------------------
RED   = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class VideoLabeler:
    """Label targets frame-by-frame from a recorded gameplay video.

    Play/Pause with Space. When paused, click to mark target centers,
    S to save, D/A to step frames.
    """

    def __init__(self, video_path: str, output_dir: str = "data/raw"):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        self._cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

        # State
        self.playing = False
        self.frame: np.ndarray | None = None       # clean frame from video
        self._frame_idx = 0
        self.points: list[tuple[int, int]] = []    # marker coordinates
        self.saved_count = self._count_existing()

        # Window
        self.window_name = "Video Labeler | Space:Play | Click:Mark | S:Save | Q:Quit"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_click)

        # Load first frame
        ret, first = self._cap.read()
        if ret:
            self.frame = first
        else:
            raise RuntimeError("Cannot read video")

    # -- Count existing files ----------------------------------------------
    def _count_existing(self) -> int:
        return len(list(self.output_dir.glob("frame_*.png")))

    # -- Mouse callback ----------------------------------------------------
    def _on_click(self, event, x, y, flags, param):
        # Only allow marking when paused
        if self.playing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    # -- HUD overlay -------------------------------------------------------
    def _draw_hud(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), BLACK, -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        pos = self._frame_idx
        if self.playing:
            status = f"PLAYING | Frame: {pos}/{self.total_frames} | Saved: {self.saved_count} | [Space] Pause"
            color = GREEN
        else:
            status = (f"PAUSED | Frame: {pos}/{self.total_frames} | "
                      f"Targets: {len(self.points)} | Saved: {self.saved_count} | "
                      f"[Space] Play  [D] Next  [A] Prev  [S] Save")
            color = RED if self.points else GREEN

        cv2.putText(img, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    # -- Frame navigation --------------------------------------------------
    def _read_frame(self) -> bool:
        """Read next frame. Returns True on success."""
        ret, frame = self._cap.read()
        if ret:
            self.frame = frame
            self._frame_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return ret

    def _step_backward(self):
        """Step back one frame."""
        target = max(0, self._frame_idx - 2)  # -2 because read() advances by 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._read_frame()

    # -- Save --------------------------------------------------------------
    def _save(self):
        if self.frame is None or not self.points:
            print("[VideoLabeler] No targets marked, skip save.")
            return

        idx = self.saved_count
        img_path = self.output_dir / f"frame_{idx:04d}.png"
        json_path = self.output_dir / f"frame_{idx:04d}.json"

        # Save clean frame (no overlay)
        cv2.imwrite(str(img_path), self.frame)

        data = {
            "targets": self.points,
            "source": str(self.video_path.name),
            "video_frame": self._frame_idx,
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self.saved_count += 1
        print(f"[VideoLabeler] Saved: {img_path.name} "
              f"(video frame {self._frame_idx}, {len(self.points)} targets)")
        self.points.clear()

    # -- Main loop ---------------------------------------------------------
    def run(self):
        print("=" * 55)
        print("  Sixshot Video Labeler - Offline Labeling Tool")
        print(f"  Video:   {self.video_path.name}")
        print(f"  Frames:  {self.total_frames}  FPS: {self._fps:.0f}")
        print(f"  Output:  {self.output_dir.resolve()}")
        print("  -------------------------------------------------")
        print("  Space: Play/Pause  |  D/-> : Next frame")
        print("  A/<- : Prev frame  |  Click: Mark target")
        print("  S: Save frame      |  R    : Clear markers")
        print("=" * 55)

        try:
            while True:
                # -- Playing: auto-advance --
                if self.playing:
                    ret = self._read_frame()
                    if not ret:
                        self.playing = False
                        print("[VideoLabeler] End of video.")

                # -- Build display --
                if self.frame is None:
                    cv2.waitKey(1)
                    continue

                display = self.frame.copy()
                # Draw markers
                for px, py in self.points:
                    cv2.circle(display, (px, py), 6, RED, -1)
                    cv2.circle(display, (px, py), 8, RED, 2)
                display = self._draw_hud(display)

                cv2.imshow(self.window_name, display)

                # Wait: 30ms if playing, 1ms if paused (responsive)
                wait = 30 if self.playing else 1
                key = cv2.waitKey(wait) & 0xFF

                # -- Key handling --
                if key == ord("q"):
                    print(f"\n[VideoLabeler] Exit. Total saved: {self.saved_count} frames.")
                    break

                elif key == ord(" "):  # Space: toggle play/pause
                    self.playing = not self.playing
                    if self.playing:
                        self.points.clear()

                elif key in (ord("d"), 0x52, 0x27):  # D, Right arrow
                    if not self.playing:
                        self._read_frame()

                elif key in (ord("a"), 0x51, 0x25):  # A, Left arrow
                    if not self.playing:
                        self._step_backward()

                elif key == ord("s") and not self.playing:
                    self._save()

                elif key == ord("r") and not self.playing:
                    self.points.clear()
                    print("[VideoLabeler] Markers cleared")

        except KeyboardInterrupt:
            print(f"\n[VideoLabeler] Interrupted. Total saved: {self.saved_count} frames.")
        finally:
            self._cap.release()
            cv2.destroyAllWindows()


# -- CLI ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sixshot Video Labeling Tool")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to gameplay video")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Output directory (default: data/raw)")
    args = parser.parse_args()

    labeler = VideoLabeler(video_path=args.video, output_dir=args.output)
    labeler.run()
