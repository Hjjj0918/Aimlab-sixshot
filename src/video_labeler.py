"""
Video Labeler - Mark target centers in recorded gameplay frames.
Usage:
    python src/video_labeler.py --video gameplay.mp4 [--output data/raw]

Controls:
    Space      Play / Pause
    D / Right  Step forward 1 frame (paused)
    A / Left   Step backward 1 frame (paused)
    Left click  Mark target center (paused)
    Right click Undo last marker
    S          Save frame + marker coordinates
    R          Clear all markers on current frame
    Q          Quit
"""
import json
import ctypes
import argparse
from pathlib import Path

import cv2
import numpy as np

# -- Color constants ----------------------------------------------------
RED   = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# -- Mouse callback state (module-level so cv2 can reach it) ------------
_points: list[tuple[int, int]] = []
_frame: np.ndarray | None = None


def _on_click(event, x, y, flags, param):
    global _points
    playing = param.get("playing", False)
    if playing:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _points:
            _points.pop()


def label_from_video(video_path: str, output_dir: str = "data/raw"):
    """Open a video, navigate frame-by-frame, mark target centers, save."""
    global _points, _frame

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = sum(1 for _ in out.glob("frame_*.png"))
    playing = False
    _points = []

    # Window setup (DPI-aware)
    win_name = "Video Labeler | Space:Play | Click:Mark | S:Save | Q:Quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, _on_click, {"playing": playing})

    # Preload first frame
    ret, _frame = cap.read()
    if not ret:
        print("Video has no frames.")
        return

    print("=" * 55)
    print("  Sixshot Video Labeler")
    print(f"  Video:  {Path(video_path).name}")
    print(f"  Frames: {total_frames}  |  Existing saves: {saved_count}")
    print(f"  Output: {out.resolve()}")
    print("  Space:Play/Pause  D:Next  A:Prev  Click:Mark  S:Save  Q:Quit")
    print("=" * 55)

    try:
        while True:
            if playing:
                ret, _frame = cap.read()
                if not ret:
                    playing = False
                    print("[VideoLabeler] End of video.")

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Build display: draw markers on a copy
            display = _frame.copy()
            for px, py in _points:
                cv2.circle(display, (px, py), 6, RED, -1)
                cv2.circle(display, (px, py), 8, RED, 2)

            # HUD bar
            h = display.shape[0]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 50), (display.shape[1], h), BLACK, -1)
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            if playing:
                status = f"PLAYING | Frame: {frame_idx}/{total_frames} | Saved: {saved_count} | [Space] Pause"
                color = GREEN
            else:
                status = (f"PAUSED | Frame: {frame_idx}/{total_frames} | "
                          f"Targets: {len(_points)} | Saved: {saved_count} | "
                          f"[Space] Play  [D] Next  [A] Prev  [S] Save  [R] Clear")
                color = RED if _points else GREEN

            cv2.putText(display, status, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(win_name, display)
            # Update mouse callback state
            cv2.setMouseCallback(win_name, _on_click, {"playing": playing})

            wait = 30 if playing else 1
            key = cv2.waitKey(wait) & 0xFF

            # -- Key handling --
            if key == ord("q"):
                print(f"\n[VideoLabeler] Exit. Total saved: {saved_count} frames.")
                break

            elif key == ord(" "):  # Space: toggle play/pause
                playing = not playing
                if playing:
                    _points.clear()

            elif key in (ord("d"), 0x52, 0x27):  # D, Right arrow
                if not playing:
                    ret, _frame = cap.read()
                    if not ret:
                        print("[VideoLabeler] End of video.")

            elif key in (ord("a"), 0x51, 0x25):  # A, Left arrow
                if not playing:
                    target = max(0, frame_idx - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ret, _frame = cap.read()

            elif key == ord("s") and not playing:
                if _frame is None or not _points:
                    print("[VideoLabeler] No targets marked, skip save.")
                    continue

                img_path = out / f"frame_{saved_count:04d}.png"
                json_path = out / f"frame_{saved_count:04d}.json"

                cv2.imwrite(str(img_path), _frame)
                data = {
                    "targets": _points,
                    "source": Path(video_path).name,
                    "video_frame": frame_idx,
                }
                json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

                saved_count += 1
                print(f"[VideoLabeler] Saved: {img_path.name} "
                      f"(frame {frame_idx}, {len(_points)} targets)")
                _points.clear()

            elif key == ord("r") and not playing:
                _points.clear()
                print("[VideoLabeler] Markers cleared")

    except KeyboardInterrupt:
        print(f"\n[VideoLabeler] Interrupted. Total saved: {saved_count} frames.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# -- CLI ------------------------------------------------------------------
if __name__ == "__main__":
    # Enable high-DPI awareness
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

    label_from_video(args.video, args.output)
