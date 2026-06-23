"""
Labeler — 交互式数据标注工具
管线位置: [Capture] -> [Labeler]

用法:
    python src/labeler.py [--region L T R B]

操作说明:
    空格    冻结/恢复画面
    鼠标左键  在冻结画面中标记目标中心
    鼠标右键  取消上一个标记点
    S       保存当前帧和标记坐标
    R       清除当前帧所有标记
    Q       退出
"""
import json
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.capture import ScreenCapture


# ── 颜色定义 ─────────────────────────────────────────────
RED   = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Labeler:
    """交互式标注器：冻结画面 → 点击标记 → 保存"""

    def __init__(
        self,
        region: tuple[int, int, int, int] | None = None,
        output_dir: str = "data/raw",
    ):
        self.region = region
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cap = ScreenCapture(region=region)
        self.frame: np.ndarray | None = None      # 当前实时帧
        self.frozen: np.ndarray | None = None      # 冻结的帧（正在标注中）
        self.frozen_display: np.ndarray | None = None  # 带标记的显示副本
        self.points: list[tuple[int, int]] = []    # 当前帧的标记点
        self.frozen_frame = False
        self.saved_count = self._count_existing()

        # 鼠标回调
        self.window_name = "Labeler — Space:Freeze | Click:Mark | S:Save | Q:Quit"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_click)

    # ── 已有文件计数 ─────────────────────────────────────
    def _count_existing(self) -> int:
        return len(list(self.output_dir.glob("frame_*.png")))

    # ── 鼠标回调 ─────────────────────────────────────────
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

    # ── 重绘标画面 ─────────────────────────────────────
    def _redraw(self):
        """将标记点画到冻结帧的副本上。"""
        if self.frozen is None:
            return
        self.frozen_display = self.frozen.copy()
        for px, py in self.points:
            cv2.circle(self.frozen_display, (px, py), 6, RED, -1)      # 实心圆
            cv2.circle(self.frozen_display, (px, py), 8, RED, 2)       # 外圈

    # ── 绘制状态栏 ─────────────────────────────────────
    def _draw_hud(self, img: np.ndarray) -> np.ndarray:
        """在画面底部叠加状态信息。"""
        h, w = img.shape[:2]

        # 半透明背景条
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

    # ── 保存 ───────────────────────────────────────────
    def _save(self):
        if self.frozen is None or not self.points:
            print("[Labeler] ⚠️  没有标记点，跳过保存。")
            return

        idx = self.saved_count
        img_path = self.output_dir / f"frame_{idx:04d}.png"
        json_path = self.output_dir / f"frame_{idx:04d}.json"

        # 保存原始帧（不含叠加信息）
        cv2.imwrite(str(img_path), self.frozen)

        # 保存标注坐标
        data = {
            "targets": self.points,
            "region": list(self.region) if self.region else None,
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self.saved_count += 1
        print(f"[Labeler] ✅ 保存: {img_path.name} ({len(self.points)} targets)")

    # ── 主循环 ─────────────────────────────────────────
    def run(self):
        print("=" * 55)
        print("  Sixshot Labeler — 数据标注工具")
        print("  空格: 冻结画面 | 左键: 标记 | 右键: 撤销")
        print("  S: 保存 | R: 清空标记 | Q: 退出")
        print(f"  输出目录: {self.output_dir.resolve()}")
        print("=" * 55)

        self.cap.start()

        try:
            while True:
                # 获取实时帧
                if not self.frozen_frame:
                    raw = self.cap.grab()
                    if raw is None:
                        cv2.waitKey(1)
                        continue
                    self.frame = raw
                    display = self.frame.copy()
                else:
                    display = self.frozen_display.copy() if self.frozen_display is not None else self.frozen.copy() # type: ignore

                # 叠加 HUD
                display = self._draw_hud(display)

                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF

                # ── 按键处理 ──
                if key == ord("q"):
                    print(f"\n[Labeler] 退出。共保存 {self.saved_count} 帧。")
                    break

                elif key == ord(" "):  # 空格
                    if self.frozen_frame:
                        # 恢复实时
                        self.frozen_frame = False
                        self.frozen = None
                        self.frozen_display = None
                        self.points = []
                    else:
                        # 冻结
                        if self.frame is not None:
                            self.frozen_frame = True
                            self.frozen = self.frame.copy()
                            self._redraw()
                            print(f"[Labeler] 🔒 画面已冻结，请在目标中心点击标记...")

                elif key == ord("s") and self.frozen_frame:
                    self._save()
                    # 保存后回到实时模式
                    self.frozen_frame = False
                    self.frozen = None
                    self.frozen_display = None
                    self.points = []

                elif key == ord("r") and self.frozen_frame:
                    self.points.clear()
                    self._redraw()
                    print("[Labeler] 🔄 标记已清除")

        except KeyboardInterrupt:
            print(f"\n[Labeler] 中断。共保存 {self.saved_count} 帧。")
        finally:
            self.cap.stop()
            cv2.destroyAllWindows()


# ── CLI ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sixshot 数据标注工具")
    parser.add_argument("--region", nargs=4, type=int, metavar=("L", "T", "R", "B"),
                        default=None, help="捕获区域 (left top right bottom)")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="输出目录 (默认: data/raw)")
    args = parser.parse_args()

    region = tuple(args.region) if args.region else None
    labeler = Labeler(region=region, output_dir=args.output)
    labeler.run()
