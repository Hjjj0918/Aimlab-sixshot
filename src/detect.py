"""
TargetDetector - Inference wrapper for MiniUNet heatmap model.
Converts captured frames -> heatmap -> target (x, y) coordinates.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class TargetDetector:
    """Load a trained MiniUNet and detect target centers in frames.

    Usage:
        detector = TargetDetector("checkpoints/best_model.pt", device="cpu")
        points = detector.detect(frame)  # -> [(x1, y1), (x2, y2), ...]
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_size: tuple[int, int] = (256, 256),
        threshold: float = 0.5,
        min_distance: int = 8,
        device: str = "cpu",
    ):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint file.
            input_size:      (width, height) the model expects.
            threshold:       Heatmap value above which a peak counts as a target.
            min_distance:    Minimum pixel distance between peaks (non-max suppression).
            device:          "cpu" or "cuda".
        """
        self.input_size = input_size
        self.threshold = threshold
        self.min_distance = min_distance
        self.device = torch.device(device)

        # Lazy import to avoid circular dependency at module level
        from src.model import MiniUNet

        self.model = MiniUNet().to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        print(f"[Detector] Loaded model from {checkpoint_path}")
        print(f"            Input size: {input_size}, threshold: {threshold}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        # Support both raw state_dict and full checkpoint dict
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Detect target centers in a BGR frame (numpy array, HxWx3).

        Returns:
            List of (x, y) pixel coordinates in the original frame's resolution.
        """
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocess: BGR -> RGB, resize, normalize, HWC -> CHW
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)          # -> (H, W, 3)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1) # -> (3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)     # -> (1, 3, H, W)

        # 2. Inference
        with torch.inference_mode():
            heatmap = self.model(tensor)                 # (1, 1, H, W)
        heatmap = heatmap.squeeze().cpu().numpy()        # -> (H, W)

        # 3. Find peaks in heatmap
        peaks = self._find_peaks(heatmap)

        # 4. Scale coordinates back to original frame size
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        points = [(int(px * scale_x), int(py * scale_y)) for (px, py) in peaks]

        return points

    def _find_peaks(self, heatmap: np.ndarray) -> list[tuple[int, int]]:
        """Find local maxima in a 2D heatmap above threshold.

        Uses max-pooling for efficient non-maximum suppression:
        a pixel is a peak if it equals the local max and exceeds threshold.
        """
        h, w = heatmap.shape
        # Pad with -inf so edges are handled correctly
        padded = np.pad(heatmap, 1, mode="constant", constant_values=-1.0)

        # Check each pixel against its 8 neighbors
        is_peak = (
            (heatmap >= padded[0:h, 0:w])       # top-left
            & (heatmap >= padded[0:h, 1:w+1])    # top
            & (heatmap >= padded[0:h, 2:w+2])    # top-right
            & (heatmap >= padded[1:h+1, 0:w])    # left
            & (heatmap >= padded[1:h+1, 2:w+2])  # right
            & (heatmap >= padded[2:h+2, 0:w])    # bottom-left
            & (heatmap >= padded[2:h+2, 1:w+1])  # bottom
            & (heatmap >= padded[2:h+2, 2:w+2])  # bottom-right
            & (heatmap > self.threshold)
        )

        peak_y, peak_x = np.where(is_peak)
        peaks = list(zip(peak_x, peak_y))  # (x, y) pairs

        # Sort by confidence (highest first) and apply min_distance filter
        peaks.sort(key=lambda p: heatmap[p[1], p[0]], reverse=True)

        filtered = []
        for p in peaks:
            if all(
                np.sqrt((p[0] - f[0]) ** 2 + (p[1] - f[1]) ** 2) >= self.min_distance
                for f in filtered
            ):
                filtered.append(p)

        return filtered


# -- Module test ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Create a dummy checkpoint for testing
    from src.model import MiniUNet
    dummy_ckpt = "checkpoints/_test_model.pt"
    Path("checkpoints").mkdir(exist_ok=True)

    model = MiniUNet()
    torch.save({"model_state_dict": model.state_dict()}, dummy_ckpt)

    detector = TargetDetector(dummy_ckpt, device="cpu")

    # Simulate a frame
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    points = detector.detect(fake_frame)
    print(f"Detected {len(points)} targets in random noise frame.")
    print(f"Points: {points}")

    # Cleanup
    Path(dummy_ckpt).unlink()
