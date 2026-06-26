"""
SixshotDataset - PyTorch Dataset for labeled Sixshot frames.
Converts point annotations (center coordinates from labeling tools)
into Gaussian heatmaps for U-Net style heatmap regression.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SixshotDataset(Dataset):
    """Load labeled frames and generate Gaussian heatmap targets.

    Each sample returns:
        (image_tensor, heatmap_tensor)
        image_tensor:    (3, H, W) float32 in [0, 1], RGB
        heatmap_tensor:  (1, H_ds, W_ds) float32 in [0, 1]
    """

    def __init__(
        self,
        data_dir: str,
        sigma: float = 3.0,
        downsample: int = 1,
        input_size: int | None = None,
    ):
        """
        Args:
            data_dir:    Directory containing frame_*.png and frame_*.json files.
            sigma:       Gaussian kernel standard deviation (pixels).
            downsample:  Downsample factor for heatmap relative to input.
            input_size:  If set, resize images to (input_size, input_size) square.
                         Coordinates are scaled proportionally.
        """
        self.data_dir = Path(data_dir)
        self.sigma = sigma
        self.downsample = downsample
        self.input_size = input_size

        # Find paired (png, json) files
        self.img_files = sorted(self.data_dir.glob("*.png"))
        paired = []
        for f in self.img_files:
            json_path = f.with_suffix(".json")
            if json_path.exists():
                paired.append((f, json_path))
        self.pairs = paired

        print(f"[Dataset] Loaded {len(self.pairs)} samples from {data_dir}")

    def __len__(self):
        return len(self.pairs)

    # -- Gaussian heatmap generation ---------------------------------------
    @staticmethod
    def _generate_heatmap(
        width: int,
        height: int,
        targets: list[tuple[float, float]],
        sigma: float = 3.0,
    ) -> np.ndarray:
        """Generate a 2D Gaussian heatmap from target center points.

        Places a 2D Gaussian blob centered at each (cx, cy) target.
        Overlapping blobs use element-wise maximum to avoid double-counting.

        Args:
            width:   Heatmap width in pixels.
            height:  Heatmap height in pixels.
            targets: List of (cx, cy) point coordinates, already downsampled.
            sigma:   Gaussian standard deviation (controls blob size).

        Returns:
            float32 numpy array of shape (height, width), values in [0, 1].
        """
        heatmap = np.zeros((height, width), dtype=np.float32)

        if not targets:
            return heatmap

        # Only compute Gaussian within 3*sigma radius (beyond is ~0)
        radius = int(math.ceil(3 * sigma))

        for cx, cy in targets:
            cx_i = int(round(cx))
            cy_i = int(round(cy))

            # Clamp to valid range
            x_min = max(0, cx_i - radius)
            x_max = min(width, cx_i + radius + 1)
            y_min = max(0, cy_i - radius)
            y_max = min(height, cy_i + radius + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            # Build local coordinate grid and compute 2D Gaussian
            X, Y = np.meshgrid(
                np.arange(x_min, x_max),
                np.arange(y_min, y_max),
            )
            gaussian = np.exp(
                -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)
            )

            # Merge into global heatmap (max handles overlapping blobs)
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                gaussian,
            )

        return heatmap

    # -- Get item ----------------------------------------------------------
    def __getitem__(self, idx):
        img_path, json_path = self.pairs[idx]

        # 1. Load image: BGR -> RGB
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # 1.5 Resize if configured (scale: original -> target)
        if self.input_size is not None:
            scale_x = self.input_size / img_w
            scale_y = self.input_size / img_h
            img = cv2.resize(img, (self.input_size, self.input_size))
            img_h, img_w = img.shape[:2]
        else:
            scale_x = 1.0
            scale_y = 1.0

        # 2. Normalize and convert to tensor: HWC -> CHW
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        # 3. Load target coordinates from JSON and scale to resized image
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_targets = data.get("targets", [])

        # 4. Build target heatmap at desired resolution
        hm_w = img_w // self.downsample
        hm_h = img_h // self.downsample

        # Scale coordinates and apply downsample
        ds_targets = [
            (x * scale_x / self.downsample, y * scale_y / self.downsample)
            for (x, y) in raw_targets
        ]

        heatmap = self._generate_heatmap(hm_w, hm_h, ds_targets, self.sigma)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)

        return img_tensor, heatmap_tensor


# -- Visualization test ----------------------------------------------------
if __name__ == "__main__":
    import sys

    # Allow running from project root: python src/dataset.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import matplotlib.pyplot as plt

    dataset = SixshotDataset(data_dir="data/raw", sigma=3.0)

    if len(dataset) == 0:
        print("[Test] No labeled data found in data/raw/.")
        print("       Run the labeler tools first to generate some samples.")
        sys.exit(1)

    # Test the n-th sample (e.g., n=0)
    img_tensor, hm_tensor = dataset[0]

    print(f"Image shape:   {img_tensor.shape}")   # expected: (3, H, W)
    print(f"Heatmap shape: {hm_tensor.shape}")    # expected: (1, H, W)
    print(f"Heatmap max:   {hm_tensor.max():.4f}")  # should be 1.0

    img_disp = img_tensor.permute(1, 2, 0).numpy()
    hm_disp = hm_tensor.squeeze(0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_disp)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(hm_disp, cmap="hot")
    axes[1].set_title("Gaussian Heatmap Target")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
