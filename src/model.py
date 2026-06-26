"""
MiniUNet - Lightweight Heatmap Regression Model
For detecting Aimlab Sixshot targets.
Total parameters: ~117,000 (lightweight, real-time on CPU/GPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) repeated twice.
    Standard U-Net building block. bias=False because BatchNorm follows.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet(nn.Module):
    """Mini U-Net for heatmap regression.

    Input:  (B, 3, H, W)  RGB image, [0, 1]
    Output: (B, 1, H, W)  heatmap, [0, 1]

    Architecture:
        Encoder:  3ch -> 16 -> 32 -> 64  (2x downsample)
        Decoder:  64 -> 32 -> 16 -> 1    (2x upsample + skip connections)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # -- Encoder (downsampling path) --
        self.inc = DoubleConv(in_channels, 16)               # H, W

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(16, 32),
        )                                                     # H/2, W/2

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64),
        )                                                     # H/4, W/4  (bottleneck)

        # -- Decoder (upsampling path) --
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64, 32)                   # 32(skip) + 32(up) = 64

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(32, 16)                   # 16(skip) + 16(up) = 32

        # -- Output head --
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)          # (B, 16, H,   W)
        x2 = self.down1(x1)       # (B, 32, H/2, W/2)
        x3 = self.down2(x2)       # (B, 64, H/4, W/4)  bottleneck

        # Decoder stage 1
        x = self.up1(x3)          # (B, 32, H/2, W/2)
        x = self._align(x, x2)
        x = torch.cat([x2, x], dim=1)  # (B, 64, H/2, W/2)
        x = self.conv_up1(x)      # (B, 32, H/2, W/2)

        # Decoder stage 2
        x = self.up2(x)           # (B, 16, H,   W)
        x = self._align(x, x1)
        x = torch.cat([x1, x], dim=1)  # (B, 32, H,   W)
        x = self.conv_up2(x)      # (B, 16, H,   W)

        # Output
        logits = self.outc(x)     # (B, 1,  H,   W)
        return torch.sigmoid(logits)

    @staticmethod
    def _align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Crop or pad tensor `a` to match the spatial size of `b`.
        Handles off-by-one size mismatches from down/up-sampling.
        """
        if a.shape[2:] != b.shape[2:]:
            a = F.interpolate(a, size=b.shape[2:], mode="bilinear", align_corners=False)
        return a


# -- Module test ----------------------------------------------------------
if __name__ == "__main__":
    model = MiniUNet()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MiniUNet built.  Trainable parameters: {total:,}")

    # Simulate a batch of 1 image, 256x256 (our dataset default size)
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape:  {dummy.shape}")
    print(f"Output shape:  {out.shape}  (expected: [1, 1, 256, 256])")
    print(f"Output range:  [{out.min().item():.4f}, {out.max().item():.4f}]  (expected: 0 ~ 1)")
