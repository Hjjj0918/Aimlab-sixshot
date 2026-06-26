# Aimlab Sixshot — Vision Pipeline

Real-time computer vision pipeline for the Aimlab Sixshot task.

## Architecture

```
Capture → DL Detector → Control
 (dxcam)  (MiniHeatmapNet)  (Win32)
```

| Module | Status | Description |
|--------|--------|-------------|
| Capture | Done | dxcam screen capture, configurable ROI, BGR output |
| Labeling | Done | Interactive tools for marking target centers in frames |
| Dataset | Done | PyTorch Dataset + Gaussian heatmap generation |
| Model | Pending | MiniHeatmapNet — U-Net style heatmap regression (~85k params) |
| Training | Pending | Training loop, data augmentation, checkpointing |
| Inference | Pending | Real-time target detection from captured frames |
| Control | Pending | Mouse movement / click via Win32 API |

## Project Structure

```
Sixshot/
├── src/
│   ├── capture.py           # Screen capture via dxcam
│   ├── labeler.py           # Live screen labeling tool
│   ├── video_labeler.py     # Video-based labeling tool
│   ├── dataset.py           # PyTorch Dataset + heatmaps
│   ├── model.py             # MiniHeatmapNet
│   ├── detect.py            # TargetDetector inference
│   ├── train.py             # Training script
|   ├── train_utils.py             # Utility functions 
│   ├── pipeline.py          # (pending) Real-time detection pipeline
│   └── bot.py               # (pending) Autonomous shooting
├── tests/
│   └── test_capture.py      # Capture live preview test
├── data/raw/                # Labeled frames (PNG + JSON)
├── checkpoints/             # Saved model weights
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Labeling

Two labeling tools are available — both produce the same output format
(`frame_XXXX.png` + `frame_XXXX.json` with target coordinates).

### Screen Labeler

Capture a live screen region, freeze frames, and mark targets:

```bash
python src/labeler.py --region L T R B
```

| Key | Action |
|-----|--------|
| Space | Freeze / unfreeze frame |
| Left click | Mark target center |
| Right click | Undo last marker |
| S | Save frame + coordinates |
| R | Clear all markers |
| Q | Quit |

### Video Labeler

Play back a recorded gameplay video and mark targets frame by frame:

```bash
python src/video_labeler.py --video gameplay.mp4
```

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| D / → | Step forward 1 frame |
| A / ← | Step backward 1 frame |
| Left click | Mark target center |
| Right click | Undo last marker |
| S | Save frame + coordinates |
| R | Clear all markers |
| Q | Quit |

### Output Format

```json
{
  "targets": [[x1, y1], [x2, y2]],
  "source": "gameplay.mp4",
  "video_frame": 123
}
```

## Run Tests

```bash
python tests/test_capture.py
```

## Dependencies

```
dxcam>=0.0.4
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

