# Aimlab Sixshot — Deep Learning Aim Bot

Real-time computer vision + hardware-level mouse control for the Aimlab Sixshot task.

## Architecture

```
Capture → DL Detector → Control
 (dxcam)  (MiniUNet)  (Interception + SendInput)
```

| Module | Status | Description |
|--------|--------|-------------|
| Capture | Done | dxcam screen capture, configurable ROI, BGR output |
| Labeling | Done | Interactive tools for marking target centers |
| Dataset | Done | PyTorch Dataset + Gaussian heatmap generation |
| Model | Done | MiniUNet — U-Net heatmap regression (~117k params) |
| Training | Done | Training loop, augmentation, checkpointing |
| Inference | Done | Real-time detection + visualization pipeline |
| Control | Done | Interception driver movement + SendInput clicks |

## Project Structure

```
Sixshot/
├── src/
│   ├── capture.py           # Screen capture via dxcam
│   ├── labeler.py            # Live screen labeling tool
│   ├── video_labeler.py      # Video-based labeling tool
│   ├── dataset.py            # PyTorch Dataset + Gaussian heatmaps
│   ├── model.py              # MiniUNet (~117k params, U-Net)
│   ├── detect.py             # TargetDetector: model → coordinates
│   ├── train.py              # Training script
│   ├── train_utils.py        # Loss plotting, prediction visualization
│   ├── pipeline.py           # Visual pipeline (capture + detection overlay)
│   ├── control.py            # Interception driver + SendInput mouse control
│   ├── bot.py                # Autonomous aim bot (capture → detect → shoot)
│   └── window_tool.py        # Force Aimlab window to screen center
├── tests/
│   └── test_capture.py       # Capture live preview test
├── data/raw/                 # Labeled frames (PNG + JSON)
├── checkpoints/              # Saved model weights
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Interception Driver

The bot requires the [Interception](https://github.com/oblitum/Interception) driver for hardware-level mouse injection (bypasses Aimlab Raw Input).

1. Download from https://github.com/oblitum/Interception/releases
2. Run as **Administrator**: `install-interception.exe /install`
3. Reboot

## Quick Start

```bash
# 1. Position Aimlab window at screen center 800x800
python src/window_tool.py --title "aimlab_tb" --size 800

# 2. Run the bot
python src/bot.py

# 3. Press G to toggle auto-shoot
#    Press Esc for emergency stop
#    Press Q to quit
```

## Data Labeling

Two tools, same output format (`frame_XXXX.png` + `frame_XXXX.json`).

### Screen Labeler

Capture a live screen region, freeze frames, mark targets:

```bash
python src/labeler.py --region L T R B
```

| Key | Action |
|-----|--------|
| Space | Freeze / unfreeze |
| Left click | Mark target center |
| Right click | Undo marker |
| S | Save |
| Q | Quit |

### Video Labeler

Mark targets frame-by-frame in recorded gameplay:

```bash
python src/video_labeler.py --video gameplay.mp4
```

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| D / A | Step forward / backward |
| Left click | Mark target center |
| S | Save |
| Q | Quit |

## Training

```bash
python src/train.py --data-dir data/raw --epochs 100 --sigma 1.5
```

Outputs to `checkpoints/`:
- `best_model.pt` — lowest validation loss
- `training_loss.png` — loss curve
- `prediction_sample.png` — input / ground truth / prediction comparison

## Visual Pipeline (no mouse control)

Verify model accuracy before enabling the bot:

```bash
python src/pipeline.py --checkpoint checkpoints/best_model.pt
```

Green circles on detected targets. Press Q to quit.

## Bot Options

```bash
python src/bot.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `checkpoints/best_model.pt` | Model path |
| `--threshold` | `0.85` | Detection confidence (higher = fewer false positives) |
| `--scale` | `6.0` | Mouse sensitivity ratio |
| `--bias-x` | `-16` | Horizontal aim correction |
| `--bias-y` | `-35` | Vertical aim correction |
| `--region L T R B` | auto | Capture region |
| `--device` | `cuda` | `cuda` or `cpu` |

### Bias Correction

Model predictions have systematic offset from true ball centers due to annotation noise and heatmap quantization. Tune `--bias-x` and `--bias-y` per trained model.

## Dependencies

```
dxcam>=0.0.4
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
tqdm>=4.65.0
interception          # Requires separate driver installation
```

