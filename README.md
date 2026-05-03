# Aimlab Sixshot — Vision Pipeline

Real-time computer vision pipeline for the Aimlab Sixshot task.

## Architecture

```
Capture → Preprocess → Detect → Control
  (dxcam)    (HSV)   (contour)  (Win32)
```

| Module | Status | Description |
|--------|--------|-------------|
| Capture | ✅ Done | dxcam screen capture, configurable ROI, BGR output |
| Preprocess | ⏳ Pending | HSV color filtering / binarization |
| Detect | ⏳ Pending | Contour detection / target localization |
| Control | ⏳ Pending | Mouse movement / click |

## Project Structure

```
Sixshot/
├── src/           # Pipeline source code
│   └── capture.py
├── tests/         # Module tests
│   └── test_capture.py
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
python tests/test_capture.py
```
