# Changelog

## [0.1.0] — 2026-05-03

### Added
- 项目骨架：`src/` 源码目录, `tests/` 测试目录
- **Capture 模块** (`src/capture.py`)：基于 dxcam 的屏幕捕获器
  - `ScreenCapture(region)`: start/grab/stop 生命周期 + 上下文管理器
  - `output_color="BGR"` 直接对接 OpenCV，省色彩转换开销
- 测试脚本 `tests/test_capture.py`：实时预览捕获区域，Q 键退出
- `.gitignore`：排除 `.venv/`, `__pycache__/`
- 依赖文件 `requirements.txt`：dxcam, numpy, opencv-python
