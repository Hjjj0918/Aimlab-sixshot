"""
MouseController - Interception movement + Win32 clicks.
Interception bypasses Raw Input for movement; mouse_event handles clicks.
"""
import ctypes
import threading
import time
import math
import interception


# SendInput structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_u",)
    _fields_ = [("type", ctypes.c_ulong), ("_u", _U)]

    def __init__(self, _type=0, dx=0, dy=0, flags=0):
        super().__init__()
        self.type = _type
        self.mi.dx = dx
        self.mi.dy = dy
        self.mi.mouseData = 0
        self.mi.dwFlags = flags
        self.mi.time = 0
        self.mi.dwExtraInfo = None


class MouseController:
    """Hybrid: Interception for cursor movement, mouse_event for clicks."""

    def __init__(self):
        self._ffi = interception.ffi
        self._lib = interception.lib
        self._ctx = self._lib.interception_create_context()

        self._lib.interception_set_filter(
            self._ctx,
            self._lib.interception_is_mouse,
            self._lib.INTERCEPTION_FILTER_MOUSE_ALL,
        )

        print("[Mouse] Move mouse to identify device...")
        self._device = self._lib.interception_wait(self._ctx)

        stroke = self._ffi.new("InterceptionMouseStroke *")
        self._lib.interception_receive(self._ctx, self._device, stroke, 1)
        self._lib.interception_send(self._ctx, self._device, stroke, 1)
        print(f"[Mouse] Device {self._device} ready.")

        self._running = True
        self._thread = threading.Thread(target=self._forward_loop, daemon=True)
        self._thread.start()

        # Win32 user32 for clicks
        self._user32 = ctypes.windll.user32

    def _forward_loop(self):
        stroke = self._ffi.new("InterceptionMouseStroke *")
        while self._running:
            try:
                dev = self._lib.interception_wait(self._ctx)
                self._lib.interception_receive(self._ctx, dev, stroke, 1)
                self._lib.interception_send(self._ctx, dev, stroke, 1)
            except Exception:
                break

    def move(self, dx: int, dy: int):
        if dx == 0 and dy == 0:
            return
        stroke = self._ffi.new("InterceptionMouseStroke *")
        stroke.state = self._lib.INTERCEPTION_MOUSE_MOVE_RELATIVE
        stroke.flags = 0
        stroke.x = dx
        stroke.y = dy
        stroke.rolling = 0
        stroke.information = 0
        self._lib.interception_send(self._ctx, self._device, stroke, 1)

    def click(self):
        """SendInput click — one level below mouse_event."""
        inp = INPUT(0, 0, 0, 0x0002)  # LEFTDOWN
        self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        time.sleep(0.015)
        inp = INPUT(0, 0, 0, 0x0004)  # LEFTUP
        self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def shoot(self, dx: float, dy: float, scale: float = 1.0):
        """Flick + fire. Always moves and shoots."""
        mx = int(round(dx * scale))
        my = int(round(dy * scale))
        self.move(mx, my)
        time.sleep(0.015)
        self.click()
        time.sleep(0.05)
        return True

    def close(self):
        self._running = False
        if self._ctx is not None:
            self._lib.interception_destroy_context(self._ctx)
            self._ctx = None
