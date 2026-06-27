"""
MouseController - Interception movement + Win32 clicks.
Interception bypasses Raw Input for movement; mouse_event handles clicks.
"""
import ctypes
import threading
import time
import math
import interception


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
        # Use mouse_event — often bypasses Raw Input for button events
        self._user32.mouse_event(0x0002, 0, 0, 0, 0)   # LEFTDOWN
        time.sleep(0.015)
        self._user32.mouse_event(0x0004, 0, 0, 0, 0)   # LEFTUP

    def shoot(self, dx: float, dy: float, scale: float = 1.0):
        """Flick to target, fire only when crosshair is on it."""
        dist = math.hypot(dx, dy)

        if dist < 10:
            self.click()
            time.sleep(0.05)
            return True

        fraction = 0.7 if dist < 40 else 1.0
        mx = int(round(dx * fraction * scale))
        my = int(round(dy * fraction * scale))
        self.move(mx, my)
        time.sleep(0.01)
        return False

    def close(self):
        self._running = False
        if self._ctx is not None:
            self._lib.interception_destroy_context(self._ctx)
            self._ctx = None
