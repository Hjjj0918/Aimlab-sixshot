"""
Window positioning tool — moves Aimlab to center 800x800 of screen.

Usage:
    python src/window_tool.py
    python src/window_tool.py --title "Aimlabs" --size 800
"""
import ctypes
import argparse

user32 = ctypes.windll.user32

SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_SHOWWINDOW = 0x0040


def find_window_by_title(title: str) -> int:
    """Return HWND of the first window whose title contains `title`."""
    hwnd = user32.FindWindowW(None, None)
    while hwnd:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if title.lower() in buf.value.lower():
            return hwnd
        hwnd = user32.GetWindow(hwnd, 2)  # GW_HWNDNEXT
    return 0


def move_window_to_center(hwnd: int, size: int = 800):
    """Move and resize a window to the center of the primary monitor."""
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)

    left = (sw - size) // 2
    top = (sh - size) // 2

    user32.SetWindowPos(hwnd, 0, left, top, size, size, SWP_SHOWWINDOW)
    print(f"Moved window to: ({left}, {top}) — {size}x{size}")


def main():
    parser = argparse.ArgumentParser(description="Move Aimlab window to screen center")
    parser.add_argument("--title", type=str, default="Aim",
                        help="Part of the window title to search for (default: Aim)")
    parser.add_argument("--size", type=int, default=800,
                        help="Target window size in pixels (default: 800)")
    parser.add_argument("--list", action="store_true",
                        help="List all visible window titles and exit")
    args = parser.parse_args()

    if args.list:
        print("Visible windows:")
        hwnd = user32.FindWindowW(None, None)
        while hwnd:
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value.strip():
                r = ctypes.c_long()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                print(f"  [{hwnd}] \"{buf.value}\"")
            hwnd = user32.GetWindow(hwnd, 2)
        return

    hwnd = find_window_by_title(args.title)
    if not hwnd:
        print(f"No window found matching \"{args.title}\".")
        print("Use --list to see all window titles, then --title to match.")
        return

    move_window_to_center(hwnd, args.size)


if __name__ == "__main__":
    main()
