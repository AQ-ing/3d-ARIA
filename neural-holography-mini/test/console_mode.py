"""Temporarily disable legacy Windows console Quick Edit for batch runs."""
from contextlib import contextmanager
import os


@contextmanager
def uninterrupted_console():
    if os.name != "nt":
        yield
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetConsoleMode.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # CONIN$ also works when conda redirects the process's standard input.
    handle = kernel.CreateFileW("CONIN$", 0xC0000000, 3, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        yield  # No attached Windows console (e.g. redirected/remote execution).
        return
    original = wintypes.DWORD()
    changed = False
    try:
        if kernel.GetConsoleMode(handle, ctypes.byref(original)):
            # ENABLE_EXTENDED_FLAGS is required when clearing ENABLE_QUICK_EDIT_MODE.
            changed = bool(kernel.SetConsoleMode(handle, (original.value | 0x80) & ~0x40))
        yield
    finally:
        if changed:
            kernel.SetConsoleMode(handle, original.value)
        kernel.CloseHandle(handle)
