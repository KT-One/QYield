"""native_picker.py — best-effort native OS file-picker dialog, launched from the
terminal. Tries (in order): zenity (Linux/GTK), kdialog (KDE), osascript (macOS),
yad (Linux fallback). Returns None if no picker is available or the user cancels —
callers should fall back to a plain path-entry prompt in that case.

No new dependency: these are external binaries invoked via subprocess, only used
if already present on the system (common on any desktop Linux/macOS install; not
expected/needed on headless servers, which is exactly when the None fallback
matters).
"""
from __future__ import annotations

import shutil
import subprocess

#: file-type filter description shown by pickers that support it
_FILE_FILTER_NAME = "Wafer maps"
_FILE_FILTER_PATTERNS = ["*.npy", "*.png", "*.jpg", "*.jpeg"]


def native_picker_available() -> bool:
    """True if at least one supported native picker binary is on PATH."""
    return any(shutil.which(b) for b in ("zenity", "kdialog", "osascript", "yad"))


def pick_file_native(title: str = "Select a wafer map") -> str | None:
    """Open a native OS file-picker dialog and return the chosen path, or None if
    unavailable, cancelled, or the call fails for any reason (never raises)."""
    for fn in (_pick_zenity, _pick_kdialog, _pick_osascript, _pick_yad):
        try:
            path = fn(title)
        except Exception:
            continue
        if path is not None:
            return path
    return None


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    return path or None


def _pick_zenity(title: str) -> str | None:
    if not shutil.which("zenity"):
        return None
    pattern = " ".join(_FILE_FILTER_PATTERNS)
    return _run(["zenity", "--file-selection", f"--title={title}",
                f"--file-filter={_FILE_FILTER_NAME} | {pattern}"])


def _pick_kdialog(title: str) -> str | None:
    if not shutil.which("kdialog"):
        return None
    pattern = " ".join(_FILE_FILTER_PATTERNS)
    return _run(["kdialog", "--title", title, "--getopenfilename", ".",
                f"{_FILE_FILTER_NAME} ({pattern})"])


def _pick_osascript(title: str) -> str | None:
    if not shutil.which("osascript"):
        return None
    script = (
        f'set thePath to POSIX path of (choose file with prompt "{title}")\n'
        f"return thePath"
    )
    return _run(["osascript", "-e", script])


def _pick_yad(title: str) -> str | None:
    if not shutil.which("yad"):
        return None
    pattern = " ".join(_FILE_FILTER_PATTERNS)
    return _run(["yad", "--file", f"--title={title}",
                f"--file-filter={_FILE_FILTER_NAME} | {pattern}"])
