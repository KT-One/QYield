"""wafer_render.py — render a wafer-map array as colored terminal text (Rich
markup), for a quick visual sanity-check inside the TUI before running inference.
No image/graphics protocol dependency — works in any terminal Textual supports.

Die-state legend (see constants.DIE_STATE_LEGEND):
  0 = blank (outside wafer)   -> dim background
  1 = normal die              -> green
  2 = defective die           -> red
Values in between (already-normalized [0,1] floats) are bucketed the same way.
"""
from __future__ import annotations

import numpy as np

#: how many terminal rows/cols the preview grid downsamples to (kept small — this
#: is a sanity-check thumbnail, not a precision viewer)
PREVIEW_SIZE = 32


def _bucket(v: float) -> str:
    if v < 1 / 6:      # ~0 (blank)
        return "  "
    if v < 3 / 4:      # ~0.5 (good die)
        return "[green]▓▓[/green]"
    return "[red]██[/red]"        # ~1.0 (defective die)


def render_wafer_ansi(wafer: np.ndarray, size: int = PREVIEW_SIZE) -> str:
    """wafer: 2D array, either raw {0,1,2} ints or normalized [0,1] float.
    Returns a Rich-markup multi-line string (2 chars/pixel wide for a roughly
    square-looking terminal cell aspect ratio)."""
    arr = np.asarray(wafer, dtype=np.float32)
    if arr.max() > 1.0:      # raw {0,1,2} -> [0,1]
        arr = arr / 2.0
    h, w = arr.shape
    # nearest-neighbor downsample to a small preview grid
    row_idx = (np.linspace(0, h - 1, min(size, h))).astype(int)
    col_idx = (np.linspace(0, w - 1, min(size, w))).astype(int)
    small = arr[np.ix_(row_idx, col_idx)]
    lines = ["".join(_bucket(v) for v in row) for row in small]
    return "\n".join(lines)
