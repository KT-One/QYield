"""preview.py — wafer-map preview widgets for the TUI.

Renders wafers with the green/red ANSI half-block style (see wafer_render):
green = good die, red = defective die, blank = outside the wafer. Pure Rich
markup — works in any terminal Textual supports, no image/graphics-protocol
dependency.
"""
from __future__ import annotations

import numpy as np
from textual.widgets import Static

from .wafer_render import PREVIEW_SIZE, render_wafer_ansi


def make_preview_widget(wafer: np.ndarray, size: int = PREVIEW_SIZE) -> Static:
    """A Static widget previewing `wafer` as green/red ANSI half-blocks."""
    return Static(render_wafer_ansi(wafer, size=size), classes="wafer-preview")


def update_preview_widget(widget: Static, wafer: np.ndarray, size: int = PREVIEW_SIZE) -> None:
    """Update an existing preview Static in place with a new wafer."""
    widget.update(render_wafer_ansi(wafer, size=size))
