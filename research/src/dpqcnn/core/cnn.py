"""Classical CNN baseline (architecture.md §3).

Added specifically to make the business document's central claim testable:
"DP-QCNN degrades more slowly than a classical CNN as labeled rare-class
examples shrink and simulated hardware noise rises." Without a classical
baseline in the comparison matrix, that crossover claim cannot be measured.

Handles native-resolution, variable-shape wafer maps directly (no resize) via
global average pooling before the classifier head, so the same model works
across every native (h, w) in WM-811K without a fixed input size — the
classical-side analogue of the ring encoders' "native resolution, no resize"
design (architecture.md §1). Parameter count is kept comparable to the
quantum models (small stack, not a modern deep CNN) per constants.py
CNN_* config.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    """Small conv stack for variable-shape single-channel wafer maps.

    conv(3x3, pad=1) -> ReLU -> [conv -> ReLU]* -> global average pool -> FC.
    Global average pooling (not a fixed-size flatten) is what lets this model
    accept any native (h, w) without resizing, matching the ring encoders'
    native-resolution treatment of the same data.
    """

    def __init__(self, channels: list[int] = (8, 16), kernel_size: int = 3,
                 fc_hidden: int = 32, num_classes: int = 2, in_channels: int = 1):
        super().__init__()
        layers = []
        c_in = in_channels
        pad = kernel_size // 2
        for c_out in channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size, padding=pad))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c_in, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, h, w) -> (b, 1, h, w)
        feat = self.conv(x)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)

    def extra_repr(self) -> str:
        return f"native-resolution (global-avg-pool head)"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["WaferCNN", "count_params"]
