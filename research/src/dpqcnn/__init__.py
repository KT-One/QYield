"""DP-QCNN: Distributed Photonic Quantum CNN via classical feed-forward.

Merges the photonic PQCNN baseline (Monbroussou et al., 2025) with the
classical-communication distributed QML scheme (Hwang et al., 2024) on the
Quandela MerLin / Perceval stack.

Public building blocks are added incrementally; see the module-level docstrings
in ``encoder``, ``blocks`` and ``model``.
"""

from __future__ import annotations

__version__ = "0.0.1"
