"""Path shim that re-exports the verified photonic_QCNN baseline infrastructure.

The DP-QCNN work reuses the *exact* convolution circuits, SLOS amplitude
engine, datasets and training loop from Quandela's reproduced photonic_QCNN
(vendored under ``src/photonic_qcnn_repro``). Reusing these guarantees that the
only differences between the monolithic baseline and our distributed model are
the components we deliberately change (pooling, dense, inter-register link),
which maximises comparability of the benchmark.

Import roots required by the baseline package:
  * ``<baseline_root>``           -> ``runtime_lib``, ``papers.shared...``
  * ``<baseline_root>/papers``    -> ``photonic_QCNN.lib...`` (top-level pkg)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the vendored baseline repo root and wire up sys.path / data dir.
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# Search upward for the vendored baseline repo (robust to package nesting).
BASELINE_ROOT = None
for _parent in _THIS.parents:
    _cand = _parent / "photonic_qcnn_repro"
    if (_cand / "implementation.py").exists():
        BASELINE_ROOT = _cand
        break
if BASELINE_ROOT is None:
    raise RuntimeError(
        "Baseline reproduction (photonic_qcnn_repro/implementation.py) not found "
        f"above {_THIS}. Expected it under src/."
    )

_PAPERS_DIR = BASELINE_ROOT / "papers"
for _p in (str(BASELINE_ROOT), str(_PAPERS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Make the bundled dataset assets resolvable regardless of CWD.
os.environ.setdefault("DATA_DIR", str(BASELINE_ROOT / "data"))

# ---------------------------------------------------------------------------
# Re-export the primitives we build on.
# ---------------------------------------------------------------------------
# Low-level photonic primitives (merlin 0.4.0 + perceval).
from merlin import (  # noqa: E402
    CircuitConverter,
    ComputationSpace,
)
from merlin import build_slos_distribution_computegraph as build_slos_graph  # noqa: E402

# Baseline QCNN building blocks and the monolithic reference model.
from photonic_QCNN.lib.src.qcnn_paper import (  # noqa: E402
    Measure,
    OneHotEncoder,
    QConv2d,
    QDense,
    QPooling,
    compute_amplitudes,
    generate_all_fock_states_list,
    get_circuit,
)
from photonic_QCNN.lib.src.merlin_pqcnn import (  # noqa: E402
    HybridModel,
    marginalize_photon_presence,
)

# Datasets + training loop (shared modules).
from papers.shared.photonic_QCNN.data import (  # noqa: E402
    convert_dataset_to_tensor,
    convert_tensor_to_loader,
    get_dataset,
)
from photonic_QCNN.lib.training.train_model import train_model  # noqa: E402

__all__ = [
    "BASELINE_ROOT",
    "CircuitConverter",
    "ComputationSpace",
    "build_slos_graph",
    "get_circuit",
    "generate_all_fock_states_list",
    "OneHotEncoder",
    "QConv2d",
    "QPooling",
    "QDense",
    "Measure",
    "compute_amplitudes",
    "HybridModel",
    "marginalize_photon_presence",
    "get_dataset",
    "convert_dataset_to_tensor",
    "convert_tensor_to_loader",
    "train_model",
]
