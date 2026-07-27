"""Central configuration for  (revised FSWMPR benchmark).

Single source of truth for the few-shot wafer-map protocol: dataset paths,
class split (C1's exact 3-base/5-novel Pareto split, architecture.md §1.2),
preprocessing, episodic N-way K-shot sweep, backbone / ProtoNet / quantum head
dims, and seeds. The cost/hardware model lives in `configs/cost.yaml` and is
loaded here via `load_cost()` so nothing downstream hardcodes a price.

Nothing in this sub-phase should hardcode a split, image size, N/K/Q, episode
count, or hyperparameter outside this file (+ cost.yaml). See
`architecture.md`.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
WM811K_RAW = DATA_DIR / "LSWMD.pkl"
PROCESSED_DIR = DATA_DIR / "processed"          # 42x42 tensors, hash-versioned
EPISODES_DIR = DATA_DIR / "episodes"            # shared seeded episode JSONs
OUTPUT_DIR = REPO_ROOT / "outputs"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "fswmpr"             # all run outputs go under ./outputs
COST_CONFIG = REPO_ROOT / "configs" / "cost.yaml"
TEMP_DIR = REPO_ROOT / "temp" / "fswmpr"

# ---------------------------------------------------------------------------
# Class split (architecture.md §1.2 — C1 FSWMPR exact Pareto 80/20 split)
# ---------------------------------------------------------------------------
# 8 real single-defect classes (drop "none" + unlabeled). Base ~82% of labeled.
BASE_CLASSES = ["Center", "Edge-Ring", "Edge-Loc"]
NOVEL_CLASSES = ["Donut", "Loc", "Near-full", "Random", "Scratch"]
ALL_DEFECT_CLASSES = BASE_CLASSES + NOVEL_CLASSES

# Short, human-readable description of each defect class (K-set manifests, UIs).
CLASS_DESCRIPTIONS = {
    "Center":    "Defective (state-2) dies clustered in the wafer's center region.",
    "Edge-Ring": "Defective dies forming a ring near the wafer's outer edge.",
    "Edge-Loc":  "Defective dies localized along one section of the wafer's edge.",
    "Donut":     "Defective dies forming a ring shape offset from both center and edge (donut pattern).",
    "Loc":       "Defective dies localized in a small region, not tied to center/edge.",
    "Near-full": "Defective dies covering almost the entire wafer surface.",
    "Random":    "Defective dies scattered with no discernible spatial pattern.",
    "Scratch":   "Defective dies forming a thin line/scratch-like pattern.",
}

# Die-state legend — what a raw wafer map's per-pixel INTEGER VALUE means (distinct
# from the defect CLASS label above, which describes the whole wafer's pattern).
DIE_STATE_LEGEND = {
    0: "blank — no die present at this position (outside the wafer's usable area)",
    1: "normal die — good/functional die, no defect",
    2: "defective die — failed/faulty die (the pattern formed by these positions "
       "across the wafer is what determines the defect CLASS label above)",
}

# WM-811K stores failureType strings; canonicalize common variants to the above.
LABEL_ALIASES = {
    "Edge-Loc": "Edge-Loc", "Edge-Ring": "Edge-Ring", "Loc": "Loc",
    "Center": "Center", "Scratch": "Scratch", "Random": "Random",
    "Donut": "Donut", "Near-full": "Near-full", "Near-Full": "Near-full",
    "none": "none", "None": "none",
}

# Secondary ablation split (REVISION.md geometric-dissimilarity), off by default.
ABLATION_BASE_CLASSES = ["Center", "Edge-Ring", "Edge-Loc", "Loc", "Random"]
ABLATION_NOVEL_CLASSES = ["Donut", "Near-full", "Scratch"]

# ---------------------------------------------------------------------------
# Preprocessing (architecture.md §1.1 — Gong-Lin conventions)
# ---------------------------------------------------------------------------
IMG_SIZE = 42                     # resize target (Gong-Lin weighted-avg optimal)
PIXEL_NORM_DIV = 2.0              # {0,1,2} -> /2 float
RESIZE_MODE = "bilinear"
# Base-set augmentation ONLY (never support/query). architecture.md §1.1.
BASE_AUGMENT = {"hflip": True, "vflip": True, "rotations": [90, 180, 270]}
# BASE within-split ratios (train/val/test) for backbone pretrain+val.
BASE_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# Episodic protocol (architecture.md §4.1 — matches C1 Table 4 tasks)
# ---------------------------------------------------------------------------
EPISODE_TASKS = [
    {"n_way": 3, "k_shot": 5},
    {"n_way": 3, "k_shot": 10},
    {"n_way": 5, "k_shot": 5},
]
Q_QUERY = 15                      # query samples per class per episode (C1)
N_EPISODES = 100                  # episodes per cell (C1 used 100 -> 6000 pts@3w5s)
# 3-way tasks draw from the 5 novel classes; set to exclude Near-full (149) to
# keep 3-way CIs honest (Near-full only unavoidable at 5-way).
THREE_WAY_EXCLUDE = ["Near-full"]

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
BACKBONE_SEED = 42

# ---------------------------------------------------------------------------
# Backbone (architecture.md §2 — Conv4, C1's baseline backbone family)
# ---------------------------------------------------------------------------
BACKBONE = {
    "channels": [64, 64, 64, 64],   # 4 conv blocks (standard Conv4)
    "kernel_size": 3,
    "embed_dim": 64,                # d: frozen embedding width fed to every head
    "pretrain_epochs": 60,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
}

# ---------------------------------------------------------------------------
# Head B — ProtoNet (architecture.md §3.2)
# ---------------------------------------------------------------------------
PROTONET = {
    "distance": "euclidean",        # softmax over -||.||^2 (Snell 2017)
    "episodic_train_episodes": 20000,
    "episodic_lr": 1e-3,
    "train_n_way": 3,               # meta-train way (on BASE classes)
    "train_k_shot": 5,
    "train_q": 15,
}

# ---------------------------------------------------------------------------
# FAITHFUL literature anchor regime (architecture.md §3.2 (a); D-2026-07-17 opt.1)
# Higher resolution + flattened Conv4 embedding to actually reproduce C1's
# published numbers (ProtoNet-Conv4 78.40% 3w5s). The quantum-vs-classical
# controlled comparison instead uses the SHARED frozen GAP-64 backbone at
# IMG_SIZE=42 (regime (b)). Two regimes, both reported.
# ---------------------------------------------------------------------------
ANCHOR = {
    "img_size": 84,                 # standard few-shot Conv4 resolution
    "embed_mode": "flatten",        # flattened conv map (rich), not GAP-64
    "pretrain_epochs": 60,          # for the frozen-pretrain+prototype variant
    "episodic_train_episodes": 10000,
    "episodic_lr": 1e-3,
    "episodic_augment": True,       # augment meta-train episodes (curbs 3-class overfit)
    "episodic_early_stop_patience": 15,   # on base_val episodic acc
}

# ---------------------------------------------------------------------------
# Head A' — C1 FSWMPR MAE-ViT (architecture.md §3.2, target 84.40% 3w5s)
# ---------------------------------------------------------------------------
MAE_VIT = {
    "img_size": IMG_SIZE,
    "patch_size": 6,                # 42/6 = 7x7 = 49 patches (42 not div by 16)
    "encoder_dim": 384,             # ViT-small-ish (scaled to 42x42; C1 used base@224)
    "encoder_depth": 12,
    "encoder_heads": 6,
    "decoder_dim": 256,
    "decoder_depth": 8,
    "mask_ratio": 0.95,             # C1's extreme mask ratio
    "recon_loss": "smooth_l1",      # C1's smooth-L1 (vs vanilla MAE l2)
    "pretrain_epochs": 200,
    "pretrain_lr": 1e-3,
    "finetune_epochs": 10,
    "finetune_iters": 10,
    "finetune_lr": 1e-3,
    "triplet_margin": 2.0,          # TCSMLoss margin (C1)
    "gradnorm_alpha": 0.3,          # C1
    "gradnorm_lr": 1e-2,
}

# ---------------------------------------------------------------------------
# Head C — Distributed Quantum ProtoNet (architecture.md §3.3)
# ---------------------------------------------------------------------------
QUANTUM = {
    "d_total": 32,                  # total projected modes D (== cost.yaml d_total_modes)
    "n_qpus": [4, 8, 16],           # distributed configs (== cost.yaml n_qpus)
    "encode": "amplitude",          # embedding chunk -> photonic amplitudes
    "add_modes": 2,
    "conv_circuit": "BS",
    "dense_circuit": "BS",
    "comm_modes": ["NC", "CC"],     # CC = paired feed-forward (ablation)
    "episodic_train_episodes": 20000,
    "episodic_lr": 1e-3,
    "ce_batch_size": 128,           # accuracy-first default; vectorized fwd is the speed win.
    "ce_lr": 1e-3,                  # (raise batch+lr together, e.g. 512/4e-3, for ~10x speed at ~4pt acc cost)
    "ce_warmup_frac": 0.0,          # 0 = constant LR (original recipe); >0 enables warmup+cosine
}

# ---------------------------------------------------------------------------
# Cost/hardware config loader (configs/cost.yaml — the ONE editable place)
# ---------------------------------------------------------------------------
def load_cost(path: Path = COST_CONFIG) -> dict:
    """Load the editable cost/hardware model. Kept out of code on purpose."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


__all__ = [
    "REPO_ROOT", "DATA_DIR", "WM811K_RAW", "PROCESSED_DIR", "EPISODES_DIR",
    "OUTPUT_DIR", "CKPT_DIR", "RESULTS_DIR", "COST_CONFIG", "TEMP_DIR",
    "BASE_CLASSES", "NOVEL_CLASSES", "ALL_DEFECT_CLASSES", "LABEL_ALIASES",
    "CLASS_DESCRIPTIONS", "DIE_STATE_LEGEND",
    "ABLATION_BASE_CLASSES", "ABLATION_NOVEL_CLASSES",
    "IMG_SIZE", "PIXEL_NORM_DIV", "RESIZE_MODE", "BASE_AUGMENT", "BASE_SPLIT",
    "EPISODE_TASKS", "Q_QUERY", "N_EPISODES", "THREE_WAY_EXCLUDE",
    "SEEDS", "BACKBONE_SEED", "BACKBONE", "PROTONET", "ANCHOR", "MAE_VIT", "QUANTUM",
    "load_cost",
]
