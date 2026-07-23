"""constants.py — editable configuration for the QYield CLI.

Values below match exactly what the shipped checkpoint was trained/verified with.
Change WM-811K-derived values (class names, die-state legend) only if your data
source changes; change preprocessing values (PIXEL_NORM_DIV, RESIZE_MODE,
IMAGENET_MEAN/STD) only if you also retrain/re-verify the checkpoint.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Defect classes (WM-811K, 8 real single-defect classes)
# ---------------------------------------------------------------------------
# "Base" classes = what the model's backbone was originally trained on.
# "Novel" classes = held out at training time (the few-shot generalization target).
# Both groups are equally valid query/support classes at inference time — this
# split only matters for understanding how the checkpoint was TRAINED, not for
# what it can classify.
BASE_CLASSES = ["Center", "Edge-Ring", "Edge-Loc"]
NOVEL_CLASSES = ["Donut", "Loc", "Near-full", "Random", "Scratch"]
ALL_DEFECT_CLASSES = BASE_CLASSES + NOVEL_CLASSES

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

# ---------------------------------------------------------------------------
# Die-state legend — every wafer map is a 2D grid of per-die integer states.
# This is what a raw .npy query image's pixel VALUES mean (0/1/2), distinct
# from the defect CLASS label above (which describes the whole wafer's overall
# defect pattern, not any single die).
# ---------------------------------------------------------------------------
DIE_STATE_LEGEND = {
    0: "blank — no die present at this position (outside the wafer's usable area)",
    1: "normal die — good/functional die, no defect",
    2: "defective die — failed/faulty die (the pattern formed by these positions "
       "across the wafer is what determines the defect CLASS label above)",
}
PIXEL_NORM_DIV = 2.0    # raw {0,1,2} die-state -> [0,1] float: value / PIXEL_NORM_DIV
RESIZE_MODE = "bilinear"

# ---------------------------------------------------------------------------
# The model's 3 SSL backbones expect ImageNet-normalized RGB input.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Default asset locations (relative to the package's repo root).
# ---------------------------------------------------------------------------
DEFAULT_CKPT_PATH = "checkpoints/qresnet_ens/qresnet_ens.pt"
DEFAULT_STEMS_DIR = "checkpoints/stems"
DEFAULT_KSET_PATH = "data/kset_k10_s42.npz"
