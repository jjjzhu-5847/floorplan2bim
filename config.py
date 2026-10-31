"""Project-wide settings. Every module reads its assumptions from here, so a change is made once.

Units: metres, square metres, USD.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CUBICASA_DIR = DATA_DIR / "cubicasa5k"          # unzip the Zenodo archive here
MODELS_DIR = ROOT / "models"
SEGFORMER_CKPT = MODELS_DIR / "segformer_best.pt"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- geometry assumptions
H_WALL = 2.7        # storey / wall height
H_DOOR = 2.1        # door head height
SILL = 0.9          # window sill height
H_WIN_TOP = 2.1     # window head height
SAMPLE_PLAN_DOOR_W = 0.9  # door width drawn in the synthetic test plan; used only by evaluation/error_propagation_demo.py
# CubiCasa5K ground truth: model.svg coordinates are 100 units = 1 m (reported by the
# TakeoffLens project; verify on a few plans, see PROJECT_PLAN.md task 0.3).
CUBICASA_UNITS_PER_M = 100.0

# ---------------------------------------------------------------- scale calibration (bim_generation/scale.py, task 2.4)
# Typical sizes (m) from the TRAINING split ground truth, per-plan medians (outputs/priors/train_priors.json):
# door 0.797 m (spread between plans 5.4%), exterior wall 0.245 m (20.2%), interior wall 0.120 m (16.2%).
# Weights = 1 / spread^2. Used when the geometry is exact (ground truth, condition B).
# Printed room areas (OCR) on exact rooms recover the scale within ~2% (5 validation plans checked by hand).
SCALE_PRIORS = dict(sizes=dict(doors=0.797, ext_walls=0.245, int_walls=0.120, area_labels=1.0),
                    weights=dict(doors=1 / 0.054 ** 2, ext_walls=1 / 0.202 ** 2, int_walls=1 / 0.162 ** 2,
                                 area_labels=1 / 0.03 ** 2))
SCALE_LABEL_MAX_DISAGREEMENT = 0.35  # drop the area-label estimate if it differs from doors/walls by more
# Effective sizes and weights as our recognition + vectorizer measures them, calibrated on the VALIDATION split
# (evaluation/calibrate_scale.py). Used for real images (condition E, run_image, live demo).
# Calibrated 2026-10-31 on single-storey validation plans (outputs/priors/scale_calibration.json, from
# outputs/evaluation_val_fullres): on the 512-px mask our doors measure ~27% wider and walls ~40% thicker than the
# ground truth; measured again on the original image they come out close to the true sizes (doors 0.77 m).
# Chosen fusion: all five + area labels (median scale error on single-storey plans 10.7%;
# mask doors + walls + labels 11.5%; doors only 13.4%; image size only 17.7%).
SCALE_CALIBRATED = dict(sizes=dict(doors=1.010, ext_walls=0.352, int_walls=0.199, doors_img=0.770, ext_walls_img=0.260, area_labels=1.026),
                        weights=dict(doors=55.4, ext_walls=33.4, int_walls=29.5, doors_img=49.0, ext_walls_img=27.0, area_labels=115.7))

# ---------------------------------------------------------------- recognition
SEG_CLASSES = ["background", "room", "wall", "door", "window"]
SEG_INPUT_SIZE = 512                       # the public checkpoint was trained on 512 x 512
SEG_WORK_SIZE = 512                        # plans are letterboxed to this size for recognition (task 2.2)
SEG_TILE = None                            # None: one pass; 512: overlapping 512-px tiles at the working size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- vectorization (pixels at 512 px working size)
VEC_DOOR_GAP_PX = (10, 60)     # gap length between collinear wall runs that may be a door
VEC_MIN_WALL_PX = 6            # drop wall runs shorter than this
VEC_MIN_ROOM_PX = 200          # drop room regions smaller than this (pixel area)
VEC_ROOM_MODE = "footprint"      # rooms from room-class pixels ("pixels") or from the building outline ("footprint")

# ---------------------------------------------------------------- cost
# PLACEHOLDER unit rates (USD). Replace with RSMeans or another cited source (task 3.1).
UNIT_RATES = {
    "ext_wall_m2": 180.0,     # exterior wall, per m2 of net face area
    "int_wall_m2": 75.0,      # interior partition, per m2 of net face area
    "floor_m2": 60.0,         # floor finish, per m2
    "door_int_ea": 450.0,     # interior door, each
    "door_entry_ea": 1200.0,  # entrance door, each
    "window_m2": 550.0,       # window, per m2
}
UNIT_RATE_SOURCE = "placeholder"

# ---------------------------------------------------------------- AACE estimate classes (task 4.2)
# Building and general construction matrix of ASTM E2516-11 (2024), which is based on AACE International RP 17R-97,
# 18R-97 and 56R-08 (building and general construction), as tabulated in Alberta Infrastructure, "Classes of
# Construction Cost Estimates", Design & Technology Series #19 (alberta.ca/system/files/infra-tr-dt-series-19-estimates.pdf),
# Table 1. Ranges are (low %, high %) at the wide end of each class. The process-industry matrix (18R-97) is wider.
AACE_CLASSES = {
    5: (-30, 50),
    4: (-20, 30),
    3: (-15, 20),
    2: (-10, 15),
    1: (-5, 10),
}
