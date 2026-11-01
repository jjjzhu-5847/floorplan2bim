"""Project-wide settings. Every module reads its assumptions from here, so a change is made once.

Units: metres, square metres, USD.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CUBICASA_DIR = DATA_DIR / "cubicasa5k"          # unzip the Zenodo archive here
MODELS_DIR = ROOT / "models"
# Recognition model: our SegFormer-B2 fine-tune at the original resolution (recognition/train_segformer.py
# native-train hf:nvidia/segformer-b2-finetuned-ade-512-512), weights stored as fp16. Chosen over our B0 fine-tune
# on the validation split (outputs/evaluation_val_dev_sep3 vs outputs/evaluation_val_b2). The public B0 checkpoint
# behind the test-set baseline (outputs/evaluation_baseline) is not kept; see recognition/segformer.py.
SEGFORMER_CKPT = MODELS_DIR / "segformer_native_b2.pt"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- geometry assumptions
H_WALL = 2.7        # storey / wall height
H_DOOR = 2.1        # door head height
SILL = 0.9          # window sill height
H_WIN_TOP = 2.1     # window head height
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
# Calibrated 2026-10-31 on single-storey validation plans for the FINAL system (SegFormer-B2 at native resolution,
# current vectorizer; outputs/evaluation_val_b2 -> outputs/priors/scale_calibration.json). Chosen fusion:
# mask: all three + area labels (median scale error on single-storey plans 6.8%).
SCALE_CALIBRATED = dict(sizes=dict(doors=0.850, ext_walls=0.277, int_walls=0.165, doors_img=0.750, ext_walls_img=0.250, area_labels=1.082),
                        weights=dict(doors=200.7, ext_walls=37.3, int_walls=62.0, doors_img=0.0, ext_walls_img=0.0, area_labels=158.6))

# ---------------------------------------------------------------- recognition
SEG_CLASSES = ["background", "room", "wall", "door", "window"]
SEG_INPUT_SIZE = 512                       # the public checkpoint was trained on 512 x 512
SEG_WORK_SIZE = None                       # plans are letterboxed to this size for recognition (task 2.2);
                                           # None = original resolution (needs SEG_TILE and a model trained on it)
SEG_TILE = 512                             # None: one pass; 512: overlapping 512-px tiles at the working size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- vectorization (pixels at 512 px working size)
VEC_DOOR_GAP_PX = (10, 60)     # gap length between collinear wall runs that may be a door
VEC_MIN_WALL_PX = 6            # drop wall runs shorter than this
VEC_MIN_ROOM_PX = 200          # drop room regions smaller than this (pixel area)
VEC_ROOM_MODE = "footprint"      # rooms from room-class pixels ("pixels") or from the building outline ("footprint")
VEC_DROP_NONSEPARATING = True  # drop interior walls with the same room on both faces (furniture lines)

# ---------------------------------------------------------------- cost
# Unit rates (USD, 2025), from the State of Michigan, State Tax Commission, 2025 Assessor's Manual Vol. II,
# Unit-in-Place Costs (UIP 2 interior construction, UIP 5 wall costs; cost data licensed from Marshall & Swift/Boeckh;
# michigan.gov/treasury -> Assessor's Manual). Midpoint of each published range, converted at 10.7639 sq ft per m2;
# costs include the general contractor's overhead and profit, not the architect's fee; no county multiplier.
# Assemblies follow the manual's wall-cost method (UIP 5, page 1-2: a wall is the sum of its layers):
#   exterior wall  2x6 wood studs 16" o.c. 3.855 + OSB sheathing 2.285 + vinyl siding 5.695 + 6" fiberglass batts
#                  1.565 + 1/2" gypsum board 2.365 + paint (prime + one coat) 1.22 = 16.985 $/sq ft
#   partition      wood studs, gypsum board both sides, 4" wall 6.99 + paint both sides 2.44 = 9.43 $/sq ft
#   floor finish   mean of carpet (medium) 6.265, vinyl sheet 8.66, hardwood strip 3/4" (common) 12.30 = 9.075 $/sq ft
#   interior door  flush hollow-core wood, incl. hardware, 25.75 $/sq ft of opening x 0.80 m x 2.10 m
#   entrance door  residential insulated metal, incl. hardware, 39.125 $/sq ft of opening x 0.90 m x 2.10 m
#   window         wood double hung 30.375 $/sq ft x 1.375 for double glazing (manual: add 25% to 50%)
_SQFT = 10.7639
UNIT_RATES_MICHIGAN = {
    "ext_wall_m2": round(16.985 * _SQFT, 1),            # exterior wall, per m2 of net face area (182.8)
    "int_wall_m2": round(9.43 * _SQFT, 1),              # interior partition, per m2 of net face area (101.5)
    "floor_m2": round(9.075 * _SQFT, 1),                # floor finish, per m2 (97.7)
    "door_int_ea": round(25.75 * 0.80 * 2.10 * _SQFT),  # interior door, each (466)
    "door_entry_ea": round(39.125 * 0.90 * 2.10 * _SQFT),  # entrance door, each (796)
    "window_m2": round(30.375 * 1.375 * _SQFT, 1),      # window, per m2 (449.6)
}
# Location: Houston. DoD area cost factors (national average 1.00; US Army Corps of Engineers, "DoD Area Cost
# Factors (ACF)", PAX Newsletter 3.2.1, 29 March 2024, Table 4-1 of UFC 3-701-01): Texas 0.82, Michigan 1.03.
# Houston has no row of its own, so the Texas state factor is used. Note: the UFC advises against applying its ACFs
# to unit-price data; we use the ratio only as an approximate location adjustment (stated in the report). All rates
# are scaled by the same factor, so every percentage result (cost errors, AACE class, cost ranges) is unchanged.
LOCATION_FACTOR = 0.82 / 1.03
UNIT_RATES = {k: v * LOCATION_FACTOR for k, v in UNIT_RATES_MICHIGAN.items()}  # not rounded: one exact factor
UNIT_RATE_SOURCE = "Houston: Michigan Assessor's Manual 2025 x DoD location factor 0.796"

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
