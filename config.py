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
# Unit rates (USD, 2025), from the State of Michigan, State Tax Commission, 2025 Assessor's Manual Volume I
# (Residential and Agricultural), Unit-in-Place Costs section (cost data licensed from Marshall & Swift/Boeckh;
# michigan.gov/treasury -> Assessor's Manual), column of quality Class C, the class of the manual's worked residential
# example; converted at 10.7639 sq ft per m2; no county multiplier. The manual measures exterior walls and interior
# partitions by gross area ("assume no window or door openings") and prices doors and windows separately, so the
# wall rates apply to gross face area (takeoff_cost/cost.py). Assemblies are the sum of their layers ($/sq ft):
#   exterior wall  2x6 studs 16" o.c. 3.82 + 6" insulation 1.49 + 1/2" OSB sheathing 2.29 + vinyl siding 5.37
#                  + 1/2" drywall 2.32 + paint, two coats 1.06 = 16.35 $/sq ft of wall
#   partition      2x4 wood studs 16" o.c. 2.67 + 1/2" drywall two sides 4.65 + paint two coats two sides 2.12
#                  = 9.44 $/sq ft of partition
#   floor finish   mix of the manual's example: 65% carpet with pad 5.03 + 35% (vinyl inlay 5.85 + 5/8" plywood
#                  underlayment 2.55) = 6.2095 $/sq ft of floor
#   interior door  flush door, 450 each
#   entrance door  metal, hollow core, insulated, 700 each
#   window         wood double-hung 28.35 $/sq ft x 1.5 ("for double glazed, add 50%") = 42.525 $/sq ft of window
_SQFT = 10.7639
UNIT_RATES_MICHIGAN = {
    "ext_wall_m2": round(16.35 * _SQFT, 1),      # exterior wall, per m2 of gross face area (176.0)
    "int_wall_m2": round(9.44 * _SQFT, 1),       # interior partition, per m2 of gross face area (101.6)
    "floor_m2": round(6.2095 * _SQFT, 1),        # floor finish, per m2 (66.8)
    "door_int_ea": 450,                          # interior door, each
    "door_entry_ea": 700,                        # entrance door, each
    "window_m2": round(42.525 * _SQFT, 1),       # window, per m2 (457.7)
}
# Location: Houston. DoD area cost factors (national average 1.00; US Army Corps of Engineers, "DoD Area Cost
# Factors (ACF)", PAX Newsletter 3.2.1, 29 March 2024, Table 4-1 of UFC 3-701-01): Texas 0.82, Michigan 1.03.
# Houston has no row of its own, so the Texas state factor is used. Note: the UFC advises against applying its ACFs
# to unit-price data; we use the ratio only as an approximate location adjustment (stated in the report). All rates
# are scaled by the same factor, so every percentage result (cost errors, AACE class, cost ranges) is unchanged.
LOCATION_FACTOR = 0.82 / 1.03
UNIT_RATES = {k: v * LOCATION_FACTOR for k, v in UNIT_RATES_MICHIGAN.items()}  # not rounded: one exact factor
UNIT_RATE_SOURCE = "Houston: Michigan Assessor's Manual 2025, Vol. I (residential), Class C x DoD location factor 0.796"

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
