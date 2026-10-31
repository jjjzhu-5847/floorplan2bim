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
STD_DOOR_W = 0.9    # assumed typical door width, used for scale calibration
# CubiCasa5K ground truth: model.svg coordinates are 100 units = 1 m (reported by the
# TakeoffLens project; verify on a few plans, see PROJECT_PLAN.md task 0.3).
CUBICASA_UNITS_PER_M = 100.0

# ---------------------------------------------------------------- recognition
SEG_CLASSES = ["background", "room", "wall", "door", "window"]
SEG_INPUT_SIZE = 512                       # the public checkpoint was trained on 512 x 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- vectorization (pixels at 512 px working size)
VEC_DOOR_GAP_PX = (10, 60)     # gap length between collinear wall runs that may be a door
VEC_MIN_WALL_PX = 6            # drop wall runs shorter than this
VEC_MIN_ROOM_PX = 200          # drop room regions smaller than this (pixel area)

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

# ---------------------------------------------------------------- AACE (verify against AACE RP 18R-97 before citing)
# Expected accuracy ranges (low %, high %) - the wide end of each class range.
AACE_CLASSES = {
    5: (-50, 100),
    4: (-30, 50),
    3: (-20, 30),
    2: (-15, 20),
    1: (-10, 15),
}
