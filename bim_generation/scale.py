"""Step 3 - scale calibration: pixels per metre.

Current method: median door width in pixels / assumed standard door width (config.STD_DOOR_W).
To add (PROJECT_PLAN.md task 2.4): OCR of printed dimensions / room areas, and a fused estimate.
For CubiCasa ground truth the true scale is known (config.CUBICASA_UNITS_PER_M in SVG units).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from bim_generation.geometry import rect_axis  # noqa: E402


def door_widths_px(raw):
    return [rect_axis(o["poly"])[2] for o in raw["openings"] if o["kind"] == "Door"]


def scale_from_doors(raw, std_width=config.STD_DOOR_W):
    """Returns (px_per_m or None, list of door widths in px)."""
    w = door_widths_px(raw)
    return (float(np.median(w)) / std_width if w else None), w
