"""Quantity takeoff from the metric model.

Rules (state them in the report):
- Wall length = wall footprint area / median thickness, per class, so overlaps at corners count once.
  Interior wall footprint excludes any part overlapping exterior walls.
- Gross wall face area = length x storey height; this is the priced quantity (cost.py), with doors and
  windows priced separately. Net wall face area = gross - opening areas (door 2.1 m high, window 1.2 m high)
  is reported for reference.
- Floor area = sum of room polygons (net, inside walls).
- Entrance door = door hosted by an exterior wall.
"""
import sys
from pathlib import Path

import numpy as np
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import H_WALL, H_DOOR, SILL, H_WIN_TOP  # noqa: E402


def quantities(m):
    """Returns a dict of quantities (floats in m / m2 / m3, ints for counts, rooms dict)."""
    q = {}
    for ext in (True, False):
        ws = [w for w in m["walls"] if w["external"] == ext]
        if not ws:
            continue
        foot = unary_union([w["poly"] for w in ws])
        if not ext:  # interior walls: do not count the part overlapping exterior walls
            foot = foot.difference(unary_union([w["poly"] for w in m["walls"] if w["external"]]))
        t = float(np.median([w["t"] for w in ws]))
        length = foot.area / t
        ops = [o for w in ws for o in w["openings"]]
        op_area = sum(o["width"] * (H_DOOR if o["kind"] == "Door" else H_WIN_TOP - SILL) for o in ops)
        key = "ext" if ext else "int"
        q[f"{key}_wall_length_m"] = length
        q[f"{key}_wall_gross_area_m2"] = length * H_WALL
        q[f"{key}_wall_net_area_m2"] = length * H_WALL - op_area
        q[f"{key}_wall_volume_m3"] = foot.area * H_WALL - sum(
            o["width"] * t * (H_DOOR if o["kind"] == "Door" else H_WIN_TOP - SILL) for o in ops)
    doors = [o for w in m["walls"] for o in w["openings"] if o["kind"] == "Door"]
    wins = [o for w in m["walls"] for o in w["openings"] if o["kind"] == "Window"]
    q["doors_entry"] = sum(1 for w in m["walls"] for o in w["openings"] if o["kind"] == "Door" and w["external"])
    q["doors_interior"] = len(doors) - q["doors_entry"]
    q["windows"] = len(wins)
    q["window_area_m2"] = sum(o["width"] * (H_WIN_TOP - SILL) for o in wins)
    q["floor_area_m2"] = sum(sp["poly"].area for sp in m["spaces"])
    q["rooms"] = {f'{sp["type"]}_{i}': round(sp["poly"].area, 2) for i, sp in enumerate(m["spaces"])}
    return q
