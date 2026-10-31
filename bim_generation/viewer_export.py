"""Step 4a - 3D viewer data: metric model -> JSON consumed by web/template.html (three.js).

boxes : [cx, cy, z0, length, thickness, height, angle_rad, is_external]  solid wall pieces around openings
panels: [kind, cx, cy, z0, width, height, angle_rad]                      door leaves / window glass
rooms : [{type, pts, area, c}]                                            floor polygons and label anchor
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import H_WALL, H_DOOR, SILL, H_WIN_TOP  # noqa: E402


def wall_pieces(w):
    """Split a wall into solid boxes around its openings: (s0, s1, z0, z1)."""
    ops = sorted(w["openings"], key=lambda o: o["s0"])
    pieces, cur = [], 0.0
    for o in ops:
        if o["s0"] > cur + 1e-6:
            pieces.append((cur, o["s0"], 0.0, H_WALL))
        if o["kind"] == "Door":
            pieces.append((o["s0"], o["s1"], H_DOOR, H_WALL))
        else:
            pieces.append((o["s0"], o["s1"], 0.0, SILL))
            pieces.append((o["s0"], o["s1"], H_WIN_TOP, H_WALL))
        cur = max(cur, o["s1"])
    if cur < w["L"] - 1e-6:
        pieces.append((cur, w["L"], 0.0, H_WALL))
    return pieces


def viewer_json(m, q, c):
    """m: metric model, q: quantities dict, c: cost dict."""
    boxes, panels = [], []
    for w in m["walls"]:
        u = (w["p1"] - w["p0"]) / w["L"]
        ang = math.atan2(u[1], u[0])
        for s0, s1, z0, z1 in wall_pieces(w):
            cx, cy = w["p0"] + u * (s0 + s1) / 2
            boxes.append([round(cx, 3), round(cy, 3), round(z0, 3), round(s1 - s0, 3),
                          round(w["t"], 3), round(z1 - z0, 3), round(ang, 4), int(w["external"])])
        for o in w["openings"]:
            cx, cy = w["p0"] + u * (o["s0"] + o["s1"]) / 2
            z0, z1 = (0.0, H_DOOR) if o["kind"] == "Door" else (SILL, H_WIN_TOP)
            panels.append([o["kind"], round(cx, 3), round(cy, 3), z0, round(o["width"], 3),
                           round(z1 - z0, 3), round(ang, 4)])
    rooms = [dict(type=sp["type"], pts=[[round(x, 3), round(y, 3)] for x, y in sp["poly"].exterior.coords],
                  area=round(sp["poly"].area, 2), c=list(np.round(sp["poly"].representative_point().coords[0], 3)))
             for sp in m["spaces"]]
    return dict(boxes=boxes, panels=panels, rooms=rooms, H=H_WALL,
                quantities={k: (round(v, 2) if isinstance(v, float) else v) for k, v in q.items()},
                cost={k: round(v) for k, v in c.items()})
