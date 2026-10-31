"""Metric building model: walls as centre-line rectangles with openings placed along them, rooms as polygons."""
import numpy as np
from shapely.geometry import Polygon


def rect_axis(poly):
    """Centre-line (p0, p1), length and thickness of a (near-)rectangular polygon.
    Direction is normalised so p0 -> p1 points to +x (or +y for vertical walls)."""
    r = poly.minimum_rotated_rectangle
    c = np.array(r.exterior.coords)[:4]
    e = [c[(i + 1) % 4] - c[i] for i in range(4)]
    i = int(np.argmax([np.hypot(*v) for v in e[:2]]))
    L, t = float(np.hypot(*e[i])), float(np.hypot(*e[1 - i]))
    u = e[i] / L if L > 0 else np.array([1.0, 0.0])
    ctr = c.mean(axis=0)
    p0, p1 = ctr - u * L / 2, ctr + u * L / 2
    if (p1 - p0)[0] < -1e-9 or (abs((p1 - p0)[0]) < 1e-9 and (p1 - p0)[1] < 0):
        p0, p1 = p1, p0
    return p0, p1, L, t


def _scaled(p, s):
    return Polygon(np.array(p.exterior.coords) * s)


def build_model(raw, px_per_m):
    """raw plan in drawing units -> metric model. Each opening is attached to the wall it overlaps most."""
    s = 1.0 / px_per_m
    walls = []
    for w in raw["walls"]:
        p0, p1, L, t = rect_axis(w["poly"])
        if L <= 0:
            continue
        walls.append(dict(p0=p0 * s, p1=p1 * s, L=float(L * s), t=float(t * s), external=bool(w["external"]),
                          poly=_scaled(w["poly"], s), openings=[]))
    dropped = 0
    for o in raw["openings"]:
        op = _scaled(o["poly"], s)
        overlaps = [w["poly"].intersection(op).area for w in walls]
        if not overlaps or max(overlaps) <= 0:
            dropped += 1
            continue
        w = walls[int(np.argmax(overlaps))]
        u = (w["p1"] - w["p0"]) / w["L"]
        proj = [float(np.dot(np.array(pt) - w["p0"], u)) for pt in op.exterior.coords]
        a, b = max(min(proj), 0.0), min(max(proj), w["L"])
        if b > a:
            w["openings"].append(dict(kind=o["kind"], s0=float(a), s1=float(b), width=float(b - a)))
    spaces = [dict(type=sp["type"], poly=_scaled(sp["poly"], s)) for sp in raw["spaces"]]
    return dict(walls=walls, spaces=spaces, px_per_m=px_per_m, dropped_openings=dropped)
