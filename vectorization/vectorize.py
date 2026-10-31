"""Mask -> raw plan. First-pass Manhattan vectorizer (skeleton + Hough, snapped to H/V).

IMPROVE THIS FILE FIRST (PROJECT_PLAN.md task 2.x). Known misses: thin interior walls, doors whose
swing arc is labelled as door, stairs labelled as door, two storeys drawn side by side.

Original notes:

wall/window mask -> skeleton -> line segments -> snapped horizontal/vertical centre-lines ->
merged collinear runs -> thickness from the distance transform. A gap between two collinear
wall runs that overlaps a door blob becomes a door opening. A window blob that sits on a wall
line becomes a window opening. Output is the raw plan dict (see vectorization/__init__.py).
"""
import sys
from pathlib import Path

import numpy as np
import cv2
from skimage.morphology import skeletonize
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _merge(segs, tol=3, gap=4):
    """segs: list of (coord, a, b, thick). Merge runs on the same line coordinate."""
    segs = sorted(segs)
    out = []
    for c, a, b, t in segs:
        for o in out:
            if abs(o[0] - c) <= tol and a <= o[2] + gap and b >= o[1] - gap:
                w = (o[2] - o[1]) / max(o[2] - o[1] + b - a, 1e-6)
                o[0] = o[0] * w + c * (1 - w)
                o[1], o[2] = min(o[1], a), max(o[2], b)
                o[3] = max(o[3], t)
                break
        else:
            out.append([c, a, b, t])
    return out


def vectorize_mask(mask, door_gap=config.VEC_DOOR_GAP_PX):
    """mask: HxW uint8 with classes 0 bg, 1 room, 2 wall, 3 door, 4 window. Returns raw plan dict (pixels)."""
    wall = np.isin(mask, [2, 4]).astype(np.uint8)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(wall, cv2.DIST_L2, 3)
    skel = skeletonize(wall > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=8, minLineLength=6, maxLineGap=3)
    H, V = [], []
    for x1, y1, x2, y2 in (lines[:, 0] if lines is not None else []):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx >= 3 * dy:
            ys = np.linspace(y1, y2, max(dx, 2)).round().astype(int)
            xs = np.linspace(x1, x2, max(dx, 2)).round().astype(int)
            t = 2 * float(np.median(dist[ys, xs])) + 1
            H.append(((y1 + y2) / 2, min(x1, x2), max(x1, x2), t))
        elif dy >= 3 * dx:
            ys = np.linspace(y1, y2, max(dy, 2)).round().astype(int)
            xs = np.linspace(x1, x2, max(dy, 2)).round().astype(int)
            t = 2 * float(np.median(dist[ys, xs])) + 1
            V.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2), t))
    H, V = _merge(H), _merge(V)
    door = (mask == 3).astype(np.uint8)
    window = (mask == 4).astype(np.uint8)

    # bridge door gaps: collinear runs separated by a gap that overlaps door pixels
    openings = []
    for runs, horiz in ((H, True), (V, False)):
        runs.sort(key=lambda r: (round(r[0] / 4), r[1]))
        i = 0
        while i < len(runs) - 1:
            a, b = runs[i], runs[i + 1]
            g0, g1 = a[2], b[1]
            if abs(a[0] - b[0]) <= 4 and door_gap[0] <= g1 - g0 <= door_gap[1]:
                c = int(round((a[0] + b[0]) / 2))
                band = door[max(c - 6, 0):c + 7, int(g0):int(g1)] if horiz else door[int(g0):int(g1), max(c - 6, 0):c + 7]
                if band.size and band.mean() > 0.05:
                    t = max(a[3], b[3])
                    runs[i] = [(a[0] + b[0]) / 2, a[1], b[2], t]
                    runs.pop(i + 1)
                    openings.append(dict(kind="Door", horiz=horiz, c=c, s0=g0, s1=g1, t=t))
                    continue
            i += 1
    # snap wall ends to perpendicular walls (close L and T junctions)
    def snap(runs, cross):
        for r in runs:
            for end in (1, 2):
                best = None
                for q in cross:
                    if q[1] - 6 <= r[0] <= q[2] + 6 and abs(q[0] - r[end]) <= 8:
                        if best is None or abs(q[0] - r[end]) < abs(best - r[end]):
                            best = q[0]
                if best is not None:
                    r[end] = best
    snap(H, V)
    snap(V, H)
    walls = []
    for runs, horiz in ((H, True), (V, False)):
        for c, a, b, t in runs:
            t = max(t, 2.0)
            if b - a < config.VEC_MIN_WALL_PX:
                continue
            h = t / 2
            poly = box(a - h, c - h, b + h, c + h) if horiz else box(c - h, a - h, c + h, b + h)
            walls.append(dict(poly=poly, horiz=horiz, c=c, a=a, b=b, t=t))
    # window openings: window blobs lying on a wall line
    n, lab, stats, _ = cv2.connectedComponentsWithStats(window, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 10:
            continue
        horiz = w >= h
        r = box(x, y, x + w, y + h)
        if any(W["horiz"] == horiz and W["poly"].intersects(r) for W in walls):
            openings.append(dict(kind="Window", poly=r))
    raw_open = []
    for o in openings:
        if o["kind"] == "Door":
            h = o["t"] / 2
            poly = box(o["s0"], o["c"] - h, o["s1"], o["c"] + h) if o["horiz"] else box(o["c"] - h, o["s0"], o["c"] + h, o["s1"])
            raw_open.append(dict(poly=poly, kind="Door"))
        else:
            raw_open.append(dict(poly=o["poly"], kind="Window"))
    # exterior = walls on the outline of the largest built-up region
    solid = np.isin(mask, [1, 2, 3, 4]).astype(np.uint8)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = [Polygon(c[:, 0, :]).buffer(0).exterior for c in cnts if cv2.contourArea(c) > 2000]
    raw_walls = [dict(poly=W["poly"], external=any(W["poly"].distance(o) < 4 for o in outlines)) for W in walls]
    # rooms: room pixels minus walls, split into regions
    wall_px = np.zeros_like(mask)
    for W in walls:
        x0, y0, x1, y1 = [int(round(v)) for v in W["poly"].bounds]
        wall_px[y0:y1 + 1, x0:x1 + 1] = 1
    room = ((mask == 1) & (wall_px == 0)).astype(np.uint8)
    room = cv2.morphologyEx(room, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    spaces = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(room, 4)
    for i in range(1, n):
        if stats[i][4] < config.VEC_MIN_ROOM_PX:
            continue
        c, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(c, key=cv2.contourArea), 2, True)
        if len(c) >= 3:
            spaces.append(dict(poly=Polygon(c[:, 0, :]).buffer(0), type="Room"))
    return dict(walls=raw_walls, openings=raw_open, spaces=spaces)
