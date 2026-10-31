"""Mask -> raw plan. Manhattan vectorizer (skeleton + Hough, snapped to H/V).

wall/window mask -> skeleton -> line segments -> horizontal/vertical centre-lines -> merged collinear runs ->
thickness measured across each run. Each door blob is hosted by the wall run whose core holds most of its
pixels; the opening width is the extent of those core pixels and the wall is continued through it. A window
blob lying on a wall line becomes a window opening. Exterior walls run along the outline of a built-up region.
Rooms: everything inside the outline minus walls ("footprint") or room-class pixels minus walls ("pixels").
Pixel parameters are set for a 512-px mask and scaled with the mask size, so any working size can be used.
Output is the raw plan dict (see vectorization/__init__.py).
"""
import sys
from pathlib import Path

import numpy as np
import cv2
from skimage.morphology import skeletonize
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from vectorization.svg_io import polygon_parts  # noqa: E402


def _merge(segs, tol=3, gap=4):
    """segs: list of (coord, a, b, thick). Merge runs on the same wall: centre-lines closer than tol or half
    the thinner wall (a ragged mask gives a wavy skeleton) that overlap or nearly touch."""
    segs = sorted(segs)
    out = []
    for c, a, b, t in segs:
        for o in out:
            if abs(o[0] - c) <= max(tol, min(o[3], t) / 2) and a <= o[2] + gap and b >= o[1] - gap:
                w = (o[2] - o[1]) / max(o[2] - o[1] + b - a, 1e-6)
                o[0] = o[0] * w + c * (1 - w)
                o[1], o[2] = min(o[1], a), max(o[2], b)
                o[3] = max(o[3], t)
                break
        else:
            out.append([c, a, b, t])
    return out


def _thickness(wall, horiz, c, a, b):
    """Median width of the wall mask measured across the run every 2 px. 2 x distance transform + 1
    overshoots by 1-2 px, which doubles a 4-px interior wall at the 512-px working size."""
    ci, widths = int(round(c)), []
    for s in range(max(int(a), 0), min(int(b), (wall.shape[1] if horiz else wall.shape[0]) - 1) + 1, 2):
        line = wall[:, s] if horiz else wall[s, :]
        if not 0 <= ci < len(line) or not line[ci]:
            continue
        lo = hi = ci
        while lo > 0 and line[lo - 1]:
            lo -= 1
        while hi < len(line) - 1 and line[hi + 1]:
            hi += 1
        widths.append(hi - lo + 1)
    return float(np.median(widths)) if widths else None


def vectorize_mask(mask, door_gap=config.VEC_DOOR_GAP_PX, room_mode=config.VEC_ROOM_MODE):
    """mask: HxW uint8 with classes 0 bg, 1 room, 2 wall, 3 door, 4 window. Returns raw plan dict (pixels)."""
    u = max(mask.shape) / config.SEG_INPUT_SIZE  # pixel parameters below are for a 512-px mask
    door_gap = (door_gap[0] * u, door_gap[1] * u)
    wall = np.isin(mask, [2, 4]).astype(np.uint8)
    k = max(3, int(round(3 * u)) | 1)  # close at the working size: an outlined wall is one wall, not two
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    dist = cv2.distanceTransform(wall, cv2.DIST_L2, 3)
    skel = skeletonize(wall > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=round(8 * u), minLineLength=6 * u, maxLineGap=3 * u)
    H, V = [], []
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):  # (N,1,4) in OpenCV 4, (N,4) in 5
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
    H, V = _merge(H, 3 * u, 4 * u), _merge(V, 3 * u, 4 * u)
    for runs, horiz in ((H, True), (V, False)):
        for r in runs:
            r[3] = min(_thickness(wall, horiz, *r[:3]) or r[3], r[3])  # the profile can run into wall-coloured blobs
    door = cv2.morphologyEx((mask == 3).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    window = (mask == 4).astype(np.uint8)

    # doors: each door blob is hosted by the wall run whose core (|offset| <= t/2) holds most of its pixels.
    # Only core pixels count, so swing arcs and leaves drawn beside a wall do not widen the opening or pick
    # the wrong wall; this also finds doors next to a corner, where there is no wall run on one side.
    openings = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(door, 8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 6 * u * u:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        ys, xs = ys + y, xs + x
        best = None
        for runs, horiz in ((H, True), (V, False)):
            perp, along = (ys, xs) if horiz else (xs, ys)
            for r in runs:
                core = np.abs(perp - r[0]) <= max(r[3] / 2, 1.5)
                if core.sum() < 3:
                    continue
                s0, s1 = float(along[core].min()), float(along[core].max() + 1)
                if max(r[1] - s1, s0 - r[2], 0) > max(s1 - s0, 4 * u):  # must touch or lie inside the run
                    continue
                if best is None or core.sum() > best[0]:
                    best = (int(core.sum()), runs, r, horiz, s0, s1)
        if best is None or not door_gap[0] <= best[5] - best[4] <= door_gap[1]:
            continue
        _, runs, r, horiz, s0, s1 = best
        openings.append(dict(kind="Door", horiz=horiz, c=r[0], s0=s0, s1=s1, t=r[3]))
        # the wall continues through the opening: extend the host run and merge collinear runs it now reaches
        r[1], r[2] = min(r[1], s0), max(r[2], s1)
        for q in [q for q in runs if q is not r and abs(q[0] - r[0]) <= 4 * u and q[1] <= r[2] + 4 * u and q[2] >= r[1] - 4 * u]:
            r[1], r[2], r[3] = min(r[1], q[1]), max(r[2], q[2]), max(r[3], q[3])
            runs.remove(q)
    # gaps between collinear runs: a door whose blob missed the wall core (only the swing was labelled) when door
    # pixels lie within one opening width of the gap, otherwise a broken wall when the gap is shorter than 2 walls
    for runs, horiz in ((H, True), (V, False)):
        runs.sort(key=lambda r: (r[0], r[1]))
        i = 0
        while i < len(runs) - 1:
            a, b = runs[i], runs[i + 1]
            j = i + 1
            while j < len(runs) and not (abs(runs[j][0] - a[0]) <= max(4 * u, a[3] / 2) and runs[j][1] >= a[2]):
                j += 1
            if j == len(runs):
                i += 1
                continue
            b = runs[j]
            g0, g1, t = a[2], b[1], max(a[3], b[3])
            c = (a[0] + b[0]) / 2
            if door_gap[0] <= g1 - g0 <= door_gap[1]:
                w = g1 - g0
                lo, hi = int(max(c - w - t / 2, 0)), int(c + w + t / 2) + 1
                zone = door[lo:hi, int(g0):int(g1)] if horiz else door[int(g0):int(g1), lo:hi]
                is_door = zone.size and zone.mean() > 0.03
            else:
                is_door = False
            if is_door or 0 < g1 - g0 <= 2 * t:
                if is_door:
                    openings.append(dict(kind="Door", horiz=horiz, c=c, s0=g0, s1=g1, t=t))
                a[0], a[2], a[3] = c, max(a[2], b[2]), t
                runs.remove(b)
                continue
            i += 1
    # snap wall ends to perpendicular walls (close L and T junctions)
    def snap(runs, cross):
        for r in runs:
            for end in (1, 2):
                best = None
                for q in cross:
                    if q[1] - 6 * u <= r[0] <= q[2] + 6 * u and abs(q[0] - r[end]) <= max(8 * u, 1.5 * r[3]):
                        if best is None or abs(q[0] - r[end]) < abs(best - r[end]):
                            best = q[0]
                if best is not None:
                    r[end] = best
    snap(H, V)
    snap(V, H)
    def connected(r, horiz, end):
        x, y = (r[end], r[0]) if horiz else (r[0], r[end])
        for q in H + V:
            if q is r:
                continue
            qh = any(q is h for h in H)
            qx0, qy0, qx1, qy1 = (q[1], q[0], q[2], q[0]) if qh else (q[0], q[1], q[0], q[2])
            d = np.hypot(max(qx0 - x, 0, x - qx1), max(qy0 - y, 0, y - qy1))
            if d <= max(r[3], q[3]) / 2 + 3 * u:
                return True
        return False
    for runs, horiz in ((H, True), (V, False)):
        # a wall free at both ends and shorter than 10 thicknesses is furniture or a fixture, not a wall
        runs[:] = [r for r in runs if r[2] - r[1] >= 10 * r[3] or connected(r, horiz, 1) or connected(r, horiz, 2)]
    walls = []
    for runs, horiz in ((H, True), (V, False)):
        for c, a, b, t in runs:
            t = max(t, 2.0 * u)
            if b - a < config.VEC_MIN_WALL_PX * u:
                continue
            h = t / 2
            poly = box(a - h, c - h, b + h, c + h) if horiz else box(c - h, a - h, c + h, b + h)
            walls.append(dict(poly=poly, horiz=horiz, c=c, a=a, b=b, t=t))
    # window openings: window blobs lying on a wall line
    n, lab, stats, _ = cv2.connectedComponentsWithStats(window, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 10 * u * u:
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
    # rooms: everything inside the building outline ("footprint") or room-class pixels, minus walls
    solid = np.isin(mask, [1, 2, 3, 4]).astype(np.uint8)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 2000 * u * u]
    inside = np.zeros_like(mask)
    cv2.drawContours(inside, cnts, -1, 1, cv2.FILLED)
    wall_px = np.zeros_like(mask)
    for W in walls:
        x0, y0, x1, y1 = [int(round(v)) for v in W["poly"].bounds]
        wall_px[y0:y1 + 1, x0:x1 + 1] = 1
    room = ((inside == 1) if room_mode == "footprint" else (mask == 1)) & (wall_px == 0)
    room = cv2.morphologyEx(room.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # open spaces (balcony, terrace): a room whose edge meets the outside, not walls, for over a quarter of its
    # length. They stay in the floor area but leave the building outline, so the wall between a flat and its
    # balcony counts as exterior (as in the ground truth)
    spaces, open_px = [], np.zeros_like(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(room, 4)
    for i in range(1, n):
        if stats[i][4] < config.VEC_MIN_ROOM_PX * u * u:
            continue
        x, y, w, h = stats[i][:4]
        x0, y0 = max(x - 2, 0), max(y - 2, 0)
        comp = (lab[y0:y + h + 2, x0:x + w + 2] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, np.ones((3, 3), np.uint8)) > 0) & (comp == 0)
        if ring.any() and (ring & (inside[y0:y + h + 2, x0:x + w + 2] == 0)).sum() > 0.25 * ring.sum():
            open_px[y0:y + h + 2, x0:x + w + 2] |= comp
        c, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(c, key=cv2.contourArea), 2 * u, True) + np.array([x0, y0])
        if len(c) >= 3:
            spaces += [dict(poly=p, type="Room") for p in polygon_parts(Polygon(c[:, 0, :]).buffer(0))]
    closed = (room == 1) & (open_px == 0)

    def sides(W):
        """Per sample along the wall: is there a closed room / an open space just beyond either face?"""
        k = max(int((W["b"] - W["a"]) / (2 * u)), 2)
        s_ = np.linspace(W["a"], W["b"], k)
        res = []
        for off in (-1, 1):
            c_ = W["c"] + off * (W["t"] / 2 + 2 * u)
            xs, ys = (s_, np.full(k, c_)) if W["horiz"] else (np.full(k, c_), s_)
            xs = np.clip(np.round(xs).astype(int), 0, mask.shape[1] - 1)
            ys = np.clip(np.round(ys).astype(int), 0, mask.shape[0] - 1)
            res.append((closed[ys, xs], open_px[ys, xs] == 1))
        return res
    # railings: walls with no closed room on either side and an open space on one side
    keep = []
    for W in walls:
        (c1, o1), (c2, o2) = sides(W)
        if not (np.mean(c1 | c2) < 0.3 and np.mean(o1 | o2) > 0.3):
            keep.append(W)
    walls = keep
    # exterior = walls running along the outline of the built-up region without its open spaces
    core = ((inside == 1) & (open_px == 0)).astype(np.uint8)
    cc, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = [p.exterior for c in cc if cv2.contourArea(c) > 2000 * u * u
                for p in polygon_parts(Polygon(c[:, 0, :]).buffer(0))]
    outline = unary_union(outlines) if outlines else None

    def external(W):
        """Most of the centre-line runs along the building outline (touching it at one end is not enough)."""
        if outline is None:
            return False
        k = max(int(W["b"] - W["a"]), 2)
        s_ = np.linspace(W["a"], W["b"], k)
        xy = np.c_[s_, np.full(k, W["c"])] if W["horiz"] else np.c_[np.full(k, W["c"]), s_]
        return float(np.mean(shapely.distance(outline, shapely.points(xy)) <= W["t"] / 2 + 3 * u)) >= 0.5
    raw_walls = [dict(poly=W["poly"], external=external(W)) for W in walls]
    return dict(walls=raw_walls, openings=raw_open, spaces=spaces, n_outlines=len(outlines))
