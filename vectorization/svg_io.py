"""Read CubiCasa5K model.svg (ground truth), write a raw plan back to SVG, rasterize a raw plan to a mask.

CubiCasa SVG conventions (from the official loader floortrans/loaders/house.py):
  <g id="Wall" class="Wall ...">    first <polygon> child = wall outline
  <g id="Door"> / <g id="Window">   first <polygon> child = opening rectangle in the wall
  <g class="Space Bedroom ...">     first <polygon> child = room outline
  <g class="Floor">                 one per storey (a sheet can hold several storeys)
"""
import numpy as np
import cv2
from xml.dom import minidom
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _poly(g):
    pol = next((p for p in g.childNodes if p.nodeName == "polygon"), None)
    if pol is None:
        return None
    xy = []
    for a in pol.getAttribute("points").split():
        try:
            x, y = a.split(",")
            xy.append((float(x), float(y)))
        except ValueError:
            continue
    if len(xy) < 3:
        return None
    p = Polygon(xy)
    return p if p.is_valid else p.buffer(0)


def polygon_parts(p):
    """Non-empty Polygons of p. buffer(0) can turn a self-intersecting polygon into a MultiPolygon,
    but the raw plan holds single Polygons only."""
    return [g for g in getattr(p, "geoms", [p]) if g.geom_type == "Polygon" and g.area > 0]


def parse_svg(path, floor=None):
    """Returns raw plan dict + 'n_floors'. floor=None reads every storey; floor=k (0-based) reads only the
    k-th <g class="Floor">, which CubiCasa draws on F{k+1}_scaled.png. External walls: from the class name
    when present, otherwise geometric (walls touching the outline of all walls + rooms)."""
    doc = minidom.parse(str(path))
    floors = [g for g in doc.getElementsByTagName("g") if g.getAttribute("class").split()[:1] == ["Floor"]]
    root = doc if floor is None else floors[floor]
    walls, openings, spaces = [], [], []
    for g in root.getElementsByTagName("g"):
        gid, cls = g.getAttribute("id"), g.getAttribute("class")
        if gid == "Wall":
            p = _poly(g)
            if p is not None:
                walls += [dict(poly=q, external="External" in cls) for q in polygon_parts(p)]
        elif gid in ("Door", "Window"):
            p = _poly(g)
            if p is not None and p.area > 0:  # one opening, so keep its largest part
                openings.append(dict(poly=max(polygon_parts(p), key=lambda q: q.area), kind=gid))
        elif cls.startswith("Space "):
            p = _poly(g)
            if p is not None:
                spaces += [dict(poly=q, type=cls.split()[1]) for q in polygon_parts(p)]
    if walls and not any(w["external"] for w in walls):
        mark_external(walls, spaces)
    return dict(walls=walls, openings=openings, spaces=spaces, n_floors=max(len(floors), 1))


def mark_external(walls, spaces, tol_frac=0.5):
    """Geometric rule: a wall is external if it touches the outer boundary of the built footprint."""
    foot = unary_union([w["poly"] for w in walls] + [s["poly"] for s in spaces]).buffer(1).buffer(-1)
    outlines = [g.exterior for g in getattr(foot, "geoms", [foot])]
    for w in walls:
        t = min(w["poly"].minimum_rotated_rectangle.exterior.length / 4, 50)
        w["external"] = any(w["poly"].distance(o) <= t * tol_frac for o in outlines)


def write_svg(raw, path, size=None):
    """Write a raw plan in CubiCasa-like SVG (useful to inspect predictions in a browser)."""
    def pts(p):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in list(p.exterior.coords)[:-1]) + " "
    w, h = size or (1000, 1000)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
           '<g class="Model"><g class="Floor"><g class="Floorplan">']
    for s in raw["spaces"]:
        out.append(f'<g class="Space {s["type"]}"><polygon points="{pts(s["poly"])}" fill="#dfe7ee"/></g>')
    for wl in raw["walls"]:
        cls = "Wall External" if wl["external"] else "Wall"
        out.append(f'<g id="Wall" class="{cls}"><polygon points="{pts(wl["poly"])}" fill="#333"/></g>')
    for o in raw["openings"]:
        col = "#2a78d6" if o["kind"] == "Door" else "#e09a3a"
        out.append(f'<g id="{o["kind"]}" class="{o["kind"]}"><polygon points="{pts(o["poly"])}" fill="{col}"/></g>')
    out += ["</g></g></g>", "</svg>"]
    open(path, "w").write("\n".join(out))


def rasterize(raw, shape, scale=1.0, offset=(0, 0)):
    """Draw a raw plan into a 5-class mask (0 bg, 1 room, 2 wall, 3 door, 4 window).
    Draw order follows the public checkpoint's label precedence: window > door > wall > room."""
    m = np.zeros(shape, np.uint8)

    def fill(p, v):
        for g in getattr(p, "geoms", [p]):
            xy = (np.array(g.exterior.coords) * scale + np.array(offset)).round().astype(np.int32)
            cv2.fillPoly(m, [xy], int(v))
    for s in raw["spaces"]:
        fill(s["poly"], 1)
    for w in raw["walls"]:
        fill(w["poly"], 2)
    for o in raw["openings"]:
        fill(o["poly"], 3 if o["kind"] == "Door" else 4)
    return m
