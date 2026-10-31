"""Write a synthetic 2-bedroom apartment in the CubiCasa5K model.svg format.

Used only to test the downstream pipeline (SVG -> quantities -> 3D/IFC) while the real
dataset is not reachable from this workspace. Conventions copied from the CubiCasa loader
(floortrans/loaders/house.py): <g id="Wall"> / <g id="Door"> / <g id="Window"> each hold a
<polygon points="x,y x,y ... ">, rooms are <g class="Space <Type>">. 100 SVG units = 1 m.
"""
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sample_plan" / "model.svg"

U = 100  # svg units per metre
EXT, INT = 0.25, 0.12  # wall thickness (m)

# wall centre-lines (x1, y1, x2, y2, thickness) in metres
walls = [
    (0, 0, 10, 0, EXT), (10, 0, 10, 8, EXT), (10, 8, 0, 8, EXT), (0, 8, 0, 0, EXT),
    (0, 4.5, 4, 4.5, INT), (4, 4.5, 4, 8, INT), (6.5, 5, 6.5, 8, INT), (6.5, 5, 10, 5, INT),
    (4, 5.5, 6.5, 5.5, INT), (6, 0, 6, 3.5, INT), (6, 3.5, 10, 3.5, INT),
]
# openings: (wall index, start along wall (m), width (m), kind)
openings = [
    (1, 4.0, 1.0, "Door"),   # entrance, east wall into hall
    (5, 0.1, 0.9, "Door"),   # bedroom 1
    (8, 0.8, 0.8, "Door"),   # bath
    (7, 0.5, 0.9, "Door"),   # bedroom 2
    (9, 1.5, 0.9, "Door"),   # kitchen <-> living
    (10, 1.0, 0.9, "Door"),  # kitchen <-> hall
    (0, 1.0, 2.0, "Window"), (0, 7.5, 1.5, "Window"), (3, 5.0, 1.5, "Window"),
    (3, 1.0, 1.5, "Window"), (2, 7.5, 1.5, "Window"), (2, 1.0, 1.5, "Window"),
    (1, 6.0, 1.2, "Window"), (2, 4.4, 0.6, "Window"),
]
rooms = {  # centre-line polygons; clear room area = polygon minus wall footprints
    "LivingRoom": [(0, 0), (6, 0), (6, 4.5), (0, 4.5)],
    "Kitchen": [(6, 0), (10, 0), (10, 3.5), (6, 3.5)],
    "Bedroom": [(0, 4.5), (4, 4.5), (4, 8), (0, 8)],
    "Bedroom2": [(6.5, 5), (10, 5), (10, 8), (6.5, 8)],
    "Bath": [(4, 5.5), (6.5, 5.5), (6.5, 8), (4, 8)],
    "Entry": [(4, 4.5), (6, 4.5), (6, 3.5), (10, 3.5), (10, 5), (6.5, 5), (6.5, 5.5), (4, 5.5)],
}


def wall_rect(x1, y1, x2, y2, t, ext=True):
    """Rectangle around a centre-line, extended by t/2 at both ends so corners close."""
    h = t / 2
    if y1 == y2:
        a, b = sorted((x1, x2))
        return box(a - h, y1 - h, b + h, y1 + h)
    a, b = sorted((y1, y2))
    return box(x1 - h, a - h, x1 + h, b + h)


def open_rect(w, s, width):
    x1, y1, x2, y2, t = w
    h = t / 2
    if y1 == y2:
        a = min(x1, x2) + s
        return box(a, y1 - h, a + width, y1 + h)
    a = min(y1, y2) + s
    return box(x1 - h, a, x1 + h, a + width)


def pts(poly):
    return " ".join(f"{x * U:.1f},{y * U:.1f}" for x, y in list(poly.exterior.coords)[:-1]) + " "


wall_polys = [wall_rect(*w) for w in walls]
footprint = unary_union(wall_polys)
out = ['<?xml version="1.0" encoding="UTF-8"?>',
       f'<svg xmlns="http://www.w3.org/2000/svg" width="{10.6 * U:.0f}" height="{8.6 * U:.0f}" '
       f'viewBox="{-0.3 * U} {-0.3 * U} {10.6 * U} {8.6 * U}">',
       '<g class="Model"><g class="Floor"><g class="Floorplan">']
for i, (w, p) in enumerate(zip(walls, wall_polys)):
    cls = "Wall External" if w[4] == EXT else "Wall"
    out.append(f'<g id="Wall" class="{cls}"><polygon points="{pts(p)}"/>')
    for wi, s, width, kind in openings:
        if wi == i:
            out.append(f'<g id="{kind}" class="{kind} Regular"><polygon points="{pts(open_rect(w, s, width))}"/></g>')
    out.append("</g>")
for name, poly in rooms.items():
    clear = Polygon(poly).difference(footprint)
    clear = max(getattr(clear, "geoms", [clear]), key=lambda g: g.area)
    out.append(f'<g class="Space {name.rstrip("2")}"><polygon points="{pts(clear)}"/></g>')
out += ["</g></g></g>", "</svg>"]
OUT.parent.mkdir(parents=True, exist_ok=True)
open(OUT, "w").write("\n".join(out))
print("rooms clear area m2:", {k: round(Polygon(v).difference(footprint).area, 2) for k, v in rooms.items()})
