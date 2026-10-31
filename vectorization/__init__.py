"""Step 2 - vectorization: pixel mask -> walls / doors / windows / rooms as polygons ("raw plan" dict).

Raw plan dict (used everywhere downstream, in drawing units = pixels or SVG units):
    {"walls":    [{"poly": shapely Polygon, "external": bool}, ...],
     "openings": [{"poly": Polygon, "kind": "Door" | "Window"}, ...],
     "spaces":   [{"poly": Polygon, "type": "Bedroom" | ... | "Room"}, ...]}
"""
