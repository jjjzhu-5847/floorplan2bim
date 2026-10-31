"""Map raw plan coordinates between the 512-px letterboxed working image and the original image."""
from shapely import affinity


def to_original(raw, transform):
    """transform = (scale, left, top) from recognition.segformer.letterbox.
    x_orig = (x_512 - left) / scale, same for y."""
    s, left, top = transform

    def f(p):
        p = affinity.translate(p, xoff=-left, yoff=-top)
        return affinity.scale(p, xfact=1 / s, yfact=1 / s, origin=(0, 0))
    out = {k: v for k, v in raw.items() if k not in ("walls", "openings", "spaces")}
    out["walls"] = [dict(w, poly=f(w["poly"])) for w in raw["walls"]]
    out["openings"] = [dict(o, poly=f(o["poly"])) for o in raw["openings"]]
    out["spaces"] = [dict(sp, poly=f(sp["poly"])) for sp in raw["spaces"]]
    return out
