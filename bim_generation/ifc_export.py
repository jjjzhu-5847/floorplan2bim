"""Step 4b - BIM export: metric model -> IFC4 file (opens in Revit, BIMvision, BlenderBIM, IfcOpenShell).

Verified on the synthetic sample: every element tessellates and wall volumes after opening cuts
match the hand calculation.
"""
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import H_WALL, H_DOOR, SILL, H_WIN_TOP  # noqa: E402


def to_ifc(m, path, name="Floor plan to BIM"):
    """Write the metric model as an IFC4 file: IfcWall (+Qto), IfcOpeningElement, IfcDoor/IfcWindow, IfcSpace, IfcSlab."""
    import ifcopenshell
    import ifcopenshell.api as api
    f = api.run("project.create_file", version="IFC4")
    proj = api.run("root.create_entity", f, ifc_class="IfcProject", name=name)
    api.run("unit.assign_unit", f)  # SI metres
    ctx = api.run("context.add_context", f, context_type="Model")
    body = api.run("context.add_context", f, context_type="Model", context_identifier="Body",
                   target_view="MODEL_VIEW", parent=ctx)
    site = api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    bldg = api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    storey = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level 1")
    api.run("aggregate.assign_object", f, relating_object=proj, products=[site])
    api.run("aggregate.assign_object", f, relating_object=site, products=[bldg])
    api.run("aggregate.assign_object", f, relating_object=bldg, products=[storey])

    def place(el, p, u, z=0.0):
        mtx = np.eye(4)
        mtx[:3, 0] = [u[0], u[1], 0]
        mtx[:3, 1] = [-u[1], u[0], 0]
        mtx[:3, 3] = [p[0], p[1], z]
        api.run("geometry.edit_object_placement", f, product=el, matrix=mtx)

    for i, w in enumerate(m["walls"]):
        u = (w["p1"] - w["p0"]) / w["L"]
        wall = api.run("root.create_entity", f, ifc_class="IfcWall",
                       name=f'{"Ext" if w["external"] else "Int"}Wall-{i + 1}')
        rep = api.run("geometry.add_wall_representation", f, context=body, length=w["L"],
                      height=H_WALL, thickness=w["t"], offset=-w["t"] / 2)
        api.run("geometry.assign_representation", f, product=wall, representation=rep)
        place(wall, w["p0"], u)
        api.run("spatial.assign_container", f, relating_structure=storey, products=[wall])
        qto = api.run("pset.add_qto", f, product=wall, name="Qto_WallBaseQuantities")
        api.run("pset.edit_qto", f, qto=qto, properties={"Length": float(w["L"]), "Height": H_WALL, "Width": float(w["t"])})
        for o in w["openings"]:
            z0, h = (0.0, H_DOOR) if o["kind"] == "Door" else (SILL, H_WIN_TOP - SILL)
            op = api.run("root.create_entity", f, ifc_class="IfcOpeningElement")
            orep = api.run("geometry.add_wall_representation", f, context=body, length=o["width"],
                           height=h, thickness=w["t"] + 0.1, offset=-(w["t"] + 0.1) / 2)
            api.run("geometry.assign_representation", f, product=op, representation=orep)
            place(op, w["p0"] + u * o["s0"], u, z0)
            api.run("feature.add_feature", f, feature=op, element=wall)
            fill = api.run("root.create_entity", f, ifc_class="IfcDoor" if o["kind"] == "Door" else "IfcWindow")
            frep = api.run("geometry.add_wall_representation", f, context=body, length=o["width"],
                           height=h, thickness=0.05, offset=-0.025)
            api.run("geometry.assign_representation", f, product=fill, representation=frep)
            place(fill, w["p0"] + u * o["s0"], u, z0)
            api.run("feature.add_filling", f, opening=op, element=fill)
            api.run("spatial.assign_container", f, relating_structure=storey, products=[fill])
    for i, sp in enumerate(m["spaces"]):
        space = api.run("root.create_entity", f, ifc_class="IfcSpace", name=f'{sp["type"]}-{i + 1}')
        pts = [tuple(map(float, p)) for p in list(sp["poly"].exterior.coords)[:-1]]
        rep = api.run("geometry.add_slab_representation", f, context=body, depth=H_WALL, polyline=pts)
        api.run("geometry.assign_representation", f, product=space, representation=rep)
        place(space, (0, 0), (1, 0))
        api.run("aggregate.assign_object", f, relating_object=storey, products=[space])
        qto = api.run("pset.add_qto", f, product=space, name="Qto_SpaceBaseQuantities")
        api.run("pset.edit_qto", f, qto=qto, properties={"NetFloorArea": float(sp["poly"].area)})
    outline = unary_union([w["poly"] for w in m["walls"]] + [sp["poly"] for sp in m["spaces"]])
    outline = Polygon(outline.exterior) if outline.geom_type == "Polygon" else outline.convex_hull
    slab = api.run("root.create_entity", f, ifc_class="IfcSlab", name="Floor slab")
    rep = api.run("geometry.add_slab_representation", f, context=body, depth=0.2,
                  polyline=[tuple(map(float, p)) for p in list(outline.exterior.coords)[:-1]])
    api.run("geometry.assign_representation", f, product=slab, representation=rep)
    place(slab, (0, 0), (1, 0), -0.2)
    api.run("spatial.assign_container", f, relating_structure=storey, products=[slab])
    f.write(path)
    return f
