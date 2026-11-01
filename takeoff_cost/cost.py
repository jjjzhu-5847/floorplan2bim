"""Cost = quantities x unit rates (config.UNIT_RATES). Scope: walls, floor finish, doors, windows only.
Walls are priced on gross face area (openings not deducted) and doors and windows separately, as the rate source
(Michigan Assessor's Manual, Vol. I, Unit-in-Place Costs) measures them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import UNIT_RATES  # noqa: E402


def cost(q, r=None):
    """Returns {item: USD, ..., "TOTAL": USD}."""
    r = r or UNIT_RATES
    items = {
        "Exterior walls (gross face)": q.get("ext_wall_gross_area_m2", 0) * r["ext_wall_m2"],
        "Interior partitions (gross face)": q.get("int_wall_gross_area_m2", 0) * r["int_wall_m2"],
        "Floor finish": q["floor_area_m2"] * r["floor_m2"],
        "Interior doors": q["doors_interior"] * r["door_int_ea"],
        "Entrance doors": q["doors_entry"] * r["door_entry_ea"],
        "Windows": q["window_area_m2"] * r["window_m2"],
    }
    items["TOTAL"] = sum(items.values())
    return items
