"""Fetch five small real CubiCasa plan images (512 px copies) used for demos before the full dataset
is downloaded. Source: github.com/daegeun-kim/neural_floorplan (outputs/vectorization/phase4_vectorization/<id>/input.png).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

REPO = "https://github.com/daegeun-kim/neural_floorplan.git"
IDS = ["1316", "10026", "10029", "12539", "13736"]


def main():
    out = config.ROOT / "samples" / "demo_images"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO, tmp], check=True)
        for i in IDS:
            with open(out / f"input_{i}.png", "wb") as f:
                subprocess.run(["git", "-C", tmp, "show", f"HEAD:outputs/vectorization/phase4_vectorization/{i}/input.png"],
                               check=True, stdout=f)
    print("saved", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
