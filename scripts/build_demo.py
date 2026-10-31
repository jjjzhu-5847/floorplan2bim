"""One command for the demo page: sample plan + demo images + error demo -> outputs/web/index.html.

    python scripts/build_demo.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print(">", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    run("samples/make_sample_svg.py")
    run("scripts/run_pipeline.py", "--svg", "samples/sample_plan/model.svg", "--name", "sample",
        "--label", "Synthetic 2-bedroom (test plan)")
    for i in ["1316", "12539"]:
        img = ROOT / "samples" / "demo_images" / f"input_{i}.png"
        if img.exists():
            run("scripts/run_pipeline.py", "--image", str(img), "--name", i, "--label", f"CubiCasa plan {i}")
    run("evaluation/error_propagation_demo.py")
    run("web/build_page.py", "--plans", "sample", "1316", "12539")


if __name__ == "__main__":
    main()
