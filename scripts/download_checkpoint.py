"""Download the public SegFormer-B0 CubiCasa5K checkpoint (44.8 MB) into models/ and verify its sha256.

Source: github.com/alifarzadjamali/floor-plan-document-intelligence (MIT licence),
        results/segmentation/phase2_dev/checkpoints/best.pt
Uses git (a sparse, blob-less clone) because the file is not on a release page.
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

REPO = "https://github.com/alifarzadjamali/floor-plan-document-intelligence.git"
PATH = "results/segmentation/phase2_dev/checkpoints/best.pt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    dst = config.SEGFORMER_CKPT
    if dst.exists() and sha256(dst) == config.SEGFORMER_SHA256:
        print(f"{dst} already present and verified")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO, tmp], check=True)
        with open(dst, "wb") as f:
            subprocess.run(["git", "-C", tmp, "show", f"HEAD:{PATH}"], check=True, stdout=f)
    got = sha256(dst)
    if got != config.SEGFORMER_SHA256:
        sys.exit(f"sha256 mismatch: {got}")
    print(f"saved {dst} ({dst.stat().st_size / 1e6:.1f} MB), sha256 verified")


if __name__ == "__main__":
    main()
