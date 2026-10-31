"""Training plans the public SegFormer checkpoint never saw (RQ4 training data, task 3.2).

    python evaluation/train_unseen.py
The checkpoint (github.com/alifarzadjamali/floor-plan-document-intelligence, configs/segmentation.yaml:
seed 42, train_subset 1200) was trained on random.Random(42).sample(train_lines, 1200), where train_lines are the
non-empty lines of the official train.txt in file order (src/floorplan_di/data/{cubicasa,segmentation}.py).
Repeating that draw gives the 1,200 seen plans; the other 3,000 are unseen. Note: the checkpoint was *selected* on
the official validation split, so validation results are slightly optimistic; the test split is untouched.
Writes outputs/rq4/train_seen.txt and train_unseen.txt (same format as the official split files).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SEED, SUBSET = 42, 1200


def main():
    lines = [ln.strip().replace("\\", "/") for ln in (config.CUBICASA_DIR / "train.txt").read_text(encoding="utf-8-sig").splitlines()]
    lines = [ln for ln in lines if ln]
    seen = set(random.Random(SEED).sample(lines, SUBSET))
    out = config.OUTPUTS_DIR / "rq4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_seen.txt").write_text("\n".join(sorted(seen)) + "\n")
    (out / "train_unseen.txt").write_text("\n".join(ln for ln in lines if ln not in seen) + "\n")
    print(f"{len(lines)} training plans: {len(seen)} seen by the checkpoint, {len(lines) - len(seen)} unseen -> {out}")


if __name__ == "__main__":
    main()
