"""Task 5.5 - fine-tune the SegFormer-B0 checkpoint on more CubiCasa5K training plans.

    python recognition/train_segformer.py prepare [768]  # letterbox + rasterize masks -> data/seg_cache/ (~4 min)
    python recognition/train_segformer.py train [768 [init.pt]]  # fine-tune -> models/segformer_ours[_768].pt (best on val)
init: checkpoint to start from (default the public checkpoint), e.g. the 512-px model when training at 768.
Optional size (default 512): training at 768 px keeps doors larger; then recognise at that working size too.

Training plans: the 1,200 the public checkpoint was trained on plus 1,800 more drawn (seed 0) from the other 3,000
training plans; the remaining 1,200 training plans are seen by neither model and are kept for RQ4
(outputs/rq4/train_heldout.txt). Masks come from our own rasterizer (vectorization/svg_io.rasterize): a door is
the opening in the wall, not the swing, which is what the vectorizer and the door-width scale need.
Input: whole plan letterboxed to 512 px, as at inference. Loss: class-weighted cross-entropy + 0.5 x soft Dice
(foreground classes), as in the original training. Augmentation (on the GPU): flips, 90-degree rotations, brightness/contrast. Data are held in RAM.
Model selection: mean foreground IoU (room, wall, door, window) on the validation split after every epoch.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from recognition.segformer import MEAN, STD, letterbox, load_model  # noqa: E402
from vectorization.svg_io import parse_svg, rasterize  # noqa: E402
from evaluation.evaluate_cubicasa import list_plans  # noqa: E402

CACHE = config.DATA_DIR / "seg_cache"
RQ4 = config.OUTPUTS_DIR / "rq4"
SIZE, EPOCHS, BATCH, LR = config.SEG_INPUT_SIZE, 30, 16, 6e-5
TAG = ""
NC = len(config.SEG_CLASSES)


def plan_lists():
    seen = [ln.strip() for ln in open(RQ4 / "train_seen.txt") if ln.strip()]
    unseen = [ln.strip() for ln in open(RQ4 / "train_unseen.txt") if ln.strip()]
    extra = random.Random(0).sample(unseen, 1800)
    heldout = [p for p in unseen if p not in set(extra)]
    (RQ4 / "train_ours.txt").write_text("\n".join(seen + extra) + "\n")
    (RQ4 / "train_heldout.txt").write_text("\n".join(heldout) + "\n")
    return seen + extra


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, folders in (("train", [config.CUBICASA_DIR / r.strip("/") for r in plan_lists()]),
                          ("val", list_plans("val"))):
        imgs = np.lib.format.open_memmap(CACHE / f"{name}{TAG}_img.npy", "w+", np.uint8, (len(folders), SIZE, SIZE, 3))
        masks = np.lib.format.open_memmap(CACHE / f"{name}{TAG}_mask.npy", "w+", np.uint8, (len(folders), SIZE, SIZE))
        t0 = time.time()
        for i, f in enumerate(folders):
            img = np.array(Image.open(f / "F1_scaled.png").convert("RGB"))
            im, (s, left, top) = letterbox(img, SIZE)
            imgs[i] = im
            masks[i] = rasterize(parse_svg(f / "model.svg"), (SIZE, SIZE), scale=s, offset=(left, top))
            if (i + 1) % 500 == 0:
                print(f"  {name}: {i + 1}/{len(folders)} ({time.time() - t0:.0f} s)")
        imgs.flush()
        masks.flush()
        print(f"{name}: {len(folders)} plans cached")


def batches(imgs, masks, idx, rng, train, dev):
    """uint8 batches -> GPU, augmented there (the CPU was the bottleneck)."""
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    for b in range(0, len(idx), BATCH):
        j = np.sort(idx[b:b + BATCH])
        x = torch.from_numpy(imgs[j]).to(dev).permute(0, 3, 1, 2).float() / 255
        y = torch.from_numpy(masks[j]).to(dev).long()
        if train:
            for k in range(len(j)):
                r = int(rng.integers(4))
                xi, yi = torch.rot90(x[k], r, (1, 2)), torch.rot90(y[k], r, (0, 1))
                if rng.random() < 0.5:
                    xi, yi = xi.flip(2), yi.flip(1)
                if rng.random() < 0.5:
                    xi = ((xi - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.08, 0.08)).clamp(0, 1)
                x[k], y[k] = xi, yi
        yield (x - mean) / std, y


def loss_fn(logits, y, w):
    logits = torch.nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    ce = torch.nn.functional.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1)[:, 1:]
    t = torch.nn.functional.one_hot(y, NC).permute(0, 3, 1, 2)[:, 1:].float()
    inter, denom = (p * t).sum((0, 2, 3)), (p + t).sum((0, 2, 3))
    return ce + 0.5 * (1 - ((2 * inter + 1) / (denom + 1)).mean())


def evaluate(model, imgs, masks, dev):
    inter, union = np.zeros(NC), np.zeros(NC)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in batches(imgs, masks, np.arange(len(imgs)), None, False, dev):
            lg = model(pixel_values=x).logits
            pred = torch.nn.functional.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear").argmax(1).cpu().numpy()
            yy = y.cpu().numpy()
            for c in range(NC):
                inter[c] += np.sum((pred == c) & (yy == c))
                union[c] += np.sum((pred == c) | (yy == c))
    iou = inter / np.maximum(union, 1)
    return dict(zip(config.SEG_CLASSES, iou.round(4).tolist())), float(iou[1:].mean())


def train(init=None):
    dev = "cuda"
    ti, tm = np.load(CACHE / f"train{TAG}_img.npy"), np.load(CACHE / f"train{TAG}_mask.npy")
    vi, vm = np.load(CACHE / f"val{TAG}_img.npy"), np.load(CACHE / f"val{TAG}_mask.npy")
    freq = np.bincount(np.asarray(tm[::10]).ravel(), minlength=NC).astype(float)
    w = np.minimum(np.median(freq) / np.maximum(freq, 1), 8.0)
    weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    model = load_model(init, device=dev) if init else load_model(device=dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(ti) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)
    base, base_score = evaluate(model, vi, vm, dev)
    print(f"start (public checkpoint): val IoU {base}  foreground mean {base_score:.4f}", flush=True)
    best, history = base_score, [dict(epoch=0, val_iou=base, fg_mean=base_score)]
    out = config.MODELS_DIR / f"segformer_ours{TAG}.pt"
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in batches(ti, tm, rng.permutation(len(ti)), rng, True, dev):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(pixel_values=x).logits.float(), y, weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        iou, score = evaluate(model, vi, vm, dev)
        history.append(dict(epoch=ep, loss=tot, val_iou=iou, fg_mean=score))
        tag = ""
        if score > best:
            best, tag = score, "  * saved"
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_iou": iou}, out)
        print(f"epoch {ep:2d}  loss {tot:8.1f}  val {iou}  fg {score:.4f}  {time.time() - t0:.0f} s{tag}", flush=True)
    json.dump(history, open(config.MODELS_DIR / f"segformer_ours{TAG}_history.json", "w"), indent=1)
    print(f"best foreground mean IoU {best:.4f} (public checkpoint {base_score:.4f}) -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        SIZE, TAG = int(sys.argv[2]), f"_{sys.argv[2]}"
    {"prepare": prepare, "train": lambda: train(sys.argv[3] if len(sys.argv) > 3 else None)}[sys.argv[1]]()
