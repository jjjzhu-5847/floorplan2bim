"""Run the public SegFormer-B0 CubiCasa5K checkpoint on a floor plan image.

Checkpoint: github.com/alifarzadjamali/floor-plan-document-intelligence
            results/segmentation/phase2_dev/checkpoints/best.pt  (committed here as models/segformer_best.pt)
The model was trained on 512 x 512 letterboxed images, so inference letterboxes to 512 as well.
`segment_image` returns the mask at 512 px plus the transform needed to map 512-px coordinates
back to the original image (see vectorization/transform.py).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MEAN = np.array(config.IMAGENET_MEAN, np.float32)
STD = np.array(config.IMAGENET_STD, np.float32)
_MODEL = None


def load_model(path=config.SEGFORMER_CKPT, device=None):
    """Rebuild SegFormer-B0 (MiT-b0 defaults) with 5 labels and load the checkpoint."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - it is in the repo, run: git pull")
    model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=len(config.SEG_CLASSES)))
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = model.to(device).eval()
    return _MODEL


def letterbox(img, size=config.SEG_INPUT_SIZE):
    """Resize the longer side to `size`, pad with white. Returns image and (scale, left, top)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    canvas = np.full((size, size, 3), 255, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)  # same as training
    return canvas, (s, left, top)


def prepare_input(img, size=config.SEG_WORK_SIZE):
    """Working image and transform (scale, left, top). size=None: the original resolution (CubiCasa scans are about
    1 px = 1 cm), padded with white to at least one tile; otherwise letterboxed to size x size."""
    if size:
        return letterbox(img, size)
    h, w = img.shape[:2]
    t = config.SEG_INPUT_SIZE
    canvas = np.full((max(h, t), max(w, t), 3), 255, np.uint8)
    canvas[:h, :w] = img
    return canvas, (1.0, 0, 0)


def _logits(model, img):
    """C x H x W logits at the size of img."""
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(np.moveaxis(x, -1, 0)[None].copy()).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(pixel_values=x).logits
        return torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False)[0]


def _confidence(logits, mask):
    """How sure the model is about what it found (RQ4 signals): mean top probability and entropy over the pixels
    it labels as anything but background, and the share of wall/door/window pixels with top probability < 0.6."""
    p = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), 0).cpu().numpy()
    top = p.max(0)
    ent = -(p * np.log(np.clip(p, 1e-8, 1))).sum(0)
    fg, el = mask > 0, np.isin(mask, [2, 3, 4])
    return dict(conf_fg=float(top[fg].mean()) if fg.any() else np.nan,
                entropy_fg=float(ent[fg].mean()) if fg.any() else np.nan,
                lowconf_elements=float((top[el] < 0.6).mean()) if el.any() else np.nan)


def segment(model, img, tile=config.SEG_TILE, with_conf=False):
    """img: HxWx3 uint8, already letterboxed to the working size. Returns HxW uint8 class mask
    (and confidence signals if with_conf). tile=None: one pass over the whole image. tile=512: overlapping 512-px
    tiles (objects keep the size they have at the working size; logits are averaged where tiles overlap)."""
    h, w = img.shape[:2]
    if not tile or max(h, w) <= tile:
        lg = _logits(model, img)
        mask = lg.argmax(0).cpu().numpy().astype(np.uint8)
        return (mask, _confidence(lg, mask)) if with_conf else mask
    step = tile * 3 // 4
    ys = sorted({*range(0, h - tile, step), h - tile})
    xs = sorted({*range(0, w - tile, step), w - tile})
    acc, cnt = None, np.zeros((h, w), np.float32)
    for y in ys:
        for x in xs:
            lg = _logits(model, img[y:y + tile, x:x + tile]).cpu().numpy()
            if acc is None:
                acc = np.zeros((lg.shape[0], h, w), np.float32)
            acc[:, y:y + tile, x:x + tile] += lg
            cnt[y:y + tile, x:x + tile] += 1
    lg = acc / cnt
    mask = lg.argmax(0).astype(np.uint8)
    return (mask, _confidence(lg, mask)) if with_conf else mask


def segment_image(path_or_array, model=None, size=config.SEG_WORK_SIZE, tile=config.SEG_TILE):
    """Full step 1 at the working size. Returns dict(image512, mask512, transform=(scale, left, top),
    original_shape); the "512" keys hold the working-size image and mask whatever the working size is."""
    img = path_or_array if isinstance(path_or_array, np.ndarray) else np.array(Image.open(path_or_array).convert("RGB"))
    model = model or load_model()
    img_w, tf = prepare_input(img, size)
    mask, conf = segment(model, img_w, tile, with_conf=True)
    return dict(image512=img_w, mask512=mask, transform=tf, original_shape=img.shape[:2], confidence=conf)


def overlay(img, mask):
    """Colour overlay for figures: wall dark, door orange, window/room blue."""
    colors = np.array([[255, 255, 255], [214, 228, 240], [40, 40, 40], [214, 120, 40], [60, 150, 210]], np.uint8)
    return (0.45 * img + 0.55 * colors[mask]).astype(np.uint8)
