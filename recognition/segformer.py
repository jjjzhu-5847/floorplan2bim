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


def segment(model, img512):
    """img512: HxWx3 uint8 (already letterboxed). Returns HxW uint8 class mask."""
    x = (img512.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(np.moveaxis(x, -1, 0)[None].copy()).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(logits, size=img512.shape[:2], mode="bilinear",
                                                 align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def segment_image(path_or_array, model=None):
    """Full step 1. Returns dict(image512, mask512, transform=(scale, left, top), original_shape)."""
    img = path_or_array if isinstance(path_or_array, np.ndarray) else np.array(Image.open(path_or_array).convert("RGB"))
    model = model or load_model()
    img512, tf = letterbox(img)
    return dict(image512=img512, mask512=segment(model, img512), transform=tf, original_shape=img.shape[:2])


def overlay(img, mask):
    """Colour overlay for figures: wall dark, door orange, window/room blue."""
    colors = np.array([[255, 255, 255], [214, 228, 240], [40, 40, 40], [214, 120, 40], [60, 150, 210]], np.uint8)
    return (0.45 * img + 0.55 * colors[mask]).astype(np.uint8)
