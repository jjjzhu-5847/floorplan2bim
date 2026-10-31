"""Printed room areas (e.g. "MH 11,5", "13.3 m²") read with OCR - an extra scale cue (task 2.4).

Finnish plans in CubiCasa5K print room areas with one decimal and a comma or point; dimensions are whole
millimetres and heights end in a bare "m", so only numbers with exactly one decimal, optionally followed by
"m2"/"m²", are kept. OCR: RapidOCR (PaddleOCR models on onnxruntime), about 1 s per plan on CPU.
"""
import re

import numpy as np

AREA = re.compile(r"(?<![\d.,])(\d{1,3})[.,](\d)(m2|m²)?$", re.I)
_OCR = None


def read_area_labels(img):
    """img: HxWx3 uint8 (original resolution). Returns [dict(m2, x, y, text)] in image pixels."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    res, _ = _OCR(img)
    out = []
    for box, text, _conf in res or []:
        t = text.strip().replace(" ", "")
        m = AREA.search(t)
        if m and 0.5 <= (v := float(f"{m.group(1)}.{m.group(2)}")) <= 300:
            x, y = np.mean(np.asarray(box, float), 0)
            out.append(dict(m2=v, x=float(x), y=float(y), text=t))
    return out
