"""
CryoSPARC-style min-area + morphology post-processing for junk masks.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def postprocess_binary_mask(
    mask: np.ndarray,
    min_area_px: float,
    opening_size: int = 0,
    closing_size: int = 0,
    structure_connectivity: int = 8,
) -> Tuple[np.ndarray, int]:
    """
    Drop small connected components and optionally apply light open/close cleanup.
    """
    from scipy.ndimage import binary_closing, binary_opening, label

    arr = np.asarray(mask).squeeze()
    arr = (arr > 0.5).astype(np.uint8)
    if arr.size == 0:
        return arr, 0

    if int(structure_connectivity) == 8:
        structure = np.ones((3, 3), dtype=np.int8)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)

    labeled, n_labels = label(arr.astype(bool), structure=structure)
    if n_labels <= 0:
        return np.zeros_like(arr, dtype=np.uint8), 0

    sizes = np.bincount(labeled.ravel())
    min_area = max(1, int(math.ceil(float(min_area_px))))
    keep = (sizes >= min_area).astype(bool)
    keep[0] = False
    out = np.where(keep[labeled], labeled, 0).astype(bool)

    if opening_size > 0 and out.any():
        radius = int(opening_size)
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure_open = (xx * xx + yy * yy) <= (radius * radius)
        out = binary_opening(out, structure=structure_open.astype(bool))
    if closing_size > 0 and out.any():
        radius = int(closing_size)
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure_close = (xx * xx + yy * yy) <= (radius * radius)
        out = binary_closing(out, structure=structure_close.astype(bool))

    labeled_out, n_kept = label(out.astype(bool), structure=structure)
    return (labeled_out > 0).astype(np.uint8), int(n_kept)
