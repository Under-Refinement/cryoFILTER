from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _center_crop_fft2(F: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    H, W = F.shape
    cy, cx = H // 2, W // 2
    hy, hx = out_h // 2, out_w // 2
    y1 = cy - hy
    y2 = y1 + out_h
    x1 = cx - hx
    x2 = x1 + out_w
    return F[y1:y2, x1:x2]


def _center_pad_fft2(F: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    H, W = F.shape
    out = np.zeros((out_h, out_w), dtype=F.dtype)
    cy, cx = out_h // 2, out_w // 2
    hy, hx = H // 2, W // 2
    y1 = cy - hy
    y2 = y1 + H
    x1 = cx - hx
    x2 = x1 + W
    out[y1:y2, x1:x2] = F
    return out


def fourier_rescale_2d(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Fourier-domain rescale for 2D real images.

    - For scale < 1: crop centered FFT (low-pass) then iFFT -> downsample
    - For scale > 1: zero-pad centered FFT then iFFT -> upsample

    Notes:
    - This preserves the *field of view*; pixel size bookkeeping must be updated by the caller:
        new_apix = old_apix / scale
    """
    x = np.asarray(image).astype(np.float32)
    if x.ndim != 2:
        x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"fourier_rescale_2d expects 2D image, got {x.shape}")

    s = float(scale)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid scale: {scale}")
    if abs(s - 1.0) < 1e-6:
        return x.astype(np.float32)

    H, W = x.shape
    out_h = max(8, int(round(H * s)))
    out_w = max(8, int(round(W * s)))

    # FFT -> shift -> crop/pad -> unshift -> iFFT
    F = np.fft.fftshift(np.fft.fft2(x))
    if s < 1.0:
        F2 = _center_crop_fft2(F, out_h, out_w)
    else:
        F2 = _center_pad_fft2(F, out_h, out_w)
    y = np.fft.ifft2(np.fft.ifftshift(F2)).real.astype(np.float32)

    # Simple energy normalization so amplitudes are not wildly scale-dependent.
    # Empirically stable for our use-case (prompting + classifier features).
    y = y * (s * s)
    return y


def cap_max_dim(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """
    If max(H,W) > max_dim, downscale in Fourier domain to max_dim on the long axis.
    Returns (rescaled_image, scale_factor).
    """
    x = np.asarray(image)
    H, W = x.shape[-2], x.shape[-1]
    md = int(max_dim)
    if md <= 0:
        return x, 1.0
    long = float(max(H, W))
    if long <= md:
        return x, 1.0
    s = float(md / long)
    return fourier_rescale_2d(x, s), s

