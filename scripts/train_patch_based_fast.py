#!/usr/bin/env python3
"""
Fine-tune cryoFILTER segmentation models from annotated micrographs.

The current recommended adaptation path is MicrographCleaner-style full-micrograph
training from the public FULL checkpoint:

- train on dense full-micrograph patches, not GT-biased crops, so the model sees
  the same contamination base rate during training and inference
- use Fourier adaptive downsampling to a 2 A/px working scale
- use deployment-matched percentile_extra_wide normalization by default
- use raw GT with --no_region_gt_fill_holes for membrane/mesh annotations where
  holes inside broad regions are true background
- rely on stitched validation as the primary model-selection signal

The run writes best_model.pt, best_score_model.pt, and best_composite_model.pt.
See docs/fine_tuning.md for the public command template and validation checklist.
"""

import argparse
import gc
import inspect
import math
import sys
import time
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os

# NOTE: Do NOT set OMP_NUM_THREADS=1 in main process - it kills scipy performance!
# Thread limiting is done only in workers via _worker_init_fn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


from models.bad_region_detector import create_model
from models.predict import predict_bad_regions_probability
from models.validation_metrics import compute_pr_auc
from utils.image_utils import normalize_image, compute_power_spectrum, apply_lowpass_filter, compute_radial_psd_image, compute_multiscale_radial_psd_image, compute_multiscale_radial_psd_image_batch, compute_multiscale_radial_psd_channels, compute_frequency_band_psd_channels, compute_edge_magnitude, compute_multires_context_psd, compute_multires_psd_image, compute_global_features, compute_global_edge_properties, compute_edge_proximity, build_constant_pixel_size_channel
from utils.region_gt import compute_dataset_min_area_px_from_train_split, save_dataset_region_rule, apply_region_gt_transform
from utils.postprocess import postprocess_binary_mask
from utils.fourier_rescale import fourier_rescale_2d
from scipy import ndimage
import warnings

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

import pickle
import json


_FULL_MICROGRAPH_CACHE_GB = float(os.environ.get("CRYOFILTER_FULL_MIC_CACHE_GB", "24"))
_FULL_MICROGRAPH_CACHE_LIMIT_BYTES = int(max(0.0, _FULL_MICROGRAPH_CACHE_GB) * (1024 ** 3))
_FULL_MICROGRAPH_CACHE: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray, float, int]]" = OrderedDict()
_FULL_MICROGRAPH_CACHE_BYTES = 0
_FULL_MICROGRAPH_CACHE_LOCK = threading.Lock()
_MODEL_FACTORY_IGNORED_KWARGS: set[str] = set()


def _create_model_compat(**kwargs):
    """Call create_model while tolerating optional args from newer recipes."""
    signature = inspect.signature(create_model)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return create_model(**kwargs)
    accepted = set(signature.parameters)
    ignored = sorted(key for key in kwargs if key not in accepted)
    if ignored:
        unseen = [key for key in ignored if key not in _MODEL_FACTORY_IGNORED_KWARGS]
        if unseen:
            print(
                "  Note: current create_model() does not support option(s): "
                + ", ".join(unseen)
                + "; ignoring for this model factory.",
                flush=True,
            )
            _MODEL_FACTORY_IGNORED_KWARGS.update(unseen)
    return create_model(**{key: value for key, value in kwargs.items() if key in accepted})


# =============================================================================
# NOISE AUGMENTATION FOR CRYO-EM
# =============================================================================
# These augmentations help the model learn that noisy/textured clean regions
# are NOT contamination (carbon). Applied only to clean patches during training.

def add_gaussian_noise(image: np.ndarray, noise_std: float = 0.1, rng: np.random.RandomState = None) -> np.ndarray:
    """Add Gaussian noise to simulate shot noise / low SNR."""
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.normal(0, noise_std * image.std(), image.shape).astype(np.float32)
    return image + noise


def add_crystalline_texture(image: np.ndarray, amplitude: float = 0.05, rng: np.random.RandomState = None) -> np.ndarray:
    """Add low-frequency crystalline/lattice texture to simulate crystalline ice."""
    if rng is None:
        rng = np.random.RandomState()
    H, W = image.shape[-2:]

    # Random frequency for lattice pattern
    freq_x = rng.uniform(0.02, 0.15)
    freq_y = rng.uniform(0.02, 0.15)
    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    texture = np.sin(freq_x * x + phase_x) * np.cos(freq_y * y + phase_y)
    texture = texture.astype(np.float32)

    # Scale by image std
    texture = amplitude * image.std() * texture
    return image + texture


def add_intensity_gradient(image: np.ndarray, strength: float = 0.1, rng: np.random.RandomState = None) -> np.ndarray:
    """Add smooth intensity gradient (simulates uneven illumination/ice thickness)."""
    if rng is None:
        rng = np.random.RandomState()
    H, W = image.shape[-2:]

    # Random gradient direction
    angle = rng.uniform(0, 2 * np.pi)
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    gradient = np.cos(angle) * x + np.sin(angle) * y
    gradient = gradient.astype(np.float32)

    # Scale by image std
    gradient = strength * image.std() * gradient
    return image + gradient


def add_local_contrast_variation(image: np.ndarray, scale: float = 32, strength: float = 0.2, rng: np.random.RandomState = None) -> np.ndarray:
    """Add local contrast variation (simulates ice thickness variation)."""
    if rng is None:
        rng = np.random.RandomState()
    H, W = image.shape[-2:]

    # Create low-frequency noise map
    small_H, small_W = max(4, H // int(scale)), max(4, W // int(scale))
    small_noise = rng.randn(small_H, small_W).astype(np.float32)

    # Upsample to full size
    from scipy.ndimage import zoom
    contrast_map = zoom(small_noise, (H / small_H, W / small_W), order=1)
    contrast_map = contrast_map[:H, :W]  # Ensure exact size

    # Normalize to [1-strength, 1+strength] range
    contrast_map = 1 + strength * (contrast_map - contrast_map.mean()) / (contrast_map.std() + 1e-6)
    contrast_map = np.clip(contrast_map, 1 - strength, 1 + strength)

    # Apply contrast variation
    mean_val = image.mean()
    return mean_val + (image - mean_val) * contrast_map


def augment_with_cryoem_noise(
    image: np.ndarray,
    contamination_fraction: float,
    rng: np.random.RandomState = None,
    strength_multiplier: float = 1.0,  # Scale all noise strengths (0.0 = no noise, 1.0 = full strength)
    noise_prob: float = 0.5,
    noise_std_range: Tuple[float, float] = (0.05, 0.15),
    texture_prob: float = 0.3,
    texture_amp_range: Tuple[float, float] = (0.02, 0.08),
    gradient_prob: float = 0.3,
    gradient_strength_range: Tuple[float, float] = (0.05, 0.15),
    contrast_prob: float = 0.3,
    contrast_strength_range: Tuple[float, float] = (0.1, 0.3),
) -> np.ndarray:
    """
    Apply realistic cryo-EM noise augmentation to clean patches.

    Key insight: Only augment patches with LOW contamination fraction.
    This teaches the model: "noisy clean images are NOT contamination"

    Args:
        image: Input image (H, W) - ONLY the real-space channel
        contamination_fraction: Fraction of patch that is contamination (0-1)
        rng: Random state for reproducibility
        strength_multiplier: Scale factor for all noise strengths (0.0-1.0).
            Used for gradual ramping during curriculum learning.
        noise_prob: Probability of adding Gaussian noise
        noise_std_range: Range for noise std (fraction of image std)
        texture_prob: Probability of adding crystalline texture
        texture_amp_range: Range for texture amplitude
        gradient_prob: Probability of adding intensity gradient
        gradient_strength_range: Range for gradient strength
        contrast_prob: Probability of adding local contrast variation
        contrast_strength_range: Range for contrast variation strength

    Returns:
        Augmented image (H, W)
    """
    if rng is None:
        rng = np.random.RandomState()

    # Only augment clean/mostly-clean patches (< 10% contamination)
    # This teaches: "noisy clean ≠ contamination"
    if contamination_fraction > 0.1:
        return image

    # If strength is 0, skip augmentation entirely
    if strength_multiplier <= 0:
        return image

    result = image.copy()

    # Scale all strength ranges by the multiplier
    scaled_noise_std_range = (noise_std_range[0] * strength_multiplier, noise_std_range[1] * strength_multiplier)
    scaled_texture_amp_range = (texture_amp_range[0] * strength_multiplier, texture_amp_range[1] * strength_multiplier)
    scaled_gradient_range = (gradient_strength_range[0] * strength_multiplier, gradient_strength_range[1] * strength_multiplier)
    scaled_contrast_range = (contrast_strength_range[0] * strength_multiplier, contrast_strength_range[1] * strength_multiplier)

    # Apply augmentations with given probabilities
    if rng.rand() < noise_prob:
        noise_std = rng.uniform(*scaled_noise_std_range)
        result = add_gaussian_noise(result, noise_std, rng)

    if rng.rand() < texture_prob:
        texture_amp = rng.uniform(*scaled_texture_amp_range)
        result = add_crystalline_texture(result, texture_amp, rng)

    if rng.rand() < gradient_prob:
        gradient_strength = rng.uniform(*scaled_gradient_range)
        result = add_intensity_gradient(result, gradient_strength, rng)

    if rng.rand() < contrast_prob:
        contrast_strength = rng.uniform(*scaled_contrast_range)
        result = add_local_contrast_variation(result, strength=contrast_strength, rng=rng)

    return result


# =============================================================================
# ADAPTIVE DOWNSAMPLING BASED ON PIXEL SIZE
# =============================================================================
# Different datasets have vastly different pixel sizes (e.g., 0.32 Å/px to 1.35 Å/px).
# To achieve consistent physical scale during training/inference, we compute
# per-dataset downsample factors to target a uniform effective pixel size.

def compute_adaptive_downsample_factors(
    df: pd.DataFrame,
    target_pixel_size: float = 5.0,  # Target effective pixel size in Å/px
    min_downsample: float = 1.0,  # Minimum allowed downsample factor
    max_downsample: float = 20.0,  # Maximum allowed downsample factor
    default_pixel_size: float = 1.2,  # Default if pixel_size column missing
) -> Dict[str, float]:
    """
    Compute per-dataset downsample factors to achieve uniform effective resolution.

    For each dataset, computes: downsample_factor = target_pixel_size / dataset_pixel_size

    Example with target_pixel_size=5.0 Å/px:
    - Dataset 10061 (0.32 Å/px): factor = 5.0/0.32 = 15.6x
    - Dataset 10005 (1.2 Å/px):  factor = 5.0/1.2 = 4.2x
    - Dataset 10028 (1.34 Å/px): factor = 5.0/1.34 = 3.7x

    This ensures all datasets are processed at ~5 Å/px effective resolution,
    so the model sees consistent physical scales regardless of acquisition settings.

    Args:
        df: DataFrame with 'dataset_id' and optionally 'pixel_size_angstrom' columns
        target_pixel_size: Target effective pixel size in Å/px (default: 5.0)
        min_downsample: Minimum allowed factor (default: 1.0 = no upsampling)
        max_downsample: Maximum allowed factor (default: 20.0)
        default_pixel_size: Fallback pixel size if column missing (default: 1.2)

    Returns:
        Dict mapping dataset_id (str) -> downsample_factor (float)
    """
    factors = {}

    has_pixel_size = 'pixel_size_angstrom' in df.columns

    for dataset_id in df['dataset_id'].unique():
        dataset_id_str = str(dataset_id)

        if has_pixel_size:
            # Get pixel size for this dataset (use first row since should be same for all)
            subset = df[df['dataset_id'] == dataset_id]
            pixel_size = subset['pixel_size_angstrom'].iloc[0]
            if pd.isna(pixel_size) or pixel_size <= 0:
                pixel_size = default_pixel_size
        else:
            pixel_size = default_pixel_size

        # Compute factor to achieve target
        factor = target_pixel_size / pixel_size

        # Clamp to valid range
        factor = max(min_downsample, min(max_downsample, factor))

        factors[dataset_id_str] = float(factor)

    return factors


def get_downsample_factor_for_row(
    row: pd.Series,
    downsample_factors: Dict[str, float],
    default_factor: float = 4.0,
) -> float:
    """Get downsample factor for a specific micrograph row."""
    dataset_id = str(row.get('dataset_id', ''))
    return downsample_factors.get(dataset_id, default_factor)


def load_particle_annotations(path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Load particle annotations from pkl/json file or directory containing particle_annotations.pkl.

    Returns dict: {dataset_id__stem: [(x, y), ...]} in full-resolution coordinates.
    Also builds alternative key lookups for flexible matching.
    """
    if path is None:
        return {}
    p = Path(path)
    if p.is_dir():
        # Look for particle_annotations.pkl or particle_annotations.json
        pkl_path = p / "particle_annotations.pkl"
        json_path = p / "particle_annotations.json"
        if pkl_path.exists():
            p = pkl_path
        elif json_path.exists():
            p = json_path
        else:
            print(f"  WARNING: No particle_annotations.pkl or .json found in {path}")
            return {}

    if not p.exists():
        print(f"  WARNING: Particle annotations file not found: {p}")
        return {}

    suffix = p.suffix.lower()
    if suffix == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
    elif suffix == ".json":
        with open(p, "r") as f:
            data = json.load(f)
    else:
        print(f"  WARNING: Unknown particle annotations format: {suffix}")
        return {}

    # Handle {"particles": {...}} structure (used by finetune_on_mc_benchmark.py)
    if isinstance(data, dict) and "particles" in data:
        data = data.get("particles", {})
    elif not isinstance(data, dict):
        print(f"  WARNING: Particle annotations not a dict, got {type(data)}")
        return {}

    # Normalize keys and values
    result = {}
    sample_keys = []
    total_picks = 0
    for key, coords in data.items():
        # Keys can be "dataset_id__stem" or just "stem"
        key_str = str(key).strip()
        # Convert coords to list of (x, y) tuples
        if isinstance(coords, (list, tuple)):
            picks = [(int(c[0]), int(c[1])) for c in coords if len(c) >= 2]
            if picks:
                result[key_str] = picks
                total_picks += len(picks)
                if len(sample_keys) < 5:
                    sample_keys.append(key_str)

    print(f"  Loaded particle annotations: {len(result)} micrographs with {total_picks} total picks", flush=True)
    if sample_keys:
        print(f"  Sample keys: {sample_keys[:5]}", flush=True)
    return result


def build_particle_mask(
    particles: List[Tuple[int, int]],
    height: int,
    width: int,
    box_size_px: int,
    downsample_factor: float = 1.0,
    use_gaussian: bool = False,  # DEFAULT TO DISK for speed (Gaussian is expensive)
    gaussian_sigma_factor: float = 0.3,  # sigma = radius * factor
) -> np.ndarray:
    """Build particle mask from particle center coordinates.

    PERFORMANCE: Default to disk (not Gaussian) for worker speed at high num_workers.

    Args:
        particles: List of (x, y) particle centers in full-resolution
        height, width: Output mask dimensions (after downsample)
        box_size_px: Box size in full-resolution pixels (determines radius)
        downsample_factor: Downsample factor applied to coordinates and box size
        use_gaussian: If True, use soft Gaussian mask; if False, use hard disk (FASTER)
        gaussian_sigma_factor: sigma = radius * factor (default 0.3 for tight Gaussian)

    Returns:
        Mask (height, width) with values in [0, 1]
    """
    mask = np.zeros((height, width), dtype=np.float32)
    if not particles:
        return mask

    # Scale to downsampled space - use smaller effective radius (not full box)
    # Typical particle box is 256px but particle itself is ~100-150px diameter
    # Use radius = box_size / 4 for tighter mask (covers particle core, not full box)
    effective_radius = max(4, int(box_size_px / (4 * downsample_factor)))

    if use_gaussian:
        # Gaussian approach: soft mask with peak at particle center (SLOW)
        sigma = effective_radius * gaussian_sigma_factor

        for (px, py) in particles:
            # Scale particle center
            cx = px / downsample_factor
            cy = py / downsample_factor

            # Compute Gaussian (only in a local window for efficiency)
            window = int(3 * sigma) + 1
            x0, x1 = max(0, int(cx) - window), min(width, int(cx) + window + 1)
            y0, y1 = max(0, int(cy) - window), min(height, int(cy) + window + 1)

            if x1 > x0 and y1 > y0:
                # Local coordinate grids
                local_xx = np.arange(x0, x1)
                local_yy = np.arange(y0, y1)[:, None]

                # Gaussian value at each pixel
                dist_sq = (local_xx - cx)**2 + (local_yy - cy)**2
                gauss = np.exp(-dist_sq / (2 * sigma**2))

                # Max with existing values (overlapping particles)
                mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], gauss)
    else:
        # Hard disk approach: binary mask within radius (FAST - preferred for high worker counts)
        # Use simple box drawing (even faster than distance check)
        r = effective_radius
        for (px, py) in particles:
            cx = int(px / downsample_factor)
            cy = int(py / downsample_factor)

            # Simple box (faster than circular disk)
            x0 = max(0, cx - r)
            y0 = max(0, cy - r)
            x1 = min(width, cx + r + 1)
            y1 = min(height, cy + r + 1)

            if x1 > x0 and y1 > y0:
                local_yy, local_xx = np.ogrid[y0:y1, x0:x1]
                dist_sq = (local_xx - cx)**2 + (local_yy - cy)**2
                disk = (dist_sq <= effective_radius**2).astype(np.float32)
                mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], disk)

    return mask


def _downsample_image(img: np.ndarray, factor: float, order: int = 1) -> np.ndarray:
    """Downsample image by factor using scipy.ndimage.zoom."""
    if factor <= 1.0:
        return img
    h, w = img.shape
    new_h, new_w = int(h / factor), int(w / factor)
    if new_h < 1 or new_w < 1:
        return img
    downsampled = ndimage.zoom(img, (new_h / h, new_w / w), order=order)
    return downsampled.astype(img.dtype) if img.dtype in [np.uint8, np.float16, np.float32] else downsampled


def _resample_mask_to_shape(mask: np.ndarray, out_shape: Tuple[int, int], order: int = 0) -> np.ndarray:
    """Resize a mask to an exact output shape and crop/pad any rounding mismatch."""
    arr = np.asarray(mask)
    target_shape = (int(out_shape[0]), int(out_shape[1]))
    if arr.shape == target_shape:
        return arr
    zoom_factors = (
        float(target_shape[0]) / float(arr.shape[0]),
        float(target_shape[1]) / float(arr.shape[1]),
    )
    resized = ndimage.zoom(arr, zoom_factors, order=order)
    if resized.shape == target_shape:
        return resized.astype(arr.dtype, copy=False)
    matched = np.zeros(target_shape, dtype=resized.dtype)
    h_copy = min(target_shape[0], resized.shape[0])
    w_copy = min(target_shape[1], resized.shape[1])
    matched[:h_copy, :w_copy] = resized[:h_copy, :w_copy]
    return matched.astype(arr.dtype, copy=False)


def _resample_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
    factor: float,
    method: str = "zoom",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample an image/mask pair with an explicit method.

    `factor` is the effective pixel-size multiplier, so `factor=2.0` means the
    output should have half as many pixels in each dimension.
    """
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 1.0:
        return np.asarray(img), np.asarray(mask)

    method_l = str(method).strip().lower() or "zoom"
    img_arr = np.asarray(img)
    mask_arr = np.asarray(mask)

    if method_l == "stride":
        ds = int(max(1, factor))
        if ds <= 1:
            return img_arr, mask_arr
        return img_arr[::ds, ::ds], mask_arr[::ds, ::ds]

    if method_l == "zoom":
        return _downsample_image(img_arr, factor, order=1), _downsample_image(mask_arr, factor, order=0)

    if method_l == "fourier":
        out_img = fourier_rescale_2d(np.asarray(img_arr, dtype=np.float32), scale=1.0 / factor)
        out_mask = _resample_mask_to_shape(mask_arr, out_img.shape, order=0)
        return out_img.astype(np.float32, copy=False), out_mask.astype(mask_arr.dtype, copy=False)

    raise ValueError(f"Unknown adaptive_downsample_method='{method}'. Expected stride, zoom, or fourier.")


# Per-dataset pixel spacing (Å) for 4 Å low-pass and export; fallback used for unknown datasets.
DATASET_PIXEL_SPACING_ANGSTROM: Dict[str, float] = {
    "10005": 1.2156, "10028": 1.34, "10033": 1.14, "10049": 1.23, "10061": 0.3185,
    "10075": 1.16, "10077": 1.16, "10081": 1.3, "10090": 0.75, "10093": 1.2156,
    "10097": 1.31, "10099": 1.35, "10168": 1.35, "10175": 0.85, "10190": 1.35,
    "10203": 1.06, "10205": 1.0651, "10217": 0.6616,
}


def _build_dataset_pixel_size_map(df: pd.DataFrame) -> Dict[str, float]:
    pixel_sizes = dict(DATASET_PIXEL_SPACING_ANGSTROM)
    if "pixel_size_angstrom" not in df.columns:
        return pixel_sizes

    for dataset_id, group in df.groupby("dataset_id"):
        values = pd.to_numeric(group["pixel_size_angstrom"], errors="coerce").dropna()
        if len(values) == 0:
            continue
        value = float(values.iloc[0])
        if value > 0:
            pixel_sizes[str(dataset_id)] = value
    return pixel_sizes


def _resolve_dataset_pixel_size(
    dataset_id: str,
    pixel_size_by_dataset: Optional[Dict[str, float]],
    fallback: float = 1.1,
) -> float:
    dsid = str(dataset_id)
    if pixel_size_by_dataset is not None:
        value = pixel_size_by_dataset.get(dsid)
        if value is not None and np.isfinite(value) and float(value) > 0:
            return float(value)
    value = DATASET_PIXEL_SPACING_ANGSTROM.get(dsid)
    if value is not None and np.isfinite(value) and float(value) > 0:
        return float(value)
    return float(fallback)


def _build_pixel_size_channel(
    image_shape: Tuple[int, int],
    dataset_id: str,
    pixel_size_by_dataset: Optional[Dict[str, float]],
    min_angstrom: float,
    max_angstrom: float,
    downsample_factor: float = 1.0,
    fallback: float = 1.1,
) -> np.ndarray:
    effective_pixel_size = _resolve_dataset_pixel_size(
        dataset_id=dataset_id,
        pixel_size_by_dataset=pixel_size_by_dataset,
        fallback=fallback,
    ) * float(max(1.0, downsample_factor))
    return build_constant_pixel_size_channel(
        image_shape=image_shape,
        pixel_size_angstrom=effective_pixel_size,
        min_angstrom=min_angstrom,
        max_angstrom=max_angstrom,
        use_log_scale=True,
    )


def _effective_pixel_size_for_dataset(
    dataset_id: str,
    pixel_size_by_dataset: Optional[Dict[str, float]],
    downsample_factor: float = 1.0,
    fallback: float = 1.1,
) -> float:
    return float(
        _resolve_dataset_pixel_size(
            dataset_id=dataset_id,
            pixel_size_by_dataset=pixel_size_by_dataset,
            fallback=fallback,
        )
        * float(max(1.0, downsample_factor))
    )


def _parse_frequency_bands(text: str) -> Tuple[Tuple[float, float], ...]:
    bands = []
    raw = str(text).strip()
    if not raw:
        return tuple()
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid frequency band '{chunk}'. Use low:high in Å^-1, e.g. 0.020:0.035"
            )
        low_str, high_str = chunk.split(":", 1)
        low = float(low_str)
        high = float(high_str)
        if low < 0 or high <= low:
            raise ValueError(f"Invalid frequency band '{chunk}': require 0 <= low < high")
        bands.append((low, high))
    return tuple(bands)


def _format_frequency_bands(bands: Tuple[Tuple[float, float], ...]) -> str:
    return ",".join(f"{float(low):.5f}:{float(high):.5f}" for low, high in tuple(bands))


def _parse_float_list(text: str) -> Tuple[float, ...]:
    values = []
    for part in str(text).split(","):
        chunk = part.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("Expected at least one comma-separated float value")
    return tuple(sorted(set(float(v) for v in values)))


def _resolve_manifest_relative_paths(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    """Resolve known manifest path columns relative to the manifest location."""
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    path_columns = (
        "micrograph_path",
        "full_mic_path",
        "gt_mask_path",
        "gt_mask",
        "gt_mask_full_path",
        "mask_path",
        "binary_mask_path",
    )
    resolved_df = df.copy()
    for column in path_columns:
        if column not in resolved_df.columns:
            continue
        resolved_values = []
        for value in resolved_df[column].tolist():
            if pd.isna(value):
                resolved_values.append(value)
                continue
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none"}:
                resolved_values.append(np.nan)
                continue
            path = Path(text).expanduser()
            if path.is_absolute():
                resolved_values.append(str(path))
                continue
            manifest_relative = manifest_dir / path
            if manifest_relative.exists():
                resolved_values.append(str(manifest_relative))
            else:
                resolved_values.append(text)
        resolved_df[column] = resolved_values
    return resolved_df


def _normalize_row_split_label(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text in {"TRAIN", "TRAINING"}:
        return "TRAINING"
    if text in {"VAL", "VALID", "VALIDATION"}:
        return "VALIDATION"
    return text


def _find_mic_path(mic_dir: Path, dataset_id: str, stem: str) -> Path:
    p = mic_dir / f"{dataset_id}_{stem}.mrc"
    if p.exists():
        return p
    p = mic_dir / f"{stem}.mrc"
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find micrograph for dataset_id={dataset_id} stem={stem} in {mic_dir}")


def _load_mrc_2d_permissive(path: Path) -> np.ndarray:
    """
    Load a 2D image from .mrc/.mrcs or .npy (used by MC downsampled data).
    Uses permissive=True to handle corrupt files gracefully.
    """
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(p)
    suf = p.suffix.lower()
    if suf == ".npy":
        x = np.load(str(p), allow_pickle=True)
    elif suf in [".mrc", ".mrcs"]:
        import mrcfile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with mrcfile.open(str(p), permissive=True) as m:
                x = np.asarray(m.data)
    else:
        raise ValueError(f"Unsupported micrograph format: {p} (suffix={suf})")
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D micrograph, got shape={x.shape}")
    return x


def _expected_input_channels(
    use_psd_input: bool,
    psd_multiscale: bool,
    psd_multiscale_separate_channels: bool,
    psd_scales: tuple,
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    include_real_space_input: bool = True,
    use_pixel_size_channel: bool = False,
) -> int:
    if (not use_psd_input) and (not include_real_space_input):
        raise ValueError("Invalid input configuration: both real-space and PSD inputs are disabled.")

    real_channels = 1 if include_real_space_input else 0
    if not use_psd_input:
        return real_channels + (1 if use_pixel_size_channel else 0)

    if psd_frequency_band_channels:
        psd_channels = len(tuple(psd_frequency_bands))
        if psd_frequency_band_include_full_spectrum:
            psd_channels += 1
        if psd_frequency_band_include_anisotropy:
            psd_channels += 1
        if psd_channels <= 0:
            raise ValueError("Frequency-band PSD mode requires at least one channel")
    elif psd_multiscale and psd_multiscale_separate_channels:
        psd_channels = len(tuple(int(s) for s in psd_scales))
    else:
        psd_channels = 1
    return real_channels + psd_channels + (1 if use_pixel_size_channel else 0)


def _compute_psd_map(
    img: np.ndarray,
    psd_use_radial_normalization: bool,
    psd_use_radial_image: bool,
    psd_radial_bins: int,
    psd_multiscale: bool,
    psd_scales: tuple,
    psd_texture_mode: bool,
    psd_highpass_cutoff: float,
    psd_multiscale_separate_channels: bool,
    pixel_size_angstrom: Optional[float] = None,
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    psd_frequency_band_anisotropy_min_freq: float = 0.0,
) -> np.ndarray:
    if psd_frequency_band_channels:
        if pixel_size_angstrom is None or float(pixel_size_angstrom) <= 0:
            raise ValueError("pixel_size_angstrom is required for frequency-band PSD channels")
        psd = compute_frequency_band_psd_channels(
            img,
            pixel_size_angstrom=float(pixel_size_angstrom),
            frequency_bands=tuple(psd_frequency_bands),
            normalize=True,
            use_radial_normalization=bool(psd_use_radial_normalization),
            include_full_spectrum_channel=bool(psd_frequency_band_include_full_spectrum),
            include_anisotropy_channel=bool(psd_frequency_band_include_anisotropy),
            anisotropy_min_frequency=float(psd_frequency_band_anisotropy_min_freq),
        )
    elif psd_multiscale:
        if psd_multiscale_separate_channels:
            psd = compute_multiscale_radial_psd_channels(
                img,
                scales=psd_scales,
                normalize=True,
                texture_mode=psd_texture_mode,
                highpass_cutoff=psd_highpass_cutoff,
            )
        else:
            psd = compute_multiscale_radial_psd_image(
                img,
                scales=psd_scales,
                normalize=True,
                texture_mode=psd_texture_mode,
                highpass_cutoff=psd_highpass_cutoff,
            )
    elif psd_use_radial_image:
        psd = compute_radial_psd_image(img, num_bins=psd_radial_bins, normalize=True)
    else:
        psd = compute_power_spectrum(
            img, normalize=True, use_radial_normalization=psd_use_radial_normalization, return_stats=False
        )
    return np.asarray(psd, dtype=np.float32)


def _stack_image_with_psd(
    img: np.ndarray,
    psd: np.ndarray,
    use_psd_input: bool,
    include_real_space_input: bool = True,
    pixel_size_channel: Optional[np.ndarray] = None,
) -> np.ndarray:
    img2d = np.asarray(img, dtype=np.float32).squeeze()
    channels: List[np.ndarray] = []
    if include_real_space_input:
        channels.append(img2d[np.newaxis])

    if use_psd_input:
        psd_arr = np.asarray(psd, dtype=np.float32)
        if psd_arr.ndim == 2:
            psd_arr = psd_arr[np.newaxis]
        elif psd_arr.ndim != 3:
            raise ValueError(f"Expected PSD ndim 2 or 3, got {psd_arr.ndim} with shape {psd_arr.shape}")

        if psd_arr.shape[1:] != img2d.shape:
            raise ValueError(f"PSD spatial shape {psd_arr.shape[1:]} does not match image shape {img2d.shape}")
        channels.append(psd_arr)

    if pixel_size_channel is not None:
        px_arr = np.asarray(pixel_size_channel, dtype=np.float32).squeeze()
        if px_arr.shape != img2d.shape:
            raise ValueError(f"Pixel-size channel shape {px_arr.shape} does not match image shape {img2d.shape}")
        channels.append(px_arr[np.newaxis])

    if not channels:
        raise ValueError("Cannot build input tensor: no channels are enabled.")

    stacked = np.concatenate(channels, axis=0)
    return np.ascontiguousarray(stacked, dtype=np.float32)


def _infer_input_channels_from_state_dict(state_dict: dict) -> Optional[int]:
    if not isinstance(state_dict, dict):
        return None
    priority_suffixes = (
        "enc1.0.weight",
        "conv1.weight",
        "features.0.weight",
        "stem.0.weight",
    )
    for suffix in priority_suffixes:
        for key, tensor in state_dict.items():
            if key.endswith(suffix) and isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                return int(tensor.shape[1])
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            return int(tensor.shape[1])
    return None


def _expand_first_conv_input_channels(state_dict: dict, target_input_channels: int) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    target_input_channels = int(target_input_channels)
    if target_input_channels <= 0:
        return state_dict

    updated_state = dict(state_dict)
    priority_suffixes = (
        "enc1.0.weight",
        "conv1.weight",
        "features.0.weight",
        "stem.0.weight",
    )

    for suffix in priority_suffixes:
        matching_keys = [
            key for key, tensor in updated_state.items()
            if key.endswith(suffix) and isinstance(tensor, torch.Tensor) and tensor.ndim == 4
        ]
        if not matching_keys:
            continue
        for key in matching_keys:
            weight = updated_state[key]
            current_channels = int(weight.shape[1])
            if current_channels == target_input_channels:
                continue
            if current_channels > target_input_channels:
                updated_state[key] = weight[:, :target_input_channels, :, :].clone()
                continue
            extra_channels = target_input_channels - current_channels
            mean_channel = weight.mean(dim=1, keepdim=True)
            extra = mean_channel.repeat(1, extra_channels, 1, 1)
            updated_state[key] = torch.cat([weight, extra], dim=1)
        return updated_state

    return updated_state


def _attach_model_input_metadata(
    model: nn.Module,
    *,
    use_power_spectrum: bool,
    include_real_space_input: bool,
    use_pixel_size_channel: bool,
    pixel_size_channel_min_angstrom: float,
    pixel_size_channel_max_angstrom: float,
    input_channels: int,
    psd_multiscale: bool = False,
    psd_multiscale_separate_channels: bool = False,
    psd_scales: tuple = (),
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    psd_frequency_band_anisotropy_min_freq: float = 0.0,
) -> None:
    target = model.module if hasattr(model, "module") else model
    for meta_target in (model, target):
        meta_target.use_power_spectrum = bool(use_power_spectrum)
        meta_target.include_real_space_input = bool(include_real_space_input)
        meta_target.use_pixel_size_channel = bool(use_pixel_size_channel)
        meta_target.pixel_size_channel_min_angstrom = float(pixel_size_channel_min_angstrom)
        meta_target.pixel_size_channel_max_angstrom = float(pixel_size_channel_max_angstrom)
        meta_target.input_channels = int(input_channels)
        meta_target.psd_multiscale = bool(psd_multiscale)
        meta_target.psd_multiscale_separate_channels = bool(psd_multiscale_separate_channels)
        meta_target.psd_scales = tuple(int(x) for x in tuple(psd_scales))
        meta_target.psd_frequency_band_channels = bool(psd_frequency_band_channels)
        meta_target.psd_frequency_bands = tuple((float(lo), float(hi)) for lo, hi in tuple(psd_frequency_bands))
        meta_target.psd_frequency_band_include_full_spectrum = bool(psd_frequency_band_include_full_spectrum)
        meta_target.psd_frequency_band_include_anisotropy = bool(psd_frequency_band_include_anisotropy)
        meta_target.psd_frequency_band_anisotropy_min_freq = float(psd_frequency_band_anisotropy_min_freq)


def _base_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _model_state_dict(model: nn.Module) -> dict:
    return _base_model(model).state_dict()


def _strip_module_prefix(state: dict) -> dict:
    if not isinstance(state, dict) or not state:
        return state
    keys = list(state.keys())
    if all(str(k).startswith("module.") for k in keys):
        return {str(k)[7:]: v for k, v in state.items()}
    return state


def build_patch_cache(
    df: pd.DataFrame,
    mic_dir: Path,
    rule,
    normalization_method: str,
    psd_use_radial_normalization: bool,
    downsample_factor,  # Can be float (uniform) or Dict[str, float] (per-dataset)
    cache_dir: Path,
    desc: str = "train",
    psd_use_radial_image: bool = False,  # NEW: Use radial PSD image for stronger signal
    psd_radial_bins: int = 64,
    use_edge_features: bool = False,  # NEW: Blend edge features into real image
    edge_blend_alpha: float = 0.3,
    edge_sigma: float = 1.0,
    psd_multiscale: bool = False,  # NEW: Multi-scale radial PSD
    psd_multiscale_separate_channels: bool = False,  # NEW: Keep one PSD channel per scale
    psd_scales: tuple = (16, 32, 64),
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    psd_frequency_band_anisotropy_min_freq: float = 0.0,
    pixel_size_by_dataset: Optional[Dict[str, float]] = None,
    psd_texture_mode: bool = False,  # Focus on fine texture (5-40 Å) for carbon detection
    psd_highpass_cutoff: float = 0.2,  # In texture mode, ignore frequencies below this
    psd_multires_context: bool = False,  # NEW: Multi-res context PSD (computed on-the-fly per patch)
    use_global_features: bool = False,  # NEW: Compute and cache global features
    use_edge_direction: bool = False,  # NEW: Compute directional edge features (3 scalars)
    adaptive_downsample_method: str = "zoom",
) -> None:
    """Precompute per-micrograph (normalized img, PSD, region_gt) at working resolution and save to disk.
    After this, training epochs only read from cache—no recompute of PSD or normalization.

    Args:
        downsample_factor: Can be a float (uniform for all datasets) or a Dict[str, float]
            mapping dataset_id -> per-dataset factor for adaptive downsampling.

    Note: If psd_multires_context=True, PSD is computed on-the-fly per patch (not cached here).
    Note: If use_global_features=True, global features (154-dim) are computed and cached.
    Note: If use_edge_direction=True, edge direction (strength, angle) is cached per micrograph.
    Note: The actual downsample factor used is stored in each .npz file for inference matching."""

    # Handle both uniform and per-dataset downsample factors
    if isinstance(downsample_factor, dict):
        downsample_dict = downsample_factor
        use_adaptive = True
    else:
        downsample_dict = {}
        use_adaptive = False
        uniform_factor = float(downsample_factor)
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped_count = 0
    cached_count = 0
    for mic_idx in tqdm(range(len(df)), desc=f"Building {desc} cache", leave=False):
        r = df.iloc[mic_idx]
        dataset_id = str(r["dataset_id"])
        stem = str(r["stem"])
        gt_path_raw = str(r["gt_mask_path"])
        gt_path = Path(gt_path_raw)

        # Try relative to cwd first (manifest paths like 'particle_annotations/merged_gt_fullres/...')
        if not gt_path.exists():
            gt_path_cwd = Path.cwd() / gt_path_raw
            if gt_path_cwd.exists():
                gt_path = gt_path_cwd

        # Handle OLD manifest format: mc_fullres_gt_masks_642/10090__10045.__gt_full.npy
        # Map to NEW format: particle_annotations/merged_gt_fullres/10090__10045..npy
        if not gt_path.exists() and "mc_fullres_gt_masks_642" in gt_path_raw:
            # Extract dataset_id and stem from old format filename
            old_filename = Path(gt_path_raw).name  # e.g., "10090__10045.__gt_full.npy"
            # Remove __gt_full suffix to get new filename
            new_filename = old_filename.replace("__gt_full", "")  # e.g., "10090__10045..npy"
            new_path = Path.cwd() / "particle_annotations" / "merged_gt_fullres" / new_filename
            if new_path.exists():
                gt_path = new_path

        # If still not found, try other locations
        if not gt_path.exists():
            gt_filename = Path(gt_path_raw).name
            alt_name = f"{dataset_id}__{stem}.npy"
            candidates = [
                Path.cwd() / "particle_annotations" / "merged_gt_fullres" / gt_filename,
                Path.cwd() / "particle_annotations" / "merged_gt_fullres" / alt_name,
                mic_dir.parent / gt_path_raw,
                mic_dir.parent.parent / gt_path_raw,
            ]
            for candidate in candidates:
                if candidate.exists():
                    gt_path = candidate
                    break

        # If GT mask not found, this is likely a clean image - create all-zeros mask
        gt_not_found = not gt_path.exists()

        if "micrograph_path" in r and pd.notna(r["micrograph_path"]):
            mic_path = Path(r["micrograph_path"])
            if not mic_path.exists():
                mic_path = _find_mic_path(mic_dir, dataset_id, stem)
        else:
            mic_path = _find_mic_path(mic_dir, dataset_id, stem)
        img = _load_mrc_2d_permissive(mic_path)
        img = np.asarray(img).squeeze().astype(np.float32)

        # Load GT mask, or create all-zeros for clean images
        if gt_not_found:
            # Clean image - no contamination, create empty mask
            gt = np.zeros(img.shape, dtype=np.uint8)
            region_gt = gt
            if skipped_count == 0:
                print(f"  Note: Creating empty GT mask for mic {mic_idx} ({dataset_id}__{stem}) - likely clean image")
            skipped_count += 1
        else:
            gt = np.load(str(gt_path))
            gt = np.asarray(gt).squeeze()
            region_gt = apply_region_gt_transform(gt, dataset_id=dataset_id, rule=rule).astype(np.uint8)

        # Get per-micrograph downsample factor (adaptive or uniform)
        if use_adaptive:
            mic_downsample = downsample_dict.get(dataset_id, 4.0)  # Default to 4x if not found
        else:
            mic_downsample = uniform_factor
        effective_pixel_size = _effective_pixel_size_for_dataset(
            dataset_id=dataset_id,
            pixel_size_by_dataset=pixel_size_by_dataset,
            downsample_factor=mic_downsample,
        )

        if mic_downsample > 1.0:
            img, region_gt = _resample_image_and_mask(
                img,
                region_gt,
                mic_downsample,
                method=adaptive_downsample_method,
            )
        img = normalize_image(img, method=normalization_method)

        # Compute PSD - use radial image if requested (stronger signal for contamination)
        # Note: For psd_multires_context, PSD is computed on-the-fly per patch (placeholder here).
        if psd_multires_context:
            if psd_frequency_band_channels:
                placeholder_channels = (
                    len(tuple(psd_frequency_bands))
                    + (1 if psd_frequency_band_include_full_spectrum else 0)
                    + (1 if psd_frequency_band_include_anisotropy else 0)
                )
                psd = np.zeros((placeholder_channels, img.shape[0], img.shape[1]), dtype=np.float32)
            elif psd_multiscale and psd_multiscale_separate_channels:
                psd = np.zeros((len(psd_scales), img.shape[0], img.shape[1]), dtype=np.float32)
            else:
                psd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            psd = _compute_psd_map(
                img=img,
                psd_use_radial_normalization=psd_use_radial_normalization,
                psd_use_radial_image=psd_use_radial_image,
                psd_radial_bins=psd_radial_bins,
                psd_multiscale=psd_multiscale,
                psd_scales=psd_scales,
                psd_texture_mode=psd_texture_mode,
                psd_highpass_cutoff=psd_highpass_cutoff,
                psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                pixel_size_angstrom=effective_pixel_size,
                psd_frequency_band_channels=psd_frequency_band_channels,
                psd_frequency_bands=psd_frequency_bands,
                psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
                psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
            )
        psd = np.asarray(psd, dtype=np.float32)
        img = img.astype(np.float32)

        # Optional: blend edge features into real image
        if use_edge_features:
            edge = compute_edge_magnitude(img, sigma=edge_sigma, normalize=True)
            img = (1 - edge_blend_alpha) * img + edge_blend_alpha * edge
            img = np.clip(img, 0, 1).astype(np.float32)

        # Compute global features if requested
        if use_global_features:
            global_feats = compute_global_features(img)
            global_features_all = global_feats['all']  # 154-dim
        else:
            global_features_all = np.zeros(154, dtype=np.float32)

        # Compute directional edge features if requested (2 values per micrograph)
        if use_edge_direction:
            dir_strength, dir_angle = compute_global_edge_properties(img, target_size=512)
            edge_direction = np.array([dir_strength, dir_angle], dtype=np.float32)
        else:
            edge_direction = np.zeros(2, dtype=np.float32)

        np.savez_compressed(
            cache_dir / f"mic_{mic_idx}.npz",
            img=img, psd=psd, gt=region_gt.astype(np.uint8),
            global_features=global_features_all,
            edge_direction=edge_direction,  # [strength, angle]
            downsample_factor=np.array([mic_downsample], dtype=np.float32),  # Store actual factor used
            psd_multiscale_separate_channels=np.array([1 if psd_multiscale_separate_channels else 0], dtype=np.uint8),
            psd_frequency_band_channels=np.array([1 if psd_frequency_band_channels else 0], dtype=np.uint8),
            psd_frequency_band_include_full_spectrum=np.array([1 if psd_frequency_band_include_full_spectrum else 0], dtype=np.uint8),
            psd_channel_count=np.array([1 if psd.ndim == 2 else int(psd.shape[0])], dtype=np.int32),
        )
        cached_count += 1

    # Print summary
    if skipped_count > 0:
        print(f"  Cache complete: {cached_count} micrographs cached ({skipped_count} with empty GT = clean images)")

    # Save the downsample factors for reference
    if use_adaptive:
        factors_path = cache_dir / "downsample_factors.json"
        with open(factors_path, 'w') as f:
            json.dump(downsample_dict, f, indent=2)
        print(f"  Adaptive downsampling factors saved to {factors_path}")


class FastPatchDataset(Dataset):
    """Fast patch dataset: sample fixed number of patches per micrograph with GT-biased sampling.
    When cache_dir is set, loads (img, PSD, gt) from disk—no recompute of norm or PSD.

    Patch types (returned as 4th element when use_patch_type_sampling=True):
        0 = random patch
        1 = bad-centered patch (centered on GT bad region)
        2 = particle-centered patch (centered on particle pick)

    When use_psd_input=False (ablation), the PSD channels are still cached/loaded but
    not included in the model input tensor (returns 1-channel image only).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mic_dir: Path,
        rule,
        normalization_method: str,
        psd_use_radial_normalization: bool,
        patch_size: int,
        patches_per_mic: int,
        gt_bias: float = 0.5,  # Fraction of patches that contain GT (DEPRECATED if use_patch_type_sampling)
        downsample_factor: float = 1.0,
        seed: int = 42,
        use_augmentation: bool = False,  # Train: True (flip H/V), Val: False
        use_rotation_aug: bool = False,  # Train: random 90° rotations (helps with ring proteins)
        cache_dir: Optional[Path] = None,  # If set, load (img, PSD, gt) from cache—no recompute
        hard_neg_fraction: float = 0.0,  # Fraction of negative patches to sample from hard negatives (particle-likeness)
        hard_neg_quantile: float = 0.75,  # Top quantile of particle-likeness to consider "hard"
        psd_use_radial_image: bool = False,  # NEW: Use radial PSD image for stronger signal
        psd_radial_bins: int = 64,
        use_edge_features: bool = False,  # NEW: Blend edge features into real image
        edge_blend_alpha: float = 0.3,
        edge_sigma: float = 1.0,
        psd_multiscale: bool = False,  # NEW: Multi-scale radial PSD
        psd_multiscale_separate_channels: bool = False,  # NEW: Use one PSD channel per scale
        psd_scales: tuple = (16, 32, 64),
        psd_frequency_band_channels: bool = False,
        psd_frequency_bands: tuple = (),
        psd_frequency_band_include_full_spectrum: bool = False,
        psd_frequency_band_include_anisotropy: bool = False,
        psd_frequency_band_anisotropy_min_freq: float = 0.0,
        psd_texture_mode: bool = False,  # Focus on fine texture (5-40 Å) for carbon detection
        psd_highpass_cutoff: float = 0.2,  # In texture mode, ignore frequencies below this
        psd_multires_context: bool = False,  # NEW: Multi-res context PSD
        psd_context_sizes: tuple = (64, 128, 256),
        psd_bins_per_scale: int = 32,
        use_global_features: bool = False,  # NEW: Include global features
        global_feature_dim: int = 154,
        use_edge_direction: bool = False,  # NEW: Directional edge features (3 scalars)
        edge_direction_dim: int = 64,
        # Actual particle picks (positive labels)
        particles_dict: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        particle_box_size_px: int = 256,
        particle_use_gaussian: bool = True,  # Use soft Gaussian mask instead of hard box
        skip_particle_mask: bool = False,  # Skip particle mask computation (for Phase A)
        # NEW: Patch-type balanced sampling
        use_patch_type_sampling: bool = False,  # Enable explicit patch-type sampling
        bad_centered_fraction: float = 0.4,  # Fraction of patches centered on GT bad
        particle_centered_fraction: float = 0.4,  # Fraction centered on particle picks
        # (remaining fraction = random)
        use_pixel_size_channel: bool = False,
        pixel_size_by_dataset: Optional[Dict[str, float]] = None,
        pixel_size_channel_min_angstrom: float = 0.3,
        pixel_size_channel_max_angstrom: float = 1.4,
    ):
        self.df = df.reset_index(drop=True)
        self.mic_dir = mic_dir
        self.rule = rule
        self.normalization_method = normalization_method
        self.psd_use_radial_normalization = psd_use_radial_normalization
        self.psd_use_radial_image = psd_use_radial_image
        self.psd_radial_bins = psd_radial_bins
        self.use_edge_features = use_edge_features
        self.edge_blend_alpha = edge_blend_alpha
        self.edge_sigma = edge_sigma
        self.psd_multiscale = psd_multiscale
        self.psd_multiscale_separate_channels = bool(psd_multiscale_separate_channels)
        self.psd_scales = psd_scales
        self.psd_frequency_band_channels = bool(psd_frequency_band_channels)
        self.psd_frequency_bands = tuple((float(lo), float(hi)) for lo, hi in tuple(psd_frequency_bands))
        self.psd_frequency_band_include_full_spectrum = bool(psd_frequency_band_include_full_spectrum)
        self.psd_frequency_band_include_anisotropy = bool(psd_frequency_band_include_anisotropy)
        self.psd_frequency_band_anisotropy_min_freq = float(psd_frequency_band_anisotropy_min_freq)
        self.psd_texture_mode = psd_texture_mode
        self.psd_highpass_cutoff = psd_highpass_cutoff
        self.psd_multires_context = psd_multires_context
        self.psd_context_sizes = psd_context_sizes
        self.psd_bins_per_scale = psd_bins_per_scale
        self.use_global_features = use_global_features
        self.global_feature_dim = global_feature_dim
        self.use_edge_direction = use_edge_direction
        self.edge_direction_dim = edge_direction_dim
        self.patch_size = int(patch_size)
        self.patches_per_mic = int(patches_per_mic)
        self.gt_bias = float(gt_bias)
        self.downsample_factor = float(downsample_factor)
        self.rng = np.random.RandomState(seed)
        self.use_augmentation = bool(use_augmentation)
        self.use_rotation_aug = bool(use_rotation_aug)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.hard_neg_fraction = float(hard_neg_fraction)
        self.hard_neg_quantile = float(hard_neg_quantile)

        # Actual particle picks (positive labels)
        self.particles_dict = particles_dict or {}
        self.particle_box_size_px = int(particle_box_size_px)
        self.particle_use_gaussian = bool(particle_use_gaussian)
        self.skip_particle_mask = bool(skip_particle_mask)  # Skip for Phase A (performance)

        # NEW: Patch-type balanced sampling
        self.use_patch_type_sampling = bool(use_patch_type_sampling)
        self.bad_centered_fraction = float(bad_centered_fraction)
        self.particle_centered_fraction = float(particle_centered_fraction)
        self.use_pixel_size_channel = bool(use_pixel_size_channel)
        self.pixel_size_by_dataset = dict(pixel_size_by_dataset or {})
        self.pixel_size_channel_min_angstrom = float(pixel_size_channel_min_angstrom)
        self.pixel_size_channel_max_angstrom = float(pixel_size_channel_max_angstrom)

        # Ablation: whether to include PSD channels
        # Controlled externally — not a constructor param to avoid breaking existing calls
        self.use_psd_input = True  # Set to False for no-PSD ablation
        self.include_real_space_input = True  # Set to False for PSD-only ablation
        self.expected_input_channels = _expected_input_channels(
            use_psd_input=self.use_psd_input,
            psd_multiscale=self.psd_multiscale,
            psd_multiscale_separate_channels=self.psd_multiscale_separate_channels,
            psd_scales=self.psd_scales,
            psd_frequency_band_channels=self.psd_frequency_band_channels,
            psd_frequency_bands=self.psd_frequency_bands,
            psd_frequency_band_include_full_spectrum=self.psd_frequency_band_include_full_spectrum,
            psd_frequency_band_include_anisotropy=self.psd_frequency_band_include_anisotropy,
            include_real_space_input=self.include_real_space_input,
            use_pixel_size_channel=self.use_pixel_size_channel,
        )

        # In-memory LRU only when not using disk cache (keep last 4)
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
        self._cache_order: List[int] = []
        self._cache_max = 4

        # Cache for particle-likeness proxy maps (computed on-demand)
        self._particle_likeness_cache: Dict[int, np.ndarray] = {}

        # Cache for full-resolution particle masks per micrograph
        self._particle_mask_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df) * self.patches_per_mic

    def _load_global_features(self, mic_idx: int) -> np.ndarray:
        """Load global features for a micrograph. Returns zeros if not available."""
        if not self.use_global_features:
            return np.zeros(self.global_feature_dim, dtype=np.float32)

        if self.cache_dir is not None:
            npz_path = self.cache_dir / f"mic_{mic_idx}.npz"
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=True) as data:
                    if "global_features" in data:
                        return np.asarray(data["global_features"], dtype=np.float32)

        # Fallback: compute on-the-fly (slower)
        x_full, _, _ = self._load_micrograph(mic_idx)
        img = x_full[0].numpy()  # Real-space image
        global_feats = compute_global_features(img)
        return global_feats['all']

    def _load_edge_direction(self, mic_idx: int) -> np.ndarray:
        """Load edge direction (strength, angle) for a micrograph."""
        if self.cache_dir is not None:
            npz_path = self.cache_dir / f"mic_{mic_idx}.npz"
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=True) as data:
                    if "edge_direction" in data:
                        return np.asarray(data["edge_direction"], dtype=np.float32)

        # Fallback: compute on-the-fly
        x_full, _, _ = self._load_micrograph(mic_idx)
        img = x_full[0].numpy()
        dir_strength, dir_angle = compute_global_edge_properties(img, target_size=512)
        return np.array([dir_strength, dir_angle], dtype=np.float32)

    def _compute_edge_features(self, mic_idx: int, patch_center: Tuple[int, int],
                                image_shape: Tuple[int, int]) -> np.ndarray:
        """Compute 3-element edge feature vector for a patch.

        Returns: [direction_strength, direction_angle_normalized, edge_proximity]
        """
        if not self.use_edge_direction:
            return np.zeros(3, dtype=np.float32)

        # Load cached edge direction
        edge_dir = self._load_edge_direction(mic_idx)
        dir_strength = edge_dir[0]
        dir_angle = edge_dir[1]

        # Compute patch-specific proximity
        edge_proximity = compute_edge_proximity(
            patch_center, image_shape, dir_angle, dir_strength
        )

        # Normalize angle to [-1, 1] range for network input
        angle_normalized = dir_angle / 180.0

        return np.array([dir_strength, angle_normalized, edge_proximity], dtype=np.float32)

    def _load_micrograph(self, mic_idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Load micrograph (image+PSD) and GT mask. From disk cache if cache_dir set, else compute (with LRU)."""
        if mic_idx in self._cache:
            self._cache_order.remove(mic_idx)
            self._cache_order.append(mic_idx)
            return self._cache[mic_idx]

        if self.cache_dir is not None:
            npz_path = self.cache_dir / f"mic_{mic_idx}.npz"
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=True) as data:
                    img = np.asarray(data["img"], dtype=np.float32)
                    psd = np.asarray(data["psd"], dtype=np.float32)
                    region_gt = np.asarray(data["gt"])
                    cached_downsample = float(np.asarray(data["downsample_factor"]).squeeze()) if "downsample_factor" in data else float(self.downsample_factor)
                dataset_id = str(self.df.iloc[mic_idx]["dataset_id"])
                pixel_size_channel = None
                if self.use_pixel_size_channel:
                    pixel_size_channel = _build_pixel_size_channel(
                        image_shape=img.shape,
                        dataset_id=dataset_id,
                        pixel_size_by_dataset=self.pixel_size_by_dataset,
                        min_angstrom=self.pixel_size_channel_min_angstrom,
                        max_angstrom=self.pixel_size_channel_max_angstrom,
                        downsample_factor=cached_downsample,
                    )
                effective_pixel_size = _effective_pixel_size_for_dataset(
                    dataset_id=dataset_id,
                    pixel_size_by_dataset=self.pixel_size_by_dataset,
                    downsample_factor=cached_downsample,
                )
                if self.use_psd_input and self.psd_frequency_band_channels:
                    expected_band_channels = (
                        len(self.psd_frequency_bands)
                        + (1 if self.psd_frequency_band_include_full_spectrum else 0)
                        + (1 if self.psd_frequency_band_include_anisotropy else 0)
                    )
                    if not (psd.ndim == 3 and psd.shape[0] == expected_band_channels):
                        psd = compute_frequency_band_psd_channels(
                            img,
                            pixel_size_angstrom=effective_pixel_size,
                            frequency_bands=self.psd_frequency_bands,
                            normalize=True,
                            use_radial_normalization=bool(self.psd_use_radial_normalization),
                            include_full_spectrum_channel=bool(self.psd_frequency_band_include_full_spectrum),
                            include_anisotropy_channel=bool(self.psd_frequency_band_include_anisotropy),
                            anisotropy_min_frequency=float(self.psd_frequency_band_anisotropy_min_freq),
                        )
                elif self.use_psd_input and self.psd_multiscale and self.psd_multiscale_separate_channels and psd.ndim == 2:
                    # Backward compatibility for old caches that only stored fused PSD.
                    psd = compute_multiscale_radial_psd_channels(
                        img,
                        scales=self.psd_scales,
                        normalize=True,
                        texture_mode=self.psd_texture_mode,
                        highpass_cutoff=self.psd_highpass_cutoff,
                    )
                if self.use_psd_input and (not self.psd_multiscale_separate_channels) and psd.ndim == 3:
                    if psd.shape[0] == 1:
                        psd = psd[0]
                    else:
                        raise ValueError(
                            f"Cache PSD has {psd.shape[0]} channels but fused PSD mode is active. "
                            "Use a fresh output_dir/cache for this configuration."
                        )
                x_np = _stack_image_with_psd(
                    img,
                    psd,
                    use_psd_input=self.use_psd_input,
                    include_real_space_input=self.include_real_space_input,
                    pixel_size_channel=pixel_size_channel,
                )
                x = torch.from_numpy(x_np).float()
                gt_t = torch.from_numpy(region_gt).float()
                if len(self._cache) >= self._cache_max:
                    oldest = self._cache_order.pop(0)
                    del self._cache[oldest]
                self._cache[mic_idx] = (x, gt_t, region_gt)
                self._cache_order.append(mic_idx)
                return x, gt_t, region_gt

        r = self.df.iloc[mic_idx]
        dataset_id = str(r["dataset_id"])
        stem = str(r["stem"])
        gt_path_raw = str(r["gt_mask_path"])
        gt_path = Path(gt_path_raw)

        # Try relative to cwd first (manifest paths like 'particle_annotations/merged_gt_fullres/...')
        if not gt_path.exists():
            gt_path_cwd = Path.cwd() / gt_path_raw
            if gt_path_cwd.exists():
                gt_path = gt_path_cwd

        # Handle OLD manifest format: mc_fullres_gt_masks_642/10090__10045.__gt_full.npy
        # Map to NEW format: particle_annotations/merged_gt_fullres/10090__10045..npy
        if not gt_path.exists() and "mc_fullres_gt_masks_642" in gt_path_raw:
            old_filename = Path(gt_path_raw).name
            new_filename = old_filename.replace("__gt_full", "")
            new_path = Path.cwd() / "particle_annotations" / "merged_gt_fullres" / new_filename
            if new_path.exists():
                gt_path = new_path

        # If still not found, try other locations
        if not gt_path.exists():
            gt_filename = Path(gt_path_raw).name
            alt_name = f"{dataset_id}__{stem}.npy"
            candidates = [
                Path.cwd() / "particle_annotations" / "merged_gt_fullres" / gt_filename,
                Path.cwd() / "particle_annotations" / "merged_gt_fullres" / alt_name,
                self.mic_dir.parent / gt_path_raw,
                self.mic_dir.parent.parent / gt_path_raw,
            ]
            for candidate in candidates:
                if candidate.exists():
                    gt_path = candidate
                    break
        if "micrograph_path" in r and pd.notna(r["micrograph_path"]):
            mic_path = Path(r["micrograph_path"])
            if not mic_path.exists():
                mic_path = _find_mic_path(self.mic_dir, dataset_id, stem)
        else:
            mic_path = _find_mic_path(self.mic_dir, dataset_id, stem)
        img = _load_mrc_2d_permissive(mic_path)
        img = normalize_image(img, method=self.normalization_method)
        img = np.asarray(img).squeeze()
        gt = np.load(str(gt_path))
        gt = np.asarray(gt).squeeze()
        region_gt = apply_region_gt_transform(gt, dataset_id=dataset_id, rule=self.rule).astype(np.uint8)
        if self.downsample_factor > 1.0:
            img = _downsample_image(img, self.downsample_factor, order=1)
            region_gt = _downsample_image(region_gt, self.downsample_factor, order=0)
        effective_pixel_size = _effective_pixel_size_for_dataset(
            dataset_id=dataset_id,
            pixel_size_by_dataset=self.pixel_size_by_dataset,
            downsample_factor=self.downsample_factor,
        )
        # Compute PSD - use radial image if requested (stronger signal)
        psd = _compute_psd_map(
            img=img,
            psd_use_radial_normalization=self.psd_use_radial_normalization,
            psd_use_radial_image=self.psd_use_radial_image,
            psd_radial_bins=self.psd_radial_bins,
            psd_multiscale=self.psd_multiscale,
            psd_scales=self.psd_scales,
            psd_texture_mode=self.psd_texture_mode,
            psd_highpass_cutoff=self.psd_highpass_cutoff,
            psd_multiscale_separate_channels=self.psd_multiscale_separate_channels,
            pixel_size_angstrom=effective_pixel_size,
            psd_frequency_band_channels=self.psd_frequency_band_channels,
            psd_frequency_bands=self.psd_frequency_bands,
            psd_frequency_band_include_full_spectrum=self.psd_frequency_band_include_full_spectrum,
            psd_frequency_band_include_anisotropy=self.psd_frequency_band_include_anisotropy,
            psd_frequency_band_anisotropy_min_freq=self.psd_frequency_band_anisotropy_min_freq,
        )
        img = img.astype(np.float32)

        # Optional: blend edge features into real image
        if self.use_edge_features:
            edge = compute_edge_magnitude(img, sigma=self.edge_sigma, normalize=True)
            img = (1 - self.edge_blend_alpha) * img + self.edge_blend_alpha * edge
            img = np.clip(img, 0, 1).astype(np.float32)
        pixel_size_channel = None
        if self.use_pixel_size_channel:
            pixel_size_channel = _build_pixel_size_channel(
                image_shape=img.shape,
                dataset_id=dataset_id,
                pixel_size_by_dataset=self.pixel_size_by_dataset,
                min_angstrom=self.pixel_size_channel_min_angstrom,
                max_angstrom=self.pixel_size_channel_max_angstrom,
                downsample_factor=self.downsample_factor,
            )

        x_np = _stack_image_with_psd(
            img,
            psd,
            use_psd_input=self.use_psd_input,
            include_real_space_input=self.include_real_space_input,
            pixel_size_channel=pixel_size_channel,
        )
        x = torch.from_numpy(x_np).float()
        gt_t = torch.from_numpy(region_gt).float()
        if len(self._cache) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[mic_idx] = (x, gt_t, region_gt)
        self._cache_order.append(mic_idx)
        return x, gt_t, region_gt

    def _get_particle_likeness_proxy(self, mic_idx: int, img: np.ndarray, gt_np: np.ndarray) -> np.ndarray:
        """Get or compute particle-likeness proxy map for a micrograph."""
        if mic_idx in self._particle_likeness_cache:
            return self._particle_likeness_cache[mic_idx]
        proxy = compute_particle_likeness_proxy(img, gt_np)
        self._particle_likeness_cache[mic_idx] = proxy
        return proxy

    def _get_particle_mask_full(self, mic_idx: int, H: int, W: int) -> np.ndarray:
        """Get or compute full-resolution particle mask for a micrograph (at downsampled resolution)."""
        if mic_idx in self._particle_mask_cache:
            return self._particle_mask_cache[mic_idx]

        # Look up particles for this micrograph with multiple key formats
        r = self.df.iloc[mic_idx]
        dataset_id = str(r['dataset_id'])
        stem = str(r['stem'])

        # Try multiple key formats
        keys_to_try = [
            f"{dataset_id}__{stem}",
            stem,
            f"{dataset_id}_{stem}",
        ]
        particles = []
        for key in keys_to_try:
            if key in self.particles_dict:
                particles = self.particles_dict[key]
                break

        # Build mask at downsampled resolution (Gaussian or hard disk)
        mask = build_particle_mask(
            particles, H, W,
            box_size_px=self.particle_box_size_px,
            downsample_factor=self.downsample_factor,
            use_gaussian=self.particle_use_gaussian,
        )

        self._particle_mask_cache[mic_idx] = mask
        return mask

    def _get_particle_coords_for_mic(self, mic_idx: int) -> List[Tuple[int, int]]:
        """Get particle coordinates for a micrograph (in downsampled coordinates)."""
        row = self.df.iloc[mic_idx]
        dataset_id = row.get("dataset_id", "")
        stem = row.get("stem", "")

        # Try multiple key formats
        keys_to_try = [
            f"{dataset_id}__{stem}",
            f"{dataset_id}_{stem}",
            stem,
            str(dataset_id),
        ]

        for key in keys_to_try:
            if key in self.particles_dict:
                particles = self.particles_dict[key]
                # Convert from full-res to downsampled coords
                ds = self.downsample_factor
                return [(int(x / ds), int(y / ds)) for x, y in particles]

        return []

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Sample a random patch with GT-biased or patch-type balanced sampling.

        Returns:
            patch_x: (C, H, W) image + PSD channels
            patch_gt: (H, W) GT mask
            patch_particle: (H, W) particle mask (zeros if no particles_dict)
            patch_type: int (0=random, 1=bad-centered, 2=particle-centered)
        """
        mic_idx = idx // self.patches_per_mic
        patch_idx = idx % self.patches_per_mic

        x_full, gt_full, gt_np = self._load_micrograph(mic_idx)
        _, H, W = x_full.shape

        # Determine patch type and sampling location
        patch_type = 0  # Default: random

        if self.use_patch_type_sampling:
            # NEW: Patch-type balanced sampling
            rand_val = self.rng.rand()
            particle_coords = self._get_particle_coords_for_mic(mic_idx)

            if rand_val < self.bad_centered_fraction and gt_np.sum() > 0:
                # Sample bad-centered patch
                patch_type = 1
                gt_coords = np.argwhere(gt_np > 0.5)
                center = gt_coords[self.rng.randint(len(gt_coords))]
                y0 = max(0, min(H - self.patch_size, center[0] - self.patch_size // 2))
                x0 = max(0, min(W - self.patch_size, center[1] - self.patch_size // 2))
            elif rand_val < self.bad_centered_fraction + self.particle_centered_fraction and len(particle_coords) > 0:
                # Sample particle-centered patch
                patch_type = 2
                px, py = particle_coords[self.rng.randint(len(particle_coords))]
                # Note: particle coords are (x, y), but numpy is (row, col) = (y, x)
                y0 = max(0, min(H - self.patch_size, py - self.patch_size // 2))
                x0 = max(0, min(W - self.patch_size, px - self.patch_size // 2))
            else:
                # Random patch
                patch_type = 0
                y0 = self.rng.randint(0, max(1, H - self.patch_size))
                x0 = self.rng.randint(0, max(1, W - self.patch_size))
        else:
            # LEGACY: GT-biased sampling
            if self.gt_bias > 0 and gt_np.sum() > 0 and self.rng.rand() < self.gt_bias:
                # Sample patch that contains GT
                patch_type = 1  # bad-centered (for compatibility)
                gt_coords = np.argwhere(gt_np > 0.5)
                if len(gt_coords) > 0:
                    center = gt_coords[self.rng.randint(len(gt_coords))]
                    y0 = max(0, min(H - self.patch_size, center[0] - self.patch_size // 2))
                    x0 = max(0, min(W - self.patch_size, center[1] - self.patch_size // 2))
                else:
                    y0 = self.rng.randint(0, max(1, H - self.patch_size))
                    x0 = self.rng.randint(0, max(1, W - self.patch_size))
            else:
                # Negative patch sampling: random OR hard-negative (particle-likeness)
                if self.hard_neg_fraction > 0 and self.rng.rand() < self.hard_neg_fraction:
                    # Hard-negative sampling: sample from top-quantile particle-likeness regions
                    img_np = x_full[0].detach().cpu().numpy()
                    proxy = self._get_particle_likeness_proxy(mic_idx, img_np, gt_np)

                    gt_for_proxy = gt_np
                    if gt_np.shape != proxy.shape:
                        from scipy.ndimage import zoom
                        scale_y = proxy.shape[0] / gt_np.shape[0]
                        scale_x = proxy.shape[1] / gt_np.shape[1]
                        gt_for_proxy = zoom(gt_np.astype(float), (scale_y, scale_x), order=0)

                    threshold = np.quantile(proxy[gt_for_proxy < 0.5], self.hard_neg_quantile) if (gt_for_proxy < 0.5).any() else 0.0
                    hard_neg_mask = (proxy >= threshold) & (gt_for_proxy < 0.5)

                    if hard_neg_mask.any():
                        hard_neg_coords = np.argwhere(hard_neg_mask)
                        center = hard_neg_coords[self.rng.randint(len(hard_neg_coords))]
                        y0 = max(0, min(H - self.patch_size, center[0] - self.patch_size // 2))
                        x0 = max(0, min(W - self.patch_size, center[1] - self.patch_size // 2))
                    else:
                        y0 = self.rng.randint(0, max(1, H - self.patch_size))
                        x0 = self.rng.randint(0, max(1, W - self.patch_size))
                else:
                    y0 = self.rng.randint(0, max(1, H - self.patch_size))
                    x0 = self.rng.randint(0, max(1, W - self.patch_size))

        patch_x = x_full[:, y0:y0+self.patch_size, x0:x0+self.patch_size]
        patch_gt = gt_full[y0:y0+self.patch_size, x0:x0+self.patch_size]

        # Multi-resolution context PSD: compute PSD on larger regions centered on this patch
        if self.psd_multires_context:
            # Get full image (channel 0 is the image, channel 1 is placeholder PSD)
            full_img_np = x_full[0].numpy()  # (H, W)
            patch_center = (y0 + self.patch_size // 2, x0 + self.patch_size // 2)

            # Compute multi-res context PSD image for this patch location
            psd_multires = compute_multires_psd_image(
                full_img_np,
                patch_center=patch_center,
                patch_size=self.patch_size,
                context_sizes=self.psd_context_sizes,
                bins_per_scale=self.psd_bins_per_scale,
            )
            # Replace PSD channels with the computed multi-res PSD.
            if patch_x.shape[0] > 1:
                psd_multires_t = torch.from_numpy(psd_multires).float()
                if patch_x.shape[0] == 2:
                    patch_x[1] = psd_multires_t
                else:
                    patch_x[1:] = psd_multires_t.unsqueeze(0).repeat(patch_x.shape[0] - 1, 1, 1)

        # Extract particle mask patch (if particles_dict provided AND not skipped)
        # Skip in Phase A for performance - workers return zeros, no expensive computation
        if self.particles_dict and not self.skip_particle_mask:
            particle_mask_full = self._get_particle_mask_full(mic_idx, H, W)
            # Safely extract patch, handling potential shape mismatches
            pm_h, pm_w = particle_mask_full.shape
            y1 = min(y0 + self.patch_size, pm_h)
            x1 = min(x0 + self.patch_size, pm_w)
            patch_pm = particle_mask_full[y0:y1, x0:x1].copy()
            # Pad to patch_size if needed
            if patch_pm.shape != (self.patch_size, self.patch_size):
                padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                padded[:patch_pm.shape[0], :patch_pm.shape[1]] = patch_pm
                patch_pm = padded
            patch_particle = torch.from_numpy(patch_pm).float()
        else:
            # Fast path: return zeros (Phase A or no particle dict)
            patch_particle = torch.zeros(self.patch_size, self.patch_size, dtype=torch.float32)

        # Ensure all patches have consistent shapes (pad if edge cases)
        expected_channels = _expected_input_channels(
            use_psd_input=self.use_psd_input,
            psd_multiscale=self.psd_multiscale,
            psd_multiscale_separate_channels=self.psd_multiscale_separate_channels,
            psd_scales=self.psd_scales,
            include_real_space_input=self.include_real_space_input,
            use_pixel_size_channel=self.use_pixel_size_channel,
        )
        if patch_x.shape != (expected_channels, self.patch_size, self.patch_size):
            padded = torch.zeros(expected_channels, self.patch_size, self.patch_size, dtype=patch_x.dtype)
            c = min(expected_channels, patch_x.shape[0])
            h_copy = min(self.patch_size, patch_x.shape[1])
            w_copy = min(self.patch_size, patch_x.shape[2])
            padded[:c, :h_copy, :w_copy] = patch_x[:c, :h_copy, :w_copy]
            patch_x = padded
        if patch_gt.shape != (self.patch_size, self.patch_size):
            padded = torch.zeros(self.patch_size, self.patch_size, dtype=patch_gt.dtype)
            padded[:patch_gt.shape[0], :patch_gt.shape[1]] = patch_gt
            patch_gt = padded

        # Augmentation (train only)
        if self.use_augmentation:
            # Random H/V flips
            flip_h = self.rng.rand() < 0.5
            flip_v = self.rng.rand() < 0.5
            if flip_h:
                patch_x = torch.flip(patch_x, [-1])
                patch_gt = torch.flip(patch_gt, [-1])
                patch_particle = torch.flip(patch_particle, [-1])
            if flip_v:
                patch_x = torch.flip(patch_x, [-2])
                patch_gt = torch.flip(patch_gt, [-2])
                patch_particle = torch.flip(patch_particle, [-2])

            # Random 90-degree rotations (0, 90, 180, 270) - fast, no interpolation
            if self.use_rotation_aug:
                k = self.rng.randint(0, 4)  # 0, 1, 2, or 3 (number of 90° rotations)
                if k > 0:
                    patch_x = torch.rot90(patch_x, k, dims=[-2, -1])
                    patch_gt = torch.rot90(patch_gt, k, dims=[-2, -1])
                    patch_particle = torch.rot90(patch_particle, k, dims=[-2, -1])

        # Load global features (154-dim) if enabled
        if self.use_global_features:
            global_features = self._load_global_features(mic_idx)
            global_features = torch.from_numpy(global_features).float()
        else:
            global_features = torch.zeros(self.global_feature_dim, dtype=torch.float32)

        # Compute edge direction features (3 scalars) if enabled
        if self.use_edge_direction:
            # Get image shape from x_full (loaded earlier)
            _, H, W = x_full.shape
            # Patch center is at (y0 + patch_size//2, x0 + patch_size//2)
            patch_center = (y0 + self.patch_size // 2, x0 + self.patch_size // 2)
            edge_features = self._compute_edge_features(mic_idx, patch_center, (H, W))
            edge_features = torch.from_numpy(edge_features).float()
        else:
            edge_features = torch.zeros(3, dtype=torch.float32)

        return patch_x, patch_gt, patch_particle, patch_type, global_features, edge_features


class ViTMicrographDataset(Dataset):
    """
    Dataset for Vision Transformer training that returns ALL patches from a micrograph.

    Unlike FastPatchDataset which returns individual random patches, this dataset
    returns the entire micrograph as a grid of patches, enabling self-attention
    across all patches.

    Each __getitem__ call returns:
    - patches: [N, C, patch_size, patch_size] - all patches (real + PSD channels)
    - labels: [N, patch_size, patch_size] - GT labels for all patches
    - particle_masks: [N, patch_size, patch_size] - particle locations for all patches
    - grid_shape: (grid_h, grid_w) - shape of patch grid
    - num_patches: N - total number of patches

    Where N = grid_h * grid_w patches arranged in a regular grid.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mic_dir: Path,
        rule,
        normalization_method: str,
        psd_use_radial_normalization: bool,
        patch_size: int,
        downsample_factor: float = 1.0,
        seed: int = 42,
        use_augmentation: bool = False,
        cache_dir: Optional[Path] = None,
        use_psd_input: bool = True,
        include_real_space_input: bool = True,
        psd_multiscale: bool = False,
        psd_scales: tuple = (16, 32, 64),
        psd_texture_mode: bool = False,  # Focus on fine texture (5-40 Å) for carbon detection
        psd_highpass_cutoff: float = 0.2,  # In texture mode, ignore frequencies below this
        # Particle annotations
        particles_dict: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        particle_box_size_px: int = 256,
        # Noise augmentation for clean patches
        use_noise_augmentation: bool = False,
        noise_aug_prob: float = 0.5,  # Probability of applying noise aug to eligible patches
        noise_strength: float = 1.0,  # Strength multiplier for noise (0.0-1.0), can be updated per-epoch
        # Overlapping patches (prevents blocky artifacts)
        stride: Optional[int] = None,  # If None, stride = patch_size (no overlap). Set to patch_size//2 for 50% overlap
    ):
        self.df = df.reset_index(drop=True)
        self.mic_dir = mic_dir
        self.rule = rule
        self.normalization_method = normalization_method
        self.psd_use_radial_normalization = psd_use_radial_normalization
        self.patch_size = int(patch_size)
        self.stride = int(stride) if stride is not None else self.patch_size  # Default: no overlap
        self.downsample_factor = float(downsample_factor)
        self.rng = np.random.RandomState(seed)
        self.use_augmentation = bool(use_augmentation)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_psd_input = bool(use_psd_input)
        self.include_real_space_input = bool(include_real_space_input)
        self.psd_multiscale = psd_multiscale
        self.psd_scales = psd_scales
        self.psd_texture_mode = psd_texture_mode
        self.psd_highpass_cutoff = psd_highpass_cutoff
        self.particles_dict = particles_dict or {}
        self.particle_box_size_px = int(particle_box_size_px)
        self.use_noise_augmentation = bool(use_noise_augmentation)
        self.noise_aug_prob = float(noise_aug_prob)
        self.noise_strength = float(noise_strength)  # Can be updated via set_noise_strength()

        # LRU cache for loaded micrographs
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}
        self._cache_order: List[int] = []
        self._cache_max = 8

    def set_noise_strength(self, strength: float):
        """Update noise strength for curriculum learning. Called at start of each epoch."""
        self.noise_strength = float(max(0.0, min(1.0, strength)))

    def __len__(self) -> int:
        return len(self.df)

    def _load_micrograph(self, mic_idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Load micrograph (image+PSD) and GT mask from cache."""
        if mic_idx in self._cache:
            self._cache_order.remove(mic_idx)
            self._cache_order.append(mic_idx)
            return self._cache[mic_idx]

        if self.cache_dir is not None:
            npz_path = self.cache_dir / f"mic_{mic_idx}.npz"
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=True) as data:
                    img = np.asarray(data["img"], dtype=np.float32)
                    psd = np.asarray(data["psd"], dtype=np.float32)
                    region_gt = np.asarray(data["gt"])
                x = torch.from_numpy(
                    _stack_image_with_psd(
                        img,
                        psd,
                        use_psd_input=self.use_psd_input,
                        include_real_space_input=self.include_real_space_input,
                    )
                ).float()
                gt_t = torch.from_numpy(region_gt).float()
                if len(self._cache) >= self._cache_max:
                    oldest = self._cache_order.pop(0)
                    del self._cache[oldest]
                self._cache[mic_idx] = (x, gt_t, region_gt)
                self._cache_order.append(mic_idx)
                return x, gt_t, region_gt

        raise FileNotFoundError(f"Cache not found for micrograph {mic_idx}. Run cache building first.")

    def _get_particle_mask(self, mic_idx: int, H: int, W: int) -> np.ndarray:
        """Get particle mask for micrograph."""
        r = self.df.iloc[mic_idx]
        dataset_id = str(r["dataset_id"])
        stem = str(r["stem"])

        # Try multiple key formats (same as FastPatchDataset)
        keys_to_try = [
            f"{dataset_id}__{stem}",
            stem,
            f"{dataset_id}_{stem}",
        ]

        particles = []
        for key in keys_to_try:
            if key in self.particles_dict:
                particles = self.particles_dict[key]
                break

        particle_mask = np.zeros((H, W), dtype=np.float32)
        if particles:
            # Downsample particle box size
            box_ds = int(self.particle_box_size_px / self.downsample_factor)
            half_box = box_ds // 2

            for (py, px) in particles:
                # Downsample particle center
                py_ds = int(py / self.downsample_factor)
                px_ds = int(px / self.downsample_factor)

                y1 = max(0, py_ds - half_box)
                y2 = min(H, py_ds + half_box)
                x1 = max(0, px_ds - half_box)
                x2 = min(W, px_ds + half_box)
                particle_mask[y1:y2, x1:x2] = 1.0

        return particle_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return all patches from a single micrograph.

        Returns:
            Dict with:
            - patches: [N, 2, patch_size, patch_size]
            - labels: [N, patch_size, patch_size]
            - particle_masks: [N, patch_size, patch_size]
            - grid_shape: (grid_h, grid_w)
            - num_patches: N
            - mic_idx: int
        """
        mic_idx = idx

        # Load full micrograph
        x_full, gt_full, gt_np = self._load_micrograph(mic_idx)
        _, H, W = x_full.shape

        # Ensure gt_full matches image size (resize if needed)
        gt_H, gt_W = gt_full.shape
        if gt_H != H or gt_W != W:
            gt_full = F.interpolate(
                gt_full.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(0).squeeze(0)

        # Calculate grid dimensions using stride (allows overlap when stride < patch_size)
        # Number of patches = (dim - patch_size) // stride + 1
        grid_h = (H - self.patch_size) // self.stride + 1
        grid_w = (W - self.patch_size) // self.stride + 1
        num_patches = grid_h * grid_w

        # Get particle mask
        particle_mask_full = self._get_particle_mask(mic_idx, H, W)

        # Extract all patches using sliding window with stride
        # When stride < patch_size, patches overlap (e.g., stride=128 with patch_size=256 = 50% overlap)
        patches = []
        labels = []
        particle_masks = []
        positions = []  # Track (y0, x0) for each patch (needed for overlapping prediction stitching)

        for i in range(grid_h):
            for j in range(grid_w):
                y0 = i * self.stride
                x0 = j * self.stride
                y1 = y0 + self.patch_size
                x1 = x0 + self.patch_size

                # Safety check - skip if out of bounds
                if y1 > H or x1 > W:
                    continue

                # Extract patch (both channels: real + PSD)
                patch = x_full[:, y0:y1, x0:x1]
                patches.append(patch)

                # Extract label
                label = gt_full[y0:y1, x0:x1]
                labels.append(label)

                # Extract particle mask
                pmask = particle_mask_full[y0:y1, x0:x1]
                particle_masks.append(torch.from_numpy(pmask).float())

                # Track position for reconstruction
                positions.append((y0, x0))

        # Stack all patches
        patches = torch.stack(patches, dim=0)  # [N, 2, H, W]
        labels = torch.stack(labels, dim=0)  # [N, H, W]
        particle_masks = torch.stack(particle_masks, dim=0)  # [N, H, W]

        # Update num_patches and grid_shape to actual count
        num_patches = len(patches)
        actual_grid_h = grid_h
        actual_grid_w = grid_w

        # Store image shape for reconstruction
        image_shape = (H, W)

        # Data augmentation (apply same transform to all patches)
        if self.use_augmentation:
            # Random horizontal flip
            if self.rng.rand() > 0.5:
                patches = torch.flip(patches, dims=[-1])
                labels = torch.flip(labels, dims=[-1])
                particle_masks = torch.flip(particle_masks, dims=[-1])
            # Random vertical flip
            if self.rng.rand() > 0.5:
                patches = torch.flip(patches, dims=[-2])
                labels = torch.flip(labels, dims=[-2])
                particle_masks = torch.flip(particle_masks, dims=[-2])
            # Random 90-degree rotations (0, 90, 180, 270)
            k = self.rng.randint(0, 4)  # 0, 1, 2, or 3 (number of 90° rotations)
            if k > 0:
                patches = torch.rot90(patches, k, dims=[-2, -1])
                labels = torch.rot90(labels, k, dims=[-2, -1])
                particle_masks = torch.rot90(particle_masks, k, dims=[-2, -1])

            # Noise augmentation for clean patches (teaches: noisy clean ≠ contamination)
            # Only apply if noise_strength > 0 (allows curriculum-based ramping)
            if self.use_noise_augmentation and self.noise_strength > 0 and self.rng.rand() < self.noise_aug_prob:
                # Apply noise augmentation to clean patches in this micrograph
                # Only augment the real-space channel (channel 0), not PSD (channel 1)
                for p_idx in range(len(patches)):
                    # Calculate contamination fraction for this patch
                    contam_frac = labels[p_idx].float().mean().item()

                    if contam_frac < 0.1:  # Only augment clean patches (<10% contamination)
                        # Convert to numpy, augment, convert back
                        real_space = patches[p_idx, 0].numpy()
                        augmented = augment_with_cryoem_noise(
                            real_space,
                            contamination_fraction=contam_frac,
                            rng=self.rng,
                            strength_multiplier=self.noise_strength,  # Curriculum-controlled
                            noise_prob=0.7,  # High chance of Gaussian noise
                            texture_prob=0.4,  # Moderate chance of crystalline texture
                            gradient_prob=0.3,  # Some ice thickness variation
                            contrast_prob=0.3,  # Local contrast variation
                        )
                        patches[p_idx, 0] = torch.from_numpy(augmented)

        return {
            'patches': patches,  # [N, 2, patch_size, patch_size]
            'labels': labels,  # [N, patch_size, patch_size]
            'particle_masks': particle_masks,  # [N, patch_size, patch_size]
            'grid_shape': (actual_grid_h, actual_grid_w),
            'num_patches': num_patches,
            'mic_idx': mic_idx,
            'positions': positions,  # List of (y0, x0) for each patch
            'image_shape': image_shape,  # (H, W) for reconstruction
            'stride': self.stride,  # Stride used for extraction
        }


def vit_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for ViT dataset.

    Handles variable number of patches per micrograph by padding to max.

    Args:
        batch: List of dicts from ViTMicrographDataset

    Returns:
        Batched dict with:
        - patches: [B, max_N, 2, H, W]
        - labels: [B, max_N, H, W]
        - particle_masks: [B, max_N, H, W]
        - grid_shapes: List of (grid_h, grid_w)
        - num_patches: [B] tensor of actual patch counts
        - mask: [B, max_N] attention mask (1 = valid, 0 = padding)
        - positions: List of List of (y0, x0) tuples for each batch item
        - image_shapes: List of (H, W) tuples for reconstruction
        - strides: List of stride values
    """
    B = len(batch)
    max_N = max(b['num_patches'] for b in batch)
    patch_size = batch[0]['patches'].shape[-1]

    # Initialize tensors
    patches = torch.zeros(B, max_N, 2, patch_size, patch_size)
    labels = torch.zeros(B, max_N, patch_size, patch_size)
    particle_masks = torch.zeros(B, max_N, patch_size, patch_size)
    mask = torch.zeros(B, max_N)
    num_patches = torch.zeros(B, dtype=torch.long)
    grid_shapes = []
    mic_indices = []
    positions = []
    image_shapes = []
    strides = []

    for i, b in enumerate(batch):
        N = b['num_patches']
        patches[i, :N] = b['patches']
        labels[i, :N] = b['labels']
        particle_masks[i, :N] = b['particle_masks']
        mask[i, :N] = 1
        num_patches[i] = N
        grid_shapes.append(b['grid_shape'])
        mic_indices.append(b['mic_idx'])
        positions.append(b.get('positions', []))
        image_shapes.append(b.get('image_shape', (0, 0)))
        strides.append(b.get('stride', patch_size))

    return {
        'patches': patches,
        'labels': labels,
        'particle_masks': particle_masks,
        'mask': mask,
        'num_patches': num_patches,
        'grid_shapes': grid_shapes,
        'mic_indices': mic_indices,
        'positions': positions,
        'image_shapes': image_shapes,
        'strides': strides,
    }


def create_tukey_edge_mask(height: int, width: int, edge_px: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a Tukey-style edge mask for patch-based loss computation.

    The mask is 1.0 in the center and tapers smoothly to 0 at the edges.
    This prevents the model from learning 'edge of context' artifacts
    that cause swath patterns at patch boundaries during inference.

    Args:
        height: Patch height
        width: Patch width
        edge_px: Number of pixels to taper at edges (flat region starts at edge_px from border)
        device: Device for tensor (default: CPU)

    Returns:
        (H, W) tensor with values in [0, 1], smooth Tukey window
    """
    if edge_px <= 0:
        return torch.ones(height, width, device=device)

    # Create 1D Tukey-like windows for each dimension
    def tukey_1d(size: int, edge: int) -> np.ndarray:
        """1D Tukey window: flat center, cosine taper at edges"""
        if edge >= size // 2:
            # All taper, use Hann
            return np.hanning(size)

        window = np.ones(size)
        # Taper at start
        taper = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
        window[:edge] = taper
        # Taper at end
        window[-edge:] = taper[::-1]
        return window

    h_window = tukey_1d(height, edge_px)
    w_window = tukey_1d(width, edge_px)

    # 2D window is outer product
    mask_2d = np.outer(h_window, w_window).astype(np.float32)

    if device is not None:
        return torch.from_numpy(mask_2d).to(device)
    return torch.from_numpy(mask_2d)


# Cache for Tukey masks (avoid recomputing every batch)
_tukey_mask_cache = {}


def get_tukey_edge_mask(height: int, width: int, edge_px: int, device: torch.device) -> torch.Tensor:
    """Cached version of create_tukey_edge_mask."""
    key = (height, width, edge_px, str(device))
    if key not in _tukey_mask_cache:
        _tukey_mask_cache[key] = create_tukey_edge_mask(height, width, edge_px, device)
    return _tukey_mask_cache[key]


def compute_dice_loss(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice loss for binary segmentation.

    Args:
        probs: Predicted probabilities (B, H, W) or (B, 1, H, W), in [0, 1]
        targets: Ground truth binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice loss (1 - Dice coefficient)
    """
    # Flatten to (B, -1)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1).float()

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def get_warmup_lambda(epoch: int, warmup_end_epoch: int, final_value: float) -> float:
    """
    Linear warmup for loss term weights.
    Returns 0 before warmup starts, linearly ramps to final_value.

    Args:
        epoch: Current epoch (1-indexed)
        warmup_end_epoch: Epoch at which warmup completes (term reaches final_value)
        final_value: Final lambda value after warmup

    Returns:
        Current lambda value
    """
    if epoch >= warmup_end_epoch:
        return final_value
    # Linear ramp from 0 at epoch 1 to final_value at warmup_end_epoch
    return final_value * (epoch / warmup_end_epoch)


def compute_stable_loss(
    logits: torch.Tensor,  # (B, 1, H, W)
    gt_mask: torch.Tensor,  # (B, 1, H, W) or (B, H, W), binary: 1=bad, 0=good
    gt_core: torch.Tensor,  # (B, H, W) eroded GT core
    good_anchor_mask: torch.Tensor = None,  # (B, H, W) particle locations
    epoch: int = 1,
    # Phase 1: BCE + Dice (always on)
    lambda_bce: float = 1.0,
    lambda_dice: float = 1.0,
    bce_neg_weight: float = 3.0,
    # Phase 2: Hit-core (warmup after warmup_phase1_epochs)
    lambda_hit_core: float = 2.0,
    warmup_phase1_epochs: int = 5,  # BCE+Dice only for this many epochs
    # Phase 3: Particle anchor suppression (warmup after warmup_phase2_epochs)
    lambda_particle_anchor: float = 0.3,
    warmup_phase2_epochs: int = 10,  # Add hit-core, then particle suppression
    tau_anchor: float = 0.15,  # Hinge threshold for particle anchors
    # Phase 4: Area budget (warmup after warmup_phase3_epochs)
    lambda_area: float = 0.3,
    warmup_phase3_epochs: int = 15,
    rho_area: float = 0.05,  # Target bad fraction in unknown regions
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stable loss with phased warmup for training transformers/large models.

    Training phases:
      Phase 1 (epochs 1-5): BCE + Dice only - establish basic segmentation
      Phase 2 (epochs 6-10): Add hit-core - ensure coverage of contamination cores
      Phase 3 (epochs 11-15): Add particle anchor suppression - reduce FPs on particles
      Phase 4 (epochs 16+): Add area budget - control overall prediction mass

    Returns:
        total_loss, stats_dict
    """
    B, _, H, W = logits.shape

    # Normalize shapes
    gt = gt_mask.squeeze(1) if gt_mask.dim() == 4 else gt_mask  # (B, H, W)
    core = gt_core.squeeze(1) if gt_core.dim() == 4 else gt_core  # (B, H, W)

    gt_binary = (gt > 0.5).float()
    probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)

    # =========================================================================
    # Phase 1: BCE + Dice (always on from epoch 1)
    # =========================================================================

    # Weighted BCE
    pos_mask = gt_binary.unsqueeze(1)  # (B, 1, H, W)
    neg_mask = (1.0 - gt_binary).unsqueeze(1)  # (B, 1, H, W)
    bce_weights = pos_mask + bce_neg_weight * neg_mask

    bce_loss = F.binary_cross_entropy_with_logits(
        logits, gt_binary.unsqueeze(1),
        weight=bce_weights, reduction='sum'
    ) / (bce_weights.sum() + 1e-6)

    # Dice loss
    dice_loss = compute_dice_loss(probs, gt_binary)

    # =========================================================================
    # Phase 2: Hit-core (warmup starting at warmup_phase1_epochs)
    # =========================================================================
    hit_core_loss = torch.tensor(0.0, device=logits.device)
    lambda_hit_core_eff = 0.0

    if epoch > warmup_phase1_epochs and lambda_hit_core > 0:
        # Linear warmup from phase1_end to phase2_end
        warmup_progress = min(1.0, (epoch - warmup_phase1_epochs) / (warmup_phase2_epochs - warmup_phase1_epochs))
        lambda_hit_core_eff = lambda_hit_core * warmup_progress

        core_mask = (core > 0.5).float()  # (B, H, W)
        core_count = core_mask.sum(dim=(1, 2)) + 1e-6
        core_has_gt = (core_count > 1.0).float()  # (B,)

        # Max probability in core region
        probs_masked = probs * core_mask + (1 - core_mask) * (-1e6)
        max_in_core = probs_masked.max(dim=2)[0].max(dim=1)[0]  # (B,)

        # Penalize if max < 0.5
        hit_core_loss = (F.relu(0.5 - max_in_core) * core_has_gt).mean()

    # =========================================================================
    # Phase 3: Particle anchor suppression (warmup starting at warmup_phase2_epochs)
    # SAFEGUARDS:
    #   1. Exclude particles within safety_margin_px of GT contamination
    #   2. Cap contribution per sample to avoid dominating loss
    # =========================================================================
    particle_anchor_loss = torch.tensor(0.0, device=logits.device)
    lambda_particle_eff = 0.0
    n_safe_particles = 0

    if epoch > warmup_phase2_epochs and lambda_particle_anchor > 0 and good_anchor_mask is not None:
        warmup_progress = min(1.0, (epoch - warmup_phase2_epochs) / (warmup_phase3_epochs - warmup_phase2_epochs))
        lambda_particle_eff = lambda_particle_anchor * warmup_progress

        G = good_anchor_mask.squeeze(1) if good_anchor_mask.dim() == 4 else good_anchor_mask

        # SAFEGUARD 1: Exclude particles near GT contamination (dilate GT by safety margin)
        # This prevents punishing the model for ambiguous boundary regions
        safety_margin_px = 4  # pixels at working resolution (~8-16px at full res)
        if safety_margin_px > 0:
            # Dilate GT to create exclusion zone
            gt_dilated = F.max_pool2d(
                gt_binary.unsqueeze(1),
                kernel_size=2*safety_margin_px + 1,
                stride=1,
                padding=safety_margin_px
            ).squeeze(1)  # (B, H, W)
            # Only keep particles outside the dilated GT zone
            G = G * (1.0 - gt_dilated)
        else:
            G = G * (1.0 - gt_binary)  # Exclude particles in bad regions only

        G = (G > 0.1).float()

        # SAFEGUARD 2: Cap contribution per sample
        # Subsample if too many particle pixels (prevents particle term from dominating)
        max_anchor_pixels_per_sample = 500  # Cap at ~500 pixels per sample
        per_sample_counts = G.sum(dim=(1, 2))  # (B,)

        if G.sum() > 0:
            # Hinge loss: penalize if prob > tau_anchor at particle locations
            probs_at_particles = probs * G
            hinge = F.relu(probs_at_particles - tau_anchor) ** 2

            # Per-sample normalization with cap
            per_sample_loss = hinge.sum(dim=(1, 2)) / (per_sample_counts.clamp(min=1))  # (B,)
            # Weight down samples with many particles (cap effect)
            sample_weights = (per_sample_counts > 0).float() * torch.clamp(
                max_anchor_pixels_per_sample / (per_sample_counts + 1e-6), max=1.0
            )
            particle_anchor_loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-6)
            n_safe_particles = int(G.sum().item())

    # =========================================================================
    # Phase 4: Area budget on unknowns (warmup starting at warmup_phase3_epochs)
    # =========================================================================
    area_loss = torch.tensor(0.0, device=logits.device)
    lambda_area_eff = 0.0

    if epoch > warmup_phase3_epochs and lambda_area > 0:
        warmup_progress = min(1.0, (epoch - warmup_phase3_epochs) / 5)  # 5 epoch warmup for area
        lambda_area_eff = lambda_area * warmup_progress

        # Unknown = not in GT bad, not in particle anchors
        G = good_anchor_mask if good_anchor_mask is not None else torch.zeros_like(gt_binary)
        G = G.squeeze(1) if G.dim() == 4 else G
        G = G * (1.0 - gt_binary)
        G = (G > 0.1).float()

        U = (1.0 - gt_binary) * (1.0 - G)  # Unknown pixels

        if U.sum() > 0:
            mean_prob_unknown = (probs * U).sum() / (U.sum() + 1e-6)
            area_loss = F.relu(mean_prob_unknown - rho_area) ** 2

    # =========================================================================
    # Total loss
    # =========================================================================
    total_loss = (
        lambda_bce * bce_loss +
        lambda_dice * dice_loss +
        lambda_hit_core_eff * hit_core_loss +
        lambda_particle_eff * particle_anchor_loss +
        lambda_area_eff * area_loss
    )

    stats = {
        "loss_bce": float(bce_loss.item()),
        "loss_dice": float(dice_loss.item()),
        "loss_hit_core": float(hit_core_loss.item()),
        "loss_particle_anchor": float(particle_anchor_loss.item()),
        "loss_area": float(area_loss.item()),
        "lambda_hit_core_eff": lambda_hit_core_eff,
        "lambda_particle_eff": lambda_particle_eff,
        "lambda_area_eff": lambda_area_eff,
        "n_safe_particles": n_safe_particles,  # Number of valid particle anchor pixels after safeguards
        "epoch": epoch,
    }

    return total_loss, stats


# =============================================================================
# DICE + FOCAL LOSS: Medical imaging's proven approach for segmentation
# =============================================================================
# Why this works when BCE-based multi-component losses fail:
#
# BCE Problem: 4/5 loss terms benefit from predicting HIGH, only fpcap votes LOW.
#   Model learns: "predict everything as contamination = lower total loss"
#   Result: Collapse after epoch 10-15 to neg_median ~0.99
#
# Dice Solution: Self-balancing loss that CANNOT be gamed:
#   - Predict nothing → Dice = 0 (bad)
#   - Predict everything → Dice ≈ 0.5 if GT is 50% (bad)
#   - Predict exactly GT → Dice = 1.0 (good)
#   The ONLY way to maximize Dice is to precisely match GT shape.
#
# Focal Loss: Automatically downweights easy examples (confident predictions)
#   Once model is confident on a pixel, that pixel stops contributing to training.
#   This prevents "practicing" making everything 1.0.
# =============================================================================

def dice_loss(
    probs: torch.Tensor,  # (B, 1, H, W) probabilities after sigmoid
    gt: torch.Tensor,     # (B, 1, H, W) binary ground truth
    smooth: float = 1.0,  # Laplace smoothing to avoid division by zero
) -> torch.Tensor:
    """
    Dice loss for segmentation. Self-balancing - cannot be gamed by predicting all 1s.

    Dice = 2 * intersection / (prediction + ground_truth)
    Loss = 1 - Dice

    Args:
        probs: Model predictions after sigmoid, shape (B, 1, H, W)
        gt: Binary ground truth mask, shape (B, 1, H, W)
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Scalar dice loss (1 - dice_coefficient)
    """
    # Flatten spatial dimensions
    probs_flat = probs.view(-1)
    gt_flat = gt.view(-1)

    intersection = (probs_flat * gt_flat).sum()
    dice = (2.0 * intersection + smooth) / (probs_flat.sum() + gt_flat.sum() + smooth)

    return 1.0 - dice


def weighted_dice_loss(
    probs: torch.Tensor,
    gt: torch.Tensor,
    weight: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Dice loss with a per-pixel weight map."""
    probs_flat = probs.reshape(-1)
    gt_flat = gt.reshape(-1)
    weight_flat = weight.reshape(-1).to(dtype=probs.dtype)

    intersection = (weight_flat * probs_flat * gt_flat).sum()
    denom = (weight_flat * probs_flat).sum() + (weight_flat * gt_flat).sum()
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice


def focal_loss(
    logits: torch.Tensor,  # (B, 1, H, W) raw logits before sigmoid
    gt: torch.Tensor,      # (B, 1, H, W) binary ground truth
    gamma: float = 2.0,    # Focusing parameter (2.0 is standard)
    alpha: float = 0.25,   # Balance factor for positive class (0.25 standard for imbalanced)
) -> torch.Tensor:
    """
    Focal loss for handling class imbalance without explicit weighting.

    Focal = -alpha * (1-p)^gamma * log(p) for positives
          = -(1-alpha) * p^gamma * log(1-p) for negatives

    When gamma=0, this reduces to standard BCE.
    When gamma=2 (standard), confident predictions (p~1 or p~0) contribute
    almost nothing to the loss, focusing training on hard examples.

    Args:
        logits: Raw model outputs before sigmoid, shape (B, 1, H, W)
        gt: Binary ground truth mask, shape (B, 1, H, W)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weight for positive class (0.25-0.5 typical)

    Returns:
        Scalar focal loss
    """
    # Use numerically stable BCE computation from PyTorch
    # BCE = max(logits, 0) - logits * gt + log(1 + exp(-|logits|))
    # This avoids log(0) and exp overflow issues

    # Clamp logits to prevent extreme values
    logits = logits.clamp(min=-50, max=50)

    # Compute stable BCE per-pixel (without reduction)
    bce = F.binary_cross_entropy_with_logits(logits, gt.float(), reduction='none')

    # Compute probabilities with clamping for focal weight calculation
    probs = torch.sigmoid(logits)
    probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)

    # Compute focal weights
    # For positives (gt=1): weight = (1-p)^gamma  (hard positives have low p)
    # For negatives (gt=0): weight = p^gamma      (hard negatives have high p)
    pt = torch.where(gt == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma

    # Clamp focal weight to prevent extreme values
    focal_weight = focal_weight.clamp(max=10.0)

    # Apply alpha weighting
    alpha_weight = torch.where(gt == 1, alpha, 1 - alpha)

    # Combine: focal_loss = alpha * (1-pt)^gamma * BCE
    focal = alpha_weight * focal_weight * bce

    return focal.mean()


# ---------- GPU-batched Sobel edge detection ----------
# Replaces per-patch scipy.ndimage.sobel loop with a single batched conv2d.
# ~50-100x faster on GPU for typical batch sizes (64 patches).
_sobel_kernel_x: Optional[torch.Tensor] = None
_sobel_kernel_y: Optional[torch.Tensor] = None

def _gpu_sobel_edge_map(x: torch.Tensor) -> torch.Tensor:
    """Compute normalised edge magnitude on GPU using Sobel filters.

    Args:
        x: (B, 1, H, W) image tensor (already on GPU)
    Returns:
        (B, H, W) edge magnitude normalised per-sample to [0, 1]
    """
    global _sobel_kernel_x, _sobel_kernel_y

    if _sobel_kernel_x is None or _sobel_kernel_x.device != x.device:
        # Sobel 3x3 kernels (same as scipy.ndimage.sobel)
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        _sobel_kernel_x = kx
        _sobel_kernel_y = ky

    with torch.no_grad():
        gx = F.conv2d(x, _sobel_kernel_x, padding=1)  # (B,1,H,W)
        gy = F.conv2d(x, _sobel_kernel_y, padding=1)  # (B,1,H,W)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(1)  # (B,H,W)
        # Per-sample normalisation to [0, 1]
        B = edge.shape[0]
        edge_flat = edge.view(B, -1)
        edge_max = edge_flat.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        edge = edge_flat / edge_max
        edge = edge.view(B, x.shape[2], x.shape[3])
    return edge


def compute_dice_focal_loss(
    logits: torch.Tensor,  # (B, 1, H, W) raw logits
    gt_mask: torch.Tensor,  # (B, 1, H, W) binary ground truth
    lambda_dice: float = 2.0,
    lambda_focal: float = 1.0,
    lambda_edge: float = 0.5,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    edge_map: Optional[torch.Tensor] = None,  # (B, H, W) edge magnitude for boundary smoothing
    edge_erosion_px: int = 0,  # Pixels to exclude from loss at patch boundaries
    use_tukey_weight: bool = True,  # Use smooth Tukey window (vs hard erosion)
    label_smoothing: float = 0.0,  # NEW: Label smoothing to prevent overconfidence
    positive_interior_weight: float = 0.0,
    positive_interior_erode_px: int = 0,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined Dice + Focal loss for segmentation.

    This is the medical imaging standard for lesion/tumor detection.
    Self-balancing and cannot be gamed by predicting everything as positive.

    NEW: Supports edge erosion to prevent patch boundary artifacts.
    Patches have no context beyond their boundaries, so edge pixels can
    create spurious patterns. By downweighting/excluding edge pixels from
    loss, we prevent the model from learning 'edge of context' artifacts.

    NEW: Supports label smoothing to prevent probability saturation (1.0/0.0).
    With label_smoothing=0.05: targets become 0.05 and 0.95 instead of 0 and 1.
    This prevents the model from becoming overconfident and overfitting.

    Args:
        logits: Raw model outputs before sigmoid
        gt_mask: Binary ground truth
        lambda_dice: Weight for Dice loss (default 2.0)
        lambda_focal: Weight for Focal loss (default 1.0)
        lambda_edge: Weight for edge smoothing loss (default 0.5)
        focal_gamma: Focal loss gamma parameter (default 2.0)
        focal_alpha: Focal loss alpha parameter (default 0.25)
        edge_map: Optional edge magnitude map for boundary penalty
        edge_erosion_px: Pixels to exclude from loss at patch boundaries (default 0)
        use_tukey_weight: If True, use smooth Tukey window; else hard erosion
        label_smoothing: Smooth hard labels (0→ε, 1→1-ε) to prevent overconfidence

    Returns:
        total_loss: Scalar combined loss
        stats: Dictionary with individual loss components
    """
    # Ensure correct shapes
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    gt = gt_mask.float()

    # Apply label smoothing: 0 → ε, 1 → 1-ε
    # This prevents the model from becoming overconfident
    if label_smoothing > 0:
        gt = gt * (1.0 - 2 * label_smoothing) + label_smoothing

    probs = torch.sigmoid(logits)

    # Create per-pixel loss weights: edge taper for patch boundaries plus an
    # optional boost for eroded GT interiors of broad contamination regions.
    loss_weight_mask = None
    if edge_erosion_px > 0:
        B, C, H, W = logits.shape
        if use_tukey_weight:
            # Smooth Tukey window - gradual taper at edges
            loss_weight_mask = get_tukey_edge_mask(H, W, edge_erosion_px, logits.device)
            loss_weight_mask = loss_weight_mask.view(1, 1, H, W).expand(B, C, H, W)
        else:
            # Hard erosion mask - binary center region
            y = torch.arange(H, device=logits.device)
            x = torch.arange(W, device=logits.device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            # Interior = 1, edges = 0
            interior = ((yy >= edge_erosion_px) & (yy < H - edge_erosion_px) &
                       (xx >= edge_erosion_px) & (xx < W - edge_erosion_px)).float()
            loss_weight_mask = interior.view(1, 1, H, W).expand(B, C, H, W)

    interior_weight_mask = None
    positive_interior_weight = max(0.0, float(positive_interior_weight))
    positive_interior_erode_px = max(0, int(positive_interior_erode_px))
    if positive_interior_weight > 0.0:
        gt_hard_for_weight = (gt_mask.float() > 0.5).float()
        if gt_hard_for_weight.ndim == 3:
            gt_hard_for_weight = gt_hard_for_weight.unsqueeze(1)
        if positive_interior_erode_px > 0:
            kernel = 2 * positive_interior_erode_px + 1
            interior_weight_mask = -F.max_pool2d(
                -gt_hard_for_weight,
                kernel_size=kernel,
                stride=1,
                padding=positive_interior_erode_px,
            )
            interior_weight_mask = (interior_weight_mask > 0.5).float()
        else:
            interior_weight_mask = gt_hard_for_weight

        boost = 1.0 + positive_interior_weight * interior_weight_mask
        if loss_weight_mask is None:
            loss_weight_mask = boost
        else:
            loss_weight_mask = loss_weight_mask * boost

    # Dice loss - self-balancing segmentation loss
    if loss_weight_mask is not None:
        loss_dice = weighted_dice_loss(probs, gt, loss_weight_mask, smooth=1.0)
    else:
        loss_dice = dice_loss(probs, gt, smooth=1.0)

    # Focal loss - handles class imbalance, focuses on hard examples
    if loss_weight_mask is not None:
        # Weighted focal: downweight edge pixels and optionally upweight
        # hard-positive interiors.
        with torch.no_grad():
            probs_clamped = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
            pt = torch.where(gt == 1, probs_clamped, 1 - probs_clamped)
            focal_weight = (1 - pt) ** focal_gamma
        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
        alpha_weight = torch.where(gt == 1, focal_alpha, 1 - focal_alpha)
        focal_per_pixel = alpha_weight * focal_weight * bce * loss_weight_mask
        loss_focal = focal_per_pixel.sum() / (loss_weight_mask.sum() + 1e-6)
    else:
        loss_focal = focal_loss(logits, gt, gamma=focal_gamma, alpha=focal_alpha)

    # Edge loss - penalize predictions in regions with strong edges (boundary smoothing)
    loss_edge = torch.tensor(0.0, device=logits.device)
    if lambda_edge > 0 and edge_map is not None:
        # Penalize high predictions in high-edge regions outside GT
        # This encourages smooth boundaries
        edge_map = edge_map.unsqueeze(1) if edge_map.ndim == 3 else edge_map
        neg_mask = (gt < 0.5).float()
        edge_penalty = probs * edge_map.detach() * neg_mask
        loss_edge = edge_penalty.mean()

    # Combine losses
    total_loss = lambda_dice * loss_dice + lambda_focal * loss_focal + lambda_edge * loss_edge

    # Compute metrics for logging (use UN-smoothed GT for metrics so dice reflects true performance)
    with torch.no_grad():
        gt_hard = gt_mask  # Original 0/1 labels (before any smoothing)

        # Dice at multiple thresholds (0.5 for legacy, 0.15/0.20 for operational)
        dice_at = {}
        for thr in (0.15, 0.20, 0.50):
            pred_bin = (probs > thr).float()
            inter = (pred_bin * gt_hard).sum()
            dice_at[thr] = float(((2.0 * inter + 1.0) / (pred_bin.sum() + gt_hard.sum() + 1.0)).cpu().item())

        # Compute clean patch loss equivalent (for comparison with old runs)
        B = gt_hard.shape[0]
        gt_sum_per_patch = gt_hard.view(B, -1).sum(dim=1)
        clean_patch_mask = (gt_sum_per_patch < 0.5).float()
        n_clean_patches = clean_patch_mask.sum()
        if n_clean_patches > 0:
            mean_pred_per_patch = probs.view(B, -1).mean(dim=1)
            clean_patch_loss = ((mean_pred_per_patch * clean_patch_mask) ** 2).sum() / (n_clean_patches + 1e-6)
        else:
            clean_patch_loss = torch.tensor(0.0)

    stats = {
        "loss_dice": float(loss_dice.detach().cpu().item()),
        "loss_focal": float(loss_focal.detach().cpu().item()),
        "loss_edge": float(loss_edge.detach().cpu().item()),
        "positive_interior_weight": float(positive_interior_weight),
        "positive_interior_fraction": (
            float(interior_weight_mask.detach().mean().cpu().item())
            if interior_weight_mask is not None
            else 0.0
        ),
        "dice_coefficient": dice_at[0.50],  # Legacy (threshold=0.5)
        "dice_at_0.15": dice_at[0.15],
        "dice_at_0.20": dice_at[0.20],
        "loss_clean_patch": float(clean_patch_loss.cpu().item()),
        "n_clean_patches": int(n_clean_patches.cpu().item()),
    }

    return total_loss, stats


def compute_fast_region_loss(
    logits: torch.Tensor,  # (B, 1, H, W)
    gt_mask: torch.Tensor,  # (B, 1, H, W) binary
    gt_core: torch.Tensor,  # (B, 1, H, W) or (B, H, W) binary, precomputed eroded GT
    lambda_hit_core: float = 1.0,
    lambda_outside_mass: float = 1.0,
    lambda_bce: float = 1.0,
    bce_neg_weight: float = 1.0,  # Weight for negative class in BCE (increase to suppress false positives)
    lambda_particle_neg: float = 0.0,  # Particle-likeness negative mining weight
    particle_like_mask: Optional[torch.Tensor] = None,  # (B, H, W) binary, particle-like pixels
    lambda_particle_neg_wide: float = 0.0,  # Wide ring particle-neg weight
    particle_like_mask_wide: Optional[torch.Tensor] = None,  # (B, H, W) binary, wider ring particle-like pixels
    outside_dilate_px: int = 0,  # Dilate GT before computing outside-mass (avoids punishing slight misalignment)
    lambda_fpcap: float = 0.0,  # Differentiable FP control term weight
    fpcap_target_mass: float = 0.03,  # Target FP mass (start loose, tighten later)
    clean_mask: Optional[torch.Tensor] = None,  # (B, H, W) binary, clean regions: (gt==0) AND (dilate(gt, r=3)==0)
    lambda_edge: float = 0.0,  # Edge-conditioned suppression on negatives (penalize p*E on GT==0)
    edge_map: Optional[torch.Tensor] = None,  # (B, H, W) edge map (Sobel magnitude, normalized to [0,1], detached)
    # Actual particle picks (positive labels) - more accurate than heuristic particle-likeness
    lambda_particle_pos: float = 0.0,  # Weight for actual particle positive label loss
    actual_particle_mask: Optional[torch.Tensor] = None,  # (B, H, W) binary, actual particle boxes from picks
    # Soft hit_core option - use MEAN instead of MAX to avoid pushing confidence too high
    use_soft_hit_core: bool = False,  # If True, use mean(probs in core) instead of max
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss = BCE + λ_hit * hit_core + λ_out * outside_mass + λ_fpcap * L_fpcap + λ_edge * L_edge.
    BCE keeps probabilities calibrated; region terms normalized by pixel counts so BCE is not drowned.

    A. L_fpcap: Stronger differentiable FP control term
       - Computed on GT-negative pixels (or clean_mask if available)
       - fp_mass = mean(p[neg])
       - L_fpcap = relu(fp_mass - cap_mass)^2 + 0.1 * logit_penalty (helps when saturated)

    B. L_edge: Edge-conditioned suppression on negatives (ring glow suppression)
       - L_edge = mean(p * normalize(E)) over GT-negative pixels
       - Interpretation: "if it looks like a sharp particle boundary, default to predicting 0"
    """
    B, _, H, W = logits.shape
    gt = gt_mask.squeeze(1) if gt_mask.dim() == 4 else gt_mask  # (B, H, W)
    core = gt_core.squeeze(1) if gt_core.dim() == 4 else gt_core  # (B, H, W)
    gt_for_bce = gt_mask if gt_mask.shape == logits.shape else gt.unsqueeze(1)  # (B, 1, H, W)

    # BCE with negative class weighting to suppress false positives
    # Weight negatives more heavily when bce_neg_weight > 1.0
    if bce_neg_weight != 1.0:
        pos_mask = (gt_for_bce > 0.5).float()
        neg_mask = (gt_for_bce <= 0.5).float()
        bce_weights = pos_mask + bce_neg_weight * neg_mask  # (B, 1, H, W)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, gt_for_bce.to(logits.dtype),
            weight=bce_weights, reduction='sum'
        ) / (bce_weights.sum() + 1e-6)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, gt_for_bce.to(logits.dtype), reduction="mean")

    probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)

    core_mask = (core > 0.5).float()  # (B, H, W)
    core_count = core_mask.sum(dim=(1, 2)) + 1e-6  # (B,) per-sample core pixels
    core_has_gt = (core_count > 1.0).float()  # (B,)

    if use_soft_hit_core:
        # SOFT HIT_CORE: Use mean probability in core instead of max
        # This is gentler - doesn't push confidence as hard, better calibration
        # BCE loss format: -log(mean_prob) where we want mean_prob to be high
        mean_in_core = (probs * core_mask).sum(dim=(1, 2)) / core_count  # (B,)
        hit_per_sample = -torch.log(torch.clamp(mean_in_core, min=1e-6) * core_has_gt + (1 - core_has_gt))
    else:
        # HARD HIT_CORE: Use max probability in core (original behavior)
        # This is more aggressive - model only needs ONE high-prob pixel to satisfy
        probs_masked = probs * core_mask + (1 - core_mask) * (-1e6)
        max_in_core = probs_masked.max(dim=2)[0].max(dim=1)[0]  # (B,)
        hit_per_sample = -torch.log(torch.clamp(max_in_core, min=1e-6) * core_has_gt + (1 - core_has_gt))

    # FIXED: Don't divide by core_count - this made small cores dominate loss and caused model to predict high probs everywhere
    # Instead, normalize by number of patches with cores to keep loss scale reasonable
    hit_core_loss = hit_per_sample.mean()  # Mean over batch (patches with cores get equal weight)

    gt_binary = (gt > 0.5).float()  # (B, H, W)

    # Dilate GT before computing outside-mass to avoid punishing slight misalignment
    if outside_dilate_px > 0:
        from scipy.ndimage import binary_dilation
        gt_binary_np = gt_binary.detach().cpu().numpy()
        gt_dilated_list = []
        for b in range(gt_binary_np.shape[0]):
            gt_dilated = binary_dilation(gt_binary_np[b] > 0.5, iterations=outside_dilate_px)
            gt_dilated_list.append(torch.from_numpy(gt_dilated.astype(np.float32)))
        gt_binary = torch.stack(gt_dilated_list).to(gt_binary.device)  # (B, H, W)

    outside_gt = 1.0 - gt_binary
    outside_count = outside_gt.sum(dim=(1, 2)) + 1e-6  # (B,)
    prob_outside = (probs * outside_gt).sum(dim=(1, 2)) / outside_count  # mean prob on outside pixels

    # STRONGER: Use BCE on clean regions (target=0) to penalize high probabilities more aggressively
    # BCE with logits is more sensitive to high probabilities than mean prob alone
    outside_mask = outside_gt.unsqueeze(1)  # (B, 1, H, W)
    target_clean = torch.zeros_like(logits)  # Target is 0 (clean) for all pixels
    # Compute BCE only on clean pixels (weighted by outside_mask)
    outside_bce = F.binary_cross_entropy_with_logits(
        logits, target_clean,
        weight=outside_mask, reduction='sum'
    ) / (outside_mask.sum() + 1e-6)  # Normalize by number of clean pixels

    # Combine: BCE (stronger penalty for high probs) + mean prob (stability)
    outside_mass_loss = 0.7 * outside_bce + 0.3 * prob_outside.mean()

    # Particle-likeness negative mining: push logits down on particle-like pixels in clean regions
    # Key insight: Particles exist in BOTH clean and bad areas, but:
    # - Clean area + particles = normal (should predict 0, not contamination)
    # - Bad area + particles = contamination area (should predict 1, but the contamination is the area, not the particles)
    # This term teaches the model that particle-like features in clean regions are NOT contamination.
    particle_neg_loss = torch.tensor(0.0, device=logits.device)
    if lambda_particle_neg > 0 and particle_like_mask is not None:
        particle_mask = particle_like_mask.unsqueeze(1) if particle_like_mask.dim() == 3 else particle_like_mask  # (B, 1, H, W)
        if particle_mask.sum() > 0:
            target_particle = torch.zeros_like(logits)  # Target is 0 (not contamination)
            particle_neg_loss = F.binary_cross_entropy_with_logits(
                logits, target_particle,
                weight=particle_mask, reduction='sum'
            ) / (particle_mask.sum() + 1e-6)

    # Wide ring particle-neg (lower weight, wider margin)
    particle_neg_wide_loss = torch.tensor(0.0, device=logits.device)
    if lambda_particle_neg_wide > 0 and particle_like_mask_wide is not None:
        particle_mask_wide = particle_like_mask_wide.unsqueeze(1) if particle_like_mask_wide.dim() == 3 else particle_like_mask_wide  # (B, 1, H, W)
        if particle_mask_wide.sum() > 0:
            target_particle = torch.zeros_like(logits)  # Target is 0 (not contamination)
            particle_neg_wide_loss = F.binary_cross_entropy_with_logits(
                logits, target_particle,
                weight=particle_mask_wide, reduction='sum'
            ) / (particle_mask_wide.sum() + 1e-6)

    # A. Differentiable FP control term (L_fpcap) - stronger formulation
    fpcap_loss = torch.tensor(0.0, device=logits.device)
    if lambda_fpcap > 0:
        # Use GT-negative pixels directly (more aggressive than clean_mask)
        gt_neg = (gt <= 0.5).float()  # (B, H, W)
        if clean_mask is not None:
            # Prefer clean_mask if available (avoids boundary ambiguity)
            clean = clean_mask if clean_mask.dim() == 3 else clean_mask.unsqueeze(1)
            if clean.dim() == 4:
                clean = clean.squeeze(1)  # (B, H, W)
            gt_neg = clean  # Use clean_mask instead

        neg_count = gt_neg.sum(dim=(1, 2)) + 1e-6  # (B,)
        fp_mass = (probs * gt_neg).sum(dim=(1, 2)) / neg_count  # (B,) mean prob on negatives
        fp_mass_mean = fp_mass.mean()  # Scalar

        # Stronger penalty: use squared penalty and also penalize logits directly when saturated
        # This gives gradients even when probs are near 1.0
        prob_penalty = F.relu(fp_mass_mean - fpcap_target_mass) ** 2

        # Also penalize logits directly on high-prob negatives (helps when saturated)
        logit_penalty = torch.tensor(0.0, device=logits.device)
        if fp_mass_mean > fpcap_target_mass:
            # On negatives with high prob, push logits down
            neg_mask_expanded = gt_neg.unsqueeze(1)  # (B, 1, H, W)
            high_prob_neg = (probs.unsqueeze(1) > 0.5) & (neg_mask_expanded > 0.5)
            if high_prob_neg.any():
                logit_penalty = (F.relu(logits[high_prob_neg]) ** 2).mean()

        fpcap_loss = prob_penalty + 0.1 * logit_penalty

    # B. Edge-conditioned suppression on negatives (ring glow suppression)
    # Penalize probability proportional to edge strength on GT-negative pixels
    # Interpretation: "if it looks like a sharp particle boundary, default to predicting 0"
    edge_loss = torch.tensor(0.0, device=logits.device)
    if lambda_edge > 0 and edge_map is not None:
        gt_neg = (gt <= 0.5).float()  # (B, H, W)
        E = edge_map if edge_map.dim() == 3 else edge_map  # (B, H, W)
        if E.dim() == 4:
            E = E.squeeze(1)  # (B, H, W)

        # Normalize edge map to [0, 1] if not already (robust normalization)
        E_min = E.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, 1, 1)
        E_max = E.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, 1, 1)
        E_range = E_max - E_min + 1e-6
        E_norm = (E - E_min) / E_range  # (B, H, W) normalized to [0, 1]

        # L_edge_neg = mean(p * E_norm) over GT-negative pixels
        neg_edge = gt_neg * E_norm  # (B, H, W)
        neg_edge_count = neg_edge.sum(dim=(1, 2)) + 1e-6  # (B,)
        edge_loss = ((probs * neg_edge).sum(dim=(1, 2)) / neg_edge_count).mean()

    # C. Actual particle positive label loss (L_particle_pos)
    # If pixel is in actual particle box AND NOT in GT bad region, push probability to 0
    # This is more accurate than heuristic particle-likeness because it uses real particle picks
    particle_pos_loss = torch.tensor(0.0, device=logits.device)
    if lambda_particle_pos > 0 and actual_particle_mask is not None:
        particle_mask = actual_particle_mask  # (B, H, W)
        if particle_mask.dim() == 4:
            particle_mask = particle_mask.squeeze(1)  # (B, H, W)

        # Good particles = in particle box AND NOT in GT bad region
        gt_binary = (gt > 0.5).float()  # (B, H, W)
        good_particle_mask = (particle_mask > 0.5) & (gt_binary < 0.5)  # (B, H, W)

        if good_particle_mask.any():
            # BCE loss with target=0 (not contamination) on good particle pixels
            good_particle_expanded = good_particle_mask.float().unsqueeze(1)  # (B, 1, H, W)
            target_clean = torch.zeros_like(logits)  # Target is 0 (clean/not contamination)
            particle_pos_loss = F.binary_cross_entropy_with_logits(
                logits, target_clean,
                weight=good_particle_expanded, reduction='sum'
            ) / (good_particle_expanded.sum() + 1e-6)

    # D. CLEAN PATCH REGULARIZATION (CRITICAL for preventing model collapse)
    # For patches that have NO GT contamination at all, the entire prediction should be low.
    # This provides a DIRECT signal to predict negative on clean regions, which counters
    # the tendency of hit_core (max-based) to push predictions high everywhere.
    #
    # Without this: hit_core uses max-pooling (satisfied by any high pixel),
    #               outside_mass uses mean (can be gamed by spreading predictions)
    #               → model learns to predict high everywhere
    # With this:    clean patches have explicit penalty for any high predictions
    #               → model learns to differentiate clean vs contaminated
    clean_patch_loss = torch.tensor(0.0, device=logits.device)
    # Identify clean patches: patches where gt.sum() == 0 (no GT contamination at all)
    gt_sum_per_patch = gt.view(B, -1).sum(dim=1)  # (B,) total GT mass per patch
    clean_patch_mask = (gt_sum_per_patch < 0.5).float()  # (B,) 1 if patch is entirely clean
    n_clean_patches = clean_patch_mask.sum()

    if n_clean_patches > 0:
        # For clean patches, mean prediction should be close to 0
        mean_pred_per_patch = probs.view(B, -1).mean(dim=1)  # (B,) mean prediction per patch
        # Weight by clean_patch_mask: only penalize clean patches
        # Use squared penalty to be aggressive about suppressing predictions
        clean_patch_loss = ((mean_pred_per_patch * clean_patch_mask) ** 2).sum() / (n_clean_patches + 1e-6)

    # Lambda for clean patch loss - hardcoded high weight because this is critical
    # to prevent model collapse. The clean_patch_loss is already squared so weight
    # doesn't need to be huge.
    lambda_clean_patch = 2.0  # Strong weight to counter hit_core's max-pooling bias

    total_loss = (
        lambda_bce * bce_loss +
        lambda_hit_core * hit_core_loss +
        lambda_outside_mass * outside_mass_loss +
        lambda_particle_neg * particle_neg_loss +
        lambda_particle_neg_wide * particle_neg_wide_loss +
        lambda_fpcap * fpcap_loss +
        lambda_edge * edge_loss +
        lambda_particle_pos * particle_pos_loss +
        lambda_clean_patch * clean_patch_loss
    )
    stats = {
        "loss_bce": float(bce_loss.detach().cpu().item()),
        "loss_hit_core": float(hit_core_loss.detach().cpu().item()),
        "loss_outside_mass": float(outside_mass_loss.detach().cpu().item()),
        "loss_particle_neg": float(particle_neg_loss.detach().cpu().item()),
        "loss_particle_neg_wide": float(particle_neg_wide_loss.detach().cpu().item()),
        "loss_fpcap": float(fpcap_loss.detach().cpu().item()),
        "loss_edge": float(edge_loss.detach().cpu().item()),
        "loss_particle_pos": float(particle_pos_loss.detach().cpu().item()),
        "loss_clean_patch": float(clean_patch_loss.detach().cpu().item()),
        "n_clean_patches": int(n_clean_patches.detach().cpu().item()),
    }
    return total_loss, stats


def compute_pu_loss(
    logits: torch.Tensor,  # (B, 1, H, W)
    gt_mask: torch.Tensor,  # (B, 1, H, W) or (B, H, W), binary: 1=bad, 0=not-bad
    gt_core: torch.Tensor,  # (B, H, W) eroded GT core (high-confidence bad)
    good_anchor_mask: torch.Tensor,  # (B, H, W) particle pick locations (small Gaussian/disk)
    # Loss weights (can be 0 to disable)
    lambda_bad: float = 1.0,
    lambda_hit_core: float = 2.0,
    lambda_good: float = 0.2,
    lambda_margin: float = 0.3,
    lambda_area: float = 0.5,
    # NEW: Entropy regularization and asymmetric particle penalty
    lambda_entropy: float = 0.0,  # Entropy regularization (encourage mid-range predictions)
    lambda_particle_asym: float = 0.0,  # Asymmetric penalty: FP on particles >> FN
    # Hyperparameters
    tau_good: float = 0.15,  # Hinge threshold for good anchors
    margin_m: float = 2.5,  # Logit margin between bad cores and good anchors
    rho_area: float = 0.05,  # Target expected bad fraction in unknown
    focal_gamma: float = 2.0,  # Focal loss gamma for L_bad
    bce_neg_weight: float = 3.0,  # Weight for negative class in BCE
    entropy_target_range: Tuple[float, float] = (0.1, 0.3),  # Target probability range
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PU-style loss for contamination detection with conservative GT and particle anchors.

    Pixel sets:
      B = labeled BAD pixels from gt_mask
      Bcore = eroded subset of B (gt_core) for high-confidence bad
      G = GOOD-ANCHOR pixels from particle picks (small support, excludes overlap with B)
      U = UNKNOWN = all pixels not in B and not in G

    Loss components:
      L_bad: BCE/focal on B (higher weight on Bcore via hit-core)
      L_good: Hinge loss on G: mean(relu(p - tau_good)^2)
      L_margin: Logit margin between Bcore and G: mean(softplus(m - (l_bad - l_good)))
      L_area: Unknown-only area budget: relu(mean(p[U]) - rho)^2

    Returns:
        total_loss, stats_dict
    """
    B_batch, _, H, W = logits.shape

    # Normalize shapes to (B, H, W)
    gt = gt_mask.squeeze(1) if gt_mask.dim() == 4 else gt_mask
    core = gt_core.squeeze(1) if gt_core.dim() == 4 else gt_core
    G = good_anchor_mask.squeeze(1) if good_anchor_mask.dim() == 4 else good_anchor_mask

    # Ensure G doesn't overlap with B (particle picks in bad regions are ambiguous)
    gt_binary = (gt > 0.5).float()
    G = G * (1.0 - gt_binary)  # Zero out G where GT=bad
    G = (G > 0.1).float()  # Re-binarize after multiplication (Gaussian values)

    # Pixel sets
    B = gt_binary  # (B, H, W) bad pixels
    Bcore = (core > 0.5).float()  # (B, H, W) bad core pixels
    U = (1.0 - B) * (1.0 - G)  # (B, H, W) unknown pixels (not in B, not in G)

    probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)
    logits_squeezed = logits.squeeze(1)  # (B, H, W)

    # =========================================================================
    # L_bad: BCE (with optional focal-style) on bad pixels
    # =========================================================================
    gt_for_bce = gt.unsqueeze(1)  # (B, 1, H, W)

    # Standard BCE with negative weighting
    pos_mask = B.unsqueeze(1)  # (B, 1, H, W)
    neg_mask = (1.0 - B).unsqueeze(1)  # (B, 1, H, W)
    bce_weights = pos_mask + bce_neg_weight * neg_mask

    bce_loss = F.binary_cross_entropy_with_logits(
        logits, gt_for_bce.to(logits.dtype),
        weight=bce_weights, reduction='sum'
    ) / (bce_weights.sum() + 1e-6)

    # Hit-core: ensure at least one high-prob pixel per patch in core
    core_mask = Bcore
    core_count = core_mask.sum(dim=(1, 2)) + 1e-6
    core_has_gt = (core_count > 1.0).float()
    probs_masked = probs * core_mask + (1 - core_mask) * (-1e6)
    max_in_core = probs_masked.max(dim=2)[0].max(dim=1)[0]
    hit_per_sample = -torch.log(torch.clamp(max_in_core, min=1e-6) * core_has_gt + (1 - core_has_gt))
    hit_core_loss = hit_per_sample.mean()

    L_bad = bce_loss + lambda_hit_core * hit_core_loss

    # =========================================================================
    # L_good: Hinge loss on good anchors - only penalize if p > tau_good
    # mean(relu(p(x) - tau_good)^2) for x in G
    # =========================================================================
    L_good = torch.tensor(0.0, device=logits.device)
    if lambda_good > 0 and G.sum() > 0:
        # Only penalize predictions above tau_good
        hinge_violation = F.relu(probs - tau_good)  # (B, H, W)
        # Apply only on G pixels
        hinge_on_G = (hinge_violation ** 2) * G  # (B, H, W)
        L_good = hinge_on_G.sum() / (G.sum() + 1e-6)

    # =========================================================================
    # L_margin: Logit margin loss between bad cores and good anchors
    # mean(softplus(m - (l(b) - l(g)))) for pairs (b, g) from Bcore, G
    # =========================================================================
    L_margin = torch.tensor(0.0, device=logits.device)
    if lambda_margin > 0:
        # Get logits at Bcore and G locations
        bcore_locations = (Bcore > 0.5)  # (B, H, W) bool
        g_locations = (G > 0.5)  # (B, H, W) bool

        # Need at least one of each to compute margin
        bcore_count = bcore_locations.sum()
        g_count = g_locations.sum()

        if bcore_count > 0 and g_count > 0:
            # Sample logits from Bcore and G
            logits_bcore = logits_squeezed[bcore_locations]  # (N_bcore,)
            logits_g = logits_squeezed[g_locations]  # (N_g,)

            # Compute all pairwise margin violations (or sample if too large)
            n_bcore = logits_bcore.shape[0]
            n_g = logits_g.shape[0]

            # For efficiency, sample up to 1000 pairs
            max_pairs = 1000
            if n_bcore * n_g > max_pairs:
                # Random sampling of pairs
                idx_b = torch.randint(0, n_bcore, (max_pairs,), device=logits.device)
                idx_g = torch.randint(0, n_g, (max_pairs,), device=logits.device)
                l_b = logits_bcore[idx_b]
                l_g = logits_g[idx_g]
            else:
                # All pairs (broadcast)
                l_b = logits_bcore.unsqueeze(1).expand(-1, n_g).flatten()  # (N_bcore * N_g,)
                l_g = logits_g.unsqueeze(0).expand(n_bcore, -1).flatten()  # (N_bcore * N_g,)

            # Margin violation: want l(b) - l(g) >= m, so penalize softplus(m - (l(b) - l(g)))
            margin_violation = F.softplus(margin_m - (l_b - l_g))
            L_margin = margin_violation.mean()

    # =========================================================================
    # L_area: Unknown-only area budget - prevent overmasking unknown regions
    # relu(mean(p[U]) - rho)^2
    # =========================================================================
    L_area = torch.tensor(0.0, device=logits.device)
    if lambda_area > 0 and U.sum() > 0:
        # Mean predicted bad probability over UNKNOWN pixels only
        mu_U = (probs * U).sum() / (U.sum() + 1e-6)
        # Penalize if mean exceeds target fraction
        L_area = F.relu(mu_U - rho_area) ** 2

    # =========================================================================
    # L_entropy: Entropy regularization - discourage overconfident predictions
    # Penalize predictions near 0 or 1 on unknown pixels; encourage 0.1-0.3 range
    # This prevents the model from being confidently wrong everywhere
    # =========================================================================
    L_entropy = torch.tensor(0.0, device=logits.device)
    if lambda_entropy > 0 and U.sum() > 0:
        # Compute entropy: -p*log(p) - (1-p)*log(1-p)
        # High entropy = uncertain (good for unknown regions)
        eps = 1e-6
        p_clamped = torch.clamp(probs, eps, 1 - eps)
        entropy = -p_clamped * torch.log(p_clamped) - (1 - p_clamped) * torch.log(1 - p_clamped)

        # We want to ENCOURAGE entropy on unknown pixels (penalize low entropy)
        # But also penalize predictions outside target range (0.1-0.3)
        # Target: predictions should be low but uncertain, not overconfidently low
        target_low, target_high = entropy_target_range

        # Penalize predictions below target_low (overconfident that it's clean)
        # Penalize predictions above target_high (overconfident that it's bad)
        too_low_penalty = F.relu(target_low - probs) ** 2  # Push up if < 0.1
        too_high_penalty = F.relu(probs - target_high) ** 2  # Push down if > 0.3

        # Apply only on unknown pixels (not B, not G)
        range_penalty = (too_low_penalty + 2.0 * too_high_penalty) * U  # 2x penalty for being too high
        L_entropy = range_penalty.sum() / (U.sum() + 1e-6)

    # =========================================================================
    # L_particle_asym: Asymmetric penalty - FP on particles costs more than FN
    # Push predictions HARD to 0 on particle regions (3x weight)
    # =========================================================================
    L_particle_asym = torch.tensor(0.0, device=logits.device)
    if lambda_particle_asym > 0 and G.sum() > 0:
        # Strong BCE loss with target=0 on particle regions
        # This is like L_good but with BCE instead of hinge (no tolerance threshold)
        G_expanded = G.unsqueeze(1)  # (B, 1, H, W)
        target_zero = torch.zeros_like(logits)
        particle_bce = F.binary_cross_entropy_with_logits(
            logits, target_zero,
            weight=G_expanded, reduction='sum'
        ) / (G.sum() + 1e-6)
        L_particle_asym = particle_bce

    # =========================================================================
    # Total loss
    # =========================================================================
    total_loss = (
        lambda_bad * L_bad +
        lambda_good * L_good +
        lambda_margin * L_margin +
        lambda_area * L_area +
        lambda_entropy * L_entropy +
        lambda_particle_asym * L_particle_asym
    )

    stats = {
        "loss_bad": float(L_bad.detach().cpu().item()) if isinstance(L_bad, torch.Tensor) else L_bad,
        "loss_good": float(L_good.detach().cpu().item()) if isinstance(L_good, torch.Tensor) else L_good,
        "loss_margin": float(L_margin.detach().cpu().item()) if isinstance(L_margin, torch.Tensor) else L_margin,
        "loss_area": float(L_area.detach().cpu().item()) if isinstance(L_area, torch.Tensor) else L_area,
        "loss_entropy": float(L_entropy.detach().cpu().item()) if isinstance(L_entropy, torch.Tensor) else L_entropy,
        "loss_particle_asym": float(L_particle_asym.detach().cpu().item()) if isinstance(L_particle_asym, torch.Tensor) else L_particle_asym,
        "loss_bce": float(bce_loss.detach().cpu().item()),
        "loss_hit_core": float(hit_core_loss.detach().cpu().item()),
        "mu_U": float((probs * U).sum() / (U.sum() + 1e-6)) if U.sum() > 0 else 0.0,
        "G_coverage": float(G.sum() / (B_batch * H * W)),
    }
    return total_loss, stats


def compute_patch_type_loss(
    logits: torch.Tensor,  # (B, 1, H, W)
    gt_mask: torch.Tensor,  # (B, 1, H, W) or (B, H, W)
    patch_types: torch.Tensor,  # (B,) int: 0=random, 1=bad-centered, 2=particle-centered
    encoder_features: Optional[torch.Tensor] = None,  # For patch-level classification head
    # Loss weights
    lambda_patch_classify: float = 0.5,  # Weight for patch-level classification
    lambda_particle_region: float = 1.0,  # Weight for particle-centered region supervision
    lambda_bad_region: float = 1.0,  # Weight for bad-centered region supervision
    # Hyperparameters
    particle_region_radius_frac: float = 0.4,  # Fraction of patch size for particle region (centered)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Patch-type aware loss: different supervision based on patch sampling type.

    For particle-centered patches (type=2):
      - Apply strong negative supervision to central region (target=0)
      - The whole central region should be "good" since patch is centered on a particle

    For bad-centered patches (type=1):
      - Apply strong positive supervision, higher weight on bad regions
      - These patches are known to contain contamination

    For random patches (type=0):
      - Standard pixel-wise supervision using GT mask

    Optional: Patch-level classification head (if encoder_features provided)
      - Classify entire patch as good (type=2) vs bad (type=1)
      - Helps encoder learn discriminative features

    Returns:
        total_loss, stats_dict
    """
    B, _, H, W = logits.shape
    device = logits.device

    gt = gt_mask.squeeze(1) if gt_mask.dim() == 4 else gt_mask  # (B, H, W)
    probs = torch.sigmoid(logits).squeeze(1)  # (B, H, W)

    total_loss = torch.tensor(0.0, device=device)
    stats = {}

    # =========================================================================
    # Particle-centered patches: full-region negative supervision
    # =========================================================================
    particle_mask = (patch_types == 2)  # (B,) bool
    n_particle = particle_mask.sum().item()

    L_particle_region = torch.tensor(0.0, device=device)
    if n_particle > 0 and lambda_particle_region > 0:
        # Create circular mask for central region of particle-centered patches
        center_y, center_x = H // 2, W // 2
        radius = int(particle_region_radius_frac * min(H, W))

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        dist_from_center = torch.sqrt((yy - center_y).float() ** 2 + (xx - center_x).float() ** 2)
        central_region = (dist_from_center <= radius).float()  # (H, W)

        # For particle-centered patches, target is 0 (good) in central region
        # BCE loss with target=0 on central region
        probs_particle = probs[particle_mask]  # (N_particle, H, W)
        central_region_batch = central_region.unsqueeze(0).expand(n_particle, -1, -1)  # (N_particle, H, W)

        # BCE loss: -log(1 - p) for target=0, weighted by central_region
        # Clamp probs to avoid log(0) = -inf
        probs_clamped = probs_particle.clamp(min=1e-6, max=1-1e-6)
        neg_log_good = -torch.log(1.0 - probs_clamped)  # (N_particle, H, W)
        L_particle_region = (neg_log_good * central_region_batch).sum() / (central_region_batch.sum() + 1e-6)

    stats["loss_particle_region"] = float(L_particle_region.item()) if isinstance(L_particle_region, torch.Tensor) else 0.0
    stats["n_particle_patches"] = n_particle

    # =========================================================================
    # Bad-centered patches: stronger positive supervision with focus on GT regions
    # =========================================================================
    bad_mask = (patch_types == 1)  # (B,) bool
    n_bad = bad_mask.sum().item()

    L_bad_region = torch.tensor(0.0, device=device)
    if n_bad > 0 and lambda_bad_region > 0:
        probs_bad = probs[bad_mask]  # (N_bad, H, W)
        gt_bad = gt[bad_mask]  # (N_bad, H, W)

        # Stronger BCE on bad regions: higher weight on positive class
        # For bad-centered patches, we want to push predictions toward 1 on GT=bad regions
        gt_positive = (gt_bad > 0.5).float()

        # Clamp probs to avoid log(0) = -inf
        probs_clamped = probs_bad.clamp(min=1e-6, max=1-1e-6)

        # BCE loss with higher weight on positives
        neg_log_bad = -torch.log(probs_clamped)  # Loss when GT=1
        neg_log_good_bce = -torch.log(1.0 - probs_clamped)  # Loss when GT=0

        # Weight positives 3x higher than negatives (focus on learning bad regions)
        weight_pos = 3.0
        weight_neg = 1.0
        bce_weighted = weight_pos * gt_positive * neg_log_bad + weight_neg * (1 - gt_positive) * neg_log_good_bce
        L_bad_region = bce_weighted.mean()

    stats["loss_bad_region"] = float(L_bad_region.item()) if isinstance(L_bad_region, torch.Tensor) else 0.0
    stats["n_bad_patches"] = n_bad

    # =========================================================================
    # Patch-level classification (if encoder features provided)
    # Binary classifier: is this patch "good" (particle-centered) or "bad" (bad-centered)?
    # Random patches excluded from this loss
    # =========================================================================
    L_patch_classify = torch.tensor(0.0, device=device)
    if encoder_features is not None and lambda_patch_classify > 0:
        # encoder_features should be (B, C) after global pooling
        # We'll add a simple linear classifier in the training loop
        # For now, use average probability as a proxy for patch classification

        # Create binary labels: 1=bad patch, 0=good patch (particle)
        # Only use bad-centered (1) and particle-centered (2) patches
        classify_mask = (patch_types == 1) | (patch_types == 2)
        n_classify = classify_mask.sum().item()

        if n_classify > 0:
            # Labels: bad-centered=1, particle-centered=0
            patch_labels = (patch_types[classify_mask] == 1).float()  # (N_classify,)

            # Patch-level prediction: mean probability over patch
            patch_probs = probs[classify_mask].mean(dim=(1, 2))  # (N_classify,)

            # BCE loss for patch classification
            L_patch_classify = F.binary_cross_entropy(
                patch_probs.clamp(1e-6, 1-1e-6),
                patch_labels,
                reduction='mean'
            )

    stats["loss_patch_classify"] = float(L_patch_classify.item()) if isinstance(L_patch_classify, torch.Tensor) else 0.0

    # =========================================================================
    # Total patch-type loss (to be added to main loss)
    # =========================================================================
    total_loss = (
        lambda_particle_region * L_particle_region +
        lambda_bad_region * L_bad_region +
        lambda_patch_classify * L_patch_classify
    )

    # NaN protection - return zero loss if something went wrong
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(0.0, device=device)
        stats["patch_type_nan_detected"] = 1.0

    stats["loss_patch_type_total"] = float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else 0.0

    return total_loss, stats


def compute_clean_mask(gt_mask: np.ndarray, dilate_radius: int = 3) -> np.ndarray:
    """
    Compute clean mask: clean = (gt==0) AND (dilate(gt, r=dilate_radius)==0)
    This avoids boundary ambiguity by excluding regions near GT boundaries.

    Args:
        gt_mask: GT mask (H, W), 1=bad, 0=good
        dilate_radius: Radius for dilation to exclude boundary regions

    Returns:
        Binary mask (H, W) where 1 = clean region (definitely negative)
    """
    from scipy.ndimage import binary_dilation
    gt_binary = (gt_mask > 0.5).astype(bool)
    gt_dilated = binary_dilation(gt_binary, iterations=dilate_radius)
    clean = (~gt_binary) & (~gt_dilated)  # Clean = not GT AND not near GT
    return clean.astype(np.float32)


def compute_stitched_train_loss(
    model: nn.Module,
    mic_path: Path,
    gt_path: Path,
    dataset_id: str,
    rule,
    device: str,
    patch_size: int,
    normalization_method: str,
    psd_multiscale: bool = False,
    psd_scales: tuple = (16, 32, 64),
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    psd_frequency_band_anisotropy_min_freq: float = 0.0,
    psd_texture_mode: bool = False,
    psd_highpass_cutoff: float = 0.2,
    use_psd: bool = True,
    include_real_space_input: bool = True,
    psd_multiscale_separate_channels: bool = False,
    use_pixel_size_channel: bool = False,
    pixel_size_by_dataset: Optional[Dict[str, float]] = None,
    pixel_size_channel_min_angstrom: float = 0.3,
    pixel_size_channel_max_angstrom: float = 1.4,
    downsample_factor: float = 1.0,
    grid_patches: int = 16,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute stitched loss on a full micrograph during training.
    Uses a sparse grid of patches for efficiency while still covering the full image.

    This forces the model to see the TRUE distribution (not GT-biased) during training,
    which prevents the train/stitched-val misalignment that causes collapse.

    Args:
        model: Model in training mode (gradients enabled)
        mic_path: Path to micrograph
        gt_path: Path to GT mask
        dataset_id: Dataset ID for rule application
        rule: Region GT transform rule
        device: Device for computation
        patch_size: Patch size at working resolution
        normalization_method: 'robust' or 'standard'
        psd_multiscale: Whether to use multi-scale PSD
        psd_scales: PSD scale bins
        psd_texture_mode: Whether to use texture mode
        psd_highpass_cutoff: Highpass cutoff for texture mode
        downsample_factor: Downsample factor for this dataset
        grid_patches: Number of patches in sparse grid (e.g., 16 = 4x4)

    Returns:
        loss: Scalar loss tensor (Dice loss on stitched prediction)
        stats: Dict with 'stitched_dice', 'stitched_fp_area', 'stitched_pos_median', 'stitched_neg_median'
    """
    from scipy import ndimage as ndi

    # Load micrograph and GT
    img = _load_mrc_2d_permissive(mic_path)
    img = np.asarray(img).squeeze().astype(np.float32)
    gt = np.load(str(gt_path))
    gt = np.asarray(gt).squeeze()
    region_gt = apply_region_gt_transform(gt, dataset_id=dataset_id, rule=rule).astype(np.float32)

    # Downsample if needed
    ds = int(max(1, downsample_factor))
    if ds > 1:
        img = img[::ds, ::ds]
        region_gt = region_gt[::ds, ::ds]

    # Ensure GT matches image dimensions (can differ by 1-2 px due to original size mismatch)
    if region_gt.shape != img.shape:
        matched_gt = np.zeros(img.shape, dtype=region_gt.dtype)
        h_copy, w_copy = min(img.shape[0], region_gt.shape[0]), min(img.shape[1], region_gt.shape[1])
        matched_gt[:h_copy, :w_copy] = region_gt[:h_copy, :w_copy]
        region_gt = matched_gt

    # Normalize image - use same function as validation for consistency!
    from utils.image_utils import normalize_image
    img = normalize_image(img, method=normalization_method)
    effective_pixel_size = _effective_pixel_size_for_dataset(
        dataset_id=dataset_id,
        pixel_size_by_dataset=pixel_size_by_dataset,
        downsample_factor=downsample_factor,
    )

    h, w = img.shape

    # Create sparse grid of patch positions
    grid_size = int(np.sqrt(grid_patches))
    step_y = max(1, (h - patch_size) // (grid_size - 1)) if grid_size > 1 else 0
    step_x = max(1, (w - patch_size) // (grid_size - 1)) if grid_size > 1 else 0

    patches = []
    patch_positions = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y = min(gy * step_y, h - patch_size) if step_y > 0 else 0
            x = min(gx * step_x, w - patch_size) if step_x > 0 else 0
            if y < 0 or x < 0 or y + patch_size > h or x + patch_size > w:
                continue
            patch_img = img[y:y+patch_size, x:x+patch_size].copy()
            patches.append(patch_img)
            patch_positions.append((y, x))

    if len(patches) == 0:
        # Image too small for patches
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'stitched_dice': 0.0, 'stitched_fp_area': 0.0,
            'stitched_pos_median': 0.0, 'stitched_neg_median': 0.0
        }

    # Build model inputs for each patch
    patch_tensors = []
    for patch_img in patches:
        pixel_size_channel = None
        if use_pixel_size_channel:
            pixel_size_channel = _build_pixel_size_channel(
                image_shape=patch_img.shape,
                dataset_id=dataset_id,
                pixel_size_by_dataset=pixel_size_by_dataset,
                min_angstrom=pixel_size_channel_min_angstrom,
                max_angstrom=pixel_size_channel_max_angstrom,
                downsample_factor=downsample_factor,
            )
        if use_psd:
            psd = _compute_psd_map(
                img=patch_img,
                psd_use_radial_normalization=True,
                psd_use_radial_image=False,
                psd_radial_bins=64,
                psd_multiscale=psd_multiscale,
                psd_scales=psd_scales,
                psd_texture_mode=psd_texture_mode,
                psd_highpass_cutoff=psd_highpass_cutoff,
                psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                pixel_size_angstrom=effective_pixel_size,
                psd_frequency_band_channels=psd_frequency_band_channels,
                psd_frequency_bands=psd_frequency_bands,
                psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
                psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
            )
            patch_x = _stack_image_with_psd(
                patch_img,
                psd,
                use_psd_input=True,
                include_real_space_input=include_real_space_input,
                pixel_size_channel=pixel_size_channel,
            )
        else:
            patch_x = _stack_image_with_psd(
                patch_img,
                np.zeros_like(patch_img, dtype=np.float32),
                use_psd_input=False,
                include_real_space_input=include_real_space_input,
                pixel_size_channel=pixel_size_channel,
            )
        patch_tensors.append(torch.from_numpy(np.ascontiguousarray(patch_x, dtype=np.float32)).float())

    # Stack into batch and run through model
    batch = torch.stack(patch_tensors, dim=0).to(device)  # (N, 2, H, W)

    # Forward pass WITH gradients
    logits = model(batch)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    probs = torch.sigmoid(logits).squeeze(1)  # (N, H, W)

    # Stitch predictions into full probability map
    prob_map = torch.zeros((h, w), device=device, dtype=torch.float32)
    weight_map = torch.zeros((h, w), device=device, dtype=torch.float32)

    for i, (y, x) in enumerate(patch_positions):
        prob_map[y:y+patch_size, x:x+patch_size] += probs[i]
        weight_map[y:y+patch_size, x:x+patch_size] += 1.0

    # Average overlapping regions
    prob_map = prob_map / (weight_map + 1e-8)

    # Convert GT to tensor
    gt_tensor = torch.from_numpy(region_gt).float().to(device)

    # Resize GT if needed to match prob_map
    if gt_tensor.shape != prob_map.shape:
        gt_tensor = F.interpolate(
            gt_tensor.unsqueeze(0).unsqueeze(0),
            size=prob_map.shape,
            mode='nearest'
        ).squeeze()

    # Compute Dice loss (differentiable)
    smooth = 1e-6
    intersection = (prob_map * gt_tensor).sum()
    dice_coef = (2.0 * intersection + smooth) / (prob_map.sum() + gt_tensor.sum() + smooth)
    dice_loss = 1.0 - dice_coef

    # Compute stats (detached for logging)
    with torch.no_grad():
        pred_binary = (prob_map > 0.5).float()
        gt_binary = (gt_tensor > 0.5).float()
        fp = ((pred_binary > 0) & (gt_binary == 0)).float().sum()
        tn = ((pred_binary == 0) & (gt_binary == 0)).float().sum()
        fp_area = float(fp / (fp + tn + 1e-10))

        pos_mask = gt_tensor > 0.5
        neg_mask = gt_tensor <= 0.5
        pos_median = float(prob_map[pos_mask].median()) if pos_mask.any() else 0.0
        neg_median = float(prob_map[neg_mask].median()) if neg_mask.any() else 0.0

    stats = {
        'stitched_dice': float(dice_coef.detach()),
        'stitched_fp_area': fp_area,
        'stitched_pos_median': pos_median,
        'stitched_neg_median': neg_median,
    }

    return dice_loss, stats


def extract_dense_patches_from_micrograph(
    mic_path: Path,
    gt_path: Path,
    dataset_id: str,
    rule,
    patch_size: int,
    stride: int,
    normalization_method: str,
    psd_multiscale: bool = False,
    psd_scales: tuple = (16, 32, 64),
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    psd_frequency_band_anisotropy_min_freq: float = 0.0,
    psd_texture_mode: bool = False,
    psd_highpass_cutoff: float = 0.2,
    psd_multiscale_separate_channels: bool = False,
    downsample_factor: float = 1.0,
    max_patches: int = 64,
    use_psd: bool = True,
    include_real_space_input: bool = True,
    adaptive_downsample_method: str = "stride",
    use_pixel_size_channel: bool = False,
    pixel_size_by_dataset: Optional[Dict[str, float]] = None,
    pixel_size_channel_min_angstrom: float = 0.3,
    pixel_size_channel_max_angstrom: float = 1.4,
    hard_positive_fraction: float = 0.0,
    hard_positive_min_gt_frac: float = 0.10,
    hard_positive_erode_px: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract ALL overlapping patches from a full micrograph (MicrographCleaner-style).
    NO GT-bias - this samples uniformly across the entire micrograph.

    Args:
        mic_path: Path to micrograph file
        gt_path: Path to GT mask (numpy file)
        dataset_id: Dataset ID for rule application
        rule: Region GT transform rule
        patch_size: Patch size at working resolution
        stride: Stride between patches (e.g., patch_size//2 for 50% overlap)
        normalization_method: 'robust' or 'standard'
        psd_multiscale: Whether to use multi-scale PSD
        psd_scales: PSD scale bins
        psd_texture_mode: Whether to use texture mode PSD
        psd_highpass_cutoff: Highpass cutoff for texture mode
        downsample_factor: Downsample factor for this dataset
        max_patches: Maximum patches to extract (randomly sample if exceeded)

    Returns:
        patches: List of (C, H, W) arrays (image + PSD channels)
        labels: List of (H, W) GT masks for each patch
    """
    import random

    img, region_gt, effective_pixel_size = _load_preprocessed_full_micrograph(
        mic_path=mic_path,
        gt_path=gt_path,
        dataset_id=dataset_id,
        rule=rule,
        normalization_method=normalization_method,
        downsample_factor=downsample_factor,
        adaptive_downsample_method=adaptive_downsample_method,
        pixel_size_by_dataset=pixel_size_by_dataset,
    )

    h, w = img.shape

    # Collect all valid patch positions
    all_positions = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            all_positions.append((y, x))

    if len(all_positions) == 0:
        return [], []

    # Randomly sample if too many patches. Optionally reserve a fraction for
    # patches with broad GT interiors so subtle large carbon regions are not
    # diluted by uniformly sampled background patches.
    if len(all_positions) > max_patches:
        n_hard_pos = int(round(max_patches * max(0.0, min(1.0, float(hard_positive_fraction)))))
        selected = []
        selected_set = set()
        if n_hard_pos > 0 and np.any(region_gt > 0):
            core_gt = None
            erode_px = max(0, int(hard_positive_erode_px))
            if erode_px > 0:
                core_gt = ndimage.binary_erosion(
                    region_gt > 0.5,
                    structure=np.ones((3, 3), dtype=bool),
                    iterations=erode_px,
                    border_value=0,
                )

            scored_positions = []
            min_gt_frac = max(0.0, float(hard_positive_min_gt_frac))
            for y, x in all_positions:
                patch_gt = region_gt[y:y+patch_size, x:x+patch_size] > 0.5
                gt_frac = float(patch_gt.mean())
                if gt_frac < min_gt_frac:
                    continue
                core_frac = 0.0
                if core_gt is not None:
                    core_frac = float(core_gt[y:y+patch_size, x:x+patch_size].mean())
                scored_positions.append((gt_frac + 0.75 * core_frac, y, x))

            if scored_positions:
                scored_positions.sort(reverse=True)
                top_k = min(len(scored_positions), max(n_hard_pos * 4, n_hard_pos))
                top_positions = [(y, x) for _, y, x in scored_positions[:top_k]]
                for pos in random.sample(top_positions, min(n_hard_pos, len(top_positions))):
                    selected.append(pos)
                    selected_set.add(pos)

        n_remaining = max_patches - len(selected)
        remaining_pool = [pos for pos in all_positions if pos not in selected_set]
        if n_remaining > 0:
            selected.extend(random.sample(remaining_pool, min(n_remaining, len(remaining_pool))))
        positions = selected
    else:
        positions = all_positions

    # Extract patches first, then compute PSD in batch when possible.
    patch_imgs = []
    labels = []
    for y, x in positions:
        patch_img = img[y:y+patch_size, x:x+patch_size].copy()
        patch_gt = region_gt[y:y+patch_size, x:x+patch_size].copy()

        # Pad if necessary to ensure exact patch_size x patch_size
        if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
            padded_img = np.zeros((patch_size, patch_size), dtype=patch_img.dtype)
            padded_gt = np.zeros((patch_size, patch_size), dtype=patch_gt.dtype)
            h_actual, w_actual = patch_img.shape
            padded_img[:h_actual, :w_actual] = patch_img
            padded_gt[:h_actual, :w_actual] = patch_gt
            # Replicate edge pixels instead of zeros
            if h_actual < patch_size:
                padded_img[h_actual:, :w_actual] = patch_img[-1:, :]
                padded_gt[h_actual:, :w_actual] = patch_gt[-1:, :]
            if w_actual < patch_size:
                padded_img[:h_actual, w_actual:] = patch_img[:, -1:]
                padded_gt[:h_actual, w_actual:] = patch_gt[:, -1:]
            if h_actual < patch_size and w_actual < patch_size:
                padded_img[h_actual:, w_actual:] = patch_img[-1, -1]
                padded_gt[h_actual:, w_actual:] = patch_gt[-1, -1]
            patch_img = padded_img
            patch_gt = padded_gt
        patch_imgs.append(np.ascontiguousarray(patch_img, dtype=np.float32))
        labels.append(patch_gt)

    if not patch_imgs:
        return [], labels

    pixel_size_channel = None
    if use_pixel_size_channel:
        pixel_size_channel = _build_pixel_size_channel(
            image_shape=patch_imgs[0].shape,
            dataset_id=dataset_id,
            pixel_size_by_dataset=pixel_size_by_dataset,
            min_angstrom=pixel_size_channel_min_angstrom,
            max_angstrom=pixel_size_channel_max_angstrom,
            downsample_factor=downsample_factor,
        )

    patch_img_batch = None
    psd_batch = None
    if use_psd and (
        psd_frequency_band_channels
        or (psd_multiscale and not psd_multiscale_separate_channels)
    ):
        patch_img_batch = np.ascontiguousarray(np.stack(patch_imgs, axis=0), dtype=np.float32)

    if use_psd and psd_frequency_band_channels:
        psd_batch = None
    elif use_psd and psd_multiscale and not psd_multiscale_separate_channels:
        psd_batch = compute_multiscale_radial_psd_image_batch(
            patch_img_batch,
            scales=psd_scales,
            normalize=True,
            texture_mode=psd_texture_mode,
            highpass_cutoff=psd_highpass_cutoff,
        )

    patches = []
    for idx, patch_img in enumerate(patch_imgs):
        if use_psd:
            if psd_batch is not None:
                psd = psd_batch[idx]
            else:
                psd = _compute_psd_map(
                    img=patch_img,
                    psd_use_radial_normalization=True,
                    psd_use_radial_image=False,
                    psd_radial_bins=64,
                    psd_multiscale=psd_multiscale,
                    psd_scales=psd_scales,
                    psd_texture_mode=psd_texture_mode,
                    psd_highpass_cutoff=psd_highpass_cutoff,
                    psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                    pixel_size_angstrom=effective_pixel_size,
                    psd_frequency_band_channels=psd_frequency_band_channels,
                    psd_frequency_bands=psd_frequency_bands,
                    psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
                    psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                    psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
                )
            patch_input = _stack_image_with_psd(
                patch_img,
                psd,
                use_psd_input=True,
                include_real_space_input=include_real_space_input,
                pixel_size_channel=pixel_size_channel,
            )
        else:
            patch_input = _stack_image_with_psd(
                patch_img,
                np.zeros_like(patch_img, dtype=np.float32),
                use_psd_input=False,
                include_real_space_input=include_real_space_input,
                pixel_size_channel=pixel_size_channel,
            )
        patches.append(patch_input)

    return patches, labels


def _load_preprocessed_full_micrograph(
    mic_path: Path,
    gt_path: Path,
    dataset_id: str,
    rule,
    normalization_method: str,
    downsample_factor: float,
    adaptive_downsample_method: str,
    pixel_size_by_dataset: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load + preprocess one full micrograph, caching the result in RAM across epochs."""
    global _FULL_MICROGRAPH_CACHE_BYTES

    cache_key = (
        str(Path(mic_path).resolve()),
        str(Path(gt_path).resolve()),
        str(dataset_id),
        round(float(downsample_factor), 6),
        str(normalization_method),
        str(adaptive_downsample_method),
        id(rule),
    )

    if _FULL_MICROGRAPH_CACHE_LIMIT_BYTES > 0:
        with _FULL_MICROGRAPH_CACHE_LOCK:
            cached = _FULL_MICROGRAPH_CACHE.get(cache_key)
            if cached is not None:
                _FULL_MICROGRAPH_CACHE.move_to_end(cache_key)
                img_cached, gt_cached, eff_px_cached, _ = cached
                return img_cached, gt_cached, eff_px_cached

    img = _load_mrc_2d_permissive(mic_path)
    img = np.asarray(img).squeeze().astype(np.float32)
    gt = np.load(str(gt_path))
    gt = np.asarray(gt).squeeze()
    region_gt = apply_region_gt_transform(gt, dataset_id=dataset_id, rule=rule).astype(np.float32)

    if float(downsample_factor) > 1.0:
        img, region_gt = _resample_image_and_mask(
            img,
            region_gt,
            downsample_factor,
            method=adaptive_downsample_method,
        )

    if region_gt.shape != img.shape:
        matched_gt = np.zeros(img.shape, dtype=region_gt.dtype)
        h_copy = min(img.shape[0], region_gt.shape[0])
        w_copy = min(img.shape[1], region_gt.shape[1])
        matched_gt[:h_copy, :w_copy] = region_gt[:h_copy, :w_copy]
        region_gt = matched_gt

    img = normalize_image(img, method=normalization_method).astype(np.float32, copy=False)
    region_gt = (region_gt > 0.5).astype(np.uint8, copy=False)
    effective_pixel_size = _effective_pixel_size_for_dataset(
        dataset_id=dataset_id,
        pixel_size_by_dataset=pixel_size_by_dataset,
        downsample_factor=downsample_factor,
    )

    if _FULL_MICROGRAPH_CACHE_LIMIT_BYTES > 0:
        img_cached = np.ascontiguousarray(img, dtype=np.float32)
        gt_cached = np.ascontiguousarray(region_gt, dtype=np.uint8)
        img_cached.setflags(write=False)
        gt_cached.setflags(write=False)
        payload_bytes = int(img_cached.nbytes + gt_cached.nbytes)
        with _FULL_MICROGRAPH_CACHE_LOCK:
            existing = _FULL_MICROGRAPH_CACHE.pop(cache_key, None)
            if existing is not None:
                _FULL_MICROGRAPH_CACHE_BYTES -= int(existing[3])
            _FULL_MICROGRAPH_CACHE[cache_key] = (img_cached, gt_cached, float(effective_pixel_size), payload_bytes)
            _FULL_MICROGRAPH_CACHE_BYTES += payload_bytes
            while _FULL_MICROGRAPH_CACHE and _FULL_MICROGRAPH_CACHE_BYTES > _FULL_MICROGRAPH_CACHE_LIMIT_BYTES:
                _, (_, _, _, evicted_bytes) = _FULL_MICROGRAPH_CACHE.popitem(last=False)
                _FULL_MICROGRAPH_CACHE_BYTES -= int(evicted_bytes)
        return img_cached, gt_cached, float(effective_pixel_size)

    return img, region_gt, float(effective_pixel_size)


def full_micrograph_training_epoch(
    model: nn.Module,
    train_mics: List[Tuple[Path, Path, str, float]],
    rule,
    device: str,
    patch_size: int,
    stride: int,
    normalization_method: str,
    psd_multiscale: bool,
    psd_multiscale_separate_channels: bool,
    psd_scales: tuple,
    psd_frequency_band_channels: bool,
    psd_frequency_bands: tuple,
    psd_frequency_band_include_full_spectrum: bool,
    psd_frequency_band_include_anisotropy: bool,
    psd_frequency_band_anisotropy_min_freq: float,
    psd_texture_mode: bool,
    psd_highpass_cutoff: float,
    batch_size: int,
    max_patches_per_mic: int,
    mics_per_epoch: int,
    use_dice_loss: bool,
    lambda_dice: float,
    lambda_focal: float,
    focal_gamma: float,
    focal_alpha: float,
    lambda_bce: float,
    bce_neg_weight: float,
    lambda_edge: float,
    opt: torch.optim.Optimizer,
    lr_scheduler,
    grad_clip_norm: float = 1.0,
    epoch: int = 0,
    edge_erosion_px: int = 0,  # NEW: Pixels to exclude from loss at patch boundaries
    use_tukey_weight: bool = True,  # NEW: Use smooth Tukey window (vs hard erosion)
    downsample_factor: float = 1.0,  # NEW: For computing effective edge erosion
    label_smoothing: float = 0.0,  # NEW: Label smoothing to prevent overconfidence
    lr_schedule_fn=None,  # Callable(step) -> lr, for per-step LR schedule
    lr_step_counter: list = None,  # Mutable list [step_count] for tracking across epochs
    use_augmentation: bool = True,  # Random flip + rot90 (8x diversity)
    use_psd: bool = True,  # Whether to include PSD channels in model input
    include_real_space_input: bool = True,  # If False, use PSD-only input channels
    adaptive_downsample_method: str = "stride",
    use_pixel_size_channel: bool = False,
    pixel_size_by_dataset: Optional[Dict[str, float]] = None,
    pixel_size_channel_min_angstrom: float = 0.3,
    pixel_size_channel_max_angstrom: float = 1.4,
    prefetch_workers: int = 1,
    train_normalization_methods: Optional[List[str]] = None,
    train_normalization_probs: Optional[List[float]] = None,
    hard_positive_fraction: float = 0.0,
    hard_positive_min_gt_frac: float = 0.10,
    hard_positive_erode_px: int = 0,
    positive_interior_weight: float = 0.0,
    positive_interior_erode_px: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Run one full epoch of MicrographCleaner-style training.

    Processes full micrographs with dense overlapping patches, NO GT-bias.
    Training distribution = Validation distribution.

    NEW: Supports edge erosion via Tukey window to prevent patch boundary artifacts.

    Args:
        model: Model in training mode
        train_mics: List of (mic_path, gt_path, dataset_id, downsample_factor)
        rule: Region GT transform rule
        device: Device for computation
        patch_size: Patch size at working resolution
        stride: Stride for overlapping patches
        normalization_method: 'robust' or 'standard'
        psd_multiscale: Whether to use multi-scale PSD
        psd_scales: PSD scale bins
        psd_texture_mode: Whether to use texture mode
        psd_highpass_cutoff: Highpass cutoff for texture mode
        batch_size: Batch size for processing patches
        max_patches_per_mic: Max patches to extract per micrograph
        mics_per_epoch: Number of micrographs to process per epoch
        use_dice_loss: Whether to use Dice+Focal loss
        lambda_dice, lambda_focal, ...: Loss weights
        opt: Optimizer
        lr_scheduler: Learning rate scheduler (optional)
        grad_clip_norm: Gradient clipping norm
        epoch: Current epoch number

    Returns:
        avg_loss: Average training loss for epoch
        stats: Dict with training statistics
    """
    import random

    model.train()

    # Select micrographs for this epoch
    # mics_per_epoch=0 means "use ALL micrographs" (recommended for stable training)
    if mics_per_epoch <= 0 or mics_per_epoch >= len(train_mics):
        # Use ALL micrographs - stable training loss, better convergence
        epoch_mics = train_mics.copy()
        random.shuffle(epoch_mics)  # Shuffle order for variety
    else:
        # Subsample (only if compute-limited)
        epoch_mics = random.sample(train_mics, mics_per_epoch)

    norm_methods = [str(normalization_method)]
    if train_normalization_methods:
        norm_methods = [str(m).strip() for m in train_normalization_methods if str(m).strip()]
        if not norm_methods:
            norm_methods = [str(normalization_method)]
    norm_probs = None
    if train_normalization_probs and len(train_normalization_probs) == len(norm_methods):
        prob_arr = np.asarray(train_normalization_probs, dtype=np.float64)
        if np.all(np.isfinite(prob_arr)) and prob_arr.sum() > 0:
            norm_probs = (prob_arr / prob_arr.sum()).tolist()

    epoch_mic_jobs = []
    norm_counts: Dict[str, int] = {}
    for mp, gp, did, dsf in epoch_mics:
        if len(norm_methods) > 1:
            mic_norm = random.choices(norm_methods, weights=norm_probs, k=1)[0]
        else:
            mic_norm = norm_methods[0]
        norm_counts[mic_norm] = norm_counts.get(mic_norm, 0) + 1
        epoch_mic_jobs.append((mp, gp, did, dsf, mic_norm))

    all_losses = []
    all_dice_coeffs = []
    all_dice_015 = []
    all_dice_020 = []
    all_pos_probs = []
    all_neg_probs = []
    skipped_errors = []
    n_mics_processed = 0
    consecutive_oom_skips = 0

    # Process each micrograph with bounded background extraction for CPU/GPU overlap.
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def _extract_mic(args_tuple):
        """Worker function for background patch extraction."""
        mp, gp, did, dsf, mic_norm = args_tuple
        effective_hard_positive_erode = (
            max(0, int(float(hard_positive_erode_px) / max(float(dsf), 1e-6)))
            if hard_positive_erode_px > 0
            else 0
        )
        return extract_dense_patches_from_micrograph(
            mic_path=mp, gt_path=gp, dataset_id=did, rule=rule,
            patch_size=patch_size, stride=stride,
            normalization_method=str(mic_norm),
            psd_multiscale=psd_multiscale, psd_scales=psd_scales,
            psd_frequency_band_channels=psd_frequency_band_channels,
            psd_frequency_bands=psd_frequency_bands,
            psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
            psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
            psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
            psd_texture_mode=psd_texture_mode, psd_highpass_cutoff=psd_highpass_cutoff,
            psd_multiscale_separate_channels=psd_multiscale_separate_channels,
            downsample_factor=dsf, max_patches=max_patches_per_mic,
            use_psd=use_psd,
            include_real_space_input=include_real_space_input,
            adaptive_downsample_method=adaptive_downsample_method,
            use_pixel_size_channel=use_pixel_size_channel,
            pixel_size_by_dataset=pixel_size_by_dataset,
            pixel_size_channel_min_angstrom=pixel_size_channel_min_angstrom,
            pixel_size_channel_max_angstrom=pixel_size_channel_max_angstrom,
            hard_positive_fraction=hard_positive_fraction,
            hard_positive_min_gt_frac=hard_positive_min_gt_frac,
            hard_positive_erode_px=effective_hard_positive_erode,
        )

    prefetch_workers = max(1, int(prefetch_workers))
    executor = ThreadPoolExecutor(max_workers=prefetch_workers)
    future_queue = deque()
    submit_cursor = 0
    while submit_cursor < min(prefetch_workers, len(epoch_mic_jobs)):
        future_queue.append((submit_cursor, executor.submit(_extract_mic, epoch_mic_jobs[submit_cursor])))
        submit_cursor += 1

    pbar = tqdm(epoch_mic_jobs, desc=f"Epoch {epoch:02d} (full-mic)", unit="mic", ncols=120)
    for mic_idx, (mic_path, gt_path, dataset_id, ds_factor, mic_norm) in enumerate(pbar):
        try:
            current_future = None
            if future_queue and future_queue[0][0] == mic_idx:
                _, current_future = future_queue.popleft()
            if submit_cursor < len(epoch_mic_jobs):
                future_queue.append((submit_cursor, executor.submit(_extract_mic, epoch_mic_jobs[submit_cursor])))
                submit_cursor += 1
            if current_future is not None:
                patches, labels = current_future.result()
            else:
                patches, labels = _extract_mic((mic_path, gt_path, dataset_id, ds_factor, mic_norm))

            if len(patches) == 0:
                skipped_errors.append(f"No patches extracted from {mic_path.name}")
                consecutive_oom_skips = 0
                continue

            # Process in batches
            n_patches = len(patches)
            mic_losses = []

            for batch_start in range(0, n_patches, batch_size):
                batch_end = min(batch_start + batch_size, n_patches)
                batch_patches = patches[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]

                # Convert to tensors (pin_memory → non_blocking for faster H2D transfer)
                x = torch.stack([torch.from_numpy(p).float() for p in batch_patches]).to(device, non_blocking=True)
                y = torch.stack([torch.from_numpy(l).float() for l in batch_labels]).to(device, non_blocking=True)

                # Augmentation: random flip + rot90 (8x diversity, applied on GPU - fast)
                if use_augmentation:
                    if random.random() < 0.5:
                        x = torch.flip(x, [-1])  # horizontal flip
                        y = torch.flip(y, [-1])
                    if random.random() < 0.5:
                        x = torch.flip(x, [-2])  # vertical flip
                        y = torch.flip(y, [-2])
                    k = random.randint(0, 3)      # 0, 90, 180, or 270 degrees
                    if k > 0:
                        x = torch.rot90(x, k, dims=[-2, -1])
                        y = torch.rot90(y, k, dims=[-2, -1])
                    # flip/rot90 return non-contiguous tensors; .view() in model will fail without this
                    x = x.contiguous()
                    y = y.contiguous()

                # Forward pass with AMP bf16 (A100 tensor cores, no scaler needed)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    if logits.ndim == 3:
                        logits = logits.unsqueeze(1)

                    # Compute loss using compute_dice_focal_loss (includes edge erosion!)
                    if use_dice_loss:
                        # Compute effective edge erosion at working resolution
                        effective_edge_erosion = max(0, int(edge_erosion_px / float(ds_factor))) if edge_erosion_px > 0 else 0
                        effective_positive_interior_erode = (
                            max(0, int(positive_interior_erode_px / float(ds_factor)))
                            if positive_interior_erode_px > 0
                            else 0
                        )

                        # Compute edge map for edge loss (optional) - GPU-batched Sobel
                        edge_map = None
                        if lambda_edge > 0:
                            edge_map = _gpu_sobel_edge_map(x[:, 0:1, :, :])  # (B,1,H,W) -> (B,H,W)

                        # Use compute_dice_focal_loss with edge erosion support
                        loss, stats = compute_dice_focal_loss(
                            logits, y.unsqueeze(1),
                            lambda_dice=lambda_dice,
                            lambda_focal=lambda_focal,
                            lambda_edge=lambda_edge,
                            focal_gamma=focal_gamma,
                            focal_alpha=focal_alpha,
                            edge_map=edge_map,
                            edge_erosion_px=effective_edge_erosion,
                            use_tukey_weight=use_tukey_weight,
                            label_smoothing=label_smoothing,
                            positive_interior_weight=positive_interior_weight,
                            positive_interior_erode_px=effective_positive_interior_erode,
                        )

                        # Compute probs for calibration tracking (needed below)
                        probs = torch.sigmoid(logits.squeeze(1))

                        all_dice_coeffs.append(stats.get("dice_coefficient", 0.0))
                        all_dice_015.append(stats.get("dice_at_0.15", 0.0))
                        all_dice_020.append(stats.get("dice_at_0.20", 0.0))
                    else:
                        # BCE loss (no edge erosion - legacy path)
                        pos_weight = torch.tensor([bce_neg_weight]).to(device)
                        probs = torch.sigmoid(logits.squeeze(1))
                        bce_loss = F.binary_cross_entropy_with_logits(
                            logits.squeeze(1), y,
                            pos_weight=pos_weight.expand_as(y)
                        )
                        loss = lambda_bce * bce_loss

                # Backward (outside autocast - standard AMP practice)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Per-step LR schedule (used in full-mic mode)
                if lr_schedule_fn is not None and lr_step_counter is not None:
                    lr_step_counter[0] += 1
                    new_lr = lr_schedule_fn(lr_step_counter[0])
                    for g in opt.param_groups:
                        g["lr"] = new_lr

                mic_losses.append(float(loss.detach()))

                # Track probability calibration
                with torch.no_grad():
                    pos_mask = y > 0.5
                    neg_mask = y <= 0.5
                    if pos_mask.any():
                        all_pos_probs.append(float(probs[pos_mask].mean()))
                    if neg_mask.any():
                        all_neg_probs.append(float(probs[neg_mask].mean()))

            if mic_losses:
                consecutive_oom_skips = 0
                n_mics_processed += 1
                all_losses.append(np.mean(mic_losses))
                pbar.set_postfix({"loss": f"{all_losses[-1]:.4f}", "avg": f"{np.mean(all_losses):.4f}"})

        except Exception as e:
            error_text = str(e)
            skipped_errors.append(error_text)
            is_cuda_oom = str(device).startswith("cuda") and "out of memory" in error_text.lower()
            if is_cuda_oom:
                x = y = logits = loss = probs = edge_map = None
                torch.cuda.empty_cache()
                consecutive_oom_skips += 1
            else:
                consecutive_oom_skips = 0
            print(f"  [Full-mic] Skip {mic_path.name}: {e}", flush=True)
            if is_cuda_oom and n_mics_processed == 0 and consecutive_oom_skips >= 3:
                raise RuntimeError(
                    "Full-micrograph training hit CUDA out-of-memory on the first "
                    f"{consecutive_oom_skips} attempted micrographs. Reduce --batch_size "
                    "and --stitched_val_batch_forward_size, or run on a GPU with more free memory."
                ) from None
            continue

    executor.shutdown(wait=False)

    # Compute epoch statistics
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    stats = {
        'train_loss': avg_loss,
        'dice_coeff': np.mean(all_dice_coeffs) if all_dice_coeffs else 0.0,
        'dice_at_0.15': np.mean(all_dice_015) if all_dice_015 else 0.0,
        'dice_at_0.20': np.mean(all_dice_020) if all_dice_020 else 0.0,
        'pos_prob_mean': np.mean(all_pos_probs) if all_pos_probs else 0.0,
        'neg_prob_mean': np.mean(all_neg_probs) if all_neg_probs else 0.0,
        'n_mics_processed': n_mics_processed,
        'n_mics_total': len(epoch_mic_jobs),
        'n_mics_skipped': len(skipped_errors),
        'first_skip_error': skipped_errors[0] if skipped_errors else "",
        'normalization_counts': norm_counts,
    }

    return avg_loss, stats


def compute_edge_map(image: np.ndarray, method: str = "sobel") -> np.ndarray:
    """
    Compute edge map from input micrograph using Sobel or LoG magnitude.
    Normalized to [0, 1] using robust scaling.

    Args:
        image: Input micrograph (H, W)
        method: "sobel" or "log"

    Returns:
        Edge map (H, W) normalized to [0, 1]
    """
    from scipy.ndimage import sobel, gaussian_laplace
    img = np.asarray(image, dtype=np.float32)

    if method == "sobel":
        # Sobel magnitude
        grad_y = sobel(img, axis=0)
        grad_x = sobel(img, axis=1)
        edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    elif method == "log":
        # Laplacian of Gaussian magnitude
        edge_mag = np.abs(gaussian_laplace(img, sigma=1.0))
    else:
        raise ValueError(f"Unknown edge method: {method}")

    # Robust normalization to [0, 1]: use percentiles to handle outliers
    p5, p95 = np.percentile(edge_mag, [5, 95])
    if p95 > p5:
        edge_norm = np.clip((edge_mag - p5) / (p95 - p5), 0.0, 1.0)
    else:
        edge_norm = np.zeros_like(edge_mag)

    return edge_norm.astype(np.float32)


def compute_particle_likeness_proxy(image: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Precompute a "particle-likeness" proxy map P for hard-negative sampling.
    Uses bandpass + local peakiness or LoG blob response magnitude.

    Args:
        image: Normalized image (H, W)
        gt_mask: GT mask (H, W), 1=bad, 0=good

    Returns:
        Particle-likeness map (H, W) where higher values = more particle-like
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation
    img = np.asarray(image, dtype=np.float32)

    # Bandpass filter: high-pass then low-pass
    sigma_low = 2.0  # Remove large-scale variations
    sigma_high = 0.5  # Smooth noise
    low_freq = gaussian_filter(img, sigma=sigma_low)
    high_pass = img - low_freq
    bandpass = gaussian_filter(high_pass, sigma=sigma_high)

    # LoG blob response (alternative: can use this instead of bandpass)
    from scipy.ndimage import gaussian_laplace
    log_response = np.abs(gaussian_laplace(img, sigma=1.0))

    # Combine: bandpass + LoG response
    combined = bandpass + 0.5 * log_response

    # Normalize to [0, 1] using robust scaling
    p5, p95 = np.percentile(combined, [5, 95])
    if p95 > p5:
        proxy = np.clip((combined - p5) / (p95 - p5), 0.0, 1.0)
    else:
        proxy = np.zeros_like(combined)

    # Zero out GT regions (we only want particle-likeness in clean areas)
    # Handle shape mismatch between image and gt_mask (can happen with edge rounding)
    gt_binary = (gt_mask > 0.5).astype(bool)
    if gt_binary.shape != proxy.shape:
        # Resize gt_binary to match proxy shape
        from scipy.ndimage import zoom
        scale_y = proxy.shape[0] / gt_binary.shape[0]
        scale_x = proxy.shape[1] / gt_binary.shape[1]
        gt_binary = zoom(gt_binary.astype(float), (scale_y, scale_x), order=0) > 0.5
    gt_dilated = binary_dilation(gt_binary, iterations=4)  # Margin away from GT
    proxy[gt_dilated] = 0.0

    return proxy.astype(np.float32)


def detect_particle_like_pixels(
    image: np.ndarray,
    gt_mask: np.ndarray,
    margin_px: int = 4,
    sigma_low: float = 2.0,  # Low-freq Gaussian sigma (pixels) - removes large-scale variations
    sigma_high: float = 0.5,  # High-freq Gaussian sigma (pixels) - smooths noise
    local_max_radius: int = 3,
    min_response: float = 0.05,  # Minimum normalized response
) -> np.ndarray:
    """
    Detect particle-like pixels using bandpass filter + local maxima.
    Only searches in clean regions (GT==0) with margin away from GT.

    Strategy: Bandpass filter (high-pass then low-pass) to isolate particle-sized features,
    then find local maxima as particle centers.

    Args:
        image: Normalized image (H, W)
        gt_mask: GT mask (H, W), 1=bad, 0=good
        margin_px: Margin away from GT to search (default: 4px)
        sigma_low: Low-frequency Gaussian sigma in pixels (removes large-scale variations)
        sigma_high: High-frequency Gaussian sigma in pixels (smooths noise)
        local_max_radius: Radius for local maxima detection
        min_response: Minimum normalized response threshold

    Returns:
        Binary mask (H, W) where 1 = particle-like pixel
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation, zoom

    # Create clean region mask (GT==0, dilated to add margin)
    gt_binary = (gt_mask > 0.5).astype(np.float32)
    # Handle shape mismatch between image and gt_mask (can happen with edge rounding)
    if gt_binary.shape != image.shape:
        scale_y = image.shape[0] / gt_binary.shape[0]
        scale_x = image.shape[1] / gt_binary.shape[1]
        gt_binary = zoom(gt_binary, (scale_y, scale_x), order=0)
    gt_dilated = binary_dilation(gt_binary > 0.5, iterations=margin_px)
    clean_mask = (1.0 - gt_dilated.astype(np.float32)).astype(bool)

    if not clean_mask.any():
        return np.zeros_like(image, dtype=np.uint8)

    # Bandpass filter: high-pass then low-pass
    # High-pass: subtract low-freq component (removes large-scale intensity variations)
    low_freq = gaussian_filter(image.astype(np.float32), sigma=sigma_low)
    high_pass = image.astype(np.float32) - low_freq

    # Low-pass: smooth high-pass result (removes noise, keeps particle-sized features)
    bandpass = gaussian_filter(high_pass, sigma=sigma_high)

    # Normalize response to [0, 1] for thresholding
    bp_min, bp_max = bandpass.min(), bandpass.max()
    if bp_max > bp_min:
        bandpass_norm = (bandpass - bp_min) / (bp_max - bp_min)
    else:
        bandpass_norm = np.zeros_like(bandpass)

    # Local maxima detection (particle centers)
    local_max = maximum_filter(bandpass_norm, size=2*local_max_radius+1)
    is_maxima = (bandpass_norm == local_max) & (bandpass_norm > min_response)

    # Only in clean regions
    particle_like = (is_maxima & clean_mask).astype(np.uint8)

    return particle_like


def precompute_gt_cores(gt_mask: torch.Tensor, erosion_px: int = 8) -> torch.Tensor:
    """Precompute eroded GT cores on GPU using unfold."""
    # Convert to numpy for scipy erosion, then back to tensor
    gt_np = gt_mask.detach().cpu().numpy()
    cores = []
    for b in range(gt_np.shape[0]):
        core = ndimage.binary_erosion(gt_np[b] > 0.5, iterations=erosion_px)
        cores.append(torch.from_numpy(core.astype(np.float32)))
    return torch.stack(cores).to(gt_mask.device)


def _build_fixed_stitched_val_set(
    val_df: pd.DataFrame,
    mic_dir: Path,
    rule,
    n_per_dataset: int,
) -> List[Tuple[Path, Path, str, str]]:
    """Fixed stitched-val set: 1–2 mics per dataset (sorted stem). Returns [(mic_path, gt_path, dataset_id, stem), ...]."""
    out = []
    for dataset_id in sorted(val_df["dataset_id"].unique()):
        sub = val_df[val_df["dataset_id"] == dataset_id].copy().sort_values("stem").head(n_per_dataset)
        for _, r in sub.iterrows():
            try:
                dsid = str(r["dataset_id"])
                stem = str(r["stem"])
                gt_path = Path(r["gt_mask_path"])
                if not gt_path.exists():
                    continue
                if "micrograph_path" in r and pd.notna(r["micrograph_path"]):
                    mic_path = Path(r["micrograph_path"])
                    if not mic_path.exists():
                        mic_path = _find_mic_path(mic_dir, dsid, stem)
                else:
                    mic_path = _find_mic_path(mic_dir, dsid, stem)
                out.append((mic_path, gt_path, dsid, stem))
            except Exception:
                continue
    return out


def run_stitched_val_fast(
    model: nn.Module,
    fixed_set: List[Tuple[Path, Path, str, str]],
    rule,
    device: str,
    patch_size: int,
    overlap: int,
    normalization_method: str,
    psd_use_radial_normalization: bool,
    downsample_factor: float = 1.0,
    min_area_px: float = 50.0,
    fp_area_cap_pct: float = 0.5,
    opening_size: int = 2,
    closing_size: int = 3,
    export_dir: Optional[Path] = None,
    pixel_size_angstrom: float = 1.1,
    pixel_size_per_dataset: Optional[Dict[str, float]] = None,
    # Per-dataset downsampling (for adaptive downsampling match with training)
    downsample_per_dataset: Optional[Dict[str, float]] = None,
    adaptive_downsample_method: str = "zoom",
    # Multi-scale PSD parameters (must match training!)
    psd_multiscale: bool = False,
    psd_multiscale_separate_channels: bool = False,
    psd_scales: tuple = (16, 32, 64),
    psd_frequency_band_channels: bool = False,
    psd_frequency_bands: tuple = (),
    psd_frequency_band_include_full_spectrum: bool = False,
    psd_frequency_band_include_anisotropy: bool = False,
    psd_frequency_band_anisotropy_min_freq: float = 0.0,
    psd_texture_mode: bool = False,
    psd_highpass_cutoff: float = 0.2,
    # Datasets to exclude from metric computation (but still run inference for visual monitoring)
    exclude_datasets_from_metrics: Optional[List[str]] = None,
    # Ablation: whether model uses PSD channels
    use_psd: bool = True,
    # Input composition: if False with use_psd=True, use PSD-only model input
    include_real_space_input: bool = True,
    use_pixel_size_channel: bool = False,
    pixel_size_channel_min_angstrom: float = 0.3,
    pixel_size_channel_max_angstrom: float = 1.4,
    # Batched patch forward for faster GPU inference (mirrors CLI --batch-forward-size)
    batch_forward_size: int = 1,
    # Skip upsample in predict; compare GT at work resolution (faster, metrics are resolution-invariant)
    return_native_resolution: bool = False,
    blending_window: str = "hann",
    blending_edge_px: int = 16,
    threshold_grid: Optional[Tuple[float, ...]] = None,
    metrics_max_samples: int = 5_000_000,
) -> Dict[str, float]:
    """
    Stitched full-micrograph validation: primary signal for checkpoint selection.
    Pixel: PR-AUC, AUROC. Operational: FP area % median, recall at FP cap.
    Logs prob histograms (pos/neg medians). When export_dir is set, writes stitched prob maps as .mrc and .npy,
    plus three-panel figures (.png/.svg): prob map | 4 Å low-pass raw | overlay.
    Uses pixel_size_per_dataset[dataset_id] for the 4 Å filter when provided; else pixel_size_angstrom.

    NEW: exclude_datasets_from_metrics allows running inference on datasets (for visual monitoring)
    while excluding them from metric computation (for early stopping optimization).
    """
    all_probs = []
    all_targets = []
    all_prob_maps = []
    all_gt_masks = []
    all_stems = []  # (dataset_id, stem) for export
    all_raw_imgs = []  # raw micrograph for export panels

    for mic_path, gt_path, dataset_id, stem in fixed_set:
        try:
            # Use per-dataset downsampling if available (matches training adaptive downsampling)
            ds_factor = downsample_factor
            if downsample_per_dataset is not None and dataset_id in downsample_per_dataset:
                ds_factor = downsample_per_dataset[dataset_id]
            px_size = _resolve_dataset_pixel_size(
                dataset_id=dataset_id,
                pixel_size_by_dataset=pixel_size_per_dataset,
                fallback=pixel_size_angstrom,
            )

            use_preprocessed_cache = bool(return_native_resolution) and export_dir is None
            if use_preprocessed_cache:
                img_for_model, region_gt, effective_px_size = _load_preprocessed_full_micrograph(
                    mic_path=mic_path,
                    gt_path=gt_path,
                    dataset_id=dataset_id,
                    rule=rule,
                    normalization_method=normalization_method,
                    downsample_factor=float(max(1.0, ds_factor)),
                    adaptive_downsample_method=adaptive_downsample_method,
                    pixel_size_by_dataset=pixel_size_per_dataset,
                )
                predict_downsample_factor = 1.0
                predict_norm_method = "none"
                img = img_for_model
                px_size_for_model = effective_px_size
            else:
                img = _load_mrc_2d_permissive(mic_path)
                img = np.asarray(img).squeeze().astype(np.float32)
                gt = np.load(str(gt_path))
                gt = np.asarray(gt).squeeze()
                region_gt = apply_region_gt_transform(gt, dataset_id=dataset_id, rule=rule).astype(np.uint8)
                if region_gt.shape != img.shape:
                    zoom_factors = [img.shape[i] / region_gt.shape[i] for i in range(2)]
                    region_gt = ndimage.zoom(region_gt, zoom_factors, order=0)
                predict_downsample_factor = float(max(1.0, ds_factor))
                predict_norm_method = normalization_method
                px_size_for_model = px_size

            prob_map = predict_bad_regions_probability(
                model=model,
                image=img,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
                pixel_size_angstrom=px_size_for_model,
                use_power_spectrum=use_psd,
                include_real_space_input=include_real_space_input,
                use_hann_blending=True,
                blending_window=blending_window,
                blending_edge_px=blending_edge_px,
                downsample_factor=predict_downsample_factor,
                adaptive_downsample_method=adaptive_downsample_method,
                return_native_resolution=return_native_resolution,
                temperature=1.0,
                normalization_method=predict_norm_method,
                psd_use_radial_normalization=psd_use_radial_normalization,
                global_prior_path=None,
                use_global_prior=False,
                batch_forward_size=batch_forward_size,
                use_pixel_size_channel=use_pixel_size_channel,
                pixel_size_channel_min_angstrom=pixel_size_channel_min_angstrom,
                pixel_size_channel_max_angstrom=pixel_size_channel_max_angstrom,
                # Multi-scale PSD parameters (must match training!)
                psd_multiscale=psd_multiscale,
                psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                psd_scales=psd_scales,
                psd_frequency_band_channels=psd_frequency_band_channels,
                psd_frequency_bands=psd_frequency_bands,
                psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
                psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
                psd_texture_mode=psd_texture_mode,
                psd_highpass_cutoff=psd_highpass_cutoff,
            )
            prob_map = np.asarray(prob_map).squeeze()
            if prob_map.shape != region_gt.shape:
                if return_native_resolution:
                    # Downsample GT to work resolution — 4x fewer pixels, metrics are resolution-invariant
                    zoom_factors = [prob_map.shape[i] / region_gt.shape[i] for i in range(2)]
                    region_gt = ndimage.zoom(region_gt, zoom_factors, order=0)
                else:
                    zoom_factors = [region_gt.shape[i] / prob_map.shape[i] for i in range(2)]
                    prob_map = ndimage.zoom(prob_map, zoom_factors, order=1)
            all_probs.append(prob_map.flatten())
            all_targets.append(region_gt.flatten().astype(np.float32))
            all_prob_maps.append(prob_map)
            all_gt_masks.append(region_gt)
            all_stems.append((dataset_id, stem))
            if export_dir is not None:
                all_raw_imgs.append(np.asarray(img).copy())
        except Exception as e:
            warnings.warn(f"Stitched val skip {stem!s}: {e}")

    if len(all_probs) == 0:
        return {"pr_auc": 0.0, "auroc": 0.0, "recall_at_fp_cap": 0.0, "fp_area_pct_median": 1.0, "pred_components_median": 0.0, "junk_pct_error": 1.0, "n_mics": 0, "prob_pos_median": 0.0, "prob_neg_median": 0.0}

    # Filter out excluded datasets for metric computation (but they're already in all_* for export)
    exclude_set = set(exclude_datasets_from_metrics or [])
    if exclude_set:
        print(f"    NOTE: Excluding datasets {sorted(exclude_set)} from metrics (but still generating visualizations)", flush=True)

    # Create filtered versions for metric computation
    metric_indices = [i for i, (dsid, _) in enumerate(all_stems) if dsid not in exclude_set]
    metric_probs = [all_probs[i] for i in metric_indices]
    metric_targets = [all_targets[i] for i in metric_indices]
    metric_prob_maps = [all_prob_maps[i] for i in metric_indices]
    metric_gt_masks = [all_gt_masks[i] for i in metric_indices]

    n_excluded = len(all_probs) - len(metric_probs)
    if n_excluded > 0:
        print(f"    Using {len(metric_probs)} micrographs for metrics ({n_excluded} excluded for visual monitoring only)", flush=True)

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        import mrcfile
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for (dsid, stem), pm, raw_img, gt_mask in zip(all_stems, all_prob_maps, all_raw_imgs, all_gt_masks):
            np.save(export_dir / f"{dsid}_{stem}.npy", pm.astype(np.float32))
            out_stem = f"{dsid}_{stem}"
            with mrcfile.new(str(export_dir / f"{out_stem}_prob.mrc"), overwrite=True) as mrc:
                mrc.set_data(pm.astype(np.float32))
                mrc.header.mode = 2
            # Use per-dataset pixel size if available, else fallback
            px_size = pixel_size_per_dataset.get(dsid, pixel_size_angstrom) if pixel_size_per_dataset else pixel_size_angstrom
            lowpass = apply_lowpass_filter(raw_img.astype(np.float64), cutoff_angstrom=4.0, pixel_size_angstrom=px_size)
            # Ensure GT mask matches image size
            if gt_mask.shape != lowpass.shape:
                from scipy.ndimage import zoom
                zoom_factors = [lowpass.shape[i] / gt_mask.shape[i] for i in range(2)]
                gt_mask = zoom(gt_mask, zoom_factors, order=0)
            gt_binary = (gt_mask > 0.5).astype(np.float32)
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(pm, cmap="viridis", vmin=0, vmax=1)
            axes[0].set_title("Probability map")
            axes[0].axis("off")
            axes[1].imshow(lowpass, cmap="gray")
            axes[1].set_title("4 Å low-pass")
            axes[1].axis("off")
            axes[2].imshow(lowpass, cmap="gray")
            axes[2].imshow(pm, cmap="magma", alpha=0.6, vmin=0, vmax=1)
            axes[2].set_title("Pred overlay")
            axes[2].axis("off")
            axes[3].imshow(lowpass, cmap="gray")
            axes[3].imshow(gt_binary, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
            axes[3].set_title("GT overlay")
            axes[3].axis("off")
            plt.tight_layout()
            fig.savefig(export_dir / f"{out_stem}_panel.png", dpi=150, bbox_inches="tight")
            fig.savefig(export_dir / f"{out_stem}_panel.svg", bbox_inches="tight")
            plt.close(fig)

            # === Individual high-res SVG panels (separate files) ===
            indiv_dir = export_dir / "individual_plots"
            indiv_dir.mkdir(parents=True, exist_ok=True)

            panel_configs = [
                ("probability", lambda ax_: ax_.imshow(pm, cmap="viridis", vmin=0, vmax=1)),
                ("lowpass", lambda ax_: ax_.imshow(lowpass, cmap="gray")),
                ("pred_overlay", lambda ax_: (ax_.imshow(lowpass, cmap="gray"), ax_.imshow(pm, cmap="magma", alpha=0.6, vmin=0, vmax=1))),
                ("gt_overlay", lambda ax_: (ax_.imshow(lowpass, cmap="gray"), ax_.imshow(gt_binary, cmap="Reds", alpha=0.5, vmin=0, vmax=1))),
            ]
            for panel_name, draw_fn in panel_configs:
                fig_i, ax_i = plt.subplots(1, 1, figsize=(8, 8))
                draw_fn(ax_i)
                ax_i.axis("off")
                fig_i.savefig(
                    indiv_dir / f"{out_stem}_{panel_name}.svg",
                    dpi=500, bbox_inches="tight", pad_inches=0,
                )
                plt.close(fig_i)

            del fig, axes, lowpass, gt_binary  # Explicit cleanup
        plt.close("all")
        gc.collect()  # Force garbage collection to release matplotlib file handles

    print(f"    Computing metrics on {len(metric_probs)} micrographs...", flush=True)
    probs_flat = np.concatenate(metric_probs)
    targets_flat = np.concatenate(metric_targets)
    targets_binary = (targets_flat >= 0.5).astype(int)
    n_pixels = len(probs_flat)
    print(f"    Total pixels: {n_pixels:,} ({n_pixels/1e6:.1f}M)", flush=True)

    # Prob histograms for calibration (pos/neg pixel medians)
    pos_mask = targets_binary == 1
    neg_mask = targets_binary == 0
    prob_pos_median = float(np.median(probs_flat[pos_mask])) if pos_mask.any() else 0.0
    prob_neg_median = float(np.median(probs_flat[neg_mask])) if neg_mask.any() else 0.0

    # For very large arrays, subsample to speed up metric computation
    # PR-AUC and AUROC are stable with ~5M samples
    MAX_SAMPLES_FOR_METRICS = max(1, int(metrics_max_samples))
    if n_pixels > MAX_SAMPLES_FOR_METRICS:
        print(f"    Subsampling to {MAX_SAMPLES_FOR_METRICS:,} pixels for metric computation...", flush=True)
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n_pixels, size=MAX_SAMPLES_FOR_METRICS, replace=False)
        probs_sub = probs_flat[idx]
        targets_sub = targets_binary[idx]
    else:
        probs_sub = probs_flat
        targets_sub = targets_binary

    print(f"    Computing PR-AUC...", flush=True)
    pr_auc = compute_pr_auc(probs_sub, targets_sub)
    print(f"    Computing AUROC...", flush=True)
    auroc = 0.0
    if roc_auc_score is not None and targets_sub.sum() > 0 and (targets_sub == 0).sum() > 0:
        try:
            auroc = float(roc_auc_score(targets_sub, probs_sub))
        except Exception:
            pass
    print(f"    PR-AUC={pr_auc:.4f}, AUROC={auroc:.4f}", flush=True)

    min_area = max(1, int(min_area_px))
    fp_area_cap = fp_area_cap_pct / 100.0

    # Operational metrics at default threshold 0.5
    print(f"    Computing operational metrics...", flush=True)
    fp_areas_05 = []
    pred_comp_counts = []
    junk_errors = []
    for i, (prob_map, gt_mask) in enumerate(zip(metric_prob_maps, metric_gt_masks)):
        pred = (prob_map >= 0.5).astype(np.uint8)
        pred_bin, n_comp = postprocess_binary_mask(pred, min_area, opening_size=opening_size, closing_size=closing_size)
        pred_comp_counts.append(n_comp)
        gt_bin = (gt_mask >= 0.5).astype(np.uint8)
        fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
        tn = ((pred_bin == 0) & (gt_bin == 0)).sum()
        fp_area_pct = float(fp) / (fp + tn + 1e-10)
        fp_areas_05.append(fp_area_pct)
        gt_junk_pct = gt_bin.sum() / max(1, gt_bin.size)
        pred_junk_pct = pred_bin.sum() / max(1, pred_bin.size)
        junk_errors.append(abs(pred_junk_pct - gt_junk_pct))
    fp_area_median = float(np.median(fp_areas_05)) if fp_areas_05 else 1.0
    pred_components_median = float(np.median(pred_comp_counts)) if pred_comp_counts else 0.0
    junk_pct_error = float(np.mean(junk_errors)) if junk_errors else 1.0

    # Pre-compute postprocessed predictions for each threshold ONCE (major speedup)
    print(f"    Computing threshold sweep (caching postprocessed predictions)...", flush=True)
    if threshold_grid is None:
        thr_grid = np.linspace(0.1, 0.9, 9)
    else:
        thr_values = [float(t) for t in threshold_grid if np.isfinite(float(t))]
        thr_values = [min(0.999, max(0.001, t)) for t in thr_values]
        thr_grid = np.asarray(sorted(set(thr_values)), dtype=np.float32)
        if thr_grid.size == 0:
            thr_grid = np.linspace(0.1, 0.9, 9)
    print(f"    Threshold grid: {', '.join(f'{float(t):.2f}' for t in thr_grid)}", flush=True)
    cached_preds = {}  # thr -> list of (pred_bin, gt_bin, fp_area, recall) tuples
    for thr in thr_grid:
        preds_at_thr = []
        for prob_map, gt_mask in zip(metric_prob_maps, metric_gt_masks):
            pred = (prob_map >= thr).astype(np.uint8)
            pred_bin, _ = postprocess_binary_mask(pred, min_area, opening_size=opening_size, closing_size=closing_size)
            gt_bin = (gt_mask >= 0.5).astype(np.uint8)
            fp = ((pred_bin > 0) & (gt_bin == 0)).sum()
            tn = ((pred_bin == 0) & (gt_bin == 0)).sum()
            fn = ((pred_bin == 0) & (gt_bin == 1)).sum()
            tp = ((pred_bin > 0) & (gt_bin == 1)).sum()
            fp_area = float(fp) / (fp + tn + 1e-10)
            recall = float(tp) / (tp + fn + 1e-10)
            preds_at_thr.append((pred_bin, gt_bin, fp_area, recall))
        cached_preds[thr] = preds_at_thr

    # Recall at fixed FP area: choose t so median FP % <= cap
    best_recall_at_fp = 0.0
    for thr in thr_grid:
        fp_areas = [x[2] for x in cached_preds[thr]]
        recalls = [x[3] for x in cached_preds[thr]]
        fp_med = float(np.median(fp_areas))
        if fp_med <= fp_area_cap and len(recalls) > 0:
            rec_at = float(np.mean(recalls))
            if rec_at > best_recall_at_fp:
                best_recall_at_fp = rec_at

    # Compute recall at multiple FP caps for early stopping (0.5%, 1%, 2%, 5%, 15%)
    # Also track OPTIMAL THRESHOLD for each FP cap
    fp_caps_multi = [0.005, 0.01, 0.02, 0.05, 0.15]  # 0.5%, 1%, 2%, 5%, 15%
    recall_at_multi_fp_caps = {}
    optimal_thresholds = {}  # Store optimal threshold for each FP cap

    for fp_cap in fp_caps_multi:
        best_recall = 0.0
        best_thr = 0.5  # Default
        for thr in thr_grid:
            # Use cached fp_area and recall values (already computed)
            fp_areas = [x[2] for x in cached_preds[thr]]
            recalls = [x[3] for x in cached_preds[thr]]
            fp_med = float(np.median(fp_areas))
            if fp_med <= fp_cap and len(recalls) > 0:
                rec_at = float(np.mean(recalls))
                if rec_at > best_recall:
                    best_recall = rec_at
                    best_thr = thr
        recall_at_multi_fp_caps[f"recall_at_fp_{fp_cap*100:.1f}pct"] = best_recall
        optimal_thresholds[f"optimal_thr_at_fp_{fp_cap*100:.1f}pct"] = float(best_thr)

    # Find the RECOMMENDED threshold (maximize recall at 2% FP, which is a good balance)
    recommended_threshold = optimal_thresholds.get("optimal_thr_at_fp_2.0pct", 0.5)

    print(f"    Done with stitched validation metrics.", flush=True)
    print(f"    Optimal thresholds: " + ", ".join([f"FP{k.split('_')[-1]}={v:.2f}" for k, v in sorted(optimal_thresholds.items())]), flush=True)
    print(f"    → RECOMMENDED threshold: {recommended_threshold:.2f} (optimizes Recall@FP≤2%)", flush=True)

    return {
        "pr_auc": pr_auc,
        "auroc": auroc,
        "recall_at_fp_cap": best_recall_at_fp,
        "fp_area_pct_median": fp_area_median,
        "pred_components_median": pred_components_median,
        "junk_pct_error": junk_pct_error,
        "n_mics": len(all_prob_maps),
        "prob_pos_median": prob_pos_median,
        "prob_neg_median": prob_neg_median,
        "recommended_threshold": recommended_threshold,
        **recall_at_multi_fp_caps,  # Add multi-FP-cap recalls
        **optimal_thresholds,  # Add optimal thresholds for each FP cap
    }


def train_vit_model(args):
    """
    Train a Vision Transformer model for contamination detection.

    This function handles the different data pipeline and training loop
    required for ViT, which processes full micrographs instead of random patches.
    """
    from models.bad_region_detector import create_vit_model

    print("\n" + "="*80)
    print("VISION TRANSFORMER TRAINING MODE")
    print("="*80)
    print("Processing full micrographs with self-attention across all patches")
    print(f"  ViT depth: {args.vit_depth} layers")
    print(f"  ViT heads: {args.vit_heads}")
    print(f"  ViT embed dim: {args.vit_embed_dim}")
    print(f"  ViT dropout: {args.vit_dropout}")
    print("="*80 + "\n")

    # Set up output directory
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = Path(args.manifest).expanduser().resolve()
    df = pd.read_csv(manifest_path)
    df = _resolve_manifest_relative_paths(df, manifest_path)
    mic_dir = Path(args.mic_dir)

    # Dataset-stratified train/val split (no data leakage between datasets)
    # All micrographs from a given EMPIAR dataset go entirely to train OR val
    np.random.seed(args.seed)
    unique_datasets = df['dataset_id'].unique()
    np.random.shuffle(unique_datasets)

    n_train_datasets = int(len(unique_datasets) * args.train_frac)
    train_datasets = set(unique_datasets[:n_train_datasets])
    val_datasets = set(unique_datasets[n_train_datasets:])

    train_df = df[df['dataset_id'].isin(train_datasets)].reset_index(drop=True)
    val_df = df[df['dataset_id'].isin(val_datasets)].reset_index(drop=True)

    print(f"Dataset-stratified split: {len(train_datasets)} train datasets, {len(val_datasets)} val datasets")
    print(f"  Train: {len(train_df)} micrographs from datasets {sorted(train_datasets)}")
    print(f"  Val: {len(val_df)} micrographs from datasets {sorted(val_datasets)}")

    # Compute downsample factors (uniform or adaptive)
    if getattr(args, 'adaptive_downsample', False):
        print(f"\n=== ADAPTIVE DOWNSAMPLING ENABLED (ViT) ===", flush=True)
        print(f"  Target pixel size: {args.target_pixel_size} Å/px", flush=True)

        downsample_factors = compute_adaptive_downsample_factors(
            df,
            target_pixel_size=args.target_pixel_size,
            min_downsample=getattr(args, 'min_downsample', 1.0),
            max_downsample=getattr(args, 'max_downsample', 20.0),
            default_pixel_size=1.2,
        )

        # Print per-dataset factors
        print(f"  Per-dataset downsample factors:", flush=True)
        for dsid in sorted(downsample_factors.keys(), key=lambda x: int(x)):
            factor = downsample_factors[dsid]
            if 'pixel_size_angstrom' in df.columns:
                subset = df[df['dataset_id'] == int(dsid)]
                if len(subset) > 0:
                    px_size = subset['pixel_size_angstrom'].iloc[0]
                    eff_px = px_size * factor
                    print(f"    {dsid}: {factor:.2f}x ({px_size:.2f} → {eff_px:.2f} Å/px)", flush=True)

        # Use median factor for effective size calculations
        median_factor = float(np.median(list(downsample_factors.values())))
        print(f"  Median factor: {median_factor:.2f}x", flush=True)
        downsample_factor = median_factor
        cache_downsample = downsample_factors
    else:
        downsample_factors = None
        downsample_factor = float(args.downsample)
        cache_downsample = downsample_factor

    # Compute region GT rules from ALL data (train+val) so validation datasets have valid thresholds
    # Without this, validation datasets return inf min_area and GT becomes all zeros
    effective_min_area = float(getattr(args, 'min_area_floor_px', 100)) / (downsample_factor * downsample_factor)
    all_rows = [(str(r["dataset_id"]), str(r["gt_mask_path"])) for _, r in df.iterrows()]  # Use full df, not just train
    rule = compute_dataset_min_area_px_from_train_split(
        all_rows,
        min_area_floor_px=effective_min_area,
        fill_holes=bool(getattr(args, "region_gt_fill_holes", True)),
    )
    save_dataset_region_rule(rule, out_path)

    # Build/check cache (same as regular training)
    cache_train = out_path / "patch_cache" / "train"
    cache_val = out_path / "patch_cache" / "val"

    # Check if we need to build cache
    train_cache_complete = cache_train.exists() and len(list(cache_train.glob("mic_*.npz"))) >= len(train_df)
    if not train_cache_complete:
        print(f"Building train cache (need all)...")
        build_patch_cache(
            train_df, mic_dir, rule,
            normalization_method=args.normalization_method,
            psd_use_radial_normalization=args.psd_use_radial_normalization,
            downsample_factor=cache_downsample,
            cache_dir=cache_train,
            desc="train",
            psd_use_radial_image=True,  # Always use radial for ViT
            psd_radial_bins=64,
            psd_multiscale=args.psd_multiscale,
            psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
            psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
            psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
            psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
            psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
            psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
            psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
            pixel_size_by_dataset=pixel_size_by_dataset,
            psd_texture_mode=args.psd_texture_mode,
            psd_highpass_cutoff=args.psd_highpass_cutoff,
            adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "zoom"),
        )

    val_cache_complete = cache_val.exists() and len(list(cache_val.glob("mic_*.npz"))) >= len(val_df)
    if not val_cache_complete:
        print("Building val cache...")
        build_patch_cache(
            val_df, mic_dir, rule,
            normalization_method=args.normalization_method,
            psd_use_radial_normalization=args.psd_use_radial_normalization,
            downsample_factor=cache_downsample,
            cache_dir=cache_val,
            desc="val",
            psd_use_radial_image=True,
            psd_radial_bins=64,
            psd_multiscale=args.psd_multiscale,
            psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
            psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
            psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
            psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
            psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
            psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
            psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
            pixel_size_by_dataset=pixel_size_by_dataset,
            psd_texture_mode=args.psd_texture_mode,
            psd_highpass_cutoff=args.psd_highpass_cutoff,
            adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "zoom"),
        )

    # Load particle annotations if provided
    particles_dict = {}
    if args.particle_annotations:
        # Try multiple paths: direct path, relative to cwd, relative to mic_dir parent
        paths_to_try = [
            Path(args.particle_annotations),  # Direct path (absolute or relative to cwd)
            mic_dir.parent / args.particle_annotations,  # Relative to mic_dir parent
        ]
        for particles_path in paths_to_try:
            if particles_path.exists():
                particles_dict = load_particle_annotations(str(particles_path))
                print(f"  Loaded particle annotations from: {particles_path}")
                print(f"  Total micrographs with particles: {len(particles_dict)}")

                # Diagnostic: count how many train/val micrographs match
                matched_train = 0
                matched_val = 0
                for _, r in train_df.iterrows():
                    keys = [f"{r['dataset_id']}__{r['stem']}", r['stem'], f"{r['dataset_id']}_{r['stem']}"]
                    if any(k in particles_dict for k in keys):
                        matched_train += 1
                for _, r in val_df.iterrows():
                    keys = [f"{r['dataset_id']}__{r['stem']}", r['stem'], f"{r['dataset_id']}_{r['stem']}"]
                    if any(k in particles_dict for k in keys):
                        matched_val += 1
                print(f"  Matched particles: {matched_train}/{len(train_df)} train, {matched_val}/{len(val_df)} val")
                break
        else:
            print(f"  WARNING: particle_annotations not found at any of: {paths_to_try}")

    # Create ViT model
    print("\nCreating ViT model...")
    model = create_vit_model(
        embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        vit_dropout=args.vit_dropout,
        input_channels=getattr(args, "resolved_input_channels", 2),
        num_classes=1,
        device=str(args.device),
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Create datasets
    # Compute effective patch size and stride (use working_* if specified for adaptive mode)
    if getattr(args, 'working_patch_size', None) is not None:
        effective_patch_size = int(args.working_patch_size)
        print(f"  Using working_patch_size={effective_patch_size} (directly specified)")
    else:
        effective_patch_size = int(args.patch_size / downsample_factor)

    if getattr(args, 'working_stride', None) is not None:
        effective_stride = int(args.working_stride)
        print(f"  Using working_stride={effective_stride} (directly specified)")
    elif args.patch_stride is not None:
        effective_stride = int(args.patch_stride / downsample_factor)
    else:
        effective_stride = None

    train_ds = ViTMicrographDataset(
        df=train_df,
        mic_dir=mic_dir,
        rule=rule,
        normalization_method=args.normalization_method,
        psd_use_radial_normalization=args.psd_use_radial_normalization,
        patch_size=effective_patch_size,
        downsample_factor=downsample_factor,
        seed=args.seed,
        use_augmentation=True,
        cache_dir=cache_train,
        use_psd_input=getattr(args, "use_psd_bool", True),
        include_real_space_input=getattr(args, "include_real_space_input", True),
        psd_multiscale=args.psd_multiscale,
        psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
        psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
        particles_dict=particles_dict,
        particle_box_size_px=args.particle_box_size_px,
        use_noise_augmentation=args.use_noise_augmentation,
        noise_aug_prob=args.noise_aug_prob,
        noise_strength=0.0,  # Start at 0, will be updated by curriculum at each epoch
        stride=effective_stride,  # Overlapping patches if stride < patch_size
    )

    val_ds = ViTMicrographDataset(
        df=val_df,
        mic_dir=mic_dir,
        rule=rule,
        normalization_method=args.normalization_method,
        psd_use_radial_normalization=args.psd_use_radial_normalization,
        patch_size=effective_patch_size,
        downsample_factor=downsample_factor,
        seed=args.seed,
        use_augmentation=False,
        cache_dir=cache_val,
        use_psd_input=getattr(args, "use_psd_bool", True),
        include_real_space_input=getattr(args, "include_real_space_input", True),
        psd_multiscale=args.psd_multiscale,
        psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
        particles_dict=particles_dict,
        particle_box_size_px=args.particle_box_size_px,
        stride=effective_stride,  # Overlapping patches for smooth validation
    )

    print(f"\nViT Training setup:")
    print(f"  Train micrographs: {len(train_ds)}")
    print(f"  Val micrographs: {len(val_ds)}")
    print(f"  Effective patch size: {effective_patch_size}")
    actual_stride = train_ds.stride
    if actual_stride < effective_patch_size:
        overlap_pct = 100 * (1 - actual_stride / effective_patch_size)
        print(f"  Effective stride: {actual_stride} ({overlap_pct:.0f}% overlap)")
    else:
        print(f"  Effective stride: {actual_stride} (no overlap)")
    print(f"  Batch size: {args.batch_size} micrographs (each contains many patches)")

    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=vit_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),  # Smaller batch for val (memory)
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vit_collate_fn,
        pin_memory=True,
    )

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    # Training history for per-term logging
    training_history = []

    # Log noise augmentation settings (with curriculum info)
    if args.use_noise_augmentation:
        print(f"\n=== NOISE AUGMENTATION WITH STAGED CURRICULUM ===")
        print(f"  Purpose: Teach model that noisy/textured clean images are NOT contamination")
        print(f"  Probability: {args.noise_aug_prob:.0%} of micrographs will have noise added to clean patches")
        print(f"  Start epoch: {args.noise_start_epoch}")
        if args.noise_ramp_epochs > 0:
            print(f"  Ramp: epochs {args.noise_start_epoch}-{args.noise_start_epoch + args.noise_ramp_epochs} (0 → {args.noise_max_strength:.2f})")
        else:
            print(f"  Strength: {args.noise_max_strength:.2f} (no ramp)")

    # Log LR schedule
    if args.lr_drop_epoch_1 > 0:
        print(f"\n=== STAGED LEARNING RATE SCHEDULE ===")
        print(f"  Initial LR: {args.lr}")
        print(f"  Drop 1: epoch {args.lr_drop_epoch_1} → LR/{args.lr_drop_factor_1:.1f}")
        if args.lr_drop_epoch_2 > 0:
            print(f"  Drop 2: epoch {args.lr_drop_epoch_2} → LR/{args.lr_drop_factor_2:.1f}")

    # Track LR drops applied
    lr_drops_applied = set()

    if args.use_stable_loss:
        print(f"\n=== STABLE LOSS WITH PHASED WARMUP ===")
        print(f"  Phase 1 (BCE+Dice only): epochs 1-{args.warmup_phase1_epochs}")
        print(f"  Phase 2 (+ hit-core): epochs {args.warmup_phase1_epochs+1}-{args.warmup_phase2_epochs}")
        print(f"  Phase 3 (+ particle suppression): epochs {args.warmup_phase2_epochs+1}-{args.warmup_phase3_epochs}")
        print(f"  Phase 4 (+ area budget): epochs {args.warmup_phase3_epochs+1}+")
        print(f"  Lambda values: BCE={args.lambda_bce}, Dice={args.lambda_dice}, "
              f"hit_core={args.stable_lambda_hit_core}, particle={args.stable_lambda_particle}, "
              f"area={args.stable_lambda_area}")

    print(f"\nStarting ViT training for {args.num_epochs} epochs...")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # =======================================================================
        # STAGED CURRICULUM: LR drops and noise ramping
        # =======================================================================

        # LR drops (apply once when epoch is reached)
        if args.lr_drop_epoch_1 > 0 and epoch == args.lr_drop_epoch_1 and 1 not in lr_drops_applied:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] / args.lr_drop_factor_1
            lr_drops_applied.add(1)
            print(f"  [LR DROP 1] Reduced LR by {args.lr_drop_factor_1:.1f}x → {opt.param_groups[0]['lr']:.2e}")

        if args.lr_drop_epoch_2 > 0 and epoch == args.lr_drop_epoch_2 and 2 not in lr_drops_applied:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] / args.lr_drop_factor_2
            lr_drops_applied.add(2)
            print(f"  [LR DROP 2] Reduced LR by {args.lr_drop_factor_2:.1f}x → {opt.param_groups[0]['lr']:.2e}")

        # Noise strength curriculum
        if args.use_noise_augmentation:
            if epoch < args.noise_start_epoch:
                # Before noise starts: strength = 0
                noise_strength = 0.0
            elif args.noise_ramp_epochs > 0:
                # Ramping period: linear increase from 0 to max_strength
                ramp_progress = min(1.0, (epoch - args.noise_start_epoch) / args.noise_ramp_epochs)
                noise_strength = args.noise_max_strength * ramp_progress
            else:
                # No ramp: instant full strength
                noise_strength = args.noise_max_strength

            # Update dataset's noise strength
            train_ds.set_noise_strength(noise_strength)

        # Per-term loss tracking for this epoch
        epoch_loss_terms = {
            'loss_bce': [], 'loss_dice': [], 'loss_hit_core': [],
            'loss_particle_anchor': [], 'loss_area': []
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.num_epochs}", ncols=100)
        for batch in pbar:
            patches = batch['patches'].to(args.device)  # [B, N, 2, H, W]
            labels = batch['labels'].to(args.device)  # [B, N, H, W]
            particle_masks = batch['particle_masks'].to(args.device)  # [B, N, H, W]
            mask = batch['mask'].to(args.device)  # [B, N] valid patch mask
            grid_shapes = batch['grid_shapes']

            opt.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Get predictions for first micrograph (varying sizes not supported in batch)
                # TODO: Handle variable grid sizes properly
                B = patches.shape[0]
                all_losses = []

                for b in range(B):
                    n_patches = batch['num_patches'][b].item()
                    mic_patches = patches[b, :n_patches].unsqueeze(0)  # [1, N, 2, H, W]
                    mic_labels = labels[b, :n_patches].unsqueeze(0)  # [1, N, H, W]
                    mic_particles = particle_masks[b, :n_patches].unsqueeze(0)
                    grid_shape = grid_shapes[b]

                    # Forward
                    preds = model(mic_patches, grid_shape=grid_shape)  # [1, N, 1, out_H, out_W]

                    # Resize labels to match prediction size
                    _, _, _, out_H, out_W = preds.shape

                    # Reshape labels for batch resize: [1, N, H, W] -> [N, 1, H, W]
                    labels_for_resize = mic_labels[0].unsqueeze(1).float()  # [N, 1, H, W]
                    labels_resized = F.interpolate(labels_for_resize, size=(out_H, out_W), mode='nearest')
                    mic_labels_resized = labels_resized.squeeze(1).unsqueeze(0)  # [1, N, out_H, out_W]

                    # Resize particle masks
                    particles_for_resize = mic_particles[0].unsqueeze(1).float()  # [N, 1, H, W]
                    particles_resized = F.interpolate(particles_for_resize, size=(out_H, out_W), mode='nearest')
                    mic_particles_resized = particles_resized.squeeze(1).unsqueeze(0)  # [1, N, out_H, out_W]

                    preds_flat = preds.squeeze(2)  # [1, N, out_H, out_W]

                    if args.use_stable_loss:
                        # Stable loss with phased warmup
                        # Reshape for loss: [1, N, H, W] -> [N, 1, H, W]
                        logits_for_loss = preds_flat[0].unsqueeze(1)  # [N, 1, out_H, out_W]
                        gt_for_loss = mic_labels_resized[0].unsqueeze(1)  # [N, 1, out_H, out_W]
                        particles_for_loss = mic_particles_resized[0]  # [N, out_H, out_W]

                        # Compute gt_core (eroded GT)
                        gt_core = precompute_gt_cores(mic_labels_resized[0], erosion_px=max(1, 4 // 2))

                        loss, loss_stats = compute_stable_loss(
                            logits_for_loss, gt_for_loss, gt_core,
                            good_anchor_mask=particles_for_loss,
                            epoch=epoch,
                            lambda_bce=args.lambda_bce,
                            lambda_dice=args.lambda_dice,
                            bce_neg_weight=args.bce_neg_weight,
                            lambda_hit_core=args.stable_lambda_hit_core,
                            warmup_phase1_epochs=args.warmup_phase1_epochs,
                            lambda_particle_anchor=args.stable_lambda_particle,
                            warmup_phase2_epochs=args.warmup_phase2_epochs,
                            tau_anchor=args.stable_tau_anchor,
                            lambda_area=args.stable_lambda_area,
                            warmup_phase3_epochs=args.warmup_phase3_epochs,
                            rho_area=args.stable_rho_area,
                        )
                        # Track per-term losses
                        for term in ['loss_bce', 'loss_dice', 'loss_hit_core', 'loss_particle_anchor', 'loss_area']:
                            if term in loss_stats:
                                epoch_loss_terms[term].append(loss_stats[term])
                    else:
                        # Simple BCE loss (original)
                        loss = F.binary_cross_entropy_with_logits(
                            preds_flat,
                            mic_labels_resized,
                            reduction='mean'
                        )
                    all_losses.append(loss)

                batch_loss = torch.stack(all_losses).mean()

            # Backward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            epoch_loss += batch_loss.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{batch_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                patches = batch['patches'].to(args.device)
                labels = batch['labels'].to(args.device)
                grid_shapes = batch['grid_shapes']

                B = patches.shape[0]
                for b in range(B):
                    n_patches = batch['num_patches'][b].item()
                    mic_patches = patches[b, :n_patches].unsqueeze(0)
                    mic_labels = labels[b, :n_patches].unsqueeze(0)
                    grid_shape = grid_shapes[b]

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        preds = model(mic_patches, grid_shape=grid_shape)
                        _, _, _, out_H, out_W = preds.shape

                        # Resize labels: [1, N, H, W] -> [N, 1, H, W] -> resize -> [1, N, out_H, out_W]
                        labels_for_resize = mic_labels[0].unsqueeze(1).float()
                        labels_resized = F.interpolate(labels_for_resize, size=(out_H, out_W), mode='nearest')
                        mic_labels_resized = labels_resized.squeeze(1).unsqueeze(0)

                        loss = F.binary_cross_entropy_with_logits(
                            preds.squeeze(2),
                            mic_labels_resized,
                            reduction='mean'
                        )
                    val_loss += loss.item()
                    n_val += 1

        avg_val_loss = val_loss / max(1, n_val)

        # Compute average per-term losses for this epoch
        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }

        if args.use_stable_loss:
            term_str_parts = []
            for term, values in epoch_loss_terms.items():
                if values:
                    avg_term = np.mean(values)
                    epoch_record[term] = avg_term
                    short_name = term.replace('loss_', '')
                    term_str_parts.append(f"{short_name}={avg_term:.4f}")

            term_str = ", ".join(term_str_parts)
            print(f"\nEpoch {epoch}: Train={avg_loss:.4f}, Val={avg_val_loss:.4f}")
            print(f"  Loss terms: {term_str}")

            # Print active phase
            if epoch <= args.warmup_phase1_epochs:
                print(f"  Phase 1: BCE+Dice only")
            elif epoch <= args.warmup_phase2_epochs:
                print(f"  Phase 2: + hit-core (warmup)")
            elif epoch <= args.warmup_phase3_epochs:
                print(f"  Phase 3: + particle suppression (warmup)")
            else:
                print(f"  Phase 4: all terms active")
        else:
            print(f"\nEpoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Print curriculum status (LR + noise)
        current_lr = opt.param_groups[0]['lr']
        if args.use_noise_augmentation:
            print(f"  Curriculum: LR={current_lr:.2e}, noise_strength={noise_strength:.3f}")

        training_history.append(epoch_record)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), out_path / "best_model.pth")
            print(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
            }, out_path / f"checkpoint_epoch_{epoch:03d}.pth")

        # Save training history every epoch
        with open(out_path / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)

    print(f"\nTraining complete! Best model at epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Model saved to: {out_path / 'best_model.pth'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mic_dir", required=True)
    ap.add_argument("--pretrained_model", default=None, help="Refiner checkpoint; or this run's best_model.pth to resume from best")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--use_two_branch_encoder", action="store_true", help="Use two-branch MIFFI (real + PSD branches). Required when loading a two-branch checkpoint.")
    ap.add_argument("--train_from_scratch", action="store_true",
                    help="Train from scratch using MIFFI pretrained weights (real-space branch) + random init (PSD branch)")
    ap.add_argument("--model_type", type=str, default="miffi",
                    choices=["miffi", "simple", "unet", "unet_attention", "resnet_unet"],
                    help="Model architecture: miffi (default), unet_attention (CNN+bottleneck attention), etc.")
    ap.add_argument("--norm_type", type=str, default="group",
                    choices=["batch", "group", "instance"],
                    help="Normalization type for U-Net. CRITICAL: Use 'group' (default) to prevent train/test "
                         "distribution mismatch! BatchNorm learns statistics from gt_bias-sampled batches, "
                         "causing predictions to drift at inference on full images. GroupNorm normalizes "
                         "within each sample independently - no drift!")
    ap.add_argument("--decoder_dropout", type=float, default=0.0,
                    help="Dropout rate for decoder conv blocks (0.0-0.3). IMPORTANT for preventing overfitting "
                         "on long training runs (100+ epochs). Recommended: 0.2 for 100 epochs, 0.3 for 200+ epochs. "
                         "Prevents probability saturation (1.0/0.0) and particle false positives.")
    ap.add_argument("--attention_type", type=str, default="global",
                    choices=["global", "dual", "none"],
                    help="Attention type for unet_attention model. 'global' (default): single global attention. "
                         "'dual': local + global attention - ELIMINATES EDGE FPs by using windowed local attention "
                         "(immune to edge artifacts) + masked global attention (outer 15%% of positions masked out). "
                         "'none': no attention (conv-only bottleneck for ablation study). "
                         "Dual mode adds ~40%% compute at bottleneck (negligible overall).")
    ap.add_argument("--use_psd", type=str, default="true", choices=["true", "false"],
                    help="Whether to use PSD channels in the model input. 'true' (default): image+PSD input "
                         "(fused PSD by default, or multi-channel with --psd_multiscale_separate_channels). "
                         "'false': 1-channel input (real-space image only, "
                         "for ablation study). When false, PSD is still cached but not fed to the model.")
    ap.add_argument("--psd_only", action="store_true", default=False,
                    help="Use PSD-only input channels (drop real-space image channel). "
                         "Requires --use_psd=true.")
    ap.add_argument("--dual_window_size", type=int, default=8,
                    help="Window size for local attention in dual mode (default 8). Must divide bottleneck spatial size.")
    ap.add_argument("--dual_edge_mask", type=float, default=0.15,
                    help="Fraction of edges to mask in global attention (default 0.15). "
                         "Outer 15%% of positions cannot attend to or be attended from.")
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing for soft targets (0.0-0.1). Prevents overconfident predictions. "
                         "Recommended: 0.05-0.1 for long training runs to prevent probability saturation.")
    ap.add_argument("--no_fullmic_augmentation", action="store_true", default=False,
                    help="Disable flip+rot90 augmentation in full-mic training mode. "
                         "By default, random H/V flips and 90-degree rotations are applied per batch "
                         "for 8x training diversity (CRITICAL for preventing overfitting).")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_dataset_ids", type=str, default="",
                    help="Optional comma-separated dataset IDs to force into the training split.")
    ap.add_argument("--val_dataset_ids", type=str, default="",
                    help="Optional comma-separated dataset IDs to force into the validation split.")

    ap.add_argument("--patch_size", type=int, default=256,
                    help="Patch size in full-resolution pixels (before downsampling). "
                         "With uniform downsampling, effective size = patch_size / downsample_factor.")
    ap.add_argument("--working_patch_size", type=int, default=None,
                    help="Patch size at WORKING resolution (after downsampling). "
                         "When set, this overrides patch_size/downsample computation. "
                         "REQUIRED for adaptive downsampling to ensure consistent sizes across datasets. "
                         "Typical values: 64 (small, fast) or 128 (more context). "
                         "Must be divisible by 16 for U-Net compatibility.")
    ap.add_argument("--patch_stride", type=int, default=None,
                    help="Stride for patch extraction in full-res pixels. Default=patch_size (no overlap). "
                         "Set to patch_size//2 for 50%% overlap to eliminate blocky artifacts.")
    ap.add_argument("--working_stride", type=int, default=None,
                    help="Stride at WORKING resolution (after downsampling). "
                         "When set with --working_patch_size, use this for consistent overlap. "
                         "Example: --working_patch_size 64 --working_stride 32 for 50%% overlap.")
    ap.add_argument("--patches_per_mic", type=int, default=16, help="Number of patches to sample per micrograph per epoch (with GT-bias)")
    ap.add_argument("--gt_bias", type=float, default=0.5, help="Fraction of patches that contain GT (GT-biased sampling)")

    # GT-BIAS DECAY: Critical for preventing "predict contamination everywhere" collapse
    # Problem: If every batch has X% contamination patches, model learns X% as the prior
    # Solution: Start high (learn features) → decay to 0 (learn true distribution)
    ap.add_argument("--gt_bias_decay", action="store_true",
                    help="Enable GT-bias decay: start high to learn features, decay to 0 to learn true distribution. "
                         "CRITICAL for preventing 'predict everything as contamination' collapse!")
    ap.add_argument("--gt_bias_decay_start", type=int, default=10,
                    help="Epoch to START decaying GT-bias (default: 10)")
    ap.add_argument("--gt_bias_decay_end", type=int, default=20,
                    help="Epoch to END decay (GT-bias reaches gt_bias_final, default: 20)")
    ap.add_argument("--gt_bias_final", type=float, default=0.0,
                    help="Final GT-bias after decay (default: 0.0 = pure random sampling)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--downsample", type=float, default=2.0,
                    help="Uniform downsample factor (ignored if --adaptive_downsample is used)")

    # Adaptive downsampling to achieve consistent physical resolution across datasets
    ap.add_argument("--adaptive_downsample", action="store_true", default=False,
                    help="Enable adaptive per-dataset downsampling to target --target_pixel_size")
    ap.add_argument("--target_pixel_size", type=float, default=5.0,
                    help="Target effective pixel size in Å/px for adaptive downsampling (default: 5.0). "
                         "With 1.2 Å/px data → 4.2x downsample; with 0.32 Å/px (10061) → 15.6x downsample.")
    ap.add_argument("--min_downsample", type=float, default=1.0,
                    help="Minimum allowed downsample factor for adaptive mode (default: 1.0 = no upsampling)")
    ap.add_argument("--max_downsample", type=float, default=20.0,
                    help="Maximum allowed downsample factor for adaptive mode (default: 20.0). "
                         "High-res datasets like 10061 (0.32 Å/px) need 15.7x to reach 5.0 Å/px target. "
                         "Keeping same target pixel size as training data is more important than limiting downsampling.")
    ap.add_argument(
        "--adaptive_downsample_method",
        type=str,
        default=None,
        choices=["stride", "zoom", "fourier"],
        help=(
            "Resampling method used with --adaptive_downsample. "
            "If omitted, legacy behavior is preserved (zoom for cache/stitched validation, "
            "stride for dense full-micrograph extraction). Use 'fourier' for the released "
            "2 A/px balanced small-pixel recipe."
        ),
    )

    ap.add_argument(
        "--normalization_method",
        choices=[
            "percentile",
            "percentile_wide",
            "percentile_extra_wide",
            "percentile_blend",
            "simplipytem",
            "simplipytem_local",
            "simplipytem_local8bit",
            "robust",
        ],
        default="robust",
    )
    ap.add_argument(
        "--full_mic_train_normalization_methods",
        type=str,
        default="",
        help=(
            "Optional comma-separated normalization methods sampled per training micrograph in "
            "full-mic mode. Validation still uses --normalization_method."
        ),
    )
    ap.add_argument(
        "--full_mic_train_normalization_probs",
        type=str,
        default="",
        help=(
            "Optional comma-separated sampling probabilities matching "
            "--full_mic_train_normalization_methods."
        ),
    )
    ap.add_argument("--use_pixel_size_channel", action="store_true", default=False,
                    help="Append a constant input channel encoding dataset pixel size in Å/px.")
    ap.add_argument("--pixel_size_channel_min_angstrom", type=float, default=None,
                    help="Lower bound for pixel-size channel encoding. Defaults to the manifest minimum.")
    ap.add_argument("--pixel_size_channel_max_angstrom", type=float, default=None,
                    help="Upper bound for pixel-size channel encoding. Defaults to the manifest maximum.")
    ap.add_argument("--psd_use_radial_normalization", dest="psd_use_radial_normalization", action="store_true", default=True, help="Use radial norm for PSD (default: True)")
    ap.add_argument("--no-psd_use_radial_normalization", dest="psd_use_radial_normalization", action="store_false", help="Disable radial norm for PSD")
    ap.add_argument("--psd_use_radial_image", action="store_true", default=False,
                    help="Use radial PSD image instead of full 2D PSD (RECOMMENDED - much stronger signal for contamination)")
    ap.add_argument("--psd_radial_bins", type=int, default=64, help="Number of frequency bins for radial PSD (default 64)")
    ap.add_argument("--psd_multiscale", action="store_true", default=False,
                    help="Use multi-scale radial PSD (captures contamination at multiple spatial scales)")
    ap.add_argument(
        "--psd_multiscale_separate_channels",
        action="store_true",
        default=False,
        help=(
            "Use one PSD input channel per multiscale bin instead of a single fused PSD channel. "
            "Effective input channels become 1 + len(psd_scales) when --use_psd=true."
        ),
    )
    ap.add_argument("--psd_scales", type=str, default="16,32,64",
                    help="Comma-separated bin counts for multi-scale PSD (default: '16,32,64' = coarse,medium,fine)")
    ap.add_argument(
        "--psd_frequency_band_channels",
        action="store_true",
        default=False,
        help="Use physically defined Fourier-frequency bands as separate PSD channels.",
    )
    ap.add_argument(
        "--psd_frequency_bands",
        type=str,
        default="",
        help="Comma-separated low:high bands in Å^-1, e.g. '0.004:0.010,0.018:0.030'.",
    )
    ap.add_argument(
        "--psd_frequency_band_include_full_spectrum",
        action="store_true",
        default=False,
        help="Append one full-spectrum 2D PSD channel in frequency-band mode.",
    )
    ap.add_argument(
        "--psd_frequency_band_include_anisotropy",
        action="store_true",
        default=False,
        help="Append one anisotropy/streak PSD channel in frequency-band mode.",
    )
    ap.add_argument(
        "--psd_frequency_band_anisotropy_min_freq",
        type=float,
        default=0.0,
        help="Ignore frequencies below this Å^-1 in the anisotropy/streak channel.",
    )
    ap.add_argument("--psd_texture_mode", action="store_true", default=False,
                    help="Focus PSD on FINE TEXTURE (5-40 Å) where carbon/ice differ. "
                         "Suppresses low-freq (80-320 Å) where they look similar. CRITICAL for calibration!")
    ap.add_argument("--psd_highpass_cutoff", type=float, default=0.2,
                    help="In texture_mode, ignore frequencies below this (norm_dist). Default 0.2 = ~50 Å cutoff. "
                         "For 64x64 patch at 5 Å/px: 0.2 = ~50 Å cutoff, keeping 5-50 Å texture only.")
    # Multi-resolution CONTEXT PSD - computes PSD on larger regions centered on each patch
    ap.add_argument("--psd_multires_context", action="store_true", default=False,
                    help="Use multi-resolution context PSD (captures large carbon edges by computing PSD on larger regions)")
    ap.add_argument("--psd_context_sizes", type=str, default="64,128,256",
                    help="Comma-separated context sizes for multi-res PSD (default: '64,128,256' at working resolution)")
    ap.add_argument("--psd_bins_per_scale", type=int, default=32,
                    help="Number of frequency bins per context scale (default: 32)")
    # GLOBAL FEATURES - captures micrograph-wide characteristics for classifying uniform regions
    ap.add_argument("--use_global_features", action="store_true", default=False,
                    help="[DEPRECATED] Add global image-level features - use --use_edge_direction instead")
    ap.add_argument("--global_feature_dim", type=int, default=154,
                    help="[DEPRECATED] Dimension of global feature vector")
    # DIRECTIONAL EDGE FEATURES (v2) - simple 3-scalar approach for carbon edges
    ap.add_argument("--use_edge_direction", action="store_true", default=False,
                    help="Use directional edge features (3 scalars: strength, angle, proximity)")
    ap.add_argument("--edge_direction_dim", type=int, default=64,
                    help="Output dimension of edge direction MLP (default: 64)")
    # VISION TRANSFORMER - enables full-image context via self-attention
    ap.add_argument("--use_vit", action="store_true", default=False,
                    help="Use Vision Transformer for patch contextualization (processes full micrographs)")
    ap.add_argument("--vit_depth", type=int, default=6,
                    help="Number of ViT transformer layers (default: 6)")
    ap.add_argument("--vit_heads", type=int, default=8,
                    help="Number of ViT attention heads (default: 8)")
    ap.add_argument("--vit_embed_dim", type=int, default=512,
                    help="ViT embedding dimension (default: 512)")
    ap.add_argument("--vit_dropout", type=float, default=0.1,
                    help="ViT dropout rate (default: 0.1)")
    # Edge features - helps catch carbon EDGES that radial PSD misses
    ap.add_argument("--use_edge_features", action="store_true", default=False,
                    help="Blend edge magnitude into real image (helps catch carbon edges)")
    ap.add_argument("--edge_blend_alpha", type=float, default=0.3,
                    help="Blend weight for edge features: real*(1-alpha) + edge*alpha (default 0.3)")
    ap.add_argument("--edge_sigma", type=float, default=1.0,
                    help="Gaussian smoothing sigma before edge detection (default 1.0)")

    ap.add_argument("--num_epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="Number of visible CUDA GPUs to use via DataParallel (default 1)")

    ap.add_argument("--min_area_floor_px", type=float, default=50.0)
    ap.add_argument(
        "--region_gt_fill_holes",
        dest="region_gt_fill_holes",
        action="store_true",
        default=True,
        help="Fill holes inside connected GT regions after the dataset min-area transform (default: enabled).",
    )
    ap.add_argument(
        "--no_region_gt_fill_holes",
        "--no-region_gt_fill_holes",
        dest="region_gt_fill_holes",
        action="store_false",
        help="Do not fill holes inside GT regions; use this for membrane/mesh masks where holes are real background.",
    )
    ap.add_argument("--core_erosion_px", type=int, default=8)

    # ==========================================================================
    # EDGE EROSION: Critical for patch-based training
    # ==========================================================================
    # Patches have no context beyond their boundaries. If we compute loss on
    # edge pixels, the model learns "edge of context = ???" which creates
    # systematic artifacts (swaths at patch boundaries during inference).
    #
    # Fix: Exclude edge pixels from loss computation using a Tukey-style mask
    # (flat 1.0 in center, tapers to 0 at edges)
    # ==========================================================================
    ap.add_argument("--edge_erosion_px", type=int, default=16,
                    help="Pixels to exclude from loss at patch boundaries (full-res). "
                         "Prevents model from learning 'edge of context' artifacts. "
                         "Default 16 → ~4px effective after 4x downsample. Set higher (32+) if seeing edge swaths.")
    ap.add_argument("--use_tukey_weight", action="store_true", default=True,
                    help="Use Tukey window (flat center, tapered edges) for loss weighting. "
                         "More gradual than hard erosion, better gradient flow.")

    # ==========================================================================
    # DICE + FOCAL LOSS: Medical imaging's proven approach (RECOMMENDED)
    # ==========================================================================
    # Use --use_dice_loss to enable. This replaces the multi-component BCE loss
    # which suffers from "predict everything as contamination" collapse.
    #
    # Dice loss is SELF-BALANCING:
    #   - Cannot game by predicting all 1s (Dice would be ~0.5 if GT is 50%)
    #   - Cannot game by predicting all 0s (Dice would be 0)
    #   - Only way to maximize Dice is to precisely match GT shape
    #
    # Focal loss PREVENTS OVERCONFIDENCE:
    #   - Once confident, that pixel stops contributing to training
    #   - No need for phase transitions or complex loss scheduling
    # ==========================================================================
    ap.add_argument("--use_dice_loss", action="store_true",
                    help="Use Dice+Focal loss instead of multi-component BCE. "
                         "RECOMMENDED: Self-balancing, cannot collapse to 'predict everything'. "
                         "Medical imaging standard for lesion/tumor detection.")
    ap.add_argument("--lambda_dice", type=float, default=2.0,
                    help="Weight for Dice loss (default 2.0)")
    ap.add_argument("--lambda_focal", type=float, default=1.0,
                    help="Weight for Focal loss (default 1.0)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma - focusing parameter (default 2.0)")
    ap.add_argument("--focal_alpha", type=float, default=0.25,
                    help="Focal loss alpha - positive class weight (default 0.25)")
    ap.add_argument(
        "--positive_interior_weight",
        type=float,
        default=0.0,
        help="Extra Dice/Focal weight for eroded GT interiors in full-mic mode (default off).",
    )
    ap.add_argument(
        "--positive_interior_erode_px",
        type=int,
        default=32,
        help="Full-res erosion radius for the positive-interior weight mask.",
    )

    # Legacy BCE-based loss (kept for comparison, but DEPRECATED)
    ap.add_argument("--lambda_bce", type=float, default=0.5, help="Weight for BCE (calibration); reduced default to allow other terms to dominate")
    ap.add_argument("--bce_neg_weight", type=float, default=3.0, help="Weight for negative class in BCE (default 3.0, increase to suppress false positives)")
    ap.add_argument("--lambda_hit_core", type=float, default=4.0, help="Weight for hit-the-core (doubled from 2.0 to encourage positive learning)")
    ap.add_argument("--soft_hit_core", action="store_true",
                    help="Use SOFT hit_core (mean instead of max). "
                         "Max-based hit_core only needs ONE pixel high, which can cause overconfidence. "
                         "Mean-based requires the WHOLE core to be predicted high, better calibration.")
    ap.add_argument("--lambda_outside_mass", type=float, default=0.5, help="Weight for outside/background suppression (reduced from 1.0; conflicts with FP-cap when high)")
    ap.add_argument("--outside_dilate_px", type=int, default=2, help="Dilate GT by N pixels before computing outside-mass loss (default 2px, avoids punishing slight misalignment)")
    ap.add_argument("--lambda_particle_neg", type=float, default=0.3, help="Weight for particle-likeness negative mining (default 0.3, pushes logits down on particle-like pixels in clean regions)")
    ap.add_argument("--particle_neg_margin_px", type=int, default=16, help="Margin away from GT to search for particles in FULL-RES pixels (default 16px, scales down with downsample to ~3-5px effective)")
    ap.add_argument("--particle_neg_margin_wide_px", type=int, default=32, help="Wider margin ring for particle-neg (full-res pixels, default 32px → ~8px effective, lower weight)")
    ap.add_argument("--lambda_particle_neg_wide", type=float, default=0.1, help="Weight for wider particle-neg ring (default 0.1, lower than main particle_neg)")
    ap.add_argument("--particle_neg_enabled", action="store_true", default=True, help="Enable particle-likeness negative mining (default: True)")

    # A. Differentiable FP control term
    ap.add_argument("--lambda_fpcap", type=float, default=0.1, help="Weight for differentiable FP control term L_fpcap (default 0.1, ramps to target over 10 epochs)")
    ap.add_argument("--fpcap_target_mass", type=float, default=0.03, help="Target FP mass for L_fpcap (default 0.03, start loose then tighten)")

    # B. Edge-conditioned suppression on negatives (ring glow suppression)
    ap.add_argument("--lambda_edge", type=float, default=0.5, help="Weight for edge-conditioned suppression on negatives L_edge (default 0.5, penalizes p*E on GT==0)")

    # DYNAMIC LOSS REWEIGHTING: Prevents "predict everything as contamination" collapse
    # Phase 1 (epochs 1 to reweight_epoch): Discovery - find the contamination
    # Phase 2 (epochs > reweight_epoch): Refinement - reduce hit_core, increase fpcap
    # This addresses the "4 losses voting for high predictions vs 1 for low" problem
    ap.add_argument("--loss_reweight_epoch", type=int, default=0,
                    help="Epoch to switch from discovery to refinement phase (0=disabled). "
                         "After this epoch, hit_core decreases and fpcap increases to prevent collapse.")
    ap.add_argument("--phase2_lambda_hit_core", type=float, default=0.1,
                    help="Hit-core weight after reweight_epoch (default 0.1, down from 1.0+)")
    ap.add_argument("--phase2_lambda_bce", type=float, default=0.3,
                    help="BCE weight after reweight_epoch (default 0.3, down from 1.0)")
    ap.add_argument("--phase2_lambda_fpcap", type=float, default=5.0,
                    help="FP-cap weight after reweight_epoch (default 5.0, up from 0.2-2.0)")
    ap.add_argument("--phase2_lambda_outside_mass", type=float, default=0.1,
                    help="Outside-mass weight after reweight_epoch (default 0.1)")

    # C. Hard-negative patch sampling
    ap.add_argument("--hard_neg_fraction", type=float, default=0.2, help="Fraction of negative patches from hard negatives (default 0.2, was 0.4; lower = more stable)")
    ap.add_argument("--hard_neg_quantile", type=float, default=0.6, help="Top quantile of particle-likeness for 'hard' (default 0.6, was 0.75; lower = less extreme)")

    ap.add_argument("--early_stop_patience", type=int, default=10, help="Stop if val loss has not improved for this many stitched-val cycles (increased from 5 to allow calibration)")
    ap.add_argument("--early_stop_min_epochs", type=int, default=40, help="Minimum epochs before early stopping is allowed (prevents stopping during calibration phase)")
    ap.add_argument("--early_stop_min_stitched_cycles", type=int, default=8, help="Minimum stitched-val cycles before early stopping is allowed (alternative to min_epochs)")
    ap.add_argument("--early_stop_mode", choices=["score", "multi_fp_cap", "composite"], default="composite", help="Early stopping mode: 'score' uses PR-AUC - α*fp_mass, 'multi_fp_cap' monitors recall at multiple FP caps, 'composite' uses 0.5*Recall@2%% + 0.5*Recall@5%%")
    ap.add_argument("--early_stop_alpha", type=float, default=1.0, help="Weight α for score-based early stopping: score = PR-AUC - α * fp_mass (default 1.0, range 0.5-2.0)")

    ap.add_argument("--stitched_val_every_n_epochs", type=int, default=5, help="Run full-micrograph stitched validation every N epochs")
    ap.add_argument("--stitched_val_mics_per_dataset", type=int, default=2)
    ap.add_argument("--exclude_datasets_from_metrics", type=str, default="",
                    help="Comma-separated list of dataset IDs to exclude from metric computation "
                         "(but still run inference for visual monitoring). E.g., '10061' for extreme "
                         "downsampling cases that would drag down aggregate metrics.")
    ap.add_argument("--stitched_val_patch_size", type=int, default=256)
    ap.add_argument("--stitched_val_overlap", type=int, default=64)
    ap.add_argument("--stitched_val_downsample", type=float, default=1.0, help="Downsample for stitched val (1=full res)")
    ap.add_argument("--stitched_val_min_area_px", type=float, default=50.0)
    ap.add_argument("--stitched_val_fp_area_cap_pct", type=float, default=0.5, help="FP area %% cap for recall-at-fp (default 0.5)")
    ap.add_argument("--stitched_val_opening_size", type=int, default=2)
    ap.add_argument("--stitched_val_closing_size", type=int, default=3)
    ap.add_argument(
        "--stitched_val_thresholds",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated thresholds for stitched-val recall/FP sweep.",
    )
    ap.add_argument(
        "--stitched_val_metrics_max_samples",
        type=int,
        default=5_000_000,
        help="Maximum pixels sampled for stitched-val PR-AUC/AUROC computation.",
    )
    ap.add_argument("--cache_dir", default=None, help="Dir for per-mic cache (default: output_dir/patch_cache). Built once before training; no PSD/norm recompute after.")
    ap.add_argument("--no-export_stitched_prob_maps", dest="export_stitched_prob_maps", action="store_false", default=True, help="Do not export stitched prob maps every stitched-val run (default: export)")
    ap.add_argument("--stitched_val_pixel_size_angstrom", type=float, default=1.1, help="Pixel size in Å for 4 Å low-pass in export panels (default 1.1)")
    ap.add_argument("--stitched_val_blending_window", choices=["hann", "tukey"], default="hann",
                    help="Blend window for stitched validation patch fusion.")
    ap.add_argument("--stitched_val_blending_edge_px", type=int, default=16,
                    help="Edge taper in pixels when using Tukey blending for stitched validation.")
    ap.add_argument("--stitched_val_batch_forward_size", type=int, default=1, help="Batch size for patch forward pass during stitched val (use 32 for A100, 0=auto)")
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (default 8, capped at 8 for NFS; higher causes contention)")
    ap.add_argument("--warmup_steps", type=int, default=400, help="LR warmup steps before cosine decay")

    # Actual particle picks (positive labels) - more accurate than heuristic particle-likeness
    ap.add_argument("--particle_annotations", default=None, help="Path to dir or pkl/json with actual particle picks (keys=dataset_id__stem, values=list of (x,y) full-res).")
    ap.add_argument("--particle_box_size_px", type=int, default=256, help="Size in full-res px; effective radius = box_size/(4*downsample) for small Gaussian (default 256)")
    ap.add_argument("--particle_use_gaussian", action="store_true", default=False, help="Use soft Gaussian mask for particles (slow, not recommended)")
    ap.add_argument("--no-particle_use_gaussian", dest="particle_use_gaussian", action="store_false", help="Use hard disk mask (default, faster for high worker counts)")

    # PU-style loss (new design): BAD=trusted positive, GOOD-PICKS=trusted negative, rest=UNKNOWN
    ap.add_argument("--use_pu_loss", action="store_true", default=True, help="Use PU-style loss with curriculum (recommended)")
    ap.add_argument("--no-use_pu_loss", dest="use_pu_loss", action="store_false", help="Use old loss (not recommended)")
    ap.add_argument("--lambda_good", type=float, default=0.2, help="Weight for good-anchor hinge loss L_good (default 0.2)")
    ap.add_argument("--lambda_margin", type=float, default=0.0, help="Weight for logit margin loss L_margin (default 0.0 - DISABLED, was causing overmasking)")
    ap.add_argument("--lambda_area", type=float, default=0.5, help="Weight for unknown-area budget L_area (default 0.5)")
    ap.add_argument("--tau_good", type=float, default=0.15, help="Hinge threshold for good anchors: only penalize if p > tau (default 0.15)")
    ap.add_argument("--margin_m", type=float, default=2.5, help="Logit margin between Bcore and G (default 2.5, try 2.0-4.0)")
    ap.add_argument("--rho_area", type=float, default=0.05, help="Target expected bad fraction in unknown (default 0.05, anneal to 0.03)")
    ap.add_argument("--rho_area_final", type=float, default=0.03, help="Final rho after annealing (default 0.03)")
    # NEW: Entropy regularization and asymmetric particle penalty
    ap.add_argument("--lambda_entropy", type=float, default=0.0,
                    help="Entropy regularization weight - penalize overconfident predictions on unknown regions (default 0.0)")
    ap.add_argument("--lambda_particle_asym", type=float, default=0.0,
                    help="Asymmetric particle penalty - strong BCE to 0 on particle regions (default 0.0)")
    ap.add_argument("--entropy_target_low", type=float, default=0.1,
                    help="Lower bound of target probability range for entropy reg (default 0.1)")
    ap.add_argument("--entropy_target_high", type=float, default=0.3,
                    help="Upper bound of target probability range for entropy reg (default 0.3)")
    # Curriculum phases
    ap.add_argument("--phase_a_epochs", type=int, default=10, help="Phase A (bad only): epochs 1 to N (default 10)")
    ap.add_argument("--phase_b_epochs", type=int, default=25, help="Phase B (+ good, area): epochs phase_a+1 to N (default 25)")
    # After phase_b_epochs, Phase C enables margin loss
    ap.add_argument("--disable_curriculum", action="store_true",
                    help="DISABLE Phase curriculum entirely - stay in Phase A forever. "
                         "Phase B/C enable hard-neg sampling, GT erosion, and expensive CPU ops "
                         "which can cause calibration drift. Use this for stable training.")

    # STITCHED TRAINING LOSS - forces model to see TRUE distribution during training
    ap.add_argument("--use_stitched_train_loss", action="store_true", default=False,
                    help="Add stitched full-micrograph loss during training. "
                         "This forces the model to see the REAL distribution (not GT-biased) every batch. "
                         "Expensive but addresses root cause of collapse.")
    ap.add_argument("--lambda_stitched", type=float, default=1.0,
                    help="Weight for stitched training loss (default 1.0)")
    ap.add_argument("--stitched_train_mics_per_batch", type=int, default=1,
                    help="Number of full micrographs to process per batch for stitched loss (default 1)")
    ap.add_argument("--stitched_train_every_n_batches", type=int, default=4,
                    help="Compute stitched loss every N batches (default 4, reduces overhead)")
    ap.add_argument("--stitched_train_grid_patches", type=int, default=16,
                    help="Number of patches in grid for stitched loss (default 16 = 4x4 sparse grid)")

    # ==========================================================================
    # FULL MICROGRAPH TRAINING (MicrographCleaner-style)
    # ==========================================================================
    # This is the PROVEN approach from MicrographCleaner (Sanchez-Garcia et al. 2020)
    # Process ENTIRE micrographs during training, not GT-biased patches.
    #
    # Key insight: GT-biased sampling creates distribution mismatch:
    #   Training sees: 50% carbon patches (GT-biased)
    #   Reality: 10-30% carbon in full micrographs
    #   Model learns wrong base rate → predicts too much carbon
    #
    # MicrographCleaner approach:
    #   - Dense overlapping patches from FULL micrograph (no GT bias)
    #   - Training distribution = Validation distribution
    #   - Requires more epochs (100-200) but NO collapse
    # ==========================================================================
    ap.add_argument("--full_micrograph_training", action="store_true", default=False,
                    help="MicrographCleaner-style training: process ALL patches from full micrographs. "
                         "NO GT-bias, training sees TRUE distribution. Requires more epochs (~100).")
    ap.add_argument("--full_mic_mics_per_epoch", type=int, default=0,
                    help="Number of micrographs to process per epoch in full_micrograph_training mode. "
                         "Default 0 = ALL micrographs (recommended for stable training loss). "
                         "Set to smaller value (e.g., 50) only if compute-limited.")
    ap.add_argument("--full_mic_patches_per_mic", type=int, default=64,
                    help="Max patches per micrograph in full_micrograph_training mode "
                         "(default 64, uses dense grid with this cap)")
    ap.add_argument("--full_mic_overlap", type=float, default=0.5,
                    help="Patch overlap ratio for full_micrograph_training mode "
                         "(default 0.5 = 50%% overlap, like MicrographCleaner)")
    ap.add_argument("--full_mic_prefetch_workers", type=int, default=1,
                    help="Background micrograph extraction workers in full_micrograph_training mode (default 1)")
    ap.add_argument("--full_mic_hard_positive_fraction", type=float, default=0.0,
                    help="Fraction of capped full-mic patches reserved for GT-rich patches (default 0=uniform only)")
    ap.add_argument("--full_mic_hard_positive_min_gt_frac", type=float, default=0.10,
                    help="Minimum GT fraction for full-mic hard-positive patch candidates")
    ap.add_argument("--full_mic_hard_positive_erode_px", type=int, default=0,
                    help="Full-res GT erosion radius used to prefer broad positive interiors for hard-positive sampling")
    ap.add_argument("--skip_patch_cache_in_full_mic", action="store_true", default=False,
                    help="Skip patch-cache building and patch validation loader setup when full_micrograph_training is active")

    # Patch-type balanced sampling (DEPRECATED - did not work well)
    ap.add_argument("--use_patch_type_sampling", action="store_true", default=False,
                    help="[DEPRECATED] Enable explicit patch-type balanced sampling - NOT RECOMMENDED")
    ap.add_argument("--bad_centered_fraction", type=float, default=0.4)
    ap.add_argument("--particle_centered_fraction", type=float, default=0.4)
    ap.add_argument("--lambda_patch_classify", type=float, default=0.5)
    ap.add_argument("--lambda_particle_region", type=float, default=1.0)
    ap.add_argument("--lambda_bad_region", type=float, default=1.0)
    ap.add_argument("--particle_region_radius_frac", type=float, default=0.4)

    # SIMPLE PARTICLE BCE - direct BCE supervision at particle locations
    ap.add_argument("--use_simple_particle_bce", action="store_true", default=False,
                    help="Use simple BCE loss at particle locations (target=0, not contamination)")
    ap.add_argument("--lambda_particle_bce", type=float, default=0.3,
                    help="Weight for simple particle BCE loss (default 0.3)")

    # STABLE LOSS with phased warmup (DEPRECATED - use --use_dice_loss instead)
    ap.add_argument("--use_stable_loss", action="store_true", default=False,
                    help="DEPRECATED: Use --use_dice_loss instead. Stable loss with phased warmup.")
    ap.add_argument("--warmup_phase1_epochs", type=int, default=5,
                    help="Phase 1 (BCE+Dice only) duration in epochs (default 5)")
    ap.add_argument("--warmup_phase2_epochs", type=int, default=10,
                    help="Phase 2 ends (add hit-core) at this epoch (default 10)")
    ap.add_argument("--warmup_phase3_epochs", type=int, default=15,
                    help="Phase 3 ends (add particle suppression) at this epoch (default 15)")
    ap.add_argument("--stable_lambda_hit_core", type=float, default=2.0,
                    help="Hit-core weight for stable loss (default 2.0)")
    ap.add_argument("--stable_lambda_particle", type=float, default=0.3,
                    help="Particle anchor suppression weight for stable loss (default 0.3)")
    ap.add_argument("--stable_lambda_area", type=float, default=0.3,
                    help="Area budget weight for stable loss (default 0.3)")
    ap.add_argument("--stable_tau_anchor", type=float, default=0.15,
                    help="Hinge threshold for particle anchors in stable loss (default 0.15)")
    ap.add_argument("--stable_rho_area", type=float, default=0.05,
                    help="Target bad fraction in unknown regions for stable loss (default 0.05)")

    # DATA AUGMENTATION
    ap.add_argument("--use_rotation_aug", action="store_true", default=False,
                    help="Enable random 90° rotations (helps with ring proteins)")
    ap.add_argument("--use_noise_augmentation", action="store_true", default=False,
                    help="Add realistic cryo-EM noise to clean patches (teaches: noisy clean ≠ contamination)")
    ap.add_argument("--noise_aug_prob", type=float, default=0.5,
                    help="Probability of applying noise augmentation to a micrograph (default: 0.5)")

    # STAGED CURRICULUM FOR NOISE AUGMENTATION
    # Prevents catastrophic forgetting by delaying noise and using LR drops
    ap.add_argument("--noise_start_epoch", type=int, default=1,
                    help="Epoch to start noise augmentation (default: 1, set to 16 for staged curriculum)")
    ap.add_argument("--noise_ramp_epochs", type=int, default=0,
                    help="Number of epochs to ramp noise from 0 to full strength (default: 0 = instant)")
    ap.add_argument("--noise_max_strength", type=float, default=1.0,
                    help="Maximum noise strength multiplier (default: 1.0, use 0.3-0.5 for gentler noise)")

    # STAGED LEARNING RATE SCHEDULE
    # Drop LR when noise starts to prevent catastrophic forgetting
    ap.add_argument("--lr_drop_epoch_1", type=int, default=0,
                    help="First epoch to drop LR (default: 0 = no drop, set to noise_start_epoch)")
    ap.add_argument("--lr_drop_factor_1", type=float, default=5.0,
                    help="Factor to divide LR by at lr_drop_epoch_1 (default: 5.0)")
    ap.add_argument("--lr_drop_epoch_2", type=int, default=0,
                    help="Second epoch to drop LR (default: 0 = no second drop)")
    ap.add_argument("--lr_drop_factor_2", type=float, default=2.0,
                    help="Factor to divide LR by at lr_drop_epoch_2 (default: 2.0)")

    args = ap.parse_args()

    # GPU performance optimizations (no effect on training accuracy)
    if torch.cuda.is_available():
        # TF32: A100/H100 can do float32 matmuls in TF32 (19-bit) for ~3x speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cuDNN benchmark: auto-tune convolution algorithms for fixed input sizes
        torch.backends.cudnn.benchmark = True
        print("  GPU optimizations: TF32 matmul + cuDNN benchmark enabled", flush=True)

    # Robust seed setting for reproducibility (ablation studies)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Parse --use_psd flag
    args.use_psd_bool = (args.use_psd.lower() == "true") if hasattr(args, 'use_psd') else True
    args.include_real_space_input = not bool(getattr(args, "psd_only", False))
    if args.psd_only and not args.use_psd_bool:
        raise ValueError("--psd_only requires --use_psd true.")
    if args.psd_only and args.model_type == "miffi" and args.use_two_branch_encoder:
        raise ValueError("--psd_only is incompatible with --use_two_branch_encoder (requires real+PSD branches).")
    if args.psd_only:
        print("  PSD-only mode enabled: real-space image channel will be removed from model input.", flush=True)
    args.psd_scales_tuple = tuple(int(x) for x in args.psd_scales.split(","))
    args.psd_frequency_bands_tuple = _parse_frequency_bands(getattr(args, "psd_frequency_bands", ""))
    args.stitched_val_threshold_grid = _parse_float_list(args.stitched_val_thresholds)
    bad_thresholds = [t for t in args.stitched_val_threshold_grid if t <= 0.0 or t >= 1.0]
    if bad_thresholds:
        raise ValueError(f"--stitched_val_thresholds values must be in (0, 1), got {bad_thresholds}")
    args.stitched_val_metrics_max_samples = max(1, int(args.stitched_val_metrics_max_samples))
    print(
        "  Stitched-val threshold grid: "
        + ", ".join(f"{float(t):.2f}" for t in args.stitched_val_threshold_grid),
        flush=True,
    )
    print(
        f"  Stitched-val metric sample cap: {args.stitched_val_metrics_max_samples:,} pixels",
        flush=True,
    )
    allowed_train_norms = {
        "percentile",
        "percentile_wide",
        "percentile_extra_wide",
        "percentile_blend",
        "simplipytem",
        "simplipytem_local",
        "simplipytem_local8bit",
        "robust",
    }
    args.full_mic_train_normalization_methods_list = []
    args.full_mic_train_normalization_probs_list = None
    train_norm_text = str(getattr(args, "full_mic_train_normalization_methods", "") or "").strip()
    train_norm_prob_text = str(getattr(args, "full_mic_train_normalization_probs", "") or "").strip()
    if train_norm_text:
        train_norm_methods = [part.strip() for part in train_norm_text.split(",") if part.strip()]
        unknown_train_norms = sorted(set(train_norm_methods) - allowed_train_norms)
        if unknown_train_norms:
            raise ValueError(
                "--full_mic_train_normalization_methods contains unknown method(s): "
                + ", ".join(unknown_train_norms)
            )
        args.full_mic_train_normalization_methods_list = train_norm_methods
        if train_norm_prob_text:
            train_norm_probs = [float(part.strip()) for part in train_norm_prob_text.split(",") if part.strip()]
            if len(train_norm_probs) != len(train_norm_methods):
                raise ValueError(
                    "--full_mic_train_normalization_probs must have one value per "
                    "--full_mic_train_normalization_methods entry"
                )
            if any((not np.isfinite(prob)) or prob < 0.0 for prob in train_norm_probs):
                raise ValueError("--full_mic_train_normalization_probs must be finite non-negative values")
            if sum(train_norm_probs) <= 0.0:
                raise ValueError("--full_mic_train_normalization_probs must sum to > 0")
            args.full_mic_train_normalization_probs_list = train_norm_probs
        print(
            "  Full-mic train normalization sampler: "
            + ", ".join(args.full_mic_train_normalization_methods_list)
            + (
                " with probs "
                + ", ".join(f"{float(p):.3f}" for p in args.full_mic_train_normalization_probs_list)
                if args.full_mic_train_normalization_probs_list is not None
                else " with uniform probabilities"
            )
            + f"; validation/deployment normalization={args.normalization_method}",
            flush=True,
        )
    elif train_norm_prob_text:
        raise ValueError("--full_mic_train_normalization_probs requires --full_mic_train_normalization_methods")

    if not (0.0 <= float(args.full_mic_hard_positive_fraction) <= 1.0):
        raise ValueError("--full_mic_hard_positive_fraction must be in [0, 1]")
    if not (0.0 <= float(args.full_mic_hard_positive_min_gt_frac) <= 1.0):
        raise ValueError("--full_mic_hard_positive_min_gt_frac must be in [0, 1]")
    args.full_mic_hard_positive_erode_px = max(0, int(args.full_mic_hard_positive_erode_px))
    args.positive_interior_weight = max(0.0, float(args.positive_interior_weight))
    args.positive_interior_erode_px = max(0, int(args.positive_interior_erode_px))
    args.num_gpus = max(1, int(getattr(args, "num_gpus", 1)))

    if not args.psd_multiscale and args.psd_multiscale_separate_channels:
        print("  WARNING: --psd_multiscale_separate_channels requires --psd_multiscale; disabling separate-channels mode.", flush=True)
        args.psd_multiscale_separate_channels = False
    if not args.use_psd_bool and args.psd_multiscale_separate_channels:
        print("  WARNING: --use_psd=false so --psd_multiscale_separate_channels is ignored.", flush=True)
        args.psd_multiscale_separate_channels = False
    if not args.use_psd_bool and args.psd_frequency_band_channels:
        print("  WARNING: --use_psd=false so --psd_frequency_band_channels is ignored.", flush=True)
        args.psd_frequency_band_channels = False
        args.psd_frequency_band_include_full_spectrum = False
        args.psd_frequency_band_include_anisotropy = False
        args.psd_frequency_bands_tuple = tuple()
    if args.psd_frequency_band_channels and args.psd_multiscale:
        raise ValueError("--psd_frequency_band_channels is incompatible with --psd_multiscale.")
    if args.psd_frequency_band_channels and args.psd_use_radial_image:
        raise ValueError("--psd_frequency_band_channels is incompatible with --psd_use_radial_image.")
    if (
        args.psd_frequency_band_channels
        and (not args.psd_frequency_bands_tuple)
        and (not args.psd_frequency_band_include_full_spectrum)
        and (not args.psd_frequency_band_include_anisotropy)
    ):
        raise ValueError(
            "--psd_frequency_band_channels requires at least one --psd_frequency_bands entry "
            "or --psd_frequency_band_include_full_spectrum or --psd_frequency_band_include_anisotropy."
        )

    args.resolved_input_channels = _expected_input_channels(
        use_psd_input=args.use_psd_bool,
        psd_multiscale=args.psd_multiscale,
        psd_multiscale_separate_channels=args.psd_multiscale_separate_channels,
        psd_scales=args.psd_scales_tuple,
        psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
        psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
        psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
        psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
        include_real_space_input=args.include_real_space_input,
        use_pixel_size_channel=bool(args.use_pixel_size_channel),
    )

    # If using ViT, use separate training function
    if args.use_vit:
        train_vit_model(args)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).expanduser().resolve()
    df = pd.read_csv(manifest_path)
    df = _resolve_manifest_relative_paths(df, manifest_path)
    pixel_size_by_dataset = _build_dataset_pixel_size_map(df)

    # Compute downsample factors (uniform or adaptive)
    if args.adaptive_downsample:
        print(f"\n=== ADAPTIVE DOWNSAMPLING ENABLED ===", flush=True)
        print(f"  Target pixel size: {args.target_pixel_size} Å/px", flush=True)
        print(f"  Allowed range: {args.min_downsample}x - {args.max_downsample}x", flush=True)

        downsample_factors = compute_adaptive_downsample_factors(
            df,
            target_pixel_size=args.target_pixel_size,
            min_downsample=args.min_downsample,
            max_downsample=args.max_downsample,
            default_pixel_size=1.2,  # Typical cryo-EM pixel size
        )

        # Print per-dataset factors
        print(f"  Per-dataset downsample factors:", flush=True)
        for dsid in sorted(downsample_factors.keys(), key=lambda x: int(x)):
            factor = downsample_factors[dsid]
            # Get original pixel size for display
            if 'pixel_size_angstrom' in df.columns:
                subset = df[df['dataset_id'] == int(dsid)]
                if len(subset) > 0:
                    px_size = subset['pixel_size_angstrom'].iloc[0]
                    eff_px = px_size * factor
                    # Check if this factor was capped
                    uncapped_factor = args.target_pixel_size / px_size
                    if uncapped_factor > args.max_downsample:
                        print(f"    {dsid}: {factor:.2f}x ({px_size:.2f} → {eff_px:.2f} Å/px) [CAPPED from {uncapped_factor:.1f}x]", flush=True)
                    else:
                        print(f"    {dsid}: {factor:.2f}x ({px_size:.2f} → {eff_px:.2f} Å/px)", flush=True)
                else:
                    print(f"    {dsid}: {factor:.2f}x", flush=True)
            else:
                print(f"    {dsid}: {factor:.2f}x", flush=True)

        # Use median factor for effective size calculations
        median_factor = float(np.median(list(downsample_factors.values())))
        print(f"  Median factor: {median_factor:.2f}x", flush=True)
        downsample_factor = median_factor  # For effective size calculations

        # REQUIRE working_patch_size for adaptive downsampling
        if args.working_patch_size is None:
            print("\n*** ERROR: --working_patch_size is REQUIRED when using --adaptive_downsample ***")
            print("    With adaptive downsampling, different datasets have different downsample factors,")
            print("    so patch_size/downsample doesn't give a consistent working resolution.")
            print("    Typical values: 64 (small, fast) or 128 (more context).")
            print("    Example: --working_patch_size 64")
            sys.exit(1)
    else:
        downsample_factors = None  # Will pass uniform factor instead
        downsample_factor = float(args.downsample)

    if downsample_factors is not None:
        effective_pixel_sizes = [
            float(pixel_size_by_dataset[dsid]) * float(max(1.0, downsample_factors.get(str(dsid), 1.0)))
            for dsid in sorted(pixel_size_by_dataset.keys())
        ]
    else:
        effective_pixel_sizes = [
            float(pixel_size_by_dataset[dsid]) * float(max(1.0, downsample_factor))
            for dsid in sorted(pixel_size_by_dataset.keys())
        ]
    if not effective_pixel_sizes:
        effective_pixel_sizes = [1.0]
    if args.pixel_size_channel_min_angstrom is None:
        args.pixel_size_channel_min_angstrom = float(min(effective_pixel_sizes))
    if args.pixel_size_channel_max_angstrom is None:
        args.pixel_size_channel_max_angstrom = float(max(effective_pixel_sizes))
    if args.use_pixel_size_channel:
        print(
            f"  Pixel-size channel enabled: "
            f"{args.pixel_size_channel_min_angstrom:.4f}–{args.pixel_size_channel_max_angstrom:.4f} Å/px (log-encoded)",
            flush=True,
        )

    # Compute effective patch size
    # If working_patch_size is specified, use it directly (REQUIRED for adaptive mode)
    # Otherwise compute from patch_size / downsample_factor (uniform mode only)
    if args.working_patch_size is not None:
        effective_patch_size = int(args.working_patch_size)
        print(f"  Using working_patch_size={effective_patch_size} (directly specified)", flush=True)
        # Validate it's divisible by 16 for U-Net compatibility
        if effective_patch_size % 16 != 0:
            print(f"  WARNING: working_patch_size={effective_patch_size} is not divisible by 16.")
            print(f"           This may cause dimension mismatches in U-Net. Recommend: 64, 128, or 256.")
    else:
        effective_patch_size = int(args.patch_size / downsample_factor) if downsample_factor > 1.0 else int(args.patch_size)

    # Compute effective stride
    # If working_stride is specified, use it directly (for adaptive mode)
    # Otherwise compute from patch_stride / downsample_factor
    if args.working_stride is not None:
        effective_stride = int(args.working_stride)
        print(f"  Using working_stride={effective_stride} (directly specified)", flush=True)
    elif args.patch_stride is not None:
        effective_stride = int(args.patch_stride / downsample_factor) if downsample_factor > 1.0 else int(args.patch_stride)
    else:
        effective_stride = None  # Will default to patch_size (no overlap)

    # Compute other effective sizes using median factor
    effective_min_area = float(args.min_area_floor_px) / (downsample_factor * downsample_factor) if downsample_factor > 1.0 else float(args.min_area_floor_px)
    effective_core_erosion = max(1, int(args.core_erosion_px / downsample_factor)) if downsample_factor > 1.0 else int(args.core_erosion_px)

    # Prefer explicit dataset IDs when supplied. Otherwise, app-generated
    # annotation manifests may carry a row-level split column for single
    # CryoSPARC workspaces; older manifests fall back to dataset-stratified
    # splitting.
    unique_datasets = sorted(int(x) for x in df["dataset_id"].unique())
    train_dataset_ids_arg = [int(x.strip()) for x in str(args.train_dataset_ids).split(",") if x.strip()]
    val_dataset_ids_arg = [int(x.strip()) for x in str(args.val_dataset_ids).split(",") if x.strip()]
    split_column = next((col for col in ("split", "split_label") if col in df.columns), None)
    train_datasets = set()
    val_datasets = set()
    if train_dataset_ids_arg or val_dataset_ids_arg:
        if not (train_dataset_ids_arg and val_dataset_ids_arg):
            raise ValueError("Must provide both --train_dataset_ids and --val_dataset_ids when overriding the split.")
        train_datasets = set(train_dataset_ids_arg)
        val_datasets = set(val_dataset_ids_arg)
        overlap = train_datasets & val_datasets
        if overlap:
            raise ValueError(f"Explicit split overlap detected: {sorted(overlap)}")
        unknown = (train_datasets | val_datasets) - set(unique_datasets)
        if unknown:
            raise ValueError(f"Explicit split contains unknown dataset IDs: {sorted(unknown)}")
        missing = set(unique_datasets) - (train_datasets | val_datasets)
        if missing:
            raise ValueError(f"Explicit split does not cover all datasets. Missing: {sorted(missing)}")
        print("Using explicit dataset split override.", flush=True)
        train_df = df[df['dataset_id'].isin(train_datasets)].reset_index(drop=True)
        val_df = df[df['dataset_id'].isin(val_datasets)].reset_index(drop=True)
        split_description = "Dataset-stratified split"
    elif split_column is not None:
        row_split = df[split_column].map(_normalize_row_split_label)
        valid = {"TRAINING", "VALIDATION"}
        invalid = sorted(set(label for label in row_split if label not in valid))
        if invalid:
            raise ValueError(
                f"Manifest column {split_column!r} contains invalid split labels: {invalid}. "
                "Use TRAINING or VALIDATION."
            )
        train_df = df[row_split == "TRAINING"].reset_index(drop=True)
        val_df = df[row_split == "VALIDATION"].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(
                f"Manifest column {split_column!r} must include at least one TRAINING "
                "row and one VALIDATION row."
            )
        train_datasets = set(int(x) for x in train_df["dataset_id"].unique())
        val_datasets = set(int(x) for x in val_df["dataset_id"].unique())
        print(f"Using row-level manifest split column: {split_column}", flush=True)
        split_description = "Row-level manifest split"
    else:
        np.random.seed(args.seed)
        shuffled_datasets = np.array(unique_datasets, dtype=np.int64)
        np.random.shuffle(shuffled_datasets)
        n_train_datasets = int(len(shuffled_datasets) * args.train_frac)
        train_datasets = set(int(x) for x in shuffled_datasets[:n_train_datasets])
        val_datasets = set(int(x) for x in shuffled_datasets[n_train_datasets:])
        train_df = df[df['dataset_id'].isin(train_datasets)].reset_index(drop=True)
        val_df = df[df['dataset_id'].isin(val_datasets)].reset_index(drop=True)
        split_description = "Dataset-stratified split"

    print(f"{split_description}: {len(train_datasets)} train datasets, {len(val_datasets)} val datasets", flush=True)
    print(f"  Train: {len(train_df)} micrographs from datasets {sorted(train_datasets)}", flush=True)
    print(f"  Val: {len(val_df)} micrographs from datasets {sorted(val_datasets)}", flush=True)
    print(
        f"  Micrograph split: train={len(train_df)}/{len(df)} ({len(train_df)/max(1, len(df))*100:.1f}%), "
        f"val={len(val_df)}/{len(df)} ({len(val_df)/max(1, len(df))*100:.1f}%)",
        flush=True,
    )

    # Load actual particle annotations (positive labels) if provided
    particles_dict = {}
    particles_dict_normalized = {}  # Additional key lookups
    if args.particle_annotations:
        particles_dict = load_particle_annotations(args.particle_annotations)
        if particles_dict:
            # Build multiple key lookups for flexible matching
            # Keys in file might be: "dataset_id__stem", "stem", "dataset_id_stem", etc.
            for k, v in particles_dict.items():
                # Original key
                particles_dict_normalized[k] = v
                # Try stripping any dataset_id prefix if present
                if "__" in k:
                    stem_part = k.split("__", 1)[1]
                    particles_dict_normalized[stem_part] = v
                if "_" in k and not "__" in k:
                    # Could be dataset_id_stem format
                    parts = k.split("_", 1)
                    if parts[0].isdigit():  # Looks like dataset_id
                        stem_part = parts[1]
                        particles_dict_normalized[stem_part] = v
                        # Also store in dataset_id__stem format
                        norm_key = f"{parts[0]}__{parts[1]}"
                        particles_dict_normalized[norm_key] = v

            # Match train micrographs with multiple key formats
            matched = 0
            sample_manifest_keys = []
            for _, row in train_df.iterrows():
                dataset_id = str(row['dataset_id'])
                stem = str(row['stem'])
                # Try multiple key formats
                keys_to_try = [
                    f"{dataset_id}__{stem}",
                    stem,
                    f"{dataset_id}_{stem}",
                ]
                found = False
                for key in keys_to_try:
                    if key in particles_dict_normalized:
                        matched += 1
                        found = True
                        break
                if not found and len(sample_manifest_keys) < 5:
                    sample_manifest_keys.append(f"{dataset_id}__{stem}")

            print(f"  Matched particle annotations: {matched}/{len(train_df)} train micrographs", flush=True)
            if matched == 0 and sample_manifest_keys:
                print(f"  Sample manifest keys (not matched): {sample_manifest_keys}", flush=True)

            # Use the normalized dict for lookups
            particles_dict = particles_dict_normalized

    # Compute region rules from ALL data (train+val) so validation datasets have valid thresholds
    # Without this, validation datasets return inf min_area and GT becomes all zeros
    all_rows = [(str(r["dataset_id"]), str(r["gt_mask_path"])) for _, r in df.iterrows()]  # Use full df
    rule = compute_dataset_min_area_px_from_train_split(
        all_rows,
        min_area_floor_px=effective_min_area,
        fill_holes=bool(args.region_gt_fill_holes),
    )
    save_dataset_region_rule(rule, out_dir)
    print(f"  Region-GT hole filling: {'on' if bool(args.region_gt_fill_holes) else 'off'}", flush=True)

    mic_dir = Path(args.mic_dir)
    cache_base = Path(args.cache_dir) if args.cache_dir else (out_dir / "patch_cache")
    cache_train = cache_base / "train"
    cache_val = cache_base / "val"
    cache_downsample = downsample_factors if downsample_factors is not None else downsample_factor
    nw = max(0, int(args.num_workers))
    val_patches_per_mic = max(4, args.patches_per_mic // 4)
    val_loader = None

    def _ensure_patch_cache_and_val_loader() -> DataLoader:
        nonlocal val_loader
        if val_loader is not None:
            return val_loader

        train_cache_complete = cache_train.exists() and len(list(cache_train.glob("mic_*.npz"))) >= len(train_df)
        if not train_cache_complete:
            n_existing = len(list(cache_train.glob("mic_*.npz"))) if cache_train.exists() else 0
            print(f"Building train cache ({n_existing}/{len(train_df)} exist, need all)...", flush=True)
            build_patch_cache(
                train_df, mic_dir, rule,
                normalization_method=args.normalization_method,
                psd_use_radial_normalization=args.psd_use_radial_normalization,
                downsample_factor=cache_downsample,
                cache_dir=cache_train,
                desc="train",
                psd_use_radial_image=args.psd_use_radial_image,
                psd_radial_bins=args.psd_radial_bins,
                use_edge_features=args.use_edge_features,
                edge_blend_alpha=args.edge_blend_alpha,
                edge_sigma=args.edge_sigma,
                psd_multiscale=args.psd_multiscale,
                psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
                psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
                psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
                psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
                psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
                psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
                psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
                pixel_size_by_dataset=pixel_size_by_dataset,
                psd_texture_mode=args.psd_texture_mode,
                psd_highpass_cutoff=args.psd_highpass_cutoff,
                psd_multires_context=args.psd_multires_context,
                use_global_features=args.use_global_features,
                use_edge_direction=args.use_edge_direction,
                adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "zoom"),
            )

        val_cache_complete = cache_val.exists() and len(list(cache_val.glob("mic_*.npz"))) >= len(val_df)
        if not val_cache_complete:
            print("Building val cache...", flush=True)
            build_patch_cache(
                val_df, mic_dir, rule,
                normalization_method=args.normalization_method,
                psd_use_radial_normalization=args.psd_use_radial_normalization,
                downsample_factor=cache_downsample,
                cache_dir=cache_val,
                desc="val",
                psd_use_radial_image=args.psd_use_radial_image,
                psd_radial_bins=args.psd_radial_bins,
                use_edge_features=args.use_edge_features,
                edge_blend_alpha=args.edge_blend_alpha,
                edge_sigma=args.edge_sigma,
                psd_multiscale=args.psd_multiscale,
                psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
                psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
                psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
                psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
                psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
                psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
                psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
                pixel_size_by_dataset=pixel_size_by_dataset,
                psd_texture_mode=args.psd_texture_mode,
                psd_highpass_cutoff=args.psd_highpass_cutoff,
                psd_multires_context=args.psd_multires_context,
                use_global_features=args.use_global_features,
                use_edge_direction=args.use_edge_direction,
                adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "zoom"),
            )

        val_ds = FastPatchDataset(
            val_df, mic_dir, rule,
            normalization_method=args.normalization_method,
            psd_use_radial_normalization=args.psd_use_radial_normalization,
            patch_size=effective_patch_size,
            patches_per_mic=val_patches_per_mic,
            gt_bias=0.0,
            downsample_factor=downsample_factor,
            seed=args.seed + 999,
            use_augmentation=False,
            cache_dir=cache_val,
            psd_use_radial_image=args.psd_use_radial_image,
            psd_radial_bins=args.psd_radial_bins,
            use_edge_features=args.use_edge_features,
            edge_blend_alpha=args.edge_blend_alpha,
            edge_sigma=args.edge_sigma,
            psd_multiscale=args.psd_multiscale,
            psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
            psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
            psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
            psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
            psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
            psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
            psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
            psd_texture_mode=args.psd_texture_mode,
            psd_highpass_cutoff=args.psd_highpass_cutoff,
            psd_multires_context=args.psd_multires_context,
            psd_context_sizes=tuple(int(x) for x in args.psd_context_sizes.split(",")),
            psd_bins_per_scale=args.psd_bins_per_scale,
            use_global_features=args.use_global_features,
            global_feature_dim=args.global_feature_dim,
            use_edge_direction=args.use_edge_direction,
            edge_direction_dim=args.edge_direction_dim,
            use_pixel_size_channel=args.use_pixel_size_channel,
            pixel_size_by_dataset=pixel_size_by_dataset,
            pixel_size_channel_min_angstrom=args.pixel_size_channel_min_angstrom,
            pixel_size_channel_max_angstrom=args.pixel_size_channel_max_angstrom,
        )
        val_ds.use_psd_input = getattr(args, 'use_psd_bool', True)
        val_ds.include_real_space_input = getattr(args, "include_real_space_input", True)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=min(2, nw) if nw else 0,
            pin_memory=True, persistent_workers=False, prefetch_factor=2 if nw else None,
        )
        return val_loader

    if getattr(args, "full_micrograph_training", False) and getattr(args, "skip_patch_cache_in_full_mic", False):
        print("  Patch cache + patch validation loader skipped for full-micrograph mode.", flush=True)
    else:
        _ensure_patch_cache_and_val_loader()

    # Fixed stitched-val set (primary signal for checkpoint selection)
    fixed_stitched_set = _build_fixed_stitched_val_set(
        val_df, mic_dir, rule, n_per_dataset=int(args.stitched_val_mics_per_dataset)
    )
    print(f"  Stitched val: {len(fixed_stitched_set)} fixed micrographs (1–{args.stitched_val_mics_per_dataset} per dataset)", flush=True)

    input_metadata_kwargs = dict(
        use_power_spectrum=getattr(args, "use_psd_bool", True),
        include_real_space_input=getattr(args, "include_real_space_input", True),
        use_pixel_size_channel=bool(args.use_pixel_size_channel),
        pixel_size_channel_min_angstrom=float(args.pixel_size_channel_min_angstrom),
        pixel_size_channel_max_angstrom=float(args.pixel_size_channel_max_angstrom),
        input_channels=int(getattr(args, "resolved_input_channels", 2 if getattr(args, "use_psd_bool", True) else 1)),
        psd_multiscale=bool(args.psd_multiscale),
        psd_multiscale_separate_channels=bool(getattr(args, "psd_multiscale_separate_channels", False)),
        psd_scales=tuple(int(x) for x in args.psd_scales_tuple),
        psd_frequency_band_channels=bool(getattr(args, "psd_frequency_band_channels", False)),
        psd_frequency_bands=tuple(getattr(args, "psd_frequency_bands_tuple", tuple())),
        psd_frequency_band_include_full_spectrum=bool(getattr(args, "psd_frequency_band_include_full_spectrum", False)),
        psd_frequency_band_include_anisotropy=bool(getattr(args, "psd_frequency_band_include_anisotropy", False)),
        psd_frequency_band_anisotropy_min_freq=float(getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0)),
    )

    # Load model: single- or two-branch MIFFI
    if args.train_from_scratch:
        # Train from scratch
        print(f"  Training from scratch with model_type={args.model_type}", flush=True)

        # For MIFFI, use pretrained weights; for other models, random init
        miffi_weights = None if args.model_type == "miffi" else "skip"

        _use_psd = getattr(args, 'use_psd_bool', True)
        model = _create_model_compat(
            model_type=args.model_type,
            device=str(args.device),
            use_power_spectrum=_use_psd,
            input_channels_override=int(getattr(args, "resolved_input_channels", 2 if _use_psd else 1)),
            num_classes=1,
            use_two_branch_encoder=args.use_two_branch_encoder if args.model_type == "miffi" else False,
            miffi_weights_path=miffi_weights,
            use_global_features=args.use_global_features,
            global_feature_dim=args.global_feature_dim,
            use_edge_direction=args.use_edge_direction,
            edge_direction_dim=args.edge_direction_dim,
            norm_type=getattr(args, 'norm_type', 'group'),
            decoder_dropout=getattr(args, 'decoder_dropout', 0.0),
            attention_type=getattr(args, 'attention_type', 'global'),
            dual_window_size=getattr(args, 'dual_window_size', 8),
            dual_edge_mask=getattr(args, 'dual_edge_mask', 0.15),
        )

        if args.model_type == "miffi":
            print(f"  ✓ Initialized real-space branch from MIFFI pretrained weights", flush=True)
            if args.use_two_branch_encoder:
                print(f"  ✓ PSD branch initialized randomly (no pretrained PSD encoder exists)", flush=True)
        else:
            print(f"  ✓ Model initialized with random weights", flush=True)
        _attach_model_input_metadata(
            model,
            **input_metadata_kwargs,
        )
    else:
        # Load from checkpoint
        if args.pretrained_model is None:
            raise ValueError("Must provide --pretrained_model or --train_from_scratch")

        ckpt = torch.load(args.pretrained_model, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if not isinstance(state, dict):
            state = {}
        else:
            state = _strip_module_prefix(dict(state))
        checkpoint_use_psd = bool(ckpt.get("use_power_spectrum", getattr(args, "use_psd_bool", True)))
        if checkpoint_use_psd != getattr(args, "use_psd_bool", True):
            print(
                f"  WARNING: checkpoint use_power_spectrum={checkpoint_use_psd} "
                f"but run requested use_psd={getattr(args, 'use_psd_bool', True)}. "
                f"Using run configuration.",
                flush=True,
            )
        resume_input_channels = int(getattr(args, "resolved_input_channels", _infer_input_channels_from_state_dict(state) or 2))
        checkpoint_input_channels = _infer_input_channels_from_state_dict(state)
        print(
            f"  Resuming with target input_channels={resume_input_channels} "
            f"(checkpoint had {checkpoint_input_channels if checkpoint_input_channels is not None else 'unknown'})",
            flush=True,
        )
        state = _expand_first_conv_input_channels(state, resume_input_channels)
        resume_model_type = ckpt.get("model_type", args.model_type)
        resume_attention_type = ckpt.get("attention_type", getattr(args, "attention_type", "global"))

        model = _create_model_compat(
            model_type=resume_model_type,
            device=str(args.device),
            use_power_spectrum=getattr(args, "use_psd_bool", True),
            input_channels_override=int(resume_input_channels),
            num_classes=1,
            use_two_branch_encoder=args.use_two_branch_encoder if resume_model_type == "miffi" else False,
            miffi_weights_path="skip",  # Skip pretrained weights, load from checkpoint instead
            use_global_features=args.use_global_features,
            global_feature_dim=args.global_feature_dim,
            use_edge_direction=args.use_edge_direction,
            edge_direction_dim=args.edge_direction_dim,
            norm_type=getattr(args, 'norm_type', 'group'),
            decoder_dropout=getattr(args, 'decoder_dropout', 0.0),
            attention_type=resume_attention_type,
            dual_window_size=getattr(args, 'dual_window_size', 8),
            dual_edge_mask=getattr(args, 'dual_edge_mask', 0.15),
        )
        _attach_model_input_metadata(
            model,
            **input_metadata_kwargs,
        )

        # Load checkpoint: strip final/binary_head if shape mismatch
        model_sd = model.state_dict()
        for k in list(state.keys()):
            if k in model_sd and state[k].shape != model_sd[k].shape:
                state.pop(k, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if "epoch" in ckpt and "val_loss" in ckpt:
            print(f"Resuming from our best: epoch {ckpt.get('epoch', '?')} val_loss={ckpt.get('val_loss', 0):.6f}", flush=True)
        else:
            print(f"Loaded checkpoint (final/binary_head reinit if shape differed)", flush=True)
        print(f"Loaded (strict=False): missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.num_gpus > 1:
        if not str(args.device).startswith("cuda"):
            raise ValueError("--num_gpus > 1 requires --device cuda")
        visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_cuda < args.num_gpus:
            raise ValueError(
                f"Requested --num_gpus={args.num_gpus} but only {visible_cuda} CUDA device(s) are visible."
            )
        device_ids = list(range(int(args.num_gpus)))
        model = nn.DataParallel(model, device_ids=device_ids)
        _attach_model_input_metadata(model, **input_metadata_kwargs)
        print(f"  ✓ DataParallel enabled across CUDA devices {device_ids}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _checkpoint_payload(epoch_value: int, **extra_metrics) -> dict:
        payload = {
            "epoch": int(epoch_value),
            "model_state_dict": _model_state_dict(model),
            "optimizer_state_dict": opt.state_dict(),
            "use_power_spectrum": getattr(args, "use_psd_bool", True),
            "include_real_space_input": getattr(args, "include_real_space_input", True),
            "use_pixel_size_channel": bool(args.use_pixel_size_channel),
            "pixel_size_channel_min_angstrom": float(args.pixel_size_channel_min_angstrom),
            "pixel_size_channel_max_angstrom": float(args.pixel_size_channel_max_angstrom),
            "psd_multiscale": bool(args.psd_multiscale),
            "psd_multiscale_separate_channels": bool(getattr(args, "psd_multiscale_separate_channels", False)),
            "psd_scales": [int(x) for x in args.psd_scales.split(",")],
            "psd_frequency_band_channels": bool(getattr(args, "psd_frequency_band_channels", False)),
            "psd_frequency_bands": [list(map(float, band)) for band in getattr(args, "psd_frequency_bands_tuple", tuple())],
            "psd_frequency_band_include_full_spectrum": bool(
                getattr(args, "psd_frequency_band_include_full_spectrum", False)
            ),
            "psd_frequency_band_include_anisotropy": bool(getattr(args, "psd_frequency_band_include_anisotropy", False)),
            "psd_frequency_band_anisotropy_min_freq": float(getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0)),
            "normalization_method": str(getattr(args, "normalization_method", "")),
            "full_mic_train_normalization_methods": list(getattr(args, "full_mic_train_normalization_methods_list", [])),
            "full_mic_train_normalization_probs": (
                list(getattr(args, "full_mic_train_normalization_probs_list", []) or [])
            ),
            "full_mic_hard_positive_fraction": float(getattr(args, "full_mic_hard_positive_fraction", 0.0)),
            "full_mic_hard_positive_min_gt_frac": float(getattr(args, "full_mic_hard_positive_min_gt_frac", 0.10)),
            "full_mic_hard_positive_erode_px": int(getattr(args, "full_mic_hard_positive_erode_px", 0)),
            "positive_interior_weight": float(getattr(args, "positive_interior_weight", 0.0)),
            "positive_interior_erode_px": int(getattr(args, "positive_interior_erode_px", 0)),
            "input_channels": int(getattr(args, "resolved_input_channels", 2)),
            "attention_type": getattr(args, "attention_type", "global"),
            "model_type": args.model_type,
        }
        payload.update(extra_metrics)
        return payload

    total_train_batches = (len(train_df) * args.patches_per_mic) // args.batch_size
    total_steps = args.num_epochs * max(1, total_train_batches)
    warmup_steps = min(int(args.warmup_steps), max(1, total_steps // 10))

    def _lr_cosine_warmup(step: int) -> float:
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    print(f"Training on {len(train_df)*args.patches_per_mic} patches ({len(train_df)} micrographs × {args.patches_per_mic} patches/mic)", flush=True)
    print(f"  Effective patch_size={effective_patch_size}, core_erosion={effective_core_erosion}", flush=True)
    if getattr(args, 'gt_bias_decay', False):
        print(f"  GT-bias={args.gt_bias} → {args.gt_bias_final} (decay from epoch {args.gt_bias_decay_start} to {args.gt_bias_decay_end})", flush=True)
        print(f"    → CRITICAL: Model learns features first, then learns true contamination distribution!", flush=True)
    else:
        if getattr(args, "full_micrograph_training", False) and getattr(args, "skip_patch_cache_in_full_mic", False):
            print(f"  GT-bias={args.gt_bias} (deterministic per epoch: seed+epoch); patch cache skipped in full-mic mode", flush=True)
        else:
            print(f"  GT-bias={args.gt_bias} (deterministic per epoch: seed+epoch); cache={cache_train}", flush=True)
    print(f"  Val: {len(val_df)} images × {val_patches_per_mic} patches/mic (fixed patches every epoch)", flush=True)
    print(f"  LR: warmup {warmup_steps} steps then cosine decay over {total_steps} steps", flush=True)
    # Early stopping: use smooth surrogate metric instead of binary recall@FP
    # Mode 1: score = PR-AUC - α * fp_mass (smooth surrogate that correlates with goal)
    # Mode 2: monitor recall at multiple FP caps (0.5%, 1%, 2%, 5%, 15%) - stop only if all plateau
    best_stitched_score = -1.0  # PR-AUC - α * fp_mass
    best_stitched_pr_auc = -1.0
    best_stitched_fp_mass = 1.0
    best_multi_fp_recalls = {}  # Track best recall at each FP cap
    epochs_without_improvement_stitched = 0
    global_step = 0

    early_stop_mode_str = args.early_stop_mode
    stitched_val_cycle_count = 0  # Track number of stitched-val cycles completed
    effective_particle_neg_margin = max(1, int(args.particle_neg_margin_px / downsample_factor))
    effective_particle_neg_margin_wide = max(1, int(args.particle_neg_margin_wide_px / downsample_factor))

    if getattr(args, "full_micrograph_training", False):
        print(
            f"  Early stop patience={args.early_stop_patience} stitched cycles "
            f"(min {args.early_stop_min_epochs} epochs or {args.early_stop_min_stitched_cycles} cycles) "
            f"| mode=PR-AUC (official) + score secondary (PR-AUC-{args.early_stop_alpha}*FP-mass)",
            flush=True,
        )
    elif early_stop_mode_str == "score":
        print(f"  Early stop patience={args.early_stop_patience} stitched cycles (min {args.early_stop_min_epochs} epochs or {args.early_stop_min_stitched_cycles} cycles) | mode=score, PR-AUC-{args.early_stop_alpha}*FP-mass", flush=True)
    elif early_stop_mode_str == "composite":
        print(f"  Early stop patience={args.early_stop_patience} stitched cycles (min {args.early_stop_min_epochs} epochs or {args.early_stop_min_stitched_cycles} cycles) | mode=composite, 0.5*Recall@2% + 0.5*Recall@5%", flush=True)
    else:
        print(f"  Early stop patience={args.early_stop_patience} stitched cycles (min {args.early_stop_min_epochs} epochs or {args.early_stop_min_stitched_cycles} cycles) | mode=multi_fp_cap, monitors recall@FP 0.5%/1%/2%/5%/15%", flush=True)
    if getattr(args, 'use_dice_loss', False):
        print(f"  ✓ DICE + FOCAL LOSS (medical imaging standard) - SELF-BALANCING, cannot collapse!", flush=True)
        print(f"    λ_dice={args.lambda_dice} λ_focal={args.lambda_focal} (γ={args.focal_gamma}, α={args.focal_alpha}) λ_edge={args.lambda_edge}", flush=True)
        print(f"    → Dice: Only way to maximize is to precisely match GT shape", flush=True)
        print(f"    → Focal: Confident predictions stop contributing to training", flush=True)
        # Edge erosion for patch boundary artifact prevention
        edge_erosion = getattr(args, 'edge_erosion_px', 0)
        if edge_erosion > 0:
            effective_edge = max(1, int(edge_erosion / downsample_factor))
            use_tukey = getattr(args, 'use_tukey_weight', True)
            window_type = "Tukey (smooth taper)" if use_tukey else "hard erosion"
            print(f"  ✓ EDGE EROSION: {edge_erosion}px full-res → {effective_edge}px effective ({window_type})", flush=True)
            print(f"    → Prevents 'edge of context' artifacts (swaths at patch boundaries)", flush=True)
        if getattr(args, "positive_interior_weight", 0.0) > 0.0:
            effective_pos_erode = max(1, int(args.positive_interior_erode_px / downsample_factor))
            print(
                f"  ✓ POSITIVE INTERIOR BOOST: weight={args.positive_interior_weight:.3f}, "
                f"erode={args.positive_interior_erode_px}px full-res → {effective_pos_erode}px effective",
                flush=True,
            )
    else:
        print(f"  λ_bce={args.lambda_bce} (neg_weight={args.bce_neg_weight}) λ_hit={args.lambda_hit_core} λ_out={args.lambda_outside_mass} λ_fpcap={args.lambda_fpcap} λ_edge={args.lambda_edge}", flush=True)
        if args.loss_reweight_epoch > 0:
            print(f"  ✓ DYNAMIC LOSS REWEIGHTING at epoch {args.loss_reweight_epoch}:", flush=True)
            print(f"    Phase 1 (DISCOVERY): Find contamination with full loss weights", flush=True)
            print(f"    Phase 2 (REFINEMENT): λ_bce→{args.phase2_lambda_bce}, λ_hit→{args.phase2_lambda_hit_core}, λ_out→{args.phase2_lambda_outside_mass}, λ_fpcap→{args.phase2_lambda_fpcap}", flush=True)
            print(f"    → Prevents 'predict everything as contamination' collapse by flipping vote balance", flush=True)
    if args.use_edge_direction:
        print(f"  ✓ EDGE DIRECTION FEATURES enabled (3 scalars → {args.edge_direction_dim}-dim)", flush=True)
        print(f"    → Features: direction_strength + direction_angle + edge_proximity (patch-specific!)", flush=True)
    elif args.use_global_features:
        print(f"  ✓ GLOBAL FEATURES enabled ({args.global_feature_dim}-dim) - helps classify uniform regions!", flush=True)
        print(f"    → Features: global PSD (32) + histogram (50) + stats (7) + spatial variance (64) + edge density (1)", flush=True)
    if args.psd_multires_context:
        print(f"  ✓ MULTI-RES CONTEXT PSD enabled (contexts={args.psd_context_sizes}, bins/scale={args.psd_bins_per_scale})", flush=True)
        print(f"    → Captures LARGE carbon edges by computing PSD on larger regions centered on each patch", flush=True)
    elif getattr(args, "psd_frequency_band_channels", False):
        input_mode = "PSD-only" if not getattr(args, "include_real_space_input", True) else "image+PSD"
        print(
            f"  ✓ MULTI-FREQUENCY PSD enabled ({input_mode}, input_channels={getattr(args, 'resolved_input_channels', 'unknown')})",
            flush=True,
        )
        if getattr(args, "psd_frequency_bands_tuple", tuple()):
            print(f"    → Bands (Å^-1): {_format_frequency_bands(getattr(args, 'psd_frequency_bands_tuple', tuple()))}", flush=True)
        if getattr(args, "psd_frequency_band_include_full_spectrum", False):
            print("    → Full-spectrum 2D PSD channel enabled", flush=True)
        if getattr(args, "psd_frequency_band_include_anisotropy", False):
            print(
                f"    → Anisotropy/streak channel enabled (min_freq={getattr(args, 'psd_frequency_band_anisotropy_min_freq', 0.0):.5f} Å^-1)",
                flush=True,
            )
    elif args.psd_multiscale:
        texture_str = f" TEXTURE MODE (highpass>{args.psd_highpass_cutoff})" if args.psd_texture_mode else ""
        print(f"  ✓ MULTI-SCALE RADIAL PSD enabled (scales={args.psd_scales}){texture_str}", flush=True)
        if getattr(args, "psd_multiscale_separate_channels", False):
            input_mode = "PSD-only" if not getattr(args, "include_real_space_input", True) else "image+PSD"
            print(f"    → Separate PSD channels enabled ({input_mode}, input_channels={getattr(args, 'resolved_input_channels', 'unknown')})", flush=True)
        else:
            input_mode = "PSD-only" if not getattr(args, "include_real_space_input", True) else "image+PSD"
            print(f"    → Fused PSD channel mode ({input_mode}, input_channels={getattr(args, 'resolved_input_channels', 'unknown')})", flush=True)
        if args.psd_texture_mode:
            print(f"    → TEXTURE MODE: Focus on FINE texture (5-40 Å), suppress coarse (80-320 Å) where carbon/ice look same", flush=True)
    elif args.psd_use_radial_image:
        print(f"  ✓ RADIAL PSD IMAGE enabled ({args.psd_radial_bins} bins) - stronger contamination signal!", flush=True)
    if args.use_edge_features:
        print(f"  ✓ EDGE FEATURES enabled (alpha={args.edge_blend_alpha}, sigma={args.edge_sigma}) - helps catch carbon edges!", flush=True)
    if args.use_pu_loss and particles_dict:
        gaussian_str = "Gaussian" if args.particle_use_gaussian else "disk"
        print(f"  PU-STYLE LOSS ENABLED ({gaussian_str} particle mask, box={args.particle_box_size_px}px)", flush=True)
        print(f"    Curriculum: Phase A (bad only) epochs 1-{args.phase_a_epochs}, Phase B (+good,area) {args.phase_a_epochs+1}-{args.phase_b_epochs}, Phase C (+margin) {args.phase_b_epochs+1}+", flush=True)
        print(f"    λ_good={args.lambda_good}, λ_margin={args.lambda_margin}, λ_area={args.lambda_area}", flush=True)
        print(f"    tau_good={args.tau_good}, margin_m={args.margin_m}, rho_area={args.rho_area}→{args.rho_area_final}", flush=True)
        if args.lambda_entropy > 0 or args.lambda_particle_asym > 0:
            print(f"    λ_entropy={args.lambda_entropy} (target range {args.entropy_target_low}-{args.entropy_target_high}), λ_particle_asym={args.lambda_particle_asym}", flush=True)
    elif particles_dict:
        print(f"  Particle-neg margins: {args.particle_neg_margin_px}px full-res → {effective_particle_neg_margin}px effective (downsample={downsample_factor})", flush=True)
    if args.particle_neg_margin_wide_px > 0:
        print(f"  Particle-neg wide margin: {args.particle_neg_margin_wide_px}px full-res → {effective_particle_neg_margin_wide}px effective (λ={args.lambda_particle_neg_wide})", flush=True)
    print(f"  Hard-negative sampling: {args.hard_neg_fraction*100:.0f}% of negatives from top {args.hard_neg_quantile*100:.0f}% quantile", flush=True)
    if getattr(args, "full_mic_hard_positive_fraction", 0.0) > 0.0:
        print(
            f"  Full-mic hard-positive sampling: {args.full_mic_hard_positive_fraction*100:.0f}% of capped patches, "
            f"min GT frac={args.full_mic_hard_positive_min_gt_frac:.3f}, "
            f"erode={args.full_mic_hard_positive_erode_px}px full-res",
            flush=True,
        )
    print(f"  Note: If hit-core is near zero, consider reducing core_erosion_px or λ_hit_core. Once L_fpcap is active, consider downweighting λ_outside_mass.", flush=True)

    # =========================================================================
    # STITCHED TRAINING LOSS SETUP
    # Build list of training micrographs for stitched loss (forces model to see TRUE distribution)
    # =========================================================================
    stitched_train_mics = []
    if getattr(args, 'use_stitched_train_loss', False):
        print(f"\n  ✓ STITCHED TRAINING LOSS ENABLED (λ={args.lambda_stitched})", flush=True)
        print(f"    → Forces model to see REAL distribution every {args.stitched_train_every_n_batches} batches", flush=True)
        print(f"    → {args.stitched_train_mics_per_batch} mic(s) × {args.stitched_train_grid_patches} patches sparse grid", flush=True)
        print(f"    → This addresses train/val misalignment that causes collapse!", flush=True)

        # Build list of (mic_path, gt_path, dataset_id, downsample_factor)
        for _, row in train_df.iterrows():
            dataset_id = str(row['dataset_id'])
            stem = str(row['stem'])

            # Get micrograph path - try multiple column names
            mic_path = None
            if 'micrograph_path' in row and pd.notna(row['micrograph_path']):
                mic_path = Path(row['micrograph_path'])
            elif 'full_mic_path' in row and pd.notna(row['full_mic_path']):
                mic_path = Path(row['full_mic_path'])
            else:
                # Try to construct from mic_dir + stem
                for ext in ['.mrc', '.tif', '.tiff', '.MRC']:
                    candidate = mic_dir / f"{dataset_id}_{stem}{ext}"
                    if candidate.exists():
                        mic_path = candidate
                        break

            if mic_path is None or not mic_path.exists():
                continue

            # Find GT path - try multiple column names
            gt_path = None
            for col in ['gt_mask_path', 'gt_mask', 'gt_mask_full_path']:
                if col in row and pd.notna(row[col]):
                    gt_path_str = str(row[col])
                    for candidate in [
                        Path(gt_path_str),
                        mic_dir.parent / gt_path_str,
                        Path("particle_annotations/merged_gt_fullres") / f"{dataset_id}__{stem}.npy",
                    ]:
                        if candidate.exists():
                            gt_path = candidate
                            break
                    if gt_path:
                        break

            if gt_path is None or not gt_path.exists():
                continue

            # Get downsample factor
            ds_factor = downsample_factors.get(dataset_id, downsample_factor) if downsample_factors else downsample_factor
            stitched_train_mics.append((mic_path, gt_path, dataset_id, ds_factor))

        print(f"    → {len(stitched_train_mics)} training micrographs available for stitched loss", flush=True)
        if len(stitched_train_mics) == 0:
            print(f"    WARNING: No training micrographs found for stitched loss! Disabling.", flush=True)
            args.use_stitched_train_loss = False

    # =========================================================================
    # FULL MICROGRAPH TRAINING MODE (MicrographCleaner-style)
    # =========================================================================
    # If enabled, this completely replaces the normal patch-based training loop.
    # Processes full micrographs with dense overlapping patches (NO GT-bias).
    # This is the PROVEN approach from MicrographCleaner (Sanchez-Garcia et al. 2020).
    # =========================================================================
    full_mic_train_mics = []
    if getattr(args, 'full_micrograph_training', False):
        print(f"\n{'='*80}", flush=True)
        print(f"FULL MICROGRAPH TRAINING MODE (MicrographCleaner-style)", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  → Processing ENTIRE micrographs with dense overlapping patches", flush=True)
        print(f"  → NO GT-bias: Training sees TRUE contamination distribution", flush=True)
        mics_per_epoch = getattr(args, 'full_mic_mics_per_epoch', 0)
        if mics_per_epoch <= 0:
            print(f"  → ALL micrographs every epoch (stable training loss, better convergence)", flush=True)
        else:
            print(f"  → {mics_per_epoch} mics/epoch (random sampling - may cause loss variance)", flush=True)
        print(f"  → {args.full_mic_patches_per_mic} patches/mic max, {args.full_mic_overlap*100:.0f}% overlap", flush=True)
        edge_erosion = getattr(args, 'edge_erosion_px', 0)
        if edge_erosion > 0:
            effective_edge = max(1, int(edge_erosion / downsample_factor))
            use_tukey = getattr(args, 'use_tukey_weight', True)
            window_type = "Tukey (smooth taper)" if use_tukey else "hard erosion"
            print(f"  → EDGE EROSION: {edge_erosion}px → {effective_edge}px effective ({window_type})", flush=True)
        print(f"  → Per-epoch val: gt_bias=0.0 (aligned with training distribution!)", flush=True)
        print(f"  → Requires more epochs (~100) but PREVENTS collapse!", flush=True)
        print(f"{'='*80}\n", flush=True)

        # Build list of training micrographs (reuse stitched_train_mics if available)
        if stitched_train_mics:
            full_mic_train_mics = stitched_train_mics
        else:
            # Build from train_df
            for _, row in train_df.iterrows():
                dataset_id = str(row['dataset_id'])
                stem = str(row['stem'])

                # Get micrograph path - try multiple column names
                mic_path = None
                if 'micrograph_path' in row and pd.notna(row['micrograph_path']):
                    mic_path = Path(row['micrograph_path'])
                elif 'full_mic_path' in row and pd.notna(row['full_mic_path']):
                    mic_path = Path(row['full_mic_path'])
                else:
                    # Try to construct from mic_dir + stem
                    for ext in ['.mrc', '.tif', '.tiff', '.MRC']:
                        candidate = mic_dir / f"{dataset_id}_{stem}{ext}"
                        if candidate.exists():
                            mic_path = candidate
                            break

                if mic_path is None or not mic_path.exists():
                    continue

                # Get GT path - try multiple column names
                gt_path = None
                for col in ['gt_mask_path', 'gt_mask', 'gt_mask_full_path']:
                    if col in row and pd.notna(row[col]):
                        gt_path_str = str(row[col])
                        for candidate in [
                            Path(gt_path_str),
                            mic_dir.parent / gt_path_str,
                            Path("particle_annotations/merged_gt_fullres") / f"{dataset_id}__{stem}.npy",
                        ]:
                            if candidate.exists():
                                gt_path = candidate
                                break
                        if gt_path:
                            break

                if gt_path is None or not gt_path.exists():
                    continue

                ds_factor = downsample_factors.get(dataset_id, downsample_factor) if downsample_factors else downsample_factor
                full_mic_train_mics.append((mic_path, gt_path, dataset_id, ds_factor))

        # Report actual mics per epoch (now that we know how many are available)
        if args.full_mic_mics_per_epoch <= 0:
            print(f"  → ALL {len(full_mic_train_mics)} mics/epoch × {args.full_mic_patches_per_mic} patches/mic (stable training!)", flush=True)
        else:
            actual_mics = min(args.full_mic_mics_per_epoch, len(full_mic_train_mics))
            print(f"  → {actual_mics}/{len(full_mic_train_mics)} mics/epoch × {args.full_mic_patches_per_mic} patches/mic", flush=True)

        if len(full_mic_train_mics) == 0:
            print(f"  WARNING: No training micrographs found! Falling back to patch-based training.", flush=True)
            args.full_micrograph_training = False
        else:
            # Compute stride from overlap
            stride = int(effective_patch_size * (1 - args.full_mic_overlap))
            stride = max(1, stride)

            print(f"  → Effective stride: {stride}px ({args.full_mic_overlap*100:.0f}% overlap)", flush=True)
            _fullmic_aug = not getattr(args, 'no_fullmic_augmentation', False)
            print(f"  → Augmentation: {'flip+rot90 (8x diversity)' if _fullmic_aug else 'DISABLED'}", flush=True)

            # Compute correct LR schedule for full-mic mode
            # Each mic ≈ ceil(max_patches / batch_size) optimizer steps
            patches_per_mic_est = max(1, int(args.full_mic_patches_per_mic))
            steps_per_mic = max(1, math.ceil(patches_per_mic_est / args.batch_size))
            n_mics_per_epoch = len(full_mic_train_mics) if args.full_mic_mics_per_epoch <= 0 else min(args.full_mic_mics_per_epoch, len(full_mic_train_mics))
            fullmic_steps_per_epoch = n_mics_per_epoch * steps_per_mic
            fullmic_total_steps = args.num_epochs * fullmic_steps_per_epoch
            fullmic_warmup_steps = min(int(args.warmup_steps), max(1, fullmic_total_steps // 10))

            def _fullmic_lr_schedule(step: int) -> float:
                """Warmup + cosine decay LR schedule for full-mic mode."""
                if step < fullmic_warmup_steps:
                    return args.lr * step / max(1, fullmic_warmup_steps)
                progress = (step - fullmic_warmup_steps) / max(1, fullmic_total_steps - fullmic_warmup_steps)
                return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            fullmic_global_step = 0  # Track steps across epochs for LR schedule
            _fullmic_step_counter = [0]  # Mutable list so full_micrograph_training_epoch can update it

            print(f"  → LR schedule: warmup {fullmic_warmup_steps} steps, cosine decay over {fullmic_total_steps} steps ({fullmic_steps_per_epoch} steps/epoch)", flush=True)
            print(f"  → Early stopping: PR-AUC patience={args.early_stop_patience} stitched cycles (min {args.early_stop_min_epochs} epochs or {args.early_stop_min_stitched_cycles} cycles)", flush=True)
            print(f"\nStarting full micrograph training for {args.num_epochs} epochs...\n", flush=True)

            # Main training loop for full micrograph mode
            best_stitched_prauc = -1.0
            best_composite = -1.0
            best_prauc_only = -1.0  # Track best PR-AUC separately (may differ from best composite)
            best_deployment_score = float("-inf")
            patience_counter = 0

            for epoch in range(1, args.num_epochs + 1):
                epoch_start = time.perf_counter()

                # Run full micrograph training epoch
                # Apply LR schedule at epoch start (so it's printed correctly)
                current_lr = _fullmic_lr_schedule(fullmic_global_step)
                for g in opt.param_groups:
                    g["lr"] = current_lr

                train_loss, train_stats = full_micrograph_training_epoch(
                    model=model,
                    train_mics=full_mic_train_mics,
                    rule=rule,
                    device=args.device,
                    patch_size=effective_patch_size,
                    stride=stride,
                    normalization_method=args.normalization_method,
                    psd_multiscale=args.psd_multiscale,
                    psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
                    psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
                    psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
                    psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
                    psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
                    psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
                    psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
                    psd_texture_mode=getattr(args, 'psd_texture_mode', False),
                    psd_highpass_cutoff=getattr(args, 'psd_highpass_cutoff', 0.2),
                    batch_size=args.batch_size,
                    max_patches_per_mic=args.full_mic_patches_per_mic,
                    mics_per_epoch=args.full_mic_mics_per_epoch,
                    use_dice_loss=getattr(args, 'use_dice_loss', False),
                    lambda_dice=getattr(args, 'lambda_dice', 2.0),
                    lambda_focal=getattr(args, 'lambda_focal', 1.0),
                    focal_gamma=getattr(args, 'focal_gamma', 2.0),
                    focal_alpha=getattr(args, 'focal_alpha', 0.25),
                    lambda_bce=args.lambda_bce,
                    bce_neg_weight=args.bce_neg_weight,
                    lambda_edge=args.lambda_edge,
                    opt=opt,
                    lr_scheduler=None,  # Per-step LR handled via lr_schedule_fn
                    grad_clip_norm=args.grad_clip_norm,
                    epoch=epoch,
                    # NEW: Edge erosion parameters (now actually used in full_mic mode!)
                    edge_erosion_px=getattr(args, 'edge_erosion_px', 0),
                    use_tukey_weight=getattr(args, 'use_tukey_weight', True),
                    downsample_factor=downsample_factor,
                    # Anti-overfitting: label smoothing
                    label_smoothing=getattr(args, 'label_smoothing', 0.0),
                    # Per-step LR schedule (warmup + cosine decay over full training)
                    lr_schedule_fn=_fullmic_lr_schedule,
                    lr_step_counter=_fullmic_step_counter,  # Mutable list persists across epochs
                    # Augmentation: flip + rot90 for 8x training diversity
                    use_augmentation=not getattr(args, 'no_fullmic_augmentation', False),
                    # Ablation: whether to include PSD channels
                    use_psd=getattr(args, 'use_psd_bool', True),
                    include_real_space_input=getattr(args, "include_real_space_input", True),
                    use_pixel_size_channel=args.use_pixel_size_channel,
                    pixel_size_by_dataset=pixel_size_by_dataset,
                    pixel_size_channel_min_angstrom=args.pixel_size_channel_min_angstrom,
                    pixel_size_channel_max_angstrom=args.pixel_size_channel_max_angstrom,
                    prefetch_workers=getattr(args, "full_mic_prefetch_workers", 1),
                    adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "stride"),
                    train_normalization_methods=getattr(args, "full_mic_train_normalization_methods_list", None),
                    train_normalization_probs=getattr(args, "full_mic_train_normalization_probs_list", None),
                    hard_positive_fraction=getattr(args, "full_mic_hard_positive_fraction", 0.0),
                    hard_positive_min_gt_frac=getattr(args, "full_mic_hard_positive_min_gt_frac", 0.10),
                    hard_positive_erode_px=getattr(args, "full_mic_hard_positive_erode_px", 0),
                    positive_interior_weight=getattr(args, "positive_interior_weight", 0.0),
                    positive_interior_erode_px=getattr(args, "positive_interior_erode_px", 0),
                )

                # Update global step counter from the mutable list
                fullmic_global_step = _fullmic_step_counter[0]
                if train_stats.get("n_mics_total", 0) > 0 and train_stats.get("n_mics_processed", 0) == 0:
                    first_error = str(train_stats.get("first_skip_error") or "unknown error")
                    hint = (
                        " Reduce --batch_size and --stitched_val_batch_forward_size, or run on a GPU "
                        "with more free memory."
                        if "out of memory" in first_error.lower()
                        else ""
                    )
                    raise RuntimeError(
                        "Full-micrograph training processed 0/"
                        f"{train_stats.get('n_mics_total', 0)} micrographs in epoch {epoch:02d}; "
                        f"all micrographs were skipped. First skip error: {first_error}.{hint}"
                    )

                epoch_time = time.perf_counter() - epoch_start

                # Print epoch summary
                print(f"\n{'='*80}", flush=True)
                print(f"Epoch {epoch:02d}/{args.num_epochs} COMPLETE [Full-Mic Mode]", flush=True)
                print(f"{'='*80}", flush=True)
                current_lr = _fullmic_lr_schedule(fullmic_global_step)
                print(f"Train Loss: {train_loss:.6f} | Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min) | LR: {current_lr:.2e}", flush=True)
                dice_15 = train_stats.get('dice_at_0.15', 0.0)
                dice_20 = train_stats.get('dice_at_0.20', 0.0)
                dice_50 = train_stats['dice_coeff']
                print(f"  Dice @0.15: {dice_15:.4f} | @0.20: {dice_20:.4f} | @0.50: {dice_50:.4f}", flush=True)
                print(f"  Pos prob mean: {train_stats['pos_prob_mean']:.4f}", flush=True)
                print(f"  Neg prob mean: {train_stats['neg_prob_mean']:.4f}", flush=True)
                print(f"  Micrographs processed: {train_stats['n_mics_processed']} | Step: {fullmic_global_step}/{fullmic_total_steps}", flush=True)
                norm_counts = train_stats.get("normalization_counts", {})
                if norm_counts:
                    print(
                        "  Train normalization counts: "
                        + ", ".join(f"{k}={v}" for k, v in sorted(norm_counts.items())),
                        flush=True,
                    )

                # Run stitched validation
                if epoch % args.stitched_val_every_n_epochs == 0:
                    print(f"\n  [STITCHED VAL] Running on {len(fixed_stitched_set)} fixed micrographs...", flush=True)
                    # No export during training — saves ~2.5h per stitched val cycle
                    export_dir = None
                    # Use training patch size for consistency
                    val_patch_size = effective_patch_size
                    val_overlap = int(args.stitched_val_overlap)
                    # Parse excluded datasets from comma-separated string
                    exclude_datasets = [s.strip() for s in args.exclude_datasets_from_metrics.split(",") if s.strip()]

                    stitched_metrics = run_stitched_val_fast(
                        model, fixed_stitched_set, rule, args.device,
                        patch_size=val_patch_size,
                        overlap=val_overlap,
                        normalization_method=args.normalization_method,
                        psd_use_radial_normalization=args.psd_use_radial_normalization,
                        downsample_factor=args.stitched_val_downsample,  # Default fallback
                        min_area_px=args.stitched_val_min_area_px,
                        fp_area_cap_pct=args.stitched_val_fp_area_cap_pct,
                        opening_size=args.stitched_val_opening_size,
                        closing_size=args.stitched_val_closing_size,
                        export_dir=export_dir,
                        pixel_size_angstrom=args.stitched_val_pixel_size_angstrom,
                        pixel_size_per_dataset=pixel_size_by_dataset,
                        # Per-dataset adaptive downsampling (CRITICAL - must match training!)
                        downsample_per_dataset=downsample_factors,
                        # Multi-scale PSD parameters (must match training!)
                        psd_multiscale=args.psd_multiscale,
                        psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
                        psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
                        psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
                        psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
                        psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
                        psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
                        psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
                        psd_texture_mode=getattr(args, 'psd_texture_mode', False),
                        psd_highpass_cutoff=getattr(args, 'psd_highpass_cutoff', 0.2),
                        # Exclude certain datasets from metrics (but still generate visualizations)
                        exclude_datasets_from_metrics=exclude_datasets if exclude_datasets else None,
                        # Ablation: whether model uses PSD channels
                        use_psd=getattr(args, 'use_psd_bool', True),
                        include_real_space_input=getattr(args, "include_real_space_input", True),
                        use_pixel_size_channel=args.use_pixel_size_channel,
                        pixel_size_channel_min_angstrom=args.pixel_size_channel_min_angstrom,
                        pixel_size_channel_max_angstrom=args.pixel_size_channel_max_angstrom,
                        batch_forward_size=getattr(args, 'stitched_val_batch_forward_size', 1),
                        return_native_resolution=True,
                        blending_window=args.stitched_val_blending_window,
                        blending_edge_px=args.stitched_val_blending_edge_px,
                        adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "zoom"),
                        threshold_grid=getattr(args, "stitched_val_threshold_grid", None),
                        metrics_max_samples=getattr(args, "stitched_val_metrics_max_samples", 5_000_000),
                    )

                    prauc = stitched_metrics.get('pr_auc', 0.0)
                    auroc = stitched_metrics.get('auroc', 0.0)
                    fp_area = stitched_metrics.get('fp_area_pct_median', 1.0)
                    recall_2pct = stitched_metrics.get('recall_at_fp_2.0pct', 0.0)
                    recall_5pct = stitched_metrics.get('recall_at_fp_5.0pct', 0.0)
                    pos_median = stitched_metrics.get('prob_pos_median', 0.0)
                    neg_median = stitched_metrics.get('prob_neg_median', 1.0)

                    print(f"  [PRIMARY METRICS] Stitched PR-AUC: {prauc:.4f} | AUROC: {auroc:.4f}", flush=True)
                    print(f"  [PRIMARY METRICS] FP area median: {fp_area*100:.2f}% | Recall@2%: {recall_2pct:.4f} | Recall@5%: {recall_5pct:.4f}", flush=True)
                    print(f"  Prob hist (pos/neg median): {pos_median:.3f} / {neg_median:.3f}", flush=True)

                    # Composite metric (reported but NOT used for early stopping)
                    composite = 0.5 * recall_2pct + 0.5 * recall_5pct
                    deployment_score = prauc - args.early_stop_alpha * fp_area
                    print(f"  Composite (0.5*R@2%+0.5*R@5%): {composite:.4f}", flush=True)
                    print(
                        f"  Deployment score (PR-AUC - {args.early_stop_alpha}*FP area): "
                        f"{deployment_score:.4f}",
                        flush=True,
                    )

                    # Save best composite model separately
                    if composite > best_composite or (composite == best_composite and prauc > best_stitched_prauc):
                        best_composite = composite
                        best_stitched_prauc = prauc

                        torch.save(
                            _checkpoint_payload(
                                epoch,
                                train_loss=train_loss,
                                stitched_prauc=prauc,
                                composite=composite,
                            ),
                            Path(args.output_dir) / "best_composite_model.pt",
                        )
                        print(f"  ✓ Saved best composite model (composite={composite:.4f}, PR-AUC={prauc:.4f})", flush=True)

                    if deployment_score > best_deployment_score or (
                        deployment_score == best_deployment_score and prauc > best_prauc_only
                    ):
                        best_deployment_score = deployment_score
                        torch.save(
                            _checkpoint_payload(
                                epoch,
                                train_loss=train_loss,
                                stitched_prauc=prauc,
                                composite=composite,
                                deployment_score=deployment_score,
                                fp_area_pct_median=fp_area,
                                recall_2pct=recall_2pct,
                                recall_5pct=recall_5pct,
                            ),
                            Path(args.output_dir) / "best_score_model.pt",
                        )
                        print(
                            f"  ✓ Saved best score model (score={deployment_score:.4f}, "
                            f"PR-AUC={prauc:.4f}, FP area={fp_area*100:.2f}%)",
                            flush=True,
                        )

                    # PR-AUC drives early stopping (more stable than composite with conservative models)
                    if prauc > best_prauc_only:
                        best_prauc_only = prauc
                        patience_counter = 0
                        torch.save(
                            _checkpoint_payload(
                                epoch,
                                train_loss=train_loss,
                                stitched_prauc=prauc,
                                composite=composite,
                                recall_2pct=recall_2pct,
                                recall_5pct=recall_5pct,
                            ),
                            Path(args.output_dir) / "best_model.pt",
                        )
                        print(f"  ✓ Saved best model (PR-AUC={prauc:.4f}, composite={composite:.4f})", flush=True)
                    else:
                        patience_counter += 1
                        print(f"  No PR-AUC improvement (best={best_prauc_only:.4f}, patience={patience_counter}/{args.early_stop_patience})", flush=True)

                    # Early stopping check (with min_epochs guard!)
                    stitched_cycle = epoch // args.stitched_val_every_n_epochs
                    min_epochs_met = epoch >= args.early_stop_min_epochs
                    min_cycles_met = stitched_cycle >= args.early_stop_min_stitched_cycles
                    can_early_stop = min_epochs_met or min_cycles_met

                    if patience_counter >= args.early_stop_patience:
                        if can_early_stop:
                            print(f"\n  Early stopping triggered: PR-AUC has not improved for {args.early_stop_patience} stitched-val cycles (best={best_prauc_only:.4f})", flush=True)
                            break
                        else:
                            print(f"  (Would early-stop but min_epochs={args.early_stop_min_epochs}/min_cycles={args.early_stop_min_stitched_cycles} not met yet)", flush=True)

                print(f"{'='*80}\n", flush=True)

            # Training complete
            print(f"\n{'='*80}", flush=True)
            print(f"FULL MICROGRAPH TRAINING COMPLETE", flush=True)
            print(f"Best model (by PR-AUC): PR-AUC={best_prauc_only:.4f} (saved to best_model.pt)", flush=True)
            print(f"Best composite model: composite={best_composite:.4f} (saved to best_composite_model.pt)", flush=True)
            print(
                f"Best score model: score={best_deployment_score:.4f} "
                f"(saved to best_score_model.pt)",
                flush=True,
            )
            print(f"{'='*80}\n", flush=True)

            # Exit - don't fall through to patch-based training
            return

    _ensure_patch_cache_and_val_loader()

    for epoch in range(1, args.num_epochs + 1):
        # =======================================================================
        # CURRICULUM PHASES for PU-style loss (compute BEFORE dataset creation)
        # Phase A (epochs 1-phase_a): bad only (L_bad + hit-core) - SKIP particle masks for speed!
        # Phase B (epochs phase_a+1 to phase_b): + L_good, L_area
        # Phase C (epochs > phase_b): + L_margin, anneal rho
        #
        # NOTE: --disable_curriculum keeps Phase A forever to prevent calibration drift
        # Phase B/C enable hard-neg sampling and GT erosion which can destabilize training
        # =======================================================================
        if args.use_pu_loss:
            # If curriculum is disabled, force Phase A forever
            effective_phase_a_epochs = args.num_epochs + 1 if getattr(args, 'disable_curriculum', False) else args.phase_a_epochs
            effective_phase_b_epochs = args.num_epochs + 1 if getattr(args, 'disable_curriculum', False) else args.phase_b_epochs

            if epoch <= effective_phase_a_epochs:
                # Phase A: Learn "bad" only - SKIP expensive ops for speed
                current_phase = "A"
                lambda_good_eff = 0.0
                lambda_margin_eff = 0.0
                lambda_area_eff = 0.0
                rho_area_eff = args.rho_area
                skip_particle_mask = True  # CRITICAL: Skip in workers for Phase A performance
                hard_neg_fraction_eff = 0.0  # Disable hard-neg in Phase A for speed
            elif epoch <= effective_phase_b_epochs:
                # Phase B: Add good anchors + area budget
                current_phase = "B"
                lambda_good_eff = args.lambda_good
                lambda_margin_eff = 0.0
                lambda_area_eff = args.lambda_area
                rho_area_eff = args.rho_area
                skip_particle_mask = False  # Need particle masks for L_good
                hard_neg_fraction_eff = args.hard_neg_fraction  # Re-enable hard-neg
            else:
                # Phase C: Add margin loss, anneal rho
                current_phase = "C"
                lambda_good_eff = args.lambda_good
                lambda_margin_eff = args.lambda_margin
                lambda_area_eff = args.lambda_area
                # Anneal rho from rho_area to rho_area_final over 10 epochs
                anneal_epochs = 10
                epochs_into_c = epoch - args.phase_b_epochs
                if epochs_into_c <= anneal_epochs:
                    rho_area_eff = args.rho_area - (args.rho_area - args.rho_area_final) * (epochs_into_c / anneal_epochs)
                else:
                    rho_area_eff = args.rho_area_final
                skip_particle_mask = False  # Need particle masks for L_good and L_margin
                hard_neg_fraction_eff = args.hard_neg_fraction
        else:
            current_phase = "legacy"
            lambda_good_eff = 0.0
            lambda_margin_eff = 0.0
            lambda_area_eff = 0.0
            rho_area_eff = args.rho_area
            skip_particle_mask = True  # No PU loss, skip particle masks
            hard_neg_fraction_eff = args.hard_neg_fraction

        # =======================================================================
        # DYNAMIC LOSS REWEIGHTING: Prevents "predict everything as contamination"
        # Phase 1 (epochs 1 to reweight_epoch): Discovery - all losses at full strength
        # Phase 2 (epochs > reweight_epoch): Refinement - reduce hit_core, increase fpcap
        #
        # The problem: 4/5 loss terms (BCE, hit_core, outside_mass, edge) benefit from
        # predicting HIGH probabilities. Only fpcap votes for LOW. By epoch 15+, the
        # model learns "predict everything as contamination" minimizes total loss.
        #
        # The fix: After finding contamination (epoch ~10), flip the vote balance:
        # - Reduce hit_core (you already found it, stop pushing for more)
        # - Reduce BCE positive pressure (already classifying positives correctly)
        # - Increase fpcap (now THIS is what matters - control false positives)
        # =======================================================================
        if args.loss_reweight_epoch > 0 and epoch > args.loss_reweight_epoch:
            # Phase 2: Refinement - reduce confidence-pushing losses, increase FP penalty
            lambda_bce_eff = args.phase2_lambda_bce
            lambda_hit_core_eff = args.phase2_lambda_hit_core
            lambda_fpcap_eff = args.phase2_lambda_fpcap
            lambda_outside_mass_eff = args.phase2_lambda_outside_mass
            loss_phase = "REFINEMENT"
            if epoch == args.loss_reweight_epoch + 1:
                print(f"\n  [LOSS REWEIGHT] Epoch {epoch}: Switching to REFINEMENT phase", flush=True)
                print(f"    λ_bce: {args.lambda_bce:.1f} → {lambda_bce_eff:.1f}", flush=True)
                print(f"    λ_hit_core: {args.lambda_hit_core:.1f} → {lambda_hit_core_eff:.1f}", flush=True)
                print(f"    λ_fpcap: {args.lambda_fpcap:.1f} → {lambda_fpcap_eff:.1f}", flush=True)
                print(f"    λ_outside_mass: {args.lambda_outside_mass:.1f} → {lambda_outside_mass_eff:.1f}", flush=True)
        else:
            # Phase 1: Discovery - full strength on all losses
            lambda_bce_eff = args.lambda_bce
            lambda_hit_core_eff = args.lambda_hit_core
            lambda_fpcap_eff = args.lambda_fpcap
            lambda_outside_mass_eff = args.lambda_outside_mass
            loss_phase = "DISCOVERY"

        # Compute effective GT-bias for this epoch (with optional decay)
        if getattr(args, 'gt_bias_decay', False):
            if epoch <= args.gt_bias_decay_start:
                # Before decay: use full gt_bias
                gt_bias_eff = args.gt_bias
            elif epoch >= args.gt_bias_decay_end:
                # After decay: use final gt_bias
                gt_bias_eff = args.gt_bias_final
            else:
                # During decay: linear interpolation
                decay_progress = (epoch - args.gt_bias_decay_start) / (args.gt_bias_decay_end - args.gt_bias_decay_start)
                gt_bias_eff = args.gt_bias + decay_progress * (args.gt_bias_final - args.gt_bias)

            # Log gt_bias changes
            if epoch == 1 or epoch == args.gt_bias_decay_start + 1 or epoch == args.gt_bias_decay_end:
                print(f"  [GT-BIAS] Epoch {epoch}: gt_bias = {gt_bias_eff:.3f}", flush=True)
        else:
            gt_bias_eff = args.gt_bias

        # Deterministic per-epoch patch sampling (seed+epoch) so epochs are comparable
        train_ds = FastPatchDataset(
            train_df, mic_dir, rule,
            normalization_method=args.normalization_method,
            psd_use_radial_normalization=args.psd_use_radial_normalization,
            patch_size=effective_patch_size,
            patches_per_mic=args.patches_per_mic,
            gt_bias=gt_bias_eff,
            downsample_factor=downsample_factor,
            seed=args.seed + epoch,
            use_augmentation=True,
            use_rotation_aug=args.use_rotation_aug,
            cache_dir=cache_train,
            hard_neg_fraction=hard_neg_fraction_eff,
            hard_neg_quantile=args.hard_neg_quantile,
            # Actual particle picks (positive labels)
            particles_dict=particles_dict,
            particle_box_size_px=args.particle_box_size_px,
            particle_use_gaussian=args.particle_use_gaussian,
            skip_particle_mask=skip_particle_mask,
            # NEW: Patch-type balanced sampling (disabled in Phase A when skip_particle_mask=True)
            use_patch_type_sampling=args.use_patch_type_sampling and not skip_particle_mask,
            bad_centered_fraction=args.bad_centered_fraction,
            particle_centered_fraction=args.particle_centered_fraction,
            # Radial PSD
            psd_use_radial_image=args.psd_use_radial_image,
            psd_radial_bins=args.psd_radial_bins,
            # Edge features
            use_edge_features=args.use_edge_features,
            edge_blend_alpha=args.edge_blend_alpha,
            edge_sigma=args.edge_sigma,
            # Multi-scale PSD
            psd_multiscale=args.psd_multiscale,
            psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
            psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
            psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
            psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
            psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
            psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
            psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
            psd_texture_mode=args.psd_texture_mode,
            psd_highpass_cutoff=args.psd_highpass_cutoff,
            # Multi-res context PSD
            psd_multires_context=args.psd_multires_context,
            psd_context_sizes=tuple(int(x) for x in args.psd_context_sizes.split(",")),
            psd_bins_per_scale=args.psd_bins_per_scale,
            # Global features
            use_global_features=args.use_global_features,
            global_feature_dim=args.global_feature_dim,
            # Edge direction features
            use_edge_direction=args.use_edge_direction,
            edge_direction_dim=args.edge_direction_dim,
            use_pixel_size_channel=args.use_pixel_size_channel,
            pixel_size_by_dataset=pixel_size_by_dataset,
            pixel_size_channel_min_angstrom=args.pixel_size_channel_min_angstrom,
            pixel_size_channel_max_angstrom=args.pixel_size_channel_max_angstrom,
        )
        train_ds.use_psd_input = getattr(args, 'use_psd_bool', True)
        train_ds.include_real_space_input = getattr(args, "include_real_space_input", True)

        # Note: We recreate DataLoader every epoch (seed+epoch changes dataset), so persistent_workers
        # doesn't provide much benefit and can cause FD leaks if workers aren't properly shut down.
        # Keep persistent_workers=False to avoid "too many open files" errors.
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=nw, pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if nw else None,
        )

        model.train()
        losses = []
        epoch_start = time.perf_counter()

        # Reset stitched training loss tracking for this epoch
        epoch_stitched_losses = []
        epoch_stitched_dices = []
        epoch_stitched_batch_counter = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.num_epochs}", unit="batch", ncols=120)
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch - now includes patch_type (4th), global_features (5th), edge_features (6th)
            if len(batch_data) == 6:
                patch_x, patch_gt, patch_particle, patch_type, global_features, edge_features = batch_data
            elif len(batch_data) == 5:
                patch_x, patch_gt, patch_particle, patch_type, global_features = batch_data
                edge_features = torch.zeros(patch_x.shape[0], 3, dtype=torch.float32)
            elif len(batch_data) == 4:
                patch_x, patch_gt, patch_particle, patch_type = batch_data
                global_features = torch.zeros(patch_x.shape[0], args.global_feature_dim, dtype=torch.float32)
                edge_features = torch.zeros(patch_x.shape[0], 3, dtype=torch.float32)
            else:
                # Legacy: 3-element batch (no patch_type)
                patch_x, patch_gt, patch_particle = batch_data
                patch_type = torch.zeros(patch_x.shape[0], dtype=torch.long)
                global_features = torch.zeros(patch_x.shape[0], args.global_feature_dim, dtype=torch.float32)
                edge_features = torch.zeros(patch_x.shape[0], 3, dtype=torch.float32)

            # Warmup + cosine LR (per-step)
            lr = _lr_cosine_warmup(global_step)
            for g in opt.param_groups:
                g["lr"] = lr

            patch_x = patch_x.to(args.device, non_blocking=True)
            patch_gt = patch_gt.to(args.device, non_blocking=True)
            patch_particle = patch_particle.to(args.device, non_blocking=True) if patch_particle is not None else None
            patch_type = patch_type.to(args.device, non_blocking=True)
            global_features = global_features.to(args.device, non_blocking=True) if args.use_global_features else None
            edge_features = edge_features.to(args.device, non_blocking=True) if args.use_edge_direction else None

            opt.zero_grad(set_to_none=True)

            # PHASE A FAST PATH: Skip ALL expensive CPU ops (scipy erosion, edge detection, etc.)
            # This makes Phase A as fast as the original code before PU-loss
            skip_expensive_cpu_ops = (current_phase == "A")

            # Precompute GT cores (on CPU with scipy, then move to GPU) - SKIP in Phase A
            if skip_expensive_cpu_ops:
                # Use GT mask directly as "core" in Phase A (no erosion)
                gt_core = patch_gt
            else:
                gt_core = precompute_gt_cores(patch_gt, erosion_px=effective_core_erosion)

            # Initialize auxiliary masks as None (will skip expensive computation in Phase A)
            particle_like_mask = None
            particle_like_mask_wide = None
            clean_mask = None
            edge_map = None

            # Only compute expensive CPU ops in Phase B/C
            if not skip_expensive_cpu_ops:
                # Extract image channel and GT mask for auxiliary computations
                patch_img_np = patch_x[:, 0].detach().cpu().numpy()  # (B, H, W) - image channel
                patch_gt_np = patch_gt.detach().cpu().numpy()  # (B, H, W)

                # Particle-likeness negative mining: detect particle-like pixels in clean regions
                if args.particle_neg_enabled and (args.lambda_particle_neg > 0 or args.lambda_particle_neg_wide > 0):
                    particle_masks = []
                    particle_masks_wide = []
                    for b in range(patch_img_np.shape[0]):
                        if args.lambda_particle_neg > 0:
                            particle_mask = detect_particle_like_pixels(
                                patch_img_np[b], patch_gt_np[b],
                                margin_px=effective_particle_neg_margin
                            )
                            particle_masks.append(torch.from_numpy(particle_mask).float())
                        if args.lambda_particle_neg_wide > 0 and args.particle_neg_margin_wide_px > 0:
                            particle_mask_wide = detect_particle_like_pixels(
                                patch_img_np[b], patch_gt_np[b],
                                margin_px=effective_particle_neg_margin_wide
                            )
                            particle_masks_wide.append(torch.from_numpy(particle_mask_wide).float())
                    if particle_masks:
                        particle_like_mask = torch.stack(particle_masks).to(args.device)
                    if particle_masks_wide:
                        particle_like_mask_wide = torch.stack(particle_masks_wide).to(args.device)

                # Compute clean mask for FP control and edge suppression
                if args.lambda_fpcap > 0 or args.lambda_edge > 0:
                    clean_masks = []
                    for b in range(patch_gt_np.shape[0]):
                        clean = compute_clean_mask(patch_gt_np[b], dilate_radius=3)
                        clean_masks.append(torch.from_numpy(clean).float())
                    clean_mask = torch.stack(clean_masks).to(args.device)

                # Compute edge map for edge-conditioned suppression
                if args.lambda_edge > 0:
                    edge_map = _gpu_sobel_edge_map(patch_x[:, 0:1, :, :])  # GPU-batched Sobel

            # Forward + Backward with AMP (bf16)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if args.use_edge_direction and edge_features is not None:
                    logits = model(patch_x, edge_features=edge_features)
                elif args.use_global_features and global_features is not None:
                    logits = model(patch_x, global_features=global_features)
                else:
                    logits = model(patch_x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim == 3:
                    logits = logits.unsqueeze(1)

                # Use PU-style loss with curriculum OR legacy loss
                if args.use_pu_loss and particles_dict:
                    # Entropy and asymmetric penalties: enable after Phase A
                    lambda_entropy_eff = args.lambda_entropy if not skip_particle_mask else 0.0
                    lambda_particle_asym_eff = args.lambda_particle_asym if not skip_particle_mask else 0.0

                    loss, stats = compute_pu_loss(
                        logits, patch_gt.unsqueeze(1), gt_core,
                        good_anchor_mask=patch_particle,
                        lambda_bad=args.lambda_bce,  # Keep same scale as before
                        lambda_hit_core=args.lambda_hit_core,
                        lambda_good=lambda_good_eff,
                        lambda_margin=lambda_margin_eff,
                        lambda_area=lambda_area_eff,
                        lambda_entropy=lambda_entropy_eff,
                        lambda_particle_asym=lambda_particle_asym_eff,
                        tau_good=args.tau_good,
                        margin_m=args.margin_m,
                        rho_area=rho_area_eff,
                        bce_neg_weight=args.bce_neg_weight,
                        entropy_target_range=(args.entropy_target_low, args.entropy_target_high),
                    )
                else:
                    # ==========================================================
                    # LOSS COMPUTATION: Dice+Focal (recommended) or legacy BCE
                    # ==========================================================
                    if getattr(args, 'use_dice_loss', False):
                        # DICE + FOCAL: Medical imaging's self-balancing approach
                        # Cannot collapse to "predict everything" - the ONLY way to
                        # maximize Dice is to precisely match GT shape.
                        #
                        # EDGE EROSION: Exclude patch boundary pixels from loss to
                        # prevent 'edge of context' artifacts (swaths at patch boundaries)
                        effective_edge_erosion = max(0, int(getattr(args, 'edge_erosion_px', 0) / downsample_factor))
                        loss, stats = compute_dice_focal_loss(
                            logits, patch_gt.unsqueeze(1),
                            lambda_dice=args.lambda_dice,
                            lambda_focal=args.lambda_focal,
                            lambda_edge=args.lambda_edge,
                            focal_gamma=args.focal_gamma,
                            focal_alpha=args.focal_alpha,
                            edge_map=edge_map,
                            edge_erosion_px=effective_edge_erosion,
                            use_tukey_weight=getattr(args, 'use_tukey_weight', True),
                        )
                    else:
                        # Legacy BCE-based multi-component loss (DEPRECATED)
                        # Known issue: 4/5 components benefit from predicting HIGH,
                        # leading to "predict everything as contamination" collapse
                        fpcap_weight = 0.0
                        # Ramp fpcap in early epochs (within current phase)
                        if lambda_fpcap_eff > 0:
                            ramp_epochs = 10
                            if args.loss_reweight_epoch > 0 and epoch > args.loss_reweight_epoch:
                                # In refinement phase, use full fpcap immediately
                                fpcap_weight = lambda_fpcap_eff
                            elif epoch <= ramp_epochs:
                                fpcap_weight = lambda_fpcap_eff * (epoch / ramp_epochs)
                            else:
                                fpcap_weight = lambda_fpcap_eff
                        else:
                            fpcap_weight = 0.0

                        loss, stats = compute_fast_region_loss(
                            logits, patch_gt.unsqueeze(1), gt_core.unsqueeze(1),
                            lambda_bce=lambda_bce_eff,
                            bce_neg_weight=args.bce_neg_weight,
                            lambda_hit_core=lambda_hit_core_eff,
                            lambda_outside_mass=lambda_outside_mass_eff,
                            lambda_particle_neg=args.lambda_particle_neg if args.particle_neg_enabled else 0.0,
                            particle_like_mask=particle_like_mask,
                            lambda_particle_neg_wide=args.lambda_particle_neg_wide if args.particle_neg_enabled else 0.0,
                            particle_like_mask_wide=particle_like_mask_wide,
                            lambda_fpcap=fpcap_weight,
                            fpcap_target_mass=args.fpcap_target_mass,
                            clean_mask=clean_mask,
                            lambda_edge=args.lambda_edge,
                            edge_map=edge_map,
                            lambda_particle_pos=0.0,
                            actual_particle_mask=None,
                            use_soft_hit_core=getattr(args, 'soft_hit_core', False),
                        )

                # DEPRECATED: Patch-type specific loss (did not work well)
                if args.use_patch_type_sampling and not skip_particle_mask:
                    patch_type_loss, pt_stats = compute_patch_type_loss(
                        logits, patch_gt.unsqueeze(1), patch_type,
                        encoder_features=None,
                        lambda_patch_classify=args.lambda_patch_classify,
                        lambda_particle_region=args.lambda_particle_region,
                        lambda_bad_region=args.lambda_bad_region,
                        particle_region_radius_frac=args.particle_region_radius_frac,
                    )
                    loss = loss + patch_type_loss
                    for k, v in pt_stats.items():
                        stats[k] = v

                # SIMPLE PARTICLE BCE (RECOMMENDED) - direct supervision at particle locations
                # This tells the model: "these particle regions are NOT contamination"
                if args.use_simple_particle_bce and not skip_particle_mask and patch_particle is not None:
                    # patch_particle is the particle mask (1 at particle locations, 0 elsewhere)
                    particle_mask_binary = (patch_particle > 0.1).float()
                    if particle_mask_binary.sum() > 0:
                        # BCE with target=0 at particle locations
                        logits_at_particles = logits.squeeze(1) * particle_mask_binary  # (B, H, W)
                        target_zeros = torch.zeros_like(logits_at_particles)

                        # Weighted BCE - only at particle locations
                        particle_bce = F.binary_cross_entropy_with_logits(
                            logits_at_particles, target_zeros,
                            weight=particle_mask_binary,
                            reduction='sum'
                        ) / (particle_mask_binary.sum() + 1e-6)

                        loss = loss + args.lambda_particle_bce * particle_bce
                        stats["loss_particle_bce"] = float(particle_bce.item())

            # =========================================================================
            # STITCHED TRAINING LOSS - forces model to see TRUE distribution
            # Compute every N batches to reduce overhead while still providing feedback
            # =========================================================================
            epoch_stitched_batch_counter += 1
            if getattr(args, 'use_stitched_train_loss', False) and stitched_train_mics and \
               epoch_stitched_batch_counter % args.stitched_train_every_n_batches == 0:
                # Sample random training micrographs
                import random
                n_mics = min(args.stitched_train_mics_per_batch, len(stitched_train_mics))
                sampled_mics = random.sample(stitched_train_mics, n_mics)

                stitched_loss_total = torch.tensor(0.0, device=args.device)
                stitched_stats_accum = {'stitched_dice': [], 'stitched_fp_area': [],
                                        'stitched_pos_median': [], 'stitched_neg_median': []}

                for mic_path, gt_path, dataset_id, ds_factor in sampled_mics:
                    try:
                        stitched_loss, stitched_stats = compute_stitched_train_loss(
                            model=model,
                            mic_path=mic_path,
                            gt_path=gt_path,
                            dataset_id=dataset_id,
                            rule=rule,
                            device=args.device,
                            patch_size=effective_patch_size,
                            normalization_method=args.normalization_method,
                            psd_multiscale=args.psd_multiscale,
                            psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
                            psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
                            psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
                            psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
                            psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
                            psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
                            psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
                            psd_texture_mode=args.psd_texture_mode,
                            psd_highpass_cutoff=args.psd_highpass_cutoff,
                            use_psd=getattr(args, "use_psd_bool", True),
                            include_real_space_input=getattr(args, "include_real_space_input", True),
                            use_pixel_size_channel=args.use_pixel_size_channel,
                            pixel_size_by_dataset=pixel_size_by_dataset,
                            pixel_size_channel_min_angstrom=args.pixel_size_channel_min_angstrom,
                            pixel_size_channel_max_angstrom=args.pixel_size_channel_max_angstrom,
                            downsample_factor=ds_factor,
                            grid_patches=args.stitched_train_grid_patches,
                        )
                        stitched_loss_total = stitched_loss_total + stitched_loss
                        for k, v in stitched_stats.items():
                            stitched_stats_accum[k].append(v)
                    except Exception as e:
                        # Skip this micrograph on error
                        if batch_idx == 0:
                            print(f"    [Stitched loss] Skip {mic_path.name}: {e}", flush=True)

                if len(stitched_stats_accum['stitched_dice']) > 0:
                    # Average stitched loss across sampled micrographs
                    stitched_loss_avg = stitched_loss_total / len(stitched_stats_accum['stitched_dice'])
                    loss = loss + args.lambda_stitched * stitched_loss_avg

                    # Track for epoch logging
                    epoch_stitched_losses.append(float(stitched_loss_avg.detach()))
                    epoch_stitched_dices.append(np.mean(stitched_stats_accum['stitched_dice']))

                    stats["loss_stitched"] = float(stitched_loss_avg.detach())
                    stats["stitched_dice"] = np.mean(stitched_stats_accum['stitched_dice'])
                    stats["stitched_fp_area"] = np.mean(stitched_stats_accum['stitched_fp_area'])
                    stats["stitched_pos_median"] = np.mean(stitched_stats_accum['stitched_pos_median'])
                    stats["stitched_neg_median"] = np.mean(stitched_stats_accum['stitched_neg_median'])

            # Backward with AMP
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            losses.append(loss_val)
            global_step += 1

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{np.mean(losses):.4f}", "lr": f"{lr:.2e}"})

        train_loss = np.mean(losses)
        epoch_time = time.perf_counter() - epoch_start

        # Validation (fixed val patches every epoch for stable metrics; secondary to stitched val)
        model.eval()
        val_losses = []
        val_hit_core = []
        val_outside_mass = []
        val_bce_list = []
        val_fpcap = []
        val_edge = []
        val_clean_patch = []
        val_n_clean_patches = []
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validating", unit="batch", leave=False, ncols=120):
                # Handle both 3-element and 4-element batches
                if len(val_batch) >= 4:
                    patch_x, patch_gt, _, _ = val_batch[:4]
                else:
                    patch_x, patch_gt, _ = val_batch[:3]

                # Ignore particle mask in validation (not used for loss)
                patch_x = patch_x.to(args.device, non_blocking=True)
                patch_gt = patch_gt.to(args.device, non_blocking=True)

                gt_core = precompute_gt_cores(patch_gt, erosion_px=effective_core_erosion)

                # Extract image channel and GT mask for auxiliary computations
                patch_img_np = patch_x[:, 0].detach().cpu().numpy()  # (B, H, W)
                patch_gt_np = patch_gt.detach().cpu().numpy()  # (B, H, W)

                # Compute clean mask and edge map for validation
                clean_mask = None
                if args.lambda_fpcap > 0 or args.lambda_edge > 0:
                    clean_masks = []
                    for b in range(patch_gt_np.shape[0]):
                        clean = compute_clean_mask(patch_gt_np[b], dilate_radius=3)
                        clean_masks.append(torch.from_numpy(clean).float())
                    clean_mask = torch.stack(clean_masks).to(args.device)  # (B, H, W)

                edge_map = None
                if args.lambda_edge > 0:
                    edge_map = _gpu_sobel_edge_map(patch_x[:, 0:1, :, :])  # GPU-batched Sobel

                # Particle-neg masks for validation (scaled margins)
                particle_like_mask_val = None
                particle_like_mask_wide_val = None
                if args.particle_neg_enabled and (args.lambda_particle_neg > 0 or args.lambda_particle_neg_wide > 0):
                    particle_masks_val = []
                    particle_masks_wide_val = []
                    for b in range(patch_img_np.shape[0]):
                        if args.lambda_particle_neg > 0:
                            particle_mask = detect_particle_like_pixels(
                                patch_img_np[b], patch_gt_np[b],
                                margin_px=effective_particle_neg_margin
                            )
                            particle_masks_val.append(torch.from_numpy(particle_mask).float())
                        if args.lambda_particle_neg_wide > 0 and args.particle_neg_margin_wide_px > 0:
                            particle_mask_wide = detect_particle_like_pixels(
                                patch_img_np[b], patch_gt_np[b],
                                margin_px=effective_particle_neg_margin_wide
                            )
                            particle_masks_wide_val.append(torch.from_numpy(particle_mask_wide).float())
                    if particle_masks_val:
                        particle_like_mask_val = torch.stack(particle_masks_val).to(args.device)
                    if particle_masks_wide_val:
                        particle_like_mask_wide_val = torch.stack(particle_masks_wide_val).to(args.device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Note: validation doesn't have global features in this loop
                    # Use model(patch_x) directly
                    logits = model(patch_x)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    if logits.ndim == 3:
                        logits = logits.unsqueeze(1)

                    # Use same loss function as training for fair comparison
                    if getattr(args, 'use_dice_loss', False):
                        effective_edge_erosion = max(0, int(getattr(args, 'edge_erosion_px', 0) / downsample_factor))
                        loss, stats = compute_dice_focal_loss(
                            logits, patch_gt.unsqueeze(1),
                            lambda_dice=args.lambda_dice,
                            lambda_focal=args.lambda_focal,
                            lambda_edge=args.lambda_edge,
                            focal_gamma=args.focal_gamma,
                            focal_alpha=args.focal_alpha,
                            edge_map=edge_map,
                            edge_erosion_px=effective_edge_erosion,
                            use_tukey_weight=getattr(args, 'use_tukey_weight', True),
                        )
                    else:
                        effective_outside_dilate = max(0, int(args.outside_dilate_px / downsample_factor))
                        val_fpcap_weight = lambda_fpcap_eff if epoch > 10 else lambda_fpcap_eff * (epoch / 10.0)
                        loss, stats = compute_fast_region_loss(
                            logits, patch_gt.unsqueeze(1), gt_core.unsqueeze(1),
                            lambda_bce=lambda_bce_eff,
                            bce_neg_weight=args.bce_neg_weight,
                            lambda_hit_core=lambda_hit_core_eff,
                            lambda_outside_mass=lambda_outside_mass_eff,
                            outside_dilate_px=effective_outside_dilate,
                            lambda_particle_neg=args.lambda_particle_neg if args.particle_neg_enabled else 0.0,
                            particle_like_mask=particle_like_mask_val,
                            lambda_particle_neg_wide=args.lambda_particle_neg_wide if args.particle_neg_enabled else 0.0,
                            particle_like_mask_wide=particle_like_mask_wide_val,
                            lambda_fpcap=val_fpcap_weight,
                            fpcap_target_mass=args.fpcap_target_mass,
                            clean_mask=clean_mask,
                            lambda_edge=args.lambda_edge,
                            edge_map=edge_map,
                            use_soft_hit_core=getattr(args, 'soft_hit_core', False),
                        )

                val_losses.append(float(loss.cpu().item()))
                # Handle different stat keys for Dice vs BCE loss
                if getattr(args, 'use_dice_loss', False):
                    val_hit_core.append(stats.get("loss_dice", 0.0))  # Use dice as primary metric
                    val_outside_mass.append(stats.get("loss_focal", 0.0))
                    val_bce_list.append(stats.get("dice_coefficient", 0.0))  # Track dice coeff in BCE slot
                    val_fpcap.append(0.0)
                else:
                    val_hit_core.append(stats["loss_hit_core"])
                    val_outside_mass.append(stats["loss_outside_mass"])
                    val_bce_list.append(stats["loss_bce"])
                    val_fpcap.append(stats.get("loss_fpcap", 0.0))
                val_edge.append(stats.get("loss_edge", 0.0))
                val_clean_patch.append(stats.get("loss_clean_patch", 0.0))
                val_n_clean_patches.append(stats.get("n_clean_patches", 0))

        val_loss = np.mean(val_losses)
        val_bce_mean = np.mean(val_bce_list)
        val_hit_core_mean = np.mean(val_hit_core)
        val_outside_mass_mean = np.mean(val_outside_mass)
        val_fpcap_mean = np.mean(val_fpcap) if val_fpcap else 0.0
        val_edge_mean = np.mean(val_edge) if val_edge else 0.0

        # Print epoch summary (patch val = secondary)
        print(f"\n{'='*80}", flush=True)
        phase_str = f" [Phase {current_phase}]" if args.use_pu_loss else ""
        # Add loss reweight phase indicator
        if args.loss_reweight_epoch > 0:
            loss_phase_str = f" [{loss_phase}]"
        else:
            loss_phase_str = ""
        print(f"Epoch {epoch:02d}/{args.num_epochs} COMPLETE{phase_str}{loss_phase_str}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Train Loss: {train_loss:.6f} | Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)", flush=True)
        # STITCHED TRAINING LOSS summary - key indicator of true distribution alignment
        if getattr(args, 'use_stitched_train_loss', False) and epoch_stitched_dices:
            avg_stitched_dice = np.mean(epoch_stitched_dices)
            avg_stitched_loss = np.mean(epoch_stitched_losses)
            print(f"  [STITCHED TRAIN] Dice: {avg_stitched_dice:.4f} | Loss: {avg_stitched_loss:.4f} (λ={args.lambda_stitched})", flush=True)
        # Show current effective lambda values if reweighting is enabled
        if args.loss_reweight_epoch > 0:
            print(f"  λ_eff: bce={lambda_bce_eff:.2f}, hit={lambda_hit_core_eff:.2f}, out={lambda_outside_mass_eff:.2f}, fpcap={lambda_fpcap_eff:.2f}", flush=True)
        if args.use_pu_loss and particles_dict:
            print(f"  PU Loss: λ_good={lambda_good_eff:.2f}, λ_margin={lambda_margin_eff:.2f}, λ_area={lambda_area_eff:.2f}, rho={rho_area_eff:.3f}", flush=True)
        # Compute clean_patch stats
        val_clean_patch_mean = np.mean(val_clean_patch) if val_clean_patch else 0.0
        val_n_clean_patches_mean = np.mean(val_n_clean_patches) if val_n_clean_patches else 0

        print(f"Patch Val (secondary): {val_loss:.6f}", flush=True)
        if getattr(args, 'use_dice_loss', False):
            # Dice+Focal loss metrics
            print(f"  ├─ Dice Loss:         {val_hit_core_mean:.6f}", flush=True)  # val_hit_core stores dice loss
            print(f"  ├─ Focal Loss:        {val_outside_mass_mean:.6f}", flush=True)  # val_outside_mass stores focal
            print(f"  ├─ Dice Coeff:        {val_bce_mean:.4f}", flush=True)  # val_bce stores dice coefficient
            if args.lambda_edge > 0:
                print(f"  ├─ Edge:              {val_edge_mean:.6f}", flush=True)
        else:
            # Legacy BCE-based loss metrics
            print(f"  ├─ BCE:               {val_bce_mean:.6f}", flush=True)
            print(f"  ├─ Hit Core:          {val_hit_core_mean:.6f}", flush=True)
            print(f"  ├─ Outside Mass:      {val_outside_mass_mean:.6f}", flush=True)
            if args.lambda_fpcap > 0:
                print(f"  ├─ FP-cap:            {val_fpcap_mean:.6f}", flush=True)
            if args.lambda_edge > 0:
                print(f"  ├─ Edge:              {val_edge_mean:.6f}", flush=True)
        # Always log clean_patch_loss - critical for diagnosing calibration drift
        print(f"  └─ Clean Patch:       {val_clean_patch_mean:.6f} (n={val_n_clean_patches_mean:.1f} clean patches/batch)", flush=True)

        # Stitched full-micrograph validation (primary signal for checkpoint selection)
        run_stitched = (epoch % args.stitched_val_every_n_epochs == 0)
        if run_stitched:
            print(f"\n  [STITCHED VAL] Running on {len(fixed_stitched_set)} fixed micrographs...", flush=True)
            export_dir = (out_dir / "stitched_prob_maps" / f"epoch_{epoch:03d}") if args.export_stitched_prob_maps else None
            # Parse excluded datasets from comma-separated string
            exclude_datasets_alt = [s.strip() for s in args.exclude_datasets_from_metrics.split(",") if s.strip()]

            stitched = run_stitched_val_fast(
                model, fixed_stitched_set, rule, args.device,
                patch_size=args.stitched_val_patch_size,
                overlap=args.stitched_val_overlap,
                normalization_method=args.normalization_method,
                psd_use_radial_normalization=args.psd_use_radial_normalization,
                downsample_factor=args.stitched_val_downsample,
                min_area_px=args.stitched_val_min_area_px,
                fp_area_cap_pct=args.stitched_val_fp_area_cap_pct,
                opening_size=args.stitched_val_opening_size,
                closing_size=args.stitched_val_closing_size,
                export_dir=export_dir,
                pixel_size_angstrom=args.stitched_val_pixel_size_angstrom,
                pixel_size_per_dataset=pixel_size_by_dataset,
                # Per-dataset downsampling (matches training adaptive downsampling)
                downsample_per_dataset=downsample_factors,
                # Multi-scale PSD parameters (must match training!)
                psd_multiscale=args.psd_multiscale,
                psd_multiscale_separate_channels=getattr(args, "psd_multiscale_separate_channels", False),
                psd_scales=tuple(int(x) for x in args.psd_scales.split(",")),
                psd_frequency_band_channels=getattr(args, "psd_frequency_band_channels", False),
                psd_frequency_bands=getattr(args, "psd_frequency_bands_tuple", tuple()),
                psd_frequency_band_include_full_spectrum=getattr(args, "psd_frequency_band_include_full_spectrum", False),
                psd_frequency_band_include_anisotropy=getattr(args, "psd_frequency_band_include_anisotropy", False),
                psd_frequency_band_anisotropy_min_freq=getattr(args, "psd_frequency_band_anisotropy_min_freq", 0.0),
                psd_texture_mode=getattr(args, 'psd_texture_mode', False),
                psd_highpass_cutoff=getattr(args, 'psd_highpass_cutoff', 0.2),
                # Exclude certain datasets from metrics (but still generate visualizations)
                exclude_datasets_from_metrics=exclude_datasets_alt if exclude_datasets_alt else None,
                # Ablation: whether model uses PSD channels
                use_psd=getattr(args, 'use_psd_bool', True),
                include_real_space_input=getattr(args, "include_real_space_input", True),
                use_pixel_size_channel=args.use_pixel_size_channel,
                pixel_size_channel_min_angstrom=args.pixel_size_channel_min_angstrom,
                pixel_size_channel_max_angstrom=args.pixel_size_channel_max_angstrom,
                batch_forward_size=getattr(args, 'stitched_val_batch_forward_size', 1),
                return_native_resolution=True,
                blending_window=args.stitched_val_blending_window,
                blending_edge_px=args.stitched_val_blending_edge_px,
                adaptive_downsample_method=(getattr(args, "adaptive_downsample_method", None) or "zoom"),
                threshold_grid=getattr(args, "stitched_val_threshold_grid", None),
                metrics_max_samples=getattr(args, "stitched_val_metrics_max_samples", 5_000_000),
            )
            print(f"  [PRIMARY METRICS] Stitched PR-AUC: {stitched['pr_auc']:.4f} | AUROC: {stitched['auroc']:.4f}", flush=True)
            print(f"  [PRIMARY METRICS] FP area median: {stitched['fp_area_pct_median']*100:.2f}%% | Pred components median: {stitched['pred_components_median']:.1f} | Junk %% error: {stitched['junk_pct_error']*100:.2f}%%", flush=True)
            print(f"  Recall@FP≤{args.stitched_val_fp_area_cap_pct}%%: {stitched['recall_at_fp_cap']:.4f}", flush=True)
            # Print multi-FP-cap recalls if available
            multi_fp_keys = [k for k in stitched.keys() if k.startswith("recall_at_fp_")]
            if multi_fp_keys:
                multi_fp_str = " | ".join([f"{k.replace('recall_at_fp_', 'R@')}={stitched[k]:.3f}" for k in sorted(multi_fp_keys)])
                print(f"  Multi-FP-cap recalls: {multi_fp_str}", flush=True)
            print(f"  Prob hist (pos/neg median): {stitched.get('prob_pos_median', 0):.3f} / {stitched.get('prob_neg_median', 0):.3f}", flush=True)
            if export_dir:
                print(f"  Exported stitched prob maps → {export_dir}", flush=True)
            if stitched["n_mics"] > 0:
                stitched_val_cycle_count += 1
                improved = False

                # Check if we're past minimum training floor
                min_epochs_met = epoch >= args.early_stop_min_epochs
                min_cycles_met = stitched_val_cycle_count >= args.early_stop_min_stitched_cycles
                can_early_stop = min_epochs_met or min_cycles_met

                if args.early_stop_mode == "score":
                    # Early stopping: use smooth surrogate metric (PR-AUC - α * fp_mass)
                    # fp_mass is computed as median FP area % (smooth, differentiable surrogate)
                    fp_mass = stitched.get("fp_area_pct_median", 1.0)  # Already normalized to [0, 1]
                    pr_auc = stitched.get("pr_auc", 0.0)
                    score = pr_auc - args.early_stop_alpha * fp_mass  # Higher is better

                    if score > best_stitched_score:
                        best_stitched_score = score
                        best_stitched_pr_auc = pr_auc
                        best_stitched_fp_mass = fp_mass
                        improved = True
                        torch.save(
                            _checkpoint_payload(
                                epoch,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                stitched_pr_auc=pr_auc,
                                stitched_auroc=stitched["auroc"],
                                stitched_score=score,
                                fp_area_pct_median=fp_mass,
                                recall_at_fp_cap=stitched.get("recall_at_fp_cap", 0.0),
                                recommended_threshold=stitched.get("recommended_threshold", 0.5),
                            ),
                            out_dir / "best_model.pth",
                        )
                        rec_thr = stitched.get("recommended_threshold", 0.5)
                        print(f"  ✓ Saved best model (score={score:.4f}, PR-AUC={pr_auc:.4f}, FP-mass={fp_mass*100:.2f}%, rec_thr={rec_thr:.2f})", flush=True)

                elif args.early_stop_mode == "composite":
                    # Composite metric: 0.5*Recall@2% + 0.5*Recall@5% (less brittle than 0.5% alone)
                    recall_2pct = stitched.get("recall_at_fp_2.0pct", 0.0)
                    recall_5pct = stitched.get("recall_at_fp_5.0pct", 0.0)
                    composite_score = 0.5 * recall_2pct + 0.5 * recall_5pct
                    fp_mass = stitched.get("fp_area_pct_median", 1.0)

                    if composite_score > best_stitched_score:
                        best_stitched_score = composite_score
                        best_stitched_pr_auc = stitched.get("pr_auc", 0.0)
                        best_stitched_fp_mass = fp_mass
                        improved = True
                        torch.save(
                            _checkpoint_payload(
                                epoch,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                stitched_pr_auc=stitched.get("pr_auc", 0.0),
                                stitched_auroc=stitched["auroc"],
                                composite_score=composite_score,
                                recall_at_fp_2pct=recall_2pct,
                                recall_at_fp_5pct=recall_5pct,
                                fp_area_pct_median=fp_mass,
                                recommended_threshold=stitched.get("recommended_threshold", 0.5),
                            ),
                            out_dir / "best_model.pth",
                        )
                        rec_thr = stitched.get("recommended_threshold", 0.5)
                        print(f"  ✓ Saved best model (composite={composite_score:.4f}, R@2%={recall_2pct:.3f}, R@5%={recall_5pct:.3f}, FP-mass={fp_mass*100:.2f}%, rec_thr={rec_thr:.2f})", flush=True)

                elif args.early_stop_mode == "multi_fp_cap":
                    # Early stopping: monitor recall at multiple FP caps - stop only if ALL plateau
                    # Check if any FP cap recall improved
                    fp_caps = [0.005, 0.01, 0.02, 0.05, 0.15]  # 0.5%, 1%, 2%, 5%, 15%
                    any_improved = False
                    for fp_cap in fp_caps:
                        key = f"recall_at_fp_{fp_cap*100:.1f}pct"
                        current_recall = stitched.get(key, 0.0)
                        best_recall = best_multi_fp_recalls.get(key, -1.0)
                        if current_recall > best_recall:
                            best_multi_fp_recalls[key] = current_recall
                            any_improved = True

                    if any_improved:
                        improved = True
                        pr_auc = stitched.get("pr_auc", 0.0)
                        fp_mass = stitched.get("fp_area_pct_median", 1.0)
                        rec_thr = stitched.get("recommended_threshold", 0.5)
                        torch.save(
                            _checkpoint_payload(
                                epoch,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                stitched_pr_auc=pr_auc,
                                stitched_auroc=stitched["auroc"],
                                fp_area_pct_median=fp_mass,
                                recall_at_fp_cap=stitched.get("recall_at_fp_cap", 0.0),
                                recommended_threshold=rec_thr,
                                **best_multi_fp_recalls,
                            ),
                            out_dir / "best_model.pth",
                        )
                        recalls_str = ", ".join([f"{k.replace('recall_at_fp_', '')}={v:.3f}" for k, v in sorted(best_multi_fp_recalls.items())])
                        print(f"  ✓ Saved best model (rec_thr={rec_thr:.2f}, recalls: {recalls_str})", flush=True)

                if improved:
                    epochs_without_improvement_stitched = 0
                else:
                    epochs_without_improvement_stitched += 1
                    # Only check early stopping if we're past minimum training floor
                    if can_early_stop and epochs_without_improvement_stitched >= args.early_stop_patience:
                        if args.early_stop_mode == "score":
                            print(f"  Early stopping: score (PR-AUC - {args.early_stop_alpha}*FP-mass) has not improved for {args.early_stop_patience} stitched-val cycles.", flush=True)
                        elif args.early_stop_mode == "composite":
                            print(f"  Early stopping: composite score (0.5*R@2% + 0.5*R@5%) has not improved for {args.early_stop_patience} stitched-val cycles.", flush=True)
                        else:
                            print(f"  Early stopping: recall at all FP caps (0.5%/1%/2%/5%/15%) has plateaued for {args.early_stop_patience} stitched-val cycles.", flush=True)
                        # Explicitly shut down persistent workers
                        try:
                            if hasattr(train_loader, '_iterator'):
                                iterator = getattr(train_loader, '_iterator', None)
                                if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                                    iterator._shutdown_workers()
                        except Exception:
                            pass
                        train_loader = None
                        train_ds = None
                        gc.collect()
                        break

        # Release train loader/dataset so worker FDs are freed before next epoch (avoids "too many open files").
        # Explicitly shut down persistent workers to prevent FD leaks
        try:
            if hasattr(train_loader, '_iterator'):
                iterator = getattr(train_loader, '_iterator', None)
                if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                    iterator._shutdown_workers()
        except Exception:
            pass
        train_loader = None
        train_ds = None
        gc.collect()
        print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    main()
