# Publication classifier UI integration.
"""Numerical inference kernels from the FINAL_top76 publication classifier.

Derived from the frozen patch-aggregation evaluation implementation. Training,
label access, filesystem layout and evaluation are deliberately kept outside
these kernels. See docs/publication_classifier.md for provenance and metrics.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
import time
from typing import Any, Sequence

import numpy as np
import torch
from scipy import ndimage as ndi

from utils.contamination_subtypes import TYPE_ID_SEQUENCE, TYPE_ID_TO_KEY
from utils.image_utils import compute_frequency_band_psd_channels_batch

# Set CRYOFILTER_TYPING_PROFILE=1 to print a per-image phase-timing breakdown from
# predict_prepared (grid sampling, embeddings, handcrafted features, rescue pass,
# probability splatting, ...). Off by default - pure diagnostic, changes no computation.
_PROFILE_TYPING = bool(os.environ.get("CRYOFILTER_TYPING_PROFILE"))


class _PhaseTimer:
    """Accumulates named phase durations for one predict_prepared() call."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.totals: dict[str, float] = {}
        self._t0 = time.monotonic() if enabled else 0.0

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        self.totals[name] = self.totals.get(name, 0.0) + (now - self._t0)
        self._t0 = now

    def report(self, *, n_centers: int, n_rescue_centers: int) -> None:
        if not self.enabled:
            return
        parts = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in self.totals.items())
        total = sum(self.totals.values())
        print(
            f"[typing-profile] centers={n_centers} rescue_centers={n_rescue_centers} "
            f"total={total:.2f}s :: {parts}",
            flush=True,
        )

def _extract_centered_patch(arr: np.ndarray, cy: int, cx: int, patch_size: int, pad_mode: str) -> np.ndarray:
    half = patch_size // 2
    y0 = int(cy) - half
    x0 = int(cx) - half
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - arr.shape[0])
    pad_right = max(0, x1 - arr.shape[1])
    if pad_top or pad_left or pad_bottom or pad_right:
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=pad_mode)
        y0 += pad_top
        y1 += pad_top
        x0 += pad_left
        x1 += pad_left
    patch = np.asarray(arr[y0:y1, x0:x1])
    if patch.shape != (patch_size, patch_size):
        raise ValueError(f"Patch shape drift: expected {(patch_size, patch_size)} got {patch.shape}")
    return patch

def _embedding_from_feature_map(features: torch.Tensor, center_size: int) -> np.ndarray:
    _, _, h, w = features.shape
    cs = min(int(center_size), int(h), int(w))
    y0 = max(0, (h - cs) // 2)
    x0 = max(0, (w - cs) // 2)
    center = features[:, :, y0 : y0 + cs, x0 : x0 + cs]
    mean = center.mean(dim=(2, 3))
    std = center.std(dim=(2, 3), unbiased=False)
    energy = torch.sqrt(torch.clamp((center ** 2).mean(dim=(2, 3)), min=1e-8))
    return torch.cat([mean, std, energy], dim=1).detach().cpu().numpy().astype(np.float32, copy=False)

def _build_input_batch(
    patches: np.ndarray,
    *,
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
    input_mode: str = "auto",
    input_channels: int | None = None,
    use_power_spectrum: bool = True,
    include_real_space_input: bool = True,
) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float32)
    real = patches[:, None, :, :].astype(np.float32, copy=False)

    def _psd_channels() -> np.ndarray:
        return compute_frequency_band_psd_channels_batch(
            patches.astype(np.float32, copy=False),
            pixel_size_angstrom=float(pixel_size_angstrom),
            frequency_bands=frequency_bands,
            normalize=True,
            use_radial_normalization=True,
            include_anisotropy_channel=False,
        )

    def _coerce_psd_channels(psd: np.ndarray, expected: int | None) -> np.ndarray:
        if expected is None:
            return psd.astype(np.float32, copy=False)
        expected_i = int(expected)
        if expected_i == psd.shape[1]:
            return psd.astype(np.float32, copy=False)
        if expected_i == 1:
            return psd.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
        if expected_i <= 0:
            return np.zeros((patches.shape[0], 0, patches.shape[1], patches.shape[2]), dtype=np.float32)
        raise ValueError(
            f"Cannot coerce {psd.shape[1]} PSD channels to expected input channel count {expected_i}."
        )

    mode = str(input_mode or "auto").strip().lower().replace("-", "_")
    aliases = {
        "default": "auto",
        "checkpoint": "auto",
        "real": "real_only",
        "no_psd": "zero_psd",
        "no_fourier": "zero_psd",
        "zero_fourier": "zero_psd",
        "psd": "psd_only",
        "fourier_only": "psd_only",
        "no_real": "zero_real",
        "no_realspace": "zero_real",
        "no_real_space": "zero_real",
        "zero_realspace": "zero_real",
    }
    mode = aliases.get(mode, mode)
    if mode == "auto":
        if bool(use_power_spectrum):
            mode = "full" if bool(include_real_space_input) else "psd_only"
        else:
            mode = "real_only"

    expected_channels = None if input_channels is None else int(input_channels)

    if mode == "real_only":
        out = real
    elif mode == "full":
        psd = _coerce_psd_channels(_psd_channels(), None if expected_channels is None else expected_channels - 1)
        out = np.concatenate([real, psd], axis=1)
    elif mode == "psd_only":
        out = _coerce_psd_channels(_psd_channels(), expected_channels)
    elif mode == "zero_psd":
        psd_count = 4 if expected_channels is None else max(expected_channels - 1, 0)
        zeros = np.zeros((patches.shape[0], psd_count, patches.shape[1], patches.shape[2]), dtype=np.float32)
        out = np.concatenate([real, zeros], axis=1)
    elif mode == "zero_real":
        psd = _coerce_psd_channels(_psd_channels(), None if expected_channels is None else expected_channels - 1)
        zeros = np.zeros_like(real)
        out = np.concatenate([zeros, psd], axis=1)
    else:
        raise ValueError(
            "input_mode must be one of auto, full, real_only, psd_only, zero_psd, or zero_real "
            f"(got {input_mode!r})."
        )

    if expected_channels is not None and out.shape[1] != expected_channels:
        raise ValueError(
            f"Built {out.shape[1]} input channels for mode {mode!r}, but checkpoint expects {expected_channels}."
        )
    return out.astype(np.float32, copy=False)

@dataclass(frozen=True)
class ScaleSpec:
    patch_size: int
    embedding_center_size: int

@dataclass(frozen=True)
class PairExpertSpec:
    name: str
    class_a_id: int
    class_b_id: int
    min_total_prob: float
    require_top2_pair: bool
    max_margin: float
    blend: float
    max_border_distance_px: float
    max_corner_distance_px: float
    require_component_touches_border: bool
    apply_on_patch_probe: bool

def _classifier_classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(getattr(model, "classes_"), dtype=np.int64)
    if hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "classes_"):
        return np.asarray(model.named_steps["clf"].classes_, dtype=np.int64)
    raise AttributeError(f"Could not resolve classes_ from classifier type {type(model).__name__}.")

def _row_normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    denom = probs.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return (probs / denom).astype(np.float32, copy=False)

def _apply_prob_power(probs: np.ndarray, power: float) -> np.ndarray:
    if abs(float(power) - 1.0) <= 1e-6:
        return np.asarray(probs, dtype=np.float32)
    safe = np.clip(np.asarray(probs, dtype=np.float32), 1e-8, None)
    sharpened = np.power(safe, float(power), dtype=np.float32)
    return _row_normalize_probs(sharpened)

def _apply_pair_refine_probs(
    probs_global: np.ndarray,
    *,
    feature_vectors: np.ndarray,
    pair_refine_model: Any | None,
    agg_idx: int,
    eth_idx: int,
    min_total_prob: float,
    require_top2_pair: bool,
) -> np.ndarray:
    if pair_refine_model is None:
        return np.asarray(probs_global, dtype=np.float32)

    refined = np.array(probs_global, dtype=np.float32, copy=True)
    pair_prob = pair_refine_model.predict_proba(feature_vectors)[:, 1].astype(np.float32, copy=False)
    pair_total = refined[:, agg_idx] + refined[:, eth_idx]
    refine_mask = pair_total >= float(min_total_prob)
    if require_top2_pair and np.any(refine_mask):
        top2 = np.sort(np.argsort(refined, axis=1)[:, -2:], axis=1)
        pair_sorted = np.asarray(sorted([int(agg_idx), int(eth_idx)]), dtype=np.int64)
        refine_mask &= np.all(top2 == pair_sorted[None, :], axis=1)
    if np.any(refine_mask):
        refined[refine_mask, agg_idx] = pair_total[refine_mask] * pair_prob[refine_mask]
        refined[refine_mask, eth_idx] = pair_total[refine_mask] * (1.0 - pair_prob[refine_mask])
    return refined

def _apply_single_pair_expert_probs(
    probs_global: np.ndarray,
    *,
    pair_prob: np.ndarray,
    class_a_idx: int,
    class_b_idx: int,
    min_total_prob: float,
    require_top2_pair: bool,
    max_margin: float,
    blend: float,
    extra_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    refined = np.array(probs_global, dtype=np.float32, copy=True)
    pair_prob = np.asarray(pair_prob, dtype=np.float32)
    pair_total = refined[:, int(class_a_idx)] + refined[:, int(class_b_idx)]
    refine_mask = pair_total >= float(min_total_prob)
    if require_top2_pair and np.any(refine_mask):
        top2 = np.sort(np.argsort(refined, axis=1)[:, -2:], axis=1)
        pair_sorted = np.asarray(sorted([int(class_a_idx), int(class_b_idx)]), dtype=np.int64)
        refine_mask &= np.all(top2 == pair_sorted[None, :], axis=1)
    if float(max_margin) > 0.0 and np.any(refine_mask):
        pair_margin = np.abs(refined[:, int(class_a_idx)] - refined[:, int(class_b_idx)])
        refine_mask &= pair_margin <= float(max_margin)
    if extra_mask is not None:
        refine_mask &= np.asarray(extra_mask, dtype=bool)
    apply_count = int(np.sum(refine_mask))
    if not np.any(refine_mask):
        return refined, apply_count

    pair_total_masked = np.maximum(pair_total[refine_mask], 1e-8)
    old_share_a = refined[refine_mask, int(class_a_idx)] / pair_total_masked
    alpha = float(np.clip(blend, 0.0, 1.0))
    new_share_a = (1.0 - alpha) * old_share_a + alpha * pair_prob[refine_mask]
    new_share_b = 1.0 - new_share_a
    refined[refine_mask, int(class_a_idx)] = pair_total_masked * new_share_a
    refined[refine_mask, int(class_b_idx)] = pair_total_masked * new_share_b
    return refined, apply_count

def _apply_pair_expert_stack_soft(
    probs_global: np.ndarray,
    *,
    feature_vectors: np.ndarray,
    pair_expert_models: Sequence[tuple[PairExpertSpec, Any]],
    class_to_global: dict[int, int],
    centers: Sequence[tuple[int, int]],
    image_shape: tuple[int, int],
    component_ids: Sequence[int],
    component_touches_border: dict[int, bool],
) -> tuple[np.ndarray, dict[str, int]]:
    refined = np.asarray(probs_global, dtype=np.float32)
    if not pair_expert_models:
        return refined, {}
    if not centers:
        return refined, {}

    h, w = int(image_shape[0]), int(image_shape[1])
    centers_np = np.asarray(centers, dtype=np.float32)
    border_dist = np.minimum.reduce(
        [
            centers_np[:, 0],
            centers_np[:, 1],
            float(h - 1) - centers_np[:, 0],
            float(w - 1) - centers_np[:, 1],
        ]
    ).astype(np.float32, copy=False)
    corners = np.asarray(
        [[0.0, 0.0], [0.0, float(w - 1)], [float(h - 1), 0.0], [float(h - 1), float(w - 1)]],
        dtype=np.float32,
    )
    corner_dist = np.min(
        np.sqrt(np.sum(np.square(centers_np[:, None, :] - corners[None, :, :]), axis=2), dtype=np.float32),
        axis=1,
    ).astype(np.float32, copy=False)
    touches_border = np.asarray(
        [bool(component_touches_border.get(int(component_id), False)) for component_id in component_ids],
        dtype=bool,
    )

    apply_counts: dict[str, int] = {}
    for spec, model in pair_expert_models:
        class_a_idx = class_to_global.get(int(spec.class_a_id))
        class_b_idx = class_to_global.get(int(spec.class_b_id))
        if class_a_idx is None or class_b_idx is None:
            continue
        pair_prob = model.predict_proba(feature_vectors)[:, 1].astype(np.float32, copy=False)
        extra_mask = np.ones((refined.shape[0],), dtype=bool)
        extra_mask &= border_dist <= float(spec.max_border_distance_px)
        extra_mask &= corner_dist <= float(spec.max_corner_distance_px)
        if bool(spec.require_component_touches_border):
            extra_mask &= touches_border
        refined, count = _apply_single_pair_expert_probs(
            refined,
            pair_prob=pair_prob,
            class_a_idx=int(class_a_idx),
            class_b_idx=int(class_b_idx),
            min_total_prob=float(spec.min_total_prob),
            require_top2_pair=bool(spec.require_top2_pair),
            max_margin=float(spec.max_margin),
            blend=float(spec.blend),
            extra_mask=extra_mask,
        )
        apply_counts[str(spec.name)] = int(count)
    return refined, apply_counts

def _top2_contains(class_probs: np.ndarray, class_idx: int) -> np.ndarray:
    if class_probs.shape[1] < 2:
        return np.zeros((class_probs.shape[0],), dtype=bool)
    top2 = np.argsort(class_probs, axis=1)[:, -2:]
    return np.any(top2 == int(class_idx), axis=1)

def _class_sigma_vector(default_sigma_px: float, overrides: dict[str, float]) -> np.ndarray:
    values: list[float] = []
    for type_id in TYPE_ID_SEQUENCE:
        key = TYPE_ID_TO_KEY[type_id]
        values.append(max(float(overrides.get(key, default_sigma_px)), 1e-3))
    return np.asarray(values, dtype=np.float32)

def _class_prob_weight_vector(overrides: dict[str, float]) -> np.ndarray:
    values: list[float] = []
    for type_id in TYPE_ID_SEQUENCE:
        key = TYPE_ID_TO_KEY[type_id]
        values.append(max(float(overrides.get(key, 1.0)), 1e-6))
    return np.asarray(values, dtype=np.float32)

def _apply_class_probability_weights(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32).reshape(1, -1)
    if weights.shape[1] != np.asarray(probs).shape[1]:
        raise ValueError(
            f"class probability weights dimension mismatch: {weights.shape[1]} vs {np.asarray(probs).shape[1]}"
        )
    if np.allclose(weights, 1.0, atol=1e-6):
        return np.asarray(probs, dtype=np.float32)
    weighted = np.asarray(probs, dtype=np.float32) * weights
    return _row_normalize_probs(weighted)

def _apply_carbon_border_corner_rescue(
    probs: np.ndarray,
    *,
    centers: Sequence[tuple[int, int]],
    image_shape: tuple[int, int],
    component_ids: Sequence[int],
    component_touches_border: dict[int, bool],
    carbon_idx: int | None,
    crystalline_idx: int | None,
    edge_enabled: bool,
    edge_max_border_distance_px: float,
    edge_min_carbon_prob: float,
    edge_max_crystalline_margin: float,
    edge_boost: float,
    corner_enabled: bool,
    corner_max_border_distance_px: float,
    corner_max_corner_distance_px: float,
    corner_min_carbon_prob: float,
    corner_max_crystalline_margin: float,
    corner_boost: float,
) -> np.ndarray:
    if carbon_idx is None or crystalline_idx is None:
        return np.asarray(probs, dtype=np.float32)
    if not edge_enabled and not corner_enabled:
        return np.asarray(probs, dtype=np.float32)

    out = np.asarray(probs, dtype=np.float32).copy()
    h, w = int(image_shape[0]), int(image_shape[1])
    corners = np.asarray(
        [[0.0, 0.0], [0.0, float(w - 1)], [float(h - 1), 0.0], [float(h - 1), float(w - 1)]],
        dtype=np.float32,
    )

    for idx, ((cy, cx), component_id) in enumerate(zip(centers, component_ids, strict=True)):
        prob = out[idx]
        if int(np.argmax(prob)) != int(crystalline_idx):
            continue

        border_dist = float(min(cy, cx, h - 1 - int(cy), w - 1 - int(cx)))
        carbon_prob = float(prob[int(carbon_idx)])
        crystalline_prob = float(prob[int(crystalline_idx)])
        margin = crystalline_prob - carbon_prob
        if margin < 0:
            continue

        boost = 0.0
        if (
            edge_enabled
            and component_touches_border.get(int(component_id), False)
            and border_dist <= float(edge_max_border_distance_px)
            and carbon_prob >= float(edge_min_carbon_prob)
            and margin <= float(edge_max_crystalline_margin)
        ):
            boost = max(boost, float(edge_boost))

        if corner_enabled:
            point = np.asarray([float(cy), float(cx)], dtype=np.float32)
            corner_dist = float(np.min(np.sqrt(np.sum(np.square(corners - point[None, :]), axis=1))))
            if (
                border_dist <= float(corner_max_border_distance_px)
                and corner_dist <= float(corner_max_corner_distance_px)
                and carbon_prob >= float(corner_min_carbon_prob)
                and margin <= float(corner_max_crystalline_margin)
            ):
                boost = max(boost, float(corner_boost))

        if boost <= 0:
            continue

        prob[int(carbon_idx)] *= boost
        prob[int(crystalline_idx)] /= max(np.sqrt(boost), 1e-6)
        out[idx] = prob

    return _row_normalize_probs(out)

def _apply_component_geometry_postprocess(
    pred_map: np.ndarray,
    pred_mask: np.ndarray,
    *,
    enabled: bool,
    aggregate_edge_to_carbon_enabled: bool,
    aggregate_edge_max_border_px: float,
    aggregate_edge_min_area_px: int,
    aggregate_edge_min_aspect: float,
    aggregate_edge_carbon_dilation_px: int,
    carbon_interior_to_aggregate_enabled: bool,
    carbon_interior_min_border_px: float,
    carbon_interior_min_area_px: int,
    carbon_interior_aggregate_dilation_px: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply conservative component-level subtype cleanup rules.

    These rules only change subtype predictions inside the already-captured
    binary contamination mask. They do not add or remove contamination pixels.
    """
    if not enabled or (not aggregate_edge_to_carbon_enabled and not carbon_interior_to_aggregate_enabled):
        return np.asarray(pred_map, dtype=np.uint8), {}

    out = np.asarray(pred_map, dtype=np.uint8).copy()
    mask = np.asarray(pred_mask, dtype=bool)
    h, w = int(out.shape[0]), int(out.shape[1])
    structure = np.ones((3, 3), dtype=np.uint8)
    apply_counts: dict[str, int] = {}

    if aggregate_edge_to_carbon_enabled:
        agg_components, _num_agg = ndi.label((out == np.uint8(3)) & mask, structure=structure)
        objects = ndi.find_objects(agg_components)
        carbon_neighborhood = ndi.binary_dilation(
            (out == np.uint8(1)) & mask,
            iterations=max(0, int(aggregate_edge_carbon_dilation_px)),
        )
        changed = 0
        for component_id, component_slice in enumerate(objects, start=1):
            if component_slice is None:
                continue
            local = agg_components[component_slice] == int(component_id)
            area = int(local.sum())
            if area < int(aggregate_edge_min_area_px):
                continue
            y0 = int(component_slice[0].start or 0)
            x0 = int(component_slice[1].start or 0)
            y1 = int(component_slice[0].stop or 0)
            x1 = int(component_slice[1].stop or 0)
            border_dist = min(y0, x0, h - y1, w - x1)
            box_h = max(y1 - y0, 1)
            box_w = max(x1 - x0, 1)
            aspect = max(box_h / float(box_w), box_w / float(box_h))
            if (
                float(border_dist) <= float(aggregate_edge_max_border_px)
                and float(aspect) >= float(aggregate_edge_min_aspect)
                and np.any(carbon_neighborhood[component_slice] & local)
            ):
                component_pixels = agg_components == int(component_id)
                out[component_pixels] = np.uint8(1)
                changed += int(component_pixels.sum())
        if changed:
            apply_counts["aggregate_edge_to_carbon"] = int(changed)

    if carbon_interior_to_aggregate_enabled:
        carbon_components, _num_carbon = ndi.label((out == np.uint8(1)) & mask, structure=structure)
        objects = ndi.find_objects(carbon_components)
        aggregate_neighborhood = ndi.binary_dilation(
            (out == np.uint8(3)) & mask,
            iterations=max(0, int(carbon_interior_aggregate_dilation_px)),
        )
        changed = 0
        for component_id, component_slice in enumerate(objects, start=1):
            if component_slice is None:
                continue
            local = carbon_components[component_slice] == int(component_id)
            area = int(local.sum())
            if area < int(carbon_interior_min_area_px):
                continue
            y0 = int(component_slice[0].start or 0)
            x0 = int(component_slice[1].start or 0)
            y1 = int(component_slice[0].stop or 0)
            x1 = int(component_slice[1].stop or 0)
            border_dist = min(y0, x0, h - y1, w - x1)
            if (
                float(border_dist) >= float(carbon_interior_min_border_px)
                and np.any(aggregate_neighborhood[component_slice] & local)
            ):
                component_pixels = carbon_components == int(component_id)
                out[component_pixels] = np.uint8(3)
                changed += int(component_pixels.sum())
        if changed:
            apply_counts["carbon_interior_to_aggregate"] = int(changed)

    return out, apply_counts

def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    cs = min(int(size), int(h), int(w))
    y0 = max(0, (h - cs) // 2)
    x0 = max(0, (w - cs) // 2)
    return np.asarray(arr[y0 : y0 + cs, x0 : x0 + cs])

def _safe_patch_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
        f"{prefix}_max": float(arr.max()),
    }

def _make_ring_mask(shape: tuple[int, int], inner_size: int) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    mask = np.ones((h, w), dtype=bool)
    inner = min(int(inner_size), h, w)
    y0 = max(0, (h - inner) // 2)
    x0 = max(0, (w - inner) // 2)
    mask[y0 : y0 + inner, x0 : x0 + inner] = False
    return mask

def _fft_anisotropy_features(center: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(center, dtype=np.float32)
    arr = arr - float(arr.mean())
    power = np.square(np.abs(np.fft.fftshift(np.fft.fft2(arr))), dtype=np.float32)
    h, w = power.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt(np.square(yy - cy) + np.square(xx - cx), dtype=np.float32)
    rr /= max(min(h, w) * 0.5, 1.0)
    theta = np.arctan2(yy - cy, xx - cx).astype(np.float32, copy=False)
    annuli = ((0.08, 0.16), (0.16, 0.28), (0.28, 0.42))
    sectors = np.linspace(-np.pi, np.pi, num=9, dtype=np.float32)
    feats: dict[str, float] = {}
    for ann_idx, (r0, r1) in enumerate(annuli):
        ann_mask = (rr >= float(r0)) & (rr < float(r1))
        vals = power[ann_mask]
        if vals.size == 0:
            feats[f"{prefix}_ann{ann_idx}_anisotropy"] = 0.0
            feats[f"{prefix}_ann{ann_idx}_peakiness"] = 0.0
            feats[f"{prefix}_ann{ann_idx}_entropy"] = 1.0
            feats[f"{prefix}_ann{ann_idx}_q95"] = 0.0
            continue
        sector_means: list[float] = []
        for s0, s1 in zip(sectors[:-1], sectors[1:], strict=True):
            sec_mask = ann_mask & (theta >= float(s0)) & (theta < float(s1))
            sec_vals = power[sec_mask]
            sector_means.append(float(sec_vals.mean()) if sec_vals.size else 0.0)
        sector_arr = np.asarray(sector_means, dtype=np.float32)
        denom = float(max(sector_arr.mean(), 1e-6))
        probs = sector_arr / max(float(sector_arr.sum()), 1e-6)
        entropy = -float(np.sum(probs * np.log(probs + 1e-8))) / math.log(len(sector_arr))
        feats[f"{prefix}_ann{ann_idx}_anisotropy"] = float(sector_arr.std() / denom)
        feats[f"{prefix}_ann{ann_idx}_peakiness"] = float(sector_arr.max() / denom)
        feats[f"{prefix}_ann{ann_idx}_entropy"] = float(entropy)
        feats[f"{prefix}_ann{ann_idx}_q95"] = float(np.quantile(vals, 0.95))
    return feats

def _structure_tensor_summary(gx: np.ndarray, gy: np.ndarray) -> tuple[float, float, float]:
    gx_arr = np.asarray(gx, dtype=np.float32)
    gy_arr = np.asarray(gy, dtype=np.float32)
    jxx = float(np.mean(np.square(gx_arr, dtype=np.float32)))
    jyy = float(np.mean(np.square(gy_arr, dtype=np.float32)))
    jxy = float(np.mean(gx_arr * gy_arr))
    trace = max(jxx + jyy, 1e-8)
    diff = jxx - jyy
    root = float(np.sqrt(max(diff * diff + 4.0 * jxy * jxy, 0.0)))
    coherence = float(root / trace)
    angle = 0.5 * math.atan2(2.0 * jxy, diff)
    axis_alignedness = float(max(abs(math.cos(angle)), abs(math.sin(angle))))
    det = max(jxx * jyy - jxy * jxy, 0.0)
    cornerness = float((4.0 * det) / max(trace * trace, 1e-8))
    return coherence, axis_alignedness, cornerness

def _gradient_polarity_features(
    center: np.ndarray,
    *,
    prefix: str,
    highpass_reference: np.ndarray | None = None,
) -> dict[str, float]:
    arr = np.asarray(center, dtype=np.float32)
    gx = ndi.sobel(arr, axis=1).astype(np.float32, copy=False)
    gy = ndi.sobel(arr, axis=0).astype(np.float32, copy=False)
    coherence, axis_alignedness, cornerness = _structure_tensor_summary(gx, gy)

    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx_n = ((xx.astype(np.float32) - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)).reshape(-1)
    yy_n = ((yy.astype(np.float32) - (h - 1) * 0.5) / max((h - 1) * 0.5, 1.0)).reshape(-1)
    design = np.stack(
        [
            xx_n,
            yy_n,
            np.ones_like(xx_n, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    target = arr.reshape(-1).astype(np.float32, copy=False)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = design @ coeffs
    resid = target - pred
    var_target = float(np.var(target))
    plane_r2 = float(1.0 - (np.var(resid) / max(var_target, 1e-8)))
    slope_x = float(coeffs[0])
    slope_y = float(coeffs[1])
    slope_mag = float(np.sqrt(slope_x * slope_x + slope_y * slope_y))
    plane_axis_alignedness = float(
        max(abs(slope_x), abs(slope_y)) / max(abs(slope_x) + abs(slope_y), 1e-8)
    )
    resid_std = float(np.std(resid))
    hp_ref = np.asarray(highpass_reference if highpass_reference is not None else resid.reshape(arr.shape), dtype=np.float32)
    hp_energy = float(np.mean(np.abs(hp_ref)))
    ramp_over_texture = float(slope_mag / max(hp_energy, 1e-6))
    texture_over_ramp = float(hp_energy / max(slope_mag, 1e-6))
    return {
        f"{prefix}_grad_coherence": coherence,
        f"{prefix}_grad_axis_alignedness": axis_alignedness,
        f"{prefix}_grad_cornerness": cornerness,
        f"{prefix}_plane_slope_mag": slope_mag,
        f"{prefix}_plane_axis_alignedness": plane_axis_alignedness,
        f"{prefix}_plane_r2": plane_r2,
        f"{prefix}_plane_resid_std": resid_std,
        f"{prefix}_ramp_over_texture": ramp_over_texture,
        f"{prefix}_texture_over_ramp": texture_over_ramp,
    }

def _fft_bragg_features(center: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(center, dtype=np.float32)
    arr = arr - float(arr.mean())
    power = np.square(np.abs(np.fft.fftshift(np.fft.fft2(arr))), dtype=np.float32)
    power = ndi.gaussian_filter(power, sigma=1.0).astype(np.float32, copy=False)
    h, w = power.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt(np.square(yy - cy) + np.square(xx - cx), dtype=np.float32)
    rr /= max(min(h, w) * 0.5, 1.0)
    theta = np.arctan2(yy - cy, xx - cx).astype(np.float32, copy=False)
    annuli = ((0.08, 0.16), (0.16, 0.28), (0.28, 0.42))
    sectors = np.linspace(-np.pi, np.pi, num=17, dtype=np.float32)
    feats: dict[str, float] = {}
    for ann_idx, (r0, r1) in enumerate(annuli):
        ann_mask = (rr >= float(r0)) & (rr < float(r1))
        vals = power[ann_mask]
        if vals.size == 0:
            feats[f"{prefix}_ann{ann_idx}_peak_count_norm"] = 0.0
            feats[f"{prefix}_ann{ann_idx}_top1_over_mean"] = 0.0
            feats[f"{prefix}_ann{ann_idx}_top4_frac"] = 0.0
            feats[f"{prefix}_ann{ann_idx}_opp_symmetry"] = 0.0
            feats[f"{prefix}_ann{ann_idx}_peak_sparsity"] = 0.0
            continue

        sector_means: list[float] = []
        for s0, s1 in zip(sectors[:-1], sectors[1:], strict=True):
            sec_mask = ann_mask & (theta >= float(s0)) & (theta < float(s1))
            sec_vals = power[sec_mask]
            sector_means.append(float(sec_vals.mean()) if sec_vals.size else 0.0)
        sector_arr = np.asarray(sector_means, dtype=np.float32)
        mean_sector = float(max(sector_arr.mean(), 1e-6))
        top_sorted = np.sort(sector_arr)
        top1 = float(top_sorted[-1])
        top4_frac = float(top_sorted[-4:].sum() / max(float(sector_arr.sum()), 1e-6))
        opp_pairs = []
        half = sector_arr.size // 2
        for i in range(half):
            opp_pairs.append(min(float(sector_arr[i]), float(sector_arr[i + half])) / mean_sector)
        opp_symmetry = float(np.mean(opp_pairs)) if opp_pairs else 0.0

        local_max = ann_mask & (power >= ndi.maximum_filter(power, size=3))
        thr = float(vals.mean() + 2.0 * vals.std())
        peak_mask = local_max & (power >= thr)
        peak_count_norm = float(peak_mask.sum() / max(np.sqrt(float(vals.size)), 1.0))
        peak_sparsity = float(np.quantile(vals, 0.995) / max(vals.mean(), 1e-6))
        feats[f"{prefix}_ann{ann_idx}_peak_count_norm"] = peak_count_norm
        feats[f"{prefix}_ann{ann_idx}_top1_over_mean"] = float(top1 / mean_sector)
        feats[f"{prefix}_ann{ann_idx}_top4_frac"] = top4_frac
        feats[f"{prefix}_ann{ann_idx}_opp_symmetry"] = opp_symmetry
        feats[f"{prefix}_ann{ann_idx}_peak_sparsity"] = peak_sparsity
    return feats

def _compute_center_geometry_feature_rows(
    contamination_mask: np.ndarray,
    centers: Sequence[tuple[int, int]],
) -> np.ndarray:
    mask = np.asarray(contamination_mask, dtype=bool)
    h, w = mask.shape
    if not np.any(mask):
        return np.zeros((len(centers), 8), dtype=np.float32)
    component_map, num_components = ndi.label(mask)
    objects = ndi.find_objects(component_map)
    dist = ndi.distance_transform_edt(mask).astype(np.float32, copy=False)
    comp_stats: dict[int, tuple[float, float, float, float, float]] = {}
    image_area = float(max(h * w, 1))
    image_scale = float(max(max(h, w), 1))
    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        local = component_map[slc] == int(component_id)
        area = int(local.sum())
        if area <= 0:
            continue
        y0 = int(slc[0].start or 0)
        x0 = int(slc[1].start or 0)
        y1 = int(slc[0].stop or 0)
        x1 = int(slc[1].stop or 0)
        bh = max(y1 - y0, 1)
        bw = max(x1 - x0, 1)
        fill = float(area / float(max(bh * bw, 1)))
        touches_border = float(y0 == 0 or x0 == 0 or y1 == h or x1 == w)
        local_dist = dist[slc][local]
        p90_thickness = float(np.quantile(local_dist, 0.90)) / image_scale if local_dist.size else 0.0
        mean_thickness = float(local_dist.mean()) / image_scale if local_dist.size else 0.0
        aspect = float(max(bh, bw) / float(max(min(bh, bw), 1)))
        comp_stats[int(component_id)] = (
            float(np.log1p(area) / np.log1p(image_area)),
            float(np.log(aspect + 1e-6)),
            fill,
            touches_border,
            p90_thickness,
            mean_thickness,
        )
    rows: list[list[float]] = []
    for cy, cx in centers:
        cy_i = int(np.clip(cy, 0, h - 1))
        cx_i = int(np.clip(cx, 0, w - 1))
        border_dist = min(cy_i, cx_i, h - 1 - cy_i, w - 1 - cx_i) / image_scale
        boundary_dist = float(dist[cy_i, cx_i]) / image_scale if mask[cy_i, cx_i] else 0.0
        component_id = int(component_map[cy_i, cx_i])
        area_log, aspect_log, fill, touches_border, p90_thickness, mean_thickness = comp_stats.get(
            component_id,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        rows.append(
            [
                float(border_dist),
                float(boundary_dist),
                float(area_log),
                float(aspect_log),
                float(fill),
                float(touches_border),
                float(p90_thickness),
                float(mean_thickness),
            ]
        )
    return np.asarray(rows, dtype=np.float32)

def _compute_center_edgecorner_geometry_feature_rows(
    contamination_mask: np.ndarray,
    centers: Sequence[tuple[int, int]],
    *,
    local_patch_size: int = 96,
    border_band_px: int = 8,
    corner_box_px: int = 64,
) -> np.ndarray:
    mask = np.asarray(contamination_mask, dtype=bool)
    h, w = mask.shape
    if not np.any(mask):
        return np.zeros((len(centers), len(_EDGECORNER_GEOMETRY_FEATURE_NAMES)), dtype=np.float32)
    component_map, _num_components = ndi.label(mask)
    objects = ndi.find_objects(component_map)
    component_stats: dict[int, tuple[float, float, float, float, float, float]] = {}
    band = max(1, int(border_band_px))
    corner_box = max(4, int(min(corner_box_px, h, w)))
    for component_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        local = component_map[slc] == int(component_id)
        area = int(local.sum())
        if area <= 0:
            continue
        y0 = int(slc[0].start or 0)
        x0 = int(slc[1].start or 0)
        y1 = int(slc[0].stop or 0)
        x1 = int(slc[1].stop or 0)
        touches_top = y0 == 0
        touches_left = x0 == 0
        touches_bottom = y1 == h
        touches_right = x1 == w
        border_touch_count = int(touches_top) + int(touches_left) + int(touches_bottom) + int(touches_right)
        adjacent_pair = float(
            (touches_top and touches_left)
            or (touches_top and touches_right)
            or (touches_bottom and touches_left)
            or (touches_bottom and touches_right)
        )
        opposite_pair = float((touches_top and touches_bottom) or (touches_left and touches_right))
        coverages: list[float] = []
        if touches_top:
            coverages.append(float(mask[:band, :].any(axis=0).mean()))
        if touches_bottom:
            coverages.append(float(mask[max(h - band, 0) :, :].any(axis=0).mean()))
        if touches_left:
            coverages.append(float(mask[:, :band].any(axis=1).mean()))
        if touches_right:
            coverages.append(float(mask[:, max(w - band, 0) :].any(axis=1).mean()))
        max_border_coverage = float(max(coverages)) if coverages else 0.0
        mean_border_coverage = float(np.mean(coverages)) if coverages else 0.0
        corner_counts = [
            int(mask[:corner_box, :corner_box].sum()),
            int(mask[:corner_box, max(w - corner_box, 0) :].sum()),
            int(mask[max(h - corner_box, 0) :, :corner_box].sum()),
            int(mask[max(h - corner_box, 0) :, max(w - corner_box, 0) :].sum()),
        ]
        corner_mass_frac = float(max(corner_counts) / max(area, 1))
        component_stats[int(component_id)] = (
            float(border_touch_count / 4.0),
            adjacent_pair,
            opposite_pair,
            max_border_coverage,
            mean_border_coverage,
            corner_mass_frac,
        )

    rows: list[list[float]] = []
    mask_float = mask.astype(np.float32, copy=False)
    for cy, cx in centers:
        cy_i = int(np.clip(cy, 0, h - 1))
        cx_i = int(np.clip(cx, 0, w - 1))
        component_id = int(component_map[cy_i, cx_i])
        stats = component_stats.get(component_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        local_patch = _extract_centered_patch(mask_float, cy_i, cx_i, int(local_patch_size), pad_mode="constant").astype(
            np.float32, copy=False
        )
        local_smooth = ndi.gaussian_filter(local_patch, sigma=1.0).astype(np.float32, copy=False)
        gx = ndi.sobel(local_smooth, axis=1).astype(np.float32, copy=False)
        gy = ndi.sobel(local_smooth, axis=0).astype(np.float32, copy=False)
        coherence, axis_alignedness, cornerness = _structure_tensor_summary(gx, gy)
        edge_density = float(np.mean(np.hypot(gx, gy)))
        rows.append(
            [
                float(stats[0]),
                float(stats[1]),
                float(stats[2]),
                float(stats[3]),
                float(stats[4]),
                float(stats[5]),
                float(coherence),
                float(axis_alignedness),
                float(cornerness),
                float(edge_density),
            ]
        )
    return np.asarray(rows, dtype=np.float32)

def _compute_single_handcrafted_feature_dict(
    patch: np.ndarray,
    *,
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
    center_size: int,
    append_real_space_handcrafted_features: bool,
    append_psd_band_features: bool,
    append_anisotropic_fourier_features: bool,
    append_background_reference_features: bool,
    append_gradient_polarity_features: bool,
    append_bragg_fourier_features: bool,
) -> dict[str, float]:
    patch_arr = np.asarray(patch, dtype=np.float32)
    if append_psd_band_features:
        psd = _build_input_batch(
            patch_arr[None, ...],
            pixel_size_angstrom=float(pixel_size_angstrom),
            frequency_bands=frequency_bands,
            input_mode="psd_only",
        )[0]
    else:
        psd = np.zeros((0, patch_arr.shape[0], patch_arr.shape[1]), dtype=np.float32)
    center = _center_crop(patch_arr, int(center_size)).astype(np.float32, copy=False)
    outer_size = min(int(center_size) * 2, int(patch_arr.shape[0]), int(patch_arr.shape[1]))
    outer = _center_crop(patch_arr, int(outer_size)).astype(np.float32, copy=False)
    ring_mask = _make_ring_mask(outer.shape, inner_size=int(center.shape[0]))
    ring_vals = outer[ring_mask]
    edge = np.hypot(ndi.sobel(center, axis=0), ndi.sobel(center, axis=1)).astype(np.float32, copy=False)
    blurred15 = ndi.gaussian_filter(center, sigma=1.5).astype(np.float32, copy=False)
    blurred40 = ndi.gaussian_filter(center, sigma=4.0).astype(np.float32, copy=False)
    high15 = np.abs(center - blurred15).astype(np.float32, copy=False)
    high40 = np.abs(center - blurred40).astype(np.float32, copy=False)
    sq = np.square(center, dtype=np.float32)
    mean_sq15 = ndi.gaussian_filter(sq, sigma=1.5).astype(np.float32, copy=False)
    mean_sq40 = ndi.gaussian_filter(sq, sigma=4.0).astype(np.float32, copy=False)
    std15 = np.sqrt(np.maximum(mean_sq15 - np.square(blurred15, dtype=np.float32), 0.0)).astype(np.float32, copy=False)
    std40 = np.sqrt(np.maximum(mean_sq40 - np.square(blurred40, dtype=np.float32), 0.0)).astype(np.float32, copy=False)

    feats: dict[str, float] = {}
    if append_real_space_handcrafted_features:
        feats.update(_safe_patch_stats(center, "center_intensity"))
        feats.update(_safe_patch_stats(edge, "center_edge"))
        feats.update(_safe_patch_stats(high15, "highpass15"))
        feats.update(_safe_patch_stats(high40, "highpass40"))
        feats.update(_safe_patch_stats(std15, "localstd15"))
        feats.update(_safe_patch_stats(std40, "localstd40"))
        if append_background_reference_features:
            feats.update(_safe_patch_stats(ring_vals, "ring_intensity"))
            feats["center_minus_ring_mean"] = (
                float(center.mean() - ring_vals.mean()) if ring_vals.size else float(center.mean())
            )
            feats["center_minus_ring_std"] = (
                float(center.std() - ring_vals.std()) if ring_vals.size else float(center.std())
            )
    for band_idx in range(psd.shape[0]):
        band = np.asarray(psd[band_idx], dtype=np.float32)
        feats.update(_safe_patch_stats(band, f"psd_band{band_idx}"))
        if append_background_reference_features:
            band_center = _center_crop(band, int(center_size)).astype(np.float32, copy=False)
            band_outer = _center_crop(band, int(outer_size)).astype(np.float32, copy=False)
            band_ring = band_outer[ring_mask]
            feats[f"psd_band{band_idx}_center_minus_ring_mean"] = (
                float(band_center.mean() - band_ring.mean()) if band_ring.size else float(band_center.mean())
            )
            feats[f"psd_band{band_idx}_center_minus_ring_q90"] = (
                float(np.quantile(band_center, 0.90) - np.quantile(band_ring, 0.90))
                if band_ring.size
                else float(np.quantile(band_center, 0.90))
            )
    if append_anisotropic_fourier_features:
        feats.update(_fft_anisotropy_features(center, "fft_orient"))
    if append_real_space_handcrafted_features and append_gradient_polarity_features:
        feats.update(_gradient_polarity_features(center, prefix="intensity_ramp", highpass_reference=high15))
    if append_bragg_fourier_features:
        feats.update(_fft_bragg_features(center, "fft_bragg"))
    return feats

def _compute_handcrafted_patch_features_batch(
    patches: np.ndarray,
    *,
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
    center_size: int,
    append_real_space_handcrafted_features: bool,
    append_psd_band_features: bool,
    append_anisotropic_fourier_features: bool,
    append_background_reference_features: bool,
    append_gradient_polarity_features: bool,
    append_bragg_fourier_features: bool,
    workers: int = 1,
) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float32)

    def _one(idx: int) -> list[float]:
        feats = _compute_single_handcrafted_feature_dict(
            patches[idx],
            pixel_size_angstrom=float(pixel_size_angstrom),
            frequency_bands=frequency_bands,
            center_size=int(center_size),
            append_real_space_handcrafted_features=bool(append_real_space_handcrafted_features),
            append_psd_band_features=bool(append_psd_band_features),
            append_anisotropic_fourier_features=bool(append_anisotropic_fourier_features),
            append_background_reference_features=bool(append_background_reference_features),
            append_gradient_polarity_features=bool(append_gradient_polarity_features),
            append_bragg_fourier_features=bool(append_bragg_fourier_features),
        )
        return [float(v) for v in feats.values()]

    # Each patch's handcrafted features (FFT/gradient/structure-tensor based) are fully
    # independent of every other patch - this is embarrassingly parallel. numpy/scipy
    # release the GIL during their C-level FFT/array compute, so a thread pool gives
    # real wall-clock parallelism here without inter-process serialization overhead.
    # executor.map preserves input order, so results still line up 1:1 with `patches`.
    if int(workers) <= 1 or patches.shape[0] <= 1:
        rows = [_one(idx) for idx in range(patches.shape[0])]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            rows = list(pool.map(_one, range(patches.shape[0])))
    return np.asarray(rows, dtype=np.float32)

def _run_probe_on_centers_chunk(
    image_norm: np.ndarray,
    chunk_centers: Sequence[tuple[int, int]],
    *,
    scale: "ScaleSpec",
    probe: Any,
    device: torch.device,
    encoder_input_mode: str,
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Run the probe forward pass for one (scale, chunk-of-centers) batch.

    On GPU OOM, bisects the chunk and retries the two halves instead of
    propagating the failure - since the embedding is computed per-sample
    (GroupNorm, no cross-batch statistics), the result is identical to what
    a single successful call at this chunk size would have produced. This
    keeps an OOM local to the batch that hit it, instead of losing every
    already-computed center in the image and permanently shrinking the
    batch size for the rest of the run (see PublicationClassifier.predict_prepared).
    """
    patches = [
        _extract_centered_patch(image_norm, cy, cx, int(scale.patch_size), pad_mode="reflect").astype(
            np.float32, copy=False
        )
        for cy, cx in chunk_centers
    ]
    batch_np = np.stack(patches, axis=0).astype(np.float32, copy=False)
    inputs = _build_input_batch(
        batch_np,
        pixel_size_angstrom=float(pixel_size_angstrom),
        frequency_bands=frequency_bands,
        input_mode=str(encoder_input_mode),
        input_channels=probe.input_channels,
        use_power_spectrum=probe.use_power_spectrum,
        include_real_space_input=probe.include_real_space_input,
    )
    try:
        tensor = torch.from_numpy(inputs).to(device, non_blocking=True)
        with torch.inference_mode():
            features, _contam_logits = probe(tensor)
        return _embedding_from_feature_map(features, center_size=int(scale.embedding_center_size))
    except torch.cuda.OutOfMemoryError:
        if len(chunk_centers) <= 1:
            raise
        torch.cuda.empty_cache()
        mid = len(chunk_centers) // 2
        first = _run_probe_on_centers_chunk(
            image_norm, chunk_centers[:mid], scale=scale, probe=probe, device=device,
            encoder_input_mode=encoder_input_mode, pixel_size_angstrom=pixel_size_angstrom,
            frequency_bands=frequency_bands,
        )
        second = _run_probe_on_centers_chunk(
            image_norm, chunk_centers[mid:], scale=scale, probe=probe, device=device,
            encoder_input_mode=encoder_input_mode, pixel_size_angstrom=pixel_size_angstrom,
            frequency_bands=frequency_bands,
        )
        return np.concatenate([first, second], axis=0)


def _compute_multiscale_embeddings_for_centers(
    image_norm: np.ndarray,
    centers: Sequence[tuple[int, int]],
    *,
    scales: Sequence[ScaleSpec],
    probe: Any,
    device: torch.device,
    encoder_input_mode: str,
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(centers), batch_size):
        batch_centers = centers[start : start + batch_size]
        emb_parts = [
            _run_probe_on_centers_chunk(
                image_norm, batch_centers, scale=scale, probe=probe, device=device,
                encoder_input_mode=encoder_input_mode, pixel_size_angstrom=pixel_size_angstrom,
                frequency_bands=frequency_bands,
            )
            for scale in scales
        ]
        out.append(np.concatenate(emb_parts, axis=1).astype(np.float32, copy=False))
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)

def _compute_feature_vectors_for_centers(
    image_norm: np.ndarray,
    centers: Sequence[tuple[int, int]],
    *,
    scales: Sequence[ScaleSpec],
    probe: Any,
    device: torch.device,
    encoder_input_mode: str,
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
    batch_size: int,
    append_handcrafted_features: bool,
    handcrafted_center_size: int,
    append_real_space_handcrafted_features: bool,
    append_psd_band_features: bool,
    append_geometry_features: bool,
    append_anisotropic_fourier_features: bool,
    append_background_reference_features: bool,
    append_edgecorner_geometry_features: bool,
    append_gradient_polarity_features: bool,
    append_bragg_fourier_features: bool,
    contamination_mask: np.ndarray | None = None,
    handcrafted_workers: int = 1,
    _timer: "_PhaseTimer | None" = None,
    _timer_prefix: str = "",
) -> np.ndarray:
    emb = _compute_multiscale_embeddings_for_centers(
        image_norm,
        centers,
        scales=scales,
        probe=probe,
        device=device,
        encoder_input_mode=encoder_input_mode,
        pixel_size_angstrom=pixel_size_angstrom,
        frequency_bands=frequency_bands,
        batch_size=batch_size,
    )
    if _timer is not None:
        _timer.mark(f"{_timer_prefix}cnn_embeddings")
    if not append_handcrafted_features:
        out = emb
    else:
        largest_patch = max(int(scale.patch_size) for scale in scales)
        handcrafted_parts: list[np.ndarray] = []
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start : start + batch_size]
            patches = [
                _extract_centered_patch(image_norm, cy, cx, largest_patch, pad_mode="reflect").astype(np.float32, copy=False)
                for cy, cx in batch_centers
            ]
            batch_np = np.stack(patches, axis=0).astype(np.float32, copy=False)
            handcrafted_parts.append(
                _compute_handcrafted_patch_features_batch(
                    batch_np,
                    pixel_size_angstrom=float(pixel_size_angstrom),
                    frequency_bands=frequency_bands,
                    center_size=int(handcrafted_center_size),
                    append_real_space_handcrafted_features=bool(append_real_space_handcrafted_features),
                    append_psd_band_features=bool(append_psd_band_features),
                    append_anisotropic_fourier_features=bool(append_anisotropic_fourier_features),
                    append_background_reference_features=bool(append_background_reference_features),
                    append_gradient_polarity_features=bool(append_gradient_polarity_features),
                    append_bragg_fourier_features=bool(append_bragg_fourier_features),
                    workers=int(handcrafted_workers),
                )
            )
        handcrafted = np.concatenate(handcrafted_parts, axis=0).astype(np.float32, copy=False)
        out = np.concatenate([emb, handcrafted], axis=1).astype(np.float32, copy=False)

    if append_geometry_features and contamination_mask is not None:
        geom = _compute_center_geometry_feature_rows(contamination_mask, centers)
        out = np.concatenate([out, geom], axis=1).astype(np.float32, copy=False)
    if append_edgecorner_geometry_features and contamination_mask is not None:
        geom_extra = _compute_center_edgecorner_geometry_feature_rows(contamination_mask, centers)
        out = np.concatenate([out, geom_extra], axis=1).astype(np.float32, copy=False)
    return out

def _sample_grid_centers_in_component(
    local_mask: np.ndarray,
    stride: int,
    *,
    offset_y: int = 0,
    offset_x: int = 0,
) -> list[tuple[int, int]]:
    h, w = local_mask.shape
    if not np.any(local_mask):
        return []
    dist = ndi.distance_transform_edt(local_mask)
    centers: list[tuple[int, int]] = []
    step = max(1, int(stride))
    y_start = -((step - (int(offset_y) % step)) % step)
    x_start = -((step - (int(offset_x) % step)) % step)
    for y0 in range(y_start, h, step):
        ys = max(0, y0)
        ye = min(h, y0 + step)
        if ys >= ye:
            continue
        for x0 in range(x_start, w, step):
            xs = max(0, x0)
            xe = min(w, x0 + step)
            if xs >= xe:
                continue
            cell_mask = np.asarray(local_mask[ys:ye, xs:xe], dtype=bool)
            if not np.any(cell_mask):
                continue
            cell_dist = np.asarray(dist[ys:ye, xs:xe], dtype=np.float32)
            coords = np.argwhere(cell_mask)
            values = cell_dist[cell_mask]
            best = coords[int(np.argmax(values))]
            centers.append((ys + int(best[0]), xs + int(best[1])))
    if centers:
        return centers
    best_global = np.argwhere(local_mask)
    values = dist[local_mask]
    best = best_global[int(np.argmax(values))]
    return [(int(best[0]), int(best[1]))]

def predict_prepared(
    image_norm: np.ndarray,
    pred_mask: np.ndarray,
    *,
    probe: Any,
    patch_model: Any,
    rescue_model: Any | None,
    target_pixel_size_angstrom: float,
    scales: Sequence[ScaleSpec],
    rescue_scales: Sequence[ScaleSpec],
    encoder_input_mode: str,
    sample_stride_px: int,
    grid_offsets_px: Sequence[tuple[int, int]],
    gaussian_sigma_px: float,
    component_constrained: bool,
    class_sigma_px: dict[str, float],
    intensity_aware_beta: float,
    intensity_aware_smooth_sigma_px: float,
    aggregate_stamp_enabled: bool,
    aggregate_stamp_threshold: float,
    aggregate_stamp_radius_px: int,
    aggregate_stamp_sigma_px: float,
    aggregate_stamp_strength: float,
    aggregate_stamp_require_top1: bool,
    soft_eval_prob_power: float,
    class_prob_weights: dict[str, float] | None,
    component_postprocess_enabled: bool,
    component_aggregate_edge_to_carbon_enabled: bool,
    component_aggregate_edge_max_border_px: float,
    component_aggregate_edge_min_area_px: int,
    component_aggregate_edge_min_aspect: float,
    component_aggregate_edge_carbon_dilation_px: int,
    component_carbon_interior_to_aggregate_enabled: bool,
    component_carbon_interior_min_border_px: float,
    component_carbon_interior_min_area_px: int,
    component_carbon_interior_aggregate_dilation_px: int,
    rescue_handcrafted_center_size: int,
    rescue_margin_threshold: float,
    rescue_aggregate_min_prob: float,
    rescue_blend: float,
    rescue_require_aggregate_top2: bool,
    rescue_require_rescue_aggregate_top1: bool,
    rescue_min_boundary_distance_px: float,
    rescue_boundary_alpha_scale_px: float,
    carbon_edge_rescue_enabled: bool,
    carbon_edge_max_border_distance_px: float,
    carbon_edge_min_carbon_prob: float,
    carbon_edge_max_crystalline_margin: float,
    carbon_edge_boost: float,
    corner_carbon_rescue_enabled: bool,
    corner_rescue_max_border_distance_px: float,
    corner_rescue_max_corner_distance_px: float,
    corner_rescue_min_carbon_prob: float,
    corner_rescue_max_crystalline_margin: float,
    corner_rescue_boost: float,
    pair_refine_model: Any | None,
    pair_refine_min_total_prob: float,
    pair_refine_require_top2_pair: bool,
    pair_expert_models: Sequence[tuple[PairExpertSpec, Any]],
    device: torch.device,
    batch_size: int,
    frequency_bands: tuple[tuple[float, float], ...],
    append_handcrafted_features: bool,
    handcrafted_center_size: int,
    append_real_space_handcrafted_features: bool,
    append_psd_band_features: bool,
    append_geometry_features: bool,
    append_anisotropic_fourier_features: bool,
    append_background_reference_features: bool,
    append_edgecorner_geometry_features: bool,
    append_gradient_polarity_features: bool,
    append_bragg_fourier_features: bool,
    handcrafted_workers: int = 1,
) -> dict[str, Any]:
    _timer = _PhaseTimer(_PROFILE_TYPING)
    classes = _classifier_classes(patch_model)
    class_to_global = {int(type_id): idx for idx, type_id in enumerate(TYPE_ID_SEQUENCE)}

    pair_expert_soft_apply_counts: dict[str, int] = {}
    component_postprocess_apply_counts: dict[str, int] = {}

    class_sigmas = _class_sigma_vector(float(gaussian_sigma_px), class_sigma_px)
    class_weight_vec = _class_prob_weight_vector(class_prob_weights or {})
    agg_type_idx = next((idx for idx, t in enumerate(TYPE_ID_SEQUENCE) if int(t) == 3), None)
    carbon_type_idx = next((idx for idx, t in enumerate(TYPE_ID_SEQUENCE) if int(t) == 1), None)
    crystalline_type_idx = next((idx for idx, t in enumerate(TYPE_ID_SEQUENCE) if int(t) == 2), None)
    radius_sigma = max(
        float(np.max(class_sigmas)) if class_sigmas.size else 0.0,
        float(aggregate_stamp_sigma_px) if aggregate_stamp_enabled else 0.0,
    )
    radius = max(1, int(math.ceil(3.0 * max(radius_sigma, 1e-3))))
    stamp_radius = max(0, int(aggregate_stamp_radius_px))
    kernel_axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_dist2 = np.square(kernel_axis[:, None]) + np.square(kernel_axis[None, :])
    gaussian_kernels = tuple(
        np.exp(-0.5 * (kernel_dist2 / max(float(sigma_px) ** 2, 1e-6))).astype(np.float32, copy=False)
        for sigma_px in class_sigmas.tolist()
    )
    stamp_kernel = None
    if stamp_radius > 0:
        stamp_axis = np.arange(-stamp_radius, stamp_radius + 1, dtype=np.float32)
        stamp_dist2 = np.square(stamp_axis[:, None]) + np.square(stamp_axis[None, :])
        stamp_kernel = np.exp(
            -0.5 * (stamp_dist2 / max(float(aggregate_stamp_sigma_px) ** 2, 1e-6))
        ).astype(np.float32, copy=False)

    smooth_image = None
    if float(intensity_aware_beta) > 0.0:
        smooth_image = ndi.gaussian_filter(
            image_norm.astype(np.float32, copy=False),
            sigma=float(intensity_aware_smooth_sigma_px),
        ).astype(np.float32, copy=False)
    component_map, num_components = ndi.label(pred_mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndi.find_objects(component_map)
    component_touches_border: dict[int, bool] = {}

    centers: list[tuple[int, int]] = []
    center_component_ids: list[int] = []
    center_boundary_distances: list[float] = []
    for component_id, component_slice in enumerate(objects, start=1):
        if component_slice is None:
            continue
        local_mask = component_map[component_slice] == component_id
        if not np.any(local_mask):
            continue
        local_dist = ndi.distance_transform_edt(local_mask).astype(np.float32, copy=False)
        y0 = int(component_slice[0].start or 0)
        x0 = int(component_slice[1].start or 0)
        y1 = int(component_slice[0].stop or 0)
        x1 = int(component_slice[1].stop or 0)
        component_touches_border[int(component_id)] = bool(y0 <= 0 or x0 <= 0 or y1 >= pred_mask.shape[0] or x1 >= pred_mask.shape[1])
        local_centers: list[tuple[int, int]] = []
        seen_local: set[tuple[int, int]] = set()
        for offset_y, offset_x in grid_offsets_px:
            for cy_local, cx_local in _sample_grid_centers_in_component(
                local_mask,
                stride=int(sample_stride_px),
                offset_y=int(offset_y),
                offset_x=int(offset_x),
            ):
                coord = (int(cy_local), int(cx_local))
                if coord in seen_local:
                    continue
                seen_local.add(coord)
                local_centers.append(coord)
        for cy_local, cx_local in local_centers:
            centers.append((y0 + int(cy_local), x0 + int(cx_local)))
            center_component_ids.append(int(component_id))
            center_boundary_distances.append(float(local_dist[int(cy_local), int(cx_local)]))
    if not centers:
        return {"pred_map": np.zeros(pred_mask.shape, dtype=np.uint8), "top1_prob": np.zeros(pred_mask.shape, dtype=np.float16), "top1_margin": np.zeros(pred_mask.shape, dtype=np.float16), "centers": 0}
    center_boundary_dist_np = np.asarray(center_boundary_distances, dtype=np.float32)
    _timer.mark("grid_sampling")

    feature_vectors = _compute_feature_vectors_for_centers(
        image_norm,
        centers,
        scales=scales,
        probe=probe,
        device=device,
        encoder_input_mode=encoder_input_mode,
        pixel_size_angstrom=target_pixel_size_angstrom,
        frequency_bands=frequency_bands,
        batch_size=batch_size,
        append_handcrafted_features=append_handcrafted_features,
        handcrafted_center_size=handcrafted_center_size,
        append_real_space_handcrafted_features=append_real_space_handcrafted_features,
        append_psd_band_features=append_psd_band_features,
        append_geometry_features=append_geometry_features,
        append_anisotropic_fourier_features=append_anisotropic_fourier_features,
        append_background_reference_features=append_background_reference_features,
        append_edgecorner_geometry_features=append_edgecorner_geometry_features,
        append_gradient_polarity_features=append_gradient_polarity_features,
        append_bragg_fourier_features=append_bragg_fourier_features,
        contamination_mask=pred_mask,
        handcrafted_workers=handcrafted_workers,
        _timer=_timer,
    )
    _timer.mark("handcrafted_features")
    probs_local = patch_model.predict_proba(feature_vectors)
    probs_global = np.zeros((probs_local.shape[0], len(TYPE_ID_SEQUENCE)), dtype=np.float32)
    for local_idx, class_id in enumerate(classes.tolist()):
        if int(class_id) in class_to_global:
            probs_global[:, class_to_global[int(class_id)]] = probs_local[:, local_idx]
    if pair_refine_model is not None:
        agg_idx = class_to_global.get(3)
        eth_idx = class_to_global.get(4)
        if agg_idx is not None and eth_idx is not None:
            probs_global = _apply_pair_refine_probs(
                probs_global,
                feature_vectors=feature_vectors,
                pair_refine_model=pair_refine_model,
                agg_idx=int(agg_idx),
                eth_idx=int(eth_idx),
                min_total_prob=float(pair_refine_min_total_prob),
                require_top2_pair=bool(pair_refine_require_top2_pair),
            )
    if pair_expert_models:
        probs_global, image_apply_counts = _apply_pair_expert_stack_soft(
            probs_global,
            feature_vectors=feature_vectors,
            pair_expert_models=pair_expert_models,
            class_to_global=class_to_global,
            centers=centers,
            image_shape=pred_mask.shape,
            component_ids=center_component_ids,
            component_touches_border=component_touches_border,
        )
        for name, count in image_apply_counts.items():
            pair_expert_soft_apply_counts[name] = pair_expert_soft_apply_counts.get(name, 0) + int(count)
    probs_global = _apply_carbon_border_corner_rescue(
        probs_global,
        centers=centers,
        image_shape=pred_mask.shape,
        component_ids=center_component_ids,
        component_touches_border=component_touches_border,
        carbon_idx=carbon_type_idx,
        crystalline_idx=crystalline_type_idx,
        edge_enabled=bool(carbon_edge_rescue_enabled),
        edge_max_border_distance_px=float(carbon_edge_max_border_distance_px),
        edge_min_carbon_prob=float(carbon_edge_min_carbon_prob),
        edge_max_crystalline_margin=float(carbon_edge_max_crystalline_margin),
        edge_boost=float(carbon_edge_boost),
        corner_enabled=bool(corner_carbon_rescue_enabled),
        corner_max_border_distance_px=float(corner_rescue_max_border_distance_px),
        corner_max_corner_distance_px=float(corner_rescue_max_corner_distance_px),
        corner_min_carbon_prob=float(corner_rescue_min_carbon_prob),
        corner_max_crystalline_margin=float(corner_rescue_max_crystalline_margin),
        corner_boost=float(corner_rescue_boost),
    )
    _timer.mark("classify_and_border_rescue")
    n_rescue_centers = 0
    if (
        rescue_model is not None
        and rescue_scales
        and agg_type_idx is not None
        and float(rescue_blend) > 0.0
        and float(rescue_margin_threshold) > 0.0
    ):
        sorted_probs = np.sort(probs_global, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        rescue_mask = margin <= float(rescue_margin_threshold)
        rescue_mask &= probs_global[:, int(agg_type_idx)] >= float(rescue_aggregate_min_prob)
        if float(rescue_min_boundary_distance_px) > 0.0:
            rescue_mask &= center_boundary_dist_np >= float(rescue_min_boundary_distance_px)
        if rescue_require_aggregate_top2:
            rescue_mask &= _top2_contains(probs_global, int(agg_type_idx))
        rescue_indices = np.flatnonzero(rescue_mask)
        n_rescue_centers = int(rescue_indices.size)
        if rescue_indices.size > 0:
            rescue_centers = [centers[int(idx)] for idx in rescue_indices.tolist()]
            rescue_features = _compute_feature_vectors_for_centers(
                image_norm,
                rescue_centers,
                scales=rescue_scales,
                probe=probe,
                device=device,
                encoder_input_mode=encoder_input_mode,
                pixel_size_angstrom=target_pixel_size_angstrom,
                frequency_bands=frequency_bands,
                batch_size=batch_size,
                append_handcrafted_features=append_handcrafted_features,
                handcrafted_center_size=rescue_handcrafted_center_size,
                append_real_space_handcrafted_features=append_real_space_handcrafted_features,
                append_psd_band_features=append_psd_band_features,
                append_geometry_features=append_geometry_features,
                append_anisotropic_fourier_features=append_anisotropic_fourier_features,
                append_background_reference_features=append_background_reference_features,
                append_edgecorner_geometry_features=append_edgecorner_geometry_features,
                append_gradient_polarity_features=append_gradient_polarity_features,
                append_bragg_fourier_features=append_bragg_fourier_features,
                contamination_mask=pred_mask,
                handcrafted_workers=handcrafted_workers,
                _timer=_timer,
                _timer_prefix="rescue_",
            )
            _timer.mark("rescue_handcrafted_features")
            rescue_local = rescue_model.predict_proba(rescue_features)
            rescue_classes = _classifier_classes(rescue_model)
            rescue_global = np.zeros((rescue_local.shape[0], len(TYPE_ID_SEQUENCE)), dtype=np.float32)
            for local_idx, class_id in enumerate(rescue_classes.tolist()):
                if int(class_id) in class_to_global:
                    rescue_global[:, class_to_global[int(class_id)]] = rescue_local[:, local_idx]
            if rescue_require_rescue_aggregate_top1:
                apply_mask = np.argmax(rescue_global, axis=1) == int(agg_type_idx)
            else:
                apply_mask = rescue_global[:, int(agg_type_idx)] > probs_global[rescue_indices, int(agg_type_idx)]
            if np.any(apply_mask):
                alpha = float(np.clip(rescue_blend, 0.0, 1.0))
                idx_apply = rescue_indices[apply_mask]
                if float(rescue_boundary_alpha_scale_px) > 0.0:
                    alpha_vec = alpha * np.clip(
                        center_boundary_dist_np[idx_apply] / max(float(rescue_boundary_alpha_scale_px), 1e-6),
                        0.0,
                        1.0,
                    )
                    blended = (
                        (1.0 - alpha_vec[:, None]) * probs_global[idx_apply]
                        + alpha_vec[:, None] * rescue_global[apply_mask]
                    ).astype(np.float32, copy=False)
                else:
                    blended = (
                        (1.0 - alpha) * probs_global[idx_apply]
                        + alpha * rescue_global[apply_mask]
                    ).astype(np.float32, copy=False)
                probs_global[idx_apply] = _row_normalize_probs(blended)
    _timer.mark("rescue_classify_and_blend")
    probs_global = _apply_class_probability_weights(probs_global, class_weight_vec)
    probs_global = _apply_prob_power(probs_global, power=float(soft_eval_prob_power))

    h, w = pred_mask.shape
    prob_acc = np.zeros((len(TYPE_ID_SEQUENCE), h, w), dtype=np.float32)
    weight_acc = np.zeros((len(TYPE_ID_SEQUENCE), h, w), dtype=np.float32)
    component_prob_sum: dict[int, np.ndarray] = {}
    component_prob_count: dict[int, int] = {}

    for (cy, cx), component_id, prob in zip(centers, center_component_ids, probs_global, strict=True):
        component_prob_sum.setdefault(int(component_id), np.zeros((len(TYPE_ID_SEQUENCE),), dtype=np.float64))
        component_prob_count[int(component_id)] = component_prob_count.get(int(component_id), 0) + 1
        component_prob_sum[int(component_id)] += prob.astype(np.float64, copy=False)

        y0 = max(0, int(cy) - radius)
        y1 = min(h, int(cy) + radius + 1)
        x0 = max(0, int(cx) - radius)
        x1 = min(w, int(cx) + radius + 1)
        ky0 = y0 - (int(cy) - radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (int(cx) - radius)
        kx1 = kx0 + (x1 - x0)
        if component_constrained:
            valid = component_map[y0:y1, x0:x1] == int(component_id)
        else:
            valid = pred_mask[y0:y1, x0:x1]
        valid_f = valid.astype(np.float32, copy=False)
        if not np.any(valid_f > 0):
            continue
        intensity_weight = None
        if smooth_image is not None:
            center_val = float(smooth_image[int(cy), int(cx)])
            delta = smooth_image[y0:y1, x0:x1].astype(np.float32, copy=False) - center_val
            intensity_weight = np.exp(
                -float(intensity_aware_beta) * np.square(delta, dtype=np.float32)
            ).astype(np.float32, copy=False)
        for class_idx, sigma_px in enumerate(class_sigmas.tolist()):
            weight = gaussian_kernels[class_idx][ky0:ky1, kx0:kx1]
            if intensity_weight is not None:
                weight = weight * intensity_weight
            weight = weight * valid_f
            if not np.any(weight > 0):
                continue
            prob_acc[class_idx, y0:y1, x0:x1] += prob[class_idx] * weight
            weight_acc[class_idx, y0:y1, x0:x1] += weight

        if (
            aggregate_stamp_enabled
            and agg_type_idx is not None
            and stamp_radius > 0
            and float(aggregate_stamp_strength) > 0.0
            and prob[agg_type_idx] >= float(aggregate_stamp_threshold)
            and (not aggregate_stamp_require_top1 or int(np.argmax(prob)) == int(agg_type_idx))
        ):
            sy0 = max(0, int(cy) - stamp_radius)
            sy1 = min(h, int(cy) + stamp_radius + 1)
            sx0 = max(0, int(cx) - stamp_radius)
            sx1 = min(w, int(cx) + stamp_radius + 1)
            sky0 = sy0 - (int(cy) - stamp_radius)
            sky1 = sky0 + (sy1 - sy0)
            skx0 = sx0 - (int(cx) - stamp_radius)
            skx1 = skx0 + (sx1 - sx0)
            assert stamp_kernel is not None
            stamp_weight = stamp_kernel[sky0:sky1, skx0:skx1]
            if component_constrained:
                stamp_valid = component_map[sy0:sy1, sx0:sx1] == int(component_id)
            else:
                stamp_valid = pred_mask[sy0:sy1, sx0:sx1]
            stamp_weight = stamp_weight * stamp_valid.astype(np.float32, copy=False)
            if np.any(stamp_weight > 0):
                prob_acc[int(agg_type_idx), sy0:sy1, sx0:sx1] += (
                    float(aggregate_stamp_strength) * float(prob[agg_type_idx]) * stamp_weight
                )

    zero_weight_mask = pred_mask & (~np.any(weight_acc > 0, axis=0))
    if np.any(zero_weight_mask):
        for component_id in range(1, int(num_components) + 1):
            comp_pixels = component_map == component_id
            missing = comp_pixels & zero_weight_mask
            if not np.any(missing):
                continue
            mean_prob = component_prob_sum.get(int(component_id))
            count = component_prob_count.get(int(component_id), 0)
            if mean_prob is None or count <= 0:
                continue
            prob_acc[:, missing] = (mean_prob / float(count))[:, None]
            weight_acc[:, missing] = 1.0

    pred_map = np.zeros(pred_mask.shape, dtype=np.uint8)
    valid_pred = pred_mask & np.any(weight_acc > 0, axis=0)
    if np.any(valid_pred):
        prob_map = np.zeros_like(prob_acc, dtype=np.float32)
        positive = weight_acc > 0
        prob_map[positive] = prob_acc[positive] / weight_acc[positive]
        pred_idx = np.argmax(prob_map, axis=0).astype(np.int64)
        pred_types = np.asarray(TYPE_ID_SEQUENCE, dtype=np.uint8)[pred_idx]
        pred_map[valid_pred] = pred_types[valid_pred]
    _timer.mark("splatting")

    pred_map, image_component_counts = _apply_component_geometry_postprocess(
        pred_map,
        pred_mask,
        enabled=bool(component_postprocess_enabled),
        aggregate_edge_to_carbon_enabled=bool(component_aggregate_edge_to_carbon_enabled),
        aggregate_edge_max_border_px=float(component_aggregate_edge_max_border_px),
        aggregate_edge_min_area_px=int(component_aggregate_edge_min_area_px),
        aggregate_edge_min_aspect=float(component_aggregate_edge_min_aspect),
        aggregate_edge_carbon_dilation_px=int(component_aggregate_edge_carbon_dilation_px),
        carbon_interior_to_aggregate_enabled=bool(component_carbon_interior_to_aggregate_enabled),
        carbon_interior_min_border_px=float(component_carbon_interior_min_border_px),
        carbon_interior_min_area_px=int(component_carbon_interior_min_area_px),
        carbon_interior_aggregate_dilation_px=int(component_carbon_interior_aggregate_dilation_px),
    )
    for name, count in image_component_counts.items():
        component_postprocess_apply_counts[name] = component_postprocess_apply_counts.get(name, 0) + int(count)
    _timer.mark("postprocess")

    top1_prob = np.zeros((h, w), dtype=np.float16)
    top1_margin = np.zeros((h, w), dtype=np.float16)
    if np.any(valid_pred):
        top2 = np.partition(prob_map, kth=-2, axis=0)
        top = top2[-1]
        second = top2[-2]
        top1_prob[valid_pred] = top[valid_pred].astype(np.float16)
        top1_margin[valid_pred] = (top - second)[valid_pred].astype(np.float16)
    _timer.report(n_centers=len(centers), n_rescue_centers=n_rescue_centers)
    return {"pred_map": pred_map, "top1_prob": top1_prob, "top1_margin": top1_margin, "centers": len(centers)}
