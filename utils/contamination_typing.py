"""Shared contamination-type heuristics used by evaluation and figure scripts.

The final-release rules keep the more selective edge/carbon behavior while
preserving the manuscript's broader component vocabulary:

- ``edge/carbon`` for strong border-dominant structures
- ``crystalline`` for components with high-frequency enrichment
- ``diffuse`` for irregular weak-fill interior contamination
- ``isolated`` for compact interior contamination
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import label as ndlabel


TYPE_ORDER = ("edge/carbon", "crystalline", "diffuse", "isolated")
NON_EDGE_TYPE_ORDER = ("crystalline", "diffuse", "isolated")

_CLEAN_FRACTION_THRESHOLD = 0.01
_LARGE_COMPONENT_MIN_AREA = 1000
_FRAGMENTED_LARGE_MIN_COMPONENT_AREA = 256
_EDGE_BAND_FRACTION = 0.10
_EDGE_MIN_COMPONENT_SHARE = 0.20
_EDGE_MIN_EDGE_FRACTION_IN_COMPONENT = 0.35
# Small discretization margin so obvious corner components with ~56/256 border
# span are not missed just for landing a hair under 0.22 after rasterization.
_EDGE_MIN_LONGEST_SIDE_SPAN = 0.218
_EDGE_MIN_ADJACENT_SIDE_SPAN = 0.12
_EDGE_MIN_CORNER_FRACTION = 0.05
_EDGE_MIN_PURE_SINGLE_SIDE_SPAN = 0.30
_EDGE_MIN_PURE_SINGLE_SIDE_EDGE_FRACTION = 0.90
_EDGE_MIN_VERY_LONG_SIDE_SPAN = 0.55
_EDGE_MIN_VERY_LONG_SIDE_EDGE_FRACTION = 0.50
_EDGE_MIN_COMPACT_CORNER_SPAN = 0.20
_EDGE_MIN_COMPACT_CORNER_FRACTION = 0.30
_EDGE_MIN_COMPACT_CORNER_EDGE_FRACTION = 0.95


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest contiguous True run in a 1D boolean array."""
    if values.size == 0:
        return 0
    padded = np.concatenate(([False], np.asarray(values, dtype=bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    if starts.size == 0:
        return 0
    return int(np.max(ends - starts))


def _component_edge_metrics(component_mask: np.ndarray, edge_px: int) -> tuple[float, float, float, float]:
    """Summarize how strongly a component looks like an edge/carbon structure."""
    h, w = component_mask.shape
    area = int(component_mask.sum())
    if area <= 0:
        return 0.0, 0.0, 0.0, 0.0

    edge_band = np.zeros_like(component_mask, dtype=bool)
    edge_band[:edge_px, :] = True
    edge_band[-edge_px:, :] = True
    edge_band[:, :edge_px] = True
    edge_band[:, -edge_px:] = True
    edge_fraction_in_component = float((component_mask & edge_band).sum() / float(area))

    top = np.any(component_mask[:edge_px, :], axis=0)
    bottom = np.any(component_mask[-edge_px:, :], axis=0)
    left = np.any(component_mask[:, :edge_px], axis=1)
    right = np.any(component_mask[:, -edge_px:], axis=1)

    top_run = _longest_true_run(top) / float(max(1, w))
    bottom_run = _longest_true_run(bottom) / float(max(1, w))
    left_run = _longest_true_run(left) / float(max(1, h))
    right_run = _longest_true_run(right) / float(max(1, h))

    side_runs = {
        "top": float(top_run),
        "bottom": float(bottom_run),
        "left": float(left_run),
        "right": float(right_run),
    }
    longest_side_span = max(side_runs.values())
    adjacent_side_span = max(
        min(side_runs["top"], side_runs["left"]),
        min(side_runs["top"], side_runs["right"]),
        min(side_runs["bottom"], side_runs["left"]),
        min(side_runs["bottom"], side_runs["right"]),
    )

    corner_px = max(1, min(2 * edge_px, min(h, w)))
    corner_fraction = max(
        float(component_mask[:corner_px, :corner_px].mean()),
        float(component_mask[:corner_px, -corner_px:].mean()),
        float(component_mask[-corner_px:, :corner_px].mean()),
        float(component_mask[-corner_px:, -corner_px:].mean()),
    )

    return edge_fraction_in_component, longest_side_span, adjacent_side_span, corner_fraction


def _is_edge_artifact_component(
    component_mask: np.ndarray,
    total_contam_pixels: int,
    edge_px: int,
) -> bool:
    """Edge/carbon detector for a single connected component.

    This stays stricter than the old "any border touch" rule, but it allows
    dominant single-edge carbon strips that do not show a perfect right-angle
    corner after cropping. It also keeps a compact corner path for beam-edge
    footprints that are visibly corner-anchored but land just under the
    longest-span cutoff after rasterization.
    """
    component_area = int(component_mask.sum())
    if component_area < _LARGE_COMPONENT_MIN_AREA or total_contam_pixels <= 0:
        return False

    component_share = float(component_area / float(total_contam_pixels))
    edge_fraction_in_component, longest_side_span, adjacent_side_span, corner_fraction = (
        _component_edge_metrics(component_mask, edge_px)
    )

    if component_share < _EDGE_MIN_COMPONENT_SHARE:
        return False
    if edge_fraction_in_component < _EDGE_MIN_EDGE_FRACTION_IN_COMPONENT:
        return False

    has_corner_like_geometry = (
        longest_side_span >= _EDGE_MIN_LONGEST_SIDE_SPAN
        and (
            adjacent_side_span >= _EDGE_MIN_ADJACENT_SIDE_SPAN
            or corner_fraction >= _EDGE_MIN_CORNER_FRACTION
        )
    )
    has_pure_single_side_strip = (
        longest_side_span >= _EDGE_MIN_PURE_SINGLE_SIDE_SPAN
        and edge_fraction_in_component >= _EDGE_MIN_PURE_SINGLE_SIDE_EDGE_FRACTION
    )
    has_extreme_single_edge_span = (
        longest_side_span >= _EDGE_MIN_VERY_LONG_SIDE_SPAN
        and edge_fraction_in_component >= _EDGE_MIN_VERY_LONG_SIDE_EDGE_FRACTION
    )
    has_compact_strong_corner = (
        longest_side_span >= _EDGE_MIN_COMPACT_CORNER_SPAN
        and corner_fraction >= _EDGE_MIN_COMPACT_CORNER_FRACTION
        and edge_fraction_in_component >= _EDGE_MIN_COMPACT_CORNER_EDGE_FRACTION
    )
    return bool(
        has_corner_like_geometry
        or has_pure_single_side_strip
        or has_extreme_single_edge_span
        or has_compact_strong_corner
    )


def _has_large_contamination_burden(component_sizes: np.ndarray) -> bool:
    """Return True when the image has a substantial non-edge contamination burden.

    The original release rule only upgraded to ``large_contam`` when one
    connected component crossed the size cutoff. That misses images where the
    contamination is clearly substantial but split across multiple medium-sized
    interior blobs. We keep the old single-component path and add a fragmented
    fallback that requires at least two non-trivial components, which prevents
    tiny speck noise from being relabeled as ``large_contam``.
    """

    sizes = np.asarray(component_sizes[1:], dtype=np.int64)
    if sizes.size == 0:
        return False

    max_size = int(sizes.max())
    if max_size > _LARGE_COMPONENT_MIN_AREA:
        return True

    fragmented = sizes[sizes >= _FRAGMENTED_LARGE_MIN_COMPONENT_AREA]
    return bool(fragmented.size >= 2 and int(fragmented.sum()) > _LARGE_COMPONENT_MIN_AREA)


def classify_non_edge_component(scores: Mapping[str, float]) -> tuple[str, float]:
    """Choose among crystalline, diffuse, and isolated using precomputed scores."""
    ordered = sorted(
        (
            (label, float(scores[label]))
            for label in NON_EDGE_TYPE_ORDER
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else best_score
    return str(best_label), float(best_score - second_score)


def assign_frequency_component_type(
    *,
    edge_artifact: bool,
    crystalline_score: float,
    diffuse_score: float,
    isolated_score: float,
) -> tuple[str, float]:
    """Assign a component to the final-release four-type vocabulary."""
    if edge_artifact:
        non_edge_label, non_edge_conf = classify_non_edge_component(
            {
                "crystalline": crystalline_score,
                "diffuse": diffuse_score,
                "isolated": isolated_score,
            }
        )
        return "edge/carbon", float(1.0 + non_edge_conf)
    return classify_non_edge_component(
        {
            "crystalline": crystalline_score,
            "diffuse": diffuse_score,
            "isolated": isolated_score,
        }
    )


def categorize_contamination_type(gt_mask: np.ndarray) -> str:
    """Categorize a contamination mask as clean, small, large, or edge-dominant."""
    gt_binary = np.asarray(gt_mask) > 0.5
    contam_fraction = float(gt_binary.mean())
    if contam_fraction < _CLEAN_FRACTION_THRESHOLD:
        return "clean"

    labeled, n_comp = ndlabel(gt_binary)
    if n_comp == 0:
        return "clean"

    component_sizes = np.bincount(labeled.ravel())
    if component_sizes.size <= 1:
        return "clean"

    total_contam_pixels = int(component_sizes[1:].sum())
    h, w = gt_binary.shape
    edge_px = max(1, int(_EDGE_BAND_FRACTION * min(h, w)))

    for comp_id in range(1, n_comp + 1):
        if _is_edge_artifact_component(labeled == comp_id, total_contam_pixels, edge_px):
            return "edge_artifact"

    if _has_large_contamination_burden(component_sizes):
        return "large_contam"
    return "small_contam"


__all__ = [
    "TYPE_ORDER",
    "NON_EDGE_TYPE_ORDER",
    "_component_edge_metrics",
    "_is_edge_artifact_component",
    "assign_frequency_component_type",
    "categorize_contamination_type",
    "classify_non_edge_component",
]
