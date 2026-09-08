"""Shared schema/helpers for typed contamination sub-annotations."""

from __future__ import annotations

from typing import Dict

import numpy as np


TYPE_SCHEMA_NAME = "cryofilter_contamination_subtypes"
TYPE_SCHEMA_VERSION = 1

BACKGROUND_LABEL = 0
UNLABELED_CONTAMINATION_LABEL = 255
TYPE_ID_SEQUENCE = (1, 2, 3, 4)

TYPE_ID_TO_KEY = {
    1: "carbon",
    2: "crystalline",
    3: "aggregate",
    4: "ethane",
}

TYPE_ID_TO_DISPLAY = {
    1: "Carbon",
    2: "Crystalline",
    3: "Aggregate",
    4: "Ethane",
}

TYPE_ID_TO_COLOR_HEX = {
    1: "#D95F02",
    2: "#1B9E77",
    3: "#CC79A7",
    4: "#E6AB02",
}

TYPE_MANIFEST_COLUMNS = (
    "typed_label_path",
    "typed_label_metadata_path",
    "typed_label_schema_name",
    "typed_label_schema_version",
    "typed_label_unlabeled_value",
)


def build_unlabeled_typed_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Create a typed label map from a binary GT mask.

    Values:
      - 0: background / clean
      - 1..4: typed contamination classes
      - 255: contamination present in GT, but not yet subtype-labeled
    """
    binary = np.asarray(binary_mask) > 0.5
    typed = np.zeros(binary.shape, dtype=np.uint8)
    typed[binary] = np.uint8(UNLABELED_CONTAMINATION_LABEL)
    return typed


def reconcile_typed_mask(binary_mask: np.ndarray, typed_mask: np.ndarray) -> np.ndarray:
    """Force a typed label map to stay consistent with the binary GT mask."""
    binary = np.asarray(binary_mask) > 0.5
    typed = np.asarray(typed_mask, dtype=np.uint8)
    if typed.shape != binary.shape:
        raise ValueError(
            f"typed mask shape {typed.shape} does not match binary mask shape {binary.shape}"
        )

    out = typed.copy()
    out[~binary] = np.uint8(BACKGROUND_LABEL)

    valid_ids = np.array(
        [BACKGROUND_LABEL, UNLABELED_CONTAMINATION_LABEL, *TYPE_ID_SEQUENCE],
        dtype=np.uint8,
    )
    valid_mask = np.isin(out, valid_ids)
    needs_unlabeled = binary & ((out == BACKGROUND_LABEL) | (~valid_mask))
    out[needs_unlabeled] = np.uint8(UNLABELED_CONTAMINATION_LABEL)
    return out


def typed_completion_stats(binary_mask: np.ndarray, typed_mask: np.ndarray) -> Dict[str, float]:
    """Summarize how much of the contamination mask has a concrete subtype label."""
    binary = np.asarray(binary_mask) > 0.5
    typed = reconcile_typed_mask(binary, typed_mask)
    total_gt = int(binary.sum())
    unlabeled_gt = int(np.sum(typed[binary] == np.uint8(UNLABELED_CONTAMINATION_LABEL)))
    labeled_gt = int(total_gt - unlabeled_gt)
    completion = float(labeled_gt / total_gt) if total_gt > 0 else 1.0
    return {
        "gt_contamination_px": float(total_gt),
        "typed_labeled_px": float(labeled_gt),
        "typed_unlabeled_px": float(unlabeled_gt),
        "typed_completion_fraction": float(completion),
    }


def typed_class_pixel_counts(binary_mask: np.ndarray, typed_mask: np.ndarray) -> Dict[str, int]:
    """Count pixels for each schema value after reconciling to the GT mask."""
    binary = np.asarray(binary_mask) > 0.5
    typed = reconcile_typed_mask(binary, typed_mask)
    counts: Dict[str, int] = {
        "background_px": int(np.sum(typed == np.uint8(BACKGROUND_LABEL))),
        "unlabeled_contamination_px": int(
            np.sum(typed == np.uint8(UNLABELED_CONTAMINATION_LABEL))
        ),
    }
    for type_id in TYPE_ID_SEQUENCE:
        counts[f"{TYPE_ID_TO_KEY[type_id]}_px"] = int(np.sum(typed == np.uint8(type_id)))
    return counts


__all__ = [
    "BACKGROUND_LABEL",
    "TYPE_ID_SEQUENCE",
    "TYPE_ID_TO_COLOR_HEX",
    "TYPE_ID_TO_DISPLAY",
    "TYPE_ID_TO_KEY",
    "TYPE_MANIFEST_COLUMNS",
    "TYPE_SCHEMA_NAME",
    "TYPE_SCHEMA_VERSION",
    "UNLABELED_CONTAMINATION_LABEL",
    "build_unlabeled_typed_mask",
    "reconcile_typed_mask",
    "typed_class_pixel_counts",
    "typed_completion_stats",
]
