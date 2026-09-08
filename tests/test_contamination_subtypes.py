from __future__ import annotations

import numpy as np

from utils.contamination_subtypes import (
    TYPE_SCHEMA_NAME,
    TYPE_SCHEMA_VERSION,
    UNLABELED_CONTAMINATION_LABEL,
    build_unlabeled_typed_mask,
    reconcile_typed_mask,
    typed_class_pixel_counts,
    typed_completion_stats,
)


def test_schema_metadata_is_stable() -> None:
    assert TYPE_SCHEMA_NAME == "cryofilter_contamination_subtypes"
    assert TYPE_SCHEMA_VERSION == 1


def test_build_unlabeled_typed_mask_marks_gt_pixels() -> None:
    binary = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    typed = build_unlabeled_typed_mask(binary)
    assert typed.dtype == np.uint8
    assert typed.tolist() == [[0, UNLABELED_CONTAMINATION_LABEL], [UNLABELED_CONTAMINATION_LABEL, 0]]


def test_reconcile_typed_mask_clears_outside_and_restores_unlabeled_inside() -> None:
    binary = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    typed = np.array([[3, 0], [9, 4]], dtype=np.uint8)
    reconciled = reconcile_typed_mask(binary, typed)
    assert reconciled.tolist() == [[0, UNLABELED_CONTAMINATION_LABEL], [UNLABELED_CONTAMINATION_LABEL, 0]]


def test_completion_and_counts_treat_zero_inside_gt_as_unlabeled() -> None:
    binary = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    typed = np.array(
        [
            [0, 1, 0],
            [0, UNLABELED_CONTAMINATION_LABEL, 0],
        ],
        dtype=np.uint8,
    )
    stats = typed_completion_stats(binary, typed)
    counts = typed_class_pixel_counts(binary, typed)

    assert stats["gt_contamination_px"] == 3
    assert stats["typed_labeled_px"] == 1
    assert stats["typed_unlabeled_px"] == 2
    assert abs(stats["typed_completion_fraction"] - (1.0 / 3.0)) < 1e-8
    assert counts["background_px"] == 3
    assert counts["carbon_px"] == 1
    assert counts["unlabeled_contamination_px"] == 2
