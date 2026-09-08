from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd

from cryofilter.cli import _build_parser
from cryofilter.gui import load_gui_manifest
from scripts.gt_mask_editor_modern import build_split_labels_from_manifest_column


def test_gui_parser_exposes_public_mask_review_options() -> None:
    args = _build_parser().parse_args(
        [
            "gui",
            "--manifest",
            "inference/contamination_typing_manifest.csv",
            "--output-dir",
            "reviewed",
        ]
    )

    assert args.command == "gui"
    assert args.output_dir == "reviewed"
    assert args.micrograph_dir is None
    assert args.max_display_dim == 1600


def test_gui_loads_inference_manifest_without_internal_columns(tmp_path: Path) -> None:
    micrograph = tmp_path / "mic_001.mrc"
    with mrcfile.new(micrograph, overwrite=True) as mrc:
        mrc.set_data(np.zeros((32, 48), dtype=np.float32))
    mask = tmp_path / "mic_001_mask.npy"
    np.save(mask, np.zeros((32, 48), dtype=np.uint8))
    manifest = tmp_path / "contamination_typing_manifest.csv"
    manifest.write_text(
        "dataset_id,stem,micrograph_path,binary_mask_path,pixel_size_angstrom\n"
        "grid_a,mic_001,mic_001.mrc,mic_001_mask.npy,1.0\n",
        encoding="utf-8",
    )

    resolved, rows, fieldnames, records = load_gui_manifest(manifest)

    assert resolved == manifest.resolve()
    assert len(rows) == 1
    assert "micrograph_path" in fieldnames
    assert records[0].dataset_id == "grid_a"
    assert records[0].stem == "mic_001"
    assert records[0].micrograph_path == micrograph.resolve()
    assert records[0].mask_path == mask.resolve()


def test_annotation_editor_accepts_manifest_row_split_column() -> None:
    labels, metadata = build_split_labels_from_manifest_column(
        pd.DataFrame({"split": ["train", "VALIDATION", "val"]})
    )

    assert labels == ["TRAINING", "VALIDATION", "VALIDATION"]
    assert metadata["split_source"] == "manifest_column"
