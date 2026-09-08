from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cryofilter.app.training_manifest import prepare_training_manifest


def test_prepare_training_manifest_filters_saved_rows_and_assigns_split(tmp_path: Path) -> None:
    mic_dir = tmp_path / "micrographs"
    mask_dir = tmp_path / "annotation_exports" / "session" / "masks"
    mic_dir.mkdir()
    mask_dir.mkdir(parents=True)
    for index in range(4):
        (mic_dir / f"mic_{index}.mrc").write_bytes(b"mrc")
        np.save(mask_dir / f"mic_{index}.npy", np.ones((4, 4), dtype=np.uint8))
    manifest = tmp_path / "annotation_exports" / "session" / "manifest_edited.csv"
    manifest.write_text(
        "dataset_id,stem,micrograph_path,gt_mask_path,annotation_status\n"
        f"1,mic_0,{mic_dir / 'mic_0.mrc'},masks/mic_0.npy,saved\n"
        f"1,mic_1,{mic_dir / 'mic_1.mrc'},masks/mic_1.npy,saved\n"
        f"1,mic_2,{mic_dir / 'mic_2.mrc'},masks/mic_2.npy,saved\n"
        f"1,mic_3,{mic_dir / 'mic_3.mrc'},masks/mic_3.npy,saved\n"
        f"1,mic_4,{mic_dir / 'mic_4.mrc'},,\n",
        encoding="utf-8",
    )
    output = tmp_path / "finetune" / "cryofilter_training_manifest.csv"

    summary = prepare_training_manifest(
        manifest=manifest,
        output_manifest=output,
        train_fraction=0.7,
        seed=42,
    )

    assert summary["rows"] == 4
    assert summary["split_counts"] == {"TRAINING": 3, "VALIDATION": 1}
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"TRAINING", "VALIDATION"}
    assert all(Path(row["gt_mask_path"]).is_absolute() for row in rows)
    assert all(Path(row["micrograph_path"]).is_absolute() for row in rows)


def test_prepare_training_manifest_reuses_cryosparc_transfer_source_paths(tmp_path: Path) -> None:
    mask_dir = tmp_path / "annotation_exports" / "session" / "masks"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "mic_0.npy", np.ones((4, 4), dtype=np.uint8))
    np.save(mask_dir / "mic_1.npy", np.ones((4, 4), dtype=np.uint8))
    manifest = tmp_path / "annotation_exports" / "session" / "manifest_edited.csv"
    manifest.write_text(
        "dataset_id,stem,micrograph_path,gt_mask_path,source_uid\n"
        "1,mic_0,/stale/mic_0.mrc,masks/mic_0.npy,7\n"
        "1,mic_1,/stale/mic_1.mrc,masks/mic_1.npy,8\n",
        encoding="utf-8",
    )
    transfer_manifest = tmp_path / "stage" / "transfer_manifest.json"
    transfer_manifest.parent.mkdir()
    transfer_manifest.write_text(
        json.dumps(
            {
                "micrographs": [
                    {"uid": "7", "source_path": "/cryosparc/project/mic_0.mrc", "transfer_filename": "micrographs/7_mic_0.mrc"},
                    {"uid": "8", "source_path": "/cryosparc/project/mic_1.mrc", "transfer_filename": "micrographs/8_mic_1.mrc"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "finetune" / "cryofilter_training_manifest.csv"

    prepare_training_manifest(
        manifest=manifest,
        output_manifest=output,
        transfer_manifest=transfer_manifest,
        prefer_source_paths=True,
        train_fraction=0.7,
        seed=42,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["micrograph_path"] for row in rows] == [
        "/cryosparc/project/mic_0.mrc",
        "/cryosparc/project/mic_1.mrc",
    ]
