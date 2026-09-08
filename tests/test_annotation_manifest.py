from __future__ import annotations

import csv
import json
from pathlib import Path

from cryofilter.app.annotation_manifest import (
    collect_micrographs,
    rows_from_micrographs,
    rows_from_transfer_manifest,
    write_manifest,
)


def test_collect_micrographs_and_write_manifest(tmp_path: Path) -> None:
    micrographs = tmp_path / "micrographs"
    nested = micrographs / "nested"
    nested.mkdir(parents=True)
    first = micrographs / "a.mrc"
    second = nested / "b.mrcs"
    first.write_bytes(b"not-a-real-mrc")
    second.write_bytes(b"not-a-real-mrcs")
    (micrographs / "ignore.txt").write_text("ignore", encoding="utf-8")

    paths = collect_micrographs(micrographs, recursive=True)
    rows = rows_from_micrographs(paths, dataset_id="1", pixel_size_angstrom="1.06")
    manifest = tmp_path / "annotation" / "source_manifest.csv"
    write_manifest(rows, manifest)

    with manifest.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert [row["stem"] for row in loaded] == ["a", "b"]
    assert all(row["dataset_id"] == "1" for row in loaded)
    assert all(row["pixel_size_angstrom"] == "1.06" for row in loaded)
    assert all(Path(row["micrograph_path"]).is_absolute() for row in loaded)
    assert sorted(row["split"] for row in loaded) == ["TRAINING", "VALIDATION"]


def test_rows_from_transfer_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "run" / "transfer_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "micrographs": [
                    {
                        "uid": 7,
                        "transfer_filename": "7_example.mrc",
                        "source_path": "/cryosparc/project/J1/imported/7_example.mrc",
                        "pixel_size_angstrom": 1.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = rows_from_transfer_manifest(manifest, dataset_id="2")

    assert rows == [
        {
            "dataset_id": "2",
            "stem": "7_example",
            "micrograph_path": str((tmp_path / "run" / "transfer" / "micrographs" / "7_example.mrc").resolve()),
            "pixel_size_angstrom": "1.25",
            "gt_mask_path": "",
            "typed_label_path": "",
            "source_uid": "7",
            "source_path": "/cryosparc/project/J1/imported/7_example.mrc",
            "split": "TRAINING",
        }
    ]

    rows_from_source = rows_from_transfer_manifest(
        manifest,
        dataset_id="2",
        prefer_source_paths=True,
    )
    assert rows_from_source[0]["micrograph_path"] == "/cryosparc/project/J1/imported/7_example.mrc"
    assert rows_from_source[0]["stem"] == "7_example"
