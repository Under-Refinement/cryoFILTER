from __future__ import annotations

import csv
import json
from pathlib import Path

import mrcfile
import numpy as np

from cryofilter.app import browser_annotation
from cryofilter.app.server import AppState


def _write_micrograph(path: Path, shape: tuple[int, int] = (32, 48)) -> None:
    values = np.linspace(-1.0, 1.0, num=shape[0] * shape[1], dtype=np.float32).reshape(shape)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(values)


def _write_manifest(path: Path, micrograph: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "stem",
                "micrograph_path",
                "pixel_size_angstrom",
                "gt_mask_path",
                "typed_label_path",
                "source_uid",
                "source_path",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset_id": "1",
                "stem": micrograph.stem,
                "micrograph_path": str(micrograph),
                "pixel_size_angstrom": "1.2",
                "gt_mask_path": "",
                "typed_label_path": "",
                "source_uid": "",
                "source_path": str(micrograph),
                "split": "TRAINING",
            }
        )


def test_browser_annotation_session_saves_mask_typed_labels_and_manifest(tmp_path: Path) -> None:
    micrograph = tmp_path / "mic_001.mrc"
    _write_micrograph(micrograph)
    manifest = tmp_path / "source_manifest.csv"
    _write_manifest(manifest, micrograph)

    summary = browser_annotation.create_browser_session(
        session_id="session1",
        manifest_path=manifest,
        run_dir=tmp_path / "annotations",
        source_mode="micrographs",
        title="Browser annotations",
    )

    assert summary["row_count"] == 1
    assert summary["saved_count"] == 0
    assert summary["first_unsaved_index"] == 0
    assert summary["next_index"] == 0
    row = browser_annotation.get_row_payload(summary, 0)
    assert row["source_width"] == 48
    assert row["source_height"] == 32

    preview, preview_info = browser_annotation.get_preview_png(summary, 0, max_dim=16)
    assert preview.startswith(b"\x89PNG")
    assert max(preview_info["preview_width"], preview_info["preview_height"]) == 16

    result = browser_annotation.save_row_annotations(
        summary,
        0,
        {
            "polygons": [
                {
                    "points": [[4, 4], [24, 4], [24, 20], [4, 20]],
                    "type_id": 1,
                }
            ]
        },
    )

    mask = np.load(result["mask_path"])
    typed = np.load(result["typed_label_path"])
    assert mask.shape == (32, 48)
    assert mask.sum() > 0
    assert set(np.unique(typed[mask > 0])) == {1}
    assert result["saved_count"] == 1
    assert result["first_unsaved_index"] is None
    assert result["last_saved_index"] == 0
    assert result["next_index"] == 0

    edited_manifest = Path(summary["manifest_edited_path"])
    with edited_manifest.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]["gt_mask_path"]
    assert exported[0]["typed_label_path"]
    assert exported[0]["typed_label_schema_name"] == "cryofilter_contamination_subtypes"
    assert exported[0]["annotation_status"] == "saved"


def test_browser_annotation_preserves_ellipse_shape_metadata(tmp_path: Path) -> None:
    micrograph = tmp_path / "mic_ellipse.mrc"
    _write_micrograph(micrograph)
    manifest = tmp_path / "source_manifest.csv"
    _write_manifest(manifest, micrograph)

    summary = browser_annotation.create_browser_session(
        session_id="session_ellipse",
        manifest_path=manifest,
        run_dir=tmp_path / "annotations",
        source_mode="micrographs",
        title="Browser annotations",
    )

    result = browser_annotation.save_row_annotations(
        summary,
        0,
        {
            "polygons": [
                {
                    "points": [[8, 8], [22, 8], [28, 16], [22, 24], [8, 24], [2, 16]],
                    "type_id": 4,
                    "shape": "ellipse",
                    "bounds": {"x0": 2, "y0": 8, "x1": 28, "y1": 24},
                    "rotation": 0.42,
                }
            ]
        },
    )

    vector = json.loads(Path(result["browser_annotation_path"]).read_text(encoding="utf-8"))
    saved_polygon = vector["polygons"][0]
    assert saved_polygon["shape"] == "ellipse"
    assert saved_polygon["bounds"] == {"x0": 2.0, "y0": 8.0, "x1": 28.0, "y1": 24.0}
    assert abs(saved_polygon["rotation"] - 0.42) < 1e-12
    typed = np.load(result["typed_label_path"])
    assert set(np.unique(typed[typed > 0])) == {4}


def test_browser_annotation_erase_shape_subtracts_from_mask(tmp_path: Path) -> None:
    micrograph = tmp_path / "mic_erase.mrc"
    _write_micrograph(micrograph)
    manifest = tmp_path / "source_manifest.csv"
    _write_manifest(manifest, micrograph)

    summary = browser_annotation.create_browser_session(
        session_id="session_erase",
        manifest_path=manifest,
        run_dir=tmp_path / "annotations",
        source_mode="micrographs",
        title="Browser annotations",
    )

    result = browser_annotation.save_row_annotations(
        summary,
        0,
        {
            "polygons": [
                {
                    "points": [[4, 4], [40, 4], [40, 28], [4, 28]],
                    "type_id": 1,
                    "mode": "add",
                },
                {
                    "points": [[24, 10], [32, 16], [24, 22], [16, 16]],
                    "mode": "erase",
                    "shape": "ellipse",
                    "bounds": {"x0": 16, "y0": 10, "x1": 32, "y1": 22},
                },
            ]
        },
    )

    mask = np.load(result["mask_path"])
    typed = np.load(result["typed_label_path"])
    assert mask[8, 8] == 1
    assert typed[8, 8] == 1
    assert mask[16, 24] == 0
    assert typed[16, 24] == 0
    vector = json.loads(Path(result["browser_annotation_path"]).read_text(encoding="utf-8"))
    assert vector["polygons"][1]["mode"] == "erase"
    assert vector["polygons"][1]["shape"] == "ellipse"


def test_app_state_creates_browser_annotation_session_from_micrographs(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    source_dir = tmp_path / "micrographs"
    source_dir.mkdir()
    _write_micrograph(source_dir / "mic_001.mrc")
    _write_micrograph(source_dir / "mic_002.mrc")

    state = AppState(work_dir)
    session = state.create_annotation_session(
        {
            "source_mode": "micrographs",
            "micrograph_source": str(source_dir),
            "output_root": "annotation_exports",
            "output_name": "browser_test",
            "dataset_id": "gridA",
            "limit_micrographs": "2",
        }
    )

    assert session["ok"] is True
    assert session["row_count"] == 2
    assert session["first_unsaved_index"] == 0
    assert session["next_index"] == 0
    assert state.list_annotation_sessions()[0]["id"] == session["id"]
    row = state.get_annotation_row(session["id"], 1)
    assert row["stem"] == "mic_002"
    preview, _info = state.get_annotation_preview(session["id"], 1, max_dim=16)
    assert preview.startswith(b"\x89PNG")
