from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from PIL import Image

from cryofilter.cryosparc.diagnostics import build_diagnostics_manifest
from cryofilter.cryosparc.remote import cli as remote_cli
from cryofilter.cryosparc.remote.config import (
    BridgeConfig,
    CryoSPARCIntegrationConfig,
)


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (80, 120, 180)).save(path)


def test_build_diagnostics_manifest_with_typing_summary(tmp_path: Path) -> None:
    overlay_dir = tmp_path / "OTF_images"
    contact_sheet = overlay_dir / "particle_overlay_contact_sheet.png"
    frame = overlay_dir / "mic_001_particle_overlay.png"
    _write_png(contact_sheet)
    _write_png(frame)

    inference_summary = tmp_path / "inference_summary.json"
    inference_summary.write_text(
        json.dumps(
            {
                "run_id": "00000000-0000-0000-0000-000000000123",
                "particle_overlay_rendering": {
                    "enabled": True,
                    "final_contact_sheet_path": str(contact_sheet),
                    "frames": [{"output_png": str(frame)}],
                },
                "particle_filtering": {
                    "particles_input": 100,
                    "particles_kept": 75,
                    "particles_removed": 25,
                },
                "inputs": [
                    {
                        "input_mrc": str(tmp_path / "mic_001.mrc"),
                        "output_image_shape": [100, 100],
                        "mask_postprocessing": {
                            "final_mask_pixels": 1200,
                            "final_mask_fraction": 0.12,
                        },
                    },
                    {
                        "input_mrc": str(tmp_path / "mic_002.mrc"),
                        "output_image_shape": [100, 100],
                        "mask_postprocessing": {
                            "final_mask_pixels": 300,
                            "final_mask_fraction": 0.03,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    typing_summary = tmp_path / "typing_summary.json"
    typing_summary.write_text(
        json.dumps(
            {
                "type_order": ["Carbon", "Crystalline", "Aggregate", "Ethane"],
                "overall_contamination_summary": {
                    "carbon_area_px": 600,
                    "carbon_pct_of_contamination": 40.0,
                    "carbon_pct_total_image": 3.0,
                    "crystalline_area_px": 450,
                    "crystalline_pct_of_contamination": 30.0,
                    "crystalline_pct_total_image": 2.25,
                    "aggregate_area_px": 300,
                    "aggregate_pct_of_contamination": 20.0,
                    "aggregate_pct_total_image": 1.5,
                    "ethane_area_px": 150,
                    "ethane_pct_of_contamination": 10.0,
                    "ethane_pct_total_image": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_diagnostics_manifest(
        inference_summary_path=inference_summary,
        typing_summary_path=typing_summary,
        output_dir=tmp_path / "diagnostics",
        max_overlay_images=1,
    )

    kinds = [asset.kind for asset in manifest.assets]
    assert kinds == [
        "otf_contact_sheet",
        "otf_overlay",
        "contamination_total_pie",
        "contamination_by_micrograph_bar",
        "contamination_type_pie",
        "contamination_type_bar",
    ]
    assert manifest.summary["particles_removed"] == 25
    assert (tmp_path / "diagnostics" / "cryosparc_diagnostics_manifest.json").exists()
    for asset in manifest.assets:
        assert Path(asset.local_path).exists()


def test_build_diagnostics_manifest_selects_top_contaminated_overlays(tmp_path: Path) -> None:
    overlay_dir = tmp_path / "OTF_images"
    contact_sheet = overlay_dir / "particle_overlay_contact_sheet.png"
    low_frame = overlay_dir / "mic_low_particle_overlay.png"
    high_frame = overlay_dir / "mic_high_particle_overlay.png"
    mid_frame = overlay_dir / "mic_mid_particle_overlay.png"
    for path in (contact_sheet, low_frame, high_frame, mid_frame):
        _write_png(path)

    inference_summary = tmp_path / "inference_summary.json"
    inference_summary.write_text(
        json.dumps(
            {
                "particle_overlay_rendering": {
                    "enabled": True,
                    "final_contact_sheet_path": str(contact_sheet),
                    "frames": [
                        {"micrograph": str(tmp_path / "mic_low.mrc"), "output_png": str(low_frame)},
                        {"micrograph": str(tmp_path / "mic_high.mrc"), "output_png": str(high_frame)},
                        {"micrograph": str(tmp_path / "mic_mid.mrc"), "output_png": str(mid_frame)},
                    ],
                },
                "inputs": [
                    {
                        "input_mrc": str(tmp_path / "mic_low.mrc"),
                        "output_image_shape": [100, 100],
                        "mask_postprocessing": {
                            "final_mask_pixels": 100,
                            "final_mask_fraction": 0.01,
                        },
                    },
                    {
                        "input_mrc": str(tmp_path / "mic_high.mrc"),
                        "output_image_shape": [100, 100],
                        "mask_postprocessing": {
                            "final_mask_pixels": 3000,
                            "final_mask_fraction": 0.30,
                        },
                    },
                    {
                        "input_mrc": str(tmp_path / "mic_mid.mrc"),
                        "output_image_shape": [100, 100],
                        "mask_postprocessing": {
                            "final_mask_pixels": 1200,
                            "final_mask_fraction": 0.12,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_diagnostics_manifest(
        inference_summary_path=inference_summary,
        output_dir=tmp_path / "diagnostics",
        max_overlay_images=2,
    )

    overlay_assets = [asset for asset in manifest.assets if asset.kind == "otf_overlay"]
    assert [Path(asset.local_path).name for asset in overlay_assets] == [
        "mic_high_particle_overlay.png",
        "mic_mid_particle_overlay.png",
    ]
    contact_sheet = next(asset for asset in manifest.assets if asset.kind == "otf_contact_sheet")
    assert Path(contact_sheet.local_path).name == "top_contaminated_otf_contact_sheet.png"
    assert manifest.summary["otf_overlays_available"] == 3
    assert manifest.summary["otf_overlays_selected"] == 2
    assert manifest.summary["otf_overlay_selection_strategy"] == "top_contaminated_micrographs"
    assert manifest.summary["otf_overlay_ranked_labels"] == [
        {"rank": 1, "label": "mic_high.mrc", "contamination_pct": 30.0},
        {"rank": 2, "label": "mic_mid.mrc", "contamination_pct": 12.0},
    ]


def test_attach_diagnostics_pushes_assets_and_remote_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    png = tmp_path / "diagnostics" / "chart.png"
    _write_png(png)
    manifest_path = tmp_path / "diagnostics" / "cryosparc_diagnostics_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "run_id": None,
                "assets": [
                    {
                        "kind": "contamination_total_pie",
                        "title": "Total Contamination",
                        "caption": "Summary chart.",
                        "local_path": str(png),
                        "remote_path": None,
                        "mime_type": "image/png",
                        "source": "test",
                    }
                ],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeTransport:
        def run(self, argv, *, timeout=None):
            calls.append(("run", tuple(argv)))
            if "attach-diagnostics" in argv:
                return _Completed(
                    stdout=(
                        '{"ok": true, "protocol_version": 1, "project_uid": "P1", '
                        '"job_uid": "J2", "diagnostics_manifest_file": "/remote/diag/cryosparc_diagnostics_manifest.json", '
                        '"attached_assets": 1, "event_ids": ["event_1"], "errors": []}'
                    )
                )
            return _Completed()

        def push(self, local_path, remote_path, *, timeout=None, contents=False, check=True):
            calls.append(("push", (Path(local_path), remote_path)))
            return _Completed()

    monkeypatch.setattr(remote_cli, "_transport", lambda config: FakeTransport())
    args = argparse.Namespace(
        project="P1",
        job="J2",
        diagnostics_manifest=str(manifest_path),
        remote_diagnostics_dir="/remote/diag",
        timeout=30.0,
        json=True,
    )
    config = CryoSPARCIntegrationConfig(
        bridge=BridgeConfig(
            command="/opt/cryofilter/bin/cryofilter-bridge",
            remote_work_root="/remote/work",
        )
    )

    assert remote_cli._run_attach_diagnostics(args, config) == 0

    assert calls[0] == ("run", ("mkdir", "-p", "/remote/diag/assets"))
    assert calls[1][0] == "push"
    assert calls[1][1][1] == "/remote/diag/assets/01_chart.png"
    assert calls[2][0] == "push"
    assert calls[2][1][1] == "/remote/diag/cryosparc_diagnostics_manifest.json"
    assert calls[3][0] == "run"
    assert "attach-diagnostics" in calls[3][1]
