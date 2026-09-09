from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cryofilter.qc_render import montage_from_paths, render_particle_overlay_png


def test_particle_overlay_renderer_writes_png_and_contact_sheet(tmp_path: Path) -> None:
    yy, xx = np.mgrid[0:96, 0:128]
    image = (
        np.sin(xx / 9.0)
        + np.cos(yy / 11.0)
        + np.random.default_rng(4).normal(0, 0.15, (96, 128))
    ).astype(np.float32)
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[24:76, 50:106] = 1
    typed_mask = np.zeros((96, 128), dtype=np.uint8)
    typed_mask[24:50, 50:106] = 1
    typed_mask[50:76, 50:106] = 3
    probability = np.clip((xx + yy) / float(xx.max() + yy.max()), 0.0, 1.0).astype(np.float32)
    coords = np.asarray([[64.0, 48.0], [18.0, 18.0]], dtype=np.float32)
    keep = np.asarray([False, True], dtype=bool)

    frame_path = render_particle_overlay_png(
        image=image,
        mask=mask,
        coords_xy=coords,
        keep=keep,
        output_path=tmp_path / "frame.png",
        label="mic_001 kept 1 / rejected 1",
        max_display_dim=96,
        particle_diameter_px=12,
    )
    assert frame_path.exists()

    arr = np.asarray(Image.open(frame_path).convert("RGB"))
    assert max(arr.shape[:2]) <= 96
    assert np.any(arr[..., 0] != arr[..., 2])

    paired_path = render_particle_overlay_png(
        image=image,
        mask=mask,
        coords_xy=coords,
        keep=keep,
        output_path=tmp_path / "frame_two_panel.png",
        probability_map=probability,
        label="mic_001 kept 1 / rejected 1",
        max_display_dim=96,
        particle_diameter_px=12,
        include_raw_panel=True,
        include_probability_panel=True,
        panel_gap_px=8,
    )
    paired = Image.open(paired_path)
    assert paired.size[0] == arr.shape[1] * 3 + 16
    assert paired.size[1] == arr.shape[0]

    typed_path = render_particle_overlay_png(
        image=image,
        mask=mask,
        coords_xy=coords,
        keep=keep,
        output_path=tmp_path / "frame_four_panel.png",
        probability_map=probability,
        typed_mask=typed_mask,
        label="mic_001 kept 1 / rejected 1",
        max_display_dim=96,
        particle_diameter_px=12,
        include_raw_panel=True,
        include_typed_mask_panel=True,
        include_probability_panel=True,
        panel_gap_px=8,
    )
    typed = Image.open(typed_path)
    assert typed.size[0] == arr.shape[1] * 4 + 24
    assert typed.size[1] == arr.shape[0]

    sheet_path = montage_from_paths(
        [paired_path],
        output_path=tmp_path / "contact_sheet.png",
        cols=1,
        tile=(240, 120),
    )
    assert sheet_path.exists()
    sheet = Image.open(sheet_path)
    assert sheet.size == (240, 120)
