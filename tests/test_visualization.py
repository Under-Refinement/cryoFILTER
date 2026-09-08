from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from cryofilter.visualization import PANEL_TITLES, PROBABILITY_CMAP, render_filtering_png


def test_render_filtering_png_writes_display_image(tmp_path: Path) -> None:
    assert PROBABILITY_CMAP == "viridis"
    assert PANEL_TITLES == (
        "Raw micrograph",
        "Final-mask overlay",
        "Contamination probability",
        "Retained image regions",
    )
    rng = np.random.default_rng(7)
    image = rng.normal(size=(96, 128)).astype(np.float32)
    probability = np.zeros((96, 128), dtype=np.float32)
    probability[16:72, 28:100] = np.linspace(0.2, 1.0, 56)[:, None]
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[24:64, 40:88] = 1
    output = tmp_path / "diagnostic.png"

    result = render_filtering_png(
        image=image,
        probability=probability,
        mask=mask,
        output_path=output,
        max_display_dim=64,
        dpi=72,
    )

    assert result == output.resolve()
    assert output.is_file()
    rendered = mpimg.imread(output)
    assert rendered.ndim == 3
    assert rendered.shape[0] > 0
    assert rendered.shape[1] > 0
