"""Optional display-image rendering for public cryoFILTER inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CONTAMINATION_COLOR = "#8B5CF6"
CONTAMINATION_EDGE_COLOR = "#C4B5FD"
FILTERED_COLOR = "#EEEAF7"
BACKGROUND_COLOR = "#F8F7FB"
PROBABILITY_CMAP = "viridis"
PANEL_TITLES = (
    "Raw micrograph",
    "Final-mask overlay",
    "Contamination probability",
    "Retained image regions",
)


def _resize_to_shape(array: np.ndarray, shape: tuple[int, int], *, order: int) -> np.ndarray:
    from scipy.ndimage import zoom

    source = np.asarray(array)
    if tuple(source.shape) == tuple(shape):
        return source
    resized = np.asarray(
        zoom(
            source,
            (float(shape[0]) / source.shape[0], float(shape[1]) / source.shape[1]),
            order=int(order),
        )
    )
    if tuple(resized.shape) == tuple(shape):
        return resized
    fixed = np.zeros(shape, dtype=resized.dtype)
    height = min(shape[0], resized.shape[0])
    width = min(shape[1], resized.shape[1])
    fixed[:height, :width] = resized[:height, :width]
    return fixed


def _display_shape(shape: tuple[int, int], max_display_dim: int) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    largest = max(height, width)
    if int(max_display_dim) <= 0 or largest <= int(max_display_dim):
        return height, width
    scale = float(max_display_dim) / float(largest)
    return max(8, int(round(height * scale))), max(8, int(round(width * scale)))


def _downsample_for_display(image: np.ndarray, max_display_dim: int) -> np.ndarray:
    from utils.fourier_rescale import fourier_rescale_2d

    source = np.asarray(image, dtype=np.float32)
    target_shape = _display_shape(tuple(source.shape), int(max_display_dim))
    if tuple(source.shape) == target_shape:
        return source
    scale = float(
        (target_shape[0] / source.shape[0] + target_shape[1] / source.shape[1]) * 0.5
    )
    resized = np.asarray(fourier_rescale_2d(source, scale), dtype=np.float32)
    if tuple(resized.shape) != target_shape:
        resized = _resize_to_shape(resized, target_shape, order=1)
    return resized.astype(np.float32, copy=False)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(source)
    if not np.any(finite):
        return np.zeros_like(source, dtype=np.float32)
    values = source[finite]
    low, high = (float(value) for value in np.percentile(values, (1.0, 99.0)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(source, dtype=np.float32)
    clean = np.where(finite, source, low)
    return np.clip((clean - low) / (high - low), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _blend_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    output = np.asarray(image_rgb, dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    if np.any(selected):
        color_array = np.asarray(color, dtype=np.float32)
        output[selected] = (1.0 - float(alpha)) * output[selected] + float(alpha) * color_array
    return np.clip(output, 0.0, 1.0)


def _blend_probability_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Blend selected pixels with confidence-weighted purple."""
    output = np.asarray(image_rgb, dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    if np.any(selected):
        confidence = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        alpha = (0.28 + 0.52 * confidence[selected])[..., None]
        color_array = np.asarray(color, dtype=np.float32)
        output[selected] = (1.0 - alpha) * output[selected] + alpha * color_array
    return np.clip(output, 0.0, 1.0)


def render_filtering_png(
    *,
    image: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    max_display_dim: int = 1400,
    dpi: int = 180,
) -> Path:
    """Write a display-sized raw, overlay, probability, and filtering diagnostic PNG."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if int(max_display_dim) < 8:
        raise ValueError("image_max_dim must be at least 8 pixels")
    if int(dpi) < 1:
        raise ValueError("image_dpi must be at least 1")

    source = np.asarray(image, dtype=np.float32)
    probability_map = np.asarray(probability, dtype=np.float32)
    mask_bool = np.asarray(mask) > 0
    if source.ndim != 2 or probability_map.ndim != 2 or mask_bool.ndim != 2:
        raise ValueError(
            "Diagnostic rendering requires 2D image, probability, and mask arrays"
        )

    displayed = _downsample_for_display(source, int(max_display_dim))
    if tuple(probability_map.shape) != tuple(displayed.shape):
        probability_map = _resize_to_shape(
            probability_map, displayed.shape, order=1
        )
    if tuple(mask_bool.shape) != tuple(displayed.shape):
        mask_bool = _resize_to_shape(mask_bool.astype(np.uint8), displayed.shape, order=0) > 0
    probability_map = np.clip(
        np.nan_to_num(probability_map, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    normalized = _normalize_for_display(displayed)
    base_rgb = np.repeat(normalized[..., None], 3, axis=2)
    color = mcolors.to_rgb(CONTAMINATION_COLOR)
    predicted = _blend_probability_mask(base_rgb, mask_bool, probability_map, color)
    filtered = _blend_mask(
        base_rgb,
        mask_bool,
        mcolors.to_rgb(FILTERED_COLOR),
        0.90,
    )
    height, width = normalized.shape
    panel_height = max(3.2, min(6.0, height / 320.0))
    panel_width = max(3.2, min(6.0, width / 320.0))
    figure, axes = plt.subplots(
        1, 4, figsize=(panel_width * 4.0, panel_height), facecolor=BACKGROUND_COLOR
    )
    panel_specs = (
        (base_rgb, PANEL_TITLES[0], False),
        (predicted, PANEL_TITLES[1], True),
        (None, PANEL_TITLES[2], False),
        (filtered, PANEL_TITLES[3], True),
    )
    probability_artist = None
    for axis, (panel, title, show_boundary) in zip(axes, panel_specs):
        if panel is None:
            probability_artist = axis.imshow(
                probability_map,
                origin="upper",
                interpolation="bilinear",
                cmap=PROBABILITY_CMAP,
                vmin=0.0,
                vmax=1.0,
            )
        else:
            axis.imshow(panel, origin="upper", interpolation="nearest")
        if show_boundary and np.any(mask_bool) and np.any(~mask_bool):
            axis.contour(
                mask_bool.astype(np.float32),
                levels=(0.5,),
                colors=(CONTAMINATION_EDGE_COLOR,),
                linewidths=0.65,
                alpha=0.90,
            )
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    if probability_artist is not None:
        colorbar = figure.colorbar(
            probability_artist,
            ax=axes[2],
            fraction=0.046,
            pad=0.025,
        )
        colorbar.set_ticks((0.0, 0.5, 1.0))
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.outline.set_visible(False)

    figure.text(
        0.5,
        0.025,
        (
            f"excluded mask area: {float(np.mean(np.asarray(mask) > 0)):.2%}"
            f"  |  mean contamination probability: {float(np.mean(probability_map)):.3f}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#3B3347",
    )
    figure.tight_layout(pad=0.6, rect=(0.0, 0.04, 1.0, 1.0))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor=BACKGROUND_COLOR,
    )
    plt.close(figure)
    return destination


__all__ = ["PANEL_TITLES", "PROBABILITY_CMAP", "render_filtering_png"]
