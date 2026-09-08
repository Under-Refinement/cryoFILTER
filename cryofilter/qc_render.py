"""Fast particle-filtering QC render helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LIGHT_PURPLE_RGB = (178, 126, 205)
MASK_OUTLINE_RGB = (255, 255, 255)
PARTICLE_KEEP_OUTLINE_RGB = (0, 114, 189)
PARTICLE_REJECT_OUTLINE_RGB = (217, 83, 25)
GOOD_BORDER_RGB = (60, 175, 90)
WARN_BORDER_RGB = (225, 70, 60)
BACKGROUND_RGB = (245, 245, 245)
PANEL_GAP_RGB = (25, 25, 25)
PROBABILITY_COLORMAP = "viridis"


def _resize_to_shape(arr: np.ndarray, target_shape: tuple[int, int], order: int) -> np.ndarray:
    from scipy.ndimage import zoom

    a = np.asarray(arr)
    if tuple(a.shape[:2]) == tuple(target_shape):
        return a
    zoom_factors = (
        float(target_shape[0]) / float(a.shape[0]),
        float(target_shape[1]) / float(a.shape[1]),
    )
    if a.ndim == 3:
        zoom_factors = (*zoom_factors, 1.0)
    out = np.asarray(zoom(a, zoom_factors, order=int(order)))
    if tuple(out.shape[:2]) == tuple(target_shape):
        return out
    fixed = np.zeros((*target_shape, *out.shape[2:]), dtype=out.dtype)
    h = min(int(target_shape[0]), int(out.shape[0]))
    w = min(int(target_shape[1]), int(out.shape[1]))
    fixed[:h, :w, ...] = out[:h, :w, ...]
    return fixed


def _contrast_normalize(
    raw: np.ndarray,
    *,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, float(percentile_low)))
    hi = float(np.percentile(arr, float(percentile_high)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _resize_display_image(img8: np.ndarray, out_long_edge: int) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(img8, dtype=np.uint8)
    if int(out_long_edge) <= 0:
        return arr
    h, w = arr.shape[:2]
    long_edge = max(int(h), int(w))
    if long_edge <= int(out_long_edge):
        return arr
    scale = float(out_long_edge) / float(long_edge)
    width = max(1, int(round(float(w) * scale)))
    height = max(1, int(round(float(h) * scale)))
    return np.asarray(Image.fromarray(arr, mode="L").resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)


def micrograph_to_display(
    raw: np.ndarray,
    *,
    out_long_edge: int = 1400,
    median: int = 3,
    percentile_low: float = 0.2,
    percentile_high: float = 99.8,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Prepare an 8-bit display micrograph and return ``(scale_x, scale_y)``."""

    arr = np.asarray(raw, dtype=np.float32)
    if int(median) > 1:
        from scipy.ndimage import median_filter

        arr = median_filter(arr, size=int(median))
    display = _contrast_normalize(
        arr,
        percentile_low=float(percentile_low),
        percentile_high=float(percentile_high),
    )
    img8 = np.clip(np.rint(display * 255.0), 0, 255).astype(np.uint8)
    img8 = _resize_display_image(img8, int(out_long_edge))
    scale_y = float(img8.shape[0]) / float(max(arr.shape[0], 1))
    scale_x = float(img8.shape[1]) / float(max(arr.shape[1], 1))
    return img8, (scale_x, scale_y)


def prep_micrograph(
    raw: np.ndarray,
    *,
    out_long_edge: int = 1400,
    median: int = 3,
    saturation_pct: float = 0.2,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Prepare a TEM-style 8-bit display image and return ``(scale_x, scale_y)``."""

    return micrograph_to_display(
        raw,
        out_long_edge=int(out_long_edge),
        median=int(median),
        percentile_low=float(saturation_pct),
        percentile_high=100.0 - float(saturation_pct),
    )


def prep_mask(mask: np.ndarray, out_shape: tuple[int, int], *, smooth: bool = False) -> np.ndarray:
    """Resize a mask to display shape."""

    arr = np.asarray(mask, dtype=np.float32)
    if tuple(arr.shape[:2]) != tuple(out_shape):
        arr = _resize_to_shape(arr, tuple(out_shape), order=0)
    if smooth:
        from scipy.ndimage import gaussian_filter

        arr = gaussian_filter(arr.astype(np.float32, copy=False), sigma=0.8)
    return (arr > 0.5).astype(np.uint8)


def _to_rgb(img8: np.ndarray) -> np.ndarray:
    arr = np.asarray(img8)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2).astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8, copy=False)
    raise ValueError(f"Expected grayscale or RGB image, got shape={arr.shape}")


def _load_label_font(font_size: int):
    from PIL import ImageFont

    candidates: list[str] = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation2/LiberationSans-Regular.ttf",
    ]
    try:
        import matplotlib

        mpl_fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates.insert(0, str(mpl_fonts / "DejaVuSans-Bold.ttf"))
        candidates.insert(1, str(mpl_fonts / "DejaVuSans.ttf"))
    except Exception:
        pass

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, int(font_size))
        except Exception:
            continue
    return ImageFont.load_default()


def _mask_outline(mask_bool: np.ndarray, width_px: int) -> np.ndarray:
    from scipy import ndimage as ndi

    width = max(1, int(width_px))
    structure = np.ones((3, 3), dtype=bool)
    outer = ndi.binary_dilation(mask_bool, structure=structure, iterations=width)
    inner = ndi.binary_erosion(mask_bool, structure=structure, iterations=max(1, width // 2))
    return np.asarray(outer & ~inner, dtype=bool)


def probability_to_rgb(
    probability_map: np.ndarray,
    out_shape: tuple[int, int],
    *,
    colormap: str = PROBABILITY_COLORMAP,
) -> np.ndarray:
    """Render a probability map as an 8-bit RGB viridis image."""

    prob = np.asarray(probability_map, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected a 2D probability map, got shape={prob.shape}")
    if tuple(prob.shape[:2]) != tuple(out_shape):
        prob = _resize_to_shape(prob, tuple(out_shape), order=1)
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(str(colormap))
        rgb = np.asarray(cmap(prob)[..., :3] * 255.0, dtype=np.float32)
    except Exception:
        rgb = np.stack(
            [
                np.clip(255.0 * (0.20 + 0.80 * prob), 0.0, 255.0),
                np.clip(255.0 * np.sqrt(prob), 0.0, 255.0),
                np.clip(255.0 * (1.0 - prob), 0.0, 255.0),
            ],
            axis=-1,
        )
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def render_probability_frame(
    probability_map: np.ndarray,
    out_shape: tuple[int, int],
    *,
    colormap: str = PROBABILITY_COLORMAP,
) -> np.ndarray:
    """Render one RGB probability-map reference panel."""

    return probability_to_rgb(probability_map, tuple(out_shape), colormap=colormap)


def _draw_label(draw, label: str, image_size: tuple[int, int]) -> None:
    if not label:
        return
    font_size = max(16, int(round(min(int(image_size[0]), int(image_size[1])) * 0.022)))
    max_text_width = int(max(1, int(image_size[0]) * 0.88))
    while font_size > 14:
        stroke_width = max(2, int(round(font_size * 0.10)))
        font = _load_label_font(font_size)
        text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=stroke_width)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width <= max_text_width:
            break
        font_size -= 2
    stroke_width = max(2, int(round(font_size * 0.10)))
    font = _load_label_font(font_size)
    pad = max(6, int(round(font_size * 0.34)))
    x0 = pad * 2
    text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=stroke_width)
    text_height = text_bbox[3] - text_bbox[1]
    y0 = max(pad * 2, int(image_size[1]) - text_height - pad * 2)
    bbox = draw.textbbox((x0, y0), str(label), font=font, stroke_width=stroke_width)
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=3,
        fill=(0, 0, 0, 165),
    )
    draw.text(
        (x0, y0),
        str(label),
        fill=(255, 255, 255, 255),
        font=font,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 225),
    )


def render_fast(
    img8: np.ndarray,
    mask: np.ndarray,
    coords_xy: np.ndarray,
    keep: np.ndarray,
    *,
    diameter_px: float = 34.0,
    label: str | None = None,
    aa: bool = True,
    mask_rgb: tuple[int, int, int] = LIGHT_PURPLE_RGB,
    mask_alpha: float = 0.35,
    mask_outline_rgb: tuple[int, int, int] = MASK_OUTLINE_RGB,
    mask_outline_alpha: float = 0.88,
    mask_outline_px: int = 2,
) -> np.ndarray:
    """Render one RGB QC frame with mask, kept particles, and rejected particles."""

    from PIL import Image, ImageDraw

    rgb = _to_rgb(img8)
    mask_bool = np.asarray(mask, dtype=bool)
    if tuple(mask_bool.shape) != tuple(rgb.shape[:2]):
        mask_bool = prep_mask(mask_bool.astype(np.uint8), tuple(rgb.shape[:2])).astype(bool)

    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    if np.any(mask_bool):
        overlay_rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        overlay_rgba[mask_bool, :3] = np.asarray(mask_rgb, dtype=np.uint8)
        overlay_rgba[mask_bool, 3] = int(round(255 * float(np.clip(mask_alpha, 0.0, 1.0))))
        base = Image.alpha_composite(base, Image.fromarray(overlay_rgba, mode="RGBA"))
        outline_bool = _mask_outline(mask_bool, int(mask_outline_px))
        if np.any(outline_bool):
            outline_rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
            outline_rgba[outline_bool, :3] = np.asarray(mask_outline_rgb, dtype=np.uint8)
            outline_rgba[outline_bool, 3] = int(round(255 * float(np.clip(mask_outline_alpha, 0.0, 1.0))))
            base = Image.alpha_composite(base, Image.fromarray(outline_rgba, mode="RGBA"))

    scale = 2 if bool(aa) else 1
    if scale > 1:
        canvas = base.resize((base.width * scale, base.height * scale), Image.Resampling.BICUBIC)
    else:
        canvas = base
    draw = ImageDraw.Draw(canvas, mode="RGBA")

    coords = np.asarray(coords_xy, dtype=np.float32)
    if coords.size:
        coords = coords.reshape(-1, 2)
    else:
        coords = np.empty((0, 2), dtype=np.float32)
    keep_arr = np.asarray(keep, dtype=bool).reshape(-1)
    if len(keep_arr) != len(coords):
        raise ValueError("keep must have one boolean value per particle coordinate")

    radius = max(2.0, float(diameter_px) * 0.625) * scale
    circle_width = max(4, int(round(float(diameter_px) / 7.0))) * scale
    x_width = max(7, int(round(float(diameter_px) / 4.8))) * scale

    kept_coords = coords[keep_arr]
    removed_coords = coords[~keep_arr]
    for x, y in kept_coords:
        sx = float(x) * scale
        sy = float(y) * scale
        box = (sx - radius, sy - radius, sx + radius, sy + radius)
        halo_width = max(circle_width + 2 * scale, int(round(circle_width * 1.45)))
        draw.ellipse(box, outline=(*PARTICLE_KEEP_OUTLINE_RGB, 230), width=halo_width)
        draw.ellipse(box, outline=(255, 255, 255, 255), width=circle_width)

    for x, y in removed_coords:
        sx = float(x) * scale
        sy = float(y) * scale
        r = radius * 0.78
        halo_width = max(x_width + 2 * scale, int(round(x_width * 1.7)))
        draw.line((sx - r, sy - r, sx + r, sy + r), fill=(*PARTICLE_REJECT_OUTLINE_RGB, 225), width=halo_width)
        draw.line((sx - r, sy + r, sx + r, sy - r), fill=(*PARTICLE_REJECT_OUTLINE_RGB, 225), width=halo_width)
        draw.line((sx - r, sy - r, sx + r, sy + r), fill=(0, 0, 0, 255), width=x_width)
        draw.line((sx - r, sy + r, sx + r, sy - r), fill=(0, 0, 0, 255), width=x_width)

    _draw_label(draw, str(label or ""), canvas.size)

    if scale > 1:
        canvas = canvas.resize(base.size, Image.Resampling.LANCZOS)
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def render_raw_frame(img8: np.ndarray, *, label: str | None = None) -> np.ndarray:
    """Render one RGB reference frame without mask or particle annotations."""

    from PIL import Image, ImageDraw

    canvas = Image.fromarray(_to_rgb(img8), mode="RGB").convert("RGBA")
    if label:
        _draw_label(ImageDraw.Draw(canvas, mode="RGBA"), str(label), canvas.size)
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gap: int = 16,
    fill: tuple[int, int, int] = PANEL_GAP_RGB,
) -> np.ndarray:
    """Place two RGB frames next to each other, centered vertically."""

    return horizontal_panels([left, right], gap=gap, fill=fill)


def horizontal_panels(
    panels: Sequence[np.ndarray],
    *,
    gap: int = 16,
    fill: tuple[int, int, int] = PANEL_GAP_RGB,
) -> np.ndarray:
    """Place two or more RGB frames side by side, centered vertically."""

    from PIL import Image

    images = [Image.fromarray(_to_rgb(panel), mode="RGB") for panel in panels]
    if not images:
        raise ValueError("Need at least one panel")
    gap = max(0, int(gap))
    width = sum(image.width for image in images) + gap * max(0, len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), tuple(int(v) for v in fill))
    x0 = 0
    for image in images:
        canvas.paste(image, (x0, max(0, (height - image.height) // 2)))
        x0 += image.width + gap
    return np.asarray(canvas, dtype=np.uint8)


def render_particle_overlay_png(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    coords_xy: np.ndarray,
    keep: np.ndarray,
    output_path: str | Path,
    probability_map: np.ndarray | None = None,
    label: str | None = None,
    max_display_dim: int = 1400,
    particle_diameter_px: float = 34.0,
    mask_alpha: float = 0.35,
    include_raw_panel: bool = False,
    include_probability_panel: bool = False,
    panel_gap_px: int = 16,
) -> Path:
    """Write one particle-filtering QC overlay PNG."""

    from PIL import Image

    img8, (scale_x, scale_y) = prep_micrograph(image, out_long_edge=int(max_display_dim), median=3)
    mask_disp = prep_mask(mask, tuple(img8.shape), smooth=False)
    coords = np.asarray(coords_xy, dtype=np.float32)
    if coords.size:
        coords = coords.reshape(-1, 2).copy()
        coords[:, 0] *= float(scale_x)
        coords[:, 1] *= float(scale_y)
    else:
        coords = np.empty((0, 2), dtype=np.float32)
    frame = render_fast(
        img8,
        mask_disp,
        coords,
        np.asarray(keep, dtype=bool),
        diameter_px=float(particle_diameter_px),
        label=label,
        aa=True,
        mask_alpha=float(mask_alpha),
    )
    if include_raw_panel:
        panels = [render_raw_frame(img8), frame]
    else:
        panels = [frame]
    if include_probability_panel:
        if probability_map is None:
            raise ValueError("include_probability_panel=True requires probability_map")
        panels.append(render_probability_frame(probability_map, tuple(img8.shape)))
    if len(panels) > 1:
        frame = horizontal_panels(panels, gap=int(panel_gap_px))
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGB").save(path)
    return path


def _load_2d_array(path: str | Path) -> np.ndarray:
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix in {".mrc", ".mrcs", ".map"}:
        import mrcfile

        with mrcfile.open(p, permissive=True) as mrc:
            arr = np.asarray(mrc.data)
    elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        from PIL import Image

        arr = np.asarray(Image.open(p).convert("F"))
    else:
        raise ValueError(f"Unsupported image file extension for {p}")

    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[0] > 0:
        return np.asarray(arr[0], dtype=np.float32)
    raise ValueError(f"Expected a 2D image or stack with at least one plane, got shape={arr.shape}")


def _masked_rgb_frame(
    img8: np.ndarray,
    mask: np.ndarray,
    *,
    mask_rgb: tuple[int, int, int],
    alpha: float,
    outline: bool = True,
) -> np.ndarray:
    from PIL import Image

    rgb = _to_rgb(img8)
    mask_bool = np.asarray(mask, dtype=bool)
    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    if np.any(mask_bool):
        overlay_rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        overlay_rgba[mask_bool, :3] = np.asarray(mask_rgb, dtype=np.uint8)
        overlay_rgba[mask_bool, 3] = int(round(255 * float(np.clip(alpha, 0.0, 1.0))))
        base = Image.alpha_composite(base, Image.fromarray(overlay_rgba, mode="RGBA"))
        if bool(outline):
            outline_bool = _mask_outline(mask_bool, 2)
            if np.any(outline_bool):
                outline_rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
                outline_rgba[outline_bool, :3] = np.asarray(MASK_OUTLINE_RGB, dtype=np.uint8)
                outline_rgba[outline_bool, 3] = 225
                base = Image.alpha_composite(base, Image.fromarray(outline_rgba, mode="RGBA"))
    return np.asarray(base.convert("RGB"), dtype=np.uint8)


def _add_title(frame: np.ndarray, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    text = str(title or "").strip()
    if not text:
        return frame
    img = Image.fromarray(_to_rgb(frame), mode="RGB")
    font = _load_label_font(max(20, int(round(img.height * 0.035))))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad = max(10, int(round(text_height * 0.55)))
    canvas = Image.new("RGB", (img.width, img.height + text_height + 2 * pad), PANEL_GAP_RGB)
    canvas.paste(img, (0, text_height + 2 * pad))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (max(0, (canvas.width - text_width) // 2), pad),
        text,
        fill=(245, 245, 245),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(canvas, dtype=np.uint8)


def render_clean_filtering_image(
    *,
    micrograph_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    title: str | None = None,
    max_display_dim: int = 1400,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    mask_alpha: float = 0.35,
    excluded_alpha: float = 0.70,
    dpi: int = 180,
    overwrite: bool = False,
    display_downsample_method: str = "fourier",
) -> Path:
    """Render a compact three-panel representative micrograph/mask figure."""

    del dpi, display_downsample_method

    from PIL import Image

    out = Path(output_path).expanduser()
    if out.exists() and not bool(overwrite):
        raise FileExistsError(f"Output already exists: {out}. Pass --overwrite to replace it.")

    micrograph = _load_2d_array(micrograph_path)
    mask = _load_2d_array(mask_path) > 0
    img8, _ = micrograph_to_display(
        micrograph,
        out_long_edge=int(max_display_dim),
        median=3,
        percentile_low=float(percentile_low),
        percentile_high=float(percentile_high),
    )
    mask_disp = prep_mask(mask.astype(np.uint8), tuple(img8.shape), smooth=False)

    raw_panel = render_raw_frame(img8, label="Raw micrograph")
    mask_panel = render_raw_frame(
        _masked_rgb_frame(
            img8,
            mask_disp,
            mask_rgb=LIGHT_PURPLE_RGB,
            alpha=float(mask_alpha),
        ),
        label="Predicted contamination",
    )
    clean_panel = render_raw_frame(
        _masked_rgb_frame(
            img8,
            mask_disp,
            mask_rgb=(30, 30, 30),
            alpha=float(excluded_alpha),
            outline=False,
        ),
        label="Filtered region",
    )
    frame = horizontal_panels([raw_panel, mask_panel, clean_panel], gap=16)
    if title:
        frame = _add_title(frame, str(title))

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGB").save(out)
    return out


def montage(
    frames: Sequence[np.ndarray],
    *,
    cols: int = 5,
    tile: int | tuple[int, int] = 400,
    gap: int = 8,
    borders: Sequence[tuple[int, int, int] | None] | None = None,
    border_px: int = 4,
) -> np.ndarray:
    """Build a contact sheet from RGB frames."""

    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("Need at least one frame for montage")

    cols = max(1, int(cols))
    if isinstance(tile, tuple):
        tile_width, tile_height = tile
    elif isinstance(tile, list):
        tile_width, tile_height = tile
    else:
        tile_width = tile_height = int(tile)
    tile_height = max(64, int(tile_height))
    if int(tile_width) <= 0:
        aspects = [
            float(np.asarray(frame).shape[1]) / float(max(np.asarray(frame).shape[0], 1))
            for frame in frames
        ]
        tile_width = int(round(float(tile_height) * float(np.median(aspects))))
    tile_width = max(64, int(tile_width))
    gap = max(0, int(gap))
    border_px = max(0, int(border_px))
    rows = int(np.ceil(len(frames) / float(cols)))
    width = cols * tile_width + (cols - 1) * gap
    height = rows * tile_height + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), BACKGROUND_RGB)

    border_values = list(borders or [])
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        x0 = col * (tile_width + gap)
        y0 = row * (tile_height + gap)
        border = border_values[i] if i < len(border_values) else None
        tile_img = Image.new("RGB", (tile_width, tile_height), BACKGROUND_RGB)
        if border is not None and border_px > 0:
            draw = ImageDraw.Draw(tile_img)
            border_color = tuple(int(v) for v in border)
            for offset in range(border_px):
                draw.rectangle(
                    (offset, offset, tile_width - 1 - offset, tile_height - 1 - offset),
                    outline=border_color,
                )
            inner_width = max(1, tile_width - 2 * border_px)
            inner_height = max(1, tile_height - 2 * border_px)
            paste_origin = (border_px, border_px)
        else:
            inner_width = tile_width
            inner_height = tile_height
            paste_origin = (0, 0)

        img = Image.fromarray(_to_rgb(frame), mode="RGB")
        img.thumbnail((inner_width, inner_height), Image.Resampling.LANCZOS)
        px = paste_origin[0] + max(0, (inner_width - img.width) // 2)
        py = paste_origin[1] + max(0, (inner_height - img.height) // 2)
        tile_img.paste(img, (px, py))
        sheet.paste(tile_img, (x0, y0))

    return np.asarray(sheet, dtype=np.uint8)


def montage_from_paths(
    frame_paths: Iterable[str | Path],
    *,
    output_path: str | Path,
    cols: int = 5,
    tile: int | tuple[int, int] = 400,
    gap: int = 8,
    borders: Sequence[tuple[int, int, int] | None] | None = None,
) -> Path:
    """Load saved frame PNGs and write a contact-sheet PNG."""

    from PIL import Image

    paths = [Path(path).expanduser() for path in frame_paths]
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
    sheet = montage(frames, cols=cols, tile=tile, gap=gap, borders=borders)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, mode="RGB").save(out)
    return out
