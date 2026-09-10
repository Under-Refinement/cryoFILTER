"""Predict release-style contamination types from binary contamination masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mrcfile
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from utils.contamination_typing import _is_edge_artifact_component, assign_frequency_component_type
from utils.contamination_subtypes import TYPE_ID_SEQUENCE, TYPE_ID_TO_DISPLAY, TYPE_ID_TO_KEY
from utils.image_utils import compute_frequency_band_psd_channels, normalize_image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BANDS_JSON = Path(__file__).resolve().parent / "configs" / "selected_frequency_bands.json"
MICROGRAPH_PATH_COLUMNS = (
    "micrograph_path",
    "full_mic_path",
    "down_mic_path",
    "mic_path",
    "resolved_mic_path",
    "staged_mic_path",
)
MASK_PATH_COLUMNS = (
    "gt_mask_path",
    "binary_mask_path",
    "mask_path",
    "pred_mask_path",
    "resolved_mask_path",
)
PIXEL_SIZE_COLUMNS = (
    "pixel_size_angstrom",
    "pixel_size",
    "apix",
)
PUBLIC_TYPE_ORDER = tuple(TYPE_ID_TO_DISPLAY[type_id] for type_id in TYPE_ID_SEQUENCE)
PUBLIC_TYPE_TO_ID = {TYPE_ID_TO_DISPLAY[type_id]: type_id for type_id in TYPE_ID_SEQUENCE}
PUBLIC_TYPE_TO_KEY = {TYPE_ID_TO_DISPLAY[type_id]: TYPE_ID_TO_KEY[type_id] for type_id in TYPE_ID_SEQUENCE}
INTERNAL_TO_PUBLIC_TYPE = {
    "edge/carbon": "Carbon",
    "crystalline": "Crystalline",
    "diffuse": "Aggregate",
    "isolated": "Ethane",
}
COMPONENT_SUMMARY_COLUMNS = (
    "image_id",
    "dataset_id",
    "stem",
    "component_id",
    "area_px",
    "small_component_auto_ethane",
    "type",
    "type_confidence",
)


def _add_type_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--manifest", type=Path, required=True, help="CSV with dataset_id, stem, micrograph path, and mask path.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV summaries and predicted typed masks.",
    )
    ap.add_argument(
        "--bands-json",
        type=Path,
        default=DEFAULT_BANDS_JSON,
        help="JSON file containing selected_frequency_bands_angstrom_inv.",
    )
    ap.add_argument(
        "--normalization-method",
        default="percentile_extra_wide",
        choices=("robust", "percentile", "percentile_wide", "percentile_extra_wide", "percentile_blend"),
        help="Real-space normalization used before PSD-band feature extraction.",
    )
    ap.add_argument(
        "--pixel-size-angstrom",
        type=float,
        default=None,
        help="Optional global pixel-size override when the manifest/header is missing or wrong.",
    )
    ap.add_argument(
        "--min-component-area-px",
        type=int,
        default=50,
        help="Components smaller than this are assigned to Ethane by default.",
    )
    ap.add_argument(
        "--expected-images",
        type=int,
        default=None,
        help="Expected full image count for live progress summaries.",
    )
    ap.add_argument("--crop-pad-px", type=int, default=16, help="Padding around each component crop before PSD extraction.")
    ap.add_argument("--min-crop-size-px", type=int, default=128, help="Minimum PSD crop size.")
    ap.add_argument("--max-psd-crop-size-px", type=int, default=768, help="Maximum PSD crop size before downsampling.")


def build_parser(*, prog: str = "predict_contamination_types.py") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    _add_type_arguments(ap)
    return ap


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    ap = subparsers.add_parser(
        "type",
        help="Predict release-style contamination types from binary contamination masks.",
    )
    _add_type_arguments(ap)
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_path(base_dir: Path, raw: object) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _first_existing_column(row: pd.Series, columns: Sequence[str]) -> Optional[str]:
    for col in columns:
        raw = row.get(col, None)
        if raw is None:
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return col
    return None


def _load_bands(path: Path) -> tuple[tuple[float, float], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple((float(lo), float(hi)) for lo, hi in payload["selected_frequency_bands_angstrom_inv"])


def _load_micrograph(path: Path) -> tuple[np.ndarray, Optional[float]]:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
        try:
            px = float(mrc.voxel_size.x)
            pixel_size = px if px > 0 else None
        except Exception:
            pixel_size = None
    if data.ndim == 2:
        image = data
    elif data.ndim == 3 and data.shape[0] > 0:
        image = data[0]
    else:
        raise ValueError(f"Unsupported micrograph dimensionality for {path}: {data.shape}")
    return np.asarray(image, dtype=np.float32), pixel_size


def _load_mask(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path)
    elif suffix in {".mrc", ".map"}:
        with mrcfile.open(path, permissive=True) as mrc:
            raw = np.asarray(mrc.data)
    else:
        raise ValueError(f"Unsupported mask format for {path}. Expected .npy or .mrc/.map")
    if raw.ndim == 2:
        arr = raw
    elif raw.ndim == 3 and raw.shape[0] > 0:
        arr = raw[0]
    else:
        raise ValueError(f"Unsupported mask dimensionality for {path}: {raw.shape}")
    return np.asarray(arr) > 0.5


def _resolve_pixel_size(
    row: pd.Series,
    *,
    header_pixel_size: Optional[float],
    override_pixel_size: Optional[float],
) -> float:
    if override_pixel_size is not None and np.isfinite(float(override_pixel_size)) and float(override_pixel_size) > 0:
        return float(override_pixel_size)
    for col in PIXEL_SIZE_COLUMNS:
        raw = row.get(col, None)
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value) and value > 0:
            return value
    if header_pixel_size is not None and np.isfinite(float(header_pixel_size)) and float(header_pixel_size) > 0:
        return float(header_pixel_size)
    raise ValueError(
        "Could not determine pixel size from --pixel-size-angstrom, manifest columns "
        f"{PIXEL_SIZE_COLUMNS}, or the MRC header."
    )


def _robust_z(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    med = np.nanmedian(arr)
    q25 = np.nanpercentile(arr, 25.0)
    q75 = np.nanpercentile(arr, 75.0)
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return pd.Series((arr - med) / scale, index=values.index)


def _expand_bounds(
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    image_shape: tuple[int, int],
    pad: int,
    min_size: int,
) -> tuple[int, int, int, int]:
    h, w = image_shape
    y0 = max(0, int(y0) - int(pad))
    y1 = min(h, int(y1) + int(pad))
    x0 = max(0, int(x0) - int(pad))
    x1 = min(w, int(x1) + int(pad))

    target_h = max(int(min_size), y1 - y0)
    target_w = max(int(min_size), x1 - x0)

    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    y0 = int(np.floor(cy - 0.5 * target_h))
    x0 = int(np.floor(cx - 0.5 * target_w))
    y1 = y0 + int(target_h)
    x1 = x0 + int(target_w)

    if y0 < 0:
        y1 += -y0
        y0 = 0
    if x0 < 0:
        x1 += -x0
        x0 = 0
    if y1 > h:
        shift = y1 - h
        y0 = max(0, y0 - shift)
        y1 = h
    if x1 > w:
        shift = x1 - w
        x0 = max(0, x0 - shift)
        x1 = w
    return y0, y1, x0, x1


def _extract_padded_crop(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    h, w = image.shape
    src_y0 = max(0, y0)
    src_y1 = min(h, y1)
    src_x0 = max(0, x0)
    src_x1 = min(w, x1)
    crop = np.asarray(image[src_y0:src_y1, src_x0:src_x1], dtype=np.float32)
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - h)
    pad_right = max(0, x1 - w)
    if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
        mode = "edge" if crop.size > 0 else "constant"
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
    return np.asarray(crop, dtype=np.float32)


def _component_band_scores(
    *,
    image_norm: np.ndarray,
    bbox: tuple[int, int, int, int],
    pixel_size_angstrom: float,
    frequency_bands: tuple[tuple[float, float], ...],
    crop_pad_px: int,
    min_crop_size_px: int,
    max_psd_crop_size_px: int,
) -> list[float]:
    min_y, max_y, min_x, max_x = bbox
    y0, y1, x0, x1 = _expand_bounds(
        y0=min_y,
        y1=max_y + 1,
        x0=min_x,
        x1=max_x + 1,
        image_shape=image_norm.shape,
        pad=crop_pad_px,
        min_size=min_crop_size_px,
    )
    crop = _extract_padded_crop(image_norm, y0, y1, x0, x1)
    effective_px = float(pixel_size_angstrom)

    max_dim = max(crop.shape)
    if int(max_psd_crop_size_px) > 0 and max_dim > int(max_psd_crop_size_px):
        scale = float(max_psd_crop_size_px) / float(max_dim)
        new_h = max(16, int(round(crop.shape[0] * scale)))
        new_w = max(16, int(round(crop.shape[1] * scale)))
        crop = ndi.zoom(crop, (new_h / crop.shape[0], new_w / crop.shape[1]), order=1).astype(np.float32, copy=False)
        effective_px = float(pixel_size_angstrom) * (1.0 / scale)

    channels = compute_frequency_band_psd_channels(
        crop,
        pixel_size_angstrom=effective_px,
        frequency_bands=frequency_bands,
        normalize=True,
        use_radial_normalization=True,
        include_anisotropy_channel=False,
    )
    return [float(np.max(channel)) if channel.size else float("nan") for channel in channels]


def _assign_types(component_df: pd.DataFrame) -> pd.DataFrame:
    df = component_df.copy()
    large_mask = ~df["small_component_auto_ethane"].astype(bool)
    large_df = df.loc[large_mask].copy()
    if not large_df.empty:
        for idx in range(1, 5):
            col = f"band_{idx}_score"
            large_df[f"band_{idx}_z"] = large_df.groupby("dataset_id")[col].transform(_robust_z)

        band_score_cols = [f"band_{idx}_score" for idx in range(1, 5)]
        band_vals = large_df[band_score_cols].to_numpy(dtype=float)
        large_df["dominant_band"] = 1 + np.argmax(band_vals, axis=1)
        large_df["hf_enrichment"] = np.maximum(large_df["band_3_z"], large_df["band_4_z"]) - np.maximum(
            large_df["band_1_z"],
            large_df["band_2_z"],
        )
        log_aspect = np.log1p(large_df["aspect_ratio"].to_numpy(dtype=float))
        positive_hf = np.maximum(large_df["hf_enrichment"].to_numpy(dtype=float), 0.0)

        large_df["crystalline_score"] = (
            1.30 * large_df["hf_enrichment"].to_numpy(dtype=float)
            + 0.45 * np.maximum(large_df["band_3_z"], large_df["band_4_z"]).to_numpy(dtype=float)
            + 0.20 * log_aspect
            - 0.35 * large_df["compactness"].to_numpy(dtype=float)
        )
        large_df["ethane_score"] = (
            1.30 * large_df["compactness"].to_numpy(dtype=float)
            + 0.95 * large_df["bbox_fill_frac"].to_numpy(dtype=float)
            - 0.45 * log_aspect
            - 0.75 * positive_hf
            - 1.20 * large_df["touches_border"].to_numpy(dtype=float)
        )
        large_df["aggregate_score"] = (
            1.10 * (1.0 - large_df["compactness"].to_numpy(dtype=float))
            + 0.85 * (1.0 - large_df["bbox_fill_frac"].to_numpy(dtype=float))
            + 0.25 * log_aspect
            - 0.25 * positive_hf
            - 0.70 * large_df["touches_border"].to_numpy(dtype=float)
        )

        assigned: list[str] = []
        confidence: list[float] = []
        for _, row in large_df.iterrows():
            label, conf = assign_frequency_component_type(
                edge_artifact=bool(int(row["edge_artifact"])),
                crystalline_score=float(row["crystalline_score"]),
                diffuse_score=float(row["aggregate_score"]),
                isolated_score=float(row["ethane_score"]),
            )
            assigned.append(INTERNAL_TO_PUBLIC_TYPE[str(label)])
            confidence.append(float(conf))
        large_df["type"] = assigned
        large_df["type_confidence"] = np.asarray(confidence, dtype=float)

        df.loc[large_df.index, large_df.columns] = large_df

    for idx in range(1, 5):
        z_col = f"band_{idx}_z"
        if z_col not in df.columns:
            df[z_col] = np.nan
    for col in ("hf_enrichment", "crystalline_score", "aggregate_score", "ethane_score"):
        if col not in df.columns:
            df[col] = np.nan
    if "dominant_band" not in df.columns:
        df["dominant_band"] = np.nan

    small_mask = df["small_component_auto_ethane"].astype(bool)
    df.loc[small_mask, "type"] = "Ethane"
    df.loc[small_mask, "type_confidence"] = 0.0
    return df


def _typed_mask_for_image(mask: np.ndarray, image_rows: pd.DataFrame) -> np.ndarray:
    labels, nlab = ndi.label(mask.astype(bool), structure=np.ones((3, 3), dtype=bool))
    typed = np.zeros(mask.shape, dtype=np.uint8)
    if nlab <= 0:
        return typed
    type_by_comp = {
        int(row["component_id"]): PUBLIC_TYPE_TO_ID[str(row["type"])]
        for _, row in image_rows.iterrows()
    }
    for comp_id, type_id in type_by_comp.items():
        typed[labels == int(comp_id)] = np.uint8(type_id)
    return typed


def _image_summary_rows(component_df: pd.DataFrame, manifest_rows: Iterable[pd.Series]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row0 in manifest_rows:
        key = f"{str(row0['dataset_id'])}__{str(row0['stem'])}"
        if component_df.empty:
            sub = component_df
        else:
            sub = component_df.loc[component_df["image_id"] == key]
        total_pixels = int(row0["_total_pixels"])
        contaminated_pixels = int(row0["_contaminated_pixels"])
        summary: dict[str, object] = {
            "image_id": str(key),
            "dataset_id": str(row0["dataset_id"]),
            "stem": str(row0["stem"]),
            "total_pixels": total_pixels,
            "contaminated_pixels": contaminated_pixels,
            "contamination_pct_total_image": float(100.0 * contaminated_pixels / max(total_pixels, 1)),
            "n_components": int(len(sub)),
            "n_large_components": int((~sub["small_component_auto_ethane"].astype(bool)).sum()) if not sub.empty else 0,
            "n_small_components": int(sub["small_component_auto_ethane"].astype(bool).sum()) if not sub.empty else 0,
        }
        for label in PUBLIC_TYPE_ORDER:
            slug = PUBLIC_TYPE_TO_KEY[label]
            area_px = int(sub.loc[sub["type"] == label, "area_px"].sum()) if not sub.empty else 0
            n_components = int((sub["type"] == label).sum()) if not sub.empty else 0
            summary[f"{slug}_components"] = n_components
            summary[f"{slug}_area_px"] = area_px
            summary[f"{slug}_pct_total_image"] = float(100.0 * area_px / max(total_pixels, 1))
            summary[f"{slug}_pct_of_contamination"] = float(100.0 * area_px / max(contaminated_pixels, 1))
        rows.append(summary)
    return rows


def _dataset_summary_df(image_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_id, sub in image_summary_df.groupby("dataset_id", sort=True):
        total_pixels = int(sub["total_pixels"].sum())
        contaminated_pixels = int(sub["contaminated_pixels"].sum())
        row: dict[str, object] = {
            "dataset_id": str(dataset_id),
            "n_images": int(len(sub)),
            "total_pixels": total_pixels,
            "contaminated_pixels": contaminated_pixels,
            "contamination_pct_total_image": float(100.0 * contaminated_pixels / max(total_pixels, 1)),
            "mean_image_contamination_pct": float(sub["contamination_pct_total_image"].mean()),
        }
        for label in PUBLIC_TYPE_ORDER:
            slug = PUBLIC_TYPE_TO_KEY[label]
            area_px = int(sub[f"{slug}_area_px"].sum())
            row[f"{slug}_area_px"] = area_px
            row[f"{slug}_pct_total_image"] = float(100.0 * area_px / max(total_pixels, 1))
            row[f"{slug}_pct_of_contamination"] = float(100.0 * area_px / max(contaminated_pixels, 1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def _overall_summary(image_summary_df: pd.DataFrame) -> dict[str, object]:
    total_pixels = int(image_summary_df["total_pixels"].sum())
    contaminated_pixels = int(image_summary_df["contaminated_pixels"].sum())
    summary: dict[str, object] = {
        "n_images": int(len(image_summary_df)),
        "total_pixels": total_pixels,
        "contaminated_pixels": contaminated_pixels,
        "contamination_pct_total_image": float(100.0 * contaminated_pixels / max(total_pixels, 1)),
    }
    for label in PUBLIC_TYPE_ORDER:
        slug = PUBLIC_TYPE_TO_KEY[label]
        area_px = int(image_summary_df[f"{slug}_area_px"].sum())
        summary[f"{slug}_area_px"] = area_px
        summary[f"{slug}_pct_total_image"] = float(100.0 * area_px / max(total_pixels, 1))
        summary[f"{slug}_pct_of_contamination"] = float(
            100.0 * area_px / max(contaminated_pixels, 1)
        )
    return summary


def _print_overall_summary(summary: dict[str, object]) -> None:
    print("Contamination summary")
    print(
        "  Total contamination: "
        f"{float(summary['contamination_pct_total_image']):.2f}% of analyzed micrograph area"
    )
    for label in PUBLIC_TYPE_ORDER:
        slug = PUBLIC_TYPE_TO_KEY[label]
        print(
            f"  {label}: {float(summary[f'{slug}_pct_of_contamination']):.2f}% of contamination "
            f"({float(summary[f'{slug}_pct_total_image']):.2f}% of total area)"
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _component_summary_df(component_rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    component_df = pd.DataFrame(component_rows)
    if component_df.empty:
        return pd.DataFrame(columns=COMPONENT_SUMMARY_COLUMNS)
    return _assign_types(component_df)


def _write_summary_outputs(
    *,
    output_dir: Path,
    bands_json: Path,
    args: argparse.Namespace,
    component_df: pd.DataFrame,
    manifest_rows: Sequence[pd.Series],
    expected_images: int,
    status: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    image_summary_df = pd.DataFrame(_image_summary_rows(component_df, manifest_rows)).sort_values("image_id").reset_index(drop=True)
    image_summary_csv = output_dir / "image_contamination_summary.csv"
    _write_csv_atomic(image_summary_df, image_summary_csv)

    dataset_summary_df = _dataset_summary_df(image_summary_df)
    dataset_summary_csv = output_dir / "dataset_contamination_summary.csv"
    _write_csv_atomic(dataset_summary_df, dataset_summary_csv)

    overall_summary = _overall_summary(image_summary_df)
    component_csv = output_dir / "component_type_assignments.csv"
    summary_payload = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "output_dir": str(output_dir),
        "bands_json": str(bands_json),
        "normalization_method": str(args.normalization_method),
        "pixel_size_override_angstrom": (None if args.pixel_size_angstrom is None else float(args.pixel_size_angstrom)),
        "min_component_area_px": int(args.min_component_area_px),
        "type_order": list(PUBLIC_TYPE_ORDER),
        "type_to_id": {label: int(PUBLIC_TYPE_TO_ID[label]) for label in PUBLIC_TYPE_ORDER},
        "overall_contamination_summary": overall_summary,
        "component_type_assignments_csv": str(component_csv),
        "image_contamination_summary_csv": str(image_summary_csv),
        "dataset_contamination_summary_csv": str(dataset_summary_csv),
        "typed_mask_dir": str(output_dir / "typed_masks"),
        "n_images": int(image_summary_df.shape[0]),
        "n_images_expected": int(expected_images),
        "n_components": int(component_df.shape[0]),
        "typing_status": str(status),
    }
    _write_json_atomic(output_dir / "summary.json", summary_payload)
    return image_summary_df, dataset_summary_df, overall_summary


def run(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    typed_mask_dir = output_dir / "typed_masks"
    typed_mask_dir.mkdir(parents=True, exist_ok=True)

    bands_json = args.bands_json.expanduser()
    if not bands_json.is_absolute():
        bands_json = (REPO_ROOT / bands_json).resolve()
    frequency_bands = _load_bands(bands_json)

    manifest_df = pd.read_csv(manifest_path).reset_index(drop=True)
    expected_images = (
        int(args.expected_images)
        if args.expected_images is not None and int(args.expected_images) > 0
        else int(len(manifest_df))
    )
    if "dataset_id" not in manifest_df.columns or "stem" not in manifest_df.columns:
        raise ValueError("Manifest must contain dataset_id and stem columns.")

    manifest_dir = manifest_path.parent
    component_rows: list[dict[str, object]] = []
    manifest_rows: list[pd.Series] = []

    for _, row in manifest_df.iterrows():
        micro_col = _first_existing_column(row, MICROGRAPH_PATH_COLUMNS)
        mask_col = _first_existing_column(row, MASK_PATH_COLUMNS)
        if micro_col is None or mask_col is None:
            raise ValueError(
                "Each manifest row must provide a micrograph path column from "
                f"{MICROGRAPH_PATH_COLUMNS} and a mask path column from {MASK_PATH_COLUMNS}."
            )

        dataset_id = str(row["dataset_id"])
        stem = str(row["stem"])
        image_id = f"{dataset_id}__{stem}"
        micro_path = _resolve_path(manifest_dir, row[micro_col])
        mask_path = _resolve_path(manifest_dir, row[mask_col])

        image, header_pixel_size = _load_micrograph(micro_path)
        mask = _load_mask(mask_path)
        if tuple(image.shape) != tuple(mask.shape):
            raise ValueError(
                f"Micrograph and mask shapes must match for {image_id}: "
                f"image={tuple(image.shape)} mask={tuple(mask.shape)}"
            )

        pixel_size_angstrom = _resolve_pixel_size(
            row,
            header_pixel_size=header_pixel_size,
            override_pixel_size=args.pixel_size_angstrom,
        )
        image_norm = normalize_image(image, method=args.normalization_method).astype(np.float32, copy=False)

        labels, nlab = ndi.label(mask.astype(bool), structure=np.ones((3, 3), dtype=bool))
        objects = ndi.find_objects(labels)
        total_contam_pixels = int(mask.sum())
        h, w = mask.shape
        edge_px = max(1, int(0.10 * min(h, w)))

        for comp_id, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            local_mask = np.asarray(labels[sl] == comp_id, dtype=bool)
            area_px = int(local_mask.sum())
            bbox_y0 = int(sl[0].start)
            bbox_y1 = int(sl[0].stop - 1)
            bbox_x0 = int(sl[1].start)
            bbox_x1 = int(sl[1].stop - 1)
            bbox_h = int(sl[0].stop - sl[0].start)
            bbox_w = int(sl[1].stop - sl[1].start)
            touches_border = int(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w)
            component_mask = np.asarray(labels == comp_id, dtype=bool)
            edge_artifact = int(
                _is_edge_artifact_component(
                    component_mask,
                    total_contam_pixels=total_contam_pixels,
                    edge_px=edge_px,
                )
            )
            eroded = ndi.binary_erosion(local_mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
            perimeter_px = float(np.count_nonzero(local_mask & (~eroded)))
            compactness = float(4.0 * np.pi * area_px / (perimeter_px * perimeter_px + 1e-6))
            bbox_fill = float(area_px / max(1, bbox_h * bbox_w))
            aspect_ratio = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))

            small_component_auto_ethane = bool(area_px < int(args.min_component_area_px))
            band_scores = [float("nan")] * 4
            if not small_component_auto_ethane:
                band_scores = _component_band_scores(
                    image_norm=image_norm,
                    bbox=(bbox_y0, bbox_y1, bbox_x0, bbox_x1),
                    pixel_size_angstrom=pixel_size_angstrom,
                    frequency_bands=frequency_bands,
                    crop_pad_px=int(args.crop_pad_px),
                    min_crop_size_px=int(args.min_crop_size_px),
                    max_psd_crop_size_px=int(args.max_psd_crop_size_px),
                )

            component_rows.append(
                {
                    "image_id": image_id,
                    "dataset_id": dataset_id,
                    "stem": stem,
                    "component_id": int(comp_id),
                    "area_px": int(area_px),
                    "pixel_size_angstrom": float(pixel_size_angstrom),
                    "bbox_y0": bbox_y0,
                    "bbox_y1": bbox_y1,
                    "bbox_x0": bbox_x0,
                    "bbox_x1": bbox_x1,
                    "bbox_h_px": bbox_h,
                    "bbox_w_px": bbox_w,
                    "touches_border": touches_border,
                    "edge_artifact": edge_artifact,
                    "compactness": compactness,
                    "bbox_fill_frac": bbox_fill,
                    "aspect_ratio": aspect_ratio,
                    "band_1_score": float(band_scores[0]),
                    "band_2_score": float(band_scores[1]),
                    "band_3_score": float(band_scores[2]),
                    "band_4_score": float(band_scores[3]),
                    "small_component_auto_ethane": bool(small_component_auto_ethane),
                    "total_pixels": int(h * w),
                    "contaminated_pixels": int(total_contam_pixels),
                }
            )
        row_with_geometry = row.copy()
        row_with_geometry["_total_pixels"] = int(h * w)
        row_with_geometry["_contaminated_pixels"] = int(total_contam_pixels)
        manifest_rows.append(row_with_geometry)
        partial_component_df = _component_summary_df(component_rows)
        _write_summary_outputs(
            output_dir=output_dir,
            bands_json=bands_json,
            args=args,
            component_df=partial_component_df,
            manifest_rows=manifest_rows,
            expected_images=expected_images,
            status="running",
        )
        print(f"Typed {len(manifest_rows)}/{expected_images} image(s); live summary updated.", flush=True)

    component_df = _component_summary_df(component_rows)
    component_csv = output_dir / "component_type_assignments.csv"
    _write_csv_atomic(component_df, component_csv)
    image_summary_df, dataset_summary_df, overall_summary = _write_summary_outputs(
        output_dir=output_dir,
        bands_json=bands_json,
        args=args,
        component_df=component_df,
        manifest_rows=manifest_rows,
        expected_images=expected_images,
        status="complete",
    )

    for key, row in manifest_df.groupby(["dataset_id", "stem"], sort=True):
        dataset_id, stem = key
        image_id = f"{dataset_id}__{stem}"
        mask_col = _first_existing_column(row.iloc[0], MASK_PATH_COLUMNS)
        mask = _load_mask(_resolve_path(manifest_dir, row.iloc[0][mask_col]))
        typed = _typed_mask_for_image(mask, component_df.loc[component_df["image_id"] == image_id].copy())
        np.save(typed_mask_dir / f"{image_id}_typed_mask.npy", typed.astype(np.uint8))

    _print_overall_summary(overall_summary)
    print(f"Wrote component table: {component_csv}")
    print(f"Wrote image summary:  {output_dir / 'image_contamination_summary.csv'}")
    print(f"Wrote dataset summary:{output_dir / 'dataset_contamination_summary.csv'}")
    print(f"Wrote typed masks:    {typed_mask_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))
