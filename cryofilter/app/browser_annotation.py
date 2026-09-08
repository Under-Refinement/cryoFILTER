"""Browser-native annotation session helpers for the local cryoFILTER app."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from utils.contamination_subtypes import (
    TYPE_ID_SEQUENCE,
    TYPE_ID_TO_DISPLAY,
    TYPE_SCHEMA_NAME,
    TYPE_SCHEMA_VERSION,
    UNLABELED_CONTAMINATION_LABEL,
    build_unlabeled_typed_mask,
    reconcile_typed_mask,
    typed_class_pixel_counts,
    typed_completion_stats,
)


IMAGE_PATH_COLUMNS = (
    "micrograph_path",
    "full_mic_path",
    "resolved_mic_path",
    "staged_mic_path",
    "mic_path",
    "source_path",
)
REQUIRED_EXPORT_FIELDS = (
    "gt_mask_path",
    "typed_label_path",
    "typed_label_metadata_path",
    "typed_label_schema_name",
    "typed_label_schema_version",
    "typed_label_unlabeled_value",
    "browser_annotation_path",
    "annotation_status",
    "annotated_at",
)
SESSION_FILENAME = "browser_session.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_browser_session(
    *,
    session_id: str,
    manifest_path: Path,
    run_dir: Path,
    source_mode: str,
    title: str,
    mic_dir: Path | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Annotation manifest does not exist: {manifest_path}")
    rows, _fieldnames = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Annotation manifest has no rows: {manifest_path}")

    timestamp = now or utc_now()
    meta: dict[str, Any] = {
        "id": str(session_id),
        "title": str(title),
        "status": "active",
        "source_mode": str(source_mode),
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "manifest_edited_path": str(run_dir / "manifest_edited.csv"),
        "mic_dir": str(mic_dir.expanduser().resolve()) if mic_dir is not None else "",
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    ensure_session_dirs(meta)
    write_session_file(meta)
    sync_manifest_edited(meta)
    return session_summary(meta)


def read_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is missing a header row: {manifest_path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def write_session_file(meta: dict[str, Any]) -> None:
    run_dir = Path(str(meta["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SESSION_FILENAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def ensure_session_dirs(meta: dict[str, Any]) -> None:
    run_dir = Path(str(meta["run_dir"]))
    for child in ("masks", "typed_labels", "typed_metadata", "browser_annotations", "browser_previews"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def session_summary(meta: dict[str, Any]) -> dict[str, Any]:
    rows, _fieldnames = read_manifest(Path(str(meta["manifest_path"])))
    progress = session_progress(meta, rows)
    split_counts: dict[str, int] = {}
    for row in rows:
        split = _optional_text(row.get("split") or row.get("split_label")).upper()
        if split:
            split_counts[split] = split_counts.get(split, 0) + 1
    cryosparc_source = _cryosparc_source_from_transfer_manifest(meta)
    payload = dict(meta)
    for key, value in cryosparc_source.items():
        payload.setdefault(key, value)
    payload.update(
        {
            "ok": True,
            "row_count": len(rows),
            **progress,
            "split_counts": split_counts,
            "type_options": type_options(),
        }
    )
    return payload


def _cryosparc_source_from_transfer_manifest(meta: dict[str, Any]) -> dict[str, str]:
    if _optional_text(meta.get("source_mode")).lower() != "cryosparc":
        return {}
    manifest_text = _optional_text(meta.get("transfer_manifest"))
    if not manifest_text:
        return {}
    manifest_path = Path(manifest_text).expanduser()
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    source: dict[str, str] = {}
    project_uid = _optional_text(data.get("project_uid"))
    workspace_uid = _optional_text(data.get("workspace_uid"))
    micrographs_ref = data.get("micrographs_ref") if isinstance(data.get("micrographs_ref"), dict) else {}
    job_uid = _optional_text(micrographs_ref.get("job_uid"))
    output_name = _optional_text(micrographs_ref.get("output_name")) or "micrographs"
    if project_uid:
        source["cryosparc_project"] = project_uid
    if workspace_uid:
        source["cryosparc_workspace"] = workspace_uid
    if job_uid:
        source["micrographs"] = f"{job_uid}:{output_name}"
    return source


def session_progress(
    meta: dict[str, Any],
    rows: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if rows is None:
        rows, _fieldnames = read_manifest(Path(str(meta["manifest_path"])))
    masks_dir = Path(str(meta["run_dir"])) / "masks"
    saved_count = 0
    first_unsaved_index: int | None = None
    last_saved_index: int | None = None
    for index, row in enumerate(rows):
        saved = (masks_dir / f"{row_key(row, index)}.npy").exists()
        if saved:
            saved_count += 1
            last_saved_index = index
        elif first_unsaved_index is None:
            first_unsaved_index = index
    if first_unsaved_index is not None:
        next_index = first_unsaved_index
    elif last_saved_index is not None:
        next_index = last_saved_index
    else:
        next_index = 0
    return {
        "saved_count": saved_count,
        "first_unsaved_index": first_unsaved_index,
        "last_saved_index": last_saved_index,
        "next_index": next_index,
        "is_complete": bool(rows) and saved_count == len(rows),
    }


def count_saved_rows(meta: dict[str, Any], rows: Sequence[dict[str, str]] | None = None) -> int:
    return int(session_progress(meta, rows)["saved_count"])



def type_options() -> list[dict[str, Any]]:
    return [{"id": None, "label": "Unlabeled"}] + [
        {"id": int(type_id), "label": TYPE_ID_TO_DISPLAY[int(type_id)]}
        for type_id in TYPE_ID_SEQUENCE
    ]


def get_row_payload(meta: dict[str, Any], index: int) -> dict[str, Any]:
    rows, _fieldnames = read_manifest(Path(str(meta["manifest_path"])))
    row = row_at(rows, index)
    image_path = resolve_micrograph_path(row, manifest_dir=Path(str(meta["manifest_path"])).parent, mic_dir=_meta_mic_dir(meta))
    height, width = image_shape(image_path)
    key = row_key(row, index)
    vector_path = annotation_path(meta, key)
    annotation = {}
    if vector_path.exists():
        annotation = json.loads(vector_path.read_text(encoding="utf-8"))
    saved = mask_path(meta, key).exists()
    return {
        "ok": True,
        "index": int(index),
        "row_count": len(rows),
        "key": key,
        "dataset_id": _optional_text(row.get("dataset_id")),
        "stem": _row_stem(row, index),
        "split": _optional_text(row.get("split") or row.get("split_label")),
        "micrograph_path": str(image_path),
        "source_width": int(width),
        "source_height": int(height),
        "image_url": f"/api/annotation/sessions/{meta['id']}/rows/{int(index)}/image",
        "saved": saved,
        "annotation": annotation,
        "mask_path": str(mask_path(meta, key)) if saved else "",
    }


def get_preview_png(
    meta: dict[str, Any],
    index: int,
    *,
    max_dim: int = 1600,
) -> tuple[bytes, dict[str, Any]]:
    rows, _fieldnames = read_manifest(Path(str(meta["manifest_path"])))
    row = row_at(rows, index)
    image_path = resolve_micrograph_path(row, manifest_dir=Path(str(meta["manifest_path"])).parent, mic_dir=_meta_mic_dir(meta))
    key = row_key(row, index)
    max_dim = max(16, min(int(max_dim), 4096))
    png_path = Path(str(meta["run_dir"])) / "browser_previews" / f"{int(index):06d}_{key}_{max_dim}.png"
    info_path = png_path.with_suffix(".json")
    stat = image_path.stat()
    cache_key = {
        "source_path": str(image_path),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
        "max_dim": int(max_dim),
    }
    if png_path.exists() and info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if all(info.get(key_) == value for key_, value in cache_key.items()):
                return png_path.read_bytes(), info
        except Exception:
            pass

    array = load_micrograph_array(image_path)
    height, width = array.shape
    image = normalize_to_image(array)
    scale = min(1.0, float(max_dim) / float(max(width, height)))
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    if (preview_width, preview_height) != (width, height):
        image = image.resize((preview_width, preview_height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()
    info = {
        **cache_key,
        "source_width": int(width),
        "source_height": int(height),
        "preview_width": int(preview_width),
        "preview_height": int(preview_height),
    }
    png_path.write_bytes(payload)
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return payload, info


def save_row_annotations(
    meta: dict[str, Any],
    index: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rows, _fieldnames = read_manifest(Path(str(meta["manifest_path"])))
    row = row_at(rows, index)
    image_path = resolve_micrograph_path(row, manifest_dir=Path(str(meta["manifest_path"])).parent, mic_dir=_meta_mic_dir(meta))
    height, width = image_shape(image_path)
    key = row_key(row, index)
    polygons = _validated_polygons(payload.get("polygons"), width=width, height=height)
    binary_mask, typed_mask = rasterize_polygons(polygons, width=width, height=height)

    ensure_session_dirs(meta)
    out_mask = mask_path(meta, key)
    out_typed = typed_label_path(meta, key)
    out_meta = typed_metadata_path(meta, key)
    out_annotation = annotation_path(meta, key)
    np.save(out_mask, binary_mask.astype(np.uint8))
    np.save(out_typed, typed_mask.astype(np.uint8))

    stats = typed_completion_stats(binary_mask, typed_mask)
    counts = typed_class_pixel_counts(binary_mask, typed_mask)
    vector_payload = {
        "dataset_id": _optional_text(row.get("dataset_id")),
        "stem": _row_stem(row, index),
        "key": key,
        "image_width": int(width),
        "image_height": int(height),
        "polygons": polygons,
        "updated_at": utc_now(),
    }
    out_annotation.write_text(json.dumps(vector_payload, indent=2) + "\n", encoding="utf-8")
    out_meta.write_text(
        json.dumps(
            {
                "dataset_id": _optional_text(row.get("dataset_id")),
                "stem": _row_stem(row, index),
                "key": key,
                "binary_mask_path": path_for_manifest(out_mask, Path(str(meta["manifest_path"])).parent),
                "typed_label_path": path_for_manifest(out_typed, Path(str(meta["manifest_path"])).parent),
                "typed_label_schema_name": TYPE_SCHEMA_NAME,
                "typed_label_schema_version": int(TYPE_SCHEMA_VERSION),
                "typed_label_unlabeled_value": int(UNLABELED_CONTAMINATION_LABEL),
                "type_id_to_name": {str(type_id): TYPE_ID_TO_DISPLAY[type_id] for type_id in TYPE_ID_SEQUENCE},
                "counts": counts,
                "stats": stats,
                "current_split_label": _optional_text(row.get("split") or row.get("split_label")),
                "browser_annotation_path": path_for_manifest(out_annotation, Path(str(meta["manifest_path"])).parent),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    meta["updated_at"] = utc_now()
    write_session_file(meta)
    sync_manifest_edited(meta)
    summary = session_summary(meta)
    summary.update(
        {
            "index": int(index),
            "key": key,
            "mask_path": str(out_mask),
            "typed_label_path": str(out_typed),
            "typed_label_metadata_path": str(out_meta),
            "browser_annotation_path": str(out_annotation),
            "stats": stats,
            "counts": counts,
        }
    )
    return summary


def finalize_session(meta: dict[str, Any]) -> dict[str, Any]:
    meta["status"] = "ready"
    meta["updated_at"] = utc_now()
    write_session_file(meta)
    sync_manifest_edited(meta)
    return session_summary(meta)


def sync_manifest_edited(meta: dict[str, Any]) -> Path:
    manifest_path = Path(str(meta["manifest_path"]))
    manifest_dir = manifest_path.parent
    rows, fieldnames = read_manifest(manifest_path)
    export_fields = list(fieldnames)
    for field in REQUIRED_EXPORT_FIELDS:
        if field not in export_fields:
            export_fields.append(field)
    for index, row in enumerate(rows):
        for field in export_fields:
            row.setdefault(field, "")
        key = row_key(row, index)
        out_mask = mask_path(meta, key)
        out_typed = typed_label_path(meta, key)
        out_meta = typed_metadata_path(meta, key)
        out_annotation = annotation_path(meta, key)
        if out_mask.exists():
            row["gt_mask_path"] = path_for_manifest(out_mask, manifest_dir)
            row["typed_label_path"] = path_for_manifest(out_typed, manifest_dir) if out_typed.exists() else ""
            row["typed_label_metadata_path"] = path_for_manifest(out_meta, manifest_dir) if out_meta.exists() else ""
            row["typed_label_schema_name"] = TYPE_SCHEMA_NAME
            row["typed_label_schema_version"] = str(TYPE_SCHEMA_VERSION)
            row["typed_label_unlabeled_value"] = str(UNLABELED_CONTAMINATION_LABEL)
            row["browser_annotation_path"] = (
                path_for_manifest(out_annotation, manifest_dir) if out_annotation.exists() else ""
            )
            row["annotation_status"] = "saved"
            if out_annotation.exists():
                try:
                    row["annotated_at"] = json.loads(out_annotation.read_text(encoding="utf-8")).get(
                        "updated_at",
                        "",
                    )
                except Exception:
                    row["annotated_at"] = ""
        else:
            row["gt_mask_path"] = _optional_text(row.get("gt_mask_path"))
            row["typed_label_path"] = _optional_text(row.get("typed_label_path"))
            row["typed_label_metadata_path"] = _optional_text(row.get("typed_label_metadata_path"))
            row["browser_annotation_path"] = _optional_text(row.get("browser_annotation_path"))
            row["annotation_status"] = _optional_text(row.get("annotation_status"))
            row["annotated_at"] = _optional_text(row.get("annotated_at"))

    output_path = Path(str(meta["manifest_edited_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=export_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(output_path)
    return output_path


def row_at(rows: Sequence[dict[str, str]], index: int) -> dict[str, str]:
    index = int(index)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Annotation row index out of range: {index}")
    return rows[index]


def row_key(row: dict[str, str], index: int) -> str:
    dataset = _optional_text(row.get("dataset_id")) or "dataset"
    stem = _row_stem(row, index)
    return f"{_slug(dataset)}__{_slug(stem)}"


def resolve_micrograph_path(
    row: dict[str, str],
    *,
    manifest_dir: Path,
    mic_dir: Path | None = None,
) -> Path:
    candidates: list[Path] = []
    for column in IMAGE_PATH_COLUMNS:
        text = _optional_text(row.get(column))
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            if mic_dir is not None:
                candidates.extend([mic_dir / path, mic_dir / path.name])
            candidates.extend([manifest_dir / path, Path.cwd() / path, path])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates[:4])
    suffix = f" Tried: {tried}" if tried else ""
    raise FileNotFoundError(f"Could not resolve micrograph path for row {_row_stem(row, 0)}.{suffix}")


def image_shape(path: Path) -> tuple[int, int]:
    array = load_micrograph_array(path)
    return int(array.shape[0]), int(array.shape[1])


def load_micrograph_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".mrc", ".mrcs"}:
        try:
            import mrcfile
        except ImportError as exc:
            raise ImportError("Reading MRC micrographs requires `mrcfile`. Install cryoFILTER dependencies.") from exc
        with mrcfile.open(path, permissive=True) as handle:
            array = np.asarray(handle.data).copy()
    elif suffix == ".npy":
        array = np.load(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("F"))
    array = np.squeeze(np.asarray(array))
    while array.ndim > 2:
        array = array[0]
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D micrograph image, got shape {array.shape} from {path}")
    return np.asarray(array, dtype=np.float32)


def normalize_to_image(array: np.ndarray) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        scaled = np.zeros(values.shape, dtype=np.uint8)
        return Image.fromarray(scaled, mode="L")
    clean = np.where(finite, values, np.nanmedian(values[finite]))
    lo, hi = np.percentile(clean, [1.0, 99.5])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo = float(np.min(clean))
        hi = float(np.max(clean))
    if hi <= lo:
        scaled = np.zeros(clean.shape, dtype=np.uint8)
    else:
        scaled = np.clip((clean - lo) / (hi - lo), 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def rasterize_polygons(
    polygons: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    binary_image = Image.new("L", (int(width), int(height)), 0)
    typed_image = Image.new("L", (int(width), int(height)), 0)
    binary_draw = ImageDraw.Draw(binary_image)
    typed_draw = ImageDraw.Draw(typed_image)
    for polygon in polygons:
        points = [(float(x), float(y)) for x, y in polygon.get("points", [])]
        if len(points) < 3:
            continue
        mode = _optional_text(polygon.get("mode")).lower() or "add"
        if mode == "erase":
            binary_draw.polygon(points, fill=0)
            typed_draw.polygon(points, fill=0)
            continue
        type_id = polygon.get("type_id")
        label = int(type_id) if type_id in TYPE_ID_SEQUENCE else int(UNLABELED_CONTAMINATION_LABEL)
        binary_draw.polygon(points, fill=1)
        typed_draw.polygon(points, fill=label)
    binary = np.asarray(binary_image, dtype=np.uint8)
    typed = np.asarray(typed_image, dtype=np.uint8)
    if np.any(binary):
        typed = build_unlabeled_typed_mask(binary)
        typed_image = Image.fromarray(typed, mode="L")
        typed_draw = ImageDraw.Draw(typed_image)
        for polygon in polygons:
            points = [(float(x), float(y)) for x, y in polygon.get("points", [])]
            if len(points) < 3:
                continue
            mode = _optional_text(polygon.get("mode")).lower() or "add"
            type_id = polygon.get("type_id")
            if mode == "erase":
                continue
            if type_id in TYPE_ID_SEQUENCE:
                typed_draw.polygon(points, fill=int(type_id))
        typed = np.asarray(typed_image, dtype=np.uint8)
        typed = reconcile_typed_mask(binary, typed)
    return binary, typed


def path_for_manifest(path: Path, manifest_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except Exception:
        return str(path)


def mask_path(meta: dict[str, Any], key: str) -> Path:
    return Path(str(meta["run_dir"])) / "masks" / f"{key}.npy"


def typed_label_path(meta: dict[str, Any], key: str) -> Path:
    return Path(str(meta["run_dir"])) / "typed_labels" / f"{key}.npy"


def typed_metadata_path(meta: dict[str, Any], key: str) -> Path:
    return Path(str(meta["run_dir"])) / "typed_metadata" / f"{key}.json"


def annotation_path(meta: dict[str, Any], key: str) -> Path:
    return Path(str(meta["run_dir"])) / "browser_annotations" / f"{key}.json"


def _validated_polygons(value: object, *, width: int, height: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("polygons must be a list")
    polygons: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_points = item.get("points")
        if not isinstance(raw_points, list):
            continue
        points: list[list[float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = min(max(float(point[0]), 0.0), float(width - 1))
                y = min(max(float(point[1]), 0.0), float(height - 1))
            except Exception:
                continue
            points.append([x, y])
        if len(points) < 3:
            continue
        raw_type = item.get("type_id")
        try:
            parsed_type = int(raw_type) if _optional_text(raw_type) else None
        except Exception:
            parsed_type = None
        type_id = parsed_type if parsed_type in TYPE_ID_SEQUENCE else None
        shape = _optional_text(item.get("shape")).lower()
        if shape not in {"polygon", "ellipse"}:
            shape = "polygon"
        polygon: dict[str, Any] = {
            "points": points,
            "type_id": type_id,
            "mode": "erase" if _optional_text(item.get("mode")).lower() == "erase" else "add",
            "shape": shape,
        }
        bounds = _validated_bounds(item.get("bounds"), width=width, height=height)
        if bounds is not None:
            polygon["bounds"] = bounds
        rotation = _validated_rotation(item.get("rotation"))
        if shape == "ellipse" and rotation is not None:
            polygon["rotation"] = rotation
        polygons.append(polygon)
    return polygons


def _validated_rotation(value: object) -> float | None:
    try:
        rotation = float(value)
    except Exception:
        return None
    if not math.isfinite(rotation):
        return None
    return ((rotation + math.pi) % (2.0 * math.pi)) - math.pi


def _validated_bounds(value: object, *, width: int, height: int) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, float] = {}
    for key, limit in (("x0", width), ("x1", width), ("y0", height), ("y1", height)):
        try:
            coordinate = float(value.get(key))
        except Exception:
            return None
        if not math.isfinite(coordinate):
            return None
        parsed[key] = min(max(coordinate, 0.0), float(limit - 1))
    x0, x1 = sorted([parsed["x0"], parsed["x1"]])
    y0, y1 = sorted([parsed["y0"], parsed["y1"]])
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _row_stem(row: dict[str, str], index: int) -> str:
    stem = _optional_text(row.get("stem"))
    if stem:
        return stem
    for column in IMAGE_PATH_COLUMNS:
        text = _optional_text(row.get(column))
        if text:
            return Path(text).stem
    return f"row_{int(index):06d}"


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "item"


def _optional_text(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _meta_mic_dir(meta: dict[str, Any]) -> Path | None:
    text = _optional_text(meta.get("mic_dir"))
    return Path(text).expanduser().resolve() if text else None
