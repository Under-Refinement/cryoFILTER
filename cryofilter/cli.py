"""Command-line interface for cryoFILTER."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from cryofilter.app.server import add_subparser as add_app_subparser
from cryofilter.cryosparc.remote.cli import add_subparser as add_cryosparc_subparser
from cryofilter.typing_cli import add_subparser as add_type_subparser
from cryofilter.typing_cli import run as run_type
from cryofilter.training import add_subparser as add_train_subparser
from cryofilter.training import run as run_train
from cryofilter.gui import add_subparser as add_gui_subparser
from cryofilter.gui import run as run_gui
from cryofilter.install_cryosparc_tools import add_subparser as add_install_cryosparc_tools_subparser
from utils.small_pixel_policy import (
    DEFAULT_INFERENCE_PROFILE,
    DEFAULT_SMALL_PIXEL_CUTOFF_ANGSTROM,
    SMALL_PIXEL_PROFILE_CONFIGS,
    resolve_small_pixel_inference_policy,
)

DEFAULT_MASK_THRESHOLD = 0.60
DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM = 100.0
DEFAULT_PUBLIC_CHECKPOINT_RELATIVE = Path("pretrained_models") / "cryoFILTER_FULL.pt"
DEFAULT_OUTPUT_DIR_SUFFIX = "_cryofilter_output"
DEFAULT_PUBLIC_TARGET_PIXEL_SIZE = 2.0
DEFAULT_PUBLIC_OVERLAP = 160
DEFAULT_PUBLIC_NORMALIZATION_METHOD = "percentile_extra_wide"
DEFAULT_PUBLIC_MULTISCALE_PSD_SOURCE = "patch"
DEFAULT_PUBLIC_BLENDING_WINDOW = "tukey"
DEFAULT_PUBLIC_BLENDING_EDGE_PX = 16
DEFAULT_PUBLIC_ADAPTIVE_DOWNSAMPLE_METHOD = "fourier"
DEFAULT_PARTICLE_OVERLAY_SUBDIR = "OTF_images"
DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM = 1800
DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX = 34.0
DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS = 1
DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE = 1100
DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY = 1
DEFAULT_PARTICLE_OVERLAY_WARN_REMOVED_FRACTION = 0.25
DEFAULT_INTERNAL_MAP_OUTPUT_SUBDIR = ".cryofilter_internal_maps"
TYPING_MANIFEST_FILENAME = "contamination_typing_manifest.csv"
TYPING_MANIFEST_FIELDS = (
    "dataset_id",
    "stem",
    "micrograph_path",
    "binary_mask_path",
    "pixel_size_angstrom",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_public_checkpoint_path() -> Path:
    return (_repo_root() / DEFAULT_PUBLIC_CHECKPOINT_RELATIVE).resolve()


def _default_output_dir_for_input(input_path: Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    stem = path.stem if path.suffix else path.name
    if path.exists() and path.is_dir():
        stem = path.name
    if not stem:
        stem = "cryofilter_input"
    return (path.parent / f"{stem}{DEFAULT_OUTPUT_DIR_SUFFIX}").resolve()


def _resolve_checkpoint_path(checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg).expanduser().resolve()

    default_path = _default_public_checkpoint_path()
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "No checkpoint was provided and the default public checkpoint was not found at "
        f"{default_path}. Download cryoFILTER_FULL.pt into pretrained_models/ or pass --checkpoint."
    )


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _parse_psd_scales(text: str) -> tuple[int, ...]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--psd-scales must contain at least one integer value")
    return tuple(values)


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _resolve_auto_batch_forward_size(gpu_name: str, total_mem_gb: Optional[float]) -> int:
    name_u = str(gpu_name or "").upper()
    fast_name_tokens = ("A100", "H100", "H200", "RTX 4090", "GEFORCE RTX 4090", "RTX 5090", "GEFORCE RTX 5090")
    if any(token in name_u for token in fast_name_tokens):
        return 32
    if total_mem_gb is not None and np.isfinite(float(total_mem_gb)):
        mem = float(total_mem_gb)
        if mem >= 23.0:
            return 32
        if mem >= 15.0:
            return 16
    return 8


def _collect_mrc_paths(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mrc":
            raise ValueError(f"Input file must have .mrc extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*.mrc" if recursive else "*.mrc"
    paths = sorted(input_path.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .mrc files found in {input_path}")
    return paths


def _resolve_inference_devices(raw_devices: str) -> list[str]:
    """Resolve simple GPU indices, CUDA device strings, or ``all``."""
    raw = str(raw_devices or "").strip()
    if not raw:
        return []

    import torch

    if raw.lower() == "all":
        count = int(torch.cuda.device_count())
        if count < 1:
            raise RuntimeError("--gpus all was requested, but no CUDA GPUs are visible")
        return [f"cuda:{index}" for index in range(count)]

    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("--gpus must be 'all' or a comma-separated list such as 0,1,2,3")
    visible_count = int(torch.cuda.device_count())
    devices: list[str] = []
    for item in requested:
        device = f"cuda:{int(item)}" if item.isdigit() else item
        parsed = torch.device(device)
        if parsed.type != "cuda":
            raise ValueError(f"--gpus only accepts GPU indices, got {item!r}")
        index = 0 if parsed.index is None else int(parsed.index)
        if index < 0 or index >= visible_count:
            raise ValueError(
                f"GPU {index} is not visible; torch reports {visible_count} GPU(s)"
            )
        devices.append(f"cuda:{index}")
    if len(set(devices)) != len(devices):
        raise ValueError("--gpus contains duplicate GPU indices")
    return devices


def _parse_optional_resource_int(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _apply_infer_resource_args(args: argparse.Namespace) -> dict[str, int | None]:
    """Apply friendly CPU/GPU count aliases before resolving public GPU shards."""

    num_cpus = _parse_optional_resource_int(
        getattr(args, "num_cpus", None) or os.environ.get("CRYOFILTER_NUM_CPUS"),
        name="--num-cpus",
        minimum=1,
    )
    if num_cpus is not None:
        value = str(num_cpus)
        os.environ["CRYOFILTER_NUM_CPUS"] = value
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[env_name] = value

    num_gpus = _parse_optional_resource_int(
        getattr(args, "num_gpus", None) or os.environ.get("CRYOFILTER_NUM_GPUS"),
        name="--num-gpus",
        minimum=0,
    )
    if num_gpus is not None:
        os.environ["CRYOFILTER_NUM_GPUS"] = str(num_gpus)
        if not str(getattr(args, "gpus", "") or "").strip() and num_gpus > 0:
            args.gpus = ",".join(str(index) for index in range(num_gpus))
        if num_gpus == 0 and str(getattr(args, "device", "")).startswith("cuda"):
            raise RuntimeError("CUDA inference was requested, but --num-gpus is 0.")
    return {"num_cpus": num_cpus, "num_gpus": num_gpus}


def _shard_mrc_paths(mrc_paths: Sequence[Path], worker_count: int) -> list[list[Path]]:
    """Round-robin micrographs so similarly named acquisitions stay balanced."""
    if int(worker_count) < 1:
        raise ValueError("worker_count must be at least 1")
    return [list(mrc_paths[index::worker_count]) for index in range(worker_count)]


def _merge_worker_summaries(
    worker_summaries: Sequence[dict[str, object]],
    mrc_paths: Sequence[Path],
) -> dict[str, object]:
    if not worker_summaries:
        raise ValueError("No multi-GPU worker summaries were produced")

    rows_by_path: dict[Path, dict[str, object]] = {}
    for worker_summary in worker_summaries:
        for raw_row in worker_summary.get("inputs", []):
            row = dict(raw_row)
            resolved = Path(str(row["input_mrc"])).expanduser().resolve()
            if resolved in rows_by_path:
                raise RuntimeError(f"Duplicate multi-GPU inference result for {resolved}")
            rows_by_path[resolved] = row

    ordered_rows: list[dict[str, object]] = []
    for path in mrc_paths:
        resolved = path.expanduser().resolve()
        if resolved not in rows_by_path:
            raise RuntimeError(f"Missing multi-GPU inference result for {resolved}")
        ordered_rows.append(rows_by_path[resolved])

    summary = dict(worker_summaries[0])
    summary["inputs"] = ordered_rows
    overlay_summaries = [
        item.get("particle_overlay_rendering")
        for item in worker_summaries
        if isinstance(item.get("particle_overlay_rendering"), dict)
        and item.get("particle_overlay_rendering", {}).get("enabled")
    ]
    if overlay_summaries:
        merged_overlay = dict(overlay_summaries[0])
        frames_by_key: dict[str, dict[str, object]] = {}
        for overlay in overlay_summaries:
            for raw_frame in overlay.get("frames", []):
                if not isinstance(raw_frame, dict):
                    continue
                frame = dict(raw_frame)
                key = str(frame.get("output_png") or frame.get("micrograph") or "")
                if key:
                    frames_by_key[key] = frame

        ordered_frames: list[dict[str, object]] = []
        seen: set[str] = set()
        for row in ordered_rows:
            particle_overlay = row.get("particle_overlay")
            if not isinstance(particle_overlay, dict):
                continue
            frame = {"micrograph": row.get("input_mrc"), **particle_overlay}
            key = str(frame.get("output_png") or frame.get("micrograph") or "")
            if not key or key in seen:
                continue
            ordered_frames.append({**frames_by_key.get(key, {}), **frame})
            seen.add(key)
        for key, frame in frames_by_key.items():
            if key not in seen:
                ordered_frames.append(frame)
        merged_overlay["frames"] = ordered_frames
        summary["particle_overlay_rendering"] = merged_overlay
    return summary


def _load_mrc_2d(path: Path) -> tuple[np.ndarray, Optional[float]]:
    import mrcfile

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
        # Some files store a stack; use the first slice for 2D micrograph inference.
        image = data[0]
    else:
        raise ValueError(f"Unsupported MRC dimensionality for {path}: shape={data.shape}")

    return image.astype(np.float32, copy=False), pixel_size


def _load_mrc_pixel_size(path: Path) -> Optional[float]:
    """Read only the MRC header when inference outputs are being resumed."""
    import mrcfile

    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        try:
            px = float(mrc.voxel_size.x)
        except Exception:
            return None
    return px if np.isfinite(px) and px > 0 else None


def _typing_dataset_id(mrc_path: Path, input_path: Path) -> str:
    """Create a stable, filesystem-safe dataset name for a typing manifest row."""
    if input_path.is_file():
        return input_path.parent.name or "dataset"

    root_name = input_path.name or "dataset"
    try:
        relative_parent = mrc_path.parent.relative_to(input_path)
    except ValueError:
        return mrc_path.parent.name or root_name
    return "__".join((root_name, *relative_parent.parts))


def _write_typing_manifest(
    output_dir: Path,
    input_path: Path,
    inference_rows: Sequence[dict[str, object]],
    *,
    pixel_size_override_angstrom: Optional[float] = None,
) -> Optional[Path]:
    """Write generated micrograph/mask pairs in the format used by ``cryofilter type``."""
    manifest_rows: list[dict[str, object]] = []
    for row in inference_rows:
        if not bool(row.get("generated_outputs")):
            continue
        raw_micrograph = row.get("input_mrc")
        raw_mask = row.get("output_mask_npy")
        if not raw_micrograph or not raw_mask:
            continue

        micrograph_path = Path(str(raw_micrograph)).expanduser().resolve()
        mask_path = Path(str(raw_mask)).expanduser().resolve()
        pixel_size = pixel_size_override_angstrom
        if pixel_size is None:
            raw_pixel_size = row.get("pixel_size_angstrom")
            try:
                candidate = float(raw_pixel_size)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                candidate = float("nan")
            if np.isfinite(candidate) and candidate > 0:
                pixel_size = candidate
            else:
                pixel_size = _load_mrc_pixel_size(micrograph_path)

        manifest_rows.append(
            {
                "dataset_id": _typing_dataset_id(micrograph_path, input_path),
                "stem": micrograph_path.stem,
                "micrograph_path": str(micrograph_path),
                "binary_mask_path": str(mask_path),
                "pixel_size_angstrom": "" if pixel_size is None else float(pixel_size),
            }
        )

    if not manifest_rows:
        return None

    manifest_path = output_dir / TYPING_MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TYPING_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def _save_mrc(path: Path, array: np.ndarray, voxel_size_angstrom: Optional[float]) -> None:
    import mrcfile

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(array.astype(np.float32, copy=False))
        if voxel_size_angstrom is not None and voxel_size_angstrom > 0:
            mrc.voxel_size = voxel_size_angstrom


def _extract_state_dict(checkpoint: object) -> dict:
    import torch

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    if isinstance(checkpoint, dict):
        raise ValueError("Checkpoint dictionary does not contain a recognized state_dict key")
    raise ValueError("Checkpoint format is not supported")


def _load_checkpoint(path: Path, map_location: str) -> object:
    import torch

    # Our project checkpoints are trusted local artifacts and may include
    # metadata objects that require legacy torch.load behavior.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _detect_model_type(state_dict: dict) -> str:
    keys = tuple(state_dict.keys())
    has_stem = any("stem." in k for k in keys)
    has_stages = any("stages." in k for k in keys)
    has_conv1 = any("conv1.weight" in k for k in keys)
    has_layer1 = any("layer1" in k and "dec" not in k for k in keys)
    has_enc1 = any("enc1" in k for k in keys)
    has_attention = any(("attention" in k.lower()) or ("transformer" in k.lower()) for k in keys)

    if has_stem and has_stages:
        raise ValueError("Checkpoint architecture is not supported by this public release")
    if has_conv1 and has_layer1:
        return "resnet_unet"
    if has_enc1 and has_attention:
        return "unet_attention"
    if has_enc1:
        return "unet"
    return "simple"


def _detect_attention_type(state_dict: dict) -> str:
    keys = tuple(state_dict.keys())
    has_dual = any("local_transformer" in k or "global_transformer" in k for k in keys)
    has_attention = any(("attention" in k.lower()) or ("transformer" in k.lower()) for k in keys)
    if has_dual:
        return "dual"
    if has_attention:
        return "global"
    return "none"


def _detect_input_channels(state_dict: dict) -> Optional[int]:
    import torch

    priority_suffixes = (
        "enc1.0.weight",
        "conv1.weight",
        "features.0.weight",
        "stem.0.weight",
    )
    for suffix in priority_suffixes:
        for key, tensor in state_dict.items():
            if key.endswith(suffix) and isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                return int(tensor.shape[1])

    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            return int(tensor.shape[1])
    return None


def _infer_num_classes(state_dict: dict) -> int:
    import torch

    for key in ("final.weight", "binary_head.2.weight", "binary_head.1.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return 2


def _infer_decoder_dropout(state_dict: dict) -> float:
    import torch

    tensor = state_dict.get("dec4.4.weight")
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
        # Checkpoint layout includes a dropout layer at dec*.3 (no parameters),
        # shifting the second conv to index 4.
        return 0.1
    return 0.0


def _resolve_use_psd(mode: str, checkpoint: object, state_dict: dict) -> bool:
    if mode == "true":
        return True
    if mode == "false":
        return False

    if isinstance(checkpoint, dict):
        maybe_flag = checkpoint.get("use_power_spectrum")
        if isinstance(maybe_flag, bool):
            return maybe_flag

    channels = _detect_input_channels(state_dict)
    if channels == 1:
        return False
    if channels == 2:
        return True
    return True


def _resolve_psd_multiscale_separate_channels(
    mode: Optional[bool],
    checkpoint: object,
    state_dict: dict,
    use_psd: bool,
) -> bool:
    if not use_psd:
        return False
    if mode is not None:
        return bool(mode)

    if isinstance(checkpoint, dict):
        maybe_flag = checkpoint.get("psd_multiscale_separate_channels")
        if isinstance(maybe_flag, bool):
            return maybe_flag

    channels = _detect_input_channels(state_dict)
    return bool(channels is not None and channels > 2)


def _resolve_model_type(mode: str, checkpoint: object, state_dict: dict) -> str:
    if mode != "auto":
        return mode
    if isinstance(checkpoint, dict):
        maybe_type = checkpoint.get("model_type")
        if isinstance(maybe_type, str) and maybe_type:
            return maybe_type
    return _detect_model_type(state_dict)


def _resolve_attention_type(mode: str, checkpoint: object, state_dict: dict) -> str:
    if mode != "auto":
        return mode
    if isinstance(checkpoint, dict):
        maybe_type = checkpoint.get("attention_type")
        if isinstance(maybe_type, str) and maybe_type:
            return maybe_type
    return _detect_attention_type(state_dict)


def _resize_to_shape(arr: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    from scipy.ndimage import zoom

    if tuple(arr.shape) == tuple(target_shape):
        return np.asarray(arr, dtype=np.float32)
    zoom_factors = (float(target_shape[0]) / float(arr.shape[0]), float(target_shape[1]) / float(arr.shape[1]))
    resized = zoom(arr, zoom_factors, order=order)
    out = np.asarray(resized, dtype=np.float32)
    if tuple(out.shape) != tuple(target_shape):
        fixed = np.zeros(target_shape, dtype=np.float32)
        h = min(target_shape[0], out.shape[0])
        w = min(target_shape[1], out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return out


def _maybe_downsample(
    image: np.ndarray,
    pixel_size_angstrom: Optional[float],
    adaptive: bool,
    target_pixel_size: float,
    max_factor: Optional[float] = None,
    min_work_dim: int = 0,
    method: str = "bilinear",
) -> tuple[np.ndarray, bool, dict]:
    from scipy.ndimage import zoom

    info = {
        "target_pixel_size": float(target_pixel_size),
        "requested_factor": None,
        "applied_factor": 1.0,
        "max_factor": (None if max_factor is None else float(max_factor)),
        "max_factor_applied": False,
        "min_work_dim": int(max(0, min_work_dim)),
        "min_work_dim_applied": False,
        "downsample_method": str(method),
        "downsample_reason": None,
    }

    if not adaptive:
        info["downsample_reason"] = "adaptive_disabled"
        return image, False, info
    if pixel_size_angstrom is None or pixel_size_angstrom <= 0:
        info["downsample_reason"] = "missing_or_invalid_pixel_size"
        return image, False, info

    factor = float(target_pixel_size) / float(pixel_size_angstrom)
    info["requested_factor"] = float(factor)

    if max_factor is not None and float(max_factor) > 1.0 and factor > float(max_factor):
        factor = float(max_factor)
        info["max_factor_applied"] = True

    if factor <= 1.05:
        info["downsample_reason"] = "factor_below_threshold"
        return image, False, info

    h, w = image.shape
    new_h = max(1, int(round(h / factor)))
    new_w = max(1, int(round(w / factor)))

    min_dim = int(max(0, min_work_dim))
    if min_dim > 0 and min(new_h, new_w) < min_dim and min(h, w) > min_dim:
        scale_up = float(min_dim) / float(min(new_h, new_w))
        guarded_h = min(h, max(1, int(round(new_h * scale_up))))
        guarded_w = min(w, max(1, int(round(new_w * scale_up))))
        if guarded_h > new_h or guarded_w > new_w:
            info["min_work_dim_applied"] = True
            new_h, new_w = guarded_h, guarded_w

    if new_h >= h and new_w >= w:
        info["downsample_reason"] = "guarded_to_input_shape"
        return image, False, info

    sy = float(new_h) / float(h)
    sx = float(new_w) / float(w)

    method_l = str(method).strip().lower()
    if method_l == "fourier":
        from utils.fourier_rescale import fourier_rescale_2d

        s = float((sy + sx) * 0.5)
        resized = fourier_rescale_2d(image, s)
        if tuple(resized.shape) != (new_h, new_w):
            resized = _resize_to_shape(np.asarray(resized), (new_h, new_w), order=1)
        else:
            resized = np.asarray(resized, dtype=np.float32)
    else:
        resized = zoom(image, (sy, sx), order=1).astype(np.float32, copy=False)

    info["applied_factor"] = float((float(h) / float(new_h) + float(w) / float(new_w)) * 0.5)
    return resized.astype(np.float32, copy=False), True, info


def _disk_structure(radius_px: int) -> np.ndarray:
    radius = int(max(0, radius_px))
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= (radius * radius)


def _remove_small_mask_components(mask: np.ndarray, min_area_pixels: int) -> np.ndarray:
    from scipy.ndimage import label

    mask_bool = np.asarray(mask, dtype=bool)
    min_area = int(max(0, min_area_pixels))
    if min_area <= 1 or not mask_bool.any():
        return mask_bool

    labels, num_labels = label(mask_bool, structure=np.ones((3, 3), dtype=bool))
    if num_labels <= 0:
        return mask_bool
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_area
    return keep[labels]


def _fill_small_mask_holes(mask: np.ndarray, max_area_pixels: int) -> tuple[np.ndarray, int]:
    from scipy.ndimage import label

    mask_bool = np.asarray(mask, dtype=bool).copy()
    max_area = int(max(0, max_area_pixels))
    if max_area <= 0 or mask_bool.all():
        return mask_bool, 0

    holes = ~mask_bool
    labels, num_labels = label(holes, structure=np.ones((3, 3), dtype=bool))
    if num_labels <= 0:
        return mask_bool, 0

    border = np.zeros_like(mask_bool, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_labels = set(int(v) for v in np.unique(labels[border]).tolist())
    sizes = np.bincount(labels.ravel(), minlength=num_labels + 1)

    filled_pixels = 0
    for label_id in range(1, num_labels + 1):
        if label_id in border_labels:
            continue
        area = int(sizes[label_id])
        if area <= max_area:
            component = labels == label_id
            mask_bool[component] = True
            filled_pixels += area
    return mask_bool, int(filled_pixels)


def _apply_hysteresis_mask(
    prob_map: np.ndarray,
    high_threshold: float,
    mode: str,
    low_delta: float,
    low_min: float,
    auto_max_growth_ratio: float,
    auto_min_added_mean_prob_margin: float,
) -> tuple[np.ndarray, dict]:
    from scipy.ndimage import binary_fill_holes, binary_propagation

    prob = np.asarray(prob_map, dtype=np.float32)
    high = float(high_threshold)
    low = max(float(low_min), high - float(low_delta))
    mask_base = prob >= high
    mode_l = str(mode).strip().lower()
    meta = {
        "mode": mode_l,
        "high_threshold": high,
        "low_threshold": low,
        "low_delta": float(low_delta),
        "low_min": float(low_min),
        "auto_max_growth_ratio": float(auto_max_growth_ratio),
        "auto_min_added_mean_prob_margin": float(auto_min_added_mean_prob_margin),
        "applied": False,
        "reason": "off",
        "base_pixels": int(mask_base.sum()),
        "grown_pixels": int(mask_base.sum()),
        "added_pixels": 0,
        "added_growth_ratio": 0.0,
        "added_mean_probability": None,
    }

    if mode_l == "off":
        return mask_base, meta
    if mode_l not in {"on", "auto"}:
        raise ValueError("--hysteresis-mode must be one of: off, on, auto")
    if low >= high:
        meta["reason"] = "low_threshold_not_below_high_threshold"
        return mask_base, meta
    if not mask_base.any():
        meta["reason"] = "no_seed_pixels"
        return mask_base, meta

    terrain = prob >= low
    grown_for_gates = binary_propagation(mask_base, mask=terrain, structure=np.ones((3, 3), dtype=bool))
    added = grown_for_gates & ~mask_base
    added_pixels = int(added.sum())
    base_pixels = int(mask_base.sum())
    growth_ratio = float(added_pixels / max(base_pixels, 1))
    added_mean = float(prob[added].mean()) if added_pixels > 0 else None
    grown = binary_fill_holes(grown_for_gates).astype(bool)
    meta.update(
        {
            "grown_pixels": int(grown.sum()),
            "grown_pixels_before_hole_fill": int(grown_for_gates.sum()),
            "added_pixels": added_pixels,
            "added_growth_ratio": growth_ratio,
            "added_mean_probability": added_mean,
        }
    )

    if added_pixels <= 0:
        meta["reason"] = "no_added_pixels"
        return mask_base, meta
    if mode_l == "auto":
        max_growth = float(auto_max_growth_ratio)
        if max_growth >= 0.0 and growth_ratio > max_growth:
            meta["reason"] = "auto_rejected_growth_ratio"
            return mask_base, meta
        min_added_mean = low + float(auto_min_added_mean_prob_margin)
        if added_mean is not None and (added_mean + 1e-6) < min_added_mean:
            meta["reason"] = "auto_rejected_added_mean_probability"
            return mask_base, meta

    meta["applied"] = True
    meta["reason"] = "accepted"
    return grown, meta


def _postprocess_inference_mask(
    prob_map: np.ndarray,
    threshold: float,
    hysteresis_mode: str,
    hysteresis_low_delta: float,
    hysteresis_low_min: float,
    hysteresis_auto_max_growth_ratio: float,
    hysteresis_auto_min_added_mean_prob_margin: float,
    min_area_pixels: int,
    fill_small_holes_max_area_pixels: int,
    mask_buffer_distance_angstrom: float,
    output_pixel_size_angstrom: Optional[float],
    fill_small_holes_after_buffer_max_area_pixels: int,
) -> tuple[np.ndarray, dict]:
    from scipy.ndimage import binary_dilation

    mask, hysteresis_meta = _apply_hysteresis_mask(
        prob_map=prob_map,
        high_threshold=float(threshold),
        mode=str(hysteresis_mode),
        low_delta=float(hysteresis_low_delta),
        low_min=float(hysteresis_low_min),
        auto_max_growth_ratio=float(hysteresis_auto_max_growth_ratio),
        auto_min_added_mean_prob_margin=float(hysteresis_auto_min_added_mean_prob_margin),
    )
    before_min_area = int(mask.sum())
    mask = _remove_small_mask_components(mask, int(min_area_pixels))
    after_min_area = int(mask.sum())
    mask, filled_before = _fill_small_mask_holes(mask, int(fill_small_holes_max_area_pixels))

    buffer_radius_px = 0
    buffer_distance = float(mask_buffer_distance_angstrom)
    if (
        buffer_distance > 0.0
        and output_pixel_size_angstrom is not None
        and float(output_pixel_size_angstrom) > 0.0
    ):
        buffer_radius_px = int(round(buffer_distance / float(output_pixel_size_angstrom)))
        if buffer_radius_px > 0:
            mask = binary_dilation(mask, structure=_disk_structure(buffer_radius_px)).astype(bool)

    mask, filled_after = _fill_small_mask_holes(mask, int(fill_small_holes_after_buffer_max_area_pixels))
    meta = {
        "threshold": float(threshold),
        "hysteresis": hysteresis_meta,
        "min_area_pixels": int(max(0, min_area_pixels)),
        "pixels_before_min_area": before_min_area,
        "pixels_after_min_area": after_min_area,
        "fill_small_holes_max_area_pixels": int(max(0, fill_small_holes_max_area_pixels)),
        "filled_hole_pixels_before_buffer": int(filled_before),
        "mask_buffer_distance_angstrom": buffer_distance,
        "mask_buffer_radius_pixels": int(buffer_radius_px),
        "fill_small_holes_after_buffer_max_area_pixels": int(max(0, fill_small_holes_after_buffer_max_area_pixels)),
        "filled_hole_pixels_after_buffer": int(filled_after),
        "final_mask_pixels": int(mask.sum()),
        "final_mask_fraction": float(mask.mean()) if mask.size else 0.0,
    }
    return mask.astype(np.uint8), meta


def _resolve_render_particle_overlays(args: argparse.Namespace) -> bool:
    return bool(args.particle_file) if args.render_particle_overlays is None else bool(args.render_particle_overlays)


def _particle_overlay_contact_sheet_tile(args: argparse.Namespace) -> int | tuple[int, int]:
    tile = int(args.particle_overlay_contact_sheet_tile)
    if bool(args.particle_overlay_raw_panel) or bool(args.particle_overlay_probability_panel):
        return (0, tile)
    return tile


def _create_particle_overlay_context(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mrc_paths: Sequence[Path],
) -> dict[str, object]:
    from cryofilter.particle_filtering import collect_particle_coordinates_by_micrograph

    overlay_dir = (
        Path(args.particle_overlay_dir).expanduser().resolve()
        if args.particle_overlay_dir
        else (output_dir / DEFAULT_PARTICLE_OVERLAY_SUBDIR).resolve()
    )
    overlay_dir.mkdir(parents=True, exist_ok=True)
    particle_index = collect_particle_coordinates_by_micrograph(
        particle_file=args.particle_file,
        micrograph_paths=mrc_paths,
        pixel_size_override_angstrom=args.pixel_size_angstrom,
        csg_file=args.particle_csg,
        allow_unmatched=bool(args.allow_unmatched_particles),
    )
    contact_sheet_path = overlay_dir / "particle_overlay_contact_sheet.png"
    context = {
        "output_dir": overlay_dir,
        "contact_sheet_path": contact_sheet_path,
        "particle_index": particle_index,
        "frame_paths": [],
        "frame_borders": [],
        "frames": [],
    }
    context["summary"] = {
        "enabled": True,
        "output_dir": str(overlay_dir),
        "contact_sheet": bool(args.particle_overlay_contact_sheet),
        "contact_sheet_path": str(contact_sheet_path),
        "max_display_dim": int(args.particle_overlay_max_display_dim),
        "particle_diameter_px": float(args.particle_overlay_diameter_px),
        "mask_alpha": float(args.particle_overlay_mask_alpha),
        "raw_reference_panel": bool(args.particle_overlay_raw_panel),
        "probability_panel": bool(args.particle_overlay_probability_panel),
        "warn_removed_fraction": float(args.particle_overlay_warn_removed_fraction),
        "particle_file_index": {
            key: value
            for key, value in particle_index.items()
            if key != "coordinates_by_micrograph"
        },
        "frames": context["frames"],
    }
    print(
        "Particle overlay rendering enabled: "
        f"{particle_index['particles_matched']}/{particle_index['particles_input']} matched particles; "
        f"writing PNGs to {overlay_dir}",
        flush=True,
    )
    return context


def _render_particle_overlay_for_micrograph(
    *,
    context: dict[str, object],
    args: argparse.Namespace,
    mrc_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    prob_map: np.ndarray,
    pixel_size_angstrom: Optional[float],
) -> dict[str, object]:
    from cryofilter.particle_filtering import classify_particle_coordinates_by_mask
    from cryofilter.qc_render import (
        GOOD_BORDER_RGB,
        WARN_BORDER_RGB,
        montage_from_paths,
        render_particle_overlay_png,
    )

    particle_index = context["particle_index"]
    coords_by_micrograph = particle_index["coordinates_by_micrograph"]
    coords = coords_by_micrograph.get(Path(mrc_path).expanduser().resolve())
    if coords is None:
        coords = np.empty((0, 2), dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)

    if len(coords) > 0:
        if pixel_size_angstrom is None or not np.isfinite(float(pixel_size_angstrom)):
            raise ValueError(
                f"Cannot render particle overlay for {mrc_path.name}: missing pixel size. "
                "Repair the MRC header or pass --pixel-size-angstrom."
            )
        keep, distances = classify_particle_coordinates_by_mask(
            coords,
            bad_mask=mask,
            input_shape=tuple(int(v) for v in image.shape[:2]),
            pixel_size_angstrom=float(pixel_size_angstrom),
            exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
        )
    else:
        keep = np.zeros(0, dtype=bool)
        distances = np.empty(0, dtype=np.float32)

    n_total = int(len(coords))
    n_kept = int(np.count_nonzero(keep))
    n_removed = int(n_total - n_kept)
    removed_fraction = float(n_removed / max(n_total, 1))
    border = (
        WARN_BORDER_RGB
        if removed_fraction >= float(args.particle_overlay_warn_removed_fraction)
        else GOOD_BORDER_RGB
    )
    output_path = context["output_dir"] / f"{mrc_path.stem}_particle_overlay.png"
    label = f"{mrc_path.name}   kept {n_kept} / rejected {n_removed}"
    render_particle_overlay_png(
        image=image,
        mask=mask,
        coords_xy=coords,
        keep=keep,
        output_path=output_path,
        probability_map=prob_map,
        label=label,
        max_display_dim=int(args.particle_overlay_max_display_dim),
        particle_diameter_px=float(args.particle_overlay_diameter_px),
        mask_alpha=float(args.particle_overlay_mask_alpha),
        include_raw_panel=bool(args.particle_overlay_raw_panel),
        include_probability_panel=bool(args.particle_overlay_probability_panel),
    )

    record = {
        "output_png": str(output_path),
        "particles": n_total,
        "particles_kept": n_kept,
        "particles_removed": n_removed,
        "removed_fraction": removed_fraction,
        "min_distance_angstrom": (
            None if len(distances) == 0 else float(np.nanmin(distances))
        ),
        "median_distance_angstrom": (
            None if len(distances) == 0 else float(np.nanmedian(distances))
        ),
    }
    context["frame_paths"].append(output_path)
    context["frame_borders"].append(border)
    context["frames"].append({"micrograph": str(mrc_path), **record})

    every = int(args.particle_overlay_contact_sheet_every)
    if bool(args.particle_overlay_contact_sheet) and every > 0:
        if len(context["frame_paths"]) % every == 0:
            sheet = montage_from_paths(
                context["frame_paths"],
                output_path=context["contact_sheet_path"],
                cols=int(args.particle_overlay_contact_sheet_cols),
                tile=_particle_overlay_contact_sheet_tile(args),
                borders=context["frame_borders"],
            )
            record["contact_sheet_updated"] = str(sheet)

    print(
        f"  particle overlay: kept {n_kept}/{n_total}, rejected {n_removed}; {output_path}",
        flush=True,
    )
    return record


def _finalize_particle_overlay_contact_sheet(
    context: dict[str, object],
    args: argparse.Namespace,
) -> Optional[Path]:
    if not bool(args.particle_overlay_contact_sheet) or not context["frame_paths"]:
        return None
    from cryofilter.qc_render import montage_from_paths

    final_sheet = montage_from_paths(
        context["frame_paths"],
        output_path=context["contact_sheet_path"],
        cols=int(args.particle_overlay_contact_sheet_cols),
        tile=_particle_overlay_contact_sheet_tile(args),
        borders=context["frame_borders"],
    )
    context["summary"]["final_contact_sheet_path"] = str(final_sheet)
    print(f"Particle overlay contact sheet: {final_sheet}", flush=True)
    return final_sheet


def _render_particle_overlays_from_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    mrc_paths: Sequence[Path],
    summary: dict[str, object],
) -> None:
    if not _resolve_render_particle_overlays(args):
        summary.setdefault("particle_overlay_rendering", {"enabled": False})
        return
    existing = summary.get("particle_overlay_rendering")
    if isinstance(existing, dict) and existing.get("enabled") and existing.get("frames"):
        rows = summary.get("inputs", [])
        expected_count = (
            sum(
                1
                for row in rows
                if isinstance(row, dict)
                and row.get("input_mrc")
                and row.get("output_prob_npy")
                and row.get("output_mask_npy")
            )
            if isinstance(rows, list)
            else len(mrc_paths)
        )
        frame_paths = [
            path
            for frame in existing.get("frames", [])
            if isinstance(frame, dict)
            and frame.get("output_png")
            for path in [Path(str(frame.get("output_png"))).expanduser()]
            if path.exists()
        ]
        if len(frame_paths) >= expected_count:
            if bool(args.particle_overlay_contact_sheet):
                overlay_dir = (
                    Path(str(existing.get("output_dir"))).expanduser().resolve()
                    if existing.get("output_dir")
                    else (output_dir / DEFAULT_PARTICLE_OVERLAY_SUBDIR).resolve()
                )
                context = {
                    "output_dir": overlay_dir,
                    "contact_sheet_path": overlay_dir / "particle_overlay_contact_sheet.png",
                    "frame_paths": frame_paths,
                    "frame_borders": [None] * len(frame_paths),
                    "summary": existing,
                }
                existing["contact_sheet"] = True
                existing["contact_sheet_path"] = str(context["contact_sheet_path"])
                _finalize_particle_overlay_contact_sheet(context, args)
            return

    context = _create_particle_overlay_context(args, output_dir=output_dir, mrc_paths=mrc_paths)
    summary["particle_overlay_rendering"] = context["summary"]
    rows = summary.get("inputs", [])
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        prob_path = row.get("output_prob_npy")
        mask_path = row.get("output_mask_npy")
        input_mrc = row.get("input_mrc")
        if not prob_path or not mask_path or not input_mrc:
            continue
        prob_file = Path(str(prob_path)).expanduser()
        mask_file = Path(str(mask_path)).expanduser()
        if not prob_file.is_file() or not mask_file.is_file():
            continue
        mrc_path = Path(str(input_mrc)).expanduser().resolve()
        image, header_pixel_size = _load_mrc_2d(mrc_path)
        row_pixel_size = row.get("pixel_size_angstrom")
        try:
            candidate_pixel_size = float(row_pixel_size)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            candidate_pixel_size = float("nan")
        pixel_size = (
            args.pixel_size_angstrom
            if args.pixel_size_angstrom is not None
            else (candidate_pixel_size if np.isfinite(candidate_pixel_size) and candidate_pixel_size > 0 else header_pixel_size)
        )
        row["particle_overlay"] = _render_particle_overlay_for_micrograph(
            context=context,
            args=args,
            mrc_path=mrc_path,
            image=image,
            mask=np.load(mask_file),
            prob_map=np.load(prob_file),
            pixel_size_angstrom=pixel_size,
        )
    _finalize_particle_overlay_contact_sheet(context, args)


def _finalize_inference_run(
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_dir: Path,
    mrc_paths: Sequence[Path],
    summary: dict[str, object],
) -> int:
    from cryofilter.particle_filtering import default_filtered_particle_path, filter_particle_file

    if args.particle_file:
        particle_input = Path(args.particle_file).expanduser().resolve()
        mask_dir_raw = summary.get("mask_output_dir")
        mask_dir = (
            Path(str(mask_dir_raw)).expanduser().resolve()
            if mask_dir_raw
            else output_dir
        )
        particle_output = (
            Path(args.filtered_particle_file).expanduser().resolve()
            if args.filtered_particle_file
            else default_filtered_particle_path(particle_input, output_dir).resolve()
        )
        particle_filter_summary = filter_particle_file(
            particle_file=particle_input,
            output_file=particle_output,
            micrograph_paths=mrc_paths,
            mask_dir=mask_dir,
            exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
            pixel_size_override_angstrom=args.pixel_size_angstrom,
            csg_file=args.particle_csg,
            allow_unmatched=bool(args.allow_unmatched_particles),
            overwrite=bool(args.overwrite_filtered_particles),
        )
        particle_summary_path = output_dir / "particle_filter_summary.json"
        particle_filter_summary["summary_file"] = str(particle_summary_path)
        with open(particle_summary_path, "w", encoding="utf-8") as handle:
            json.dump(particle_filter_summary, handle, indent=2)
        summary["particle_filtering"] = particle_filter_summary
        print(
            "Particle filtering: "
            f"kept {particle_filter_summary['particles_kept']}/"
            f"{particle_filter_summary['particles_input']} "
            f"({particle_filter_summary['particles_removed']} removed)",
            flush=True,
        )
        print(f"Filtered particles: {particle_output}", flush=True)
        if particle_filter_summary.get("output_csg_file"):
            print(f"Filtered CryoSPARC group: {particle_filter_summary['output_csg_file']}", flush=True)
        extra_files = [
            path
            for path in particle_filter_summary.get("output_particle_files", [])
            if path != particle_filter_summary["output_particle_file"]
        ]
        if extra_files:
            print("Additional filtered .cs files:", flush=True)
            for path in extra_files:
                print(f"  {path}", flush=True)
        print(f"Particle summary: {particle_summary_path}", flush=True)

    _render_particle_overlays_from_summary(
        args=args,
        output_dir=output_dir,
        mrc_paths=mrc_paths,
        summary=summary,
    )

    typing_manifest_path = _write_typing_manifest(
        output_dir,
        input_path,
        summary["inputs"],
        pixel_size_override_angstrom=args.pixel_size_angstrom,
    )
    summary["typing_manifest"] = None if typing_manifest_path is None else str(typing_manifest_path)

    summary_path = output_dir / "inference_summary.json"
    _write_inference_summary(summary_path, summary)

    print(f"Finished inference on {len(mrc_paths)} micrograph(s).", flush=True)
    if typing_manifest_path is not None:
        print(f"Typing manifest: {typing_manifest_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


def _write_inference_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _run_infer(
    args: argparse.Namespace,
    *,
    mrc_paths_override: Optional[Sequence[Path]] = None,
    finalize: bool = True,
    live_summary_path: Optional[Path] = None,
) -> int | dict[str, object]:
    import torch
    from models.bad_region_detector import create_model
    from models.predict import (
        _attach_model_input_metadata,
        _resolve_checkpoint_model_config,
        predict_bad_regions_probability,
    )

    if args.overlap >= args.patch_size:
        raise ValueError("--overlap must be smaller than --patch-size")
    if not args.particle_file and (args.filtered_particle_file or args.particle_csg):
        raise ValueError("--filtered-particle-file and --particle-csg require --particle-file")
    render_particle_overlays = _resolve_render_particle_overlays(args)
    if render_particle_overlays and not args.particle_file:
        raise ValueError("--render-particle-overlays requires --particle-file")
    if args.particle_file and args.no_resample_or_gen:
        raise ValueError(
            "Particle filtering requires final mask outputs; do not combine --particle-file "
            "with --no-resample-or-gen."
        )
    if args.image_output_dir and not args.render_images:
        raise ValueError("--image-output-dir requires --render-images")
    if args.render_images and args.no_resample_or_gen:
        raise ValueError("--render-images cannot be combined with --no-resample-or-gen")
    if args.render_images and int(args.image_max_dim) < 8:
        raise ValueError("--image-max-dim must be at least 8")
    if args.render_images and int(args.image_dpi) < 1:
        raise ValueError("--image-dpi must be at least 1")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir_for_input(input_path)
    )
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    image_output_dir = (
        Path(args.image_output_dir).expanduser().resolve()
        if args.image_output_dir
        else output_dir / "diagnostic_images"
    )
    export_mask_outputs = bool(getattr(args, "export_masks", True))
    force_disk_map_outputs = bool(getattr(args, "force_disk_mask_outputs", False))
    needs_disk_map_outputs = bool(
        export_mask_outputs or args.particle_file or force_disk_map_outputs
    )
    map_output_dir = (
        output_dir
        if export_mask_outputs
        else output_dir / DEFAULT_INTERNAL_MAP_OUTPUT_SUBDIR
    )
    save_mrc_outputs = bool(args.save_mrc and export_mask_outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    if needs_disk_map_outputs:
        map_output_dir.mkdir(parents=True, exist_ok=True)
    if args.render_images:
        image_output_dir.mkdir(parents=True, exist_ok=True)
    mrc_paths = (
        list(mrc_paths_override)
        if mrc_paths_override is not None
        else _collect_mrc_paths(input_path, recursive=args.recursive)
    )
    if not mrc_paths:
        raise ValueError("No micrographs were assigned to this inference worker")

    particle_overlay_context: Optional[dict[str, object]] = None
    if render_particle_overlays:
        particle_overlay_context = _create_particle_overlay_context(
            args,
            output_dir=output_dir,
            mrc_paths=mrc_paths,
        )

    checkpoint = _load_checkpoint(checkpoint_path, map_location=args.device)
    state_dict = _extract_state_dict(checkpoint)
    checkpoint_config = _resolve_checkpoint_model_config(checkpoint)

    model_type = _resolve_model_type(args.model_type, checkpoint, state_dict)
    attention_type = _resolve_attention_type(args.attention_type, checkpoint, state_dict)
    use_psd = _resolve_use_psd(args.use_psd, checkpoint, state_dict)
    checkpoint_use_psd = bool(checkpoint_config.get("use_power_spectrum", use_psd))
    checkpoint_attention_type = str(checkpoint_config.get("attention_type", attention_type))
    checkpoint_model_type = str(checkpoint_config.get("model_type", model_type))
    if args.use_psd != "auto" and use_psd != checkpoint_use_psd:
        raise ValueError(
            "This checkpoint has a fixed PSD input layout. Leave --use-psd at auto and "
            "choose the matching ablation checkpoint instead."
        )
    if args.attention_type != "auto" and attention_type != checkpoint_attention_type:
        raise ValueError(
            "This checkpoint has a fixed attention layout. Leave --attention-type at auto and "
            "choose the matching ablation checkpoint instead."
        )
    if args.model_type != "auto" and model_type != checkpoint_model_type:
        raise ValueError(
            "This checkpoint has a fixed architecture. Leave --model-type at auto and "
            "choose the matching checkpoint instead."
        )

    psd_multiscale = bool(checkpoint_config.get("psd_multiscale", False)) and use_psd
    psd_frequency_band_channels = bool(checkpoint_config.get("psd_frequency_band_channels", False)) and use_psd
    psd_frequency_bands = tuple(
        (float(lo), float(hi)) for lo, hi in tuple(checkpoint_config.get("psd_frequency_bands", ()))
    )
    psd_frequency_band_include_anisotropy = bool(
        checkpoint_config.get("psd_frequency_band_include_anisotropy", False)
    ) and use_psd
    psd_frequency_band_anisotropy_min_freq = float(
        checkpoint_config.get("psd_frequency_band_anisotropy_min_freq", 0.0)
    )
    if psd_multiscale:
        psd_multiscale_separate_channels = _resolve_psd_multiscale_separate_channels(
            args.psd_multiscale_separate_channels,
            checkpoint,
            state_dict,
            use_psd,
        )
    else:
        psd_multiscale_separate_channels = False

    detected_input_channels = int(checkpoint_config.get("input_channels", 0) or 0)
    if detected_input_channels <= 0:
        maybe_channels = _detect_input_channels(state_dict)
        detected_input_channels = int(maybe_channels) if maybe_channels is not None else 0
    num_classes = int(checkpoint_config.get("num_classes", _infer_num_classes(state_dict)))
    decoder_dropout = float(checkpoint_config.get("decoder_dropout", _infer_decoder_dropout(state_dict)))
    psd_scales = _parse_psd_scales(args.psd_scales)
    if use_psd:
        if detected_input_channels > 0:
            resolved_input_channels = int(detected_input_channels)
        else:
            resolved_input_channels = 1 + (len(psd_scales) if psd_multiscale_separate_channels else 1)
    else:
        resolved_input_channels = 1
    multiscale_targets = _parse_float_list(args.multi_scale_target_pixel_sizes)
    if not multiscale_targets:
        multiscale_targets = [float(args.target_pixel_size)]
    if any((not np.isfinite(v) or v <= 0.0) for v in multiscale_targets):
        raise ValueError("All target pixel sizes must be finite and > 0")
    if len(multiscale_targets) > 1 and bool(args.no_resample or args.no_resample_or_gen):
        raise ValueError("Multi-scale target fusion requires resampling outputs to a common shape; disable --no-resample/--no-resample-or-gen")
    if len(multiscale_targets) > 1 and (not bool(args.adaptive_downsample)):
        raise ValueError("Multi-scale target fusion requires --adaptive-downsample")

    standard_use_tta = bool(args.use_tta or int(args.tta) > 1)

    adaptive_max_factor: Optional[float] = None
    if float(args.adaptive_max_factor) > 1.0:
        adaptive_max_factor = float(args.adaptive_max_factor)

    def _resolve_policy(pixel_size_angstrom: Optional[float]):
        return resolve_small_pixel_inference_policy(
            pixel_size_angstrom=pixel_size_angstrom,
            standard_inference_recipe="adaptive_resample",
            standard_target_pixel_size=float(args.target_pixel_size),
            standard_multiscale_targets=multiscale_targets,
            standard_normalization_method=str(args.normalization_method),
            standard_overlap=int(args.overlap),
            standard_adaptive_downsample=bool(args.adaptive_downsample),
            standard_adaptive_downsample_method=str(args.adaptive_downsample_method),
            standard_multiscale_psd_source=str(args.multiscale_psd_source),
            standard_blending_window=str(args.blending_window),
            standard_blending_edge_px=int(args.blending_edge_px),
            standard_use_tta=bool(standard_use_tta),
            inference_profile=str(args.inference_profile),
            auto_enabled=bool(args.small_pixel_auto),
            cutoff_angstrom=float(args.small_pixel_cutoff_angstrom),
        )

    print(f"Using checkpoint={checkpoint_path}", flush=True)
    print(f"Using output_dir={output_dir}", flush=True)
    if export_mask_outputs:
        print(f"Exporting mask/probability arrays to {map_output_dir}", flush=True)
    elif needs_disk_map_outputs:
        print(
            "Visible mask/probability export disabled; keeping internal arrays for "
            f"downstream filtering/typing in {map_output_dir}",
            flush=True,
        )
    else:
        print("Visible mask/probability export disabled", flush=True)
    print(f"Using device={args.device}", flush=True)
    print(f"Using model_type={model_type}", flush=True)
    print(f"Using attention_type={attention_type}", flush=True)
    print(f"Using PSD channel={use_psd}", flush=True)
    if not use_psd:
        psd_mode = "none"
    elif psd_frequency_band_channels:
        psd_mode = "frequency_band"
    elif psd_multiscale:
        psd_mode = "multiscale"
    else:
        psd_mode = "single_channel"
    print(f"Using PSD mode={psd_mode}", flush=True)
    if psd_frequency_band_channels:
        print(f"Using frequency bands={psd_frequency_bands}", flush=True)
    print(f"Using separate multiscale PSD channels={psd_multiscale_separate_channels}", flush=True)
    print(f"Using input_channels={resolved_input_channels}", flush=True)
    print(f"Using num_classes={num_classes}", flush=True)
    if model_type == "unet_attention":
        print(f"Using decoder_dropout={decoder_dropout}", flush=True)
    if len(multiscale_targets) > 1:
        print(f"Using multi-scale target fusion (A/px): {multiscale_targets}", flush=True)
    if standard_use_tta:
        print("Standard TTA requested: enabled (4-view identity/flip/rotate ensemble)", flush=True)
    print(
        f"Small-pixel auto={bool(args.small_pixel_auto)} "
        f"(cutoff={float(args.small_pixel_cutoff_angstrom):.3f} A/px, "
        f"profile={args.inference_profile})",
        flush=True,
    )
    print(
        "Mask post-processing defaults: "
        f"threshold={float(args.threshold):.2f}, "
        f"hysteresis={args.hysteresis_mode}, "
        f"min_area={int(args.min_area_pixels)} px, "
        f"hole_fill={int(args.fill_small_holes_max_area_pixels)}/"
        f"{int(args.fill_small_holes_after_buffer_max_area_pixels)} px, "
        f"mask_buffer={float(args.mask_buffer_distance_angstrom):.1f} A, "
        f"particle_exclusion={float(args.particle_exclusion_distance_angstrom):.1f} A"
        f", skip_existing={bool(args.skip_existing)}",
        flush=True,
    )

    model = create_model(
        model_type=model_type,
        device=args.device,
        use_power_spectrum=use_psd,
        num_classes=num_classes,
        norm_type=str(checkpoint_config.get("norm_type") or args.norm_type),
        decoder_dropout=decoder_dropout,
        attention_type=attention_type,
        input_channels_override=resolved_input_channels,
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys while loading checkpoint.")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys while loading checkpoint.")
    runtime_model_config = dict(checkpoint_config)
    runtime_model_config.update(
        {
            "use_power_spectrum": bool(use_psd),
            "input_channels": int(resolved_input_channels),
            "psd_multiscale": bool(psd_multiscale),
            "psd_multiscale_separate_channels": bool(psd_multiscale_separate_channels),
            "psd_frequency_band_channels": bool(psd_frequency_band_channels),
            "psd_frequency_bands": tuple(psd_frequency_bands),
            "psd_frequency_band_include_anisotropy": bool(psd_frequency_band_include_anisotropy),
            "psd_frequency_band_anisotropy_min_freq": float(psd_frequency_band_anisotropy_min_freq),
        }
    )
    _attach_model_input_metadata(model, runtime_model_config)
    model.eval()

    resolved_batch_forward_size = 1
    if int(args.batch_forward_size) > 0:
        resolved_batch_forward_size = int(args.batch_forward_size)
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = ""
        total_mem_gb = None
        parsed_device = torch.device(args.device)
        gpu_index = torch.cuda.current_device() if parsed_device.index is None else int(parsed_device.index)
        try:
            gpu_name = str(torch.cuda.get_device_name(gpu_index))
        except Exception:
            gpu_name = ""
        try:
            props = torch.cuda.get_device_properties(gpu_index)
            total_mem_gb = float(props.total_memory) / (1024 ** 3)
        except Exception:
            total_mem_gb = None

        resolved_batch_forward_size = _resolve_auto_batch_forward_size(gpu_name, total_mem_gb)

        print(
            f"Auto batch_forward_size={resolved_batch_forward_size} "
            f"(gpu={gpu_name or 'unknown'}, mem_gb={total_mem_gb if total_mem_gb is not None else 'unknown'})",
            flush=True,
        )
    else:
        resolved_batch_forward_size = 1

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": args.device,
        "model_type": model_type,
        "attention_type": attention_type,
        "use_psd": use_psd,
        "psd_mode": psd_mode,
        "psd_multiscale": bool(psd_multiscale),
        "psd_multiscale_separate_channels": bool(psd_multiscale_separate_channels),
        "psd_frequency_band_channels": bool(psd_frequency_band_channels),
        "psd_frequency_bands": [list(b) for b in psd_frequency_bands],
        "detected_input_channels": (None if detected_input_channels <= 0 else int(detected_input_channels)),
        "resolved_input_channels": int(resolved_input_channels),
        "batch_forward_size_requested": int(args.batch_forward_size),
        "batch_forward_size_resolved": int(resolved_batch_forward_size),
        "standard_use_tta_requested": bool(standard_use_tta),
        "tta_argument": int(args.tta),
        "target_pixel_size": float(args.target_pixel_size),
        "multi_scale_target_pixel_sizes": [float(v) for v in multiscale_targets],
        "inference_profile": str(args.inference_profile),
        "small_pixel_auto": bool(args.small_pixel_auto),
        "small_pixel_cutoff_angstrom": float(args.small_pixel_cutoff_angstrom),
        "adaptive_downsample": bool(args.adaptive_downsample),
        "adaptive_max_factor": (None if adaptive_max_factor is None else float(adaptive_max_factor)),
        "adaptive_min_work_dim": int(max(0, args.adaptive_min_work_dim)),
        "adaptive_downsample_method": str(args.adaptive_downsample_method),
        "multiscale_psd_source": str(args.multiscale_psd_source),
        "blending_window": str(args.blending_window),
        "blending_edge_px": int(args.blending_edge_px),
        "stitch_phase_mode": str(args.stitch_phase_mode),
        "stitch_phase_reduction": str(args.stitch_phase_reduction),
        "stitch_phase_blend_alpha": float(args.stitch_phase_blend_alpha),
        "stitch_output_space": str(args.stitch_output_space),
        "stitch_output_blend_alpha": float(args.stitch_output_blend_alpha),
        "stitch_output_preserve_threshold": float(args.stitch_output_preserve_threshold),
        "stitch_output_transition": float(args.stitch_output_transition),
        "patch_prob_floor_quantile": float(args.patch_prob_floor_quantile),
        "patch_prob_floor_strength": float(args.patch_prob_floor_strength),
        "normalization_method": str(args.normalization_method),
        "mask_postprocessing_defaults": {
            "threshold": float(args.threshold),
            "hysteresis_mode": str(args.hysteresis_mode),
            "hysteresis_low_delta": float(args.hysteresis_low_delta),
            "hysteresis_low_min": float(args.hysteresis_low_min),
            "hysteresis_auto_max_growth_ratio": float(args.hysteresis_auto_max_growth_ratio),
            "hysteresis_auto_min_added_mean_prob_margin": float(args.hysteresis_auto_min_added_mean_prob_margin),
            "min_area_pixels": int(max(0, args.min_area_pixels)),
            "fill_small_holes_max_area_pixels": int(max(0, args.fill_small_holes_max_area_pixels)),
            "mask_buffer_distance_angstrom": float(args.mask_buffer_distance_angstrom),
            "fill_small_holes_after_buffer_max_area_pixels": int(
                max(0, args.fill_small_holes_after_buffer_max_area_pixels)
            ),
            "particle_exclusion_distance_angstrom": float(args.particle_exclusion_distance_angstrom),
        },
        "no_resample": bool(args.no_resample),
        "no_resample_or_gen": bool(args.no_resample_or_gen),
        "export_masks": bool(export_mask_outputs),
        "mask_outputs_exported": bool(export_mask_outputs),
        "mask_output_dir": (
            str(map_output_dir)
            if needs_disk_map_outputs and not bool(args.no_resample_or_gen)
            else None
        ),
        "binned_outputs": bool(args.no_resample),
        "render_images": bool(args.render_images),
        "image_output_dir": str(image_output_dir) if args.render_images else None,
        "image_max_dim": int(args.image_max_dim),
        "image_dpi": int(args.image_dpi),
        "skip_existing": bool(args.skip_existing),
        "particle_overlay_rendering": (
            particle_overlay_context["summary"]
            if particle_overlay_context is not None
            else {"enabled": False}
        ),
        "inputs": [],
    }
    incremental_summary_path = (
        live_summary_path
        if live_summary_path is not None
        else (output_dir / "inference_summary.json" if finalize else None)
    )

    for idx, mrc_path in enumerate(mrc_paths, start=1):
        stem = mrc_path.stem
        prob_npy = map_output_dir / f"{stem}_prob.npy"
        mask_npy = map_output_dir / f"{stem}_mask.npy"
        prob_mrc = output_dir / f"{stem}_prob.mrc"
        mask_mrc = output_dir / f"{stem}_mask.mrc"
        diagnostic_png = image_output_dir / f"{stem}_clean_filtering.png"
        particle_overlay_png = (
            particle_overlay_context["output_dir"] / f"{stem}_particle_overlay.png"
            if particle_overlay_context is not None
            else None
        )
        skip_resample = bool(args.no_resample or args.no_resample_or_gen)
        skip_generation = bool(args.no_resample_or_gen)
        expected_outputs = []
        if not skip_generation and needs_disk_map_outputs:
            expected_outputs.extend([prob_npy, mask_npy])
        if not skip_generation and save_mrc_outputs:
            expected_outputs.extend([prob_mrc, mask_mrc])
        if args.render_images:
            expected_outputs.append(diagnostic_png)
        if particle_overlay_png is not None:
            expected_outputs.append(particle_overlay_png)
        if (
            bool(args.skip_existing)
            and (not skip_generation)
            and expected_outputs
            and all(p.exists() for p in expected_outputs)
        ):
            print(
                f"[{idx}/{len(mrc_paths)}] {mrc_path.name}: skipping existing outputs",
                flush=True,
            )
            skipped_record = {
                "input_mrc": str(mrc_path),
                "status": "skipped_existing",
                "generated_outputs": bool(needs_disk_map_outputs),
                "exported_outputs": bool(export_mask_outputs and needs_disk_map_outputs),
                "output_prob_npy": str(prob_npy) if needs_disk_map_outputs else None,
                "output_mask_npy": str(mask_npy) if needs_disk_map_outputs else None,
                "output_prob_mrc": str(prob_mrc) if save_mrc_outputs else None,
                "output_mask_mrc": str(mask_mrc) if save_mrc_outputs else None,
                "output_diagnostic_png": str(diagnostic_png) if args.render_images else None,
                "particle_overlay_png": (
                    str(particle_overlay_png)
                    if particle_overlay_png is not None and particle_overlay_png.exists()
                    else None
                ),
            }
            if particle_overlay_context is not None and particle_overlay_png is not None and particle_overlay_png.exists():
                particle_overlay_context["frame_paths"].append(particle_overlay_png)
                particle_overlay_context["frame_borders"].append(None)
            summary["inputs"].append(skipped_record)
            if incremental_summary_path is not None:
                _write_inference_summary(incremental_summary_path, summary)
            continue

        image, header_pixel_size = _load_mrc_2d(mrc_path)
        pixel_size = args.pixel_size_angstrom if args.pixel_size_angstrom is not None else header_pixel_size
        policy = _resolve_policy(pixel_size)
        if int(policy.overlap) >= int(args.patch_size):
            raise ValueError(
                f"Resolved overlap {int(policy.overlap)} from policy {policy.policy_name}/"
                f"{policy.inference_profile} must be smaller than --patch-size {int(args.patch_size)}"
            )

        px_label = "unknown" if pixel_size is None else f"{float(pixel_size):.4f} A/px"
        targets_label = ",".join(f"{float(v):.3f}" for v in policy.multiscale_targets)
        print(f"[{idx}/{len(mrc_paths)}] {mrc_path.name}", flush=True)
        print(
            f"  pixel_size={px_label}; policy={policy.policy_name}; "
            f"profile={policy.inference_profile}; target(s)={targets_label} A/px; "
            f"overlap={int(policy.overlap)}; tta={bool(policy.use_tta)}",
            flush=True,
        )
        work_pixel_size = pixel_size

        if skip_generation:
            print("  skipping output map generation (--no-resample-or-gen)", flush=True)

        per_scale_results: list[dict] = []
        prob_maps_for_fusion: list[np.ndarray] = []
        prob_map: Optional[np.ndarray] = None
        output_pixel_size = pixel_size

        for s_idx, target_px in enumerate(policy.multiscale_targets, start=1):
            work_image, downsampled, downsample_meta = _maybe_downsample(
                image=image,
                pixel_size_angstrom=pixel_size,
                adaptive=bool(policy.adaptive_downsample),
                target_pixel_size=float(target_px),
                max_factor=adaptive_max_factor,
                min_work_dim=int(max(0, args.adaptive_min_work_dim)),
                method=str(policy.adaptive_downsample_method),
            )

            work_pixel_size = pixel_size
            if (
                downsampled
                and pixel_size is not None
                and pixel_size > 0
                and work_image.shape[0] > 0
                and work_image.shape[1] > 0
            ):
                scale_y = image.shape[0] / work_image.shape[0]
                scale_x = image.shape[1] / work_image.shape[1]
                work_pixel_size = float(pixel_size) * float((scale_y + scale_x) * 0.5)

            if len(policy.multiscale_targets) > 1:
                print(
                    f"  scale {s_idx}/{len(policy.multiscale_targets)} target={target_px:.3f} A/px: {image.shape} -> {work_image.shape}",
                    flush=True,
                )
            elif downsampled:
                print(f"  adaptive downsample: {image.shape} -> {work_image.shape}", flush=True)

            if downsampled and skip_resample:
                print("  keeping downsampled outputs (no resample to original size)", flush=True)

            effective_pixel_size = work_pixel_size if work_pixel_size is not None else float(target_px)
            prob_scale = predict_bad_regions_probability(
                model=model,
                image=work_image,
                device=args.device,
                patch_size=args.patch_size,
                overlap=int(policy.overlap),
                pixel_size_angstrom=effective_pixel_size,
                use_power_spectrum=use_psd,
                use_tta=bool(policy.use_tta),
                temperature=args.temperature,
                logit_bias=args.logit_bias,
                batch_forward_size=resolved_batch_forward_size,
                psd_multiscale=psd_multiscale,
                psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                psd_frequency_band_channels=psd_frequency_band_channels,
                psd_frequency_bands=psd_frequency_bands,
                psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
                multiscale_psd_source=str(policy.multiscale_psd_source),
                blending_window=str(policy.blending_window),
                blending_edge_px=int(policy.blending_edge_px),
                stitch_phase_mode=str(args.stitch_phase_mode),
                stitch_phase_reduction=str(args.stitch_phase_reduction),
                stitch_phase_blend_alpha=float(args.stitch_phase_blend_alpha),
                stitch_output_space=str(args.stitch_output_space),
                stitch_output_blend_alpha=float(args.stitch_output_blend_alpha),
                stitch_output_preserve_threshold=float(args.stitch_output_preserve_threshold),
                stitch_output_transition=float(args.stitch_output_transition),
                patch_prob_floor_quantile=float(args.patch_prob_floor_quantile),
                patch_prob_floor_strength=float(args.patch_prob_floor_strength),
                psd_scales=psd_scales,
                normalization_method=str(policy.normalization_method),
            )

            resampled_to_original = False
            if tuple(prob_scale.shape) != tuple(image.shape) and not skip_resample:
                prob_scale = _resize_to_shape(prob_scale, tuple(image.shape), order=1)
                resampled_to_original = True

            prob_scale = np.clip(np.asarray(prob_scale, dtype=np.float32), 0.0, 1.0)
            if not skip_resample:
                prob_maps_for_fusion.append(prob_scale)
                output_pixel_size = pixel_size
            else:
                # Single-scale only when skip_resample is enabled.
                prob_map = prob_scale
                output_pixel_size = work_pixel_size if tuple(prob_scale.shape) != tuple(image.shape) else pixel_size

            per_scale_results.append(
                {
                    "scale_index": int(s_idx),
                    "target_pixel_size_angstrom": float(target_px),
                    "policy_name": str(policy.policy_name),
                    "input_image_shape": list(image.shape),
                    "work_image_shape": list(work_image.shape),
                    "work_pixel_size_angstrom": work_pixel_size,
                    "adaptive_downsample_applied": bool(downsampled),
                    "downsample_meta": downsample_meta,
                    "resampled_to_original": bool(resampled_to_original),
                    "prob_min": float(prob_scale.min()),
                    "prob_max": float(prob_scale.max()),
                    "prob_mean": float(prob_scale.mean()),
                }
            )

        if not skip_resample:
            if len(prob_maps_for_fusion) == 1:
                prob_map = prob_maps_for_fusion[0]
            else:
                prob_map = np.mean(np.stack(prob_maps_for_fusion, axis=0), axis=0).astype(np.float32, copy=False)

        if prob_map is None:
            raise RuntimeError("Internal error: prob_map was not produced")

        prob_map = np.clip(prob_map.astype(np.float32, copy=False), 0.0, 1.0)
        output_is_downsampled = tuple(prob_map.shape) != tuple(image.shape)
        mask, mask_postprocess_meta = _postprocess_inference_mask(
            prob_map=prob_map,
            threshold=float(args.threshold),
            hysteresis_mode=str(args.hysteresis_mode),
            hysteresis_low_delta=float(args.hysteresis_low_delta),
            hysteresis_low_min=float(args.hysteresis_low_min),
            hysteresis_auto_max_growth_ratio=float(args.hysteresis_auto_max_growth_ratio),
            hysteresis_auto_min_added_mean_prob_margin=float(args.hysteresis_auto_min_added_mean_prob_margin),
            min_area_pixels=int(args.min_area_pixels),
            fill_small_holes_max_area_pixels=int(args.fill_small_holes_max_area_pixels),
            mask_buffer_distance_angstrom=float(args.mask_buffer_distance_angstrom),
            output_pixel_size_angstrom=output_pixel_size,
            fill_small_holes_after_buffer_max_area_pixels=int(args.fill_small_holes_after_buffer_max_area_pixels),
        )

        prob_npy_path: Optional[str] = None
        mask_npy_path: Optional[str] = None
        prob_mrc_path: Optional[str] = None
        mask_mrc_path: Optional[str] = None
        diagnostic_png_path: Optional[str] = None

        if not skip_generation:
            if needs_disk_map_outputs:
                np.save(prob_npy, prob_map)
                np.save(mask_npy, mask)
                prob_npy_path = str(prob_npy)
                mask_npy_path = str(mask_npy)

            if save_mrc_outputs:
                _save_mrc(prob_mrc, prob_map, output_pixel_size)
                _save_mrc(mask_mrc, mask.astype(np.float32, copy=False), output_pixel_size)
                prob_mrc_path = str(prob_mrc)
                mask_mrc_path = str(mask_mrc)
        elif args.save_mrc:
            print("  note: --save-mrc ignored because --no-resample-or-gen skips output generation", flush=True)

        if args.render_images:
            from cryofilter.visualization import render_filtering_png

            render_filtering_png(
                image=image,
                probability=prob_map,
                mask=mask,
                output_path=diagnostic_png,
                max_display_dim=int(args.image_max_dim),
                dpi=int(args.image_dpi),
            )
            diagnostic_png_path = str(diagnostic_png)
            print(f"  diagnostic image: {diagnostic_png}", flush=True)

        input_record = {
            "input_mrc": str(mrc_path),
            "pixel_size_angstrom": pixel_size,
            "policy": policy.to_json_dict(),
            "work_pixel_size_angstrom": per_scale_results[0]["work_pixel_size_angstrom"],
            "input_image_shape": list(image.shape),
            "work_image_shape": per_scale_results[0]["work_image_shape"],
            "output_image_shape": list(prob_map.shape),
            "adaptive_downsample_applied": bool(per_scale_results[0]["adaptive_downsample_applied"]),
            "resampled_to_original": bool(any(bool(r["resampled_to_original"]) for r in per_scale_results)),
            "multi_scale_fusion_applied": bool(len(policy.multiscale_targets) > 1),
            "per_scale": per_scale_results,
            "output_is_downsampled": bool(output_is_downsampled),
            "generated_outputs": bool((not skip_generation) and needs_disk_map_outputs),
            "exported_outputs": bool((not skip_generation) and export_mask_outputs and needs_disk_map_outputs),
            "output_prob_npy": prob_npy_path,
            "output_mask_npy": mask_npy_path,
            "output_prob_mrc": prob_mrc_path,
            "output_mask_mrc": mask_mrc_path,
            "output_diagnostic_png": diagnostic_png_path,
            "output_pixel_size_angstrom": output_pixel_size,
            "mask_threshold": args.threshold,
            "mask_postprocessing": mask_postprocess_meta,
            "particle_exclusion_distance_angstrom": float(args.particle_exclusion_distance_angstrom),
            "prob_min": float(prob_map.min()),
            "prob_max": float(prob_map.max()),
            "prob_mean": float(prob_map.mean()),
        }
        if particle_overlay_context is not None:
            input_record["particle_overlay"] = _render_particle_overlay_for_micrograph(
                context=particle_overlay_context,
                args=args,
                mrc_path=mrc_path,
                image=image,
                mask=mask,
                prob_map=prob_map,
                pixel_size_angstrom=pixel_size,
            )
        summary["inputs"].append(input_record)
        if incremental_summary_path is not None:
            _write_inference_summary(incremental_summary_path, summary)

    if particle_overlay_context is not None:
        _finalize_particle_overlay_contact_sheet(particle_overlay_context, args)

    if not finalize:
        print(f"Finished worker inference on {len(mrc_paths)} micrograph(s).", flush=True)
        return summary

    return _finalize_inference_run(
        args=args,
        input_path=input_path,
        output_dir=output_dir,
        mrc_paths=mrc_paths,
        summary=summary,
    )


def _multi_gpu_worker(
    args_payload: dict[str, object],
    mrc_paths: Sequence[Path],
    worker_summary_path: Path,
) -> None:
    """Spawn-safe worker entry point; particle filtering is finalized by the parent."""
    worker_args = argparse.Namespace(**args_payload)
    worker_args.filtered_particle_file = None
    worker_args.allow_unmatched_particles = True
    worker_args.particle_overlay_contact_sheet = False
    result = _run_infer(
        worker_args,
        mrc_paths_override=mrc_paths,
        finalize=False,
        live_summary_path=worker_summary_path,
    )
    if not isinstance(result, dict):
        raise RuntimeError("Multi-GPU inference worker did not return an inference summary")
    _write_inference_summary(worker_summary_path, result)


def _run_multi_gpu_infer(args: argparse.Namespace, devices: Sequence[str]) -> int:
    if args.overlap >= args.patch_size:
        raise ValueError("--overlap must be smaller than --patch-size")
    if not args.particle_file and (args.filtered_particle_file or args.particle_csg):
        raise ValueError("--filtered-particle-file and --particle-csg require --particle-file")
    render_particle_overlays = _resolve_render_particle_overlays(args)
    if render_particle_overlays and not args.particle_file:
        raise ValueError("--render-particle-overlays requires --particle-file")
    if args.particle_file and args.no_resample_or_gen:
        raise ValueError(
            "Particle filtering requires final mask outputs; do not combine --particle-file "
            "with --no-resample-or-gen."
        )
    if args.image_output_dir and not args.render_images:
        raise ValueError("--image-output-dir requires --render-images")
    if args.render_images and args.no_resample_or_gen:
        raise ValueError("--render-images cannot be combined with --no-resample-or-gen")
    if args.render_images and int(args.image_max_dim) < 8:
        raise ValueError("--image-max-dim must be at least 8")
    if args.render_images and int(args.image_dpi) < 1:
        raise ValueError("--image-dpi must be at least 1")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir_for_input(input_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mrc_paths = _collect_mrc_paths(input_path, recursive=args.recursive)

    active_devices = list(devices[: len(mrc_paths)])
    if len(active_devices) <= 1:
        args.device = active_devices[0]
        return int(_run_infer(args))

    stems: dict[str, Path] = {}
    for path in mrc_paths:
        previous = stems.get(path.stem)
        if previous is not None:
            raise ValueError(
                "Multi-GPU inference requires unique micrograph stems because outputs are "
                f"written to one directory: {previous} and {path}"
            )
        stems[path.stem] = path

    shards = _shard_mrc_paths(mrc_paths, len(active_devices))
    worker_summary_paths = [
        output_dir / f".cryofilter_worker_{index:02d}_summary.json"
        for index in range(len(active_devices))
    ]
    print(
        f"Using {len(active_devices)} GPUs for {len(mrc_paths)} micrograph(s): "
        + ", ".join(active_devices),
        flush=True,
    )
    print(
        "Worker progress counters are per-GPU shard; the app aggregates completed micrographs across workers.",
        flush=True,
    )

    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    assignments: list[dict[str, object]] = []
    for index, (device, shard, worker_summary_path) in enumerate(
        zip(active_devices, shards, worker_summary_paths)
    ):
        worker_payload = dict(vars(args))
        worker_payload["device"] = device
        worker_payload["force_disk_mask_outputs"] = bool(args.particle_file)
        process = context.Process(
            target=_multi_gpu_worker,
            args=(worker_payload, shard, worker_summary_path),
            name=f"cryofilter-{device.replace(':', '-')}",
        )
        process.start()
        processes.append(process)
        assignments.append(
            {
                "worker": int(index),
                "device": device,
                "micrograph_count": len(shard),
                "micrographs": [str(path) for path in shard],
                "worker_summary": str(worker_summary_path),
            }
        )

    for process in processes:
        process.join()
    failures = [
        f"{process.name} (exit code {process.exitcode})"
        for process in processes
        if process.exitcode != 0
    ]
    if failures:
        raise RuntimeError("Multi-GPU inference failed: " + ", ".join(failures))

    worker_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in worker_summary_paths
    ]
    summary = _merge_worker_summaries(worker_summaries, mrc_paths)
    summary["device"] = "multi_gpu"
    summary["devices"] = active_devices
    summary["multi_gpu"] = {
        "enabled": True,
        "worker_count": len(active_devices),
        "assignments": assignments,
    }

    return _finalize_inference_run(
        args=args,
        input_path=input_path,
        output_dir=output_dir,
        mrc_paths=mrc_paths,
        summary=summary,
    )


def _run_filter_particles(args: argparse.Namespace) -> int:
    from cryofilter.particle_filtering import default_filtered_particle_path, filter_particle_file

    input_path = Path(args.input).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else mask_dir
    )
    particle_input = Path(args.particle_file).expanduser().resolve()
    particle_output = (
        Path(args.filtered_particle_file).expanduser().resolve()
        if args.filtered_particle_file
        else default_filtered_particle_path(particle_input, output_dir).resolve()
    )
    summary_file = (
        Path(args.summary_file).expanduser().resolve()
        if args.summary_file
        else (output_dir / "particle_filter_summary.json").resolve()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    micrograph_paths = _collect_mrc_paths(input_path, recursive=bool(args.recursive))
    particle_filter_summary = filter_particle_file(
        particle_file=particle_input,
        output_file=particle_output,
        micrograph_paths=micrograph_paths,
        mask_dir=mask_dir,
        exclusion_distance_angstrom=float(args.particle_exclusion_distance_angstrom),
        pixel_size_override_angstrom=args.pixel_size_angstrom,
        csg_file=args.particle_csg,
        allow_unmatched=bool(args.allow_unmatched_particles),
        overwrite=bool(args.overwrite_filtered_particles),
    )
    particle_filter_summary["summary_file"] = str(summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(particle_filter_summary, handle, indent=2)

    print(
        "Particle filtering: "
        f"kept {particle_filter_summary['particles_kept']}/"
        f"{particle_filter_summary['particles_input']} "
        f"({particle_filter_summary['particles_removed']} removed)",
        flush=True,
    )
    print(f"Filtered particles: {particle_filter_summary['output_particle_file']}", flush=True)
    if particle_filter_summary.get("output_csg_file"):
        print(f"Filtered CryoSPARC group: {particle_filter_summary['output_csg_file']}", flush=True)
    extra_files = [
        path
        for path in particle_filter_summary.get("output_particle_files", [])
        if path != particle_filter_summary["output_particle_file"]
    ]
    if extra_files:
        print("Additional filtered .cs files:", flush=True)
        for path in extra_files:
            print(f"  {path}", flush=True)
    print(f"Particle summary: {summary_file}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryofilter", description="cryoFILTER command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run contamination segmentation on one MRC file or a directory of MRCs.")
    infer.add_argument(
        "--checkpoint",
        "--weights",
        dest="checkpoint",
        default=None,
        help="Path to trained .pt checkpoint. Defaults to pretrained_models/cryoFILTER_FULL.pt when present.",
    )
    infer.add_argument(
        "--input",
        "--input-directory",
        "--input_directory",
        "--input-dir",
        "--input_dir",
        "--input-path",
        "--input_path",
        "--input-file",
        "--input_file",
        dest="input",
        required=True,
        help="Input .mrc file or directory containing .mrc files.",
    )
    infer.add_argument(
        "--output-dir",
        "--output_directory",
        "--output_dir",
        "--output",
        dest="output_dir",
        default=None,
        help="Directory for segmentation outputs. Defaults to <input>_cryofilter_output.",
    )
    infer.add_argument("--device", default=_default_device(), help="Device for inference, e.g. cuda or cpu.")
    infer.add_argument(
        "--gpus",
        "--devices",
        dest="gpus",
        default="",
        help=(
            "Run directory inference across multiple GPUs using image-level sharding. "
            "Pass GPU indices such as 0 or 0,1,2,3; use 'all' for every visible GPU. "
            "The legacy cuda:0,cuda:1 form is also accepted. "
            "Overrides --device when supplied."
        ),
    )
    infer.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Optional CPU thread count for preprocessing, PNG generation, and math libraries.",
    )
    infer.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Optional number of CUDA GPUs to use. If --gpus is omitted, values >0 "
            "select the first N visible GPUs."
        ),
    )
    infer.add_argument(
        "--model-type",
        default="auto",
        choices=("auto", "unet_attention", "unet", "resnet_unet", "simple"),
        help="Model architecture. Use auto to infer from checkpoint.",
    )
    infer.add_argument(
        "--attention-type",
        default="auto",
        choices=("auto", "global", "dual", "none"),
        help="Attention mode for unet_attention checkpoints. Use auto to infer when possible.",
    )
    infer.add_argument(
        "--use-psd",
        default="auto",
        choices=("auto", "true", "false"),
        help="Whether to run image+PSD (2-channel) inference. auto infers from checkpoint.",
    )
    infer.add_argument("--norm-type", default="group", choices=("group", "batch", "instance"), help="Model normalization mode.")
    infer.add_argument("--patch-size", type=int, default=256, help="Patch size for sliding-window inference.")
    infer.add_argument("--overlap", type=int, default=DEFAULT_PUBLIC_OVERLAP, help="Patch overlap in pixels.")
    infer.add_argument(
        "--multiscale-psd-source",
        default=DEFAULT_PUBLIC_MULTISCALE_PSD_SOURCE,
        choices=("patch", "full_image"),
        help="For multiscale PSD inputs, compute PSD per patch (legacy) or once on the full working-resolution image.",
    )
    infer.add_argument(
        "--blending-window",
        default=DEFAULT_PUBLIC_BLENDING_WINDOW,
        choices=("hann", "tukey"),
        help="Patch-stitching blend window.",
    )
    infer.add_argument(
        "--blending-edge-px",
        type=int,
        default=DEFAULT_PUBLIC_BLENDING_EDGE_PX,
        help="Edge taper size in pixels for Tukey blending.",
    )
    infer.add_argument(
        "--stitch-phase-mode",
        default="single",
        choices=("single", "diag_halfstride", "quad_halfstride"),
        help="Optional shifted-grid ensemble used to suppress tiling haze.",
    )
    infer.add_argument(
        "--stitch-phase-reduction",
        default="mean",
        choices=("mean", "median", "max"),
        help="How to combine shifted-grid predictions when --stitch-phase-mode is enabled.",
    )
    infer.add_argument(
        "--stitch-phase-blend-alpha",
        type=float,
        default=1.0,
        help="Blend between the base grid and shifted-grid consensus (0=base only, 1=consensus only).",
    )
    infer.add_argument(
        "--stitch-output-space",
        default="probability",
        choices=("probability", "logit", "hybrid", "adaptive_hybrid"),
        help="Space used for overlap stitching inside inference.",
    )
    infer.add_argument(
        "--stitch-output-blend-alpha",
        type=float,
        default=0.5,
        help="When --stitch-output-space=hybrid, weight placed on the logit-stitched map.",
    )
    infer.add_argument("--stitch-output-preserve-threshold", type=float, default=0.8)
    infer.add_argument("--stitch-output-transition", type=float, default=0.15)
    infer.add_argument(
        "--patch-prob-floor-quantile",
        type=float,
        default=0.0,
        help="Optional per-patch low-quantile background floor used before probability-space stitching.",
    )
    infer.add_argument(
        "--patch-prob-floor-strength",
        type=float,
        default=0.0,
        help="Strength of the per-patch probability-floor subtraction.",
    )
    infer.add_argument(
        "--batch-forward-size",
        type=int,
        default=0,
        help=(
            "Number of patches to batch per model forward pass (GPU speed optimization). "
            "Use 0 for auto (A100/H100/H200 and RTX 4090/24 GB-class GPUs=32, 16 GB-class GPUs=16, CPU=1)."
        ),
    )
    infer.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MASK_THRESHOLD,
        help="High threshold for final binary mask export. Raise for more conservative masks.",
    )
    infer.add_argument(
        "--hysteresis-mode",
        default="auto",
        choices=("off", "on", "auto"),
        help="Final mask hysteresis growth mode. auto accepts conservative low-threshold growth only when safety gates pass.",
    )
    infer.add_argument(
        "--hysteresis-low-delta",
        type=float,
        default=0.10,
        help="Low hysteresis threshold is max(--hysteresis-low-min, --threshold - this value).",
    )
    infer.add_argument(
        "--hysteresis-low-min",
        type=float,
        default=0.20,
        help="Minimum low threshold allowed for hysteresis growth.",
    )
    infer.add_argument(
        "--hysteresis-auto-max-growth-ratio",
        type=float,
        default=1.0,
        help="In auto mode, reject hysteresis if added pixels exceed this multiple of seed pixels. Use -1 to disable.",
    )
    infer.add_argument(
        "--hysteresis-auto-min-added-mean-prob-margin",
        type=float,
        default=0.05,
        help="In auto mode, added hysteresis pixels must average at least low_threshold plus this margin.",
    )
    infer.add_argument(
        "--min-area-pixels",
        type=int,
        default=400,
        help="Remove final-mask connected components smaller than this many pixels. Use 0 to disable.",
    )
    infer.add_argument(
        "--fill-small-holes-max-area-pixels",
        type=int,
        default=400,
        help="Fill enclosed good islands up to this area before optional mask buffering. Use 0 to disable.",
    )
    infer.add_argument(
        "--mask-buffer-distance-angstrom",
        type=float,
        default=0.0,
        help="Dilate final contamination mask by this physical distance before export. Default 0 A.",
    )
    infer.add_argument(
        "--fill-small-holes-after-buffer-max-area-pixels",
        type=int,
        default=400,
        help="Fill enclosed good islands up to this area after optional mask buffering. Use 0 to disable.",
    )
    infer.add_argument(
        "--particle-exclusion-distance-angstrom",
        type=float,
        default=DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM,
        help=(
            "Remove supplied particles at or within this physical distance of the final contamination mask. "
            "A practical value is the radius of the original particle-extraction box: "
            "0.5 * original box pixels * original particle pixel size (do not use a Fourier-cropped box). "
            f"Default: {DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM:.0f} A."
        ),
    )
    infer.add_argument(
        "--particle-file",
        "--particles",
        dest="particle_file",
        default=None,
        help=(
            "Optional CryoSPARC .cs or RELION coordinate .star file. When supplied, inference "
            "also writes a filtered particle file using the final post-processed masks."
        ),
    )
    infer.add_argument(
        "--filtered-particle-file",
        "--particle-output",
        dest="filtered_particle_file",
        default=None,
        help=(
            "Optional filtered .cs or .star output path. Defaults to "
            "<output-dir>/<particle-stem>_cryofiltered.<ext>."
        ),
    )
    infer.add_argument(
        "--particle-csg",
        default=None,
        help=(
            "CryoSPARC .csg template paired with --particle-file. Recommended for CryoSPARC "
            "so the filtered output can be re-imported as a .cs/.csg pair. A sibling .csg is "
            "used automatically when present."
        ),
    )
    infer.add_argument(
        "--allow-unmatched-particles",
        action="store_true",
        help=(
            "Preserve particles whose micrograph names do not match the inference inputs. "
            "By default, unmatched particle rows stop the run to prevent partial filtering."
        ),
    )
    infer.add_argument(
        "--overwrite-filtered-particles",
        action="store_true",
        help="Replace an existing filtered particle output. The input particle file is never overwritten.",
    )
    infer.add_argument(
        "--render-particle-overlays",
        dest="render_particle_overlays",
        action="store_true",
        default=None,
        help=(
            "With --particle-file, write one on-the-fly QC PNG per processed micrograph "
            "showing the raw micrograph, final mask with kept/rejected particles, and probability map. "
            "Default: on when --particle-file is supplied."
        ),
    )
    infer.add_argument(
        "--no-render-particle-overlays",
        dest="render_particle_overlays",
        action="store_false",
        help="Disable on-the-fly QC PNG generation when filtering a particle file.",
    )
    infer.add_argument(
        "--particle-overlay-dir",
        default=None,
        help=f"Directory for on-the-fly QC PNGs. Defaults to <output-dir>/{DEFAULT_PARTICLE_OVERLAY_SUBDIR}.",
    )
    infer.add_argument(
        "--particle-overlay-max-display-dim",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_MAX_DISPLAY_DIM,
        help="Longest display edge for per-micrograph particle overlay PNGs.",
    )
    infer.add_argument(
        "--particle-overlay-diameter-px",
        type=float,
        default=DEFAULT_PARTICLE_OVERLAY_DIAMETER_PX,
        help="Display-space diameter for particle circles/X marks in overlay PNGs.",
    )
    infer.add_argument(
        "--particle-overlay-mask-alpha",
        type=float,
        default=0.35,
        help="Opacity for the light-purple predicted-mask overlay.",
    )
    infer.add_argument(
        "--particle-overlay-raw-panel",
        dest="particle_overlay_raw_panel",
        action="store_true",
        default=True,
        help="Include the raw micrograph as the left panel in each particle-overlay QC PNG. Default: on.",
    )
    infer.add_argument(
        "--no-particle-overlay-raw-panel",
        dest="particle_overlay_raw_panel",
        action="store_false",
        help="Write compact overlay-only particle QC PNGs.",
    )
    infer.add_argument(
        "--particle-overlay-probability-panel",
        dest="particle_overlay_probability_panel",
        action="store_true",
        default=True,
        help="Include the viridis probability map as the rightmost panel in particle-overlay QC PNGs. Default: on.",
    )
    infer.add_argument(
        "--no-particle-overlay-probability-panel",
        dest="particle_overlay_probability_panel",
        action="store_false",
        help="Do not include the probability-map panel in particle-overlay QC PNGs.",
    )
    infer.add_argument(
        "--particle-overlay-contact-sheet",
        dest="particle_overlay_contact_sheet",
        action="store_true",
        default=True,
        help="Write and periodically refresh particle_overlay_contact_sheet.png. Default: on when overlays are enabled.",
    )
    infer.add_argument(
        "--no-particle-overlay-contact-sheet",
        dest="particle_overlay_contact_sheet",
        action="store_false",
        help="Write per-micrograph overlay PNGs only.",
    )
    infer.add_argument(
        "--particle-overlay-contact-sheet-cols",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_COLS,
        help="Number of columns in the live particle overlay contact sheet.",
    )
    infer.add_argument(
        "--particle-overlay-contact-sheet-tile",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_TILE,
        help="Tile size in pixels for the particle overlay contact sheet.",
    )
    infer.add_argument(
        "--particle-overlay-contact-sheet-every",
        type=int,
        default=DEFAULT_PARTICLE_OVERLAY_CONTACT_SHEET_EVERY,
        help="Refresh the contact sheet after this many newly rendered frames; <=0 only writes it at the end.",
    )
    infer.add_argument(
        "--particle-overlay-warn-removed-fraction",
        type=float,
        default=DEFAULT_PARTICLE_OVERLAY_WARN_REMOVED_FRACTION,
        help="Contact-sheet border turns red when rejected-particle fraction is at least this value.",
    )
    infer.add_argument("--psd-scales", default="8,16,32,64,128", help="Comma-separated PSD scales for multiscale PSD.")
    infer.add_argument(
        "--psd-multiscale-separate-channels",
        dest="psd_multiscale_separate_channels",
        action="store_true",
        default=None,
        help=(
            "Use one PSD channel per multiscale bin (image + N PSD channels). "
            "By default this is auto-detected from checkpoint metadata/input shape."
        ),
    )
    infer.add_argument(
        "--no-psd-multiscale-separate-channels",
        dest="psd_multiscale_separate_channels",
        action="store_false",
        help="Force fused single-channel multiscale PSD mode.",
    )
    infer.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for logits.")
    infer.add_argument("--logit-bias", type=float, default=0.0, help="Additive bias for bad-class logits.")
    infer.add_argument(
        "--normalization-method",
        type=str,
        default=DEFAULT_PUBLIC_NORMALIZATION_METHOD,
        choices=(
            "robust",
            "percentile",
            "percentile_wide",
            "percentile_extra_wide",
            "percentile_blend",
            "simplipytem",
            "simplipytem_local",
            "simplipytem_local8bit",
        ),
        help="Real-space normalization passed into model inference.",
    )
    infer.add_argument("--use-tta", action="store_true", help="Enable 4-view test-time augmentation during inference.")
    infer.add_argument(
        "--tta",
        type=int,
        default=0,
        help="Compatibility alias for explicit TTA counts. Use --tta 4 to enable the same 4-view TTA as --use-tta; 0/1 disables it.",
    )
    infer.add_argument("--recursive", action="store_true", help="Recursively discover .mrc files under input directory.")
    infer.add_argument(
        "--pixel-size-angstrom",
        type=float,
        default=None,
        help="Pixel size override (A/px). If omitted, read from MRC header when available.",
    )
    infer.add_argument(
        "--target-pixel-size",
        "--target-apix",
        dest="target_pixel_size",
        type=float,
        default=DEFAULT_PUBLIC_TARGET_PIXEL_SIZE,
        help="Target pixel size for adaptive downsampling (A/px).",
    )
    profile_help = (
        "Small-pixel automatic inference profile. "
        "fast: target 2.0 A/px, Fourier resampling, patch-local PSD, Tukey blending, "
        f"overlap {SMALL_PIXEL_PROFILE_CONFIGS['fast']['overlap']}, no TTA. "
        "balanced: same recipe with "
        f"overlap {SMALL_PIXEL_PROFILE_CONFIGS['balanced']['overlap']}, no TTA. "
        "quality: manuscript-style recipe with "
        f"overlap {SMALL_PIXEL_PROFILE_CONFIGS['quality']['overlap']} and 4-view TTA."
    )
    infer.add_argument(
        "--inference-profile",
        default=DEFAULT_INFERENCE_PROFILE,
        choices=("fast", "balanced", "quality"),
        help=profile_help,
    )
    infer.add_argument(
        "--small-pixel-auto",
        dest="small_pixel_auto",
        action="store_true",
        default=True,
        help=(
            "Automatically switch to the selected small-pixel profile at or below "
            "--small-pixel-cutoff-angstrom."
        ),
    )
    infer.add_argument(
        "--no-small-pixel-auto",
        dest="small_pixel_auto",
        action="store_false",
        help="Disable the automatic small-pixel recipe override.",
    )
    infer.add_argument(
        "--small-pixel-cutoff-angstrom",
        type=float,
        default=DEFAULT_SMALL_PIXEL_CUTOFF_ANGSTROM,
        help="Pixel-size cutoff (A/px) for the automatic small-pixel recipe.",
    )
    infer.add_argument(
        "--multi-scale-target-pixel-sizes",
        type=str,
        default="",
        help=(
            "Optional comma-separated target A/px values for multi-scale fusion "
            "(example: '1.5,2.0'). If set, each scale is inferred and fused by averaging."
        ),
    )
    infer.add_argument(
        "--adaptive-downsample",
        dest="adaptive_downsample",
        action="store_true",
        default=True,
        help="Enable adaptive downsampling to target physical scale before inference.",
    )
    infer.add_argument(
        "--no-adaptive-downsample",
        dest="adaptive_downsample",
        action="store_false",
        help="Disable adaptive downsampling.",
    )
    infer.add_argument(
        "--adaptive-max-factor",
        type=float,
        default=0.0,
        help=(
            "Optional cap on adaptive downsample factor (target/pixel_size). "
            "Set >1 to enable; <=1 disables the cap."
        ),
    )
    infer.add_argument(
        "--adaptive-min-work-dim",
        type=int,
        default=0,
        help=(
            "Optional minimum short-side size (pixels) after adaptive downsampling. "
            "If downsampling would go below this, reduce the downsample amount."
        ),
    )
    infer.add_argument(
        "--adaptive-downsample-method",
        type=str,
        default=DEFAULT_PUBLIC_ADAPTIVE_DOWNSAMPLE_METHOD,
        choices=("bilinear", "fourier"),
        help="Resampling kernel for adaptive downsampling.",
    )
    infer.add_argument(
        "--save-mrc",
        dest="save_mrc",
        action="store_true",
        default=True,
        help="Save .mrc probability and mask outputs in addition to .npy files.",
    )
    infer.add_argument(
        "--no-save-mrc",
        dest="save_mrc",
        action="store_false",
        help="Disable .mrc output export (keep only .npy outputs).",
    )
    infer.add_argument(
        "--export-masks",
        dest="export_masks",
        action="store_true",
        default=True,
        help="Export per-micrograph probability and mask arrays. Enabled by default.",
    )
    infer.add_argument(
        "--no-export-masks",
        dest="export_masks",
        action="store_false",
        help=(
            "Do not export visible probability/mask arrays. Runs with particle "
            "filtering or typing may still keep hidden internal arrays until completion."
        ),
    )
    infer.add_argument(
        "--render-images",
        action="store_true",
        default=False,
        help=(
            "Also write one display-sized four-panel PNG per micrograph showing the raw "
            "micrograph, a purple final-mask overlay, the probability map, and retained regions. "
            "Disabled by default."
        ),
    )
    infer.add_argument(
        "--image-output-dir",
        default=None,
        help=(
            "Directory for --render-images PNGs. Defaults to "
            "<output-dir>/diagnostic_images."
        ),
    )
    infer.add_argument(
        "--image-max-dim",
        type=int,
        default=1400,
        help="Maximum displayed PNG dimension in pixels. Default: 1400.",
    )
    infer.add_argument(
        "--image-dpi",
        type=int,
        default=180,
        help="Diagnostic PNG resolution in dots per inch. Default: 180.",
    )
    infer.add_argument(
        "--no-resample",
        "--no_resample",
        dest="no_resample",
        action="store_true",
        default=False,
        help=(
            "If adaptive downsampling is applied, keep outputs at downsampled resolution "
            "(still generates .npy/.mrc outputs unless --no-resample-or-gen is also used)."
        ),
    )
    infer.add_argument(
        "--no-resample-or-gen",
        "--no_resample_or_gen",
        dest="no_resample_or_gen",
        action="store_true",
        default=False,
        help=(
            "Skip resampling back to original size and skip probability/mask file generation. "
            "This low-latency mode cannot be combined with --particle-file because particle filtering needs final masks."
        ),
    )
    infer.add_argument(
        "--skip-existing",
        "--resume",
        dest="skip_existing",
        action="store_true",
        default=False,
        help=(
            "Skip micrographs whose expected output files already exist. "
            "Useful when restarting a crashed directory run without rebuilding staging directories."
        ),
    )

    add_train_subparser(subparsers)

    filter_particles = subparsers.add_parser(
        "filter-particles",
        aliases=["filter"],
        help="Filter an existing particle file using previously generated cryoFILTER masks.",
    )
    filter_particles.add_argument(
        "--input",
        "--input-directory",
        "--input_directory",
        "--input-dir",
        "--input_dir",
        "--input-path",
        "--input_path",
        "--input-file",
        "--input_file",
        dest="input",
        required=True,
        help="Original .mrc file or directory of .mrc files used to make the masks.",
    )
    filter_particles.add_argument(
        "--mask-dir",
        "--mask_dir",
        required=True,
        help="Directory containing existing <micrograph-stem>_mask.npy files from cryofilter infer.",
    )
    filter_particles.add_argument(
        "--output-dir",
        "--output_directory",
        "--output_dir",
        "--output",
        dest="output_dir",
        default=None,
        help="Directory for filtered particle outputs. Defaults to --mask-dir.",
    )
    filter_particles.add_argument(
        "--particle-file",
        "--particles",
        dest="particle_file",
        required=True,
        help="CryoSPARC .cs or RELION coordinate .star file to filter.",
    )
    filter_particles.add_argument(
        "--filtered-particle-file",
        "--particle-output",
        dest="filtered_particle_file",
        default=None,
        help="Optional filtered .cs or .star output path. Defaults to <output-dir>/<particle-stem>_cryofiltered.<ext>.",
    )
    filter_particles.add_argument(
        "--particle-csg",
        default=None,
        help="Optional CryoSPARC .csg paired with --particle-file. A sibling .csg is used automatically when present.",
    )
    filter_particles.add_argument(
        "--particle-exclusion-distance-angstrom",
        type=float,
        default=DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM,
        help=(
            "Remove supplied particles at or within this physical distance of the final contamination mask. "
            f"Default: {DEFAULT_PARTICLE_EXCLUSION_DISTANCE_ANGSTROM:.0f} A."
        ),
    )
    filter_particles.add_argument(
        "--pixel-size-angstrom",
        type=float,
        default=None,
        help="Pixel size override (A/px). If omitted, read from MRC header when available.",
    )
    filter_particles.add_argument("--recursive", action="store_true", help="Recursively discover .mrc files under input directory.")
    filter_particles.add_argument(
        "--allow-unmatched-particles",
        action="store_true",
        help=(
            "Preserve particles whose micrograph names do not match the input micrographs. "
            "By default, unmatched particle rows stop the run to prevent partial filtering."
        ),
    )
    filter_particles.add_argument(
        "--overwrite-filtered-particles",
        action="store_true",
        help="Replace existing filtered particle outputs. The input particle file is never overwritten.",
    )
    filter_particles.add_argument(
        "--summary-file",
        default=None,
        help="Optional JSON summary path. Defaults to <output-dir>/particle_filter_summary.json.",
    )

    add_type_subparser(subparsers)
    add_gui_subparser(subparsers)
    add_install_cryosparc_tools_subparser(subparsers)
    add_cryosparc_subparser(subparsers)
    add_app_subparser(subparsers)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "infer":
        _apply_infer_resource_args(args)
        devices = _resolve_inference_devices(args.gpus)
        if devices:
            return _run_multi_gpu_infer(args, devices)
        return _run_infer(args)
    if args.command == "train":
        return run_train(args)
    if args.command in {"filter-particles", "filter"}:
        return _run_filter_particles(args)
    if args.command == "type":
        return run_type(args)
    if args.command == "gui":
        return run_gui(args)
    if args.command == "install-cryosparc-tools":
        return args.func(args)
    if args.command == "cryosparc":
        return args.func(args)
    if args.command in {"app", "studio"}:
        return args.func(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
