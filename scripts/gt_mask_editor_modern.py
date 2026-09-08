#!/usr/bin/env python3
"""
Simple GT mask editor using matplotlib/Tk (no OpenGL required).

This version adds:
  - dataset-level filtering in the GUI (exact dataset IDs; comma/space separated)
  - split labels + filtering (ALL / TRAINING / VALIDATION)
  - resumable output/session directories for masks + typed contamination sub-labels
  - safer deletion behavior (explicit zero-mask files are written for cleared/modified masks)
  - typed contamination sub-labeling inside GT masks:
      1 = Carbon, 2 = Crystalline, 3 = Aggregate, 4 = Ethane

Usage:
    python scripts/gt_mask_editor_simple.py --manifest manifest_cleaned_final_gt_masks_selected.csv

Controls:
    Mouse:
      - GT mask / Polygon label: Left click adds polygon point, Right click / Ctrl+click / Enter completes polygon
      - Full label: Left click inside GT applies the active subtype to the full current scope
      - Middle click: Select region at cursor

    Keyboard:
      - d: Delete selected region
      - c: Clear entire mask
      - r: Reset current GT mask and redraw it from scratch
      - b / x / f / l: Switch GT-mask / erase-polygon / full-label / polygon-label mode
      - 1/2/3/4: Label the current full scope or drawn submask as Carbon/Crystalline/Aggregate/Ethane
      - 0 or u: Clear typed label back to "unlabeled contamination" inside the current full scope / submask
      - n / p: Next / previous image (within current filter)
      - ] / [: Next / previous dataset (within current filter)
      - s: Save the current row to the session directory
      - e: Export masks + manifest (stay open; heavier)
      - q: Save a fast resumable session checkpoint and quit
      - t / v / a: Split filter = TRAINING / VALIDATION / ALL
      - Escape: Cancel current polygon
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import importlib
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

# Bootstrap with a safe noninteractive backend, then switch to an available
# interactive backend (TkAgg / QtAgg) when the editor window is created.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, RadioButtons, TextBox

from scipy import ndimage

try:
    from skimage.draw import polygon as draw_polygon
except ModuleNotFoundError:
    def draw_polygon(ys, xs, shape):
        """
        Fallback polygon rasterization when scikit-image is unavailable.

        Returns (rr, cc) index arrays similar to skimage.draw.polygon.
        """
        h, w = shape
        poly = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        if poly.shape[0] < 3 or h <= 0 or w <= 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        min_x = max(int(np.floor(np.min(poly[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(poly[:, 0]))), w - 1)
        min_y = max(int(np.floor(np.min(poly[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(poly[:, 1]))), h - 1)
        if max_x < min_x or max_y < min_y:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        points = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(poly, closed=True).contains_points(points, radius=0.5)
        rr = yy.ravel()[inside].astype(np.intp, copy=False)
        cc = xx.ravel()[inside].astype(np.intp, copy=False)
        return rr, cc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.contamination_subtypes import (
    BACKGROUND_LABEL,
    TYPE_ID_SEQUENCE,
    TYPE_ID_TO_COLOR_HEX,
    TYPE_ID_TO_DISPLAY,
    TYPE_ID_TO_KEY,
    TYPE_MANIFEST_COLUMNS,
    TYPE_SCHEMA_NAME,
    TYPE_SCHEMA_VERSION,
    UNLABELED_CONTAMINATION_LABEL,
    build_unlabeled_typed_mask,
    reconcile_typed_mask,
    typed_class_pixel_counts,
    typed_completion_stats,
)


DEFAULT_AUTO_VAL_MANIFEST = Path(
    "testing_particle_filtering_with_crYOLO/manifests/heldout_val_manifest.csv"
)
VALID_SPLIT_FILTERS = ("ALL", "TRAINING", "VALIDATION")
VALID_EDIT_MODES = ("GT mask", "Erase polygon", "Full label", "Polygon label")
SORT_MODE_OPTIONS = (
    "Dataset / Filename",
    "Least Typed First",
    "Most Typed First",
    "Most Labeled First",
    "Least Labeled First",
    "Most Untyped Pixels",
)
DEFAULT_SELECTED_MASK_MANIFEST = Path("manifest_cleaned_final_gt_masks_selected.csv")
DEFAULT_HYBRID_SESSION_ROOT = Path("02_FINAL_RELEASE_FREQ_HYBRIDMASKS/contamination_typing_sessions")
DEFAULT_GUI_BACKEND_CANDIDATES = ("QtAgg", "TkAgg")
DEFAULT_GT_MASK_PATH_COLUMNS = (
    "gt_mask_path",
    "gt_mask_full_path",
    "gt_mask_path_selected_review",
    "gt_mask_path_v4_final",
    "gt_mask_full_path_v4_final",
    "gt_mask_path_original",
)


def normalize_dataset_id(value: object) -> str:
    """Normalize dataset_id to a stable string key."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return str(int(f)) if f.is_integer() else str(value).strip()

    text = str(value).strip()
    if text.endswith(".0"):
        try:
            f = float(text)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass
    return text


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _has_display_server() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _interactive_backend_candidates() -> List[str]:
    forced = os.environ.get("CRYOFILTER_ANNOTATION_BACKEND", "").strip()
    if forced:
        return [forced]

    candidates: List[str] = []
    if _module_available("PyQt5") or _module_available("PySide6") or _module_available("PySide2"):
        candidates.append("QtAgg")
    if _module_available("tkinter"):
        candidates.append("TkAgg")

    for fallback in DEFAULT_GUI_BACKEND_CANDIDATES:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def normalize_stem(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def row_key(dataset_id: object, stem: object) -> str:
    return f"{normalize_dataset_id(dataset_id)}__{normalize_stem(stem)}"


def resolve_existing_path(path_value: object, manifest_dir: Path) -> Optional[Path]:
    """Resolve a path from manifest (relative or absolute) to an existing file when possible."""
    if pd.isna(path_value):
        return None
    text = str(path_value).strip()
    if not text:
        return None

    p = Path(text)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([manifest_dir / p, Path.cwd() / p, p])

    for cand in candidates:
        if cand.exists():
            return cand

    # Return manifest-relative path for future writes/debugging even if it doesn't exist.
    return p if p.is_absolute() else (manifest_dir / p)


def path_for_manifest(path: Path, manifest_dir: Path) -> str:
    """Prefer manifest-relative paths when possible."""
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except Exception:
        return str(path)


def load_image(path: Path) -> Optional[np.ndarray]:
    """Load image from supported formats."""
    try:
        if path.suffix == ".npy":
            img = np.load(str(path))
            return np.squeeze(img)
        if path.suffix in [".mrc", ".mrcs"]:
            import mrcfile

            with warnings.catch_warnings():
                # Some legacy MRCs in this dataset are permissive-loadable but noisy.
                warnings.simplefilter("ignore", RuntimeWarning)
                with mrcfile.open(str(path), permissive=True) as mrc:
                    return np.squeeze(mrc.data.copy())
        if path.suffix in [".tif", ".tiff"]:
            from PIL import Image

            return np.squeeze(np.array(Image.open(str(path))))

        # Fallback: try MRC
        import mrcfile

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with mrcfile.open(str(path), permissive=True) as mrc:
                return np.squeeze(mrc.data.copy())
    except Exception as exc:
        print(f"  Failed to load {path}: {exc}")
        return None


def find_image(
    mic_path_value: object,
    mic_dir: Optional[Path],
    stem: str,
    manifest_dir: Path,
) -> Optional[np.ndarray]:
    """Try multiple locations/extensions to load the micrograph image."""
    paths_to_try: List[Path] = []

    mic_path = resolve_existing_path(mic_path_value, manifest_dir)

    if mic_dir is not None:
        mic_stem = Path(str(mic_path_value)).stem if not pd.isna(mic_path_value) else stem
        for ext in [".mrc", ".mrcs", ".npy", ".tif", ".tiff"]:
            paths_to_try.append(mic_dir / (stem + ext))
            if mic_stem:
                paths_to_try.append(mic_dir / (mic_stem + ext))

    if mic_path is not None:
        paths_to_try.append(mic_path)

    seen = set()
    for path in paths_to_try:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            img = load_image(path)
            if img is not None:
                return img
    return None


def candidate_gt_mask_columns_for_row(
    row: pd.Series,
    available_columns: Sequence[str],
) -> List[str]:
    """Order GT mask columns so the reviewed hybrid mask is reconstructed first."""
    source = str(row.get("gt_mask_review_selected_source", "")).strip().lower()

    candidates: List[str] = [
        "gt_mask_path",
        "gt_mask_full_path",
        "gt_mask_path_selected_review",
    ]
    if source == "mc":
        candidates.extend(
            [
                "gt_mask_path_original",
                "gt_mask_path_v4_final",
                "gt_mask_full_path_v4_final",
            ]
        )
    elif source == "final":
        candidates.extend(
            [
                "gt_mask_path_v4_final",
                "gt_mask_full_path_v4_final",
                "gt_mask_path_original",
            ]
        )
    else:
        candidates.extend(
            [
                "gt_mask_path_v4_final",
                "gt_mask_path_original",
                "gt_mask_full_path_v4_final",
            ]
        )

    candidates.extend(DEFAULT_GT_MASK_PATH_COLUMNS)
    available = set(str(col) for col in available_columns)
    return [col for col in dict.fromkeys(candidates) if col in available]


def resolve_gt_mask_path_for_row(
    row: pd.Series,
    manifest_dir: Path,
    available_columns: Sequence[str],
) -> Tuple[Optional[Path], Optional[str]]:
    """Return the best existing GT mask path for this row plus its source column."""
    for col in candidate_gt_mask_columns_for_row(row, available_columns):
        gt_path = resolve_existing_path(row.get(col, None), manifest_dir)
        if gt_path is not None and gt_path.exists():
            return gt_path, col
    return None, None


def parse_dataset_filter_text(text: str) -> List[str]:
    tokens = [t.strip() for t in re.split(r"[,\s]+", text.strip()) if t.strip()]
    return [normalize_dataset_id(t) for t in tokens]


def build_split_labels_from_seed(
    dataset_ids: Sequence[str],
    seed: int,
    train_frac: float,
) -> Tuple[List[str], Dict[str, object]]:
    """Reproduce dataset-stratified TRAINING/VALIDATION split labels from seed/train_frac."""
    ds_series = pd.Series(dataset_ids, dtype=object)
    unique_datasets = [x for x in ds_series.unique().tolist() if str(x) != ""]

    np.random.seed(seed)
    unique_datasets = np.array(unique_datasets, dtype=object)
    np.random.shuffle(unique_datasets)

    n_train = int(len(unique_datasets) * train_frac)
    train_set = set(str(x) for x in unique_datasets[:n_train])

    labels = [
        "TRAINING" if (ds and ds in train_set) else "VALIDATION"
        for ds in dataset_ids
    ]

    metadata = {
        "split_source": "seeded_dataset_split",
        "seed": int(seed),
        "train_frac": float(train_frac),
        "n_unique_datasets": int(len(unique_datasets)),
        "n_train_datasets": int(n_train),
        "n_val_datasets": int(len(unique_datasets) - n_train),
        "train_datasets": [str(x) for x in unique_datasets[:n_train].tolist()],
        "val_datasets": [str(x) for x in unique_datasets[n_train:].tolist()],
    }
    return labels, metadata


def build_split_labels_from_val_manifest(
    keys: Sequence[str],
    val_manifest_path: Path,
) -> Tuple[List[str], Dict[str, object]]:
    """Label rows by explicit heldout validation manifest membership."""
    val_df = pd.read_csv(val_manifest_path)
    if "dataset_id" not in val_df.columns or "stem" not in val_df.columns:
        raise ValueError(
            f"Validation manifest missing required columns dataset_id/stem: {val_manifest_path}"
        )

    val_keys = {
        row_key(row["dataset_id"], row["stem"])
        for _, row in val_df.iterrows()
    }
    labels = ["VALIDATION" if k in val_keys else "TRAINING" for k in keys]
    metadata = {
        "split_source": "val_manifest_membership",
        "val_manifest": str(val_manifest_path),
        "n_val_keys_from_manifest": int(len(val_keys)),
    }
    return labels, metadata


def normalize_split_label(value: object) -> Optional[str]:
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text in {"TRAIN", "TRAINING"}:
        return "TRAINING"
    if text in {"VAL", "VALID", "VALIDATION"}:
        return "VALIDATION"
    return None


def build_split_labels_from_manifest_column(df: pd.DataFrame) -> Tuple[List[str], Dict[str, object]]:
    """Use explicit row-level split labels from a manifest column."""
    split_column = "split" if "split" in df.columns else "split_label"
    labels = [normalize_split_label(value) for value in df[split_column].tolist()]
    if any(label is None for label in labels):
        bad = sorted(
            {
                str(value)
                for value, label in zip(df[split_column].tolist(), labels)
                if label is None
            }
        )
        raise ValueError(
            f"Manifest column {split_column!r} contains invalid split labels: {bad}. "
            "Use TRAINING or VALIDATION."
        )
    normalized = [str(label) for label in labels]
    return normalized, {
        "split_source": "manifest_column",
        "split_column": split_column,
        "n_training_rows": int(sum(1 for label in normalized if label == "TRAINING")),
        "n_validation_rows": int(sum(1 for label in normalized if label == "VALIDATION")),
    }


def resolve_editor_session_dir(
    *,
    manifest_dir: Path,
    workspace_root: Path,
    output_dir: Optional[str],
    output_root: str,
    output_name: Optional[str],
    export_timestamp: str,
) -> Path:
    """Resolve the resumable session directory, prompting the user when needed."""
    if output_dir:
        out_path = Path(output_dir).expanduser()
        run_dir = out_path if out_path.is_absolute() else (manifest_dir / out_path)
    else:
        out_root_path = Path(output_root).expanduser()
        resolved_root = out_root_path if out_root_path.is_absolute() else (manifest_dir / out_root_path)
        default_name = output_name or f"session_{export_timestamp}"
        default_dir = (resolved_root / default_name).resolve()
        run_dir = _prompt_for_session_dir(default_dir=default_dir)

    run_dir = run_dir.resolve()
    workspace_root = workspace_root.resolve()
    try:
        run_dir.relative_to(workspace_root)
    except Exception as exc:
        raise ValueError(
            f"Session/output directory must stay inside the workspace root {workspace_root}: {run_dir}"
        ) from exc
    return run_dir


def _prompt_for_session_dir(default_dir: Path) -> Path:
    """Prompt for a session directory using Tk when available, with a safe fallback."""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        value = simpledialog.askstring(
            "Annotation Session Directory",
            (
                "Enter a session/output directory.\n\n"
                "Use an existing directory to resume a previous labeling session,\n"
                "or enter a new directory path to start a fresh session."
            ),
            initialvalue=str(default_dir),
            parent=root,
        )
        try:
            root.destroy()
        except Exception:
            pass
        if value and str(value).strip():
            return Path(str(value).strip()).expanduser()
    except Exception:
        pass
    return default_dir


class SimpleMaskEditor:
    """Matplotlib-based mask editor with dataset/split filtering and isolated exports."""

    @staticmethod
    def _create_editor_figure():
        """Create the matplotlib figure using the first working interactive backend."""
        if not _has_display_server():
            raise RuntimeError(
                "No interactive display was detected for the annotation GUI.\n"
                "If you are connecting remotely, make sure X forwarding is active and "
                "`$DISPLAY` is set before launching the editor.\n"
                "Typical check: `echo $DISPLAY`\n"
                "Typical launch: `ssh -Y <host>` (or `ssh -XY <host>`) from a machine "
                "running an X server such as XQuartz, X11, or VcXsrv."
            )

        errors: List[str] = []
        tried: List[str] = []
        for backend in _interactive_backend_candidates():
            backend_name = str(backend).strip()
            if not backend_name:
                continue
            tried.append(backend_name)
            try:
                plt.switch_backend(backend_name)
                fig, ax = plt.subplots(1, 1, figsize=(12.5, 10.5))
                return fig, ax
            except Exception as exc:
                errors.append(f"{backend_name}: {exc}")

        tried_text = ", ".join(tried) if tried else "none"
        error_text = "\n".join(errors) if errors else "No backend attempts were made."
        raise RuntimeError(
            "Failed to initialize an interactive matplotlib backend for the annotation GUI.\n"
            f"Tried backends: {tried_text}\n"
            f"DISPLAY={os.environ.get('DISPLAY', '')!r}, "
            f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')!r}\n"
            "Backend errors:\n"
            f"{error_text}"
        )

    def __init__(
        self,
        manifest_path: str,
        mic_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_root: str = "gt_mask_editor_exports",
        output_name: Optional[str] = None,
        val_manifest: Optional[str] = None,
        split_seed: int = 42,
        split_train_frac: float = 0.7,
        initial_split: str = "ALL",
        initial_dataset_filter: str = "",
        initial_row_key: str = "",
        export_only_modified: bool = False,
        prefetch_workers: Optional[int] = None,
        prefetch_radius: int = 0,
        row_cache_size: int = 8,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        if not self.manifest_path.is_absolute():
            self.manifest_path = (Path.cwd() / self.manifest_path).resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.manifest_dir = self.manifest_path.parent

        self.mic_dir = None
        if mic_dir:
            mic_dir_path = Path(mic_dir).expanduser()
            self.mic_dir = (
                mic_dir_path
                if mic_dir_path.is_absolute()
                else (self.manifest_dir / mic_dir_path).resolve()
            )

        self.export_only_modified = bool(export_only_modified)
        self.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workspace_root = Path.cwd().resolve()
        default_prefetch_workers = 0
        requested_prefetch_workers = max(
            0,
            int(default_prefetch_workers if prefetch_workers is None else prefetch_workers),
        )
        requested_prefetch_radius = max(0, int(prefetch_radius))
        self.prefetch_workers = min(requested_prefetch_workers, 2)
        self.prefetch_radius = min(requested_prefetch_radius, 2)
        if self.prefetch_workers != requested_prefetch_workers:
            print(
                f"Capping prefetch_workers from {requested_prefetch_workers} to {self.prefetch_workers} "
                "to keep interactive navigation responsive."
            )
        if self.prefetch_radius != requested_prefetch_radius:
            print(
                f"Capping prefetch_radius from {requested_prefetch_radius} to {self.prefetch_radius} "
                "to avoid overloading the shared filesystem during labeling."
            )
        self.row_cache_size = max(2, int(row_cache_size))
        self.row_payload_cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.row_prefetch_futures: Dict[int, Future] = {}
        self.prefetch_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.prefetch_workers, thread_name_prefix="cf-modern-prefetch")
            if self.prefetch_workers > 0
            else None
        )
        self.interactive_load_executor: Optional[ThreadPoolExecutor] = None
        self.active_row_load_future: Optional[Future] = None
        self.active_row_load_token: int = 0
        self.active_row_load_row_idx: Optional[int] = None
        self.row_loading: bool = False

        if output_dir and output_name:
            raise ValueError("Use either --output_dir or --output_name, not both.")
        self.run_dir = resolve_editor_session_dir(
            manifest_dir=self.manifest_dir,
            workspace_root=self.workspace_root,
            output_dir=output_dir,
            output_root=output_root,
            output_name=output_name,
            export_timestamp=self.export_timestamp,
        )
        self.output_root = self.run_dir.parent
        self.masks_out_dir = self.run_dir / "masks"
        self.typed_labels_out_dir = self.run_dir / "typed_labels"
        self.typed_metadata_out_dir = self.run_dir / "typed_metadata"
        self.session_manifest_out = self.run_dir / "manifest_edited.csv"
        self.session_metadata_out = self.run_dir / "export_metadata.json"

        # Load manifest
        self.df = pd.read_csv(self.manifest_path).reset_index(drop=True)
        print(f"Loaded manifest: {self.manifest_path} ({len(self.df)} rows)")
        if "dataset_id" not in self.df.columns or "stem" not in self.df.columns:
            raise ValueError("Manifest must contain columns: dataset_id, stem")

        # Find micrograph column (prefer downsampled image for editing if available)
        self.mic_col = None
        for col in [
            "down_mic_path",
            "full_mic_path",
            "micrograph_path",
            "mic_path",
            "resolved_mic_path",
            "staged_mic_path",
        ]:
            if col in self.df.columns:
                self.mic_col = col
                break
        if self.mic_col is None:
            raise ValueError("No micrograph path column found in manifest")
        print(f"Using micrograph column: {self.mic_col}")
        self.image_path_columns = [
            col
            for col in [
                self.mic_col,
                "full_mic_path",
                "down_mic_path",
                "micrograph_path",
                "mic_path",
                "resolved_mic_path",
                "staged_mic_path",
            ]
            if col in self.df.columns
        ]
        # Deduplicate while preserving order
        self.image_path_columns = list(dict.fromkeys(self.image_path_columns))

        self.gt_mask_path_columns = [
            col for col in DEFAULT_GT_MASK_PATH_COLUMNS if col in self.df.columns
        ]
        self.has_gt_col = bool(self.gt_mask_path_columns)
        self.has_typed_col = "typed_label_path" in self.df.columns

        # Precompute per-row keys and split labels
        self.dataset_ids: List[str] = [normalize_dataset_id(v) for v in self.df["dataset_id"].tolist()]
        self.stems: List[str] = [normalize_stem(v) for v in self.df["stem"].tolist()]
        self.keys: List[str] = [f"{did}__{stem}" for did, stem in zip(self.dataset_ids, self.stems)]
        self.row_idx_by_key: Dict[str, int] = {key: idx for idx, key in enumerate(self.keys)}

        self.val_manifest_path: Optional[Path] = None
        if val_manifest:
            vp = Path(val_manifest).expanduser()
            self.val_manifest_path = vp if vp.is_absolute() else (self.manifest_dir / vp)
        else:
            auto_vp = self.manifest_dir / DEFAULT_AUTO_VAL_MANIFEST
            if auto_vp.exists():
                self.val_manifest_path = auto_vp

        if self.val_manifest_path is not None and self.val_manifest_path.exists():
            self.split_labels, self.split_metadata = build_split_labels_from_val_manifest(
                self.keys, self.val_manifest_path
            )
            print(f"Split labels: using explicit val manifest {self.val_manifest_path}")
        elif "split" in self.df.columns or "split_label" in self.df.columns:
            self.split_labels, self.split_metadata = build_split_labels_from_manifest_column(
                self.df
            )
            print("Split labels: using manifest row-level split column")
        else:
            self.split_labels, self.split_metadata = build_split_labels_from_seed(
                self.dataset_ids, split_seed, split_train_frac
            )
            print(
                f"Split labels: derived from seed={split_seed}, train_frac={split_train_frac} "
                f"(matches export_heldout_val_split.py logic)"
            )

        train_rows = sum(1 for s in self.split_labels if s == "TRAINING")
        val_rows = sum(1 for s in self.split_labels if s == "VALIDATION")
        print(
            f"Rows by split: TRAINING={train_rows}, VALIDATION={val_rows}, TOTAL={len(self.df)}"
        )

        self.available_datasets = sorted({d for d in self.dataset_ids if d})
        print(f"Unique datasets in editor: {len(self.available_datasets)}")

        # Editor state
        self.modified_masks: Dict[str, np.ndarray] = {}
        self.modified_typed_masks: Dict[str, np.ndarray] = {}
        self.current_polygon: List[Tuple[float, float]] = []
        self.current_image: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_typed_mask: Optional[np.ndarray] = None
        self.current_display_limits: Optional[Tuple[float, float]] = None
        self.current_image_display: Optional[np.ndarray] = None
        self.current_display_stride: int = 1
        self.regions: List[np.ndarray] = []
        self.selected_region: int = -1
        self.current_row_idx: Optional[int] = None
        self.edit_mode = "GT mask"
        self.active_type_id: Optional[int] = None
        self.persisted_mask_keys: set[str] = set()
        self.persisted_typed_keys: set[str] = set()
        self.dirty_mask_keys: set[str] = set()
        self.dirty_typed_keys: set[str] = set()
        self._skip_close_event_checkpoint = False
        self._close_cleanup_done = False
        self._checkpoint_in_progress = False
        self.sort_mode = SORT_MODE_OPTIONS[0]
        self.sort_view_dirty = False
        self.row_sort_stats_cache: Dict[str, Dict[str, float]] = {}
        self.export_process: Optional[subprocess.Popen] = None
        self.export_status_detail: str = ""
        self.export_last_command: Optional[List[str]] = None
        self.show_mask_overlays = True

        self._ensure_session_dirs()
        self._refresh_session_inventory()

        # Optional Tk left-side image browser (dataset -> filename tree)
        self._tk_browser_ready = False
        self._tk_tree_item_to_row_idx: Dict[str, int] = {}
        self._tk_tree_dataset_item_ids: Dict[str, str] = {}
        self._tk_row_item_ids: Dict[int, str] = {}
        self._suspend_tk_tree_event = False
        self.tk_controls_enabled = False
        self.tk_dataset_var = None
        self.tk_status_var = None
        self.tk_sort_var = None
        self.tk_overlay_button = None
        self.tk_after_host = None
        self.tk_split_buttons: Dict[str, object] = {}
        self.tk_mode_buttons: Dict[str, object] = {}
        self.tk_type_buttons: Dict[Optional[int], object] = {}
        self._qt_browser_ready = False
        self.qt_controls_enabled = False
        self.qt_after_host = None
        self.qt_status_label = None
        self.qt_browser_status_label = None
        self.qt_view_info_label = None
        self.qt_dataset_edit = None
        self.qt_sort_combo = None
        self.qt_overlay_button = None
        self.qt_image_tree = None
        self.qt_image_view = None
        self._qt = None
        self._qtcore = None
        self._qtgui = None
        self._qt_row_items: Dict[int, object] = {}
        self._qt_dataset_items: Dict[str, object] = {}
        self._suspend_qt_tree_event = False
        self.qt_split_buttons: Dict[str, object] = {}
        self.qt_mode_buttons: Dict[str, object] = {}
        self.qt_type_buttons: Dict[Optional[int], object] = {}
        self.qt_shortcuts: List[object] = []

        # Filter/view state
        self.dataset_filter_text = initial_dataset_filter.strip()
        self.dataset_filter_ids = set(parse_dataset_filter_text(self.dataset_filter_text))
        self.initial_row_key = str(initial_row_key).strip()
        self.split_filter = initial_split.upper()
        if self.split_filter not in VALID_SPLIT_FILTERS:
            raise ValueError(f"initial_split must be one of {VALID_SPLIT_FILTERS}")
        self.view_indices: List[int] = []
        self.view_pos: int = 0

        # Setup figure + widgets
        self.fig, self.ax = self._create_editor_figure()
        self.fig.subplots_adjust(bottom=0.34)
        try:
            self.fig.canvas.manager.set_window_title("cryoFILTER Contamination Labeling Studio (Modern)")
        except Exception:
            pass

        self._build_widgets()
        disable_native_browser = os.environ.get("CRYOFILTER_DISABLE_NATIVE_BROWSER", "").strip().lower()
        if disable_native_browser in {"1", "true", "yes", "on"}:
            print("Native image browser disabled by CRYOFILTER_DISABLE_NATIVE_BROWSER")
        else:
            self._build_native_browser()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close_event)

        # Apply initial filters and load first image
        self._rebuild_view(preserve_current=False)

    def _build_widgets(self):
        """Create lightweight GUI controls under the plot."""
        self.widget_axes = []

        # Split filter
        ax_split = self.fig.add_axes([0.02, 0.11, 0.15, 0.19], facecolor="whitesmoke")
        self.widget_axes.append(ax_split)
        self.split_radio = RadioButtons(ax_split, VALID_SPLIT_FILTERS, active=0)
        self.split_radio.on_clicked(self._on_split_radio_changed)
        if self.split_filter in VALID_SPLIT_FILTERS:
            self.split_radio.set_active(VALID_SPLIT_FILTERS.index(self.split_filter))

        ax_mode = self.fig.add_axes([0.19, 0.11, 0.15, 0.19], facecolor="whitesmoke")
        self.widget_axes.append(ax_mode)
        self.mode_radio = RadioButtons(ax_mode, VALID_EDIT_MODES, active=0)
        self.mode_radio.on_clicked(self._on_mode_radio_changed)

        type_labels = ["None"] + [f"{type_id} {TYPE_ID_TO_DISPLAY[type_id]}" for type_id in TYPE_ID_SEQUENCE]
        ax_type = self.fig.add_axes([0.35, 0.11, 0.15, 0.19], facecolor="whitesmoke")
        self.widget_axes.append(ax_type)
        self.type_radio = RadioButtons(ax_type, type_labels, active=0)
        self.type_radio.on_clicked(self._on_type_radio_changed)

        # Dataset filter textbox + buttons
        ax_ds = self.fig.add_axes([0.52, 0.24, 0.23, 0.06])
        self.widget_axes.append(ax_ds)
        self.dataset_box = TextBox(ax_ds, "Dataset(s)", initial=self.dataset_filter_text)
        self.dataset_box.on_submit(self._on_dataset_submit)

        ax_apply = self.fig.add_axes([0.76, 0.24, 0.07, 0.06])
        self.widget_axes.append(ax_apply)
        self.btn_apply_dataset = Button(ax_apply, "Apply")
        self.btn_apply_dataset.on_clicked(self._on_apply_dataset_clicked)

        ax_clear = self.fig.add_axes([0.84, 0.24, 0.07, 0.06])
        self.widget_axes.append(ax_clear)
        self.btn_clear_dataset = Button(ax_clear, "Clear")
        self.btn_clear_dataset.on_clicked(self._on_clear_dataset_clicked)

        ax_prev_ds = self.fig.add_axes([0.52, 0.16, 0.11, 0.05])
        self.widget_axes.append(ax_prev_ds)
        self.btn_prev_dataset = Button(ax_prev_ds, "Prev Dataset")
        self.btn_prev_dataset.on_clicked(self._on_prev_dataset_clicked)

        ax_next_ds = self.fig.add_axes([0.64, 0.16, 0.11, 0.05])
        self.widget_axes.append(ax_next_ds)
        self.btn_next_dataset = Button(ax_next_ds, "Next Dataset")
        self.btn_next_dataset.on_clicked(self._on_next_dataset_clicked)

        ax_save = self.fig.add_axes([0.77, 0.16, 0.08, 0.05])
        self.widget_axes.append(ax_save)
        self.btn_save = Button(ax_save, "Save")
        self.btn_save.on_clicked(self._on_save_clicked)

        ax_export = self.fig.add_axes([0.86, 0.16, 0.11, 0.05])
        self.widget_axes.append(ax_export)
        self.btn_export = Button(ax_export, "Export (stay)")
        self.btn_export.on_clicked(self._on_export_clicked)

        ax_help = self.fig.add_axes([0.52, 0.08, 0.45, 0.05])
        self.widget_axes.append(ax_help)
        self.help_button = Button(ax_help, "Mode: b/x/f/l | Types: 1-4, 0/u | Edit: d/c/r | Nav: n/p, [/] | s/e/q")
        self.help_button.label.set_fontsize(8)
        # No callback needed; informational only.
        self.help_button.on_clicked(lambda _evt: None)

    def _set_tk_button_palette(self, button, *, selected: bool, fill: str, fg: str = "#0f172a") -> None:
        if button is None:
            return
        if selected:
            button.configure(bg=fill, fg="white", activebackground=fill, activeforeground="white")
        else:
            button.configure(bg="#f4efe6", fg=fg, activebackground="#e8dfcf", activeforeground=fg)

    def _set_qt_button_palette(self, button, *, selected: bool, fill: str, fg: str = "#0f172a") -> None:
        if button is None:
            return
        if selected:
            button.setStyleSheet(
                f"QPushButton {{ background-color: {fill}; color: white; border: 1px solid {fill}; border-radius: 9px; }}"
            )
            try:
                button.setChecked(True)
            except Exception:
                pass
        else:
            button.setStyleSheet(
                f"QPushButton {{ background-color: #f4efe6; color: {fg}; border: 1px solid #e7ddd0; border-radius: 9px; }}"
                f"QPushButton:hover {{ background-color: #e8dfcf; color: {fg}; border: 1px solid #e0d2be; }}"
            )
            try:
                button.setChecked(False)
            except Exception:
                pass

    def _current_status_text(self) -> str:
        export_suffix = ""
        if self.export_process is not None:
            export_suffix = " | export=running"
        elif self.export_status_detail:
            export_suffix = f" | export={self.export_status_detail}"
        overlay_suffix = "on" if self.show_mask_overlays else "off"
        if self.row_loading and self.current_row_idx is not None:
            return (
                f"Loading {self.keys[self.current_row_idx]} | "
                f"view {self.view_pos + 1}/{len(self.view_indices)} | "
                f"cache={len(self.row_payload_cache)}/{self.row_cache_size} | "
                f"prefetch={self.prefetch_workers}x | masks={overlay_suffix}"
                f"{export_suffix}"
            )
        if self.current_row_idx is None:
            return f"No active image | masks={overlay_suffix}{export_suffix}"

        completion = 0.0
        unlabeled_px = 0
        if self.current_mask is not None:
            stats = typed_completion_stats(
                self.current_mask,
                self.current_typed_mask
                if self.current_typed_mask is not None
                else build_unlabeled_typed_mask(self.current_mask),
            )
            completion = 100.0 * float(stats["typed_completion_fraction"])
            unlabeled_px = int(stats["typed_unlabeled_px"])
        return (
            f"{self.keys[self.current_row_idx]} | split={self.split_labels[self.current_row_idx]} | "
            f"mode={self.edit_mode} | unlabeled_px={unlabeled_px} | "
            f"typed={completion:.0f}% | masks={overlay_suffix} | "
            f"cache={len(self.row_payload_cache)}/{self.row_cache_size} | "
            f"prefetch={self.prefetch_workers}x"
            f"{export_suffix}"
        )

    def _invalidate_row_sort_stats(self, row_idx: Optional[int]) -> None:
        if row_idx is None or row_idx < 0 or row_idx >= len(self.keys):
            return
        self.row_sort_stats_cache.pop(self.keys[row_idx], None)

    def _row_sort_stats(self, row_idx: int) -> Dict[str, float]:
        key = self.keys[row_idx]
        cached = self.row_sort_stats_cache.get(key)
        if cached is not None:
            return cached

        stats_payload: Dict[str, float]
        if row_idx == self.current_row_idx and self.current_mask is not None:
            typed = (
                self.current_typed_mask
                if self.current_typed_mask is not None
                else build_unlabeled_typed_mask(self.current_mask)
            )
            stats = typed_completion_stats(self.current_mask, typed)
            total_px = float(np.asarray(self.current_mask).size)
            stats_payload = {
                "typed_completion_fraction": float(stats["typed_completion_fraction"]),
                "typed_unlabeled_px": float(stats["typed_unlabeled_px"]),
                "gt_contamination_px": float(stats["gt_contamination_px"]),
                "labeled_fraction": float(stats["gt_contamination_px"] / total_px) if total_px > 0 else 0.0,
            }
        elif key in self.modified_masks:
            mask = self.modified_masks[key]
            typed = self.modified_typed_masks.get(key, build_unlabeled_typed_mask(mask))
            stats = typed_completion_stats(mask, typed)
            total_px = float(np.asarray(mask).size)
            stats_payload = {
                "typed_completion_fraction": float(stats["typed_completion_fraction"]),
                "typed_unlabeled_px": float(stats["typed_unlabeled_px"]),
                "gt_contamination_px": float(stats["gt_contamination_px"]),
                "labeled_fraction": float(stats["gt_contamination_px"] / total_px) if total_px > 0 else 0.0,
            }
        else:
            meta_path = self._session_typed_metadata_path_for_key(key)
            if meta_path.exists():
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    stats = payload.get("stats", {})
                    counts = payload.get("counts", {})
                    total_px = float(sum(int(v) for v in counts.values())) if counts else 0.0
                    gt_px = float(stats.get("gt_contamination_px", 0.0))
                    stats_payload = {
                        "typed_completion_fraction": float(stats.get("typed_completion_fraction", 1.0 if gt_px <= 0 else 0.0)),
                        "typed_unlabeled_px": float(stats.get("typed_unlabeled_px", 0.0)),
                        "gt_contamination_px": gt_px,
                        "labeled_fraction": float(gt_px / total_px) if total_px > 0 else 0.0,
                    }
                except Exception:
                    stats_payload = {}
            else:
                stats_payload = {}

            if not stats_payload:
                mask, _mask_source = self._load_mask_for_export(row_idx)
                if mask is None:
                    stats_payload = {
                        "typed_completion_fraction": 1.0,
                        "typed_unlabeled_px": 0.0,
                        "gt_contamination_px": 0.0,
                        "labeled_fraction": 0.0,
                    }
                else:
                    typed_mask, _typed_source = self._load_typed_mask_for_export(row_idx, binary_mask=mask)
                    typed_mask = (
                        reconcile_typed_mask(mask, typed_mask)
                        if typed_mask is not None
                        else build_unlabeled_typed_mask(mask)
                    )
                    stats = typed_completion_stats(mask, typed_mask)
                    total_px = float(np.asarray(mask).size)
                    stats_payload = {
                        "typed_completion_fraction": float(stats["typed_completion_fraction"]),
                        "typed_unlabeled_px": float(stats["typed_unlabeled_px"]),
                        "gt_contamination_px": float(stats["gt_contamination_px"]),
                        "labeled_fraction": float(stats["gt_contamination_px"] / total_px) if total_px > 0 else 0.0,
                    }

        self.row_sort_stats_cache[key] = stats_payload
        return stats_payload

    def _row_browser_label(self, row_idx: int) -> str:
        label = self._row_display_name(row_idx)
        if self._key_has_session_artifacts(self.keys[row_idx]):
            label = "* " + label
        if self.split_filter == "ALL":
            label = f"{label} [{self.split_labels[row_idx][0]}]"
        if self.sort_mode != SORT_MODE_OPTIONS[0]:
            stats = self._row_sort_stats(row_idx)
            label = (
                f"{label} | typed {100.0 * float(stats['typed_completion_fraction']):.0f}%"
                f" | labeled {100.0 * float(stats['labeled_fraction']):.1f}%"
                f" | untyped_px {int(stats['typed_unlabeled_px'])}"
            )
        return label

    def _sort_key_for_row(self, row_idx: int):
        if self.sort_mode == SORT_MODE_OPTIONS[0]:
            return (self._row_display_name(row_idx).lower(), row_idx)

        stats = self._row_sort_stats(row_idx)
        typed = float(stats["typed_completion_fraction"])
        labeled = float(stats["labeled_fraction"])
        unlabeled_px = float(stats["typed_unlabeled_px"])
        name_key = self._row_display_name(row_idx).lower()

        if self.sort_mode == SORT_MODE_OPTIONS[1]:
            return (typed, -unlabeled_px, name_key)
        if self.sort_mode == SORT_MODE_OPTIONS[2]:
            return (-typed, -unlabeled_px, name_key)
        if self.sort_mode == SORT_MODE_OPTIONS[3]:
            return (-labeled, typed, name_key)
        if self.sort_mode == SORT_MODE_OPTIONS[4]:
            return (labeled, typed, name_key)
        if self.sort_mode == SORT_MODE_OPTIONS[5]:
            return (-unlabeled_px, typed, name_key)
        return (name_key, row_idx)

    def _count_rows_with_untyped_pixels(self, indices: Sequence[int]) -> int:
        count = 0
        for row_idx in indices:
            if float(self._row_sort_stats(int(row_idx))["typed_unlabeled_px"]) > 0.0:
                count += 1
        return count

    def _browser_status_text(self) -> str:
        if not self.view_indices:
            return f"No rows | split={self.split_filter} | dataset={self.dataset_filter_text or 'ALL'}"
        dataset_count = len({self.dataset_ids[row_idx] for row_idx in self.view_indices})
        status = (
            f"{len(self.view_indices)} images | {dataset_count} datasets | "
            f"split={self.split_filter} | dataset={self.dataset_filter_text or 'ALL'} | "
            f"sort={self.sort_mode}"
        )
        if self.sort_mode != SORT_MODE_OPTIONS[0]:
            remaining_untyped = self._count_rows_with_untyped_pixels(self.view_indices)
            status = f"{status} | remaining_untyped={remaining_untyped}"
        return status

    def _force_canvas_redraw(self) -> None:
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass

    def _schedule_qt_redraw(self) -> None:
        if not self._qt_browser_ready or self.qt_after_host is None:
            return
        try:
            self._qtcore.QTimer.singleShot(0, self._force_canvas_redraw)
            self._qtcore.QTimer.singleShot(120, self._force_canvas_redraw)
        except Exception:
            pass

    def _sync_tk_control_state(self) -> None:
        if self.tk_controls_enabled:
            if self.tk_dataset_var is not None:
                current_text = self.dataset_filter_text or ""
                if self.tk_dataset_var.get() != current_text:
                    self.tk_dataset_var.set(current_text)
            if self.tk_sort_var is not None and self.tk_sort_var.get() != self.sort_mode:
                self.tk_sort_var.set(self.sort_mode)

            for split_label, button in self.tk_split_buttons.items():
                self._set_tk_button_palette(button, selected=(self.split_filter == split_label), fill="#155e75")

            for mode_label, button in self.tk_mode_buttons.items():
                self._set_tk_button_palette(button, selected=(self.edit_mode == mode_label), fill="#0f766e")

            for type_id, button in self.tk_type_buttons.items():
                if type_id is None:
                    self._set_tk_button_palette(button, selected=(self.active_type_id is None), fill="#64748b")
                    continue
                self._set_tk_button_palette(
                    button,
                    selected=(self.active_type_id == type_id),
                    fill=TYPE_ID_TO_COLOR_HEX[int(type_id)],
                )

            if self.tk_status_var is not None:
                self.tk_status_var.set(self._current_status_text())
            if self.tk_overlay_button is not None:
                self.tk_overlay_button.configure(text="Masks On" if self.show_mask_overlays else "Masks Off")

        if self.qt_controls_enabled:
            if self.qt_dataset_edit is not None:
                current_text = self.dataset_filter_text or ""
                if self.qt_dataset_edit.text() != current_text:
                    self.qt_dataset_edit.setText(current_text)
            if self.qt_sort_combo is not None and self.qt_sort_combo.currentText() != self.sort_mode:
                self.qt_sort_combo.setCurrentText(self.sort_mode)

            for split_label, button in self.qt_split_buttons.items():
                self._set_qt_button_palette(button, selected=(self.split_filter == split_label), fill="#155e75")

            for mode_label, button in self.qt_mode_buttons.items():
                self._set_qt_button_palette(button, selected=(self.edit_mode == mode_label), fill="#0f766e")

            for type_id, button in self.qt_type_buttons.items():
                if type_id is None:
                    self._set_qt_button_palette(button, selected=(self.active_type_id is None), fill="#64748b")
                    continue
                self._set_qt_button_palette(
                    button,
                    selected=(self.active_type_id == type_id),
                    fill=TYPE_ID_TO_COLOR_HEX[int(type_id)],
                )

            if self.qt_status_label is not None:
                self.qt_status_label.setText(self._current_status_text())
            if self.qt_overlay_button is not None:
                self.qt_overlay_button.setText("Masks On" if self.show_mask_overlays else "Masks Off")

    def _build_tk_modern_toolbar(self, parent):
        tk = self._tk

        toolbar = tk.Frame(parent, bg="#f3efe7", padx=14, pady=10)
        toolbar.pack(side="top", fill="x")

        title = tk.Label(
            toolbar,
            text="cryoFILTER Contamination Labeling Studio",
            bg="#f3efe7",
            fg="#0f172a",
            font=("TkDefaultFont", 13, "bold"),
            anchor="w",
        )
        title.pack(side="top", anchor="w")

        self.tk_status_var = tk.StringVar(value="Initializing modern labeling workspace...")
        status = tk.Label(
            toolbar,
            textvariable=self.tk_status_var,
            bg="#f3efe7",
            fg="#475569",
            font=("TkDefaultFont", 10),
            anchor="w",
        )
        status.pack(side="top", anchor="w", pady=(3, 8))

        row1 = tk.Frame(toolbar, bg="#f3efe7")
        row1.pack(side="top", fill="x", pady=(0, 6))
        row2 = tk.Frame(toolbar, bg="#f3efe7")
        row2.pack(side="top", fill="x")

        def make_group(host, title_text: str):
            outer = tk.Frame(host, bg="#f3efe7")
            outer.pack(side="left", padx=(0, 12))
            tk.Label(
                outer,
                text=title_text,
                bg="#f3efe7",
                fg="#64748b",
                font=("TkDefaultFont", 9, "bold"),
            ).pack(side="top", anchor="w", pady=(0, 4))
            buttons = tk.Frame(outer, bg="#f3efe7")
            buttons.pack(side="top", anchor="w")
            return buttons

        split_group = make_group(row1, "Split")
        for split_label in VALID_SPLIT_FILTERS:
            button = tk.Button(
                split_group,
                text=split_label.title(),
                relief="flat",
                bd=0,
                padx=12,
                pady=7,
                font=("TkDefaultFont", 10, "bold"),
                command=lambda lbl=split_label: self._set_split_filter(lbl),
            )
            button.pack(side="left", padx=(0, 6))
            self.tk_split_buttons[split_label] = button

        mode_group = make_group(row1, "Mode")
        for mode_label in VALID_EDIT_MODES:
            short = {
                "GT mask": "GT Mask",
                "Erase polygon": "Erase Poly",
                "Full label": "Full Label",
                "Polygon label": "Polygon Label",
            }[mode_label]
            button = tk.Button(
                mode_group,
                text=short,
                relief="flat",
                bd=0,
                padx=12,
                pady=7,
                font=("TkDefaultFont", 10, "bold"),
                command=lambda lbl=mode_label: self._set_edit_mode(lbl),
            )
            button.pack(side="left", padx=(0, 6))
            self.tk_mode_buttons[mode_label] = button

        action_group = make_group(row1, "Navigate / Save")
        action_specs = [
            ("Prev", lambda: self._step(-1)),
            ("Next", lambda: self._step(1)),
            ("Prev DS", lambda: self._step_dataset(-1)),
            ("Next DS", lambda: self._step_dataset(1)),
            ("Masks On", self._toggle_mask_overlays),
            ("Redraw GT", self._reset_current_row_for_redraw),
            ("Save Row", self._save_current),
            ("Finalize Export", self._start_headless_export),
        ]
        for label, callback in action_specs:
            button = tk.Button(
                action_group,
                text=label,
                relief="flat",
                bd=0,
                padx=12,
                pady=7,
                bg="#111827" if "Save" in label or "Export" in label else "#f4efe6",
                fg="white" if "Save" in label or "Export" in label else "#0f172a",
                activebackground="#1f2937" if "Save" in label or "Export" in label else "#e8dfcf",
                activeforeground="white" if "Save" in label or "Export" in label else "#0f172a",
                font=("TkDefaultFont", 10, "bold"),
                command=callback,
            )
            button.pack(side="left", padx=(0, 6))
            if label.startswith("Masks"):
                self.tk_overlay_button = button

        type_group = make_group(row2, "Subtype")
        clear_button = tk.Button(
            type_group,
            text="Unlabeled",
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            font=("TkDefaultFont", 10, "bold"),
            command=lambda: self._set_active_type_id(None),
        )
        clear_button.pack(side="left", padx=(0, 6))
        self.tk_type_buttons[None] = clear_button
        for type_id in TYPE_ID_SEQUENCE:
            button = tk.Button(
                type_group,
                text=TYPE_ID_TO_DISPLAY[type_id],
                relief="flat",
                bd=0,
                padx=12,
                pady=8,
                font=("TkDefaultFont", 10, "bold"),
                command=lambda tid=type_id: self._set_active_type_id(tid),
            )
            button.pack(side="left", padx=(0, 6))
            self.tk_type_buttons[type_id] = button

        tips_group = make_group(row2, "Fast Flow")
        tips = tk.Label(
            tips_group,
            text="Full label: choose type, click GT region. Polygon: choose type, draw, right-click.",
            bg="#f3efe7",
            fg="#334155",
            justify="left",
        )
        tips.pack(side="left")

        sort_group = make_group(row2, "Sort Browser")
        self.tk_sort_var = tk.StringVar(value=self.sort_mode)
        try:
            sort_menu = tk.OptionMenu(sort_group, self.tk_sort_var, *SORT_MODE_OPTIONS, command=self._on_sort_option_changed)
            sort_menu.configure(
                relief="flat",
                bd=0,
                padx=8,
                pady=6,
                bg="#f4efe6",
                fg="#0f172a",
                activebackground="#e8dfcf",
                activeforeground="#0f172a",
                highlightthickness=0,
            )
            sort_menu["menu"].configure(bg="white", fg="#0f172a")
            sort_menu.pack(side="left")
        except Exception:
            pass

    def _build_native_browser(self):
        if self._build_qt_left_browser():
            return
        self._build_tk_left_browser()

    def _display_needs_reload_for_row(self, row_idx: int) -> bool:
        if self.current_row_idx != int(row_idx):
            return True
        if self.row_loading:
            return False
        if self.current_image is None or self.current_mask is None:
            return True
        if self._qt_browser_ready and self.qt_image_view is not None:
            pixmap = getattr(self.qt_image_view, "_pixmap", None)
            return bool(pixmap is None or pixmap.isNull())
        if self.current_image_display is None:
            return True
        return False

    def _select_browser_row_idx(self, row_idx: int, *, force_reload: bool = False) -> None:
        try:
            new_pos = self.view_indices.index(int(row_idx))
        except ValueError:
            return

        if force_reload and self.row_loading and self.current_row_idx == int(row_idx):
            return

        if force_reload or self._display_needs_reload_for_row(int(row_idx)):
            print(f"  Loading selected image: {self.keys[int(row_idx)]}", flush=True)
            self.view_pos = new_pos
            if force_reload:
                self._invalidate_row_payload(int(row_idx))
            self._load_current()
            return

        self.view_pos = new_pos

    def _qt_row_idx_from_item(self, item) -> Optional[int]:
        if item is None or self._qtcore is None:
            return None
        row_idx = item.data(0, self._qtcore.Qt.ItemDataRole.UserRole)
        if row_idx is None and item.childCount() > 0:
            item = item.child(0)
            row_idx = item.data(0, self._qtcore.Qt.ItemDataRole.UserRole)
        if row_idx is None:
            return None
        return int(row_idx)

    def _build_qt_modern_toolbar(self, host_layout) -> None:
        qt = self._qt

        toolbar = qt.QFrame()
        toolbar.setStyleSheet("QFrame { background: #f3efe7; border-radius: 16px; }")
        toolbar.setSizePolicy(qt.QSizePolicy.Policy.Preferred, qt.QSizePolicy.Policy.Maximum)
        toolbar.setMaximumHeight(250)
        toolbar_layout = qt.QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 14, 16, 14)
        toolbar_layout.setSpacing(6)
        host_layout.addWidget(toolbar)

        title = qt.QLabel("cryoFILTER Contamination Labeling Studio")
        title.setStyleSheet("color: #0f172a; font-size: 16px; font-weight: 700;")
        toolbar_layout.addWidget(title)

        self.qt_status_label = qt.QLabel("Initializing modern labeling workspace...")
        self.qt_status_label.setWordWrap(True)
        self.qt_status_label.setStyleSheet("color: #475569; font-size: 11px;")
        toolbar_layout.addWidget(self.qt_status_label)

        row1 = qt.QHBoxLayout()
        row1.setSpacing(12)
        toolbar_layout.addLayout(row1)
        row2 = qt.QHBoxLayout()
        row2.setSpacing(12)
        toolbar_layout.addLayout(row2)

        def make_group(host, title_text: str):
            outer = qt.QWidget()
            outer.setSizePolicy(qt.QSizePolicy.Policy.Maximum, qt.QSizePolicy.Policy.Maximum)
            outer_layout = qt.QVBoxLayout(outer)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_layout.setSpacing(3)
            label = qt.QLabel(title_text)
            label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700;")
            outer_layout.addWidget(label)
            inner = qt.QHBoxLayout()
            inner.setSpacing(5)
            outer_layout.addLayout(inner)
            host.addWidget(outer)
            return inner

        split_group = make_group(row1, "Split")
        for split_label in VALID_SPLIT_FILTERS:
            button = qt.QPushButton(split_label.title())
            button.setMinimumHeight(34)
            button.clicked.connect(lambda _checked=False, lbl=split_label: self._set_split_filter(lbl))
            split_group.addWidget(button)
            self.qt_split_buttons[split_label] = button

        mode_group = make_group(row1, "Mode")
        for mode_label in VALID_EDIT_MODES:
            short = {
                "GT mask": "GT Mask",
                "Erase polygon": "Erase Poly",
                "Full label": "Full Label",
                "Polygon label": "Polygon Label",
            }[mode_label]
            button = qt.QPushButton(short)
            button.setMinimumHeight(34)
            button.clicked.connect(lambda _checked=False, lbl=mode_label: self._set_edit_mode(lbl))
            mode_group.addWidget(button)
            self.qt_mode_buttons[mode_label] = button

        action_group = make_group(row1, "Navigate / Save")
        action_specs = [
            ("Prev", lambda: self._step(-1)),
            ("Next", lambda: self._step(1)),
            ("Prev DS", lambda: self._step_dataset(-1)),
            ("Next DS", lambda: self._step_dataset(1)),
            ("Masks On", self._toggle_mask_overlays),
            ("Redraw GT", self._reset_current_row_for_redraw),
            ("Save Row", self._save_current),
            ("Finalize Export", self._start_headless_export),
        ]
        for label, callback in action_specs:
            button = qt.QPushButton(label)
            button.setMinimumHeight(34)
            if "Save" in label or "Export" in label:
                button.setStyleSheet(
                    "QPushButton { background-color: #111827; color: white; border: 1px solid #111827; border-radius: 9px; }"
                    "QPushButton:hover { background-color: #1f2937; border: 1px solid #1f2937; }"
                )
            else:
                self._set_qt_button_palette(button, selected=False, fill="#111827")
            button.clicked.connect(lambda _checked=False, cb=callback: cb())
            action_group.addWidget(button)
            if label.startswith("Masks"):
                self.qt_overlay_button = button

        type_group = make_group(row2, "Subtype")
        clear_button = qt.QPushButton("Unlabeled")
        clear_button.setMinimumHeight(34)
        clear_button.clicked.connect(lambda _checked=False: self._set_active_type_id(None))
        type_group.addWidget(clear_button)
        self.qt_type_buttons[None] = clear_button
        for type_id in TYPE_ID_SEQUENCE:
            button = qt.QPushButton(TYPE_ID_TO_DISPLAY[type_id])
            button.setMinimumHeight(34)
            button.clicked.connect(lambda _checked=False, tid=type_id: self._set_active_type_id(tid))
            type_group.addWidget(button)
            self.qt_type_buttons[type_id] = button

        tips_group = make_group(row2, "Fast Flow")
        tips = qt.QLabel(
            "Full label: choose type, click GT region. Polygon: choose type, draw, right-click."
        )
        tips.setWordWrap(False)
        tips.setStyleSheet("color: #334155; font-size: 11px;")
        tips_group.addWidget(tips)

        sort_group = make_group(row2, "Sort Browser")
        self.qt_sort_combo = qt.QComboBox()
        self.qt_sort_combo.addItems(list(SORT_MODE_OPTIONS))
        self.qt_sort_combo.setCurrentText(self.sort_mode)
        self.qt_sort_combo.currentTextChanged.connect(self._on_sort_option_changed)
        self.qt_sort_combo.setStyleSheet(
            "QComboBox { background: white; border: 1px solid #d6d3d1; border-radius: 9px; padding: 6px 10px; }"
        )
        sort_group.addWidget(self.qt_sort_combo)
        row1.addStretch(1)
        row2.addStretch(1)

    def _build_qt_left_browser(self) -> bool:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception:
            return False

        manager = getattr(self.fig.canvas, "manager", None)
        window = getattr(manager, "window", None)
        canvas_widget = self.fig.canvas
        if window is None or canvas_widget is None or not hasattr(canvas_widget, "setParent"):
            return False

        self._qt = QtWidgets
        self._qtcore = QtCore
        self._qtgui = QtGui
        self.qt_after_host = window

        class _QtImageView(QtWidgets.QWidget):
            def __init__(self, editor):
                super().__init__()
                self._editor = editor
                self._pixmap = None
                self._pixmap_rect = QtCore.QRect()
                self.setMinimumSize(640, 640)
                self.setMouseTracking(True)
                self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
                self.setStyleSheet("background: white; border: 1px solid #e7e5e4; border-radius: 12px;")

            def set_pixmap(self, pixmap):
                self._pixmap = pixmap
                self.update()

            def _target_rect(self):
                if self._pixmap is None or self._pixmap.isNull():
                    return QtCore.QRect()
                size = self._pixmap.size()
                scaled = size.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                x = max(0, (self.width() - scaled.width()) // 2)
                y = max(0, (self.height() - scaled.height()) // 2)
                return QtCore.QRect(x, y, scaled.width(), scaled.height())

            def paintEvent(self, event):
                super().paintEvent(event)
                painter = QtGui.QPainter(self)
                painter.fillRect(self.rect(), QtGui.QColor("#ffffff"))
                if self._pixmap is not None and not self._pixmap.isNull():
                    self._pixmap_rect = self._target_rect()
                    painter.drawPixmap(self._pixmap_rect, self._pixmap)
                else:
                    self._pixmap_rect = QtCore.QRect()
                    painter.setPen(QtGui.QColor("#64748b"))
                    painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Loading image...")
                painter.end()

            def mousePressEvent(self, event):
                if self._pixmap is None or self._pixmap.isNull() or self._pixmap_rect.isNull():
                    return
                if not self._pixmap_rect.contains(event.position().toPoint()):
                    return

                px = event.position().x() - float(self._pixmap_rect.x())
                py = event.position().y() - float(self._pixmap_rect.y())
                if self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
                    return
                scale_x = self._pixmap.width() / float(self._pixmap_rect.width())
                scale_y = self._pixmap.height() / float(self._pixmap_rect.height())
                x_disp = px * scale_x
                y_disp = py * scale_y
                mapped = None
                if (
                    event.button() == QtCore.Qt.MouseButton.LeftButton
                    and bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
                ):
                    mapped = 3
                else:
                    button_map = {
                        QtCore.Qt.MouseButton.LeftButton: 1,
                        QtCore.Qt.MouseButton.MiddleButton: 2,
                        QtCore.Qt.MouseButton.RightButton: 3,
                    }
                    mapped = button_map.get(event.button())
                if mapped is not None:
                    self._editor._handle_canvas_click(mapped, x_disp, y_disp)
                try:
                    self.setFocus()
                except Exception:
                    pass
                event.accept()

            def mouseDoubleClickEvent(self, event):
                if self._pixmap is None or self._pixmap.isNull() or self._pixmap_rect.isNull():
                    return
                if not self._pixmap_rect.contains(event.position().toPoint()):
                    return
                if event.button() != QtCore.Qt.MouseButton.LeftButton:
                    return
                if self._editor._is_full_label_mode():
                    return

                px = event.position().x() - float(self._pixmap_rect.x())
                py = event.position().y() - float(self._pixmap_rect.y())
                if self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
                    return
                scale_x = self._pixmap.width() / float(self._pixmap_rect.width())
                scale_y = self._pixmap.height() / float(self._pixmap_rect.height())
                x_disp = px * scale_x
                y_disp = py * scale_y
                self._editor._handle_canvas_click(3, x_disp, y_disp)
                try:
                    self.setFocus()
                except Exception:
                    pass
                event.accept()

        try:
            shell = QtWidgets.QWidget()
            shell.setStyleSheet("background: #ede7dc;")
            root = QtWidgets.QHBoxLayout(shell)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(12)

            left_panel = QtWidgets.QFrame()
            left_panel.setFixedWidth(360)
            left_panel.setStyleSheet("QFrame { background: #fbfaf6; border-radius: 16px; }")
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            left_layout.setContentsMargins(14, 14, 14, 14)
            left_layout.setSpacing(10)
            root.addWidget(left_panel)

            right_panel = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(10)
            root.addWidget(right_panel, 1)

            window.setCentralWidget(shell)
            if getattr(manager, "toolbar", None) is not None:
                try:
                    manager.toolbar.hide()
                except Exception:
                    pass

            self._build_qt_modern_toolbar(right_layout)

            self.qt_view_info_label = QtWidgets.QLabel("")
            self.qt_view_info_label.setWordWrap(True)
            self.qt_view_info_label.setStyleSheet("color: #334155; font-size: 11px;")
            right_layout.addWidget(self.qt_view_info_label)

            self.qt_image_view = _QtImageView(self)
            self.qt_image_view.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            right_layout.addWidget(self.qt_image_view, 1)

            canvas_widget.setParent(right_panel)
            canvas_widget.hide()
            try:
                self.qt_image_view.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
                self.qt_image_view.setFocus()
            except Exception:
                pass
            right_layout.setStretch(0, 0)
            right_layout.setStretch(1, 0)
            right_layout.setStretch(2, 1)

            title = QtWidgets.QLabel("Image Browser")
            title.setStyleSheet("color: #0f172a; font-size: 15px; font-weight: 700;")
            left_layout.addWidget(title)

            hint = QtWidgets.QLabel(
                "Choose datasets, jump between files, and keep labeling without leaving the keyboard."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #475569; font-size: 11px;")
            left_layout.addWidget(hint)

            filter_label = QtWidgets.QLabel("Dataset Filter")
            filter_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700;")
            left_layout.addWidget(filter_label)

            self.qt_dataset_edit = QtWidgets.QLineEdit(self.dataset_filter_text)
            self.qt_dataset_edit.setPlaceholderText("dataset ids, comma or space separated")
            self.qt_dataset_edit.returnPressed.connect(self._apply_dataset_filter_from_widget)
            self.qt_dataset_edit.setStyleSheet(
                "QLineEdit { background: white; border: 1px solid #d6d3d1; border-radius: 9px; padding: 8px; }"
            )
            left_layout.addWidget(self.qt_dataset_edit)

            filter_buttons = QtWidgets.QHBoxLayout()
            filter_buttons.setSpacing(6)
            apply_button = QtWidgets.QPushButton("Apply Filter")
            apply_button.setMinimumHeight(34)
            apply_button.setStyleSheet(
                "QPushButton { background-color: #0f766e; color: white; border: 1px solid #0f766e; border-radius: 9px; }"
                "QPushButton:hover { background-color: #115e59; border: 1px solid #115e59; }"
            )
            apply_button.clicked.connect(self._apply_dataset_filter_from_widget)
            filter_buttons.addWidget(apply_button)
            clear_button = QtWidgets.QPushButton("Clear")
            clear_button.setMinimumHeight(34)
            self._set_qt_button_palette(clear_button, selected=False, fill="#64748b")
            clear_button.clicked.connect(lambda _checked=False: self._on_clear_dataset_clicked(None))
            filter_buttons.addWidget(clear_button)
            filter_buttons.addStretch(1)
            left_layout.addLayout(filter_buttons)

            self.qt_browser_status_label = QtWidgets.QLabel("Initializing...")
            self.qt_browser_status_label.setWordWrap(True)
            self.qt_browser_status_label.setStyleSheet("color: #475569; font-size: 11px;")
            left_layout.addWidget(self.qt_browser_status_label)

            self.qt_image_tree = QtWidgets.QTreeWidget()
            self.qt_image_tree.setHeaderHidden(True)
            self.qt_image_tree.setUniformRowHeights(True)
            self.qt_image_tree.setAnimated(False)
            self.qt_image_tree.itemSelectionChanged.connect(self._on_qt_tree_select)
            self.qt_image_tree.itemClicked.connect(self._on_qt_tree_item_clicked)
            self.qt_image_tree.itemActivated.connect(self._on_qt_tree_item_clicked)
            self.qt_image_tree.setStyleSheet(
                "QTreeWidget { background: white; border: 1px solid #e7e5e4; border-radius: 10px; }"
            )
            left_layout.addWidget(self.qt_image_tree, 1)

            shortcut_map = [
                ("n", "n"), ("p", "p"), ("[", "["), ("]", "]"),
                ("b", "b"), ("x", "x"), ("f", "f"), ("l", "l"),
                ("0", "0"), ("u", "u"),
                ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"),
                ("a", "a"), ("t", "t"), ("v", "v"),
                ("s", "s"), ("e", "e"), ("q", "q"),
                ("d", "d"), ("c", "c"), ("r", "r"), ("Escape", "escape"),
                ("Return", "enter"), ("Enter", "enter"),
            ]
            self.qt_shortcuts = []
            for sequence, key_name in shortcut_map:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), window)
                shortcut.activated.connect(lambda kn=key_name: self._on_key(SimpleNamespace(key=kn)))
                self.qt_shortcuts.append(shortcut)

            self._qt_browser_ready = True
            self.qt_controls_enabled = True
            for ax in getattr(self, "widget_axes", []):
                try:
                    ax.set_visible(False)
                except Exception:
                    pass
            self.fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.02)
            self._sync_tk_control_state()
            self._schedule_qt_redraw()
            return True
        except Exception as exc:
            print(f"Qt image browser disabled (PySide6 shell setup failed): {exc}")
            self._qt_browser_ready = False
            self.qt_controls_enabled = False
            return False

    def _on_qt_tree_select(self) -> None:
        if not self._qt_browser_ready or self._suspend_qt_tree_event or self.qt_image_tree is None:
            return

        selected = self.qt_image_tree.selectedItems()
        if not selected:
            return
        row_idx = self._qt_row_idx_from_item(selected[0])
        if row_idx is None:
            return
        self._select_browser_row_idx(int(row_idx))

    def _on_qt_tree_item_clicked(self, item, _column=0) -> None:
        if not self._qt_browser_ready or self.qt_image_tree is None:
            return
        row_idx = self._qt_row_idx_from_item(item)
        if row_idx is None:
            return
        self._select_browser_row_idx(int(row_idx))

    def _on_sort_option_changed(self, value: str) -> None:
        self._set_sort_mode(str(value).strip())

    def _set_sort_mode(self, sort_mode: str) -> None:
        if sort_mode not in SORT_MODE_OPTIONS:
            return
        if sort_mode == self.sort_mode:
            return
        self.sort_mode = sort_mode
        self._sync_tk_control_state()
        self._rebuild_view(preserve_current=True)

    def _mark_sorted_view_dirty(self) -> None:
        if self.sort_mode != SORT_MODE_OPTIONS[0]:
            self.sort_view_dirty = True

    def _maybe_refresh_sorted_view(self) -> None:
        if self.sort_mode == SORT_MODE_OPTIONS[0] or not self.sort_view_dirty:
            return
        self._rebuild_view(preserve_current=True)

    def _toggle_mask_overlays(self) -> None:
        self.show_mask_overlays = not self.show_mask_overlays
        self._sync_tk_control_state()
        self._update_display()

    def _build_tk_left_browser(self):
        """
        Add a fast, collapsible dataset/file selector on the left (Tk Treeview).

        Only enabled when running with the TkAgg backend.
        """
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:
            print(f"Tk image browser disabled (tkinter unavailable): {exc}")
            return

        manager = getattr(self.fig.canvas, "manager", None)
        window = getattr(manager, "window", None)
        get_tk_widget = getattr(self.fig.canvas, "get_tk_widget", None)
        if window is None or get_tk_widget is None:
            return

        try:
            canvas_widget = self.fig.canvas.get_tk_widget()
        except Exception as exc:
            print(f"Tk image browser disabled (no Tk canvas widget): {exc}")
            return

        try:
            canvas_widget.pack_forget()
        except Exception:
            pass

        self._tk = tk
        self._ttk = ttk
        window.configure(bg="#ede7dc")
        try:
            ttk.Style(window).theme_use("clam")
        except Exception:
            pass

        self.tk_left_panel = tk.Frame(window, width=360, bg="#fbfaf6")
        self.tk_left_panel.pack(side="left", fill="y")
        self.tk_left_panel.pack_propagate(False)

        self.tk_right_panel = tk.Frame(window, bg="#ede7dc")
        self.tk_right_panel.pack(side="right", fill="both", expand=True)
        self.tk_after_host = self.tk_right_panel

        self._build_tk_modern_toolbar(self.tk_right_panel)

        canvas_host = tk.Frame(self.tk_right_panel, bg="#ede7dc")
        canvas_host.pack(side="top", fill="both", expand=True)
        canvas_widget.pack(in_=canvas_host, side="top", fill="both", expand=True, padx=12, pady=(0, 12))

        title = tk.Label(
            self.tk_left_panel,
            text="Image Browser",
            bg="#fbfaf6",
            fg="#0f172a",
            font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        )
        title.pack(side="top", anchor="w", padx=12, pady=(12, 4))

        hint = tk.Label(
            self.tk_left_panel,
            text="Choose datasets, jump between files, and keep labeling without leaving the keyboard.",
            bg="#fbfaf6",
            fg="#475569",
            wraplength=330,
            justify="left",
        )
        hint.pack(side="top", anchor="w", padx=12, pady=(0, 10))

        filter_frame = tk.Frame(self.tk_left_panel, bg="#fbfaf6")
        filter_frame.pack(side="top", fill="x", padx=12, pady=(0, 10))
        tk.Label(
            filter_frame,
            text="Dataset Filter",
            bg="#fbfaf6",
            fg="#64748b",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="top", anchor="w", pady=(0, 4))

        self.tk_dataset_var = tk.StringVar(value=self.dataset_filter_text)
        dataset_entry = tk.Entry(
            filter_frame,
            textvariable=self.tk_dataset_var,
            relief="flat",
            bd=1,
            highlightthickness=1,
            highlightbackground="#d6d3d1",
            highlightcolor="#0f766e",
            font=("TkDefaultFont", 10),
        )
        dataset_entry.pack(side="top", fill="x", ipady=6)
        dataset_entry.bind("<Return>", lambda _event: self._apply_dataset_filter_from_widget())

        filter_buttons = tk.Frame(filter_frame, bg="#fbfaf6")
        filter_buttons.pack(side="top", fill="x", pady=(6, 0))
        tk.Button(
            filter_buttons,
            text="Apply Filter",
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            font=("TkDefaultFont", 10, "bold"),
            command=self._apply_dataset_filter_from_widget,
        ).pack(side="left")
        tk.Button(
            filter_buttons,
            text="Clear",
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            bg="#f4efe6",
            fg="#0f172a",
            activebackground="#e8dfcf",
            activeforeground="#0f172a",
            font=("TkDefaultFont", 10, "bold"),
            command=lambda: self._on_clear_dataset_clicked(None),
        ).pack(side="left", padx=(6, 0))

        self.tk_browser_status_var = tk.StringVar(value="Initializing...")
        status = tk.Label(
            self.tk_left_panel,
            textvariable=self.tk_browser_status_var,
            bg="#fbfaf6",
            fg="#475569",
            wraplength=330,
            justify="left",
        )
        status.pack(side="top", anchor="w", padx=12, pady=(0, 8))

        tree_frame = tk.Frame(self.tk_left_panel, bg="#fbfaf6")
        tree_frame.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 12))

        self.tk_image_tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse", height=22)
        yscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tk_image_tree.yview)
        self.tk_image_tree.configure(yscrollcommand=yscroll.set)

        self.tk_image_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.tk_image_tree.bind("<<TreeviewSelect>>", self._on_tk_tree_select)
        self.tk_image_tree.bind("<ButtonRelease-1>", self._on_tk_tree_click)

        self._tk_browser_ready = True
        self.tk_controls_enabled = True
        for ax in getattr(self, "widget_axes", []):
            try:
                ax.set_visible(False)
            except Exception:
                pass
        self.fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.02)
        self._sync_tk_control_state()

    def _ensure_session_dirs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.masks_out_dir.mkdir(parents=True, exist_ok=True)
        self.typed_labels_out_dir.mkdir(parents=True, exist_ok=True)
        self.typed_metadata_out_dir.mkdir(parents=True, exist_ok=True)

    def _session_mask_path_for_key(self, key: str) -> Path:
        return self.masks_out_dir / f"{key}.npy"

    def _session_typed_label_path_for_key(self, key: str) -> Path:
        return self.typed_labels_out_dir / f"{key}.npy"

    def _session_typed_metadata_path_for_key(self, key: str) -> Path:
        return self.typed_metadata_out_dir / f"{key}.json"

    def _refresh_session_inventory(self):
        self.persisted_mask_keys = {p.stem for p in self.masks_out_dir.glob("*.npy")}
        self.persisted_typed_keys = {p.stem for p in self.typed_labels_out_dir.glob("*.npy")}

    def _key_has_session_artifacts(self, key: str) -> bool:
        return (
            key in self.modified_masks
            or key in self.modified_typed_masks
            or key in self.persisted_mask_keys
            or key in self.persisted_typed_keys
        )

    def _load_typed_mask_from_known_paths(
        self,
        row_idx: int,
        *,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        key = self.keys[row_idx]
        if key in self.modified_typed_masks:
            return reconcile_typed_mask(binary_mask, self.modified_typed_masks[key].copy())

        session_typed_path = self._session_typed_label_path_for_key(key)
        if session_typed_path.exists():
            try:
                typed = np.squeeze(np.load(str(session_typed_path))).astype(np.uint8)
                if typed.shape == binary_mask.shape:
                    return reconcile_typed_mask(binary_mask, typed)
            except Exception as exc:
                print(f"  Failed to load typed labels for {key} from {session_typed_path}: {exc}")

        if self.has_typed_col:
            row = self.df.iloc[row_idx]
            typed_path = resolve_existing_path(row.get("typed_label_path", None), self.manifest_dir)
            if typed_path is not None and typed_path.exists():
                try:
                    typed = np.squeeze(np.load(str(typed_path))).astype(np.uint8)
                    if typed.shape == binary_mask.shape:
                        return reconcile_typed_mask(binary_mask, typed)
                except Exception as exc:
                    print(f"  Failed to load typed labels for {key} from {typed_path}: {exc}")

        return build_unlabeled_typed_mask(binary_mask)

    def _persist_row_artifacts(
        self,
        row_idx: int,
        binary_mask: np.ndarray,
        typed_mask: np.ndarray,
        *,
        write_mask: bool = True,
        write_typed: bool = True,
    ) -> None:
        self._ensure_session_dirs()
        key = self.keys[row_idx]
        binary_mask = (np.asarray(binary_mask) > 0.5).astype(np.float32)
        typed_mask = reconcile_typed_mask(binary_mask, np.asarray(typed_mask, dtype=np.uint8))

        if write_mask:
            mask_path = self._session_mask_path_for_key(key)
            np.save(mask_path, binary_mask.astype(np.uint8))
            self.persisted_mask_keys.add(key)

        if write_typed:
            typed_path = self._session_typed_label_path_for_key(key)
            np.save(typed_path, typed_mask.astype(np.uint8))
            self.persisted_typed_keys.add(key)

        self.modified_masks[key] = binary_mask.copy()
        self.modified_typed_masks[key] = typed_mask.copy()
        if self.current_row_idx == row_idx:
            self.current_mask = binary_mask
            self.current_typed_mask = typed_mask
        if write_mask or write_typed:
            self._write_typed_metadata_for_row(row_idx, binary_mask, typed_mask)
        self._invalidate_row_sort_stats(row_idx)
        self._refresh_tk_current_leaf_label()

    def _persist_current_row_artifacts(
        self,
        *,
        write_mask: bool = True,
        write_typed: bool = True,
    ):
        if self.current_row_idx is None or self.current_mask is None or self.current_typed_mask is None:
            return

        binary_mask = (np.asarray(self.current_mask) > 0.5).astype(np.float32)
        typed_mask = reconcile_typed_mask(binary_mask, np.asarray(self.current_typed_mask, dtype=np.uint8))
        self._persist_row_artifacts(
            int(self.current_row_idx),
            binary_mask,
            typed_mask,
            write_mask=write_mask,
            write_typed=write_typed,
        )

    def _cache_current_row_artifacts(self):
        """Keep current edits in memory so labeling stays responsive between saves."""
        if self.current_row_idx is None or self.current_mask is None:
            return

        key = self.keys[self.current_row_idx]
        binary_mask = (np.asarray(self.current_mask) > 0.5).astype(np.float32)
        if self.current_typed_mask is None or self.current_typed_mask.shape != binary_mask.shape:
            typed_mask = build_unlabeled_typed_mask(binary_mask)
        else:
            typed_mask = reconcile_typed_mask(binary_mask, np.asarray(self.current_typed_mask, dtype=np.uint8))

        self.current_mask = binary_mask
        self.current_typed_mask = typed_mask
        self.modified_masks[key] = binary_mask.copy()
        self.modified_typed_masks[key] = typed_mask.copy()
        self._refresh_tk_current_leaf_label()

    def _write_typed_metadata_for_row(
        self,
        row_idx: int,
        binary_mask: np.ndarray,
        typed_mask: np.ndarray,
    ) -> None:
        if row_idx is None:
            return

        key = self.keys[row_idx]
        row = self.df.iloc[row_idx]
        resolved_gt_path, resolved_gt_col = resolve_gt_mask_path_for_row(
            row,
            self.manifest_dir,
            self.gt_mask_path_columns,
        )
        typed_path = self._session_typed_label_path_for_key(key)
        session_mask_path = self._session_mask_path_for_key(key)
        if session_mask_path.exists():
            binary_mask_path = session_mask_path
        else:
            binary_mask_path = resolved_gt_path
        binary_mask = (np.asarray(binary_mask) > 0.5).astype(np.float32)
        typed_mask = reconcile_typed_mask(binary_mask, np.asarray(typed_mask, dtype=np.uint8))
        stats = typed_completion_stats(binary_mask, typed_mask)
        counts = typed_class_pixel_counts(binary_mask, typed_mask)
        payload = {
            "dataset_id": self.dataset_ids[row_idx],
            "stem": self.stems[row_idx],
            "key": key,
            "binary_mask_path": (
                path_for_manifest(binary_mask_path, self.manifest_dir)
                if binary_mask_path is not None
                else ""
            ),
            "typed_label_path": path_for_manifest(typed_path, self.manifest_dir),
            "typed_label_schema_name": TYPE_SCHEMA_NAME,
            "typed_label_schema_version": int(TYPE_SCHEMA_VERSION),
            "typed_label_unlabeled_value": int(UNLABELED_CONTAMINATION_LABEL),
            "type_id_to_name": {
                str(type_id): TYPE_ID_TO_DISPLAY[type_id] for type_id in TYPE_ID_SEQUENCE
            },
            "counts": counts,
            "stats": stats,
            "current_split_label": self.split_labels[row_idx],
            "source_gt_mask_path": (
                path_for_manifest(resolved_gt_path, self.manifest_dir)
                if resolved_gt_path is not None
                else (str(row.get("gt_mask_path", "")) if self.has_gt_col else "")
            ),
            "source_gt_mask_column": resolved_gt_col or "",
        }
        meta_path = self._session_typed_metadata_path_for_key(key)
        meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_current_typed_metadata(self):
        if self.current_row_idx is None or self.current_mask is None or self.current_typed_mask is None:
            return
        self._write_typed_metadata_for_row(
            int(self.current_row_idx),
            self.current_mask,
            self.current_typed_mask,
        )

    def _typed_scope_mask_for_selected_region(self) -> Optional[np.ndarray]:
        if self.current_mask is None:
            return None
        if 0 <= self.selected_region < len(self.regions):
            return np.asarray(self.regions[self.selected_region], dtype=bool)
        return np.asarray(self.current_mask > 0.5, dtype=bool)

    def _typed_scope_mask_for_full_label(self) -> Optional[np.ndarray]:
        if self.current_mask is None:
            return None
        if 0 <= self.selected_region < len(self.regions):
            return np.asarray(self.regions[self.selected_region], dtype=bool)
        if len(self.regions) == 1:
            return np.asarray(self.regions[0], dtype=bool)
        return None

    def _row_has_resolved_source_gt_mask(self, row_idx: int) -> bool:
        if row_idx is None or not self.has_gt_col:
            return False
        row = self.df.iloc[row_idx]
        gt_path, _ = resolve_gt_mask_path_for_row(
            row,
            self.manifest_dir,
            self.gt_mask_path_columns,
        )
        return bool(gt_path is not None and gt_path.exists())

    def _row_display_name(self, row_idx: int) -> str:
        """Human-friendly filename label for the left browser."""
        row = self.df.iloc[row_idx]
        if "original_filename" in self.df.columns:
            value = row.get("original_filename", None)
            if not pd.isna(value):
                text = str(value).strip()
                if text:
                    return text
        return self.stems[row_idx]

    def _refresh_tk_image_browser(self):
        """Rebuild the left-side dataset/file tree from the current filtered view."""
        if self._qt_browser_ready:
            tree = self.qt_image_tree
            if tree is None:
                return
            self._suspend_qt_tree_event = True
            try:
                tree.clear()
            finally:
                self._suspend_qt_tree_event = False

            self._qt_row_items = {}
            self._qt_dataset_items = {}

            if not self.view_indices:
                if self.qt_browser_status_label is not None:
                    self.qt_browser_status_label.setText(self._browser_status_text())
                return

            self._suspend_qt_tree_event = True
            try:
                if self.sort_mode == SORT_MODE_OPTIONS[0]:
                    grouped: Dict[str, List[int]] = {}
                    for row_idx in self.view_indices:
                        grouped.setdefault(self.dataset_ids[row_idx], []).append(row_idx)

                    def _dataset_sort_key(ds: str):
                        return (0, int(ds)) if ds.isdigit() else (1, ds)

                    dataset_items = sorted(grouped.items(), key=lambda kv: _dataset_sort_key(kv[0]))
                    for dataset_id, indices in dataset_items:
                        indices_sorted = sorted(indices, key=lambda i: self._row_display_name(i).lower())
                        parent = self._qt.QTreeWidgetItem([f"{dataset_id} ({len(indices_sorted)})"])
                        tree.addTopLevelItem(parent)
                        self._qt_dataset_items[dataset_id] = parent
                        if len(dataset_items) <= 12:
                            parent.setExpanded(True)

                        for row_idx in indices_sorted:
                            child = self._qt.QTreeWidgetItem([self._row_browser_label(row_idx)])
                            child.setData(0, self._qtcore.Qt.ItemDataRole.UserRole, int(row_idx))
                            parent.addChild(child)
                            self._qt_row_items[int(row_idx)] = child
                else:
                    parent = self._qt.QTreeWidgetItem([f"Sorted by {self.sort_mode} ({len(self.view_indices)})"])
                    tree.addTopLevelItem(parent)
                    parent.setExpanded(True)
                    for row_idx in self.view_indices:
                        child = self._qt.QTreeWidgetItem([self._row_browser_label(row_idx)])
                        child.setData(0, self._qtcore.Qt.ItemDataRole.UserRole, int(row_idx))
                        parent.addChild(child)
                        self._qt_row_items[int(row_idx)] = child
            finally:
                self._suspend_qt_tree_event = False

            if self.qt_browser_status_label is not None:
                self.qt_browser_status_label.setText(self._browser_status_text())
            self._sync_tk_tree_selection()
            return

        if not self._tk_browser_ready:
            return

        tree = self.tk_image_tree
        self._suspend_tk_tree_event = True
        try:
            for item in tree.get_children(""):
                tree.delete(item)
        finally:
            self._suspend_tk_tree_event = False

        self._tk_tree_item_to_row_idx = {}
        self._tk_tree_dataset_item_ids = {}
        self._tk_row_item_ids = {}

        if not self.view_indices:
            self.tk_browser_status_var.set(self._browser_status_text())
            return

        self._suspend_tk_tree_event = True
        try:
            if self.sort_mode == SORT_MODE_OPTIONS[0]:
                grouped: Dict[str, List[int]] = {}
                for row_idx in self.view_indices:
                    grouped.setdefault(self.dataset_ids[row_idx], []).append(row_idx)

                def _dataset_sort_key(ds: str):
                    return (0, int(ds)) if ds.isdigit() else (1, ds)

                dataset_items = sorted(grouped.items(), key=lambda kv: _dataset_sort_key(kv[0]))
                for dataset_id, indices in dataset_items:
                    indices_sorted = sorted(indices, key=lambda i: self._row_display_name(i).lower())
                    parent_iid = f"ds::{dataset_id}"
                    tree.insert(
                        "",
                        "end",
                        iid=parent_iid,
                        text=f"{dataset_id} ({len(indices_sorted)})",
                        open=(len(dataset_items) <= 12),
                    )
                    self._tk_tree_dataset_item_ids[dataset_id] = parent_iid

                    for row_idx in indices_sorted:
                        leaf_iid = f"row::{row_idx}"
                        tree.insert(parent_iid, "end", iid=leaf_iid, text=self._row_browser_label(row_idx))
                        self._tk_tree_item_to_row_idx[leaf_iid] = row_idx
                        self._tk_row_item_ids[row_idx] = leaf_iid
            else:
                parent_iid = "sort::all"
                tree.insert(
                    "",
                    "end",
                    iid=parent_iid,
                    text=f"Sorted by {self.sort_mode} ({len(self.view_indices)})",
                    open=True,
                )
                for row_idx in self.view_indices:
                    leaf_iid = f"row::{row_idx}"
                    tree.insert(parent_iid, "end", iid=leaf_iid, text=self._row_browser_label(row_idx))
                    self._tk_tree_item_to_row_idx[leaf_iid] = row_idx
                    self._tk_row_item_ids[row_idx] = leaf_iid
        finally:
            self._suspend_tk_tree_event = False

        self.tk_browser_status_var.set(self._browser_status_text())
        self._sync_tk_tree_selection()

    def _sync_tk_tree_selection(self):
        """Keep the left tree selection in sync with the current image."""
        if self._qt_browser_ready and self.current_row_idx is not None:
            item = self._qt_row_items.get(int(self.current_row_idx))
            if item is None or self.qt_image_tree is None:
                return
            self._suspend_qt_tree_event = True
            try:
                parent = item.parent()
                if parent is not None:
                    parent.setExpanded(True)
                self.qt_image_tree.setCurrentItem(item)
                self.qt_image_tree.scrollToItem(item)
            except Exception:
                pass
            finally:
                self._suspend_qt_tree_event = False
            return

        if not self._tk_browser_ready or self.current_row_idx is None:
            return
        leaf_iid = self._tk_row_item_ids.get(self.current_row_idx)
        if leaf_iid is None:
            return

        tree = self.tk_image_tree
        self._suspend_tk_tree_event = True
        try:
            dataset_id = self.dataset_ids[self.current_row_idx]
            parent_iid = self._tk_tree_dataset_item_ids.get(dataset_id)
            if parent_iid is not None:
                tree.item(parent_iid, open=True)
            tree.selection_set(leaf_iid)
            tree.focus(leaf_iid)
            tree.see(leaf_iid)
        except Exception:
            pass
        finally:
            self._suspend_tk_tree_event = False

    def _refresh_tk_current_leaf_label(self):
        """Refresh the current filename label (e.g., add modified marker) without rebuilding the tree."""
        if self._qt_browser_ready and self.current_row_idx is not None:
            item = self._qt_row_items.get(int(self.current_row_idx))
            if item is None:
                return
            item.setText(0, self._row_browser_label(self.current_row_idx))
            return

        if not self._tk_browser_ready or self.current_row_idx is None:
            return
        leaf_iid = self._tk_row_item_ids.get(self.current_row_idx)
        if leaf_iid is None:
            return

        try:
            self.tk_image_tree.item(leaf_iid, text=self._row_browser_label(self.current_row_idx))
        except Exception:
            pass

    def _on_tk_tree_select(self, _event):
        """Jump to the selected filename when clicked in the left tree."""
        if self._qt_browser_ready:
            self._on_qt_tree_select()
            return

        if not self._tk_browser_ready or self._suspend_tk_tree_event:
            return

        tree = self.tk_image_tree
        selected = tree.selection()
        if not selected:
            return
        item_id = selected[0]

        row_idx = self._tk_tree_item_to_row_idx.get(item_id)
        if row_idx is None:
            children = tree.get_children(item_id)
            if not children:
                return
            item_id = children[0]
            row_idx = self._tk_tree_item_to_row_idx.get(item_id)
            if row_idx is None:
                return
        self._select_browser_row_idx(int(row_idx))

    def _on_tk_tree_click(self, _event):
        """Force-load the clicked filename, including already-selected rows."""
        if self._qt_browser_ready or not self._tk_browser_ready or self._suspend_tk_tree_event:
            return
        tree = self.tk_image_tree
        selected = tree.selection()
        if not selected:
            return
        item_id = selected[0]
        row_idx = self._tk_tree_item_to_row_idx.get(item_id)
        if row_idx is None:
            children = tree.get_children(item_id)
            if not children:
                return
            row_idx = self._tk_tree_item_to_row_idx.get(children[0])
        if row_idx is None:
            return
        self._select_browser_row_idx(int(row_idx))

    def _on_split_radio_changed(self, label: str):
        self._set_split_filter(label)

    def _on_mode_radio_changed(self, label: str):
        self._set_edit_mode(label)

    def _on_type_radio_changed(self, label: str):
        text = str(label).strip()
        if text.lower() == "none":
            self._set_active_type_id(None, sync_widget=False)
        else:
            try:
                self._set_active_type_id(int(text.split()[0]), sync_widget=False)
            except Exception:
                self._set_active_type_id(None, sync_widget=False)
        self._update_display()

    def _on_dataset_submit(self, _text: str):
        self._apply_dataset_filter_from_widget()

    def _on_apply_dataset_clicked(self, _event):
        self._apply_dataset_filter_from_widget()

    def _on_clear_dataset_clicked(self, _event):
        if self.qt_controls_enabled and self.qt_dataset_edit is not None:
            self.qt_dataset_edit.setText("")
            self.dataset_filter_text = ""
            self.dataset_filter_ids = set()
            self._rebuild_view(preserve_current=True)
            return
        if self.tk_controls_enabled and self.tk_dataset_var is not None:
            self.tk_dataset_var.set("")
            self.dataset_filter_text = ""
            self.dataset_filter_ids = set()
            self._rebuild_view(preserve_current=True)
            return
        self.dataset_box.set_val("")

    def _on_prev_dataset_clicked(self, _event):
        self._step_dataset(-1)

    def _on_next_dataset_clicked(self, _event):
        self._step_dataset(1)

    def _on_save_clicked(self, _event):
        self._save_current()

    def _on_export_clicked(self, _event):
        self._start_headless_export()

    def _set_edit_mode(self, mode_label: str):
        if mode_label not in VALID_EDIT_MODES:
            return
        if self.edit_mode == mode_label:
            return
        self.edit_mode = mode_label
        if self.edit_mode != "Polygon label" and self.current_polygon:
            self.current_polygon = []
        self._sync_tk_control_state()
        self._update_display()

    def _is_full_label_mode(self) -> bool:
        return self.edit_mode == "Full label"

    def _is_erase_polygon_mode(self) -> bool:
        return self.edit_mode == "Erase polygon"

    def _is_polygon_label_mode(self) -> bool:
        return self.edit_mode == "Polygon label"

    def _set_split_filter(self, split_filter: str):
        split_filter = split_filter.upper().strip()
        if split_filter not in VALID_SPLIT_FILTERS:
            return
        if split_filter == self.split_filter:
            return
        self.split_filter = split_filter
        self._sync_tk_control_state()
        self._rebuild_view(preserve_current=True)

    def _apply_dataset_filter_from_widget(self):
        if self.qt_controls_enabled and self.qt_dataset_edit is not None:
            text = self.qt_dataset_edit.text()
        elif self.tk_controls_enabled and self.tk_dataset_var is not None:
            text = self.tk_dataset_var.get()
        else:
            text = getattr(self.dataset_box, "text", "")
        self.dataset_filter_text = str(text).strip()
        self.dataset_filter_ids = set(parse_dataset_filter_text(self.dataset_filter_text))
        self._rebuild_view(preserve_current=True)

    def _current_key(self) -> Optional[str]:
        if self.current_row_idx is None:
            return None
        return self.keys[self.current_row_idx]

    def _rebuild_view(self, preserve_current: bool = True):
        """Recompute filtered row list and reload display."""
        prev_key = self._current_key() if preserve_current else None
        self._cancel_active_row_load()

        filtered: List[int] = []
        for idx in range(len(self.df)):
            if self.split_filter != "ALL" and self.split_labels[idx] != self.split_filter:
                continue
            if self.dataset_filter_ids and self.dataset_ids[idx] not in self.dataset_filter_ids:
                continue
            filtered.append(idx)

        if self.sort_mode != SORT_MODE_OPTIONS[0]:
            filtered = sorted(filtered, key=self._sort_key_for_row)

        self.view_indices = filtered
        self.sort_view_dirty = False
        self.view_pos = 0

        if prev_key is not None:
            for pos, idx in enumerate(self.view_indices):
                if self.keys[idx] == prev_key:
                    self.view_pos = pos
                    break
        elif self.initial_row_key:
            for pos, idx in enumerate(self.view_indices):
                if self.keys[idx] == self.initial_row_key:
                    self.view_pos = pos
                    break

        print(
            f"View filter -> split={self.split_filter}, "
            f"dataset_filter={sorted(self.dataset_filter_ids) if self.dataset_filter_ids else 'ALL'}, "
            f"rows={len(self.view_indices)}"
        )
        self._refresh_tk_image_browser()

        if not self.view_indices:
            self.current_row_idx = None
            self.current_image = None
            self.current_mask = None
            self.current_typed_mask = None
            self.regions = []
            self.current_polygon = []
            self.selected_region = -1
            self._update_display()
            return

        self.view_pos = max(0, min(self.view_pos, len(self.view_indices) - 1))
        self._load_current()

    def _coerce_loaded_image_2d(
        self,
        image: Optional[np.ndarray],
        *,
        key: str,
        source_col: str,
        verbose: bool = True,
    ) -> Optional[np.ndarray]:
        """Squeeze and validate that a loaded image is 2D."""
        if image is None:
            return None
        arr = np.asarray(image)
        try:
            arr = np.squeeze(arr)
        except Exception:
            pass
        if arr.ndim != 2:
            if verbose:
                print(
                    f"  Warning: image for {key} from {source_col} not 2D after squeeze "
                    f"(shape={getattr(arr, 'shape', None)}), skipping"
                )
            return None
        return arr.astype(np.float32, copy=False)

    def _reconcile_image_shape_to_target(
        self,
        img: np.ndarray,
        target_shape: Tuple[int, int],
        *,
        max_delta_px: int = 16,
    ) -> Optional[np.ndarray]:
        """
        Center crop/pad an image to target_shape when the mismatch is small.

        This handles common off-by-1/2 border differences between full-res images
        and masks without blanking the display.
        """
        h, w = int(img.shape[0]), int(img.shape[1])
        th, tw = int(target_shape[0]), int(target_shape[1])
        if (h, w) == (th, tw):
            return img
        if abs(h - th) > max_delta_px or abs(w - tw) > max_delta_px:
            return None

        out = img

        # Height reconcile
        if out.shape[0] > th:
            extra = out.shape[0] - th
            top = extra // 2
            out = out[top : top + th, :]
        elif out.shape[0] < th:
            pad = th - out.shape[0]
            top = pad // 2
            bottom = pad - top
            out = np.pad(out, ((top, bottom), (0, 0)), mode="edge")

        # Width reconcile
        if out.shape[1] > tw:
            extra = out.shape[1] - tw
            left = extra // 2
            out = out[:, left : left + tw]
        elif out.shape[1] < tw:
            pad = tw - out.shape[1]
            left = pad // 2
            right = pad - left
            out = np.pad(out, ((0, 0), (left, right)), mode="edge")

        if out.shape[:2] != (th, tw):
            return None
        return out.astype(np.float32, copy=False)

    def _cache_put_row_payload(self, row_idx: int, payload: Dict[str, Any]) -> None:
        self.row_payload_cache[row_idx] = payload
        self.row_payload_cache.move_to_end(row_idx)
        while len(self.row_payload_cache) > self.row_cache_size:
            self.row_payload_cache.popitem(last=False)

    def _cache_get_row_payload(self, row_idx: int) -> Optional[Dict[str, Any]]:
        payload = self.row_payload_cache.get(row_idx)
        if payload is not None:
            self.row_payload_cache.move_to_end(row_idx)
        return payload

    def _invalidate_row_payload(self, row_idx: Optional[int]) -> None:
        if row_idx is None:
            return
        self.row_payload_cache.pop(int(row_idx), None)
        future = self.row_prefetch_futures.pop(int(row_idx), None)
        if future is not None:
            future.cancel()

    def _cancel_prefetch_futures(self, keep_row_idx: Optional[int] = None) -> None:
        keep = None if keep_row_idx is None else int(keep_row_idx)
        for row_idx, future in list(self.row_prefetch_futures.items()):
            if keep is not None and int(row_idx) == keep:
                continue
            self.row_prefetch_futures.pop(int(row_idx), None)
            try:
                future.cancel()
            except Exception:
                pass

    def _can_async_load_rows(self) -> bool:
        return False

    def _cancel_active_row_load(self) -> None:
        future = self.active_row_load_future
        self.active_row_load_future = None
        self.active_row_load_row_idx = None
        self.active_row_load_token += 1
        self.row_loading = False
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass

    def _schedule_active_row_poll(self, token: int) -> None:
        host = self.tk_after_host
        if host is None:
            return
        try:
            host.after(25, lambda tok=token: self._poll_active_row_load(tok))
        except Exception:
            pass

    def _begin_async_row_load(self, row_idx: int) -> None:
        if self.interactive_load_executor is None:
            return

        self._cancel_active_row_load()
        self.current_row_idx = int(row_idx)
        self.current_image = None
        self.current_mask = None
        self.current_typed_mask = None
        self.current_image_display = None
        self.current_display_stride = 1
        self.current_display_limits = (0.0, 1.0)
        self.current_polygon = []
        self.regions = []
        self.selected_region = -1
        self.row_loading = True
        self._sync_tk_tree_selection()
        self._update_display()

        token = self.active_row_load_token + 1
        self.active_row_load_token = token
        self.active_row_load_row_idx = int(row_idx)
        self.active_row_load_future = self.interactive_load_executor.submit(
            self._load_source_row_payload,
            int(row_idx),
        )
        self._schedule_active_row_poll(token)

    def _poll_active_row_load(self, token: int) -> None:
        if token != self.active_row_load_token:
            return

        future = self.active_row_load_future
        row_idx = self.active_row_load_row_idx
        if future is None or row_idx is None:
            return
        if not future.done():
            self._schedule_active_row_poll(token)
            return

        self.active_row_load_future = None
        self.active_row_load_row_idx = None
        try:
            payload = future.result()
        except Exception as exc:
            self.row_loading = False
            print(f"  Failed to load row {row_idx}: {exc}")
            self.current_row_idx = int(row_idx)
            self.current_image = None
            self.current_mask = None
            self.current_typed_mask = None
            self._update_display()
            return

        if token != self.active_row_load_token:
            return
        self._cache_put_row_payload(int(row_idx), payload)
        self._apply_row_payload(int(row_idx), payload)

    def _compute_display_payload(
        self,
        img: Optional[np.ndarray],
        row_idx: int,
    ) -> Tuple[Optional[np.ndarray], int, Tuple[float, float]]:
        if img is None:
            return None, 1, (0.0, 1.0)

        arr = np.asarray(img)
        if arr.ndim != 2:
            return None, 1, (0.0, 1.0)

        stride = self._display_stride_for_shape((int(arr.shape[0]), int(arr.shape[1])))
        disp_src = self._mean_bin_for_display_stride(arr, stride)
        try:
            from utils.filtering_visualization import (
                SIMPLIPYTEM_DISPLAY_STYLE,
                prepare_simplipytem_display,
            )

            display_img = prepare_simplipytem_display(
                disp_src,
                max_display_dim=max(int(disp_src.shape[0]), int(disp_src.shape[1])),
                saturation_pct=float(SIMPLIPYTEM_DISPLAY_STYLE["saturation_pct"]),
                median_kernel=int(SIMPLIPYTEM_DISPLAY_STYLE["median_kernel"]),
                gaussian_sigma_px=float(SIMPLIPYTEM_DISPLAY_STYLE["gaussian_sigma_px"]),
                local_normalization_patches=int(SIMPLIPYTEM_DISPLAY_STYLE["local_normalization_patches"]),
                local_normalization_padding_pct=int(SIMPLIPYTEM_DISPLAY_STYLE["local_normalization_padding_pct"]),
                local_normalization_pad=bool(SIMPLIPYTEM_DISPLAY_STYLE["local_normalization_pad"]),
            )
        except Exception:
            pixel_size = None
            try:
                row = self.df.iloc[row_idx]
                if "pixel_size_angstrom" in self.df.columns:
                    p = float(row.get("pixel_size_angstrom", np.nan))
                    if np.isfinite(p) and p > 0:
                        pixel_size = p * float(stride)
            except Exception:
                pixel_size = None

            display_img = self._normalize_for_display_like_figures(disp_src, pixel_size)
        return np.asarray(display_img, dtype=np.float32), int(max(1, stride)), (0.0, 1.0)

    def _load_source_row_payload(self, row_idx: int, verbose: bool = True) -> Dict[str, Any]:
        row = self.df.iloc[row_idx]
        key = self.keys[row_idx]
        load_t0 = time.perf_counter()
        stage_t0 = load_t0
        if verbose:
            print(f"  Loading row payload: {key}", flush=True)

        mask: Optional[np.ndarray] = None
        session_mask_path = self._session_mask_path_for_key(key)
        if session_mask_path.exists():
            try:
                mask = np.squeeze(np.load(str(session_mask_path))).astype(np.float32)
            except Exception as exc:
                if verbose:
                    print(f"  Failed to load session mask for {key} from {session_mask_path}: {exc}")

        if mask is None and self.has_gt_col:
            gt_path, gt_col = resolve_gt_mask_path_for_row(
                row,
                self.manifest_dir,
                self.gt_mask_path_columns,
            )
            if gt_path is not None and gt_path.exists():
                try:
                    mask = np.squeeze(np.load(str(gt_path))).astype(np.float32)
                    if verbose and gt_col and gt_col != "gt_mask_path":
                        print(f"  Using fallback GT mask column {gt_col} for {key}")
                except Exception as exc:
                    if verbose:
                        print(f"  Failed to load mask for {key} from {gt_path}: {exc}")
        if verbose:
            print(f"    mask stage: {time.perf_counter() - stage_t0:.2f}s", flush=True)

        target_shape: Optional[Tuple[int, int]] = None
        if mask is not None and getattr(mask, "ndim", None) == 2:
            target_shape = tuple(int(x) for x in mask.shape[:2])

        stage_t0 = time.perf_counter()
        image, image_source_col, shape_match = self._find_best_image_for_row(
            row_idx,
            target_shape=target_shape,
            verbose=verbose,
        )
        if verbose:
            print(f"    image stage: {time.perf_counter() - stage_t0:.2f}s", flush=True)
        if image is None:
            if target_shape is not None:
                if verbose:
                    print(f"Could not load image for {key}; using blank placeholder matching mask shape {target_shape}")
                image = np.zeros(target_shape, dtype=np.float32)
            else:
                if verbose:
                    print(f"Could not load image for {key}; using blank placeholder")
                image = np.zeros((1024, 1024), dtype=np.float32)
        else:
            if verbose and image_source_col is not None and image_source_col != self.mic_col:
                print(f"  Using fallback image column {image_source_col} for {key}")
            if verbose and target_shape is not None and not shape_match and image.shape[:2] != target_shape:
                print(
                    f"  Warning: no image matched mask shape for {key}; "
                    f"mask={target_shape}, image={image.shape[:2]} from {image_source_col}. "
                    "Using blank image matching mask to preserve mask display."
                )
                image = np.zeros(target_shape, dtype=np.float32)

        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        elif mask.shape != image.shape[:2]:
            print(
                f"  Warning: mask/image shape mismatch persists for {key}: "
                f"mask={mask.shape}, image={image.shape[:2]}; using blank image at mask shape"
            )
            image = np.zeros(mask.shape[:2], dtype=np.float32)

        mask = (mask > 0.5).astype(np.float32)
        stage_t0 = time.perf_counter()
        typed_mask = self._load_typed_mask_from_known_paths(row_idx, binary_mask=mask)
        if verbose:
            print(f"    typed-label stage: {time.perf_counter() - stage_t0:.2f}s", flush=True)
        stage_t0 = time.perf_counter()
        display_img, display_stride, display_limits = self._compute_display_payload(image, row_idx)
        if verbose:
            print(
                f"    display stage: {time.perf_counter() - stage_t0:.2f}s "
                f"(stride={display_stride}x, display_shape={None if display_img is None else display_img.shape})",
                flush=True,
            )
            print(f"  Loaded row payload in {time.perf_counter() - load_t0:.2f}s", flush=True)
        return {
            "row_idx": int(row_idx),
            "image": np.asarray(image, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.float32),
            "typed_mask": np.asarray(typed_mask, dtype=np.uint8),
            "image_source_col": image_source_col,
            "display_image": display_img,
            "display_stride": int(display_stride),
            "display_limits": display_limits,
        }

    def _get_row_payload(self, row_idx: int, *, verbose: bool = False) -> Dict[str, Any]:
        payload = self._cache_get_row_payload(row_idx)
        if payload is not None:
            return payload

        future = self.row_prefetch_futures.get(row_idx)
        if future is not None and future.done():
            try:
                payload = future.result()
                self.row_prefetch_futures.pop(row_idx, None)
                self._cache_put_row_payload(row_idx, payload)
                return payload
            except Exception:
                self.row_prefetch_futures.pop(row_idx, None)
        elif future is not None:
            # Never block interactive navigation on a queued/running prefetch job.
            self.row_prefetch_futures.pop(row_idx, None)
            try:
                future.cancel()
            except Exception:
                pass

        payload = self._load_source_row_payload(row_idx, verbose)
        self._cache_put_row_payload(row_idx, payload)
        return payload

    def _schedule_row_prefetch(self, row_idx: int) -> None:
        if self.prefetch_executor is None:
            return
        if row_idx < 0 or row_idx >= len(self.df):
            return
        if row_idx in self.row_payload_cache or row_idx in self.row_prefetch_futures:
            return
        self.row_prefetch_futures[row_idx] = self.prefetch_executor.submit(
            self._load_source_row_payload,
            row_idx,
            False,
        )

    def _prefetch_neighbor_rows(self, center_view_pos: int) -> None:
        if self.prefetch_executor is None or self.prefetch_radius <= 0:
            return
        for delta in range(1, self.prefetch_radius + 1):
            prev_view_pos = center_view_pos - delta
            next_view_pos = center_view_pos + delta
            if 0 <= prev_view_pos < len(self.view_indices):
                self._schedule_row_prefetch(int(self.view_indices[prev_view_pos]))
            if 0 <= next_view_pos < len(self.view_indices):
                self._schedule_row_prefetch(int(self.view_indices[next_view_pos]))

    def _find_best_image_for_row(
        self,
        row_idx: int,
        *,
        target_shape: Optional[Tuple[int, int]] = None,
        verbose: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[str], bool]:
        """
        Load an image for a row, preferring one whose shape matches target_shape.

        Returns (image, source_col, shape_matched_target).
        """
        row = self.df.iloc[row_idx]
        stem = self.stems[row_idx]
        key = self.keys[row_idx]
        fallback_img: Optional[np.ndarray] = None
        fallback_col: Optional[str] = None
        seen_direct_paths: set[str] = set()

        for col in self.image_path_columns:
            raw_img: Optional[np.ndarray]
            direct_path = resolve_existing_path(row.get(col, None), self.manifest_dir)
            direct_key: Optional[str] = None
            if direct_path is not None and direct_path.exists():
                direct_key = str(direct_path)

            # Avoid loading the same large file repeatedly when multiple manifest columns
            # reference the same underlying micrograph path.
            if direct_key is not None and direct_key in seen_direct_paths:
                continue
            if direct_key is not None:
                seen_direct_paths.add(direct_key)

            if direct_path is not None and direct_path.exists() and self.mic_dir is None:
                raw_img = load_image(direct_path)
            else:
                raw_img = find_image(
                    mic_path_value=row.get(col, None),
                    mic_dir=self.mic_dir,
                    stem=stem,
                    manifest_dir=self.manifest_dir,
                )
            img = self._coerce_loaded_image_2d(raw_img, key=key, source_col=col, verbose=verbose)
            if img is None:
                continue

            if fallback_img is None:
                fallback_img = img
                fallback_col = col

            if target_shape is not None:
                if tuple(img.shape[:2]) == tuple(target_shape):
                    return img, col, True
                reconciled = self._reconcile_image_shape_to_target(img, target_shape)
                if reconciled is not None:
                    if verbose:
                        print(
                            f"  Reconciled image shape for {key} from {img.shape[:2]} "
                            f"to mask shape {tuple(target_shape)} using {col}"
                        )
                    return reconciled, col, True

        if fallback_img is not None:
            return fallback_img, fallback_col, False
        return None, None, False

    def _display_stride_for_shape(self, shape: Tuple[int, int], max_dim: int = 1400) -> int:
        h, w = int(shape[0]), int(shape[1])
        m = max(h, w)
        if m <= max_dim:
            return 1
        return int(np.ceil(m / float(max_dim)))

    def _mean_bin_for_display_stride(self, img: np.ndarray, stride: int) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        factor = max(1, int(stride))
        if factor <= 1:
            return arr.astype(np.float32, copy=False)

        h, w = int(arr.shape[0]), int(arr.shape[1])
        hc = (h // factor) * factor
        wc = (w // factor) * factor
        if hc >= factor and wc >= factor:
            return (
                arr[:hc, :wc]
                .reshape(hc // factor, factor, wc // factor, factor)
                .mean(axis=(1, 3))
                .astype(np.float32, copy=False)
            )

        return arr[::factor, ::factor].astype(np.float32, copy=False)

    def _normalize_for_display_like_figures(self, img: np.ndarray, pixel_size_angstrom: Optional[float]) -> np.ndarray:
        """
        Match the qualitative raw-panel rendering style used in figure scripts:
        low-pass display (if available) + robust percentile normalization to [0, 1].
        """
        x = np.asarray(img, dtype=np.float32)
        if x.ndim != 2:
            return np.zeros((1, 1), dtype=np.float32)

        # Low-pass display similar to scripts/ablation_visualize_probability_panels.py::_to_display_image
        if pixel_size_angstrom is not None and np.isfinite(pixel_size_angstrom) and pixel_size_angstrom > 0:
            try:
                from utils.image_utils import apply_lowpass_filter

                lowpass = apply_lowpass_filter(
                    x.astype(np.float64),
                    cutoff_angstrom=4.0,
                    pixel_size_angstrom=float(pixel_size_angstrom),
                )
                x = np.asarray(lowpass, dtype=np.float32)
            except Exception:
                pass

        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return np.zeros_like(x, dtype=np.float32)

        lo, hi = np.percentile(finite, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.percentile(finite, [2.0, 98.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.zeros_like(x, dtype=np.float32)

        x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return x.astype(np.float32, copy=False)

    def _prepare_display_cache_for_current_row(self, row_idx: int) -> None:
        """Prepare a faster display preview (possibly downsampled) for interactive rendering."""
        display_img, display_stride, display_limits = self._compute_display_payload(self.current_image, row_idx)
        self.current_image_display = display_img
        self.current_display_stride = int(display_stride)
        self.current_display_limits = display_limits

    def _apply_row_payload(self, row_idx: int, payload: Dict[str, Any]) -> None:
        key = self.keys[row_idx]
        self.current_row_idx = int(row_idx)
        self.current_image = np.asarray(payload.get("image"), dtype=np.float32).copy()
        source_mask = np.asarray(payload.get("mask"), dtype=np.float32)
        if key in self.modified_masks:
            self.current_mask = self.modified_masks[key].copy()
        else:
            self.current_mask = source_mask.copy()

        self.current_mask = (self.current_mask > 0.5).astype(np.float32)
        if key in self.modified_typed_masks:
            self.current_typed_mask = reconcile_typed_mask(self.current_mask, self.modified_typed_masks[key].copy())
        else:
            source_typed = payload.get("typed_mask")
            if source_typed is not None:
                self.current_typed_mask = reconcile_typed_mask(
                    self.current_mask,
                    np.asarray(source_typed, dtype=np.uint8).copy(),
                )
            else:
                self.current_typed_mask = self._load_typed_mask_from_known_paths(
                    row_idx,
                    binary_mask=self.current_mask,
                )

        self.current_image_display = (
            None
            if payload.get("display_image") is None
            else np.asarray(payload.get("display_image"), dtype=np.float32)
        )
        self.current_display_stride = int(payload.get("display_stride", 1))
        self.current_display_limits = tuple(payload.get("display_limits", (0.0, 1.0)))
        self.row_loading = False
        self._extract_regions()
        self.current_polygon = []
        self._prefetch_neighbor_rows(self.view_pos)
        self._update_display()
        self._sync_tk_tree_selection()

    def _load_current(self):
        """Load image + mask for current filtered row."""
        if not self.view_indices:
            self._cancel_active_row_load()
            self._cancel_prefetch_futures()
            self.current_row_idx = None
            self.current_image = None
            self.current_mask = None
            self.current_typed_mask = None
            self.regions = []
            self.current_polygon = []
            self.selected_region = -1
            self._update_display()
            return

        self.view_pos = max(0, min(self.view_pos, len(self.view_indices) - 1))
        row_idx = int(self.view_indices[self.view_pos])
        payload = self._cache_get_row_payload(row_idx)
        if payload is not None:
            self._cancel_active_row_load()
            self._apply_row_payload(row_idx, payload)
            return

        self._cancel_active_row_load()
        self._cancel_prefetch_futures(keep_row_idx=row_idx)
        payload = self._get_row_payload(row_idx, verbose=True)
        self._apply_row_payload(row_idx, payload)

    def _extract_regions(self):
        """Extract connected regions from current mask."""
        if self.current_mask is None:
            self.regions = []
            self.selected_region = -1
            return

        binary = (self.current_mask > 0.5).astype(np.uint8)
        labeled, num = ndimage.label(binary)

        self.regions = [(labeled == i) for i in range(1, num + 1)]
        self.selected_region = -1

    def _image_display_limits(self) -> Tuple[float, float]:
        if self.current_display_limits is not None:
            return self.current_display_limits
        if self.current_image is None:
            return (0.0, 1.0)

        arr = np.asarray(self.current_image)
        if arr.ndim != 2:
            return (0.0, 1.0)

        # Subsample for speed; this function is used often during interactive redraws.
        step_y = max(1, arr.shape[0] // 1024)
        step_x = max(1, arr.shape[1] // 1024)
        sample = arr[::step_y, ::step_x]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return (0.0, 1.0)

        # Slightly narrower range than 1/99 improves raw cryo-EM visibility.
        vmin, vmax = np.percentile(finite, [0.2, 99.8])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            center = float(np.nanmean(finite)) if finite.size else 0.0
            spread = float(np.nanstd(finite)) if finite.size else 1.0
            spread = spread if np.isfinite(spread) and spread > 0 else 1.0
            return (center - 2.5 * spread, center + 2.5 * spread)
        return (float(vmin), float(vmax))

    def _overlay_rgb_mask(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        color_rgb: Sequence[float],
        alpha: float,
    ) -> None:
        mask_bool = np.asarray(mask, dtype=bool)
        if not np.any(mask_bool):
            return
        color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
        rgb[mask_bool] = ((1.0 - alpha) * rgb[mask_bool]) + (alpha * color.reshape(3))

    def _downsample_overlay_to_display(
        self,
        arr: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        data = np.asarray(arr)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D overlay array, got shape={data.shape}")

        th, tw = int(target_shape[0]), int(target_shape[1])
        stride = max(1, int(getattr(self, "current_display_stride", 1)))
        down = data[::stride, ::stride]

        if down.shape == (th, tw):
            return down

        fixed = np.zeros((th, tw), dtype=down.dtype)
        h = min(th, int(down.shape[0]))
        w = min(tw, int(down.shape[1]))
        fixed[:h, :w] = down[:h, :w]
        return fixed

    def _compose_qt_display_image(self) -> Optional[np.ndarray]:
        display_img = self.current_image_display
        if display_img is None and self.current_image is not None:
            display_img = np.asarray(self.current_image, dtype=np.float32)
        if display_img is None:
            return None

        base = np.asarray(display_img, dtype=np.float32)
        if base.ndim != 2:
            return None
        base = np.clip(base, 0.0, 1.0)
        rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32, copy=False)
        target_shape = tuple(int(x) for x in base.shape[:2])
        stride = max(1, int(getattr(self, "current_display_stride", 1)))

        if (
            self.show_mask_overlays
            and self.current_mask is not None
            and self.current_mask.size
            and np.any(self.current_mask > 0.5)
        ):
            mask_disp = self._downsample_overlay_to_display(self.current_mask, target_shape) > 0.5
            typed_disp = None
            if self.current_typed_mask is not None and self.current_typed_mask.shape == self.current_mask.shape:
                typed_disp = self._downsample_overlay_to_display(
                    reconcile_typed_mask(self.current_mask, self.current_typed_mask),
                    target_shape,
                )

            unlabeled_disp = mask_disp.copy()
            if typed_disp is not None:
                unlabeled_disp &= typed_disp == np.uint8(UNLABELED_CONTAMINATION_LABEL)
                for type_id in TYPE_ID_SEQUENCE:
                    type_mask = typed_disp == np.uint8(type_id)
                    if np.any(type_mask):
                        self._overlay_rgb_mask(
                            rgb,
                            type_mask,
                            matplotlib.colors.to_rgb(TYPE_ID_TO_COLOR_HEX[type_id]),
                            0.55,
                        )

            self._overlay_rgb_mask(rgb, unlabeled_disp, (1.0, 0.0, 0.0), 0.34)

        if (
            self.show_mask_overlays
            and
            self.current_mask is not None
            and 0 <= self.selected_region < len(self.regions)
            and self.regions[self.selected_region].shape == self.current_mask.shape
        ):
            region_disp = self._downsample_overlay_to_display(
                self.regions[self.selected_region],
                target_shape,
            )
            self._overlay_rgb_mask(rgb, region_disp, (1.0, 1.0, 0.0), 0.55)

        out = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        if self.current_polygon:
            try:
                from PIL import Image, ImageDraw

                pil_img = Image.fromarray(out, mode="RGB")
                draw = ImageDraw.Draw(pil_img)
                xs = [float(p[0]) / stride for p in self.current_polygon]
                ys = [float(p[1]) / stride for p in self.current_polygon]
                pts = list(zip(xs, ys))
                if self._is_erase_polygon_mode():
                    preview_color = "#FF7A00"
                elif self.edit_mode != "GT mask" and self.active_type_id in TYPE_ID_SEQUENCE:
                    preview_color = TYPE_ID_TO_COLOR_HEX[int(self.active_type_id)]
                elif self.edit_mode != "GT mask":
                    preview_color = "#00B8D4"
                else:
                    preview_color = "#39B54A"
                draw.line(pts, fill=preview_color, width=2)
                for x, y in pts:
                    r = 4
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=preview_color, fill=preview_color)
                out = np.asarray(pil_img, dtype=np.uint8)
            except Exception:
                pass

        return out

    def _qt_view_info_text(self) -> str:
        if self.current_row_idx is None or self.current_image is None:
            if self.row_loading and self.current_row_idx is not None:
                return f"Loading {self.keys[self.current_row_idx]}..."
            return f"Split={self.split_filter} | Dataset filter={self.dataset_filter_text or 'ALL'}"

        row_idx = self.current_row_idx
        key = self.keys[row_idx]
        split_label = self.split_labels[row_idx]
        dataset_id = self.dataset_ids[row_idx]
        modified = " [SESSION]" if self._key_has_session_artifacts(key) else ""
        stride = max(1, int(getattr(self, "current_display_stride", 1)))
        n_regions = len(self.regions)
        typed_stats = typed_completion_stats(
            self.current_mask,
            self.current_typed_mask if self.current_typed_mask is not None else build_unlabeled_typed_mask(self.current_mask),
        )
        active_type_name = (
            TYPE_ID_TO_DISPLAY[int(self.active_type_id)]
            if self.active_type_id in TYPE_ID_SEQUENCE
            else "None"
        )
        return (
            f"{key} | {split_label}{modified} | dataset={dataset_id} | regions={n_regions} | stride={stride}x | "
            f"mode={self.edit_mode} | active={active_type_name} | "
            f"typed={100.0 * float(typed_stats['typed_completion_fraction']):.1f}% | "
            f"masks={'on' if self.show_mask_overlays else 'off'} | "
            f"unlabeled_px={int(typed_stats['typed_unlabeled_px'])}"
        )

    def _update_display(self):
        """Redraw image, mask overlays, and status."""
        if self._qt_browser_ready and self.qt_image_view is not None:
            rgb = self._compose_qt_display_image()
            if rgb is not None:
                h, w, _ = rgb.shape
                qimg = self._qtgui.QImage(
                    rgb.data,
                    w,
                    h,
                    int(rgb.strides[0]),
                    self._qtgui.QImage.Format.Format_RGB888,
                ).copy()
                self.qt_image_view.set_pixmap(self._qtgui.QPixmap.fromImage(qimg))
            else:
                self.qt_image_view.set_pixmap(self._qtgui.QPixmap())
            if self.qt_view_info_label is not None:
                self.qt_view_info_label.setText(self._qt_view_info_text())
            self._sync_tk_control_state()
            return

        self.ax.clear()

        if self.current_row_idx is None or self.current_image is None:
            if self.row_loading and self.current_row_idx is not None:
                key = self.keys[self.current_row_idx]
                self.ax.text(
                    0.5,
                    0.55,
                    f"Loading {key}",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    fontsize=14,
                )
                self.ax.text(
                    0.5,
                    0.45,
                    "Navigation stays live while the next image is read from disk.",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    fontsize=10,
                )
            else:
                self.ax.text(
                    0.5,
                    0.55,
                    "No images match the current filters",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    fontsize=14,
                )
                self.ax.text(
                    0.5,
                    0.45,
                    f"Split={self.split_filter} | Dataset filter={self.dataset_filter_text or 'ALL'}",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    fontsize=10,
                )
            self.ax.set_axis_off()
            self._sync_tk_control_state()
            self._force_canvas_redraw()
            return

        display_img = self.current_image_display
        if display_img is None and self.current_image is not None:
            # Fallback if preview cache was not built for some reason.
            display_img = np.asarray(self.current_image, dtype=np.float32)
        if display_img is None:
            display_img = np.zeros((64, 64), dtype=np.float32)
        stride = max(1, int(getattr(self, "current_display_stride", 1)))
        target_shape = tuple(int(x) for x in np.asarray(display_img).shape[:2])

        # Base image (already normalized to [0,1] in preview cache)
        vmin, vmax = self._image_display_limits()
        self.ax.imshow(
            display_img,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # GT mask and typed-submask overlays.
        if (
            self.show_mask_overlays
            and self.current_mask is not None
            and self.current_mask.size
            and np.any(self.current_mask > 0.5)
        ):
            mask_disp = self._downsample_overlay_to_display(self.current_mask, target_shape) > 0.5
            typed_disp = None
            if self.current_typed_mask is not None and self.current_typed_mask.shape == self.current_mask.shape:
                typed_disp = self._downsample_overlay_to_display(
                    reconcile_typed_mask(self.current_mask, self.current_typed_mask),
                    target_shape,
                )

            unlabeled_disp = mask_disp.copy()
            if typed_disp is not None:
                unlabeled_disp &= typed_disp == np.uint8(UNLABELED_CONTAMINATION_LABEL)
                for type_id in TYPE_ID_SEQUENCE:
                    type_mask = typed_disp == np.uint8(type_id)
                    if not np.any(type_mask):
                        continue
                    rgb = matplotlib.colors.to_rgb(TYPE_ID_TO_COLOR_HEX[type_id])
                    type_rgba = np.zeros((*type_mask.shape, 4), dtype=np.float32)
                    type_rgba[type_mask, 0] = rgb[0]
                    type_rgba[type_mask, 1] = rgb[1]
                    type_rgba[type_mask, 2] = rgb[2]
                    type_rgba[type_mask, 3] = 0.55
                    self.ax.imshow(type_rgba, interpolation="nearest")

            gt_rgba = np.zeros((*mask_disp.shape, 4), dtype=np.float32)
            gt_rgba[unlabeled_disp, 0] = 1.0
            gt_rgba[unlabeled_disp, 3] = 0.34
            self.ax.imshow(gt_rgba, interpolation="nearest")

        # Selected region highlight in yellow
        if (
            self.show_mask_overlays
            and
            self.current_mask is not None
            and 0 <= self.selected_region < len(self.regions)
            and self.regions[self.selected_region].shape == self.current_mask.shape
        ):
            region_disp = self._downsample_overlay_to_display(
                self.regions[self.selected_region],
                target_shape,
            )
            region_rgba = np.zeros((*region_disp.shape, 4), dtype=np.float32)
            region_rgba[region_disp, 0] = 1.0
            region_rgba[region_disp, 1] = 1.0
            region_rgba[region_disp, 3] = 0.55
            self.ax.imshow(region_rgba, interpolation="nearest")

        # Current polygon preview (green)
        if self.current_polygon:
            xs = [p[0] / stride for p in self.current_polygon]
            ys = [p[1] / stride for p in self.current_polygon]
            if self._is_erase_polygon_mode():
                preview_color = "#FF7A00"
            elif self.edit_mode != "GT mask" and self.active_type_id in TYPE_ID_SEQUENCE:
                preview_color = TYPE_ID_TO_COLOR_HEX[int(self.active_type_id)]
            elif self.edit_mode != "GT mask":
                preview_color = "#00B8D4"
            else:
                preview_color = "#39B54A"
            self.ax.plot(xs, ys, "-", color=preview_color, linewidth=2)
            self.ax.plot(xs, ys, "o", color=preview_color, markersize=6)

        row_idx = self.current_row_idx
        key = self.keys[row_idx]
        split_label = self.split_labels[row_idx]
        dataset_id = self.dataset_ids[row_idx]
        modified = " [SESSION]" if self._key_has_session_artifacts(key) else ""
        n_regions = len(self.regions)
        typed_stats = typed_completion_stats(
            self.current_mask,
            self.current_typed_mask if self.current_typed_mask is not None else build_unlabeled_typed_mask(self.current_mask),
        )
        active_type_name = (
            TYPE_ID_TO_DISPLAY[int(self.active_type_id)]
            if self.active_type_id in TYPE_ID_SEQUENCE
            else "None"
        )

        title_line1 = (
            f"[view {self.view_pos + 1}/{len(self.view_indices)} | total {len(self.df)}] "
            f"{key} | {split_label}{modified}"
        )
        title_line2 = (
            f"dataset={dataset_id} | regions={n_regions} | "
            f"split_filter={self.split_filter} | dataset_filter={self.dataset_filter_text or 'ALL'} | "
            f"display_stride={stride}x"
        )
        title_line3 = (
            f"mode={self.edit_mode} | active_type={active_type_name} | "
            f"typed_completion={100.0 * float(typed_stats['typed_completion_fraction']):.1f}% | "
            f"typed_unlabeled_px={int(typed_stats['typed_unlabeled_px'])} | "
            f"masks={'on' if self.show_mask_overlays else 'off'} | session={self.run_dir.name}"
        )
        title_line4 = (
            "GT: L add R/Ctrl+L/Enter complete | Erase: L add R/Ctrl+L/Enter subtract | "
            "Full: L apply whole scope | Poly: L add R/Ctrl+L/Enter apply | "
            "M:select scope  b/x/f/l:mode  1-4 label  0/u clear  d:del-reg  c:clear  r:redraw  "
            "o:masks  n/p:[/] nav  s:row e:finalize-export q:checkpoint+quit"
        )
        self.ax.set_title(f"{title_line1}\n{title_line2}\n{title_line3}\n{title_line4}", fontsize=8.5)
        self.ax.set_axis_off()
        self._sync_tk_control_state()
        self._force_canvas_redraw()

    def _step(self, delta: int):
        if not self.view_indices:
            return
        self._maybe_refresh_sorted_view()
        if not self.view_indices:
            return
        self.view_pos = int(np.clip(self.view_pos + delta, 0, len(self.view_indices) - 1))
        self._load_current()

    def _on_close_event(self, _event) -> None:
        if not self._skip_close_event_checkpoint:
            try:
                self._write_session_checkpoint(reason="window_close")
            except Exception as exc:
                print(f"  Warning: failed to save session checkpoint on close: {exc}")
        self._skip_close_event_checkpoint = False
        if self._close_cleanup_done:
            return
        self._close_cleanup_done = True
        self._cancel_active_row_load()
        executor = getattr(self, "prefetch_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.prefetch_executor = None
        interactive_executor = getattr(self, "interactive_load_executor", None)
        if interactive_executor is not None:
            try:
                interactive_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.interactive_load_executor = None

    def _step_dataset(self, delta: int):
        """Jump to next/prev dataset in the current filtered view."""
        if not self.view_indices or self.current_row_idx is None:
            return
        self._maybe_refresh_sorted_view()
        if not self.view_indices or self.current_row_idx is None:
            return

        current_dataset = self.dataset_ids[self.current_row_idx]
        pos = self.view_pos
        step = 1 if delta >= 0 else -1
        i = pos + step
        while 0 <= i < len(self.view_indices):
            row_idx = self.view_indices[i]
            if self.dataset_ids[row_idx] != current_dataset:
                self.view_pos = i
                self._load_current()
                return
            i += step
        print("  No further dataset in that direction within current filter")

    def _on_click(self, event):
        """Handle mouse clicks for polygon drawing and region selection."""
        if event.inaxes != self.ax or self.current_row_idx is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self._handle_canvas_click(int(event.button), float(x), float(y))

    def _handle_canvas_click(self, button: int, x_disp: float, y_disp: float):
        if self.current_row_idx is None:
            return
        stride = max(1, int(getattr(self, "current_display_stride", 1)))
        x_full = float(x_disp) * stride
        y_full = float(y_disp) * stride

        if button == 1 and self._is_full_label_mode():
            self._select_region_at(int(x_full), int(y_full), announce=False)
            if self.selected_region < 0:
                print("  Click directly on a GT region to label it")
                self._update_display()
                return
            if self.active_type_id in TYPE_ID_SEQUENCE:
                self._apply_typed_label_to_selected_region(int(self.active_type_id))
            else:
                print("  Region selected. Press 1/2/3/4 or choose a subtype to assign it.")
                self._update_display()
            return

        if button == 1:  # left click = add point
            self.current_polygon.append((x_full, y_full))
            self._update_display()
            return

        if button == 3:  # right click = complete polygon
            self._complete_current_polygon()
            return

        if button == 2:  # middle click = select region
            self._select_region_at(int(x_full), int(y_full))
            self._update_display()

    def _complete_current_polygon(self) -> bool:
        if len(self.current_polygon) < 3:
            print("  Polygon needs at least 3 points before it can be completed")
            self._update_display()
            return False

        if self.edit_mode == "GT mask":
            self._add_polygon_as_region()
            self.current_polygon = []
            self._update_display()
            return True

        if self._is_erase_polygon_mode():
            self._subtract_polygon_from_mask()
            return True

        if self._is_polygon_label_mode() and self.active_type_id in TYPE_ID_SEQUENCE:
            self._apply_typed_label_to_current_polygon(int(self.active_type_id))
            self._update_display()
            return True

        if self._is_polygon_label_mode():
            print("  Polygon ready. Press 1/2/3/4 to assign a subtype.")
            self._update_display()
            return False

        self._update_display()
        return False

    def _on_key(self, event):
        """Handle keyboard shortcuts."""
        if event.key is None:
            return
        key = str(event.key).lower()

        if key == "n":
            self._step(1)
        elif key == "p":
            self._step(-1)
        elif key == "]":
            self._step_dataset(1)
        elif key == "[":
            self._step_dataset(-1)
        elif key == "d":
            self._delete_selected()
        elif key == "c":
            self._clear_mask()
        elif key == "r":
            self._reset_current_row_for_redraw()
        elif key == "o":
            self._toggle_mask_overlays()
        elif key == "b":
            if self.edit_mode != "GT mask":
                self.mode_radio.set_active(VALID_EDIT_MODES.index("GT mask"))
        elif key == "x":
            if self.edit_mode != "Erase polygon":
                self.mode_radio.set_active(VALID_EDIT_MODES.index("Erase polygon"))
        elif key == "f":
            if self.edit_mode != "Full label":
                self.mode_radio.set_active(VALID_EDIT_MODES.index("Full label"))
        elif key == "l":
            if self.edit_mode != "Polygon label":
                self.mode_radio.set_active(VALID_EDIT_MODES.index("Polygon label"))
        elif key in {"1", "2", "3", "4"}:
            self._assign_type_shortcut(int(key))
        elif key in {"0", "u"}:
            self._clear_typed_label_scope()
        elif key == "s":
            self._save_current()
        elif key == "e":
            self._start_headless_export()
        elif key == "q":
            self._save_and_quit()
        elif key in {"enter", "return"}:
            self._complete_current_polygon()
        elif key == "t":
            if self.split_filter != "TRAINING":
                self.split_radio.set_active(VALID_SPLIT_FILTERS.index("TRAINING"))
        elif key == "v":
            if self.split_filter != "VALIDATION":
                self.split_radio.set_active(VALID_SPLIT_FILTERS.index("VALIDATION"))
        elif key == "a":
            if self.split_filter != "ALL":
                self.split_radio.set_active(VALID_SPLIT_FILTERS.index("ALL"))
        elif key == "escape":
            self.current_polygon = []
            self._update_display()

    def _select_region_at(self, x: int, y: int, *, announce: bool = True) -> int:
        """Select connected region under the cursor."""
        for i, region in enumerate(self.regions):
            if 0 <= y < region.shape[0] and 0 <= x < region.shape[1] and region[y, x]:
                self.selected_region = i
                if announce:
                    print(f"  Selected region {i + 1}")
                return i
        self.selected_region = -1
        return -1

    def _mark_current_modified(self, *, mask_changed: bool = False, typed_changed: bool = False):
        if self.current_row_idx is None or self.current_mask is None:
            return
        if self.current_typed_mask is None or self.current_typed_mask.shape != self.current_mask.shape:
            self.current_typed_mask = build_unlabeled_typed_mask(self.current_mask)
        self.current_typed_mask = reconcile_typed_mask(self.current_mask, self.current_typed_mask)
        key = self.keys[self.current_row_idx]
        if mask_changed:
            self.dirty_mask_keys.add(key)
        if typed_changed:
            self.dirty_typed_keys.add(key)
        self._cache_current_row_artifacts()
        self._invalidate_row_payload(self.current_row_idx)
        self._invalidate_row_sort_stats(self.current_row_idx)
        self._mark_sorted_view_dirty()
        self._refresh_tk_current_leaf_label()

    def _add_polygon_as_region(self):
        """Rasterize current polygon into mask."""
        if self.current_mask is None or len(self.current_polygon) < 3:
            return

        h, w = self.current_mask.shape
        xs = [p[0] for p in self.current_polygon]
        ys = [p[1] for p in self.current_polygon]
        before = self.current_mask > 0.5
        rr, cc = draw_polygon(ys, xs, shape=(h, w))
        self.current_mask[rr, cc] = 1.0
        after = self.current_mask > 0.5

        if self.current_typed_mask is None or self.current_typed_mask.shape != self.current_mask.shape:
            self.current_typed_mask = build_unlabeled_typed_mask(self.current_mask)
        else:
            self.current_typed_mask = reconcile_typed_mask(self.current_mask, self.current_typed_mask)
            added = after & (~before)
            self.current_typed_mask[added] = np.uint8(UNLABELED_CONTAMINATION_LABEL)

        self._mark_current_modified(mask_changed=True, typed_changed=True)
        self._extract_regions()
        print(f"  Added polygon region ({len(self.current_polygon)} points)")

    def _subtract_polygon_from_mask(self):
        """Rasterize current polygon and remove that area from the current GT mask."""
        if self.current_mask is None or len(self.current_polygon) < 3:
            return
        poly_mask = self._rasterize_current_polygon_mask()
        if poly_mask is None:
            return

        current = np.asarray(self.current_mask > 0.5, dtype=bool)
        erase_mask = np.asarray(poly_mask & current, dtype=bool)
        erased_px = int(np.sum(erase_mask))
        self.current_polygon = []

        if erased_px <= 0:
            print("  Erase polygon did not overlap the existing GT mask")
            self._update_display()
            return

        self.current_mask[erase_mask] = 0.0
        if self.current_typed_mask is None or self.current_typed_mask.shape != self.current_mask.shape:
            self.current_typed_mask = build_unlabeled_typed_mask(self.current_mask)
        else:
            self.current_typed_mask[erase_mask] = np.uint8(BACKGROUND_LABEL)
            self.current_typed_mask = reconcile_typed_mask(self.current_mask, self.current_typed_mask)

        self._mark_current_modified(mask_changed=True, typed_changed=True)
        self._extract_regions()
        self._update_display()
        print(f"  Erased polygon from GT mask ({erased_px} px)")

    def _delete_selected(self):
        """Delete selected connected component."""
        if self.current_mask is None:
            return
        if 0 <= self.selected_region < len(self.regions):
            region = np.asarray(self.regions[self.selected_region], dtype=bool)
            self.current_mask[region] = 0.0
            if self.current_typed_mask is not None and self.current_typed_mask.shape == self.current_mask.shape:
                self.current_typed_mask[region] = np.uint8(BACKGROUND_LABEL)
            self._mark_current_modified(mask_changed=True, typed_changed=True)
            self._extract_regions()
            self._update_display()
            print("  Deleted region")

    def _clear_mask(self):
        """Clear all mask regions."""
        if self.current_mask is None:
            return
        self.current_mask = np.zeros_like(self.current_mask, dtype=np.float32)
        self.current_typed_mask = np.zeros_like(self.current_mask, dtype=np.uint8)
        self._mark_current_modified(mask_changed=True, typed_changed=True)
        self._extract_regions()
        self._update_display()
        print("  Cleared all regions")

    def _reset_current_row_for_redraw(self):
        """Clear the current GT/typed mask and switch back to GT-mask drawing mode."""
        if self.current_mask is None:
            return
        self._clear_mask()
        self.selected_region = -1
        self.current_polygon = []
        if self.edit_mode != "GT mask":
            self._set_edit_mode("GT mask")
        else:
            self._update_display()
        print("  Current GT mask reset. Redraw it in GT mask mode, then save/checkpoint when ready.")

    def _save_current(self):
        """Mark current mask as saved/modified in-session."""
        if self.current_row_idx is None or self.current_mask is None:
            return False
        key = self.keys[self.current_row_idx]
        write_mask = key in self.dirty_mask_keys or (
            key not in self.persisted_mask_keys and not self._row_has_resolved_source_gt_mask(self.current_row_idx)
        )
        write_typed = key in self.dirty_typed_keys

        if not write_mask and not write_typed:
            print(f"  No new row changes to save for {key}")
            return False

        self._persist_current_row_artifacts(write_mask=write_mask, write_typed=write_typed)
        self.dirty_mask_keys.discard(key)
        self.dirty_typed_keys.discard(key)
        print(f"  Saved current row session files for {key}")
        self._mark_sorted_view_dirty()
        return True

    def _save_all_dirty_rows(self) -> int:
        dirty_keys = sorted(
            self.dirty_mask_keys | self.dirty_typed_keys,
            key=lambda key: self.row_idx_by_key.get(key, len(self.df)),
        )
        saved_rows = 0

        for key in dirty_keys:
            row_idx = self.row_idx_by_key.get(key)
            if row_idx is None:
                continue

            row_mask = self.modified_masks.get(key)
            if row_mask is None:
                continue
            binary_mask = (np.asarray(row_mask) > 0.5).astype(np.float32)

            row_typed = self.modified_typed_masks.get(key)
            if row_typed is None or np.asarray(row_typed).shape != binary_mask.shape:
                typed_mask = build_unlabeled_typed_mask(binary_mask)
            else:
                typed_mask = reconcile_typed_mask(binary_mask, np.asarray(row_typed, dtype=np.uint8))

            write_mask = key in self.dirty_mask_keys or (
                key not in self.persisted_mask_keys and not self._row_has_resolved_source_gt_mask(row_idx)
            )
            write_typed = key in self.dirty_typed_keys
            if not write_mask and not write_typed:
                continue

            self._persist_row_artifacts(
                int(row_idx),
                binary_mask,
                typed_mask,
                write_mask=write_mask,
                write_typed=write_typed,
            )
            self.dirty_mask_keys.discard(key)
            self.dirty_typed_keys.discard(key)
            saved_rows += 1

        return saved_rows

    def _write_session_checkpoint(self, *, reason: str) -> int:
        if self._checkpoint_in_progress:
            return 0

        self._checkpoint_in_progress = True
        try:
            saved_rows = self._save_all_dirty_rows()
            self._ensure_session_dirs()
            self._refresh_session_inventory()

            new_df = self.df.copy()
            if "gt_mask_path" not in new_df.columns:
                new_df["gt_mask_path"] = ""
            for col in TYPE_MANIFEST_COLUMNS:
                if col not in new_df.columns:
                    new_df[col] = ""

            for row_idx in range(len(self.df)):
                key = self.keys[row_idx]
                row = self.df.iloc[row_idx]

                if key in self.persisted_mask_keys:
                    new_df.loc[row_idx, "gt_mask_path"] = path_for_manifest(
                        self._session_mask_path_for_key(key),
                        self.manifest_dir,
                    )
                elif self.has_gt_col:
                    value = row.get("gt_mask_path", "")
                    new_df.loc[row_idx, "gt_mask_path"] = "" if pd.isna(value) else str(value)
                else:
                    new_df.loc[row_idx, "gt_mask_path"] = ""

                if key in self.persisted_typed_keys:
                    typed_out_path = self._session_typed_label_path_for_key(key)
                    meta_out_path = self._session_typed_metadata_path_for_key(key)
                    new_df.loc[row_idx, "typed_label_path"] = path_for_manifest(
                        typed_out_path,
                        self.manifest_dir,
                    )
                    new_df.loc[row_idx, "typed_label_metadata_path"] = (
                        path_for_manifest(meta_out_path, self.manifest_dir)
                        if meta_out_path.exists()
                        else ""
                    )
                elif self.has_typed_col:
                    typed_value = row.get("typed_label_path", "")
                    meta_value = row.get("typed_label_metadata_path", "")
                    new_df.loc[row_idx, "typed_label_path"] = "" if pd.isna(typed_value) else str(typed_value)
                    new_df.loc[row_idx, "typed_label_metadata_path"] = "" if pd.isna(meta_value) else str(meta_value)
                else:
                    new_df.loc[row_idx, "typed_label_path"] = ""
                    new_df.loc[row_idx, "typed_label_metadata_path"] = ""

                new_df.loc[row_idx, "typed_label_schema_name"] = TYPE_SCHEMA_NAME
                new_df.loc[row_idx, "typed_label_schema_version"] = int(TYPE_SCHEMA_VERSION)
                new_df.loc[row_idx, "typed_label_unlabeled_value"] = int(UNLABELED_CONTAMINATION_LABEL)

            new_df.to_csv(self.session_manifest_out, index=False)

            metadata = {
                "timestamp": self.export_timestamp,
                "source_manifest": str(self.manifest_path),
                "val_manifest": str(self.val_manifest_path) if self.val_manifest_path else None,
                "run_dir": str(self.run_dir),
                "masks_out_dir": str(self.masks_out_dir),
                "typed_labels_out_dir": str(self.typed_labels_out_dir),
                "typed_metadata_out_dir": str(self.typed_metadata_out_dir),
                "exported_manifest": str(self.session_manifest_out),
                "export_only_modified": bool(self.export_only_modified),
                "full_export": False,
                "checkpoint_only": True,
                "checkpoint_reason": str(reason),
                "n_manifest_rows": int(len(self.df)),
                "n_session_mask_files": int(len(self.persisted_mask_keys)),
                "n_session_typed_label_files": int(len(self.persisted_typed_keys)),
                "n_dirty_binary_rows_remaining": int(len(self.dirty_mask_keys)),
                "n_dirty_typed_rows_remaining": int(len(self.dirty_typed_keys)),
                "n_rows_flushed_in_checkpoint": int(saved_rows),
                "filter_state_at_export": {
                    "split_filter": self.split_filter,
                    "dataset_filter_text": self.dataset_filter_text,
                    "dataset_filter_ids": sorted(self.dataset_filter_ids),
                    "n_rows_in_current_view": int(len(self.view_indices)),
                    "edit_mode": self.edit_mode,
                    "active_type_id": int(self.active_type_id) if self.active_type_id in TYPE_ID_SEQUENCE else None,
                },
                "split_metadata": self.split_metadata,
                "modified_binary_keys": sorted(self.modified_masks.keys()),
                "modified_typed_keys": sorted(self.modified_typed_masks.keys()),
                "typed_label_schema_name": TYPE_SCHEMA_NAME,
                "typed_label_schema_version": int(TYPE_SCHEMA_VERSION),
                "typed_label_unlabeled_value": int(UNLABELED_CONTAMINATION_LABEL),
            }
            self.session_metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            self.output_root.mkdir(parents=True, exist_ok=True)
            latest_run_txt = self.output_root / "LATEST_RUN.txt"
            latest_manifest_txt = self.output_root / "LATEST_MANIFEST.txt"
            latest_run_txt.write_text(str(self.run_dir) + "\n", encoding="utf-8")
            latest_manifest_txt.write_text(str(self.session_manifest_out) + "\n", encoding="utf-8")

            print(
                f"  Session checkpoint saved ({saved_rows} dirty row"
                f"{'' if saved_rows == 1 else 's'} flushed) -> {self.run_dir}"
            )
            return saved_rows
        finally:
            self._checkpoint_in_progress = False

    def _can_finalize_existing_session_files(self) -> bool:
        self._ensure_session_dirs()
        self._refresh_session_inventory()
        typed_metadata_count = sum(1 for _ in self.typed_metadata_out_dir.glob("*.json"))
        return (
            len(self.persisted_mask_keys) >= len(self.df)
            and len(self.persisted_typed_keys) >= len(self.df)
            and typed_metadata_count >= len(self.df)
        )

    def _schedule_export_poll(self, delay_ms: int = 750) -> None:
        if self.export_process is None:
            return
        if self._qt_browser_ready and self.qt_after_host is not None:
            try:
                self._qtcore.QTimer.singleShot(int(delay_ms), self._poll_headless_export)
                return
            except Exception:
                pass
        host = self.tk_after_host
        if host is None:
            return
        try:
            host.after(int(delay_ms), self._poll_headless_export)
        except Exception:
            pass

    def _poll_headless_export(self) -> None:
        process = self.export_process
        if process is None:
            return
        exit_code = process.poll()
        if exit_code is None:
            self.export_status_detail = "running"
            self._sync_tk_control_state()
            self._schedule_export_poll()
            return

        self.export_process = None
        if exit_code == 0:
            self.export_status_detail = "done"
            print(f"  Finalize export finished -> {self.session_manifest_out}")
            self._refresh_session_inventory()
            self.row_sort_stats_cache.clear()
            self._rebuild_view(preserve_current=True)
        else:
            self.export_status_detail = f"failed({exit_code})"
            print(f"  Finalize export failed with exit code {exit_code}")
            if self.export_last_command:
                print("   Command: " + " ".join(str(part) for part in self.export_last_command))
            self._sync_tk_control_state()

    def _start_headless_export(self) -> None:
        if self.export_process is not None and self.export_process.poll() is None:
            print("  Finalize export is already running")
            self.export_status_detail = "running"
            self._sync_tk_control_state()
            return

        try:
            self._write_session_checkpoint(reason="pre_export")
        except Exception as exc:
            print(f"  Could not checkpoint before export: {exc}")
            self.export_status_detail = "checkpoint-failed"
            self._sync_tk_control_state()
            return

        script_path = Path(__file__).resolve().parent / "export_contamination_typing_session.py"
        command = [
            sys.executable,
            str(script_path),
            "--session_dir",
            str(self.run_dir),
            "--manifest",
            str(self.manifest_path),
        ]
        if self.mic_dir is not None:
            command.extend(["--mic_dir", str(self.mic_dir)])
        if self.val_manifest_path is not None:
            command.extend(["--val_manifest", str(self.val_manifest_path)])
        if self.export_only_modified:
            command.append("--export_only_modified")
        if self._can_finalize_existing_session_files():
            command.append("--finalize_existing_only")

        self.export_last_command = command
        self.export_status_detail = "running"
        print("  Starting headless finalize export")
        print("   " + " ".join(str(part) for part in command))
        try:
            self.export_process = subprocess.Popen(command, cwd=str(self.workspace_root))
        except Exception as exc:
            self.export_process = None
            self.export_status_detail = "launch-failed"
            print(f"  Failed to start finalize export: {exc}")
            self._sync_tk_control_state()
            return
        self._sync_tk_control_state()
        self._schedule_export_poll()

    def _save_and_quit(self) -> None:
        try:
            self._write_session_checkpoint(reason="quit")
        except Exception as exc:
            print(f"  Save-and-quit checkpoint failed: {exc}")
            return
        self._skip_close_event_checkpoint = True
        plt.close(self.fig)

    def _rasterize_current_polygon_mask(self) -> Optional[np.ndarray]:
        if self.current_mask is None or len(self.current_polygon) < 3:
            return None
        h, w = self.current_mask.shape
        xs = [p[0] for p in self.current_polygon]
        ys = [p[1] for p in self.current_polygon]
        rr, cc = draw_polygon(ys, xs, shape=(h, w))
        poly_mask = np.zeros((h, w), dtype=bool)
        poly_mask[rr, cc] = True
        return poly_mask

    def _set_active_type_id(self, type_id: Optional[int], *, sync_widget: bool = True) -> None:
        self.active_type_id = int(type_id) if type_id in TYPE_ID_SEQUENCE else None
        if sync_widget and hasattr(self, "type_radio"):
            target_idx = 0 if self.active_type_id is None else TYPE_ID_SEQUENCE.index(int(self.active_type_id)) + 1
            try:
                self.type_radio.set_active(target_idx)
            except Exception:
                pass
        self._sync_tk_control_state()

    def _assign_type_shortcut(self, type_id: int):
        if type_id not in TYPE_ID_SEQUENCE:
            return
        if self._is_erase_polygon_mode():
            print("  Erase-polygon mode is active. Complete the erase polygon or switch modes before assigning types.")
            self._update_display()
            return
        self._set_active_type_id(type_id)
        if self.edit_mode == "GT mask":
            try:
                self.mode_radio.set_active(VALID_EDIT_MODES.index("Full label"))
            except Exception:
                self.edit_mode = "Full label"

        if self._is_polygon_label_mode() and self.current_polygon and len(self.current_polygon) >= 3:
            self._apply_typed_label_to_current_polygon(type_id)
            return

        if self._is_full_label_mode() and 0 <= self.selected_region < len(self.regions):
            self._apply_typed_label_to_selected_region(type_id)
            return

        if self._is_full_label_mode():
            print(
                f"  Active type set to {TYPE_ID_TO_DISPLAY[type_id]}. "
                "Click a GT region to assign it."
            )
        else:
            print(
                f"  Active type set to {TYPE_ID_TO_DISPLAY[type_id]}. "
                "Draw a polygon in Polygon-label mode, then right-click to apply it."
            )
        self._update_display()

    def _clear_typed_label_scope(self):
        if self._is_erase_polygon_mode():
            print("  Erase-polygon mode is active. Complete the erase polygon or switch modes before clearing typed labels.")
            self._update_display()
            return
        if self.edit_mode == "GT mask":
            try:
                self.mode_radio.set_active(VALID_EDIT_MODES.index("Full label"))
            except Exception:
                self.edit_mode = "Full label"
        self._set_active_type_id(None)

        if self._is_full_label_mode() and self._typed_scope_mask_for_full_label() is not None:
            self._apply_typed_label_to_full_scope(None)
            return

        if self._is_polygon_label_mode() and self.current_polygon and len(self.current_polygon) >= 3:
            self._apply_typed_label_to_current_polygon(None)
            return
        if self._is_full_label_mode() and 0 <= self.selected_region < len(self.regions):
            self._apply_typed_label_to_selected_region(None)
            return
        print("  Cleared active type selection")
        self._update_display()

    def _apply_typed_label_to_full_scope(self, type_id: Optional[int]) -> None:
        if self.current_mask is None:
            return
        scope = self._typed_scope_mask_for_full_label()
        if scope is None or not np.any(scope):
            print("  No GT mask is available for full-label assignment")
            return
        source = "selected full-label region" if 0 <= self.selected_region < len(self.regions) else "full GT mask"
        self._apply_typed_label_to_scope(scope, type_id, source=source)

    def _apply_typed_label_to_selected_region(self, type_id: Optional[int]) -> None:
        if self.current_mask is None:
            return
        scope = self._typed_scope_mask_for_selected_region()
        if scope is None or not np.any(scope):
            print("  No GT region selected for typed labeling")
            return
        self._apply_typed_label_to_scope(scope, type_id, source="selected region")

    def _apply_typed_label_to_current_polygon(self, type_id: Optional[int]) -> None:
        if self.current_mask is None:
            return
        poly_mask = self._rasterize_current_polygon_mask()
        if poly_mask is None:
            return
        scope = self._typed_scope_mask_for_selected_region()
        if scope is None:
            return
        apply_mask = np.asarray(poly_mask & scope, dtype=bool)
        if not np.any(apply_mask):
            print("  Typed polygon does not overlap the current GT mask / selected region")
            return
        self._apply_typed_label_to_scope(apply_mask, type_id, source="polygon")
        self.current_polygon = []

    def _apply_typed_label_to_scope(self, scope_mask: np.ndarray, type_id: Optional[int], *, source: str) -> None:
        if self.current_mask is None:
            return
        if self.current_typed_mask is None or self.current_typed_mask.shape != self.current_mask.shape:
            self.current_typed_mask = build_unlabeled_typed_mask(self.current_mask)
        else:
            self.current_typed_mask = reconcile_typed_mask(self.current_mask, self.current_typed_mask)

        scope = np.asarray(scope_mask, dtype=bool) & np.asarray(self.current_mask > 0.5, dtype=bool)
        if not np.any(scope):
            print("  Nothing inside GT mask to subtype-label")
            return

        if type_id in TYPE_ID_SEQUENCE:
            self.current_typed_mask[scope] = np.uint8(int(type_id))
            msg = f"  Assigned {TYPE_ID_TO_DISPLAY[int(type_id)]} to {source}"
        else:
            self.current_typed_mask[scope] = np.uint8(UNLABELED_CONTAMINATION_LABEL)
            msg = f"  Cleared typed label back to unlabeled contamination for {source}"

        self._mark_current_modified(typed_changed=True)
        self._update_display()
        print(msg)

    def _load_mask_for_export(self, row_idx: int) -> Tuple[Optional[np.ndarray], str]:
        """
        Return mask for export and source label.

        Source labels:
          - "modified": session mask (write even if empty)
          - "session": persisted session mask from a previous run
          - "existing": manifest gt_mask_path
          - "none": no mask available yet (clean image or missing path)
        """
        key = self.keys[row_idx]
        if key in self.modified_masks:
            return self.modified_masks[key].copy().astype(np.float32), "modified"

        session_mask_path = self._session_mask_path_for_key(key)
        if session_mask_path.exists():
            try:
                mask = np.squeeze(np.load(str(session_mask_path))).astype(np.float32)
                if mask.ndim != 2:
                    print(
                        f"  Warning: skipping non-2D session GT mask for {key} at "
                        f"{session_mask_path} (shape={mask.shape})"
                    )
                    return None, "none"
                return (mask > 0.5).astype(np.float32), "session"
            except Exception as exc:
                print(f"  Warning: failed to load session GT mask for export {key} from {session_mask_path}: {exc}")
                return None, "none"

        if not self.has_gt_col:
            return None, "none"

        row = self.df.iloc[row_idx]
        gt_path, _gt_col = resolve_gt_mask_path_for_row(
            row,
            self.manifest_dir,
            self.gt_mask_path_columns,
        )
        if gt_path is None or not gt_path.exists():
            return None, "none"

        try:
            mask = np.squeeze(np.load(str(gt_path))).astype(np.float32)
            if mask.ndim != 2:
                print(f"  Warning: skipping non-2D GT mask for {key} at {gt_path} (shape={mask.shape})")
                return None, "none"
            return (mask > 0.5).astype(np.float32), "existing"
        except Exception as exc:
            print(f"  Warning: failed to load GT mask for export {key} from {gt_path}: {exc}")
            return None, "none"

    def _load_typed_mask_for_export(
        self,
        row_idx: int,
        *,
        binary_mask: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], str]:
        """Return typed labels for export, synthesizing unlabeled-inside-GT when needed."""
        if binary_mask is None:
            return None, "none"

        key = self.keys[row_idx]
        if key in self.modified_typed_masks:
            return reconcile_typed_mask(binary_mask, self.modified_typed_masks[key].copy()), "modified"

        session_typed_path = self._session_typed_label_path_for_key(key)
        if session_typed_path.exists():
            try:
                typed = np.squeeze(np.load(str(session_typed_path))).astype(np.uint8)
                return reconcile_typed_mask(binary_mask, typed), "session"
            except Exception as exc:
                print(
                    f"  Warning: failed to load session typed labels for export {key} "
                    f"from {session_typed_path}: {exc}"
                )

        if self.has_typed_col:
            row = self.df.iloc[row_idx]
            typed_path = resolve_existing_path(row.get("typed_label_path", None), self.manifest_dir)
            if typed_path is not None and typed_path.exists():
                try:
                    typed = np.squeeze(np.load(str(typed_path))).astype(np.uint8)
                    return reconcile_typed_mask(binary_mask, typed), "existing"
                except Exception as exc:
                    print(f"  Warning: failed to load typed labels for export {key} from {typed_path}: {exc}")

        return build_unlabeled_typed_mask(binary_mask), "synthesized_unlabeled"

    def _synthesize_empty_mask_for_export_row(self, row_idx: int) -> Optional[np.ndarray]:
        """
        Create an explicit all-zero mask for rows without GT, using image shape.

        This is used during full exports so the output directory contains one mask
        per manifest row (including clean/unedited images).
        """
        # Reuse the currently loaded image when possible.
        if (
            self.current_row_idx == row_idx
            and self.current_image is not None
            and np.asarray(self.current_image).ndim == 2
        ):
            return np.zeros(np.asarray(self.current_image).shape[:2], dtype=np.float32)

        row = self.df.iloc[row_idx]
        stem = self.stems[row_idx]

        def _load_from_col(col_name: str) -> Optional[np.ndarray]:
            if col_name not in self.df.columns:
                return None
            return find_image(
                mic_path_value=row.get(col_name, None),
                mic_dir=self.mic_dir,
                stem=stem,
                manifest_dir=self.manifest_dir,
            )

        def _down_shape_from_full_shape(full_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            try:
                sx = float(row.get("scale_x_full_over_down", np.nan))
                sy = float(row.get("scale_y_full_over_down", np.nan))
            except Exception:
                return None
            if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
                return None
            h_full, w_full = int(full_shape[0]), int(full_shape[1])
            h_down = max(1, int(round(h_full / sy)))
            w_down = max(1, int(round(w_full / sx)))
            return (h_down, w_down)

        # 1) Prefer true downsampled image shape if available.
        for col_name in [self.mic_col, "down_mic_path"]:
            if col_name not in self.df.columns:
                continue
            img = _load_from_col(col_name)
            if img is None:
                continue
            img = np.squeeze(np.asarray(img))
            if img.ndim == 2:
                return np.zeros(img.shape[:2], dtype=np.float32)

        # 2) Fallback: use full-res image + manifest scale factors to infer downsampled mask shape.
        for col_name in ["full_mic_path", "micrograph_path", "resolved_mic_path", "staged_mic_path", "mic_path"]:
            if col_name not in self.df.columns:
                continue
            img = _load_from_col(col_name)
            if img is None:
                continue
            img = np.squeeze(np.asarray(img))
            if img.ndim != 2:
                continue
            inferred = _down_shape_from_full_shape((int(img.shape[0]), int(img.shape[1])))
            if inferred is not None:
                return np.zeros(inferred, dtype=np.float32)
            # Last resort: use loaded image shape directly.
            return np.zeros(img.shape[:2], dtype=np.float32)

        return None

    def _write_session_manifest_and_metadata(self, *, full_export: bool):
        """Write a resumable manifest/metadata package for the current session directory."""
        self._ensure_session_dirs()
        self._refresh_session_inventory()

        new_df = self.df.copy()
        if "gt_mask_path" not in new_df.columns:
            new_df["gt_mask_path"] = ""
        for col in TYPE_MANIFEST_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = ""

        exported_mask_files = 0
        exported_typed_files = 0
        zero_mask_files = 0
        fully_typed_rows = 0
        partially_typed_rows = 0
        unresolved_rows = 0
        failed_full_export_keys: List[str] = []

        for row_idx in range(len(self.df)):
            key = self.keys[row_idx]
            row = self.df.iloc[row_idx]
            resolved_gt_path, resolved_gt_col = resolve_gt_mask_path_for_row(
                row,
                self.manifest_dir,
                self.gt_mask_path_columns,
            )
            force_write_row = bool(full_export and (not self.export_only_modified or self._key_has_session_artifacts(key)))
            mask, mask_source = self._load_mask_for_export(row_idx)

            if mask is None and force_write_row:
                mask = self._synthesize_empty_mask_for_export_row(row_idx)
                if mask is not None:
                    mask_source = "synthesized_zero"

            if mask is None:
                if resolved_gt_path is not None:
                    new_df.loc[row_idx, "gt_mask_path"] = path_for_manifest(resolved_gt_path, self.manifest_dir)
                elif self.has_gt_col:
                    value = row.get("gt_mask_path", "")
                    new_df.loc[row_idx, "gt_mask_path"] = "" if pd.isna(value) else str(value)
                else:
                    new_df.loc[row_idx, "gt_mask_path"] = ""
                if self.has_typed_col:
                    typed_value = row.get("typed_label_path", "")
                    meta_value = row.get("typed_label_metadata_path", "")
                    new_df.loc[row_idx, "typed_label_path"] = "" if pd.isna(typed_value) else str(typed_value)
                    new_df.loc[row_idx, "typed_label_metadata_path"] = "" if pd.isna(meta_value) else str(meta_value)
                if force_write_row:
                    failed_full_export_keys.append(key)
                continue

            mask = np.squeeze(mask).astype(np.float32)
            if mask.ndim != 2:
                print(f"  Warning: skipping export for {key}; mask not 2D after squeeze (shape={mask.shape})")
                if force_write_row:
                    failed_full_export_keys.append(key)
                continue
            mask = (mask > 0.5).astype(np.float32)

            write_session_binary = bool(
                force_write_row or key in self.persisted_mask_keys or mask_source in {"modified", "session", "synthesized_zero"}
            )
            if write_session_binary:
                out_mask_path = self._session_mask_path_for_key(key)
                np.save(out_mask_path, mask)
                self.persisted_mask_keys.add(key)
                new_df.loc[row_idx, "gt_mask_path"] = path_for_manifest(out_mask_path, self.manifest_dir)
                exported_mask_files += 1
            elif resolved_gt_path is not None:
                new_df.loc[row_idx, "gt_mask_path"] = path_for_manifest(resolved_gt_path, self.manifest_dir)
            elif self.has_gt_col:
                value = row.get("gt_mask_path", "")
                new_df.loc[row_idx, "gt_mask_path"] = "" if pd.isna(value) else str(value)
            else:
                new_df.loc[row_idx, "gt_mask_path"] = ""

            typed_mask, typed_source = self._load_typed_mask_for_export(row_idx, binary_mask=mask)
            typed_mask = reconcile_typed_mask(mask, typed_mask) if typed_mask is not None else build_unlabeled_typed_mask(mask)
            write_session_typed = bool(
                force_write_row or key in self.persisted_typed_keys or typed_source in {"modified", "session", "synthesized_unlabeled"}
            )
            if write_session_typed:
                typed_out_path = self._session_typed_label_path_for_key(key)
                np.save(typed_out_path, typed_mask.astype(np.uint8))
                self.persisted_typed_keys.add(key)
                exported_typed_files += 1
                new_df.loc[row_idx, "typed_label_path"] = path_for_manifest(typed_out_path, self.manifest_dir)

                meta_path = self._session_typed_metadata_path_for_key(key)
                stats = typed_completion_stats(mask, typed_mask)
                counts = typed_class_pixel_counts(mask, typed_mask)
                meta_payload = {
                    "dataset_id": self.dataset_ids[row_idx],
                    "stem": self.stems[row_idx],
                    "key": key,
                    "binary_mask_path": path_for_manifest(
                        self._session_mask_path_for_key(key) if write_session_binary else Path(str(new_df.loc[row_idx, "gt_mask_path"])),
                        self.manifest_dir,
                    )
                    if str(new_df.loc[row_idx, "gt_mask_path"]).strip()
                    else "",
                    "typed_label_path": path_for_manifest(typed_out_path, self.manifest_dir),
                    "typed_label_schema_name": TYPE_SCHEMA_NAME,
                    "typed_label_schema_version": int(TYPE_SCHEMA_VERSION),
                    "typed_label_unlabeled_value": int(UNLABELED_CONTAMINATION_LABEL),
                    "type_id_to_name": {
                        str(type_id): TYPE_ID_TO_DISPLAY[type_id] for type_id in TYPE_ID_SEQUENCE
                    },
                    "counts": counts,
                    "stats": stats,
                    "source_gt_mask_path": (
                        path_for_manifest(resolved_gt_path, self.manifest_dir)
                        if resolved_gt_path is not None
                        else (str(row.get("gt_mask_path", "")) if self.has_gt_col else "")
                    ),
                    "source_gt_mask_column": resolved_gt_col or "",
                }
                meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
                new_df.loc[row_idx, "typed_label_metadata_path"] = path_for_manifest(meta_path, self.manifest_dir)
            elif self.has_typed_col:
                typed_value = row.get("typed_label_path", "")
                meta_value = row.get("typed_label_metadata_path", "")
                new_df.loc[row_idx, "typed_label_path"] = "" if pd.isna(typed_value) else str(typed_value)
                new_df.loc[row_idx, "typed_label_metadata_path"] = "" if pd.isna(meta_value) else str(meta_value)
            else:
                new_df.loc[row_idx, "typed_label_path"] = ""
                new_df.loc[row_idx, "typed_label_metadata_path"] = ""

            new_df.loc[row_idx, "typed_label_schema_name"] = TYPE_SCHEMA_NAME
            new_df.loc[row_idx, "typed_label_schema_version"] = int(TYPE_SCHEMA_VERSION)
            new_df.loc[row_idx, "typed_label_unlabeled_value"] = int(UNLABELED_CONTAMINATION_LABEL)

            zero_mask_files += int(float(mask.sum()) == 0.0)
            row_stats = typed_completion_stats(mask, typed_mask)
            if row_stats["gt_contamination_px"] <= 0:
                fully_typed_rows += 1
            elif row_stats["typed_completion_fraction"] >= 0.999999:
                fully_typed_rows += 1
            elif row_stats["typed_labeled_px"] > 0:
                partially_typed_rows += 1
            else:
                unresolved_rows += 1

        if full_export and failed_full_export_keys:
            sample = ", ".join(failed_full_export_keys[:10])
            more = "" if len(failed_full_export_keys) <= 10 else f" ... (+{len(failed_full_export_keys) - 10} more)"
            raise RuntimeError(
                "Full export did not produce a binary GT mask for every manifest row. "
                f"Failed keys: {sample}{more}"
            )

        new_df.to_csv(self.session_manifest_out, index=False)

        metadata = {
            "timestamp": self.export_timestamp,
            "source_manifest": str(self.manifest_path),
            "val_manifest": str(self.val_manifest_path) if self.val_manifest_path else None,
            "run_dir": str(self.run_dir),
            "masks_out_dir": str(self.masks_out_dir),
            "typed_labels_out_dir": str(self.typed_labels_out_dir),
            "typed_metadata_out_dir": str(self.typed_metadata_out_dir),
            "exported_manifest": str(self.session_manifest_out),
            "export_only_modified": bool(self.export_only_modified),
            "full_export": bool(full_export),
            "n_manifest_rows": int(len(self.df)),
            "n_session_mask_files": int(len(self.persisted_mask_keys)),
            "n_session_typed_label_files": int(len(self.persisted_typed_keys)),
            "n_exported_mask_files": int(exported_mask_files),
            "n_exported_typed_label_files": int(exported_typed_files),
            "n_zero_mask_files_written": int(zero_mask_files),
            "n_fully_typed_rows": int(fully_typed_rows),
            "n_partially_typed_rows": int(partially_typed_rows),
            "n_unresolved_typed_rows": int(unresolved_rows),
            "filter_state_at_export": {
                "split_filter": self.split_filter,
                "dataset_filter_text": self.dataset_filter_text,
                "dataset_filter_ids": sorted(self.dataset_filter_ids),
                "n_rows_in_current_view": int(len(self.view_indices)),
                "edit_mode": self.edit_mode,
                "active_type_id": int(self.active_type_id) if self.active_type_id in TYPE_ID_SEQUENCE else None,
            },
            "split_metadata": self.split_metadata,
            "modified_binary_keys": sorted(self.modified_masks.keys()),
            "modified_typed_keys": sorted(self.modified_typed_masks.keys()),
            "typed_label_schema_name": TYPE_SCHEMA_NAME,
            "typed_label_schema_version": int(TYPE_SCHEMA_VERSION),
            "typed_label_unlabeled_value": int(UNLABELED_CONTAMINATION_LABEL),
        }
        self.session_metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        self.output_root.mkdir(parents=True, exist_ok=True)
        latest_run_txt = self.output_root / "LATEST_RUN.txt"
        latest_manifest_txt = self.output_root / "LATEST_MANIFEST.txt"
        latest_run_txt.write_text(str(self.run_dir) + "\n", encoding="utf-8")
        latest_manifest_txt.write_text(str(self.session_manifest_out) + "\n", encoding="utf-8")

        print("\n" + "=" * 72)
        print("EXPORT COMPLETE" if full_export else "SESSION MANIFEST UPDATED")
        print("=" * 72)
        print(f"Session directory:         {self.run_dir}")
        print(f"Masks directory:           {self.masks_out_dir}")
        print(f"Typed labels directory:    {self.typed_labels_out_dir}")
        print(f"Typed metadata directory:  {self.typed_metadata_out_dir}")
        print(f"Manifest:                  {self.session_manifest_out}")
        print(f"Metadata:                  {self.session_metadata_out}")
        print(f"Exported binary masks:     {exported_mask_files}")
        print(f"Exported typed label maps: {exported_typed_files}")
        print(f"Fully typed rows:          {fully_typed_rows}")
        print(f"Partially typed rows:      {partially_typed_rows}")
        print(f"Unresolved typed rows:     {unresolved_rows}")
        print(f"Latest pointers:           {latest_run_txt}, {latest_manifest_txt}")
        print("=" * 72)

    def _export_all(self):
        """Write a fully packaged session export (blocking fallback path)."""
        self._write_session_manifest_and_metadata(full_export=True)

    def run(self):
        """Start the GUI."""
        print("\nControls:")
        print("  GT mask: Left click adds polygon points, Right click / Ctrl+click / Enter completes the GT polygon")
        print("  Erase polygon: Left click adds polygon points, Right click / Ctrl+click / Enter subtracts from GT")
        print("  Full label: Left click inside GT applies the active subtype to the full current scope")
        print("  Polygon label: Left click adds polygon points, Right click / Ctrl+click / Enter applies the active subtype")
        print("  Middle click = Select region")
        print("  d = Delete selected region")
        print("  c = Clear entire mask")
        print("  r = Reset current GT mask and redraw it from scratch")
        print("  o = Toggle contamination mask overlays on/off")
        print("  b / x / f / l = Switch GT-mask / erase-polygon / full-label / polygon-label mode")
        print("  1 / 2 / 3 / 4 = Carbon / Crystalline / Aggregate / Ethane")
        print("  0 or u = Clear typed label back to unlabeled contamination")
        print("  n/p = Next/Previous image (current filter)")
        print("  [ / ] = Prev/Next dataset (current filter)")
        print("  a / t / v = Split filter ALL / TRAINING / VALIDATION")
        print("  Sort menu = Dataset/Filename or progress-based review order")
        print("  s = Save the current row to the session directory")
        print("  e = Start a headless finalize export (recommended; stays responsive)")
        print("  q = Save a fast resumable session checkpoint and quit")
        print("  Enter = Complete/apply current polygon")
        print("  Escape = Cancel current polygon")
        print()
        print("Dataset filter GUI:")
        print("  - Enter one or more dataset IDs (comma or space separated), then Apply/Enter.")
        print("  - Clear resets to all datasets.")
        print()
        print("Typed-label schema:")
        print("  - 0 = clean background")
        print("  - 1 = Carbon")
        print("  - 2 = Crystalline")
        print("  - 3 = Aggregate")
        print("  - 4 = Ethane")
        print(f"  - 255 = contamination present in GT but not yet subtype-labeled")
        print()
        print(f"Session directory: {self.run_dir}")
        print()

        if self._qt_browser_ready:
            self._schedule_qt_redraw()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simple GT Mask Editor (matplotlib)")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_SELECTED_MASK_MANIFEST),
        help=(
            "Path to manifest CSV. Default: manifest_cleaned_final_gt_masks_selected.csv "
            "(the hybrid selected-mask GT manifest)."
        ),
    )
    parser.add_argument("--mic_dir", type=str, help="Optional directory containing micrographs (overrides manifest paths)")

    # Export layout (isolated runs by default)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_HYBRID_SESSION_ROOT),
        help=(
            "Root directory for annotation sessions when no exact --output_dir is provided "
            f"(default: {DEFAULT_HYBRID_SESSION_ROOT})"
        ),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional export run name under --output_root (default: run_<timestamp>)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Legacy/advanced: exact export run directory (overrides --output_root/--output_name)",
    )
    parser.add_argument(
        "--export_only_modified",
        action="store_true",
        help="Only write masks changed in this session (manifest still exported; unmodified rows retain original paths)",
    )

    # Split labeling / filtering
    parser.add_argument(
        "--val_manifest",
        type=str,
        default=None,
        help=(
            "Optional held-out validation manifest. If omitted, auto-detects "
            "testing_particle_filtering_with_crYOLO/manifests/heldout_val_manifest.csv "
            "relative to the manifest directory; otherwise falls back to seed/train_frac split."
        ),
    )
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for derived TRAINING/VALIDATION split labels")
    parser.add_argument(
        "--split_train_frac",
        type=float,
        default=0.7,
        help="Training fraction by dataset for derived split labels",
    )
    parser.add_argument(
        "--initial_split",
        type=str,
        default="ALL",
        choices=list(VALID_SPLIT_FILTERS),
        help="Initial split filter shown in GUI",
    )
    parser.add_argument(
        "--initial_dataset_filter",
        type=str,
        default="",
        help="Initial dataset filter text (comma/space separated dataset IDs)",
    )
    parser.add_argument(
        "--initial_row_key",
        type=str,
        default="",
        help="Initial row key to select after filtering, e.g. dataset_id__stem",
    )
    parser.add_argument(
        "--prefetch_workers",
        type=int,
        default=None,
        help="Background row-prefetch workers for faster navigation (default: 0/off)",
    )
    parser.add_argument(
        "--prefetch_radius",
        type=int,
        default=0,
        help="How many previous/next rows in the current filtered view to prefetch",
    )
    parser.add_argument(
        "--row_cache_size",
        type=int,
        default=8,
        help="Maximum number of loaded row payloads to keep warm in memory",
    )

    args = parser.parse_args()

    editor = SimpleMaskEditor(
        manifest_path=args.manifest,
        mic_dir=args.mic_dir,
        output_dir=args.output_dir,
        output_root=args.output_root,
        output_name=args.output_name,
        val_manifest=args.val_manifest,
        split_seed=args.split_seed,
        split_train_frac=args.split_train_frac,
        initial_split=args.initial_split,
        initial_dataset_filter=args.initial_dataset_filter,
        initial_row_key=args.initial_row_key,
        export_only_modified=args.export_only_modified,
        prefetch_workers=args.prefetch_workers,
        prefetch_radius=args.prefetch_radius,
        row_cache_size=args.row_cache_size,
    )
    editor.run()


if __name__ == "__main__":
    main()
