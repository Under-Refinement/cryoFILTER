# Publication classifier UI integration.
"""Incremental UI/CLI output adapter for publication subtype inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import typing_cli as io
from .classifier_weights import resolve_classifier_checkpoint
from .publication_typing import MODEL_DIR, PublicationClassifier

# Increment if preprocessing, inference kernels, or native-mask output changes.
PUBLICATION_CACHE_VERSION = 1


def _summaries(row, mask, result):
    dataset_id, stem = str(row["dataset_id"]), str(row["stem"])
    image_id = f"{dataset_id}__{stem}"
    typed = result["pred_map"]
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    by_type = np.stack([np.bincount(labels[typed == type_id], minlength=count + 1)
                        for type_id in io.TYPE_ID_SEQUENCE])
    confidence = np.bincount(labels.ravel(), weights=result["top1_prob"].astype(float).ravel(),
                             minlength=count + 1)
    majority = np.argmax(by_type, axis=0)
    contaminated = int(mask.sum())
    summary = {
        "image_id": image_id, "dataset_id": dataset_id, "stem": stem,
        "total_pixels": int(mask.size), "contaminated_pixels": contaminated,
        "contamination_pct_total_image": 100.0 * contaminated / max(1, mask.size),
        "n_components": int(count), "n_large_components": int(count), "n_small_components": 0,
        "unclassified_area_px": int(np.count_nonzero(mask & (typed == 0))),
    }
    components = []
    for index, label in enumerate(io.PUBLIC_TYPE_ORDER):
        slug = io.PUBLIC_TYPE_TO_KEY[label]
        area = int(np.count_nonzero(typed == io.PUBLIC_TYPE_TO_ID[label]))
        summary.update({f"{slug}_area_px": area,
                        f"{slug}_pct_total_image": 100.0 * area / max(1, mask.size),
                        f"{slug}_pct_of_contamination": 100.0 * area / max(1, contaminated),
                        f"{slug}_components": int(np.count_nonzero((majority[1:] == index) & (by_type[:, 1:].sum(axis=0) > 0)))})
    for component_id in range(1, count + 1):
        n_labeled = int(by_type[:, component_id].sum())
        item = {"image_id": image_id, "dataset_id": dataset_id, "stem": stem,
                "component_id": component_id, "area_px": int(areas[component_id]),
                "small_component_auto_ethane": False,
                "type": io.PUBLIC_TYPE_ORDER[int(majority[component_id])] if n_labeled else "Unclassified",
                "type_confidence": float(confidence[component_id] / max(1, areas[component_id])),
                "classification_scope": "component_majority",
                "unclassified_area_px": int(areas[component_id] - n_labeled)}
        for index, label in enumerate(io.PUBLIC_TYPE_ORDER):
            item[f"{io.PUBLIC_TYPE_TO_KEY[label]}_area_px"] = int(by_type[index, component_id])
        components.append(item)
    return summary, components


def run(args):
    workers = int(args.workers)
    interval = float(args.summary_interval)
    if workers < 1 or not np.isfinite(interval) or interval < 0:
        raise ValueError("Workers must be positive and summary interval finite and nonnegative")
    if args.normalization_method != "percentile_extra_wide":
        raise ValueError("The publication classifier requires percentile_extra_wide normalization")
    if int(args.typing_batch_size) < 1:
        raise ValueError("Typing batch size must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / "typed_masks"
    masks_dir.mkdir(exist_ok=True)
    cache_dir = output / ".publication_cache"
    cache_dir.mkdir(exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty or not {"dataset_id", "stem"}.issubset(manifest.columns):
        raise ValueError("Manifest requires dataset_id/stem and at least one image")
    keys = manifest["dataset_id"].astype(str) + "__" + manifest["stem"].astype(str)
    if keys.duplicated().any():
        raise ValueError("Manifest contains duplicate dataset_id/stem image IDs")
    metadata = json.loads((MODEL_DIR / "manifest.json").read_text())
    checkpoint = resolve_classifier_checkpoint(
        args.typing_checkpoint, model_dir=MODEL_DIR,
        search_dirs=(manifest_path.parent / "pretrained_models", manifest_path.parent),
    )
    fingerprint = io._digest([metadata, [io._file_stamp(MODEL_DIR / name)
                                       for name in ("models.json", "settings.json", "model_config.json")],
                              io._file_stamp(checkpoint) if checkpoint.exists() else str(checkpoint)])
    expected = max(len(manifest), int(args.expected_images or 0))
    rows = [row for _, row in manifest.iterrows()]

    def signature(row):
        paths = []
        for columns in (io.MICROGRAPH_PATH_COLUMNS, io.MASK_PATH_COLUMNS):
            column = io._first_existing_column(row, columns)
            if column is None:
                raise ValueError("Each row needs a micrograph and binary mask path")
            paths.append(io._file_stamp(io._resolve_path(manifest_path.parent, row[column])))
        return io._digest([PUBLICATION_CACHE_VERSION, fingerprint, paths, {str(k): str(v) for k, v in row.items()},
                           args.pixel_size_angstrom, args.normalization_method,
                           args.typing_batch_size, args.typing_device])

    signatures = [signature(row) for row in rows]
    records = {}
    pending = []
    for index, key in enumerate(keys):
        path = masks_dir / f"{key}_typed_mask.npy"
        cached = io._read_cache(cache_dir / f"{io._digest(key)}.json") if args.incremental else {}
        if (cached.get("signature") == signatures[index] and path.exists()
                and cached.get("mask_stamp") == io._file_stamp(path)
                and isinstance(cached.get("summary"), dict) and isinstance(cached.get("components"), list)):
            records[index] = cached
        else:
            pending.append(index)
    print(f"Publication typing: {len(records)} cached, {len(pending)} new/changed image(s).", flush=True)
    run_signature = io._digest([signatures, expected, str(manifest_path)])
    summary_files = [output / name for name in ("summary.json", "component_type_assignments.csv",
                                                "image_contamination_summary.csv", "dataset_contamination_summary.csv")]
    state_path = cache_dir / "published.json"
    state = io._read_cache(state_path) if args.incremental else {}
    if (not pending and state.get("signature") == run_signature
            and all(path.exists() for path in summary_files)
            and state.get("summary_stamps") == [io._file_stamp(path) for path in summary_files]):
        print("Publication typing already current; reused published outputs.", flush=True)
        return 0

    def publish(final=False):
        selected = [records[index] for index in sorted(records)]
        images = pd.DataFrame([item["summary"] for item in selected])
        components = pd.DataFrame([row for item in selected for row in item["components"]])
        if components.empty:
            components = pd.DataFrame(columns=io.COMPONENT_SUMMARY_COLUMNS)
        datasets = io._dataset_summary_df(images)
        overall = io._overall_summary(images)
        overall["unclassified_area_px"] = int(images["unclassified_area_px"].sum())
        io._write_csv_atomic(components, output / "component_type_assignments.csv")
        io._write_csv_atomic(images, output / "image_contamination_summary.csv")
        io._write_csv_atomic(datasets, output / "dataset_contamination_summary.csv")
        io._write_json_atomic(output / "summary.json", {
            "manifest": str(manifest_path), "classifier": "publication", "model_id": metadata["model_id"],
            "classifier_checkpoint": str(checkpoint),
            "classifier_manifest_sha256": io._digest(metadata),
            "output_dir": str(output), "normalization_method": "percentile_extra_wide",
            "type_order": list(io.PUBLIC_TYPE_ORDER), "type_to_id": dict(io.PUBLIC_TYPE_TO_ID),
            "overall_contamination_summary": overall,
            "component_type_assignments_csv": str(output / "component_type_assignments.csv"),
            "image_contamination_summary_csv": str(output / "image_contamination_summary.csv"),
            "dataset_contamination_summary_csv": str(output / "dataset_contamination_summary.csv"),
            "typed_mask_dir": str(masks_dir), "n_images": len(images), "n_images_expected": expected,
            "n_components": int(images["n_components"].sum()),
            "typing_status": "complete" if final and len(images) >= expected else "running",
        })
        io._write_json_atomic(state_path, {"signature": run_signature if final else None,
                                           "summary_stamps": [io._file_stamp(path) for path in summary_files]})
        print(f"Publication typed {len(images)}/{expected} image(s); summary updated.", flush=True)
        if final:
            io._print_overall_summary(overall)

    classifier = None
    last_publish = time.monotonic()
    if records and pending:
        publish()
    for index in pending:
        row = rows[index]
        key = str(keys.iloc[index])
        micro_column = io._first_existing_column(row, io.MICROGRAPH_PATH_COLUMNS)
        mask_column = io._first_existing_column(row, io.MASK_PATH_COLUMNS)
        image, header_px = io._load_micrograph(io._resolve_path(manifest_path.parent, row[micro_column]))
        mask = io._load_mask(io._resolve_path(manifest_path.parent, row[mask_column]))
        px = io._resolve_pixel_size(row, header_pixel_size=header_px,
                                    override_pixel_size=args.pixel_size_angstrom)
        if classifier is None:
            import torch
            torch.set_num_threads(workers)
            classifier = PublicationClassifier(checkpoint=checkpoint, device=args.typing_device,
                                                batch_size=args.typing_batch_size)
        result = classifier.predict(image, mask, px)
        if signature(row) != signatures[index]:
            raise ValueError(f"Typing inputs changed while reading {key}; retry after inference finishes")
        path = masks_dir / f"{key}_typed_mask.npy"
        io._save_npy_atomic(path, result["pred_map"].astype(np.uint8))
        summary, components = _summaries(row, mask, result)
        record = {"signature": signatures[index], "mask_stamp": io._file_stamp(path),
                  "summary": summary, "components": components}
        records[index] = record
        io._write_json_atomic(cache_dir / f"{io._digest(key)}.json", record)
        if len(records) < len(rows) and time.monotonic() - last_publish >= interval:
            publish()
            last_publish = time.monotonic()
    publish(final=True)
    return 0
