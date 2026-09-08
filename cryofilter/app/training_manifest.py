"""Prepare annotation exports for the cryoFILTER fine-tuning recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PATH_COLUMNS = (
    "micrograph_path",
    "full_mic_path",
    "resolved_mic_path",
    "staged_mic_path",
    "mic_path",
    "source_path",
)
MASK_COLUMNS = (
    "gt_mask_path",
    "mask_path",
    "binary_mask_path",
    "gt_mask",
    "gt_mask_full_path",
)
EXTRA_FIELDS = ("micrograph_path", "gt_mask_path", "split", "annotation_status")


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _resolve_manifest_path(value: object, *, manifest_dir: Path, must_exist: bool = False) -> Path | None:
    text = _optional_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path] if path.is_absolute() else [manifest_dir / path, Path.cwd() / path, path]
    for candidate in candidates:
        if not must_exist or candidate.exists():
            return candidate.resolve()
    return None


def _split_labels(keys: Sequence[str], *, train_fraction: float, seed: int) -> list[str]:
    if not keys:
        return []
    if len(keys) == 1:
        raise ValueError("Automatic 70/30 splitting needs at least two saved annotation rows.")
    keyed = []
    for index, key in enumerate(keys):
        digest = hashlib.sha256(f"{seed}:{key}:{index}".encode("utf-8")).hexdigest()
        keyed.append((int(digest[:16], 16), index))
    keyed.sort()
    n_train = int(round(len(keys) * float(train_fraction)))
    n_train = min(max(1, n_train), len(keys) - 1)
    train_indices = {index for _, index in keyed[:n_train]}
    return ["TRAINING" if index in train_indices else "VALIDATION" for index in range(len(keys))]


def _transfer_path(
    item: dict[str, Any],
    *,
    transfer_dir: Path,
    prefer_source_paths: bool,
) -> Path | None:
    source_path = _optional_text(item.get("source_path"))
    filename = _optional_text(item.get("transfer_filename"))
    if prefer_source_paths and source_path:
        return Path(source_path).expanduser()
    if filename:
        relative = Path(filename)
        if len(relative.parts) == 1:
            relative = Path("micrographs") / relative
        return (transfer_dir / relative).resolve()
    if source_path:
        return Path(source_path).expanduser()
    return None


def _transfer_index(
    transfer_manifest: str | Path,
    *,
    prefer_source_paths: bool,
) -> dict[tuple[str, str], dict[str, str]]:
    manifest_path = Path(transfer_manifest).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    micrographs = data.get("micrographs")
    if not isinstance(micrographs, list):
        raise ValueError("Transfer manifest does not contain a micrographs list.")
    transfer_dir = manifest_path.parent / "transfer"
    index: dict[tuple[str, str], dict[str, str]] = {}
    for item in micrographs:
        if not isinstance(item, dict):
            continue
        path = _transfer_path(item, transfer_dir=transfer_dir, prefer_source_paths=prefer_source_paths)
        if path is None:
            continue
        source_path = _optional_text(item.get("source_path"))
        source_uid = _optional_text(item.get("uid"))
        transfer_filename = _optional_text(item.get("transfer_filename"))
        record = {
            "micrograph_path": str(path),
            "source_path": source_path,
            "source_uid": source_uid,
            "pixel_size_angstrom": _optional_text(item.get("pixel_size_angstrom")),
        }
        keys = {
            ("stem", path.stem),
            ("basename", path.name),
        }
        if source_uid:
            keys.add(("source_uid", source_uid))
        if source_path:
            source = Path(source_path)
            keys.update({("source_path", source_path), ("source_stem", source.stem), ("source_basename", source.name)})
        if transfer_filename:
            transfer_name = Path(transfer_filename).name
            keys.update({("transfer_filename", transfer_filename), ("transfer_basename", transfer_name)})
        for key in keys:
            index.setdefault(key, record)
    return index


def _matching_transfer_record(
    row: dict[str, str],
    index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str] | None:
    candidates = []
    for column in ("source_uid", "uid"):
        value = _optional_text(row.get(column))
        if value:
            candidates.append(("source_uid", value))
    for column in PATH_COLUMNS:
        value = _optional_text(row.get(column))
        if value:
            path = Path(value)
            candidates.extend(
                [
                    (column, value),
                    ("source_path", value),
                    ("stem", path.stem),
                    ("source_stem", path.stem),
                    ("basename", path.name),
                    ("source_basename", path.name),
                ]
            )
    stem = _optional_text(row.get("stem"))
    if stem:
        candidates.extend([("stem", stem), ("source_stem", stem)])
    for key in candidates:
        match = index.get(key)
        if match is not None:
            return match
    return None


def _saved_mask_path(row: dict[str, str], *, manifest_dir: Path) -> Path | None:
    for column in MASK_COLUMNS:
        path = _resolve_manifest_path(row.get(column), manifest_dir=manifest_dir, must_exist=True)
        if path is not None and path.is_file():
            return path
    return None


def _micrograph_path(row: dict[str, str], *, manifest_dir: Path) -> Path | None:
    for column in PATH_COLUMNS:
        path = _resolve_manifest_path(row.get(column), manifest_dir=manifest_dir, must_exist=False)
        if path is not None:
            return path
    return None


def prepare_training_manifest(
    *,
    manifest: str | Path,
    output_manifest: str | Path,
    transfer_manifest: str | Path | None = None,
    prefer_source_paths: bool = True,
    train_fraction: float = 0.7,
    seed: int = 42,
    assign_split: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_dir = manifest_path.parent
    transfer_rows = (
        _transfer_index(transfer_manifest, prefer_source_paths=prefer_source_paths)
        if transfer_manifest is not None
        else {}
    )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is missing a header row: {manifest_path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]

    prepared: list[dict[str, str]] = []
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        transfer = _matching_transfer_record(row, transfer_rows) if transfer_rows else None
        if transfer is not None:
            row["micrograph_path"] = transfer["micrograph_path"]
            row["source_path"] = transfer["source_path"] or row.get("source_path", "")
            row["source_uid"] = transfer["source_uid"] or row.get("source_uid", "")
            if transfer["pixel_size_angstrom"] and not _optional_text(row.get("pixel_size_angstrom")):
                row["pixel_size_angstrom"] = transfer["pixel_size_angstrom"]

        mask_path = _saved_mask_path(row, manifest_dir=manifest_dir)
        if mask_path is None:
            continue
        micrograph_path = _micrograph_path(row, manifest_dir=manifest_dir)
        if micrograph_path is None:
            raise ValueError(
                f"Saved annotation row {index + 1} has a mask but no micrograph path. "
                "Choose Local directory or CryoSPARC output in the Fine-tune Micrograph source."
            )
        row["gt_mask_path"] = str(mask_path)
        row["micrograph_path"] = str(micrograph_path)
        row["annotation_status"] = "saved"
        prepared.append(row)

    if not prepared:
        raise ValueError(
            "No saved annotation rows with gt_mask_path were found. "
            "Save at least two annotated micrographs before fine-tuning."
        )

    if assign_split:
        splits = _split_labels(
            [
                f"{row.get('dataset_id', '')}:{row.get('stem', '')}:{row.get('micrograph_path', '')}"
                for row in prepared
            ],
            train_fraction=float(train_fraction),
            seed=int(seed),
        )
        for row, split in zip(prepared, splits):
            row["split"] = split

    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    output_path = Path(output_manifest).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prepared)

    split_counts = {"TRAINING": 0, "VALIDATION": 0}
    for row in prepared:
        label = _optional_text(row.get("split")).upper()
        if label in {"TRAIN", "TRAINING"}:
            split_counts["TRAINING"] += 1
        elif label in {"VAL", "VALID", "VALIDATION"}:
            split_counts["VALIDATION"] += 1
    return {
        "output_manifest": str(output_path),
        "source_manifest": str(manifest_path),
        "rows": len(prepared),
        "split_counts": {key: value for key, value in split_counts.items() if value},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a fine-tuning manifest from annotation exports.")
    parser.add_argument("--manifest", required=True, help="Annotation manifest_edited.csv.")
    parser.add_argument("--output-manifest", required=True, help="Training-ready manifest to write.")
    parser.add_argument("--transfer-manifest", help="Optional CryoSPARC transfer_manifest.json for resolving source paths.")
    parser.add_argument(
        "--prefer-source-paths",
        action="store_true",
        default=False,
        help="Use original CryoSPARC source paths from the transfer manifest.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Fraction of saved rows labeled TRAINING.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic row split seed.")
    parser.add_argument(
        "--preserve-split",
        action="store_true",
        help="Keep existing split labels instead of assigning a fresh row-level split.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = prepare_training_manifest(
        manifest=args.manifest,
        output_manifest=args.output_manifest,
        transfer_manifest=args.transfer_manifest,
        prefer_source_paths=bool(args.prefer_source_paths),
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
        assign_split=not bool(args.preserve_split),
    )
    split = summary["split_counts"]
    split_text = ", ".join(f"{key}={value}" for key, value in split.items()) or "preserved"
    print(
        f"Wrote training manifest: {summary['output_manifest']} "
        f"({summary['rows']} saved rows; {split_text})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
