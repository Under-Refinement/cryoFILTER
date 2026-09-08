"""Build annotation manifests for the local cryoFILTER app."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


MRC_SUFFIXES = {".mrc", ".mrcs"}
ANNOTATION_FIELDS = [
    "dataset_id",
    "stem",
    "micrograph_path",
    "pixel_size_angstrom",
    "gt_mask_path",
    "typed_label_path",
    "source_uid",
    "source_path",
    "split",
]


def _split_labels(keys: Sequence[str], *, train_fraction: float, seed: int) -> list[str]:
    if not keys:
        return []
    if len(keys) == 1:
        return ["TRAINING"]
    keyed = []
    for index, key in enumerate(keys):
        digest = hashlib.sha256(f"{seed}:{key}:{index}".encode("utf-8")).hexdigest()
        keyed.append((int(digest[:16], 16), index))
    keyed.sort()
    n_train = int(round(len(keys) * float(train_fraction)))
    n_train = min(max(1, n_train), len(keys) - 1)
    train_indices = {index for _, index in keyed[:n_train]}
    return ["TRAINING" if index in train_indices else "VALIDATION" for index in range(len(keys))]


def _optional_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def collect_micrographs(
    source: str | Path,
    *,
    recursive: bool = True,
    limit: int | None = None,
) -> list[Path]:
    root = Path(source).expanduser()
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        iterator = root.rglob("*") if recursive else root.glob("*")
        paths = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in MRC_SUFFIXES
        )
    else:
        raise FileNotFoundError(f"Micrograph source not found: {root}")
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise ValueError(f"No .mrc/.mrcs micrographs found in {root}")
    return [path.resolve() for path in paths]


def rows_from_micrographs(
    paths: Sequence[Path],
    *,
    dataset_id: str,
    pixel_size_angstrom: str | None = None,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    splits = _split_labels(
        [f"{dataset_id}:{path.resolve()}" for path in paths],
        train_fraction=train_fraction,
        seed=seed,
    )
    for path, split in zip(paths, splits):
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "stem": path.stem,
                "micrograph_path": str(path),
                "pixel_size_angstrom": _optional_text(pixel_size_angstrom),
                "gt_mask_path": "",
                "typed_label_path": "",
                "source_uid": "",
                "source_path": str(path),
                "split": split,
            }
        )
    return rows


def rows_from_transfer_manifest(
    transfer_manifest: str | Path,
    *,
    dataset_id: str,
    prefer_source_paths: bool = False,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> list[dict[str, str]]:
    manifest_path = Path(transfer_manifest).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Transfer manifest must contain a JSON object")
    micrographs = data.get("micrographs")
    if not isinstance(micrographs, list) or not micrographs:
        raise ValueError("Transfer manifest does not contain staged micrographs")

    transfer_dir = manifest_path.parent / "transfer"
    rows: list[dict[str, str]] = []
    for item in micrographs:
        if not isinstance(item, dict):
            continue
        filename = _optional_text(item.get("transfer_filename"))
        source_path = _optional_text(item.get("source_path"))
        if prefer_source_paths and source_path:
            path = Path(source_path).expanduser()
        elif filename:
            relative_transfer = Path(filename)
            if len(relative_transfer.parts) == 1:
                relative_transfer = Path("micrographs") / relative_transfer
            path = (transfer_dir / relative_transfer).resolve()
        else:
            continue
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "stem": path.stem,
                "micrograph_path": str(path),
                "pixel_size_angstrom": _optional_text(item.get("pixel_size_angstrom")),
                "gt_mask_path": "",
                "typed_label_path": "",
                "source_uid": _optional_text(item.get("uid")),
                "source_path": source_path,
                "split": "",
            }
        )
    if not rows:
        raise ValueError("Transfer manifest did not yield any micrograph rows")
    splits = _split_labels(
        [f"{row['source_uid']}:{row['micrograph_path']}" for row in rows],
        train_fraction=train_fraction,
        seed=seed,
    )
    for row, split in zip(rows, splits):
        row["split"] = split
    return rows


def write_manifest(
    rows: Sequence[dict[str, str]],
    output_manifest: str | Path,
    *,
    reuse_existing: bool = False,
) -> Path:
    output_path = Path(output_manifest).expanduser()
    if output_path.exists() and reuse_existing:
        print(f"Using existing annotation manifest: {output_path}", flush=True)
        return output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing annotation manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote annotation manifest: {output_path} ({len(rows)} micrographs)", flush=True)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a cryoFILTER annotation manifest.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--micrographs", help="Input .mrc/.mrcs file or directory.")
    source.add_argument("--transfer-manifest", help="CryoSPARC transfer_manifest.json.")
    parser.add_argument("--output-manifest", required=True, help="Manifest CSV to write.")
    parser.add_argument("--dataset-id", default="1", help="dataset_id written to every row.")
    parser.add_argument("--pixel-size-angstrom", default=None, help="Optional A/px value.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum micrographs.")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Fraction of rows labeled TRAINING.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic row split seed.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories.")
    parser.add_argument(
        "--prefer-source-paths",
        action="store_true",
        help="For CryoSPARC manifests, write original source paths instead of staged transfer paths.",
    )
    parser.add_argument("--reuse-existing", action="store_true", help="Use an existing manifest instead of replacing it.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.micrographs:
        rows = rows_from_micrographs(
            collect_micrographs(
                args.micrographs,
                recursive=not args.no_recursive,
                limit=args.limit,
            ),
            dataset_id=str(args.dataset_id),
            pixel_size_angstrom=args.pixel_size_angstrom,
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )
    else:
        rows = rows_from_transfer_manifest(
            args.transfer_manifest,
            dataset_id=str(args.dataset_id),
            prefer_source_paths=bool(args.prefer_source_paths),
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )
        if args.limit is not None:
            rows = rows[: int(args.limit)]
    write_manifest(rows, args.output_manifest, reuse_existing=bool(args.reuse_existing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
