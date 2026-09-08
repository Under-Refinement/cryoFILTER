import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetRegionRule:
    dataset_min_area_px: Dict[str, float]
    fill_holes: bool = True
    closing_radius_px: int = 0

    def min_area(self, dataset_id: str) -> float:
        value = self.dataset_min_area_px.get(str(dataset_id), math.inf)
        try:
            value = float(value)
        except Exception:
            value = math.inf
        return value


def _to_2d_binary(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"GT mask must be 2D after squeeze; got shape={getattr(arr, 'shape', None)}")
    return (arr > 0.5).astype(np.uint8)


def compute_dataset_min_area_px_from_train_split(
    train_rows: Iterable[Tuple[str, str]],
    min_area_floor_px: float = 0.0,
    fill_holes: bool = True,
    closing_radius_px: int = 0,
) -> DatasetRegionRule:
    """
    Compute dataset-specific region-GT area thresholds in pixels.

    For each dataset, we keep the smallest connected-component area observed in
    its training masks, floored by `min_area_floor_px`. Datasets with no
    positive components get `+inf`, which zeros them out under the transform.
    """
    from scipy.ndimage import label

    rows = list(train_rows)
    areas_by_dataset: Dict[str, list[int]] = {}
    structure = np.ones((3, 3), dtype=np.int8)
    for dataset_id, gt_path in rows:
        path = Path(gt_path)
        if not path.exists():
            continue
        mask = _to_2d_binary(np.load(str(path)))
        labeled, n_labels = label(mask.astype(bool), structure=structure)
        if n_labels <= 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]
        if sizes.size:
            areas_by_dataset.setdefault(str(dataset_id), []).extend(
                int(value) for value in sizes.tolist() if int(value) > 0
            )

    floor_value = float(min_area_floor_px) if min_area_floor_px is not None else 0.0
    if not np.isfinite(floor_value) or floor_value < 0:
        floor_value = 0.0

    out: Dict[str, float] = {}
    for dataset_id, areas in areas_by_dataset.items():
        if len(areas) == 0:
            out[str(dataset_id)] = math.inf
        else:
            out[str(dataset_id)] = float(max(float(min(areas)), floor_value))

    for dataset_id, _ in rows:
        dataset_key = str(dataset_id)
        if dataset_key not in out:
            out[dataset_key] = math.inf

    return DatasetRegionRule(
        dataset_min_area_px=out,
        fill_holes=bool(fill_holes),
        closing_radius_px=int(closing_radius_px),
    )


def save_dataset_region_rule(rule: DatasetRegionRule, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_min_area_px.json"
    csv_path = out_dir / "dataset_min_area_px.csv"
    config_path = out_dir / "region_gt_config.json"
    with open(json_path, "w") as handle:
        json.dump(rule.dataset_min_area_px, handle, indent=2, sort_keys=True)
    with open(config_path, "w") as handle:
        json.dump(
            {
                "fill_holes": bool(getattr(rule, "fill_holes", True)),
                "closing_radius_px": int(getattr(rule, "closing_radius_px", 0)),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    try:
        import pandas as pd

        pd.DataFrame(
            [
                {"dataset_id": dataset_id, "min_area_px": min_area_px}
                for dataset_id, min_area_px in sorted(rule.dataset_min_area_px.items(), key=lambda kv: kv[0])
            ]
        ).to_csv(csv_path, index=False)
    except Exception:
        pass


def apply_region_gt_transform(
    gt_mask: np.ndarray,
    dataset_id: str,
    rule: DatasetRegionRule,
    fill_holes: Optional[bool] = None,
    closing_radius_px: Optional[int] = None,
) -> np.ndarray:
    """
    Convert raw GT into the deterministic region-GT target used for training.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, label

    mask = _to_2d_binary(gt_mask)

    min_area = rule.min_area(dataset_id)
    if not np.isfinite(min_area) or min_area <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    structure = np.ones((3, 3), dtype=np.int8)
    labeled, n_labels = label(mask.astype(bool), structure=structure)
    if n_labels <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= int(math.ceil(float(min_area)))
    keep[0] = False
    out = keep[labeled].astype(bool)

    if fill_holes is None:
        fill_holes = bool(getattr(rule, "fill_holes", True))
    if closing_radius_px is None:
        closing_radius_px = int(getattr(rule, "closing_radius_px", 0))

    if bool(fill_holes) and out.any():
        out = binary_fill_holes(out)

    radius = int(closing_radius_px) if closing_radius_px is not None else 0
    if radius > 0 and out.any():
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure = (xx * xx + yy * yy) <= (radius * radius)
        out = binary_closing(out, structure=structure.astype(bool))

    return out.astype(np.uint8)
