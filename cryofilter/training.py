"""Lightweight public training workflow for the cryoFILTER FULL model.

For the current recommended full-micrograph fine-tuning recipe, use
scripts/train_patch_based_fast.py as documented in docs/fine_tuning.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


FULL_MODEL_TYPE = "unet_attention"
FULL_ATTENTION_TYPE = "global"
FULL_NORM_TYPE = "group"
FULL_DECODER_DROPOUT = 0.1
FULL_NUM_CLASSES = 1
FULL_TARGET_PIXEL_SIZE_ANGSTROM = 2.0
FULL_NORMALIZATION_METHOD = "percentile_extra_wide"
FULL_INPUT_CHANNELS = 5


@dataclass(frozen=True)
class TrainingRecord:
    """One paired micrograph and binary contamination mask."""

    micrograph_path: Path
    mask_path: Path
    pixel_size_angstrom: float
    split: str


def _frequency_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "selected_frequency_bands.json"


def load_full_frequency_bands() -> tuple[tuple[float, float], ...]:
    with _frequency_config_path().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    bands = tuple(
        (float(low), float(high))
        for low, high in payload["selected_frequency_bands_angstrom_inv"]
    )
    if len(bands) != FULL_INPUT_CHANNELS - 1:
        raise ValueError(
            f"FULL requires four frequency bands; configuration contains {len(bands)}"
        )
    return bands


def _resolve_manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _normalize_split(value: str) -> str:
    split = str(value or "").strip().lower()
    aliases = {"training": "train", "validation": "val", "valid": "val"}
    return aliases.get(split, split)


def load_training_manifest(
    manifest_path: str | Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Load and validate paired inputs, creating a deterministic split if needed."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest}")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")

    records: list[TrainingRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        mask_column = "mask_path" if "mask_path" in fieldnames else "binary_mask_path"
        required = {"micrograph_path", "pixel_size_angstrom", mask_column}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Training manifest is missing required column(s): {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            micrograph = _resolve_manifest_path(row["micrograph_path"], manifest.parent)
            mask = _resolve_manifest_path(row[mask_column], manifest.parent)
            if not micrograph.is_file():
                raise FileNotFoundError(
                    f"Manifest line {line_number}: micrograph not found: {micrograph}"
                )
            if not mask.is_file():
                raise FileNotFoundError(
                    f"Manifest line {line_number}: mask not found: {mask}"
                )
            try:
                pixel_size = float(row["pixel_size_angstrom"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest line {line_number}: invalid pixel_size_angstrom"
                ) from exc
            if not np.isfinite(pixel_size) or pixel_size <= 0:
                raise ValueError(
                    f"Manifest line {line_number}: pixel_size_angstrom must be positive"
                )
            split = _normalize_split(row.get("split", ""))
            if split not in {"", "train", "val"}:
                raise ValueError(
                    f"Manifest line {line_number}: split must be train or val, got {split!r}"
                )
            records.append(
                TrainingRecord(
                    micrograph_path=micrograph,
                    mask_path=mask,
                    pixel_size_angstrom=pixel_size,
                    split=split,
                )
            )

    if len(records) < 2:
        raise ValueError("Training requires at least two paired micrographs/masks")

    explicit = [record for record in records if record.split]
    if explicit:
        if len(explicit) != len(records):
            raise ValueError("Either set split for every manifest row or omit it from every row")
        train_records = [record for record in records if record.split == "train"]
        val_records = [record for record in records if record.split == "val"]
    else:
        shuffled = list(records)
        random.Random(int(seed)).shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(validation_fraction))))
        val_count = min(val_count, len(shuffled) - 1)
        val_records = shuffled[:val_count]
        train_records = shuffled[val_count:]

    if not train_records or not val_records:
        raise ValueError("Training manifest must produce at least one train and one val row")
    return train_records, val_records


def _load_2d_array(path: Path, *, is_mask: bool) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix in {".mrc", ".mrcs"}:
        from utils.file_io import load_mrc

        array = load_mrc(str(path))
    elif is_mask and suffix in {".tif", ".tiff"}:
        from skimage.io import imread

        array = imread(path)
    else:
        expected = ".npy, .mrc, or .mrcs"
        if is_mask:
            expected += ", with .tif/.tiff also accepted for masks"
        raise ValueError(f"Unsupported file {path}; expected {expected}")

    array = np.asarray(array).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array at {path}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values found in {path}")
    return np.asarray(array, dtype=np.float32)


def _prepare_record(record: TrainingRecord) -> tuple[np.ndarray, np.ndarray]:
    from skimage.transform import resize
    from utils.fourier_rescale import fourier_rescale_2d
    from utils.image_utils import normalize_image

    image = _load_2d_array(record.micrograph_path, is_mask=False)
    mask = _load_2d_array(record.mask_path, is_mask=True)
    if image.shape != mask.shape:
        raise ValueError(
            f"Micrograph/mask shape mismatch for {record.micrograph_path.name}: "
            f"{image.shape} versus {mask.shape}"
        )

    image = normalize_image(image, method=FULL_NORMALIZATION_METHOD)
    scale = float(record.pixel_size_angstrom) / FULL_TARGET_PIXEL_SIZE_ANGSTROM
    if abs(scale - 1.0) > 1e-6:
        image = fourier_rescale_2d(image, scale=scale)
        mask = resize(
            mask,
            image.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )

    binary_mask = (np.asarray(mask) > 0).astype(np.int64)
    return (
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(binary_mask, dtype=np.int64),
    )


class FullPatchDataset:
    """Random, contamination-aware patch sampling with a small per-worker cache."""

    def __init__(
        self,
        records: Sequence[TrainingRecord],
        *,
        patch_size: int,
        patches_per_micrograph: int,
        positive_patch_fraction: float,
        augment: bool,
        seed: int,
        cache_size: int = 4,
    ) -> None:
        if patch_size < 32 or patch_size % 16:
            raise ValueError("--patch-size must be at least 32 and divisible by 16")
        if patches_per_micrograph < 1:
            raise ValueError("patches per micrograph must be positive")
        if not 0.0 <= positive_patch_fraction <= 1.0:
            raise ValueError("--positive-patch-fraction must be between 0 and 1")
        self.records = list(records)
        self.patch_size = int(patch_size)
        self.patches_per_micrograph = int(patches_per_micrograph)
        self.positive_patch_fraction = float(positive_patch_fraction)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.frequency_bands = load_full_frequency_bands()

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_micrograph

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _prepared(self, record_index: int) -> tuple[np.ndarray, np.ndarray]:
        if record_index in self._cache:
            value = self._cache.pop(record_index)
            self._cache[record_index] = value
            return value
        value = _prepare_record(self.records[record_index])
        self._cache[record_index] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _pad(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pad_h = max(0, self.patch_size - image.shape[0])
        pad_w = max(0, self.patch_size - image.shape[1])
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        return image, mask

    def __getitem__(self, index: int):
        import torch
        from utils.image_utils import compute_frequency_band_psd_channels

        record_index = int(index) % len(self.records)
        image, mask = self._prepared(record_index)
        image, mask = self._pad(image, mask)
        rng = np.random.default_rng(
            self.seed + self.epoch * max(1, len(self)) + int(index)
        )
        height, width = image.shape

        use_positive = bool(mask.any()) and rng.random() < self.positive_patch_fraction
        if use_positive:
            positive_y, positive_x = np.nonzero(mask)
            chosen = int(rng.integers(0, len(positive_y)))
            center_y = int(positive_y[chosen])
            center_x = int(positive_x[chosen])
            jitter = self.patch_size // 4
            center_y += int(rng.integers(-jitter, jitter + 1))
            center_x += int(rng.integers(-jitter, jitter + 1))
            y0 = center_y - self.patch_size // 2
            x0 = center_x - self.patch_size // 2
        else:
            y0 = int(rng.integers(0, max(1, height - self.patch_size + 1)))
            x0 = int(rng.integers(0, max(1, width - self.patch_size + 1)))
        y0 = int(np.clip(y0, 0, height - self.patch_size))
        x0 = int(np.clip(x0, 0, width - self.patch_size))

        patch = image[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        target = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.augment:
            turns = int(rng.integers(0, 4))
            patch = np.rot90(patch, turns)
            target = np.rot90(target, turns)
            if rng.random() < 0.5:
                patch = np.fliplr(patch)
                target = np.fliplr(target)

        patch = np.ascontiguousarray(patch, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.int64)
        psd = compute_frequency_band_psd_channels(
            patch,
            pixel_size_angstrom=FULL_TARGET_PIXEL_SIZE_ANGSTROM,
            frequency_bands=self.frequency_bands,
            normalize=True,
            use_radial_normalization=True,
        )
        inputs = np.concatenate((patch[None, ...], psd), axis=0)
        if inputs.shape != (FULL_INPUT_CHANNELS, self.patch_size, self.patch_size):
            raise RuntimeError(f"Unexpected FULL input shape: {inputs.shape}")
        return torch.from_numpy(inputs), torch.from_numpy(target)


def _resolve_device(value: str) -> str:
    import torch

    requested = str(value).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return str(value)


def _extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a model state dictionary")


def _strip_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def _validate_full_checkpoint(checkpoint: object) -> None:
    if not isinstance(checkpoint, dict):
        return
    expected = {
        "model_type": FULL_MODEL_TYPE,
        "attention_type": FULL_ATTENTION_TYPE,
        "input_channels": FULL_INPUT_CHANNELS,
        "num_classes": FULL_NUM_CLASSES,
    }
    for key, value in expected.items():
        if key in checkpoint and checkpoint[key] != value:
            raise ValueError(
                f"Initial checkpoint is not FULL-compatible: {key}={checkpoint[key]!r}, "
                f"expected {value!r}"
            )
    if "psd_frequency_bands" in checkpoint:
        observed = tuple(
            tuple(float(x) for x in band)
            for band in checkpoint["psd_frequency_bands"]
        )
        if observed != load_full_frequency_bands():
            raise ValueError("Initial checkpoint uses different PSD frequency bands")


def build_full_model(device: str, initial_checkpoint: Optional[Path] = None):
    import torch
    from models.bad_region_detector import create_model

    model = create_model(
        model_type=FULL_MODEL_TYPE,
        device=device,
        use_power_spectrum=True,
        num_classes=FULL_NUM_CLASSES,
        pretrained=False,
        norm_type=FULL_NORM_TYPE,
        decoder_dropout=FULL_DECODER_DROPOUT,
        attention_type=FULL_ATTENTION_TYPE,
        input_channels_override=FULL_INPUT_CHANNELS,
    )
    if initial_checkpoint is not None:
        try:
            checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(initial_checkpoint, map_location="cpu")
        _validate_full_checkpoint(checkpoint)
        state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
        model.load_state_dict(state_dict, strict=True)
        print(f"Initialized FULL weights from {initial_checkpoint}", flush=True)
    return model


def _loss_and_metrics(logits, targets, class_weights, dice_weight: float):
    import torch
    import torch.nn.functional as functional

    target_bad = (targets == 1).float()
    if logits.shape[1] == 1:
        cross_entropy = functional.binary_cross_entropy_with_logits(
            logits[:, 0], target_bad, pos_weight=class_weights[1]
        )
        probabilities = torch.sigmoid(logits[:, 0])
    elif logits.shape[1] == 2:
        cross_entropy = functional.cross_entropy(logits, targets, weight=class_weights)
        probabilities = functional.softmax(logits, dim=1)[:, 1]
    else:
        raise ValueError(f"Expected one or two FULL output channels, got {logits.shape[1]}")
    intersection = (probabilities * target_bad).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + target_bad.sum(dim=(1, 2))
    dice = ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    loss = (1.0 - float(dice_weight)) * cross_entropy + float(dice_weight) * (1.0 - dice)
    predicted = probabilities >= 0.5
    target_bool = target_bad > 0.5
    union = (predicted | target_bool).sum(dim=(1, 2)).float()
    iou = ((predicted & target_bool).sum(dim=(1, 2)).float() + 1.0) / (union + 1.0)
    return loss, float(dice.detach().cpu()), float(iou.mean().detach().cpu())


def _run_epoch(
    model,
    loader,
    *,
    device: str,
    class_weights,
    dice_weight: float,
    optimizer=None,
    scaler=None,
) -> dict[str, float]:
    import torch

    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "samples": 0}
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = int(inputs.shape[0])
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = str(device).startswith("cuda")
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss, dice, iou = _loss_and_metrics(
                    logits, targets, class_weights, dice_weight
                )
            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["dice"] += dice * batch_size
            totals["iou"] += iou * batch_size
            totals["samples"] += batch_size
    samples = max(1, int(totals.pop("samples")))
    return {key: float(value) / samples for key, value in totals.items()}


def _checkpoint_payload(model, optimizer, *, epoch: int, metrics: dict, args) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "model_type": FULL_MODEL_TYPE,
        "attention_type": FULL_ATTENTION_TYPE,
        "norm_type": FULL_NORM_TYPE,
        "decoder_dropout": FULL_DECODER_DROPOUT,
        "num_classes": FULL_NUM_CLASSES,
        "use_power_spectrum": True,
        "include_real_space_input": True,
        "use_pixel_size_channel": False,
        "pixel_size_channel_min_angstrom": FULL_TARGET_PIXEL_SIZE_ANGSTROM,
        "pixel_size_channel_max_angstrom": FULL_TARGET_PIXEL_SIZE_ANGSTROM,
        "psd_multiscale": False,
        "psd_multiscale_separate_channels": False,
        "psd_scales": [16, 32, 64],
        "psd_frequency_band_channels": True,
        "psd_frequency_bands": [list(band) for band in load_full_frequency_bands()],
        "psd_frequency_band_include_full_spectrum": False,
        "psd_frequency_band_include_anisotropy": False,
        "psd_frequency_band_anisotropy_min_freq": 0.0,
        "input_channels": FULL_INPUT_CHANNELS,
        "target_pixel_size_angstrom": FULL_TARGET_PIXEL_SIZE_ANGSTROM,
        "normalization_method": FULL_NORMALIZATION_METHOD,
        "validation_metrics": dict(metrics),
        "training_config": {
            "patch_size": int(args.patch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "positive_weight": float(args.positive_weight),
            "dice_weight": float(args.dice_weight),
            "seed": int(args.seed),
        },
    }


def _write_history(path: Path, history: Sequence[dict]) -> None:
    columns = (
        "epoch",
        "learning_rate",
        "train_loss",
        "train_dice",
        "train_iou",
        "val_loss",
        "val_dice",
        "val_iou",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def add_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "train",
        help="Run lightweight patch-based FULL training from paired micrographs and masks.",
    )
    parser.add_argument("--manifest", required=True, help="CSV of paired training inputs.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--initial-checkpoint",
        "--resume-checkpoint",
        dest="initial_checkpoint",
        default=None,
        help="Optional FULL checkpoint whose weights initialize fine-tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for this lightweight command. The full fine-tuning recipe uses 120 epochs via scripts/train_patch_based_fast.py.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-micrograph", type=int, default=16)
    parser.add_argument("--validation-patches-per-micrograph", type=int, default=8)
    parser.add_argument("--positive-patch-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and one FULL input patch without training.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.positive_weight <= 0:
        raise ValueError("--positive-weight must be positive")
    if not 0.0 <= args.dice_weight <= 1.0:
        raise ValueError("--dice-weight must be between 0 and 1")

    initial_checkpoint = (
        Path(args.initial_checkpoint).expanduser().resolve()
        if args.initial_checkpoint
        else None
    )
    if initial_checkpoint is not None and not initial_checkpoint.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {initial_checkpoint}")
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if initial_checkpoint is not None else 2e-5

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    train_records, val_records = load_training_manifest(
        args.manifest,
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    train_dataset = FullPatchDataset(
        train_records,
        patch_size=int(args.patch_size),
        patches_per_micrograph=int(args.patches_per_micrograph),
        positive_patch_fraction=float(args.positive_patch_fraction),
        augment=True,
        seed=int(args.seed),
    )
    val_dataset = FullPatchDataset(
        val_records,
        patch_size=int(args.patch_size),
        patches_per_micrograph=int(args.validation_patches_per_micrograph),
        positive_patch_fraction=0.0,
        augment=False,
        seed=int(args.seed) + 10_000,
    )

    first_inputs, first_target = train_dataset[0]
    plan = {
        "model": "FULL",
        "model_type": FULL_MODEL_TYPE,
        "attention_type": FULL_ATTENTION_TYPE,
        "input_channels": FULL_INPUT_CHANNELS,
        "frequency_bands_angstrom_inv": [list(b) for b in load_full_frequency_bands()],
        "target_pixel_size_angstrom": FULL_TARGET_PIXEL_SIZE_ANGSTROM,
        "normalization_method": FULL_NORMALIZATION_METHOD,
        "train_micrographs": len(train_records),
        "validation_micrographs": len(val_records),
        "sample_input_shape": list(first_inputs.shape),
        "sample_mask_shape": list(first_target.shape),
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; inputs are valid.", flush=True)
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    device = _resolve_device(args.device)
    model = build_full_model(device, initial_checkpoint=initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))
    class_weights = torch.tensor(
        [1.0, float(args.positive_weight)], dtype=torch.float32, device=device
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": str(device).startswith("cuda"),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    for epoch_index in range(int(args.epochs)):
        epoch = epoch_index + 1
        train_dataset.set_epoch(epoch_index)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            class_weights=class_weights,
            dice_weight=float(args.dice_weight),
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            class_weights=class_weights,
            dice_weight=float(args.dice_weight),
        )
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        _write_history(output_dir / "history.csv", history)
        payload = _checkpoint_payload(
            model, optimizer, epoch=epoch, metrics=val_metrics, args=args
        )
        torch.save(payload, last_path)
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(payload, best_path)
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
            + (" [best]" if improved else ""),
            flush=True,
        )
        if epochs_without_improvement >= int(args.early_stop_patience):
            print(f"Early stopping after epoch {epoch}.", flush=True)
            break

    summary = {
        **plan,
        "device": device,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Best FULL checkpoint: {best_path}", flush=True)
    return 0


__all__ = [
    "FULL_ATTENTION_TYPE",
    "FULL_DECODER_DROPOUT",
    "FULL_INPUT_CHANNELS",
    "FULL_MODEL_TYPE",
    "FULL_NORMALIZATION_METHOD",
    "FULL_NUM_CLASSES",
    "FULL_TARGET_PIXEL_SIZE_ANGSTROM",
    "FullPatchDataset",
    "TrainingRecord",
    "add_subparser",
    "build_full_model",
    "load_full_frequency_bands",
    "load_training_manifest",
    "run",
]
