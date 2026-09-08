from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

from cryofilter.cli import _build_parser
from cryofilter.training import (
    FULL_INPUT_CHANNELS,
    FULL_TARGET_PIXEL_SIZE_ANGSTROM,
    FullPatchDataset,
    load_full_frequency_bands,
    load_training_manifest,
)
import scripts.train_patch_based_fast as fast_train
from scripts.train_patch_based_fast import _normalize_row_split_label


def _write_mrc(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(data, dtype=np.float32))
        handle.voxel_size = FULL_TARGET_PIXEL_SIZE_ANGSTROM


def _training_manifest(tmp_path: Path) -> Path:
    rng = np.random.default_rng(12)
    rows = []
    for index, split in enumerate(("train", "val")):
        image_path = tmp_path / f"micrograph_{index}.mrc"
        mask_path = tmp_path / f"mask_{index}.npy"
        image = rng.normal(size=(48, 48)).astype(np.float32)
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[16:32, 16:32] = 1
        _write_mrc(image_path, image)
        np.save(mask_path, mask)
        rows.append(
            f"{image_path.name},{mask_path.name},{FULL_TARGET_PIXEL_SIZE_ANGSTROM},{split}"
        )
    manifest = tmp_path / "training.csv"
    manifest.write_text(
        "micrograph_path,mask_path,pixel_size_angstrom,split\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_training_parser_supports_new_and_finetune_runs() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train",
            "--manifest",
            "training.csv",
            "--output-dir",
            "output",
            "--resume-checkpoint",
            "cryoFILTER_FULL.pt",
            "--dry-run",
        ]
    )

    assert args.command == "train"
    assert args.initial_checkpoint == "cryoFILTER_FULL.pt"
    assert args.dry_run is True


def test_fast_training_model_factory_ignores_unsupported_optional_kwargs() -> None:
    original = fast_train.create_model
    fast_train._MODEL_FACTORY_IGNORED_KWARGS.clear()

    def fake_create_model(model_type: str, device: str, num_classes: int = 1) -> dict[str, object]:
        return {"model_type": model_type, "device": device, "num_classes": num_classes}

    fast_train.create_model = fake_create_model
    try:
        model = fast_train._create_model_compat(
            model_type="unet_attention",
            device="cpu",
            num_classes=1,
            use_two_branch_encoder=False,
            miffi_weights_path="skip",
        )
    finally:
        fast_train.create_model = original
        fast_train._MODEL_FACTORY_IGNORED_KWARGS.clear()

    assert model == {"model_type": "unet_attention", "device": "cpu", "num_classes": 1}


def test_full_patch_dataset_builds_five_channel_input(tmp_path: Path) -> None:
    train_records, val_records = load_training_manifest(_training_manifest(tmp_path))
    assert len(train_records) == 1
    assert len(val_records) == 1
    assert len(load_full_frequency_bands()) == 4

    dataset = FullPatchDataset(
        train_records,
        patch_size=32,
        patches_per_micrograph=1,
        positive_patch_fraction=1.0,
        augment=False,
        seed=4,
    )
    inputs, target = dataset[0]

    assert tuple(inputs.shape) == (FULL_INPUT_CHANNELS, 32, 32)
    assert tuple(target.shape) == (32, 32)
    assert np.isfinite(inputs.numpy()).all()
    assert set(np.unique(target.numpy())) <= {0, 1}


def test_full_micrograph_training_normalizes_row_split_labels() -> None:
    assert _normalize_row_split_label("train") == "TRAINING"
    assert _normalize_row_split_label("val") == "VALIDATION"
    assert _normalize_row_split_label("VALID") == "VALIDATION"
