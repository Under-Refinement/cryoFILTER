# Publication classifier UI integration.
"""Regression coverage for cached features and dataset-relative typing."""

import json
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest

from cryofilter import typing_cli


def _inputs(root: Path, count: int = 3) -> tuple[Path, pd.DataFrame]:
    rows = []
    for index in range(count):
        mic = root / f"mic_{index}.mrc"
        mask_path = root / f"mic_{index}_mask.npy"
        image = np.random.default_rng(index).normal(size=(128, 128)).astype(np.float32)
        with mrcfile.new(mic) as handle:
            handle.set_data(image)
            handle.voxel_size = 2.0
        mask = np.zeros(image.shape, dtype=np.uint8)
        if index != 2:  # Include clean, small, edge and PSD-scored components.
            mask[:45, :40] = 1
            mask[65:85, 65:90] = 1
            mask[110:112, 110:112] = 1
        np.save(mask_path, mask)
        rows.append(dict(dataset_id="demo", stem=mic.stem, micrograph_path=str(mic),
                         binary_mask_path=str(mask_path), pixel_size_angstrom=2.0))
    frame = pd.DataFrame(rows)
    manifest = root / "manifest.csv"
    frame.to_csv(manifest, index=False)
    return manifest, frame


def _run(manifest: Path, output: Path, *extra: str) -> None:
    assert typing_cli.main(["--classifier", "heuristic", "--manifest", str(manifest), "--output-dir", str(output), *extra]) == 0


def _assert_same_outputs(first: Path, second: Path) -> None:
    for name in ("component_type_assignments.csv", "image_contamination_summary.csv", "dataset_contamination_summary.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(first / name), pd.read_csv(second / name))
    for path in (first / "typed_masks").glob("*.npy"):
        np.testing.assert_array_equal(np.load(path), np.load(second / "typed_masks" / path.name))


def test_incremental_features_match_full_dataset_and_finalization_is_noop(tmp_path, monkeypatch):
    manifest, rows = _inputs(tmp_path)
    calls = []
    original = typing_cli._extract_image_features

    def extract(task):
        calls.append(task[0]["stem"])
        return original(task)

    monkeypatch.setattr(typing_cli, "_extract_image_features", extract)
    output = tmp_path / "incremental"
    rows.iloc[:1].to_csv(manifest, index=False)
    _run(manifest, output, "--incremental", "--expected-images", "3")
    partial = json.loads((output / "summary.json").read_text())
    assert (partial["n_images"], partial["n_images_expected"], partial["typing_status"]) == (1, 3, "running")
    rows.to_csv(manifest, index=False)
    _run(manifest, output, "--incremental", "--expected-images", "3")
    assert calls == ["mic_0", "mic_1", "mic_2"]
    stamps = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    _run(manifest, output, "--incremental", "--expected-images", "3")
    assert calls == ["mic_0", "mic_1", "mic_2"]
    assert stamps == {path: path.stat().st_mtime_ns for path in stamps}
    _run(manifest, tmp_path / "full")
    _assert_same_outputs(output, tmp_path / "full")


@pytest.mark.parametrize("change", ["mask", "micrograph", "pixel_size", "settings", "bands", "corrupt_cache"])
def test_incremental_invalidates_changed_inputs_and_settings(tmp_path, monkeypatch, change):
    manifest, rows = _inputs(tmp_path, count=1)
    output = tmp_path / "typing"
    _run(manifest, output, "--incremental")
    extra = []
    if change == "mask":
        mask = np.load(rows.iloc[0].binary_mask_path)
        mask[90:100, 90:100] = 1
        np.save(rows.iloc[0].binary_mask_path, mask)
    elif change == "micrograph":
        with mrcfile.open(rows.iloc[0].micrograph_path, mode="r+") as handle:
            handle.data[:] *= 0.5
    elif change == "pixel_size":
        rows["pixel_size_angstrom"] = 3.0
        rows.to_csv(manifest, index=False)
    elif change == "settings":
        extra = ["--min-component-area-px", "600"]
    elif change == "bands":
        bands = json.loads(typing_cli.DEFAULT_BANDS_JSON.read_text())
        bands["selected_frequency_bands_angstrom_inv"][0][0] *= 0.9
        bands_path = tmp_path / "bands.json"
        bands_path.write_text(json.dumps(bands))
        extra = ["--bands-json", str(bands_path)]
    else:
        cache = next(path for path in (output / ".feature_cache").glob("*.json") if path.name != "published.json")
        cache.write_text("incomplete write")
    calls = []
    original = typing_cli._extract_image_features
    monkeypatch.setattr(typing_cli, "_extract_image_features", lambda task: (calls.append(1), original(task))[1])
    _run(manifest, output, "--incremental", *extra)
    assert calls == [1]
    _run(manifest, tmp_path / "full", *extra)
    _assert_same_outputs(output, tmp_path / "full")


def test_parallel_typing_matches_serial(tmp_path):
    manifest, _ = _inputs(tmp_path)
    _run(manifest, tmp_path / "serial")
    _run(manifest, tmp_path / "parallel", "--workers", "2", "--incremental", "--summary-interval", "0")
    _assert_same_outputs(tmp_path / "serial", tmp_path / "parallel")


def test_replaced_typed_mask_is_repaired_without_reextracting_features(tmp_path, monkeypatch):
    manifest, _ = _inputs(tmp_path, count=1)
    output = tmp_path / "typing"
    _run(manifest, output, "--incremental")
    path = next((output / "typed_masks").glob("*.npy"))
    expected = np.load(path)
    np.save(path, np.zeros_like(expected))
    monkeypatch.setattr(typing_cli, "_extract_image_features", lambda task: pytest.fail("cached features must be reused"))
    _run(manifest, output, "--incremental")
    np.testing.assert_array_equal(np.load(path), expected)


def test_unchanged_masks_are_not_rewritten_for_live_summaries(tmp_path, monkeypatch):
    manifest, rows = _inputs(tmp_path)
    # Small/clean components have stable types regardless of dataset statistics.
    for path in rows.binary_mask_path:
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:32, 30:32] = 1
        np.save(path, mask)
    saved = []
    original = typing_cli._save_npy_atomic
    monkeypatch.setattr(typing_cli, "_save_npy_atomic", lambda path, array: (saved.append(path.name), original(path, array))[1])
    _run(manifest, tmp_path / "typing", "--incremental", "--summary-interval", "0")
    assert len(saved) == len(rows)
    assert len(set(saved)) == len(rows)


def test_inputs_changing_during_extraction_are_not_cached(tmp_path, monkeypatch):
    manifest, rows = _inputs(tmp_path, count=1)
    original = typing_cli._extract_image_features

    def extract(task):
        result = original(task)
        np.save(rows.iloc[0].binary_mask_path, np.zeros((128, 128), dtype=np.uint8))
        return result

    monkeypatch.setattr(typing_cli, "_extract_image_features", extract)
    with pytest.raises(ValueError, match="inputs changed"):
        _run(manifest, tmp_path / "typing", "--incremental")
    assert not list((tmp_path / "typing" / ".feature_cache").glob("*.json"))
