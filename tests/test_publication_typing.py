# Publication classifier UI integration.
"""Public output contracts and cache behavior for learned pixel typing."""
import json

import mrcfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cryofilter import typing_cli, publication_runner
from cryofilter.publication_typing import BinaryModel, resize_labels


def test_portable_logistic_probabilities_match_sklearn():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(80, 9)).astype(np.float32)
    labels = (x[:, 2] + x[:, 5] > 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, labels)
    scaler, clf = [step for _, step in model.steps]
    portable = BinaryModel(dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                                coef=clf.coef_.tolist(), intercept=clf.intercept_.tolist()))
    np.testing.assert_array_equal(portable.predict_proba(x), model.predict_proba(x))


def test_publication_is_cli_default():
    args = typing_cli.parse_args(["--manifest", "inputs.csv", "--output-dir", "outputs"])
    assert args.classifier == "publication"
    assert args.typing_checkpoint is None


def test_mixed_component_summary_counts_pixels_not_majority():
    mask = np.ones((10, 10), dtype=bool)
    typed = np.ones((10, 10), dtype=np.uint8)
    typed[:, 7:] = 3
    summary, components = publication_runner._summaries(
        {"dataset_id": "example", "stem": "mixed"}, mask,
        {"pred_map": typed, "top1_prob": np.full(mask.shape, .8, np.float16)})
    assert len(components) == summary["n_components"] == 1
    assert components[0]["type"] == "Carbon"
    assert summary["carbon_area_px"] == 70 and summary["aggregate_area_px"] == 30
    assert summary["aggregate_pct_of_contamination"] == 30


def test_publication_incremental_pipeline_and_monitor_schema(tmp_path, monkeypatch):
    calls = []

    class FakeClassifier:
        def __init__(self, **kwargs):
            pass

        def predict(self, image, mask, px):
            calls.append(px)
            typed = np.where(mask, 3, 0).astype(np.uint8)
            typed[:, :8] = np.where(mask[:, :8], 1, 0)
            return {"pred_map": typed, "top1_prob": mask.astype(np.float16),
                    "top1_margin": mask.astype(np.float16)}

    monkeypatch.setattr(publication_runner, "PublicationClassifier", FakeClassifier)
    rows = []
    for index in range(2):
        mic = tmp_path / f"mic{index}.mrc"
        with mrcfile.new(mic) as handle:
            handle.set_data(np.ones((16, 16), dtype=np.float32))
            handle.voxel_size = 2
        mask_path = tmp_path / f"mask{index}.npy"
        np.save(mask_path, np.ones((16, 16), dtype=np.uint8))
        rows.append(dict(dataset_id="example", stem=f"mic{index}", micrograph_path=str(mic),
                         binary_mask_path=str(mask_path), pixel_size_angstrom=2))
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "typing"

    def run(count):
        pd.DataFrame(rows[:count]).to_csv(manifest, index=False)
        return typing_cli.main(["--manifest", str(manifest), "--output-dir", str(output),
                                "--incremental", "--expected-images", "2", "--summary-interval", "0"])

    assert run(1) == 0
    first_mask = output / "typed_masks/example__mic0_typed_mask.npy"
    stamp = first_mask.stat().st_mtime_ns
    assert json.loads((output / "summary.json").read_text())["typing_status"] == "running"
    assert run(2) == 0
    assert calls == [2, 2] and first_mask.stat().st_mtime_ns == stamp
    summary = json.loads((output / "summary.json").read_text())
    assert summary["typing_status"] == "complete" and summary["n_images"] == 2
    assert summary["type_order"] == ["Carbon", "Crystalline", "Aggregate", "Ethane"]
    assert summary["overall_contamination_summary"]["aggregate_area_px"] == 256
    assert summary["classifier"] == "publication"
    stamps = {path: path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    assert run(2) == 0
    assert calls == [2, 2]
    assert stamps == {path: path.stat().st_mtime_ns for path in stamps}
    # Changed source masks must invalidate classification.
    np.save(rows[1]["binary_mask_path"], np.zeros((16, 16), dtype=np.uint8))
    assert run(2) == 0 and len(calls) == 3
    assert not np.load(output / "typed_masks/example__mic1_typed_mask.npy").any()


def test_resampling_preserves_label_ids():
    labels = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    resized = resize_labels(labels, (13, 17))
    assert resized.shape == (13, 17) and resized.dtype == np.uint8
    assert set(np.unique(resized)) <= set(np.unique(labels))


def test_native_resolution_confidence_resampling_supports_float16():
    confidence = np.array([[.1, .5], [.8, .9]], dtype=np.float16)
    actual = resize_labels(confidence, (9, 11))
    assert actual.shape == (9, 11) and actual.dtype == np.float16
    assert set(np.unique(actual)) <= set(np.unique(confidence))


@pytest.mark.parametrize("pixel_size", [0.5, 1.2, 2.0, 3.0])
def test_native_raster_roundtrip_preserves_binary_mask(pixel_size):
    from cryofilter.publication_typing import PublicationClassifier
    classifier = object.__new__(PublicationClassifier)
    classifier.settings = {"target_pixel_size_angstrom": 2.0,
                           "normalization_method": "percentile_extra_wide"}
    shapes = []

    def prepared(image, mask):
        shapes.append(image.shape)
        return {"pred_map": (mask * 3).astype(np.uint8),
                "top1_prob": (mask * .8).astype(np.float16),
                "top1_margin": (mask * .2).astype(np.float16), "centers": 1}

    classifier.predict_prepared = prepared
    image = np.random.default_rng(4).normal(size=(65, 91)).astype(np.float32)
    mask = np.zeros(image.shape, bool)
    mask[12:51, 18:75] = True
    result = classifier.predict(image, mask, pixel_size)
    assert shapes and (shapes[0] == image.shape) == (pixel_size == 2.0)
    for key in ("pred_map", "top1_prob", "top1_margin"):
        assert result[key].shape == mask.shape
        assert np.all(result[key][~mask] == 0)
    assert np.all(result["pred_map"][20:40, 25:60] == 3)


def test_gpu_memory_retry_reduces_batch_without_changing_recipe(monkeypatch):
    import torch
    from cryofilter import publication_typing
    classifier = object.__new__(publication_typing.PublicationClassifier)
    classifier.kwargs = {"batch_size": 4}
    calls = []

    def predict(image, mask, **kwargs):
        calls.append(kwargs["batch_size"])
        if kwargs["batch_size"] > 1:
            raise torch.cuda.OutOfMemoryError("test allocation")
        return {"pred_map": mask.astype(np.uint8)}

    monkeypatch.setattr(publication_typing, "predict_prepared", predict)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    mask = np.ones((8, 8), dtype=bool)
    with pytest.warns(RuntimeWarning, match="Reducing publication typing batch size"):
        result = classifier.predict_prepared(np.zeros(mask.shape, np.float32), mask)
    assert calls == [4, 2, 1]
    np.testing.assert_array_equal(result["pred_map"], mask)


def test_missing_lfs_weights_fail_without_falling_back(tmp_path):
    from cryofilter.publication_typing import PublicationClassifier
    pointer = tmp_path / "encoder.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        PublicationClassifier(checkpoint=pointer, device="cpu")


def test_segmentation_weights_cannot_replace_feature_encoder(tmp_path):
    from cryofilter.publication_typing import PublicationClassifier
    checkpoint = tmp_path / "wrong_FULL.pt"
    checkpoint.write_bytes(b"incorrect pretrained checkpoint" * 100)
    with pytest.raises(ValueError, match="not interchangeable"):
        PublicationClassifier(checkpoint=checkpoint, device="cpu")
