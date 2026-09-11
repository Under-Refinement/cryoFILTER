"""Classifier downloads must work across UI, CLI, and relocated checkouts."""
from pathlib import Path

import pytest

from cryofilter import classifier_weights as weights


@pytest.fixture
def locations(tmp_path, monkeypatch):
    install = tmp_path / "installation"
    working = tmp_path / "project"
    assets = install / "cryofilter/data/publication"
    working.mkdir()
    assets.mkdir(parents=True)
    monkeypatch.setattr(weights, "INSTALL_ROOT", install)
    monkeypatch.setattr(weights, "MODEL_DIR", assets)
    monkeypatch.chdir(working)
    monkeypatch.delenv("CRYOFILTER_CLASSIFIER_CHECKPOINT", raising=False)
    monkeypatch.delenv("CRYOFILTER_WEIGHTS_DIR", raising=False)
    return install, working, assets


def checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test model data" * 128)
    return path


@pytest.mark.parametrize("root,relative", [
    (0, "pretrained_models/classifier.pt"),
    (0, "classifier.pt"),
    (1, "pretrained_models/classifier.pt"),
    (1, "classifier.pt"),
    (2, "classifier.pt"),
    (2, "encoder.pt"),
])
def test_default_discovery_from_another_working_directory(locations, root, relative):
    expected = checkpoint(locations[root] / relative)
    assert weights.resolve_classifier_checkpoint() == expected


def test_zenodo_download_precedes_legacy_lfs_weights(locations):
    install, working, assets = locations
    checkpoint(assets / "encoder.pt")
    downloaded = checkpoint(install / "pretrained_models/classifier.pt")
    assert weights.resolve_classifier_checkpoint() == downloaded
    local = checkpoint(working / "pretrained_models/classifier.pt")
    assert weights.resolve_classifier_checkpoint() == local


def test_unmaterialized_lfs_pointer_does_not_block_zenodo_download(locations):
    install, working, assets = locations
    (assets / "encoder.pt").write_text("version https://git-lfs.github.com/spec/v1\n")
    downloaded = checkpoint(install / "pretrained_models/classifier.pt")
    assert weights.resolve_classifier_checkpoint() == downloaded


def test_explicit_and_environment_paths_precede_auto_discovery(locations, monkeypatch):
    install, working, assets = locations
    checkpoint(install / "pretrained_models/classifier.pt")
    custom = checkpoint(working / "custom/classifier.pt")
    explicit = checkpoint(working / "explicit/classifier.pt")
    monkeypatch.setenv("CRYOFILTER_CLASSIFIER_CHECKPOINT", str(custom))
    assert weights.resolve_classifier_checkpoint() == custom
    assert weights.resolve_classifier_checkpoint(explicit) == explicit
    assert weights.resolve_classifier_checkpoint(explicit.parent) == explicit
    monkeypatch.setenv("CRYOFILTER_CLASSIFIER_CHECKPOINT", str(custom.parent))
    assert weights.resolve_classifier_checkpoint() == custom


def test_missing_explicit_path_never_silently_falls_back(locations, monkeypatch):
    install, working, assets = locations
    checkpoint(assets / "encoder.pt")
    missing = working / "missing-classifier.pt"
    assert weights.resolve_classifier_checkpoint(missing) == missing
    monkeypatch.setenv("CRYOFILTER_CLASSIFIER_CHECKPOINT", str(missing))
    assert weights.resolve_classifier_checkpoint() == missing


def test_custom_segmentation_directory_reaches_background_typing(locations, monkeypatch):
    from cryofilter.cryosparc.remote import predict

    install, working, assets = locations
    checkpoint(assets / "encoder.pt")
    custom = checkpoint(working / "shared-weights/classifier.pt")
    calls = []
    monkeypatch.setattr(predict, "_run_polled_command", lambda command, **kw: calls.append(kw))
    predict.run_local_typing(manifest_path=working / "manifest.csv", output_dir=working / "typing",
                             weights_dir=custom.parent)
    assert calls[0]["env"]["CRYOFILTER_WEIGHTS_DIR"] == str(custom.parent)
    monkeypatch.setenv("CRYOFILTER_WEIGHTS_DIR", calls[0]["env"]["CRYOFILTER_WEIGHTS_DIR"])
    assert weights.resolve_classifier_checkpoint() == custom
    predict.run_local_typing(manifest_path=working / "manifest.csv", output_dir=working / "typing",
                             weights_dir=install)
    assert calls[1]["env"]["CRYOFILTER_WEIGHTS_DIR"] == str(custom.parent)


def test_manifest_adjacent_download_precedes_legacy_package(locations):
    install, working, assets = locations
    checkpoint(assets / "encoder.pt")
    manifest_dir = working / "analysis"
    downloaded = checkpoint(manifest_dir / "classifier.pt")
    assert weights.resolve_classifier_checkpoint(search_dirs=(manifest_dir,)) == downloaded


def test_missing_weights_returns_documented_install_location(locations):
    install, working, assets = locations
    assert weights.resolve_classifier_checkpoint() == install / "pretrained_models/classifier.pt"
