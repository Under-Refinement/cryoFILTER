"""Locate publication classifier weights without requiring Git LFS."""
from __future__ import annotations

import os
from pathlib import Path

INSTALL_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent / "data" / "publication"
CLASSIFIER_FILENAME = "classifier.pt"
ZENODO_RECORD_URL = "https://zenodo.org/records/22700873"


def _checkpoint_path(value):
    path = Path(value).expanduser().resolve()
    return path / CLASSIFIER_FILENAME if path.is_dir() else path


def resolve_classifier_checkpoint(checkpoint=None, *, model_dir=None, search_dirs=()):
    """Prefer explicit paths, then local weights, then the legacy packaged file.

    Directory probes are bounded so live typing never scans a filesystem tree.
    Content verification remains the responsibility of PublicationClassifier.
    """
    if checkpoint is not None:
        return _checkpoint_path(checkpoint)
    configured = os.environ.get("CRYOFILTER_CLASSIFIER_CHECKPOINT")
    if configured:
        return _checkpoint_path(configured)

    directories = []
    weights_dir = os.environ.get("CRYOFILTER_WEIGHTS_DIR")
    if weights_dir:
        directories.append(Path(weights_dir))
    directories.extend(Path(path) for path in search_dirs)
    directories.extend((Path.cwd() / "pretrained_models", Path.cwd(),
                        INSTALL_ROOT / "pretrained_models", INSTALL_ROOT))
    asset_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    directories.append(asset_dir)
    candidates = [path.expanduser().resolve() / CLASSIFIER_FILENAME for path in directories]
    candidates.append(asset_dir.expanduser().resolve() / "encoder.pt")
    for path in dict.fromkeys(candidates):
        # Ignore unmaterialized LFS pointers during automatic discovery.
        if path.is_file() and path.stat().st_size >= 1024:
            return path
    return (INSTALL_ROOT / "pretrained_models" / CLASSIFIER_FILENAME).resolve()
