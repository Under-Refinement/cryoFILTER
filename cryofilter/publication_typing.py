"""Portable, frozen publication contamination subtype inference."""
from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit
import torch
from torch import nn

from models.bad_region_detector import create_model
from models.predict import _attach_model_input_metadata, _resolve_checkpoint_model_config
from utils.fourier_rescale import fourier_rescale_2d
from utils.image_utils import normalize_image
from .publication_kernels import PairExpertSpec, ScaleSpec, predict_prepared

MODEL_DIR = Path(__file__).resolve().parent / "data" / "publication"


class BinaryModel:
    """Frozen StandardScaler + binary logistic regression, without pickle."""

    def __init__(self, spec):
        self.constant = spec.get("constant")
        if self.constant is None:
            self.mean = np.asarray(spec["mean"], dtype=np.float64)
            self.scale = np.asarray(spec["scale"], dtype=np.float64)
            self.coef = np.asarray(spec["coef"], dtype=np.float64)
            self.intercept = np.asarray(spec["intercept"], dtype=np.float64)

    def predict_proba(self, x):
        if self.constant is not None:
            positive = np.full(len(x), self.constant, dtype=np.float64)
        else:
            # Match sklearn's in-place float32 scaling before float64 regression.
            scaled = np.array(x, copy=True)
            scaled -= self.mean
            scaled /= self.scale
            positive = expit((scaled @ self.coef.T + self.intercept).reshape(-1))
        return np.column_stack((1.0 - positive, positive))


class HierarchicalModel:
    classes_ = np.asarray([1, 2, 3, 4], dtype=np.int64)

    def __init__(self, spec):
        self.ordered = BinaryModel(spec["ordered"])
        self.ethane = BinaryModel(spec["ethane"])
        self.aggregate = BinaryModel(spec["aggregate"])

    def predict_proba(self, x):
        ordered = self.ordered.predict_proba(x)[:, 1].astype(np.float32)
        ethane = self.ethane.predict_proba(x)[:, 1].astype(np.float32)
        aggregate = self.aggregate.predict_proba(x)[:, 1].astype(np.float32)
        probs = np.stack(((1 - ordered) * (1 - aggregate), ordered * (1 - ethane),
                          (1 - ordered) * aggregate, ordered * ethane), axis=1).astype(np.float32)
        return probs / np.where(probs.sum(axis=1, keepdims=True) > 0,
                                probs.sum(axis=1, keepdims=True), 1.0)


class FullFeatureProbe(nn.Module):
    """Publication FULL backbone exposing frozen pre-final decoder features."""

    def __init__(self, checkpoint_path, device, model_config):
        super().__init__()
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        checkpoint.update(model_config)
        cfg = _resolve_checkpoint_model_config(checkpoint)
        self.model = create_model(
            model_type=str(cfg["model_type"]), device=str(device),
            use_power_spectrum=bool(cfg["use_power_spectrum"]),
            input_channels_override=int(cfg["input_channels"]), num_classes=int(cfg["num_classes"]),
            attention_type=str(cfg.get("attention_type", "global")),
            norm_type=str(cfg.get("norm_type", "group")), decoder_dropout=float(cfg.get("decoder_dropout", 0.1)),
        )
        self.model.load_state_dict(dict(cfg["state_dict"]), strict=True)
        _attach_model_input_metadata(self.model, cfg)
        self.input_channels = int(cfg["input_channels"])
        self.use_power_spectrum = bool(cfg["use_power_spectrum"])
        self.include_real_space_input = bool(cfg.get("include_real_space_input", True))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._features = None
        self.model.final.register_forward_pre_hook(self._capture)
        self.to(device)

    def _capture(self, module, inputs):
        self._features = inputs[0]

    @torch.inference_mode()
    def forward(self, x):
        self._features = None
        output = self.model(x)
        if self._features is None:
            raise RuntimeError("Could not capture publication decoder features")
        return self._features, output[0] if isinstance(output, tuple) else output


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_labels(array, shape):
    """Original publication nearest-neighbor mask resampling."""
    if tuple(array.shape) == tuple(shape):
        return array
    # SciPy does not support float16 inputs; confidence maps use that dtype.
    working = array.astype(np.float32) if array.dtype == np.float16 else array
    resized = ndi.zoom(working, (shape[0] / array.shape[0], shape[1] / array.shape[1]), order=0).astype(array.dtype, copy=False)
    if resized.shape == tuple(shape):
        return resized
    result = np.zeros(shape, dtype=array.dtype)
    h, w = min(shape[0], resized.shape[0]), min(shape[1], resized.shape[1])
    result[:h, :w] = resized[:h, :w]
    return result


class PublicationClassifier:
    def __init__(self, *, model_dir=None, checkpoint=None, device="auto", batch_size=16):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.metadata = json.loads((self.model_dir / "manifest.json").read_text())
        for name in ("models.json", "settings.json", "model_config.json"):
            if file_sha256(self.model_dir / name) != self.metadata["files"][name]["sha256"]:
                raise ValueError(f"Publication classifier artifact checksum mismatch: {name}")
        self.checkpoint = Path(checkpoint) if checkpoint else self.model_dir / "encoder.pt"
        expected = self.metadata["files"]["encoder.pt"]["sha256"]
        if not self.checkpoint.is_file() or self.checkpoint.stat().st_size < 1024:
            raise FileNotFoundError(
                "Publication classifier weights are missing. Run git lfs pull "
                "--include='cryofilter/data/publication/encoder.pt' in the repository, "
                "or pass --typing-checkpoint with the released publication encoder.pt."
            )
        if file_sha256(self.checkpoint) != expected:
            raise ValueError("The subtype classifier requires its exact publication encoder.pt; "
                             "the segmentation FULL checkpoint is not interchangeable.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if batch_size < 1:
            raise ValueError("Typing batch size must be positive")
        self.probe = FullFeatureProbe(self.checkpoint, self.device,
                                      json.loads((self.model_dir / "model_config.json").read_text()))
        models = json.loads((self.model_dir / "models.json").read_text())
        self.settings = json.loads((self.model_dir / "settings.json").read_text())
        self.kwargs = dict(self.settings)
        self.kwargs.update(
            probe=self.probe, patch_model=HierarchicalModel(models["main"]),
            rescue_model=HierarchicalModel(models["rescue"]),
            pair_expert_models=[(PairExpertSpec(**item["spec"]), BinaryModel(item["model"]))
                                for item in models["pair_experts"]],
            pair_refine_model=None, device=self.device, batch_size=int(batch_size),
            scales=[ScaleSpec(**item) for item in self.settings["scales"]],
            rescue_scales=[ScaleSpec(**item) for item in self.settings["rescue_scales"]],
        )
        self.kwargs = {key: value for key, value in self.kwargs.items()
                       if key in inspect.signature(predict_prepared).parameters}

    def predict_prepared(self, image_norm, mask):
        image = np.asarray(image_norm, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if image.ndim != 2 or image.shape != mask.shape or not np.all(np.isfinite(image)):
            raise ValueError("Typing requires finite 2-D image data aligned with the mask")
        while True:
            try:
                return predict_prepared(image, mask, **self.kwargs)
            except torch.cuda.OutOfMemoryError:
                if self.kwargs["batch_size"] <= 1:
                    raise
                self.kwargs["batch_size"] = max(1, self.kwargs["batch_size"] // 2)
                warnings.warn(f"Reducing publication typing batch size to {self.kwargs['batch_size']} "
                              "after GPU memory exhaustion", RuntimeWarning)
            # The exception frame must be released before clearing GPU cache.
            torch.cuda.empty_cache()

    def predict(self, image, mask, pixel_size_angstrom):
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if image.ndim != 2 or image.shape != mask.shape:
            raise ValueError("Micrograph and binary mask shapes must match")
        px = float(pixel_size_angstrom)
        if not np.isfinite(px) or px <= 0:
            raise ValueError("Pixel size must be finite and positive")
        target = float(self.settings["target_pixel_size_angstrom"])
        scale = px / target
        scaled = image if abs(scale - 1.0) < 1e-6 else fourier_rescale_2d(image, scale=scale)
        small_mask = resize_labels(mask.astype(np.uint8), scaled.shape).astype(bool)
        normalized = normalize_image(scaled, method=self.settings["normalization_method"]).astype(np.float32)
        result = self.predict_prepared(normalized, small_mask)
        for key in ("pred_map", "top1_prob", "top1_margin"):
            result[key] = resize_labels(result[key], mask.shape)
            result[key][~mask] = 0
        # Very small native components can disappear during downsampling. Keep
        # those pixels explicitly unclassified instead of inventing a subtype.
        return result
