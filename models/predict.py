"""
Inference script for running inference using trained model to filter particle picks.

This module runs inference on micrographs using fine-tuned model weights,
saves probability maps to pickle files, and filters particle picks based on
proximity to detected bad regions.
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import os
import sys
import pickle
from datetime import datetime
from tqdm import tqdm
from scipy import ndimage

# Add parent directory to path for imports when running as script
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.file_io import load_mrc, load_cs, save_cs, picks_to_coordinates, create_csg_file
from utils.image_utils import (
    normalize_image,
    compute_power_spectrum,
    compute_multiscale_radial_psd_image,
    compute_multiscale_radial_psd_image_batch,
    compute_multiscale_radial_psd_channels,
    compute_frequency_band_psd_channels,
    build_constant_pixel_size_channel,
)
from utils.fourier_rescale import fourier_rescale_2d

# Handle both relative import (when used as module) and absolute import (when run as script)
try:
    from .bad_region_detector import create_model
except ImportError:
    from models.bad_region_detector import create_model


def _infer_first_conv_input_channels(state_dict: dict) -> Optional[int]:
    if not isinstance(state_dict, dict):
        return None
    priority_suffixes = (
        "enc1.0.weight",
        "conv1.weight",
        "features.0.weight",
        "stem.0.weight",
    )
    for suffix in priority_suffixes:
        for key, tensor in state_dict.items():
            if key.endswith(suffix) and isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                return int(tensor.shape[1])
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            return int(tensor.shape[1])
    return None


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(str(k).startswith("module.") for k in keys):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def _detect_model_type_from_state_dict(state_dict: dict) -> str:
    has_stem = any("stem." in k for k in state_dict.keys())
    has_stages = any("stages." in k for k in state_dict.keys())
    has_conv1 = any("conv1.weight" in k for k in state_dict.keys())
    has_layer1 = any("layer1" in k and "dec" not in k for k in state_dict.keys())
    has_enc1 = any("enc1" in k for k in state_dict.keys())

    if has_stem and has_stages:
        raise ValueError("Checkpoint architecture is not supported by this public release")
    if has_conv1 and has_layer1:
        return "resnet_unet"
    if has_enc1:
        return "unet_attention"
    return "simple"


def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    for key in ("final.weight", "binary_head.2.weight", "binary_head.1.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return 2


def _infer_decoder_dropout_from_state_dict(state_dict: dict) -> float:
    tensor = state_dict.get("dec4.4.weight")
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
        # A parameterless dropout layer at dec*.3 shifts the second decoder conv
        # to index 4 in our trained GroupNorm checkpoints.
        return 0.1
    return 0.0


def _infer_norm_type_from_state_dict(state_dict: dict) -> str:
    if any(str(k).endswith("num_batches_tracked") or str(k).endswith("running_mean") for k in state_dict.keys()):
        return "batch"
    return "group"


def _resolve_checkpoint_model_config(checkpoint) -> Dict[str, object]:
    meta = checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")
    state_dict = _strip_module_prefix(state_dict)

    input_channels = meta.get("input_channels")
    if input_channels is None:
        input_channels = _infer_first_conv_input_channels(state_dict)
    if input_channels is None:
        input_channels = 1
    input_channels = int(input_channels)

    include_real_space_input = bool(meta.get("include_real_space_input", True))
    use_pixel_size_channel = bool(meta.get("use_pixel_size_channel", False))
    use_power_spectrum = meta.get("use_power_spectrum")
    if use_power_spectrum is None:
        base_real_channels = 1 if include_real_space_input else 0
        pixel_channels = 1 if use_pixel_size_channel else 0
        use_power_spectrum = input_channels > (base_real_channels + pixel_channels)

    psd_scales = tuple(int(x) for x in tuple(meta.get("psd_scales", (16, 32, 64))))
    raw_frequency_bands = meta.get("psd_frequency_bands", ())
    psd_frequency_bands = []
    for band in tuple(raw_frequency_bands):
        if isinstance(band, (list, tuple)) and len(band) == 2:
            psd_frequency_bands.append((float(band[0]), float(band[1])))

    norm_type = str(meta.get("norm_type") or _infer_norm_type_from_state_dict(state_dict))
    decoder_dropout = float(meta.get("decoder_dropout", _infer_decoder_dropout_from_state_dict(state_dict)))
    num_classes = int(meta.get("num_classes", _infer_num_classes_from_state_dict(state_dict)))

    return {
        "state_dict": state_dict,
        "model_type": meta.get("model_type") or _detect_model_type_from_state_dict(state_dict),
        "attention_type": meta.get("attention_type", "global"),
        "num_classes": num_classes,
        "use_power_spectrum": bool(use_power_spectrum),
        "include_real_space_input": include_real_space_input,
        "use_pixel_size_channel": use_pixel_size_channel,
        "pixel_size_channel_min_angstrom": float(meta.get("pixel_size_channel_min_angstrom", 0.3)),
        "pixel_size_channel_max_angstrom": float(meta.get("pixel_size_channel_max_angstrom", 1.4)),
        "input_channels": input_channels,
        "norm_type": norm_type,
        "decoder_dropout": decoder_dropout,
        "psd_multiscale": bool(meta.get("psd_multiscale", False)),
        "psd_multiscale_separate_channels": bool(meta.get("psd_multiscale_separate_channels", False)),
        "psd_scales": psd_scales,
        "psd_frequency_band_channels": bool(meta.get("psd_frequency_band_channels", False)),
        "psd_frequency_bands": tuple(psd_frequency_bands),
        "psd_frequency_band_include_full_spectrum": bool(
            meta.get("psd_frequency_band_include_full_spectrum", False)
        ),
        "psd_frequency_band_include_anisotropy": bool(meta.get("psd_frequency_band_include_anisotropy", False)),
        "psd_frequency_band_anisotropy_min_freq": float(meta.get("psd_frequency_band_anisotropy_min_freq", 0.0)),
    }


def _attach_model_input_metadata(model: torch.nn.Module, config: Dict[str, object]) -> None:
    target = model.module if hasattr(model, "module") else model
    for meta_target in (model, target):
        meta_target.use_power_spectrum = bool(config.get("use_power_spectrum", False))
        meta_target.include_real_space_input = bool(config.get("include_real_space_input", True))
        meta_target.use_pixel_size_channel = bool(config.get("use_pixel_size_channel", False))
        meta_target.pixel_size_channel_min_angstrom = float(config.get("pixel_size_channel_min_angstrom", 0.3))
        meta_target.pixel_size_channel_max_angstrom = float(config.get("pixel_size_channel_max_angstrom", 1.4))
        meta_target.input_channels = int(config.get("input_channels", 1))
        meta_target.psd_multiscale = bool(config.get("psd_multiscale", False))
        meta_target.psd_multiscale_separate_channels = bool(config.get("psd_multiscale_separate_channels", False))
        meta_target.psd_scales = tuple(int(x) for x in tuple(config.get("psd_scales", (16, 32, 64))))
        meta_target.psd_frequency_band_channels = bool(config.get("psd_frequency_band_channels", False))
        meta_target.psd_frequency_bands = tuple(
            (float(lo), float(hi)) for lo, hi in tuple(config.get("psd_frequency_bands", ()))
        )
        meta_target.psd_frequency_band_include_full_spectrum = bool(
            config.get("psd_frequency_band_include_full_spectrum", False)
        )
        meta_target.psd_frequency_band_include_anisotropy = bool(
            config.get("psd_frequency_band_include_anisotropy", False)
        )
        meta_target.psd_frequency_band_anisotropy_min_freq = float(
            config.get("psd_frequency_band_anisotropy_min_freq", 0.0)
        )


def predict_bad_regions_probability(model: torch.nn.Module, image: np.ndarray,
                                   device: str = "cuda", patch_size: int = 512,
                                   overlap: int = 64, pixel_size_angstrom: float = 1.1,
                                   use_power_spectrum: bool = False,
                                   include_real_space_input: bool = True,
                                   use_pixel_size_channel: Optional[bool] = None,
                                   pixel_size_channel_min_angstrom: Optional[float] = None,
                                   pixel_size_channel_max_angstrom: Optional[float] = None,
                                   use_hann_blending: bool = True,
                                   blending_window: str = "hann",
                                   blending_edge_px: int = 16,
                                   stitch_phase_mode: str = "single",
                                   stitch_phase_reduction: str = "mean",
                                   stitch_phase_blend_alpha: float = 1.0,
                                   stitch_output_space: str = "probability",
                                   stitch_output_blend_alpha: float = 0.5,
                                   stitch_output_preserve_threshold: float = 0.8,
                                   stitch_output_transition: float = 0.15,
                                   patch_prob_floor_quantile: float = 0.0,
                                   patch_prob_floor_strength: float = 0.0,
                                   downsample_factor: float = 1.0,
                                   adaptive_downsample_method: str = "zoom",
                                   return_native_resolution: bool = False,
                                   batch_forward_size: int = 1,
                                   use_tta: bool = False,
                                   temperature: float = 1.0,
                                   logit_bias: float = 0.0,
                                   normalization_method: str = "robust",
                                   use_particle_size_downsampling: bool = False,
                                   target_pixels_per_particle: float = 6.0,
                                   particle_size_angstrom: float = 100.0,
                                   global_prior_path: Optional[str] = None,
                                   use_global_prior: bool = True,
                                   psd_use_radial_normalization: bool = True,
                                   # Multi-scale PSD parameters (must match training!)
                                   psd_multiscale: Optional[bool] = None,
                                   psd_multiscale_separate_channels: Optional[bool] = None,
                                   multiscale_psd_source: str = "patch",
                                   psd_scales: Optional[Tuple[int, ...]] = None,
                                   psd_frequency_band_channels: Optional[bool] = None,
                                   psd_frequency_bands: Optional[Tuple[Tuple[float, float], ...]] = None,
                                   psd_frequency_band_include_full_spectrum: Optional[bool] = None,
                                   psd_frequency_band_include_anisotropy: Optional[bool] = None,
                                   psd_frequency_band_anisotropy_min_freq: Optional[float] = None,
                                   psd_texture_mode: bool = False,
                                   psd_highpass_cutoff: float = 0.2,
                                   tile_offset_y: int = 0,
                                   tile_offset_x: int = 0) -> np.ndarray:
    """
    Predict bad region probabilities (returns probability map, not binary mask).

    Args:
        model: Trained model
        image: Input image (will be normalized)
        device: Device to run inference on
        patch_size: Size of patches for inference
        overlap: Overlap between patches
        pixel_size_angstrom: Pixel size (for power spectrum computation if needed)
        use_power_spectrum: If True, include power spectrum as additional channel
        include_real_space_input: If False (and use_power_spectrum=True), feed PSD-only
                                  channels without the real-space image channel.
        use_hann_blending: If True, use Hann/cosine window for smoother patch blending
        blending_window: Blend window used when stitching overlapping patches.
                         Supported: "hann" (legacy) or "tukey" (flat center with tapered edges).
        blending_edge_px: Edge taper size in pixels for Tukey blending.
        stitch_phase_mode: Optional shifted-grid ensemble used to suppress tiling haze.
                           "single" keeps the legacy grid, "diag_halfstride" averages
                           the base grid with a half-stride diagonal shift, and
                           "quad_halfstride" averages four half-stride grid phases.
        stitch_phase_reduction: Reduction used when combining shifted-grid predictions.
                                "mean" is the default, "median" is more robust to
                                phase-specific blocks, and "max" preserves peaks most
                                aggressively.
        stitch_phase_blend_alpha: Blend between the base grid prediction and the
                                  shifted-grid consensus. 1.0 uses the consensus
                                  only, 0.0 keeps the base grid only.
        stitch_output_space: Space used for overlap stitching. "probability" keeps
                             the legacy behavior, "logit" averages a bad-vs-good
                             logit margin and converts to probability only after stitching,
                             "hybrid" blends the two globally, and "adaptive_hybrid"
                             blends them mostly in low-confidence regions.
        stitch_output_blend_alpha: When stitch_output_space="hybrid", weight placed on
                                   the logit-stitched map (0=probability stitch only,
                                   1=logit stitch only).
        stitch_output_preserve_threshold: For adaptive_hybrid, probabilities above this
                                          remain mostly on the legacy probability stitch.
        stitch_output_transition: Width of the adaptive_hybrid transition band.
        patch_prob_floor_quantile: Optional patch-local quantile used to estimate a
                                   background floor when correcting patch-to-patch
                                   probability drift during stitching.
        patch_prob_floor_strength: Strength of the patch-baseline drift correction.
        downsample_factor: If > 1, downsample image before inference then upsample result (fast mode)
        adaptive_downsample_method: Resampling method for fast mode. "fourier" matches the
                                    released 2 A/px balanced small-pixel recipe.
        return_native_resolution: If True, skip final upsample and return the work-resolution
                                 probability map when internal downsampling is used.
        batch_forward_size: If > 1, batch multiple patches per model forward pass (faster on GPU)
        use_tta: If True, use test-time augmentation (default: False)
        temperature: Temperature scaling for calibration (default: 1.0, no scaling)
                     T > 1: softer probabilities (reduces confident false positives)
                     T < 1: sharper probabilities
        logit_bias: Logit bias for calibration (default: 0.0, no bias)
                    Negative bias: reduces false positives (subtract from bad class logits)
                    Positive bias: increases false positives (add to bad class logits)
        normalization_method: Normalization method - "robust" (default) or "percentile"
        psd_use_radial_normalization: If True, radial-whiten PSD (highlights rings/anisotropy).
                                     If False, keep raw log-PSD slope cues (often helps carbon vs hole).
        use_particle_size_downsampling: If True, downsample to standardize resolution relative to particle size (MC-inspired)
        target_pixels_per_particle: Target pixels per particle for downsampling (default: 6.0)
        particle_size_angstrom: Particle size in Angstrom for downsampling (default: 100.0)
        global_prior_path: Optional path to global_prior.json file (MC-inspired statistical prior)
        use_global_prior: If True, use global prior for prevalence correction (default: True)
        psd_multiscale_separate_channels: If True, provide one PSD channel per radial scale
                                         instead of a single fused multiscale PSD image.
        multiscale_psd_source: For PSD inputs, either compute PSD from each real-space
                              patch ("patch", training-matched legacy behavior) or once
                              on the full working-resolution image ("full_image") before slicing.

    Returns:
        Probability map (0-1 range, not thresholded) at original image resolution
    """
    model.eval()
    original_h, original_w = image.shape
    original_image = image.copy()  # Store original for upsampling

    if str(blending_window) not in {"hann", "tukey"}:
        raise ValueError("blending_window must be 'hann' or 'tukey'")
    if str(stitch_phase_mode) not in {"single", "diag_halfstride", "quad_halfstride"}:
        raise ValueError("stitch_phase_mode must be 'single', 'diag_halfstride', or 'quad_halfstride'")
    if str(stitch_phase_reduction) not in {"mean", "median", "max"}:
        raise ValueError("stitch_phase_reduction must be 'mean', 'median', or 'max'")
    if str(stitch_output_space) not in {"probability", "logit", "hybrid", "adaptive_hybrid"}:
        raise ValueError("stitch_output_space must be 'probability', 'logit', 'hybrid', or 'adaptive_hybrid'")
    stitch_phase_blend_alpha = float(stitch_phase_blend_alpha)
    if not (0.0 <= stitch_phase_blend_alpha <= 1.0):
        raise ValueError("stitch_phase_blend_alpha must be between 0.0 and 1.0")
    stitch_output_blend_alpha = float(stitch_output_blend_alpha)
    if not (0.0 <= stitch_output_blend_alpha <= 1.0):
        raise ValueError("stitch_output_blend_alpha must be between 0.0 and 1.0")
    stitch_output_preserve_threshold = float(stitch_output_preserve_threshold)
    stitch_output_transition = float(stitch_output_transition)
    if not (0.0 <= stitch_output_preserve_threshold <= 1.0):
        raise ValueError("stitch_output_preserve_threshold must be between 0.0 and 1.0")
    if stitch_output_transition <= 0.0:
        raise ValueError("stitch_output_transition must be > 0")
    patch_prob_floor_quantile = float(patch_prob_floor_quantile)
    patch_prob_floor_strength = float(patch_prob_floor_strength)
    if not (0.0 <= patch_prob_floor_quantile <= 1.0):
        raise ValueError("patch_prob_floor_quantile must be between 0.0 and 1.0")
    if not (0.0 <= patch_prob_floor_strength <= 1.0):
        raise ValueError("patch_prob_floor_strength must be between 0.0 and 1.0")
    tile_offset_y = int(max(0, tile_offset_y))
    tile_offset_x = int(max(0, tile_offset_x))
    if use_pixel_size_channel is None:
        use_pixel_size_channel = bool(getattr(model, "use_pixel_size_channel", False))
    if pixel_size_channel_min_angstrom is None:
        pixel_size_channel_min_angstrom = float(getattr(model, "pixel_size_channel_min_angstrom", 0.3))
    if pixel_size_channel_max_angstrom is None:
        pixel_size_channel_max_angstrom = float(getattr(model, "pixel_size_channel_max_angstrom", 1.4))
    if psd_multiscale is None:
        psd_multiscale = bool(getattr(model, "psd_multiscale", False))
    if psd_multiscale_separate_channels is None:
        psd_multiscale_separate_channels = bool(getattr(model, "psd_multiscale_separate_channels", False))
    if psd_scales is None:
        psd_scales = tuple(int(x) for x in tuple(getattr(model, "psd_scales", (16, 32, 64))))
    else:
        psd_scales = tuple(int(x) for x in tuple(psd_scales))
    if psd_frequency_band_channels is None:
        psd_frequency_band_channels = bool(getattr(model, "psd_frequency_band_channels", False))
    if psd_frequency_bands is None:
        psd_frequency_bands = tuple(
            (float(lo), float(hi)) for lo, hi in tuple(getattr(model, "psd_frequency_bands", tuple()))
        )
    else:
        psd_frequency_bands = tuple((float(lo), float(hi)) for lo, hi in tuple(psd_frequency_bands))
    if psd_frequency_band_include_full_spectrum is None:
        psd_frequency_band_include_full_spectrum = bool(
            getattr(model, "psd_frequency_band_include_full_spectrum", False)
        )
    if psd_frequency_band_include_anisotropy is None:
        psd_frequency_band_include_anisotropy = bool(
            getattr(model, "psd_frequency_band_include_anisotropy", False)
        )
    if psd_frequency_band_anisotropy_min_freq is None:
        psd_frequency_band_anisotropy_min_freq = float(
            getattr(model, "psd_frequency_band_anisotropy_min_freq", 0.0)
        )
    if psd_frequency_band_channels and psd_multiscale:
        raise ValueError("Frequency-band PSD mode is incompatible with multiscale PSD mode.")

    if str(stitch_output_space) in {"hybrid", "adaptive_hybrid"}:
        print(
            f"  Hybrid stitch blend: mode={stitch_output_space} "
            f"alpha_logit={stitch_output_blend_alpha:.3f} "
            f"preserve_threshold={stitch_output_preserve_threshold:.3f} "
            f"transition={stitch_output_transition:.3f}",
            flush=True,
        )
        prob_pred = predict_bad_regions_probability(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            pixel_size_angstrom=pixel_size_angstrom,
            use_power_spectrum=use_power_spectrum,
            include_real_space_input=include_real_space_input,
            use_pixel_size_channel=use_pixel_size_channel,
            pixel_size_channel_min_angstrom=pixel_size_channel_min_angstrom,
            pixel_size_channel_max_angstrom=pixel_size_channel_max_angstrom,
            use_hann_blending=use_hann_blending,
            blending_window=blending_window,
            blending_edge_px=blending_edge_px,
            stitch_phase_mode=stitch_phase_mode,
            stitch_phase_reduction=stitch_phase_reduction,
            stitch_phase_blend_alpha=stitch_phase_blend_alpha,
            stitch_output_space="probability",
            stitch_output_blend_alpha=stitch_output_blend_alpha,
            stitch_output_preserve_threshold=stitch_output_preserve_threshold,
            stitch_output_transition=stitch_output_transition,
            patch_prob_floor_quantile=patch_prob_floor_quantile,
            patch_prob_floor_strength=patch_prob_floor_strength,
            downsample_factor=downsample_factor,
            adaptive_downsample_method=adaptive_downsample_method,
            return_native_resolution=return_native_resolution,
            batch_forward_size=batch_forward_size,
            use_tta=use_tta,
            temperature=temperature,
            logit_bias=logit_bias,
            normalization_method=normalization_method,
            use_particle_size_downsampling=use_particle_size_downsampling,
            target_pixels_per_particle=target_pixels_per_particle,
            particle_size_angstrom=particle_size_angstrom,
            global_prior_path=global_prior_path,
            use_global_prior=use_global_prior,
            psd_use_radial_normalization=psd_use_radial_normalization,
            psd_multiscale=psd_multiscale,
            psd_multiscale_separate_channels=psd_multiscale_separate_channels,
            multiscale_psd_source=multiscale_psd_source,
            psd_scales=psd_scales,
            psd_frequency_band_channels=psd_frequency_band_channels,
            psd_frequency_bands=psd_frequency_bands,
            psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
            psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
            psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
            psd_texture_mode=psd_texture_mode,
            psd_highpass_cutoff=psd_highpass_cutoff,
            tile_offset_y=tile_offset_y,
            tile_offset_x=tile_offset_x,
        )
        logit_pred = predict_bad_regions_probability(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            pixel_size_angstrom=pixel_size_angstrom,
            use_power_spectrum=use_power_spectrum,
            include_real_space_input=include_real_space_input,
            use_pixel_size_channel=use_pixel_size_channel,
            pixel_size_channel_min_angstrom=pixel_size_channel_min_angstrom,
            pixel_size_channel_max_angstrom=pixel_size_channel_max_angstrom,
            use_hann_blending=use_hann_blending,
            blending_window=blending_window,
            blending_edge_px=blending_edge_px,
            stitch_phase_mode=stitch_phase_mode,
            stitch_phase_reduction=stitch_phase_reduction,
            stitch_phase_blend_alpha=stitch_phase_blend_alpha,
            stitch_output_space="logit",
            stitch_output_blend_alpha=stitch_output_blend_alpha,
            stitch_output_preserve_threshold=stitch_output_preserve_threshold,
            stitch_output_transition=stitch_output_transition,
            patch_prob_floor_quantile=patch_prob_floor_quantile,
            patch_prob_floor_strength=patch_prob_floor_strength,
            downsample_factor=downsample_factor,
            adaptive_downsample_method=adaptive_downsample_method,
            return_native_resolution=return_native_resolution,
            batch_forward_size=batch_forward_size,
            use_tta=use_tta,
            temperature=temperature,
            logit_bias=logit_bias,
            normalization_method=normalization_method,
            use_particle_size_downsampling=use_particle_size_downsampling,
            target_pixels_per_particle=target_pixels_per_particle,
            particle_size_angstrom=particle_size_angstrom,
            global_prior_path=global_prior_path,
            use_global_prior=use_global_prior,
            psd_use_radial_normalization=psd_use_radial_normalization,
            psd_multiscale=psd_multiscale,
            psd_multiscale_separate_channels=psd_multiscale_separate_channels,
            multiscale_psd_source=multiscale_psd_source,
            psd_scales=psd_scales,
            psd_frequency_band_channels=psd_frequency_band_channels,
            psd_frequency_bands=psd_frequency_bands,
            psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
            psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
            psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
            psd_texture_mode=psd_texture_mode,
            psd_highpass_cutoff=psd_highpass_cutoff,
            tile_offset_y=tile_offset_y,
            tile_offset_x=tile_offset_x,
        )
        prob_pred = np.asarray(prob_pred, dtype=np.float32)
        logit_pred = np.asarray(logit_pred, dtype=np.float32)
        if str(stitch_output_space) == "adaptive_hybrid":
            alpha_map = np.clip(
                (stitch_output_preserve_threshold - prob_pred) / max(stitch_output_transition, 1e-6),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            alpha_map = stitch_output_blend_alpha * alpha_map
            return (prob_pred + alpha_map * (logit_pred - prob_pred)).astype(np.float32, copy=False)
        return (prob_pred + stitch_output_blend_alpha * (logit_pred - prob_pred)).astype(np.float32, copy=False)

    if str(stitch_phase_mode) != "single":
        stride = patch_size - overlap
        if stride <= 0:
            raise ValueError("patch_size must be larger than overlap for shifted-grid stitching")
        phase_step = max(1, stride // 2)
        if str(stitch_phase_mode) == "diag_halfstride":
            phase_offsets = [(0, 0), (phase_step, phase_step)]
        else:
            phase_offsets = [
                (0, 0),
                (phase_step, 0),
                (0, phase_step),
                (phase_step, phase_step),
            ]
        print(
            f"  Stitch phase ensemble: mode={stitch_phase_mode} "
            f"patch_size={patch_size} overlap={overlap} phase_step={phase_step} "
            f"n_phases={len(phase_offsets)} reduction={stitch_phase_reduction} "
            f"blend_alpha={stitch_phase_blend_alpha:.3f}",
            flush=True,
        )
        phase_predictions = []
        for phase_idx, (phase_y, phase_x) in enumerate(phase_offsets, start=1):
            print(
                f"    Phase {phase_idx}/{len(phase_offsets)} offset=({phase_y}, {phase_x})",
                flush=True,
            )
            phase_pred = predict_bad_regions_probability(
                model=model,
                image=image,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
                pixel_size_angstrom=pixel_size_angstrom,
                use_power_spectrum=use_power_spectrum,
                include_real_space_input=include_real_space_input,
                use_pixel_size_channel=use_pixel_size_channel,
                pixel_size_channel_min_angstrom=pixel_size_channel_min_angstrom,
                pixel_size_channel_max_angstrom=pixel_size_channel_max_angstrom,
                use_hann_blending=use_hann_blending,
                blending_window=blending_window,
                blending_edge_px=blending_edge_px,
                stitch_phase_mode="single",
                stitch_phase_reduction=stitch_phase_reduction,
                stitch_phase_blend_alpha=stitch_phase_blend_alpha,
                stitch_output_space=stitch_output_space,
                stitch_output_blend_alpha=stitch_output_blend_alpha,
                stitch_output_preserve_threshold=stitch_output_preserve_threshold,
                stitch_output_transition=stitch_output_transition,
                patch_prob_floor_quantile=patch_prob_floor_quantile,
                patch_prob_floor_strength=patch_prob_floor_strength,
                downsample_factor=downsample_factor,
                adaptive_downsample_method=adaptive_downsample_method,
                return_native_resolution=return_native_resolution,
                batch_forward_size=batch_forward_size,
                use_tta=use_tta,
                temperature=temperature,
                logit_bias=logit_bias,
                normalization_method=normalization_method,
                use_particle_size_downsampling=use_particle_size_downsampling,
                target_pixels_per_particle=target_pixels_per_particle,
                particle_size_angstrom=particle_size_angstrom,
                global_prior_path=global_prior_path,
                use_global_prior=use_global_prior,
                psd_use_radial_normalization=psd_use_radial_normalization,
                psd_multiscale=psd_multiscale,
                psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                multiscale_psd_source=multiscale_psd_source,
                psd_scales=psd_scales,
                psd_frequency_band_channels=psd_frequency_band_channels,
                psd_frequency_bands=psd_frequency_bands,
                psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
                psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
                psd_texture_mode=psd_texture_mode,
                psd_highpass_cutoff=psd_highpass_cutoff,
                tile_offset_y=phase_y,
                tile_offset_x=phase_x,
            )
            phase_predictions.append(np.asarray(phase_pred, dtype=np.float32))
        phase_stack = np.stack(phase_predictions, axis=0).astype(np.float32, copy=False)
        if str(stitch_phase_reduction) == "median":
            combined = np.median(phase_stack, axis=0).astype(np.float32, copy=False)
        elif str(stitch_phase_reduction) == "max":
            combined = np.max(phase_stack, axis=0).astype(np.float32, copy=False)
        else:
            combined = np.mean(phase_stack, axis=0).astype(np.float32, copy=False)
        if stitch_phase_blend_alpha < 1.0:
            base_pred = phase_stack[0]
            combined = (
                base_pred + stitch_phase_blend_alpha * (combined - base_pred)
            ).astype(np.float32, copy=False)
        return combined

    # Global Prior Learning: Load global prior if provided
    global_prior = None
    if use_global_prior and global_prior_path is not None:
        try:
            from utils.global_prior import load_global_prior, get_expected_prevalence
            global_prior = load_global_prior(Path(global_prior_path))
            expected_prevalence = get_expected_prevalence(global_prior, method='median')
            print(f"  Global prior loaded: expected contamination = {expected_prevalence:.4f} (median)", flush=True)
        except Exception as e:
            print(f"  Warning: Could not load global prior from {global_prior_path}: {e}", flush=True)
            global_prior = None

    # MC Integration: Normalize image using specified method
    from utils.image_utils import normalize_image
    image = normalize_image(image, method=normalization_method)

    # Global Prior: Estimate current image contamination prevalence (for prevalence correction)
    # Use low-frequency intensity to estimate support vs. hole (darker = more contamination)
    estimated_prevalence = None
    if global_prior is not None:
        # Estimate contamination from image intensity (darker regions = more contamination)
        # This is a simple heuristic: lower intensity suggests more support/contamination
        intensity_mean = float(image.mean())
        intensity_std = float(image.std())

        # Use global prior intensity statistics to normalize
        if 'intensity_statistics' in global_prior:
            train_intensity_mean = global_prior['intensity_statistics'].get('mean', 0.5)
            # If current image is darker than training mean, estimate higher contamination
            # Simple linear mapping: darker = more contamination
            intensity_diff = train_intensity_mean - intensity_mean
            # Map intensity difference to prevalence estimate (heuristic)
            # If image is 0.1 darker than training mean, estimate +0.05 contamination
            estimated_prevalence = expected_prevalence + (intensity_diff * 0.5)  # Scale factor
            estimated_prevalence = max(0.0, min(1.0, estimated_prevalence))  # Clamp to [0, 1]
        else:
            # Fallback: use expected prevalence from training distribution
            estimated_prevalence = expected_prevalence

    # MC Integration: Particle-size based downsampling (applied before fast mode downsampling)
    downsampled = False
    scale_factor = 1.0
    new_pixel_size = pixel_size_angstrom
    pixel_size_for_model = float(pixel_size_angstrom)
    if use_particle_size_downsampling:
        from utils.image_utils import downsample_to_particle_size
        image, scale_factor, new_pixel_size = downsample_to_particle_size(
            image, pixel_size_angstrom=pixel_size_angstrom,
            target_pixels_per_particle=target_pixels_per_particle,
            particle_size_angstrom=particle_size_angstrom
        )
        downsampled = (scale_factor != 1.0)
        if downsampled:
            print(f"  Particle-size downsampling: {original_h}x{original_w} @ {pixel_size_angstrom:.2f} Å/px → {image.shape[0]}x{image.shape[1]} @ {new_pixel_size:.2f} Å/px (factor={scale_factor:.3f})", flush=True)
            # Update pixel size for power spectrum computation
            pixel_size_angstrom = new_pixel_size
            pixel_size_for_model = float(new_pixel_size)
            original_h, original_w = image.shape

    # Item 5: Optional downsample inference mode (fast mode) - applied after particle-size downsampling
    if downsample_factor > 1:
        factor = float(max(1.0, downsample_factor))
        if str(adaptive_downsample_method) == "fourier":
            image = fourier_rescale_2d(np.asarray(image, dtype=np.float32), scale=1.0 / factor)
        else:
            from scipy.ndimage import zoom

            down_h = max(1, int(round(original_h / factor)))
            down_w = max(1, int(round(original_w / factor)))
            image = zoom(image, (down_h / original_h, down_w / original_w), order=1)
        h, w = image.shape
        print(
            f"  Fast mode: Downsampled from {original_h}x{original_w} to {h}x{w} "
            f"(factor={factor:.3f}, method={adaptive_downsample_method})",
            flush=True,
        )
        pixel_size_for_model = float(pixel_size_for_model) * factor
    else:
        h, w = original_h, original_w

    # Compute power spectrum if needed (on potentially downsampled image)
    if use_power_spectrum and not include_real_space_input:
        pass  # Explicit PSD-only mode
    elif (not use_power_spectrum) and (not include_real_space_input):
        raise ValueError("Invalid input mode: include_real_space_input=False requires use_power_spectrum=True")

    if multiscale_psd_source not in {"patch", "full_image"}:
        raise ValueError("multiscale_psd_source must be 'patch' or 'full_image'")

    # NOTE: The training-matched behavior is patch-local PSD. We also support full-image
    # PSD as an opt-in diagnostic path to test whether patch-local frequency normalization
    # is creating grid artifacts on atypical datasets.
    power_spectrum = None
    use_per_patch_psd = str(multiscale_psd_source) == "patch"
    compute_multiscale_psd_single = None
    compute_multiscale_psd_batch = None
    compute_multiscale_psd_channels_single = None
    if use_power_spectrum and use_per_patch_psd and psd_multiscale:
        compute_multiscale_psd_single = compute_multiscale_radial_psd_image
        compute_multiscale_psd_batch = compute_multiscale_radial_psd_image_batch
        compute_multiscale_psd_channels_single = compute_multiscale_radial_psd_channels
    if use_power_spectrum and psd_frequency_band_channels and (not use_per_patch_psd):
        power_spectrum = compute_frequency_band_psd_channels(
            image,
            pixel_size_angstrom=float(pixel_size_for_model),
            frequency_bands=tuple(psd_frequency_bands),
            normalize=True,
            use_radial_normalization=bool(psd_use_radial_normalization),
            include_full_spectrum_channel=bool(psd_frequency_band_include_full_spectrum),
            include_anisotropy_channel=bool(psd_frequency_band_include_anisotropy),
            anisotropy_min_frequency=float(psd_frequency_band_anisotropy_min_freq),
        )
    elif use_power_spectrum and psd_multiscale and (not use_per_patch_psd):
        if psd_multiscale_separate_channels:
            power_spectrum = compute_multiscale_radial_psd_channels(
                image,
                scales=psd_scales,
                normalize=True,
                texture_mode=psd_texture_mode,
                highpass_cutoff=psd_highpass_cutoff,
            )
        else:
            power_spectrum = compute_multiscale_radial_psd_image(
                image,
                scales=psd_scales,
                normalize=True,
                texture_mode=psd_texture_mode,
                highpass_cutoff=psd_highpass_cutoff,
            )
    elif use_power_spectrum and not use_per_patch_psd:
        # Single-scale PSD: compute once for full image
        # Note: Fingerprint verification should be done at dataset level, not per-inference
        # to avoid performance impact. See evaluate_for_fig1.py for fingerprint checks.
        power_spectrum = compute_power_spectrum(
            image,
            normalize=True,
            use_radial_normalization=bool(psd_use_radial_normalization),
        )

    # Create output mask (at potentially downsampled resolution)
    output_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)
    use_patch_prob_floor_correction = (
        str(stitch_output_space) == "probability"
        and patch_prob_floor_quantile > 0.0
        and patch_prob_floor_strength > 0.0
    )
    prob_floor_accum = np.zeros((h, w), dtype=np.float32) if use_patch_prob_floor_correction else None

    # Item 5: Create a blend window for overlapping patch stitching.
    blend_window = None
    if use_hann_blending:
        if str(blending_window) == "tukey":
            edge_px = int(max(0, blending_edge_px))

            def _tukey_1d(size: int, edge: int) -> np.ndarray:
                if edge <= 0:
                    return np.ones(size, dtype=np.float32)
                if edge >= size // 2:
                    return np.hanning(size).astype(np.float32)
                window = np.ones(size, dtype=np.float32)
                taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge, dtype=np.float32) / float(edge)))
                window[:edge] = taper
                window[-edge:] = taper[::-1]
                return window

            blend_window = np.outer(
                _tukey_1d(patch_size, edge_px),
                _tukey_1d(patch_size, edge_px),
            ).astype(np.float32)
        else:
            # Create 2D Hann window (separable: outer product of 1D Hann windows)
            hann_1d_h = np.hanning(patch_size)
            hann_1d_w = np.hanning(patch_size)
            blend_window = np.outer(hann_1d_h, hann_1d_w).astype(np.float32)

            # PHASE 1 FIX: Normalize by expected overlap to get weight-sum ≈1.0 in interior.
            # The final division by count_mask cancels global scaling, but we keep the
            # legacy Hann normalization path unchanged for backward compatibility.
            stride = patch_size - overlap
            overlap_ratio = overlap / patch_size
            if overlap_ratio >= 0.75:
                expected_weight_sum = 4.0
            elif overlap_ratio >= 0.5:
                expected_weight_sum = overlap_ratio * 4.0 / 0.75
            else:
                expected_weight_sum = max(1.0, overlap_ratio * 2.0)
            blend_window = blend_window / expected_weight_sum

        if np.max(blend_window) > 0:
            blend_window = blend_window / np.max(blend_window)

    stride = patch_size - overlap

    def _axis_positions(length: int, offset: int) -> list:
        if length <= 0:
            return [0]
        if offset <= 0:
            return list(range(0, length, stride))
        positions = [0]
        positions.extend(range(int(offset), length, stride))
        return sorted(set(int(p) for p in positions if 0 <= int(p) < length))

    y_positions = _axis_positions(h, tile_offset_y)
    x_positions = _axis_positions(w, tile_offset_x)

    # Calculate total number of patches for progress tracking
    total_patches = max(1, len(y_positions) * len(x_positions))
    patch_count = 0

    # Item 4: Mixed precision inference (only for CUDA)
    use_amp = device == "cuda" and torch.cuda.is_available()
    batch_forward_size = int(max(1, batch_forward_size))
    use_batched_forward = batch_forward_size > 1
    if global_prior is not None and estimated_prevalence is not None and use_batched_forward:
        # Global prior correction currently converts patch logits to NumPy and applies
        # per-patch prevalence adjustment; keep the existing sequential path to avoid
        # changing calibrated behavior until a batched version is validated.
        print("  Note: batch_forward_size disabled because global prior correction is active", flush=True)
        use_batched_forward = False

    def _extract_full_image_psd_patch(y: int, y_end: int, x: int, x_end: int):
        if power_spectrum is None:
            return np.zeros((patch_size, patch_size), dtype=np.float32)

        if power_spectrum.ndim == 3:
            patch_ps = power_spectrum[:, y:y_end, x:x_end]
            actual_h = patch_ps.shape[1]
            actual_w = patch_ps.shape[2]
            if actual_h != patch_size or actual_w != patch_size:
                pad_h = patch_size - actual_h
                pad_w = patch_size - actual_w
                if actual_h > 0 and actual_w > 0:
                    patch_ps = np.pad(patch_ps, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                else:
                    patch_ps = np.zeros((power_spectrum.shape[0], patch_size, patch_size), dtype=np.float32)
            return np.ascontiguousarray(np.asarray(patch_ps, dtype=np.float32), dtype=np.float32)

        patch_ps = power_spectrum[y:y_end, x:x_end]
        actual_h = patch_ps.shape[0]
        actual_w = patch_ps.shape[1]
        if actual_h != patch_size or actual_w != patch_size:
            pad_h = patch_size - actual_h
            pad_w = patch_size - actual_w
            if actual_h > 0 and actual_w > 0:
                patch_ps = np.pad(patch_ps, ((0, pad_h), (0, pad_w)), mode='reflect')
            else:
                patch_ps = np.zeros((patch_size, patch_size), dtype=np.float32)
        return np.ascontiguousarray(np.asarray(patch_ps, dtype=np.float32), dtype=np.float32)

    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32, copy=False)

    def _bad_prob_from_logits_torch(logits_t: torch.Tensor) -> torch.Tensor:
        probs_t = torch.softmax(logits_t, dim=1)
        ncls = probs_t.shape[1]
        if ncls == 1:
            return torch.sigmoid(logits_t[:, 0])
        if ncls > 2:
            return probs_t[:, 1:].sum(dim=1)
        return probs_t[:, 1]

    def _bad_margin_from_logits_torch(logits_t: torch.Tensor) -> torch.Tensor:
        ncls = logits_t.shape[1]
        if ncls == 1:
            return logits_t[:, 0]
        if ncls == 2:
            return logits_t[:, 1] - logits_t[:, 0]
        return torch.logsumexp(logits_t[:, 1:, :, :], dim=1) - logits_t[:, 0, :, :]

    def _stitch_field_from_logits_torch(logits_t: torch.Tensor) -> torch.Tensor:
        if logits_t.ndim == 3:
            logits_t = logits_t.unsqueeze(1)
        if str(stitch_output_space) == "logit":
            return _bad_margin_from_logits_torch(logits_t)
        return _bad_prob_from_logits_torch(logits_t)

    def _estimate_patch_prob_floor(field_np: np.ndarray) -> float:
        if not use_patch_prob_floor_correction:
            return 0.0
        field = np.clip(np.asarray(field_np, dtype=np.float32), 0.0, 1.0)
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            return 0.0
        return float(np.quantile(finite, patch_prob_floor_quantile))

    def _pixel_size_channel_for_patch(shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not use_pixel_size_channel:
            return None
        return build_constant_pixel_size_channel(
            image_shape=shape,
            pixel_size_angstrom=float(pixel_size_for_model),
            min_angstrom=float(pixel_size_channel_min_angstrom),
            max_angstrom=float(pixel_size_channel_max_angstrom),
            use_log_scale=True,
        )

    def _stack_patch_inputs(patch_img: np.ndarray, patch_ps: Optional[np.ndarray]) -> np.ndarray:
        patch_img = np.ascontiguousarray(np.asarray(patch_img, dtype=np.float32), dtype=np.float32)
        pixel_size_channel_np = _pixel_size_channel_for_patch(patch_img.shape)
        components = []
        if include_real_space_input:
            components.append(patch_img[np.newaxis])
        if use_power_spectrum:
            if patch_ps is None:
                raise ValueError("use_power_spectrum=True requires a PSD patch.")
            patch_ps = np.ascontiguousarray(np.asarray(patch_ps, dtype=np.float32), dtype=np.float32)
            if patch_ps.ndim == 2:
                components.append(patch_ps[np.newaxis])
            elif patch_ps.ndim == 3:
                components.append(patch_ps)
            else:
                raise ValueError(f"Unsupported PSD ndim: {patch_ps.ndim}")
        if pixel_size_channel_np is not None:
            components.append(pixel_size_channel_np[np.newaxis])
        if not components:
            raise ValueError("No input channels enabled for inference.")
        return np.ascontiguousarray(np.concatenate(components, axis=0), dtype=np.float32)

    with torch.inference_mode():  # More efficient than torch.no_grad() for inference
        # Use autocast for mixed precision when on CUDA
        autocast_context = torch.amp.autocast('cuda', enabled=use_amp) if use_amp else torch.amp.autocast('cpu', enabled=False)

        with autocast_context:
            if use_batched_forward:
                patch_buf = []
                patch_ps_buf = []
                meta_buf = []

                def _extract_stitch_field_batch(output_obj):
                    if isinstance(output_obj, tuple) and len(output_obj) >= 2:
                        # Match existing inference behavior: use multiclass head only.
                        logits_b = output_obj[0].float()
                    else:
                        logits_b = output_obj.float()

                    if logits_b.ndim == 3:
                        logits_b = logits_b.unsqueeze(1)

                    if temperature != 1.0:
                        logits_b = logits_b / temperature
                    if logit_bias != 0.0 and logits_b.shape[1] > 1:
                        logits_b[:, 1:, :, :] = logits_b[:, 1:, :, :] + logit_bias

                    return _stitch_field_from_logits_torch(logits_b).detach().cpu().numpy()

                def _flush_batch() -> None:
                    if not patch_buf:
                        return
                    patch_ps_for_stack = patch_ps_buf
                    if (
                        use_power_spectrum
                        and use_per_patch_psd
                        and psd_multiscale
                        and not psd_multiscale_separate_channels
                    ):
                        if compute_multiscale_psd_batch is None:
                            raise RuntimeError("Batched multiscale PSD helper not initialized")
                        patch_ps_for_stack = compute_multiscale_psd_batch(
                            np.ascontiguousarray(np.stack(patch_buf, axis=0), dtype=np.float32),
                            scales=psd_scales,
                            normalize=True,
                            texture_mode=psd_texture_mode,
                            highpass_cutoff=psd_highpass_cutoff,
                        )
                        expected_shape = (len(patch_buf), patch_size, patch_size)
                        if patch_ps_for_stack.shape != expected_shape:
                            raise ValueError(
                                f"Batched patch PSD shape {patch_ps_for_stack.shape} != {expected_shape}"
                            )
                    stacked = np.ascontiguousarray(
                        np.stack(
                            [
                                _stack_patch_inputs(
                                    patch_buf[i],
                                    patch_ps_for_stack[i] if use_power_spectrum else None,
                                )
                                for i in range(len(patch_buf))
                            ],
                            axis=0,
                        ),
                        dtype=np.float32,
                    )

                    patch_tensor_b = torch.from_numpy(stacked).to(device)
                    try:
                        output_b = model(patch_tensor_b)
                    except RuntimeError as e:
                        if "Sizes of tensors must match" in str(e):
                            raise ValueError(
                                f"Model incompatible with patch_size={patch_size}. "
                                "Use a patch size divisible by 16 "
                                f"(for example, 128 or 256) instead of {patch_size}."
                            ) from e
                        raise
                    stitch_field_batch = _extract_stitch_field_batch(output_b)

                    for i_meta, (y, y_end, x, x_end, actual_h, actual_w) in enumerate(meta_buf):
                        stitch_field = np.asarray(stitch_field_batch[i_meta], dtype=np.float32)
                        if stitch_field.shape[0] < actual_h or stitch_field.shape[1] < actual_w:
                            from scipy.ndimage import zoom
                            zoom_h = actual_h / stitch_field.shape[0]
                            zoom_w = actual_w / stitch_field.shape[1]
                            stitch_field = zoom(stitch_field, (zoom_h, zoom_w), order=1, mode='nearest')
                        stitch_field_cropped = np.asarray(stitch_field[:actual_h, :actual_w], dtype=np.float32)
                        patch_floor_value = _estimate_patch_prob_floor(stitch_field_cropped)

                        if use_hann_blending and blend_window is not None:
                            blend_cropped = blend_window[:actual_h, :actual_w]
                            output_mask[y:y_end, x:x_end] += stitch_field_cropped * blend_cropped
                            count_mask[y:y_end, x:x_end] += blend_cropped
                            if prob_floor_accum is not None:
                                prob_floor_accum[y:y_end, x:x_end] += patch_floor_value * blend_cropped
                        else:
                            output_mask[y:y_end, x:x_end] += stitch_field_cropped
                            count_mask[y:y_end, x:x_end] += 1.0
                            if prob_floor_accum is not None:
                                prob_floor_accum[y:y_end, x:x_end] += patch_floor_value

                    patch_buf.clear()
                    patch_ps_buf.clear()
                    meta_buf.clear()

                for y in y_positions:
                    for x in x_positions:
                        patch_count += 1
                        if patch_count % 50 == 0:
                            print(f"    Processing patch {patch_count}/{total_patches}...", flush=True)

                        y_end = min(y + patch_size, h)
                        x_end = min(x + patch_size, w)
                        actual_h, actual_w = y_end - y, x_end - x

                        patch = image[y:y_end, x:x_end]
                        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                            pad_h = patch_size - patch.shape[0]
                            pad_w = patch_size - patch.shape[1]
                            if patch.shape[0] > 0 and patch.shape[1] > 0:
                                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')
                            else:
                                patch = np.zeros((patch_size, patch_size), dtype=patch.dtype)
                        if patch.shape != (patch_size, patch_size):
                            raise ValueError(f"Patch shape {patch.shape} != ({patch_size}, {patch_size})")

                        patch = np.ascontiguousarray(np.asarray(patch, dtype=np.float32), dtype=np.float32)

                        if use_power_spectrum:
                            if use_per_patch_psd and psd_frequency_band_channels:
                                patch_ps = compute_frequency_band_psd_channels(
                                    patch,
                                    pixel_size_angstrom=float(pixel_size_for_model),
                                    frequency_bands=tuple(psd_frequency_bands),
                                    normalize=True,
                                    use_radial_normalization=bool(psd_use_radial_normalization),
                                    include_full_spectrum_channel=bool(psd_frequency_band_include_full_spectrum),
                                    include_anisotropy_channel=bool(psd_frequency_band_include_anisotropy),
                                    anisotropy_min_frequency=float(psd_frequency_band_anisotropy_min_freq),
                                )
                                patch_ps = np.ascontiguousarray(np.asarray(patch_ps, dtype=np.float32), dtype=np.float32)
                                patch_buf.append(patch)
                                patch_ps_buf.append(patch_ps)
                            elif use_per_patch_psd and psd_multiscale:
                                if psd_multiscale_separate_channels:
                                    if compute_multiscale_psd_channels_single is None:
                                        raise RuntimeError("Multiscale PSD channels helper not initialized")
                                    patch_ps = compute_multiscale_psd_channels_single(
                                        patch,
                                        scales=psd_scales,
                                        normalize=True,
                                        texture_mode=psd_texture_mode,
                                        highpass_cutoff=psd_highpass_cutoff,
                                    )
                                    if patch_ps.ndim != 3 or patch_ps.shape[1:] != (patch_size, patch_size):
                                        raise ValueError(
                                            f"Patch PSD channels shape {patch_ps.shape} != "
                                            f"(S, {patch_size}, {patch_size})"
                                        )
                                else:
                                    patch_ps = None
                                if patch_ps is not None:
                                    patch_ps = np.ascontiguousarray(np.asarray(patch_ps, dtype=np.float32), dtype=np.float32)
                                patch_buf.append(patch)
                                patch_ps_buf.append(patch_ps)
                            elif use_per_patch_psd:
                                patch_ps = compute_power_spectrum(
                                    patch,
                                    normalize=True,
                                    use_radial_normalization=bool(psd_use_radial_normalization),
                                )
                                if patch_ps.shape != (patch_size, patch_size):
                                    raise ValueError(f"Patch PS shape {patch_ps.shape} != ({patch_size}, {patch_size})")
                                patch_ps = np.ascontiguousarray(np.asarray(patch_ps, dtype=np.float32), dtype=np.float32)
                                patch_buf.append(patch)
                                patch_ps_buf.append(patch_ps)
                            else:
                                patch_ps = _extract_full_image_psd_patch(y, y_end, x, x_end)
                                if patch_ps.ndim == 2 and patch_ps.shape != (patch_size, patch_size):
                                    raise ValueError(f"Patch PS shape {patch_ps.shape} != ({patch_size}, {patch_size})")
                                if patch_ps.ndim == 3 and patch_ps.shape[1:] != (patch_size, patch_size):
                                    raise ValueError(
                                        f"Patch PSD channels shape {patch_ps.shape} != "
                                        f"(S, {patch_size}, {patch_size})"
                                    )
                                patch_buf.append(patch)
                                patch_ps_buf.append(patch_ps)
                        else:
                            patch_buf.append(patch)

                        meta_buf.append((y, y_end, x, x_end, actual_h, actual_w))

                        if len(patch_buf) >= batch_forward_size:
                            _flush_batch()

                _flush_batch()
            else:
                for y in y_positions:
                    for x in x_positions:
                        patch_count += 1
                        if patch_count % 50 == 0:
                            print(f"    Processing patch {patch_count}/{total_patches}...", flush=True)

                        # Extract patch
                        y_end = min(y + patch_size, h)
                        x_end = min(x + patch_size, w)

                        patch = image[y:y_end, x:x_end]

                        # Always pad to exactly patch_size x patch_size
                        # Use REFLECT padding to create smoother, more natural transitions at image edges
                        # This is superior to edge replication which can create visible seams
                        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                            pad_h = patch_size - patch.shape[0]
                            pad_w = patch_size - patch.shape[1]
                            if patch.shape[0] > 0 and patch.shape[1] > 0:
                                # Reflect padding creates mirrored content at boundaries
                                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')
                            else:
                                # Fallback for empty patches
                                patch = np.zeros((patch_size, patch_size), dtype=patch.dtype)

                        # Verify patch shape
                        if patch.shape != (patch_size, patch_size):
                            raise ValueError(f"Patch shape {patch.shape} != ({patch_size}, {patch_size})")
                        patch = np.ascontiguousarray(np.asarray(patch, dtype=np.float32), dtype=np.float32)

                        # Stack with power spectrum if needed
                        if use_power_spectrum:
                            if use_per_patch_psd and psd_frequency_band_channels:
                                patch_ps = compute_frequency_band_psd_channels(
                                    patch,
                                    pixel_size_angstrom=float(pixel_size_for_model),
                                    frequency_bands=tuple(psd_frequency_bands),
                                    normalize=True,
                                    use_radial_normalization=bool(psd_use_radial_normalization),
                                    include_full_spectrum_channel=bool(psd_frequency_band_include_full_spectrum),
                                    include_anisotropy_channel=bool(psd_frequency_band_include_anisotropy),
                                    anisotropy_min_frequency=float(psd_frequency_band_anisotropy_min_freq),
                                )
                            elif use_per_patch_psd and psd_multiscale:
                                if psd_multiscale_separate_channels:
                                    if compute_multiscale_psd_channels_single is None:
                                        raise RuntimeError("Multiscale PSD channels helper not initialized")
                                    patch_ps = compute_multiscale_psd_channels_single(
                                        patch,
                                        scales=psd_scales,
                                        normalize=True,
                                        texture_mode=psd_texture_mode,
                                        highpass_cutoff=psd_highpass_cutoff,
                                    )
                                else:
                                    # Multi-scale PSD: compute per-patch (must match training!)
                                    if compute_multiscale_psd_single is None:
                                        raise RuntimeError("Multiscale PSD helper not initialized")
                                    patch_ps = compute_multiscale_psd_single(
                                        patch, scales=psd_scales, normalize=True,
                                        texture_mode=psd_texture_mode, highpass_cutoff=psd_highpass_cutoff
                                    )
                            elif use_per_patch_psd:
                                patch_ps = compute_power_spectrum(
                                    patch,
                                    normalize=True,
                                    use_radial_normalization=bool(psd_use_radial_normalization),
                                )
                            elif power_spectrum is not None:
                                # Single-scale PSD or optional full-image multiscale PSD:
                                # extract from pre-computed full-image frequency map(s).
                                patch_ps = _extract_full_image_psd_patch(y, y_end, x, x_end)
                            else:
                                # Fallback: zeros
                                patch_ps = np.zeros((patch_size, patch_size), dtype=np.float32)

                            # Ensure arrays are proper numpy arrays (not masked arrays or other types)
                            patch = np.asarray(patch, dtype=np.float32)
                            patch_ps = np.asarray(patch_ps, dtype=np.float32)
                            stacked = _stack_patch_inputs(patch, patch_ps)
                            # Convert to tensor - NumPy 1.26.4 works fine with torch.from_numpy()
                            patch_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
                        else:
                            stacked = _stack_patch_inputs(patch, None)
                            patch_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)

                        # Predict with mixed precision (autocast handles fp16/fp32 conversion)
                        try:
                            output = model(patch_tensor)
                        except RuntimeError as e:
                            if "Sizes of tensors must match" in str(e):
                                raise ValueError(
                                    f"Model incompatible with patch_size={patch_size}. "
                                    "Use a patch size divisible by 16 "
                                    f"(for example, 128 or 256) instead of {patch_size}."
                                ) from e
                            raise

                        # Handle dual-head models (multiclass + binary + optional particle) or single-head models
                        if isinstance(output, tuple) and len(output) >= 2:
                            # Dual-head model: (multiclass_logits, binary_logits) or (multiclass_logits, binary_logits, particle_logits)
                            # CRITICAL: Use multiclass_logits for inference to match training!
                            # Training optimizes multiclass_logits, so binary_logits is untrained.
                            multiclass_logits = output[0]
                            binary_logits = output[1]
                            # particle_logits = output[2] if len(output) >= 3 else None  # Not used for inference

                            # PHASE 2: Apply calibration (temperature scaling and logit bias)
                            logits = multiclass_logits.float()
                            if temperature != 1.0:
                                logits = logits / temperature
                            if logit_bias != 0.0:
                                # Apply bias to bad class logits (class 1, 2, 3, ...)
                                # For binary: bias on class 1
                                # For multi-class: bias on classes 1, 2, 3, ...
                                if logits.shape[1] > 1:
                                    # Apply bias to all non-good classes (classes 1, 2, 3, ...)
                                    logits[:, 1:, :, :] = logits[:, 1:, :, :] + logit_bias

                            # Global Prior Learning: Apply prevalence correction if global prior is available
                            # This adjusts logits based on expected vs. estimated contamination prevalence
                            # Integration with Fourier branch: Uses global prior to guide interpretation of PSD features
                            if global_prior is not None and estimated_prevalence is not None:
                                from utils.calibration import compute_logit_adjustment
                                # Expected prevalence from training distribution (from global prior)
                                train_prevalence = expected_prevalence
                                # Estimated prevalence for current image (from intensity, can be enhanced with PSD)
                                true_prevalence = estimated_prevalence

                                # Apply logit adjustment: adjust bad class logits based on prevalence mismatch
                                # Formula: logit_bad = logit_bad + log(π_true / (1-π_true)) - log(π_train / (1-π_train))
                                # This corrects for mismatch between training and inference contamination levels
                                logits_np = logits[0].cpu().numpy()  # (C, H, W)
                                if logits_np.shape[0] > 1:
                                    # Apply adjustment to bad class logits (class 1, 2, 3, ...)
                                    # For multi-class, adjust all bad classes uniformly
                                    for bad_class_idx in range(1, logits_np.shape[0]):
                                        bad_logits = logits_np[bad_class_idx, :, :]  # (H, W)
                                        adjusted_bad_logits = compute_logit_adjustment(
                                            bad_logits, train_prevalence, true_prevalence, eps=1e-10
                                        )
                                        logits_np[bad_class_idx, :, :] = adjusted_bad_logits
                                    logits = torch.from_numpy(logits_np).float().unsqueeze(0).to(device)

                            # Use multiclass head (matches training loss)
                            stitch_field = _stitch_field_from_logits_torch(logits)[0].detach().cpu().numpy()
                        else:
                            # Single-head model: old behavior
                            # PHASE 2: Apply calibration (temperature scaling and logit bias)
                            logits = output.float()
                            if temperature != 1.0:
                                logits = logits / temperature
                            if logit_bias != 0.0:
                                # Apply bias to bad class logits (class 1, 2, 3, ...)
                                if logits.shape[1] > 1:
                                    # Apply bias to all non-good classes (classes 1, 2, 3, ...)
                                    logits[:, 1:, :, :] = logits[:, 1:, :, :] + logit_bias

                            # Global Prior Learning: Apply prevalence correction if global prior is available
                            if global_prior is not None and estimated_prevalence is not None:
                                from utils.calibration import compute_logit_adjustment
                                train_prevalence = expected_prevalence
                                true_prevalence = estimated_prevalence

                                logits_np = logits[0].cpu().numpy()  # (C, H, W)
                                if logits_np.shape[0] > 1:
                                    # Apply adjustment to bad class logits (class 1, 2, 3, ...)
                                    for bad_class_idx in range(1, logits_np.shape[0]):
                                        bad_logits = logits_np[bad_class_idx, :, :]  # (H, W)
                                        adjusted_bad_logits = compute_logit_adjustment(
                                            bad_logits, train_prevalence, true_prevalence, eps=1e-10
                                        )
                                        logits_np[bad_class_idx, :, :] = adjusted_bad_logits
                                    logits = torch.from_numpy(logits_np).float().unsqueeze(0).to(device)

                            stitch_field = _stitch_field_from_logits_torch(logits)[0].detach().cpu().numpy()

                        # Ensure it's a proper numpy array (not a tensor or other type)
                        stitch_field = np.asarray(stitch_field, dtype=np.float32)

                        # Extract actual patch region (may be smaller than patch_size at edges)
                        actual_h, actual_w = y_end - y, x_end - x

                        # Model output might be smaller than input due to pooling - handle this
                        if stitch_field.shape[0] < actual_h or stitch_field.shape[1] < actual_w:
                            # Upsample model output to match expected size
                            from scipy.ndimage import zoom
                            zoom_h = actual_h / stitch_field.shape[0]
                            zoom_w = actual_w / stitch_field.shape[1]
                            stitch_field = zoom(stitch_field, (zoom_h, zoom_w), order=1, mode='nearest')

                        stitch_field_cropped = stitch_field[:actual_h, :actual_w]
                        # Ensure cropped version is also a proper numpy array
                        stitch_field_cropped = np.asarray(stitch_field_cropped, dtype=np.float32)
                        patch_floor_value = _estimate_patch_prob_floor(stitch_field_cropped)

                        # Item 5: Apply Hann window blending if enabled
                        if use_hann_blending and blend_window is not None:
                            # Crop blend window to match actual patch size
                            blend_cropped = blend_window[:actual_h, :actual_w]
                            # Weighted accumulation: prob * weight
                            output_mask[y:y_end, x:x_end] += stitch_field_cropped * blend_cropped
                            count_mask[y:y_end, x:x_end] += blend_cropped
                            if prob_floor_accum is not None:
                                prob_floor_accum[y:y_end, x:x_end] += patch_floor_value * blend_cropped
                        else:
                            # Simple averaging (original method)
                            output_mask[y:y_end, x:x_end] += stitch_field_cropped
                            count_mask[y:y_end, x:x_end] += 1.0
                            if prob_floor_accum is not None:
                                prob_floor_accum[y:y_end, x:x_end] += patch_floor_value

    # Average overlapping regions
    output_mask = output_mask / (count_mask + 1e-10)
    if prob_floor_accum is not None:
        floor_map = prob_floor_accum / (count_mask + 1e-10)
        finite_floor = floor_map[np.isfinite(floor_map)]
        if finite_floor.size > 0:
            reference_floor = float(np.median(finite_floor))
            floor_excess = np.clip(floor_map - reference_floor, 0.0, None).astype(np.float32, copy=False)
            preserve_alpha = np.clip(
                (stitch_output_preserve_threshold - output_mask) / max(stitch_output_transition, 1e-6),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            floor_correction = (patch_prob_floor_strength * floor_excess * preserve_alpha).astype(np.float32, copy=False)
            output_mask = np.clip(output_mask - floor_correction, 0.0, 1.0).astype(np.float32, copy=False)
            print(
                "  Patch baseline drift correction: "
                f"q={patch_prob_floor_quantile:.3f} strength={patch_prob_floor_strength:.3f} "
                f"median_floor={reference_floor:.4f} mean_correction={float(np.mean(floor_correction)):.4f} "
                f"max_correction={float(np.max(floor_correction)):.4f}",
                flush=True,
            )

    # Upsample if we used fast mode (downsampling) or particle-size downsampling
    if (downsample_factor > 1 or downsampled) and not return_native_resolution:
        from scipy.ndimage import zoom
        # Get original dimensions (stored at start)
        orig_h = original_image.shape[0] if 'original_image' in locals() else original_h
        orig_w = original_image.shape[1] if 'original_image' in locals() else original_w
        # Upsample probability map back to original resolution
        output_mask = zoom(output_mask, (orig_h / h, orig_w / w), order=1)  # Bilinear interpolation
        if downsample_factor > 1:
            print(f"  Fast mode: Upsampled from {h}x{w} back to {orig_h}x{orig_w}", flush=True)
        elif downsampled:
            print(f"  Particle-size downsampling: Upsampled from {h}x{w} back to {orig_h}x{orig_w}", flush=True)

    if str(stitch_output_space) == "logit":
        output_mask = _sigmoid_np(output_mask)
    else:
        output_mask = np.clip(output_mask, 0.0, 1.0)

    # Apply test-time augmentation if enabled
    if use_tta:
        # Run predictions on augmented versions and average
        # Note: This calls the function recursively, but with use_tta=False to avoid infinite loop
        aug_predictions = [output_mask]  # Include original

        # Helper function to predict on augmented image
        def predict_on_augmented(aug_image, transform_back_fn):
            """Predict on augmented image and transform prediction back."""
            # Recursively call but disable TTA to avoid infinite recursion
            pred = predict_bad_regions_probability(
                model=model,
                image=aug_image,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
                pixel_size_angstrom=pixel_size_angstrom,
                use_power_spectrum=use_power_spectrum,
                include_real_space_input=include_real_space_input,
                use_pixel_size_channel=use_pixel_size_channel,
                pixel_size_channel_min_angstrom=pixel_size_channel_min_angstrom,
                pixel_size_channel_max_angstrom=pixel_size_channel_max_angstrom,
                use_hann_blending=use_hann_blending,
                blending_window=blending_window,
                blending_edge_px=blending_edge_px,
                stitch_phase_mode=stitch_phase_mode,
                stitch_phase_reduction=stitch_phase_reduction,
                stitch_phase_blend_alpha=stitch_phase_blend_alpha,
                stitch_output_space=stitch_output_space,
                stitch_output_blend_alpha=stitch_output_blend_alpha,
                stitch_output_preserve_threshold=stitch_output_preserve_threshold,
                stitch_output_transition=stitch_output_transition,
                patch_prob_floor_quantile=patch_prob_floor_quantile,
                patch_prob_floor_strength=patch_prob_floor_strength,
                downsample_factor=downsample_factor,
                adaptive_downsample_method=adaptive_downsample_method,
                return_native_resolution=return_native_resolution,
                batch_forward_size=batch_forward_size,
                use_tta=False,
                temperature=temperature,
                logit_bias=logit_bias,
                normalization_method=normalization_method,
                use_particle_size_downsampling=use_particle_size_downsampling,
                target_pixels_per_particle=target_pixels_per_particle,
                particle_size_angstrom=particle_size_angstrom,
                global_prior_path=global_prior_path,
                use_global_prior=(global_prior_path is not None),
                psd_use_radial_normalization=psd_use_radial_normalization,
                psd_multiscale=psd_multiscale,
                psd_multiscale_separate_channels=psd_multiscale_separate_channels,
                multiscale_psd_source=multiscale_psd_source,
                psd_scales=psd_scales,
                psd_frequency_band_channels=psd_frequency_band_channels,
                psd_frequency_bands=psd_frequency_bands,
                psd_frequency_band_include_full_spectrum=psd_frequency_band_include_full_spectrum,
                psd_frequency_band_include_anisotropy=psd_frequency_band_include_anisotropy,
                psd_frequency_band_anisotropy_min_freq=psd_frequency_band_anisotropy_min_freq,
                psd_texture_mode=psd_texture_mode,
                psd_highpass_cutoff=psd_highpass_cutoff,
                tile_offset_y=tile_offset_y,
                tile_offset_x=tile_offset_x,
            )
            return transform_back_fn(pred)

        # Flip horizontal
        image_hflip = np.fliplr(image)
        pred_hflip = predict_on_augmented(image_hflip, lambda p: np.fliplr(p))
        aug_predictions.append(pred_hflip)

        # Flip vertical
        image_vflip = np.flipud(image)
        pred_vflip = predict_on_augmented(image_vflip, lambda p: np.flipud(p))
        aug_predictions.append(pred_vflip)

        # Rotate 90 degrees
        image_rot90 = np.rot90(image, k=1)
        pred_rot90 = predict_on_augmented(image_rot90, lambda p: np.rot90(p, k=-1))
        aug_predictions.append(pred_rot90)

        # Average all predictions (original + 3 augmentations)
        output_mask = np.mean(aug_predictions, axis=0)

        print(f"  TTA: Averaged predictions from {len(aug_predictions)} augmentations (original + 3)", flush=True)

    return output_mask


def predict_bad_regions(model: torch.nn.Module, image: np.ndarray,
                       device: str = "cuda", patch_size: int = 512,
                       overlap: int = 64, pixel_size_angstrom: float = 1.1,
                       use_power_spectrum: bool = False, threshold: float = 0.6,
                       min_area_pixels: int = 10000) -> np.ndarray:
    """
    Predict bad regions in an image using the trained model.

    This function calls predict_bad_regions_probability() ONCE and then applies
    thresholding and minimum area filtering to produce a binary mask.

    Args:
        model: Trained model
        image: Input image (will be normalized)
        device: Device to run inference on
        patch_size: Size of patches for inference
        overlap: Overlap between patches
        pixel_size_angstrom: Pixel size (for power spectrum computation if needed)
        use_power_spectrum: If True, include power spectrum as additional channel
        threshold: Probability threshold for bad region classification
        min_area_pixels: Minimum area (in pixels) for a connected bad region to be kept
        use_particle_size_filtering: If True, compute min_area from particle size (ASOCEM-inspired)
        particle_size_angstrom: Particle size in Angstrom for filtering (default: 100.0)

    Returns:
        Binary mask (1 = bad region, 0 = good region)
    """
    # ASOCEM Integration: Compute min_area from particle size if enabled
    if use_particle_size_filtering:
        from utils.postprocessing import compute_min_area_from_particle_size
        min_area_pixels = compute_min_area_from_particle_size(
            pixel_size_angstrom=pixel_size_angstrom,
            particle_size_angstrom=particle_size_angstrom
        )
        print(f"  Particle-size filtering: min_area={min_area_pixels} pixels (particle_size={particle_size_angstrom} Å, pixel_size={pixel_size_angstrom:.2f} Å/px)", flush=True)

    # Call probability prediction ONCE (no redundant inference)
    prob_map = predict_bad_regions_probability(
        model, image, device=device, patch_size=patch_size,
        overlap=overlap, pixel_size_angstrom=pixel_size_angstrom,
        use_power_spectrum=use_power_spectrum
    )

    # Debug: print prediction statistics (only for first call to avoid spam)
    if not hasattr(predict_bad_regions, '_debug_printed'):
        max_prob = np.max(prob_map)
        min_prob = np.min(prob_map)
        mean_prob = np.mean(prob_map)
        median_prob = np.median(prob_map)
        std_prob = np.std(prob_map)
        # Count pixels above different thresholds
        above_01 = np.sum(prob_map > 0.1)
        above_05 = np.sum(prob_map > 0.5)
        above_09 = np.sum(prob_map > 0.9)
        total_pixels = prob_map.size

        print(f"\n{'='*60}")
        print(f"PREDICTION STATISTICS (Bad Region Probabilities)")
        print(f"{'='*60}")
        print(f"  Min:    {min_prob:.6f}")
        print(f"  Max:    {max_prob:.6f}")
        print(f"  Mean:   {mean_prob:.6f}")
        print(f"  Median: {median_prob:.6f}")
        print(f"  Std:    {std_prob:.6f}")
        print(f"\n  Pixels above 0.1: {above_01:8d} ({100*above_01/total_pixels:6.2f}%)")
        print(f"  Pixels above 0.5: {above_05:8d} ({100*above_05/total_pixels:6.2f}%)")
        print(f"  Pixels above 0.9: {above_09:8d} ({100*above_09/total_pixels:6.2f}%)")
        print(f"  Current threshold: {threshold:.2f}")
        print(f"{'='*60}\n")

        # Warning if probabilities are very low
        if max_prob < 0.1:
            print("⚠️  WARNING: Maximum probability is very low (<0.1). This suggests:")
            print("   1. Model may not be detecting bad regions well")
            print("   2. Model may need more training")
            print("   3. Threshold may need to be lowered significantly")
            print("   4. Training data may not match inference data distribution\n")

        # Warning if probabilities are very high (untrained decoder)
        if mean_prob > 0.9 or (above_09 / total_pixels) > 0.8:
            print("⚠️  NOTE: Very high probabilities detected (mean > 0.9 or >80% pixels > 0.9).")
            print("   This is expected for an untrained decoder (baseline checkpoint).")
            print("   The decoder needs training to properly convert encoder features to segmentation.\n")

        predict_bad_regions._debug_printed = True

    # Threshold to create binary mask
    binary_mask = (prob_map > threshold).astype(np.uint8)

    # Filter out small regions (minimum area filtering)
    if min_area_pixels > 0:
        # Label connected components
        labeled_mask, num_features = ndimage.label(binary_mask)

        # Calculate area of each component
        component_sizes = np.bincount(labeled_mask.ravel())
        # component_sizes[0] is background (0), skip it

        # Create mask for components that meet minimum area requirement
        valid_components = np.zeros(num_features + 1, dtype=bool)
        valid_components[1:] = component_sizes[1:] >= min_area_pixels  # Skip background (index 0)

        # Create filtered mask
        filtered_mask = valid_components[labeled_mask].astype(np.uint8)
        binary_mask = filtered_mask

    return binary_mask


def _compute_mask_space_scales(
    image_shape: Tuple[int, int],
    mask_shape: Tuple[int, int],
) -> Tuple[float, float]:
    """Return (scale_x, scale_y) mapping original image coordinates to mask coordinates."""
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    mask_h, mask_w = int(mask_shape[0]), int(mask_shape[1])
    if img_h <= 0 or img_w <= 0 or mask_h <= 0 or mask_w <= 0:
        return 1.0, 1.0
    scale_x = float(mask_w) / float(img_w)
    scale_y = float(mask_h) / float(img_h)
    return scale_x, scale_y


def _scale_filter_params_for_mask(
    *,
    min_area_pixels: int,
    distance_threshold_pixels: float,
    image_shape: Tuple[int, int],
    mask_shape: Tuple[int, int],
) -> Tuple[int, float, float, float]:
    """
    Scale area/distance thresholds from original-image pixel units to mask-space units.

    Returns:
        (min_area_mask_px, distance_mask_px, scale_x, scale_y)
    """
    scale_x, scale_y = _compute_mask_space_scales(image_shape, mask_shape)
    area_scale = float(scale_x * scale_y)
    # Distance transform is in isotropic pixel units on the mask grid; use mean linear scale.
    dist_scale = float(0.5 * (scale_x + scale_y))

    if int(min_area_pixels) > 0:
        min_area_mask_px = max(1, int(round(float(min_area_pixels) * area_scale)))
    else:
        min_area_mask_px = int(min_area_pixels)

    distance_mask_px = float(distance_threshold_pixels) * dist_scale
    return min_area_mask_px, distance_mask_px, scale_x, scale_y


def _lookup_dt_distances_for_pick_coords(
    dt: np.ndarray,
    coords: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Sample a distance-transform map at pick coordinates defined in original-image coordinates.
    """
    if len(coords) == 0:
        return np.array([], dtype=np.float32)

    coords_arr = np.asarray(coords, dtype=np.float32)
    dt_h, dt_w = int(dt.shape[0]), int(dt.shape[1])
    img_h, img_w = int(image_shape[0]), int(image_shape[1])

    if (dt_h, dt_w) == (img_h, img_w):
        xs = np.clip(coords_arr[:, 0].astype(np.int64), 0, dt_w - 1)
        ys = np.clip(coords_arr[:, 1].astype(np.int64), 0, dt_h - 1)
    else:
        scale_x, scale_y = _compute_mask_space_scales((img_h, img_w), (dt_h, dt_w))
        xs = np.clip((coords_arr[:, 0] * scale_x).astype(np.int64), 0, dt_w - 1)
        ys = np.clip((coords_arr[:, 1] * scale_y).astype(np.int64), 0, dt_h - 1)

    return np.asarray(dt[ys, xs], dtype=np.float32)


def filter_picks_with_model(cs_data: np.ndarray, mrc_dir: str, model_path: str,
                            output_path: str, device: str = "cuda",
                            threshold: float = 0.6, max_images: Optional[int] = None,
                            min_area_pixels: int = 10000, distance_threshold_pixels: int = 100,
                            original_cs_path: Optional[str] = None,
                            original_csg_path: Optional[str] = None,
                            cryosparc_project_dir: Optional[str] = None,
                            use_combined_format: bool = True,
                            patch_size: int = 512,
                            overlap: int = 64,
                            downsample_factor: int = 1,
                            filter_on_work_resolution: bool = False):
    """
    Run inference using trained model to filter particle picks.

    This function runs inference on micrographs using checkpoint weights,
    saves probability maps to a pickle file, and filters particle picks based on
    proximity to detected bad regions.

    Args:
        cs_data: Loaded CS file data
        mrc_dir: Directory containing MRC files
        model_path: Path to trained model checkpoint (used to detect model_type and use_power_spectrum).
        output_path: Path to save filtered CS file
        device: Device to run inference on
        threshold: Threshold for bad region detection
        max_images: Maximum number of images to process. If None, process all images.
        min_area_pixels: Minimum area (in pixels) for a bad region to be considered
        distance_threshold_pixels: Distance threshold (in pixels) for removing particles near bad regions
        original_cs_path: Path to original CS file (for .csg file creation)
        original_csg_path: Path to original .csg file (template for filtered .csg)
        cryosparc_project_dir: CryoSPARC project root directory for resolving relative paths in CS file
        use_combined_format: If True, creates a single .cs file with all fields (combined format).
                            If False, creates separate blob and passthrough files (split format).
                            Default is True (combined format) as it's more compatible when input files
                            are missing CTF/pick_stats fields.
    """
    checkpoint = torch.load(model_path, map_location=device)
    model_config = _resolve_checkpoint_model_config(checkpoint)
    state_dict = model_config["state_dict"]
    use_power_spectrum = bool(model_config["use_power_spectrum"])
    model_type = str(model_config["model_type"])
    input_channels = int(model_config["input_channels"])

    # Create model with correct type and input channels
    print(
        f"Loading model: type={model_type}, use_power_spectrum={use_power_spectrum}, "
        f"input_channels={input_channels}, pixel_size_channel={model_config['use_pixel_size_channel']}",
        flush=True,
    )

    model = create_model(
        model_type=model_type,
        device=device,
        use_power_spectrum=use_power_spectrum,
        input_channels_override=input_channels,
        attention_type=str(model_config.get("attention_type", "global")),
    )

    print(f"Loading checkpoint weights...")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️  Missing keys (not loaded): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
        if not missing_keys and not unexpected_keys:
            print(f"  ✓ All checkpoint weights loaded successfully")
        else:
            print(f"  ✓ Loaded checkpoint weights (partial load)")
    except Exception as e:
        print(f"  ⚠️  Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
    _attach_model_input_metadata(model, model_config)

    model.eval()
    print(f"Model loaded and set to evaluation mode")

    # Get unique micrographs
    # Build mapping of actual files (handles _particles suffix, subdirectories, symlinks)
    mrc_dir_path = Path(mrc_dir)
    actual_files = {}
    base_to_actual = {}  # Maps filename without _particles to actual filename with _particles
    if mrc_dir_path.exists():
        # Check direct files and subdirectories
        for mrc_file in mrc_dir_path.rglob("*.mrc"):
            resolved = mrc_file.resolve()
            actual_files[mrc_file.name] = resolved
            # Handle _particles suffix: if file ends with _particles.mrc, also map without _particles
            if mrc_file.name.endswith("_particles.mrc"):
                base_name = mrc_file.name[:-len("_particles.mrc")] + ".mrc"
                if base_name not in base_to_actual:  # Don't overwrite if already mapped
                    base_to_actual[base_name] = resolved

    # Also check CryoSPARC project directory if provided (for relative paths in CS file)
    cryosparc_project_path = Path(cryosparc_project_dir) if cryosparc_project_dir else None

    micrograph_paths = {}
    for pick in cs_data:
        mrc_path = pick['location/micrograph_path']
        if isinstance(mrc_path, bytes):
            mrc_path = mrc_path.decode('utf-8')
        mrc_name = Path(mrc_path).name
        if mrc_name not in micrograph_paths:
            mrc_file = None

            # 1. Try resolving relative path in CryoSPARC project directory (if path is relative and cryosparc_dir provided)
            if cryosparc_project_path and cryosparc_project_path.exists():
                cs_path_obj = Path(mrc_path)
                if not cs_path_obj.is_absolute():
                    # Try resolving relative path in CryoSPARC project directory
                    resolved_path = cryosparc_project_path / mrc_path
                    if resolved_path.exists():
                        mrc_file = resolved_path.resolve()

            # 2. Try exact filename in mrc_dir
            if mrc_file is None or not mrc_file.exists():
                mrc_file = mrc_dir_path / mrc_name
                if not mrc_file.exists():
                    # 3. Try in actual_files dict (handles subdirectories/symlinks)
                    if mrc_name in actual_files:
                        mrc_file = actual_files[mrc_name]
                    # 4. Try base_to_actual mapping (handles _particles suffix mismatch)
                    elif mrc_name in base_to_actual:
                        mrc_file = base_to_actual[mrc_name]
                    else:
                        # Still set the path even if not found - will be checked later
                        mrc_file = mrc_dir_path / mrc_name
            micrograph_paths[mrc_name] = mrc_file

    # Limit number of images if specified
    all_micrograph_names = set(micrograph_paths.keys())
    if max_images is not None:
        micrograph_items = list(micrograph_paths.items())[:max_images]
        micrograph_paths = dict(micrograph_items)
        print(f"Limiting processing to {len(micrograph_paths)} images (requested: {max_images})")
    else:
        all_micrograph_names = None  # Process all, no need to track unprocessed

    # Pre-index picks by micrograph (Item 3: O(M×P) -> O(P) optimization)
    # Build idx_by_mrc dict once before the loop to avoid scanning cs_data repeatedly
    print("Pre-indexing picks by micrograph...", flush=True)
    idx_by_mrc: Dict[str, np.ndarray] = {}
    for i, pick in enumerate(cs_data):
        pick_path = pick['location/micrograph_path']
        if isinstance(pick_path, bytes):
            pick_path = pick_path.decode('utf-8')
        pick_name = os.path.basename(pick_path)
        if pick_name not in idx_by_mrc:
            idx_by_mrc[pick_name] = []
        idx_by_mrc[pick_name].append(i)
    # Convert lists to numpy arrays for faster indexing
    for mrc_name in idx_by_mrc:
        idx_by_mrc[mrc_name] = np.array(idx_by_mrc[mrc_name], dtype=np.int64)
    print(f"Pre-indexed picks for {len(idx_by_mrc)} micrographs", flush=True)

    # Create a boolean mask to track which picks to keep (True = keep, False = remove)
    # Start with all picks marked as keep
    keep_mask = np.ones(len(cs_data), dtype=bool)
    total_removed = 0
    processed_micrographs = set()

    # Dictionary to store inference results for saving to pickle
    inference_results: Dict[str, Dict] = {}

    for mrc_name, mrc_path in tqdm(micrograph_paths.items(), desc="Running inference on micrographs"):
        try:
            print(f"Processing: {mrc_name}", flush=True)
            if not mrc_path.exists():
                print(f"  Warning: {mrc_path} not found, skipping", flush=True)
                # Keep all picks if micrograph not found
                continue

            # Load image
            print(f"  Loading image...", flush=True)
            image = load_mrc(str(mrc_path))
            if len(image.shape) == 3:
                image = image[0]  # Take first slice if 3D
            # Keep consistent with training defaults (MC-style robust scaling)
            image = normalize_image(image, method="robust")
            print(f"  Image shape: {image.shape}", flush=True)

            # Get picks for this micrograph using pre-indexed dictionary (faster than scanning)
            pick_indices_for_mrc = idx_by_mrc.get(mrc_name, np.array([]))
            if len(pick_indices_for_mrc) > 0:
                picks = cs_data[pick_indices_for_mrc]
            else:
                picks = np.array([], dtype=cs_data.dtype)

            if len(picks) == 0:
                print(f"  No picks for this micrograph, skipping", flush=True)
                continue

            # Get pixel size
            pixel_size = 1.1  # default
            if len(picks) > 0:
                pixel_size = float(picks[0]['location/micrograph_psize_A'])
            elif len(cs_data) > 0:
                pixel_size = float(cs_data[0]['location/micrograph_psize_A'])

            # Predict bad regions - get probability map first for saving
            print(f"  Running inference (this may take a minute for large images)...", flush=True)
            use_work_resolution_filter = bool(filter_on_work_resolution and int(max(1, downsample_factor)) > 1)
            prob_map = predict_bad_regions_probability(
                model,
                image,
                device=device,
                pixel_size_angstrom=pixel_size,
                use_power_spectrum=use_power_spectrum,
                patch_size=patch_size,
                overlap=overlap,
                downsample_factor=int(max(1, downsample_factor)),
                return_native_resolution=use_work_resolution_filter,
            )
            print(f"  Inference complete, probability map shape: {prob_map.shape}", flush=True)
            min_area_eff, distance_threshold_eff, scale_x, scale_y = _scale_filter_params_for_mask(
                min_area_pixels=min_area_pixels,
                distance_threshold_pixels=float(distance_threshold_pixels),
                image_shape=tuple(image.shape),
                mask_shape=tuple(prob_map.shape),
            )
            if tuple(prob_map.shape) != tuple(image.shape):
                print(
                    f"  Work-resolution filtering enabled: mask space {prob_map.shape} "
                    f"(scale_x={scale_x:.4f}, scale_y={scale_y:.4f}, "
                    f"min_area={min_area_pixels}->{min_area_eff}, "
                    f"distance={distance_threshold_pixels}->{distance_threshold_eff:.2f}px)",
                    flush=True,
                )

            # Print prediction statistics for debugging (especially important for no-annotations model)
            max_prob = np.max(prob_map)
            min_prob = np.min(prob_map)
            mean_prob = np.mean(prob_map)
            median_prob = np.median(prob_map)
            std_prob = np.std(prob_map)
            above_thresh = np.sum(prob_map > threshold)
            total_pixels = prob_map.size
            print(f"  Prediction stats: min={min_prob:.4f}, max={max_prob:.4f}, mean={mean_prob:.4f}, "
                  f"median={median_prob:.4f}, std={std_prob:.4f}")
            print(f"  Pixels above threshold {threshold:.2f}: {above_thresh}/{total_pixels} ({100*above_thresh/total_pixels:.2f}%)", flush=True)

            # Store inference result for this micrograph
            print(f"  Storing inference result...", flush=True)
            inference_results[mrc_name] = {
                'probability_map': prob_map,  # Full probability map (0-1)
                'threshold': threshold,
                'pixel_size_angstrom': pixel_size,
                'image_shape': image.shape,
                'probability_map_shape': prob_map.shape,
                'filter_on_work_resolution': bool(use_work_resolution_filter),
                'downsample_factor': int(max(1, downsample_factor)),
                'filter_scale_x': float(scale_x),
                'filter_scale_y': float(scale_y),
                'timestamp': datetime.now().isoformat()
            }
            print(f"  Inference result stored", flush=True)

            # Threshold and filter to get binary mask for filtering
            print(f"  Thresholding probability map (threshold={threshold})...", flush=True)
            binary_mask = (prob_map > threshold).astype(np.uint8)
            print(f"  Binary mask created, {np.sum(binary_mask)} bad pixels", flush=True)

            # Apply minimum area filtering - EXACT SAME AS GUI (_display_image lines 1234-1244)
            if min_area_eff > 0:
                area_msg = f"{min_area_pixels}" if min_area_eff == min_area_pixels else f"{min_area_pixels} -> {min_area_eff} (mask px)"
                print(f"  Applying minimum area filter (min_area={area_msg})...", flush=True)
                labeled_mask, num_features = ndimage.label(binary_mask)
                print(f"  Found {num_features} connected components", flush=True)

                # EXACT GUI METHOD: use np.bincount and valid_components array
                component_sizes = np.bincount(labeled_mask.ravel())
                valid_components = np.zeros(num_features + 1, dtype=bool)
                valid_components[1:] = component_sizes[1:] >= min_area_eff  # Skip background (index 0)
                model_mask = valid_components[labeled_mask].astype(np.uint8)

                removed_count = num_features - np.sum(valid_components[1:])
                print(f"  Removed {removed_count} small regions, kept {np.sum(valid_components[1:])} large regions", flush=True)
                print(f"  After filtering: {np.sum(model_mask)} bad pixels", flush=True)
            else:
                model_mask = binary_mask

            bad_mask = model_mask  # Use model_mask (GUI naming convention)

            # Debug: print statistics about predictions
            bad_pixels = np.sum(bad_mask == 1)
            total_pixels = bad_mask.size
            bad_percentage = (bad_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            # Filter picks based on distance from bad regions (Item 2: Replace cdist with distance transform)
            print(f"  Filtering picks based on distance...", flush=True)
            coords = picks_to_coordinates(picks, image.shape)
            print(f"  Got {len(coords)} pick coordinates", flush=True)

            removed_for_this_image = 0

            # Get pick indices for this micrograph from pre-indexed dict (Item 3 optimization)
            pick_indices = idx_by_mrc.get(mrc_name, np.array([], dtype=np.int64))
            print(f"  Found {len(pick_indices)} pick indices from pre-indexed dict", flush=True)

            if len(pick_indices) > 0 and distance_threshold_eff > 0 and np.any(bad_mask):
                thresh_msg = f"{distance_threshold_pixels}px"
                if abs(float(distance_threshold_eff) - float(distance_threshold_pixels)) > 1e-6:
                    thresh_msg = f"{distance_threshold_pixels}px -> {distance_threshold_eff:.2f}px (mask)"
                print(f"  Computing distance transform (threshold={thresh_msg})...", flush=True)

                # Item 2: Use distance transform instead of per-pick cdist
                # EXACT SAME AS GUI (_display_image line 1262): use 1 - bad_mask
                # distance_transform_edt computes distance to nearest 0-valued pixel
                # Input: 1 - bad_mask means 1=good, 0=bad (inverted)
                # Output: dt[y, x] = distance from (y, x) to nearest bad pixel
                dt = ndimage.distance_transform_edt(1 - bad_mask)  # EXACT match GUI: 1 - model_mask

                # Coordinate convention check:
                # picks_to_coordinates returns [x, y] pairs where x=col, y=row
                # Image indexing is [row, col] = [y, x]
                # So for pick at (x, y), we access dt[y, x] (row=y, col=x)

                # Filter picks: if distance to nearest bad pixel <= threshold, remove pick
                # DEBUG: Track distance stats
                if len(coords) > 0:
                    distances = _lookup_dt_distances_for_pick_coords(dt, coords, tuple(image.shape))
                    within = distances <= float(distance_threshold_eff)
                    if np.any(within):
                        idx_remove = np.asarray(pick_indices, dtype=np.int64)[within]
                        keep_mask[idx_remove] = False
                        removed_n = int(np.sum(within))
                        total_removed += removed_n
                        removed_for_this_image += removed_n
                else:
                    distances = np.array([], dtype=np.float32)

                # DEBUG: Print distance stats for first few images to diagnose
                if len(distances) > 0:
                    print(f"  Distance stats: min={np.min(distances):.2f}, max={np.max(distances):.2f}, mean={np.mean(distances):.2f}, threshold={distance_threshold_eff:.2f}, picks_within_threshold={np.sum(distances <= distance_threshold_eff)}/{len(distances)}", flush=True)
                print(f"  Distance transform filtering complete, removed {removed_for_this_image} picks", flush=True)
            else:
                if len(pick_indices) == 0:
                    print(f"  Skipping distance filtering (no picks for this micrograph)", flush=True)
                elif distance_threshold_eff == 0:
                    print(f"  Skipping distance filtering (threshold=0)", flush=True)
                elif not np.any(bad_mask):
                    print(f"  Skipping distance filtering (no bad pixels)", flush=True)

            # Debug output for first few images
            if len(micrograph_paths) <= 5 or mrc_name == list(micrograph_paths.keys())[0]:
                print(f"  {mrc_name}: {bad_percentage:.2f}% bad pixels, {removed_for_this_image}/{len(picks)} picks removed", flush=True)

            processed_micrographs.add(mrc_name)
            print(f"  Completed: {mrc_name}", flush=True)
        except Exception as e:
            print(f"ERROR processing {mrc_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            # Continue with next micrograph
            continue

    # If we limited images, keep all picks from unprocessed micrographs
    # Unprocessed micrographs are already marked as keep (keep_mask starts as all True)
    if all_micrograph_names is not None:
        unprocessed_micrographs = all_micrograph_names - processed_micrographs
        if unprocessed_micrographs:
            print(f"Keeping all picks from {len(unprocessed_micrographs)} unprocessed micrographs")
            sys.stdout.flush()
            # No need to do anything - keep_mask already has all picks marked as True
            # Unprocessed picks will remain in the filtered array

    # Apply the keep mask to filter the array
    # This directly removes rows from the structured numpy array
    filtered_array = cs_data[keep_mask]

    print(f"DEBUG: About to save filtered picks...")
    print(f"DEBUG: Original picks: {len(cs_data)}, Filtered picks: {len(filtered_array)}")
    sys.stdout.flush()

    try:
        # Ensure output path has .cs extension
        if not output_path.endswith('.cs'):
            output_path += '.cs'

        # Determine file paths
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        base_name = output_path_obj.stem

        if use_combined_format:
            # Combined format: single file with all fields (like user's notebook example)
            save_cs(filtered_array, output_path)
            print(f"Saved combined .cs file: {output_path_obj.name} ({len(filtered_array)} picks, {len(filtered_array.dtype.names) if filtered_array.dtype.names else 0} fields)")
            combined_file_path = output_path
            blob_file_path = None
            passthrough_file_path = None
        else:
            # Split format: create separate blob and passthrough files
            blob_array = create_blob_cs_data(filtered_array)
            blob_file_path = output_dir / f"{base_name}_blob.cs"
            save_cs(blob_array, str(blob_file_path))
            print(f"Saved blob file: {blob_file_path.name}")

            passthrough_array = create_passthrough_cs_data(filtered_array)
            passthrough_file_path = output_dir / f"{base_name}_passthrough.cs"
            save_cs(passthrough_array, str(passthrough_file_path))
            print(f"Saved passthrough file: {passthrough_file_path.name}")
            combined_file_path = None

        sys.stdout.flush()

        # Create matching .csg file for CryoSPARC import
        # Priority: 1) User-provided original_csg_path, 2) Same directory as .cs file, 3) Reference directory
        csg_path = None

        # First, use user-provided .csg file if available
        if original_csg_path and os.path.exists(original_csg_path):
            csg_path = original_csg_path
            print(f"DEBUG: Using user-provided .csg file: {csg_path}")

        # Second, try to find original .csg file in the same directory as the input CS file
        if not csg_path and original_cs_path:
            original_cs_dir = os.path.dirname(original_cs_path)
            original_cs_name = os.path.basename(original_cs_path).replace('.cs', '.csg')
            potential_csg = os.path.join(original_cs_dir, original_cs_name)
            if os.path.exists(potential_csg):
                csg_path = potential_csg
                print(f"DEBUG: Found original .csg file: {csg_path}")

        # Third, check for reference_csg_file directory in the project root
        if not csg_path:
            # Get project root (go up from models/ to project root)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            reference_csg_dir = os.path.join(project_root, 'reference_csg_file')
            if os.path.exists(reference_csg_dir):
                # Look for any .csg file in the reference directory
                for ref_file in os.listdir(reference_csg_dir):
                    if ref_file.endswith('.csg'):
                        csg_path = os.path.join(reference_csg_dir, ref_file)
                        print(f"DEBUG: Using reference .csg file: {csg_path}")
                        sys.stdout.flush()
                        break

        output_csg_path = None
        if csg_path and os.path.exists(csg_path):
            try:
                csg_output_path = output_dir / f"{base_name}.csg"
                if use_combined_format:
                    output_csg_path = create_csg_file(
                        str(combined_file_path),
                        original_csg_path=csg_path,
                        output_csg_path=str(csg_output_path),
                        use_combined_format=True
                    )
                else:
                    output_csg_path = create_csg_file(
                        str(passthrough_file_path),
                        original_csg_path=csg_path,
                        output_csg_path=str(csg_output_path),
                        blob_file_path=str(blob_file_path) if blob_file_path else None,
                        passthrough_file_path=str(passthrough_file_path) if passthrough_file_path else None,
                        use_combined_format=False
                    )
                print(f"Created CSG file: {output_csg_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"ERROR: Could not create .csg file: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        else:
            print(f"ERROR: No original .csg file found. Cannot create .csg file.")
            print(f"  Looked for: {potential_csg if original_cs_path else 'N/A'}")
            reference_csg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference_csg_file')
            print(f"  Reference directory: {reference_csg_dir}")
            print(f"  Reference directory exists: {os.path.exists(reference_csg_dir)}")
            if os.path.exists(reference_csg_dir):
                ref_files = [f for f in os.listdir(reference_csg_dir) if f.endswith('.csg')]
                print(f"  Reference .csg files found: {ref_files}")
            print(f"  The .cs file was saved, but you'll need to create the .csg file manually for CryoSPARC import.")
            print(f"  You can copy the reference .csg file and update the metafile paths manually.")
            sys.stdout.flush()

        # Save inference results to pickle file
        main_file_path = combined_file_path if use_combined_format else str(passthrough_file_path)
        inference_output_path = main_file_path.replace('.cs', '_inference_results.pkl')
        if inference_results:
            inference_data = {
                'inference_results': inference_results,  # Dict mapping mrc_name -> results
                'model_path': model_path,
                'threshold': threshold,
                'min_area_pixels': min_area_pixels,
                'distance_threshold_pixels': distance_threshold_pixels,
                'processed_micrographs': list(processed_micrographs),
                'total_original_picks': len(cs_data),
                'total_filtered_picks': len(filtered_array),
                'total_removed': total_removed,
                'timestamp': datetime.now().isoformat(),
                'output_cs_path': main_file_path
            }
            with open(inference_output_path, 'wb') as f:
                pickle.dump(inference_data, f)
            print(f"Saved inference results to: {inference_output_path}")
            sys.stdout.flush()

        print(f"Filtering complete!")
        print(f"Original picks: {len(cs_data)}")
        print(f"Filtered picks: {len(filtered_array)}")
        print(f"Removed: {total_removed}")
        if use_combined_format:
            print(f"Combined .cs file: {output_path_obj.name}")
        else:
            print(f"Blob file: {blob_file_path.name}")
            print(f"Passthrough file: {passthrough_file_path.name}")
        if output_csg_path:
            print(f"CSG file: {output_csg_path}")
        else:
            print(f"CSG file: Not created (see error messages above)")
        sys.stdout.flush()

        # Return list of processed micrograph names
        processed_list = list(processed_micrographs)
        print(f"DEBUG: Returning {len(processed_list)} processed micrograph names")
        if len(processed_list) > 0:
            print(f"DEBUG: First few: {processed_list[:3]}")
        else:
            print(f"DEBUG: WARNING - processed_list is empty!")
        sys.stdout.flush()

        return processed_list
    except Exception as e:
        import traceback
        print(f"DEBUG: ERROR in filter_picks_with_model: {e}")
        print(traceback.format_exc())
        sys.stdout.flush()
        # Return what we have
        return list(processed_micrographs)


def filter_picks_multiple_thresholds(cs_data: np.ndarray, mrc_dir: str, model_path: str,
                                     output_dir: str, thresholds: List[float],
                                     device: str = "cuda", max_images: Optional[int] = None,
                                     min_area_pixels: int = 10000, distance_threshold_pixels: int = 100,
                                     original_cs_path: Optional[str] = None,
                                     original_csg_path: Optional[str] = None,
                                     cryosparc_project_dir: Optional[str] = None,
                                     patch_size: int = 512, overlap: int = 64,
                                     downsample_factor: int = 1,
                                     inference_cache_file: Optional[str] = None,
                                     filter_on_work_resolution: bool = False) -> Dict[float, str]:
    """
    Run inference ONCE and apply multiple thresholds to filter particle picks.

    This is an optimized version that runs inference once per micrograph (caching probability maps)
    and then applies multiple thresholds to the same cached maps. This is much faster than calling
    filter_picks_with_model multiple times.

    Args:
        cs_data: Loaded CS file data
        mrc_dir: Directory containing MRC files
        model_path: Path to trained model checkpoint
        output_dir: Directory to save filtered CS files (one per threshold)
        thresholds: List of thresholds to apply
        device: Device to run inference on
        max_images: Maximum number of images to process. If None, process all images.
        min_area_pixels: Minimum area (in pixels) for a bad region to be considered
        distance_threshold_pixels: Distance threshold (in pixels) for removing particles near bad regions
        original_cs_path: Path to original CS file (for .csg file creation)
        original_csg_path: Path to original .csg file (template for filtered .csg)
        cryosparc_project_dir: CryoSPARC project root directory for resolving relative paths in CS file
        patch_size: Size of patches for inference (default: 512)
        overlap: Overlap between patches (default: 64)
        downsample_factor: If > 1, downsample image before inference then upsample result (default: 1, no downsampling)
        inference_cache_file: Optional path to pickle file to save/load inference results.
                             If file exists, loads probability maps from it instead of running inference.
                             If file doesn't exist, runs inference and saves results to this file.

    Returns:
        Dict mapping threshold -> output_file_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, map_location=device)
    model_config = _resolve_checkpoint_model_config(checkpoint)
    state_dict = model_config["state_dict"]
    use_power_spectrum = bool(model_config["use_power_spectrum"])
    model_type = str(model_config["model_type"])
    input_channels = int(model_config["input_channels"])

    print(
        f"Loading model: type={model_type}, use_power_spectrum={use_power_spectrum}, "
        f"input_channels={input_channels}, pixel_size_channel={model_config['use_pixel_size_channel']}",
        flush=True,
    )

    model = create_model(
        model_type=model_type,
        device=device,
        use_power_spectrum=use_power_spectrum,
        input_channels_override=input_channels,
        attention_type=str(model_config.get("attention_type", "global")),
    )

    print(f"Loading checkpoint weights...")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️  Missing keys (not loaded): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
        if not missing_keys and not unexpected_keys:
            print(f"  ✓ All checkpoint weights loaded successfully")
        else:
            print(f"  ✓ Loaded checkpoint weights (partial load)")
    except Exception as e:
        print(f"  ⚠️  Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
    _attach_model_input_metadata(model, model_config)

    model.eval()
    print(f"Model loaded and set to evaluation mode")

    # Get unique micrographs (same logic as filter_picks_with_model)
    mrc_dir_path = Path(mrc_dir)
    actual_files = {}
    base_to_actual = {}
    if mrc_dir_path.exists():
        for mrc_file in mrc_dir_path.rglob("*.mrc"):
            resolved = mrc_file.resolve()
            actual_files[mrc_file.name] = resolved
            if mrc_file.name.endswith("_particles.mrc"):
                base_name = mrc_file.name[:-len("_particles.mrc")] + ".mrc"
                if base_name not in base_to_actual:
                    base_to_actual[base_name] = resolved

    cryosparc_project_path = Path(cryosparc_project_dir) if cryosparc_project_dir else None

    micrograph_paths = {}
    for pick in cs_data:
        mrc_path = pick['location/micrograph_path']
        if isinstance(mrc_path, bytes):
            mrc_path = mrc_path.decode('utf-8')
        mrc_name = Path(mrc_path).name
        if mrc_name not in micrograph_paths:
            mrc_file = None

            if cryosparc_project_path and cryosparc_project_path.exists():
                cs_path_obj = Path(mrc_path)
                if not cs_path_obj.is_absolute():
                    resolved_path = cryosparc_project_path / mrc_path
                    if resolved_path.exists():
                        mrc_file = resolved_path.resolve()

            if mrc_file is None or not mrc_file.exists():
                mrc_file = mrc_dir_path / mrc_name
                if not mrc_file.exists():
                    if mrc_name in actual_files:
                        mrc_file = actual_files[mrc_name]
                    elif mrc_name in base_to_actual:
                        mrc_file = base_to_actual[mrc_name]
                    else:
                        mrc_file = mrc_dir_path / mrc_name
            micrograph_paths[mrc_name] = mrc_file

    all_micrograph_names = set(micrograph_paths.keys())
    if max_images is not None:
        micrograph_items = list(micrograph_paths.items())[:max_images]
        micrograph_paths = dict(micrograph_items)
        print(f"Limiting processing to {len(micrograph_paths)} images (requested: {max_images})")
    else:
        all_micrograph_names = None

    # Pre-index picks by micrograph
    print("Pre-indexing picks by micrograph...", flush=True)
    idx_by_mrc: Dict[str, np.ndarray] = {}
    for i, pick in enumerate(cs_data):
        pick_path = pick['location/micrograph_path']
        if isinstance(pick_path, bytes):
            pick_path = pick_path.decode('utf-8')
        pick_name = os.path.basename(pick_path)
        if pick_name not in idx_by_mrc:
            idx_by_mrc[pick_name] = []
        idx_by_mrc[pick_name].append(i)
    for mrc_name in idx_by_mrc:
        idx_by_mrc[mrc_name] = np.array(idx_by_mrc[mrc_name], dtype=np.int64)
    print(f"Pre-indexed picks for {len(idx_by_mrc)} micrographs", flush=True)

    # STEP 1: Run inference ONCE per micrograph (cache probability maps) OR load from cache
    print("=" * 80)
    print("STEP 1: Running inference ONCE per micrograph (caching probability maps)")
    print("=" * 80)
    prob_maps_cache: Dict[str, Dict] = {}  # mrc_name -> {'prob_map': ..., 'pixel_size': ..., 'image_shape': ..., 'picks': ..., 'pick_indices': ...}
    processed_micrographs = set()

    # Try to load cached inference results if cache file exists
    cached_prob_maps = {}
    if inference_cache_file and Path(inference_cache_file).exists():
        try:
            print(f"Loading cached inference results from: {inference_cache_file}")
            with open(inference_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            cached_prob_maps = cached_data.get('prob_maps_cache', {})
            print(f"  Loaded probability maps for {len(cached_prob_maps)} micrographs")
        except Exception as e:
            print(f"  Warning: Could not load inference cache: {e}")
            print(f"  Will run inference instead")
            cached_prob_maps = {}

    for mrc_name, mrc_path in tqdm(micrograph_paths.items(), desc="Running inference (once)"):
        try:
            if not mrc_path.exists():
                continue

            # Load image
            image = load_mrc(str(mrc_path))
            if len(image.shape) == 3:
                image = image[0]
            # Keep consistent with training defaults (MC-style robust scaling)
            image = normalize_image(image, method="robust")

            # Get picks for this micrograph
            pick_indices_for_mrc = idx_by_mrc.get(mrc_name, np.array([]))
            if len(pick_indices_for_mrc) > 0:
                picks = cs_data[pick_indices_for_mrc]
            else:
                picks = np.array([], dtype=cs_data.dtype)

            if len(picks) == 0:
                continue

            # Get pixel size
            pixel_size = 1.1
            if len(picks) > 0:
                pixel_size = float(picks[0]['location/micrograph_psize_A'])
            elif len(cs_data) > 0:
                pixel_size = float(cs_data[0]['location/micrograph_psize_A'])

            # Check if we have cached probability map for this micrograph
            if mrc_name in cached_prob_maps:
                # Load from cache
                cached_entry = cached_prob_maps[mrc_name]
                prob_map = cached_entry.get('prob_map')
                if prob_map is None:
                    # Cache entry exists but no prob_map - run inference
                    prob_map = predict_bad_regions_probability(
                        model, image, device=device, pixel_size_angstrom=pixel_size,
                        use_power_spectrum=use_power_spectrum,
                        patch_size=patch_size, overlap=overlap,
                        downsample_factor=downsample_factor,
                        return_native_resolution=bool(filter_on_work_resolution and int(max(1, downsample_factor)) > 1),
                    )
            else:
                # Run inference ONCE
                prob_map = predict_bad_regions_probability(
                    model, image, device=device, pixel_size_angstrom=pixel_size,
                    use_power_spectrum=use_power_spectrum,
                    patch_size=patch_size, overlap=overlap,
                    downsample_factor=downsample_factor,
                    return_native_resolution=bool(filter_on_work_resolution and int(max(1, downsample_factor)) > 1),
                )

            # Cache probability map and related data
            prob_maps_cache[mrc_name] = {
                'prob_map': prob_map,
                'pixel_size': pixel_size,
                'image_shape': image.shape,
                'picks': picks,
                'pick_indices': pick_indices_for_mrc
            }

            processed_micrographs.add(mrc_name)
        except Exception as e:
            print(f"ERROR processing {mrc_name} during inference: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    print(f"Inference complete: cached probability maps for {len(prob_maps_cache)} micrographs")

    # Save inference cache if cache file was provided
    if inference_cache_file:
        try:
            # Prepare cache data (just probability maps, pixel sizes, and image shapes - picks are recomputed)
            cache_data = {
                'prob_maps_cache': {
                    mrc_name: {
                        'prob_map': entry['prob_map'],
                        'pixel_size': entry['pixel_size'],
                        'image_shape': entry['image_shape']
                    }
                    for mrc_name, entry in prob_maps_cache.items()
                },
                'model_path': model_path,
                'patch_size': patch_size,
                'overlap': overlap,
                'downsample_factor': downsample_factor,
                'use_power_spectrum': use_power_spectrum,
                'timestamp': datetime.now().isoformat()
            }
            cache_path = Path(inference_cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved inference cache to: {inference_cache_file}")
        except Exception as e:
            print(f"Warning: Could not save inference cache: {e}")

    # STEP 2: For each threshold, apply filtering to cached probability maps
    print("=" * 80)
    print(f"STEP 2: Applying {len(thresholds)} thresholds to cached probability maps")
    print("=" * 80)

    output_files = {}

    for threshold in thresholds:
        print(f"\nProcessing threshold {threshold:.2f}...")
        keep_mask = np.ones(len(cs_data), dtype=bool)
        total_removed = 0

        for mrc_name, cache_data in prob_maps_cache.items():
            try:
                prob_map = cache_data['prob_map']
                picks = cache_data['picks']
                pick_indices = cache_data['pick_indices']
                image_shape = cache_data['image_shape']
                min_area_eff, distance_threshold_eff, _sx, _sy = _scale_filter_params_for_mask(
                    min_area_pixels=min_area_pixels,
                    distance_threshold_pixels=float(distance_threshold_pixels),
                    image_shape=tuple(image_shape),
                    mask_shape=tuple(prob_map.shape),
                )

                # Apply threshold to cached probability map
                binary_mask = (prob_map > threshold).astype(np.uint8)

                # Apply minimum area filtering
                if min_area_eff > 0:
                    labeled_mask, num_features = ndimage.label(binary_mask)
                    if num_features > 0:
                        component_sizes = np.bincount(labeled_mask.ravel())
                        valid_components = np.zeros(num_features + 1, dtype=bool)
                        valid_components[1:] = component_sizes[1:] >= min_area_eff
                        model_mask = valid_components[labeled_mask].astype(np.uint8)
                    else:
                        model_mask = binary_mask
                else:
                    model_mask = binary_mask

                bad_mask = model_mask

                # Filter picks based on distance from bad regions
                if len(pick_indices) > 0 and distance_threshold_eff > 0 and np.any(bad_mask):
                    coords = picks_to_coordinates(picks, image_shape)
                    dt = ndimage.distance_transform_edt(1 - bad_mask)

                    if len(coords) > 0:
                        distances = _lookup_dt_distances_for_pick_coords(dt, coords, tuple(image_shape))
                        within = distances <= float(distance_threshold_eff)
                        if np.any(within):
                            idx_remove = np.asarray(pick_indices, dtype=np.int64)[within]
                            keep_mask[idx_remove] = False
                            total_removed += int(np.sum(within))
            except Exception as e:
                print(f"ERROR processing {mrc_name} at threshold {threshold:.2f}: {e}", flush=True)
                continue

        # Filter the array (simple approach: just delete rows)
        filtered_array = cs_data[keep_mask]

        # Save filtered .cs file
        base_name = f"our_local_t{threshold:.2f}"
        output_file = output_dir / f"{base_name}.cs"
        save_cs(filtered_array, str(output_file))
        output_files[threshold] = str(output_file)

        # Create .csg file pointing to the filtered .cs file
        if original_csg_path and os.path.exists(original_csg_path):
            try:
                output_csg_path = output_dir / f"{base_name}.csg"
                create_csg_file(
                    str(output_file),
                    original_csg_path=str(original_csg_path),
                    output_csg_path=str(output_csg_path),
                    use_combined_format=True
                )
            except Exception as e:
                print(f"  Warning: Could not create .csg file: {e}")

        print(f"  Threshold {threshold:.2f}: {len(filtered_array)}/{len(cs_data)} picks retained ({total_removed} removed)")

    print("\n" + "=" * 80)
    print("Multi-threshold filtering complete!")
    print("=" * 80)
    print(f"Generated {len(output_files)} filtered files")

    return output_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter particle picks using trained model")
    parser.add_argument("--cs_file", type=str, required=True, help="Input CS file")
    parser.add_argument("--mrc_dir", type=str, required=True, help="Directory with MRC files")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, required=True, help="Output CS file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Bad region threshold (default: 0.6)")
    parser.add_argument("--min_area", type=int, default=400, help="Minimum area for bad regions in pixels (default: 400 = 20x20)")
    parser.add_argument("--distance", type=int, default=100, help="Distance threshold in pixels for removing particles near bad regions (default: 100)")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for inference (default: 512)")
    parser.add_argument("--overlap", type=int, default=64, help="Patch overlap for inference (default: 64)")
    parser.add_argument("--downsample_factor", type=int, default=1, help="Inference downsample factor (default: 1 = no downsampling)")
    parser.add_argument("--filter_on_work_resolution", action="store_true", help="If downsample_factor > 1, perform threshold/min-area/distance filtering in work-resolution mask space (faster, opt-in).")
    parser.add_argument("--csg_file", type=str, default=None, help="Path to original .csg file template (optional, for CryoSPARC import)")

    args = parser.parse_args()

    # Load CS data
    cs_data = load_cs(args.cs_file)

    # Filter
    filter_picks_with_model(cs_data, args.mrc_dir, args.model, args.output,
                           args.device, args.threshold,
                           min_area_pixels=args.min_area,
                           distance_threshold_pixels=args.distance,
                           original_cs_path=args.cs_file,
                           original_csg_path=args.csg_file,
                           patch_size=args.patch_size,
                           overlap=args.overlap,
                           downsample_factor=args.downsample_factor,
                           filter_on_work_resolution=args.filter_on_work_resolution)
