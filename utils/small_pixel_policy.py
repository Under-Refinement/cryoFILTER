from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np


DEFAULT_SMALL_PIXEL_CUTOFF_ANGSTROM = 1.20
DEFAULT_INFERENCE_PROFILE = "balanced"

SMALL_PIXEL_PROFILE_CONFIGS = {
    "fast": {
        "overlap": 128,
        "use_tta": False,
    },
    "balanced": {
        "overlap": 160,
        "use_tta": False,
    },
    "quality": {
        "overlap": 192,
        "use_tta": True,
    },
}


@dataclass(frozen=True)
class InferencePolicy:
    policy_name: str
    inference_profile: str
    pixel_size_angstrom: Optional[float]
    cutoff_angstrom: float
    auto_enabled: bool
    inference_recipe: str
    target_pixel_size: float
    multiscale_targets: tuple[float, ...]
    normalization_method: str
    overlap: int
    adaptive_downsample: bool
    adaptive_downsample_method: str
    multiscale_psd_source: str
    blending_window: str
    blending_edge_px: int
    use_tta: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["multiscale_targets"] = [float(v) for v in self.multiscale_targets]
        return payload


def _clean_pixel_size(pixel_size_angstrom: Optional[float]) -> Optional[float]:
    if pixel_size_angstrom is None:
        return None
    try:
        value = float(pixel_size_angstrom)
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def resolve_small_pixel_inference_policy(
    *,
    pixel_size_angstrom: Optional[float],
    standard_inference_recipe: str,
    standard_target_pixel_size: float,
    standard_multiscale_targets: Sequence[float],
    standard_normalization_method: str,
    standard_overlap: int,
    standard_adaptive_downsample: bool,
    standard_adaptive_downsample_method: str,
    standard_multiscale_psd_source: str,
    standard_blending_window: str,
    standard_blending_edge_px: int,
    standard_use_tta: bool,
    inference_profile: str = DEFAULT_INFERENCE_PROFILE,
    auto_enabled: bool = True,
    cutoff_angstrom: float = DEFAULT_SMALL_PIXEL_CUTOFF_ANGSTROM,
) -> InferencePolicy:
    px = _clean_pixel_size(pixel_size_angstrom)
    profile_name = str(inference_profile).strip().lower() or DEFAULT_INFERENCE_PROFILE
    if profile_name not in SMALL_PIXEL_PROFILE_CONFIGS:
        valid = ", ".join(sorted(SMALL_PIXEL_PROFILE_CONFIGS))
        raise ValueError(f"Unknown inference profile '{inference_profile}'. Expected one of: {valid}")
    use_small = bool(auto_enabled) and px is not None and px <= float(cutoff_angstrom)
    if use_small:
        profile = SMALL_PIXEL_PROFILE_CONFIGS[profile_name]
        use_tta = bool(profile["use_tta"] or standard_use_tta)
        return InferencePolicy(
            policy_name="small_pixel_auto",
            inference_profile=profile_name,
            pixel_size_angstrom=px,
            cutoff_angstrom=float(cutoff_angstrom),
            auto_enabled=bool(auto_enabled),
            inference_recipe="adaptive_resample",
            target_pixel_size=2.0,
            multiscale_targets=(2.0,),
            normalization_method=str(standard_normalization_method),
            overlap=int(profile["overlap"]),
            adaptive_downsample=True,
            adaptive_downsample_method="fourier",
            multiscale_psd_source="patch",
            blending_window="tukey",
            blending_edge_px=16,
            use_tta=use_tta,
        )

    targets = tuple(float(v) for v in standard_multiscale_targets)
    if not targets:
        targets = (float(standard_target_pixel_size),)
    return InferencePolicy(
        policy_name="standard",
        inference_profile=profile_name,
        pixel_size_angstrom=px,
        cutoff_angstrom=float(cutoff_angstrom),
        auto_enabled=bool(auto_enabled),
        inference_recipe=str(standard_inference_recipe),
        target_pixel_size=float(standard_target_pixel_size),
        multiscale_targets=targets,
        normalization_method=str(standard_normalization_method),
        overlap=int(standard_overlap),
        adaptive_downsample=bool(standard_adaptive_downsample),
        adaptive_downsample_method=str(standard_adaptive_downsample_method),
        multiscale_psd_source=str(standard_multiscale_psd_source),
        blending_window=str(standard_blending_window),
        blending_edge_px=int(standard_blending_edge_px),
        use_tta=bool(standard_use_tta),
    )
