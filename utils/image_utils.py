"""
Utilities for image processing and visualization.
"""
import numpy as np
from typing import Any, Tuple, Optional, Dict
from scipy import ndimage
from scipy.ndimage import sobel, gaussian_filter
from scipy.stats import skew, kurtosis


def _normalize_image_percentile(
    image: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    normalized = (image - p_low) / (p_high - p_low + 1e-10)
    return np.clip(normalized, 0, 1)


def normalize_image(image: np.ndarray, percentile_low: float = 1.0,
                   percentile_high: float = 99.0, method: str = "percentile") -> np.ndarray:
    """
    Normalize image for display using percentile-based or robust scaling.
    
    Args:
        image: Input image array
        percentile_low: Lower percentile for scaling (used if method="percentile")
        percentile_high: Upper percentile for scaling (used if method="percentile")
        method: Normalization method. Supported values:
            "percentile", "percentile_wide", "percentile_extra_wide",
            "percentile_blend", "simplipytem", "simplipytem_local",
            "simplipytem_local8bit", or "robust".
        
    Returns:
        Normalized image (0-1 range)
    """
    if method in {"none", "identity"}:
        return np.asarray(image, dtype=np.float32)
    if method == "robust":
        return normalize_image_robust(image)
    if method == "percentile_wide":
        return _normalize_image_percentile(image, 0.5, 99.5)
    if method == "percentile_extra_wide":
        return _normalize_image_percentile(image, 0.25, 99.75)
    if method == "percentile_blend":
        percentile_img = _normalize_image_percentile(image, 0.5, 99.5)
        robust_img = normalize_image_robust(image)
        return np.clip(0.7 * percentile_img + 0.3 * robust_img, 0, 1)
    if method == "simplipytem":
        return normalize_image_simplipytem(image)
    if method in {"simplipytem_local", "simplipytem_local8bit"}:
        return normalize_image_simplipytem(
            image,
            saturation_pct=0.2,
            median_kernel=3,
            local_normalization_patches=8,
            local_normalization_padding_pct=30,
            local_normalization_pad=True,
            local_normalization_prescale_8bit=True,
        )
    return _normalize_image_percentile(image, percentile_low, percentile_high)


def _simplipytem_local_normalize(
    image: np.ndarray,
    *,
    num_patches: int,
    padding_pct: int = 15,
    pad: bool = True,
) -> np.ndarray:
    """Scale local patch medians to the global median, following SimpliPyTEM."""
    arr = np.asarray(image, dtype=np.float32)
    patches = max(0, int(num_patches))
    if patches <= 1:
        return arr.astype(np.float32, copy=False)

    h, w = int(arr.shape[0]), int(arr.shape[1])
    xconst = h // patches
    yconst = w // patches
    if xconst <= 0 or yconst <= 0:
        return arr.astype(np.float32, copy=False)

    global_median = float(np.nanmedian(arr))
    if not np.isfinite(global_median) or abs(global_median) < 1e-12:
        return arr.astype(np.float32, copy=False)

    pad_frac = max(0.0, float(int(padding_pct)) / 100.0) if pad else 0.0
    x_pad = int(round(pad_frac * xconst))
    y_pad = int(round(pad_frac * yconst))

    if pad:
        acc = np.zeros_like(arr, dtype=np.float32)
        cnt = np.zeros_like(arr, dtype=np.float32)
    else:
        out = arr.copy()

    for xi in range(patches):
        x_low = xi * xconst
        x_high = h if xi == patches - 1 else min(h, x_low + xconst + x_pad)
        for yi in range(patches):
            y_low = yi * yconst
            y_high = w if yi == patches - 1 else min(w, y_low + yconst + y_pad)
            local_patch = arr[x_low:x_high, y_low:y_high]
            if local_patch.size == 0:
                continue
            local_median = float(np.nanmedian(local_patch))
            if not np.isfinite(local_median) or abs(local_median) < 1e-12:
                scale = 1.0
            else:
                scale = global_median / local_median
            scaled_patch = local_patch * scale
            if pad:
                acc[x_low:x_high, y_low:y_high] += scaled_patch
                cnt[x_low:x_high, y_low:y_high] += 1.0
            else:
                out[x_low:x_high, y_low:y_high] = scaled_patch

    if not pad:
        return out.astype(np.float32, copy=False)

    normalized = arr.copy()
    valid = cnt > 0
    normalized[valid] = acc[valid] / cnt[valid]
    return normalized.astype(np.float32, copy=False)


def _minmax_to_8bit_float(image: np.ndarray) -> np.ndarray:
    """Return a float32 0..255 image matching SimpliPyTEM's 8-bit pre-scale."""
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[finite]
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    clean = np.where(finite, arr, lo)
    return (np.clip((clean - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.float32, copy=False)


def normalize_image_simplipytem(
    image: np.ndarray,
    saturation_pct: float = 0.2,
    median_kernel: int = 3,
    local_normalization_patches: int = 0,
    local_normalization_padding_pct: int = 15,
    local_normalization_pad: bool = True,
    local_normalization_prescale_8bit: bool = False,
) -> np.ndarray:
    """Training-safe SimpliPyTEM-style contrast normalization.

    This mirrors the figure/display contrast path while preserving the incoming
    spatial shape: optional local median scaling, optional 3x3 median cleanup,
    then percentile saturation to [0, 1].
    """
    from scipy.ndimage import median_filter

    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return arr

    work = arr
    if int(local_normalization_patches) > 1:
        if bool(local_normalization_prescale_8bit):
            work = _minmax_to_8bit_float(work)
        work = _simplipytem_local_normalize(
            work,
            num_patches=int(local_normalization_patches),
            padding_pct=int(local_normalization_padding_pct),
            pad=bool(local_normalization_pad),
        )

    kernel = max(0, int(median_kernel))
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        work = median_filter(work, size=kernel, mode="reflect").astype(np.float32, copy=False)

    sat = float(np.clip(float(saturation_pct), 0.0, 49.9))
    return _normalize_image_percentile(work, sat, 100.0 - sat)


def encode_pixel_size_scalar(
    pixel_size_angstrom: float,
    min_angstrom: float,
    max_angstrom: float,
    use_log_scale: bool = True,
) -> float:
    """
    Encode a pixel size to a bounded scalar in [0, 1].

    A log-space encoding preserves multiplicative scale differences, which is
    more natural for cryo-EM pixel spacings that span a broad range.
    """
    px = float(pixel_size_angstrom)
    lo = float(min_angstrom)
    hi = float(max_angstrom)
    if not np.isfinite(px) or px <= 0:
        px = lo if lo > 0 else 1.0
    lo = max(lo, 1e-6)
    hi = max(hi, lo + 1e-6)

    if use_log_scale:
        px = float(np.log(px))
        lo = float(np.log(lo))
        hi = float(np.log(hi))

    encoded = (px - lo) / (hi - lo + 1e-12)
    return float(np.clip(encoded, 0.0, 1.0))


def build_constant_pixel_size_channel(
    image_shape: Tuple[int, int],
    pixel_size_angstrom: float,
    min_angstrom: float,
    max_angstrom: float,
    use_log_scale: bool = True,
) -> np.ndarray:
    """Create a constant feature channel carrying the encoded pixel size."""
    encoded = encode_pixel_size_scalar(
        pixel_size_angstrom=pixel_size_angstrom,
        min_angstrom=min_angstrom,
        max_angstrom=max_angstrom,
        use_log_scale=use_log_scale,
    )
    return np.full(tuple(int(v) for v in image_shape), encoded, dtype=np.float32)


def normalize_image_robust(image: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """
    Normalize image using robust statistics (MC-inspired).
    
    Uses median and MAD (Median Absolute Deviation) for robust normalization,
    which is more resistant to outliers than percentile-based methods.
    
    Args:
        image: Input image array
        n_sigma: Number of standard deviations to use for scaling (default: 3.0)
                 Values within ±n_sigma * MAD from median are mapped to [0, 1]
        
    Returns:
        Normalized image (0-1 range)
    """
    if image.size == 0:
        return image
    
    # Compute robust statistics
    median = np.median(image)
    
    # Compute MAD (Median Absolute Deviation)
    # MAD = median(|x - median(x)|)
    mad = np.median(np.abs(image - median))
    
    # If MAD is very small (e.g., uniform image), fall back to IQR
    eps = 1e-10
    if mad < eps:
        q25 = np.percentile(image, 25.0)
        q75 = np.percentile(image, 75.0)
        iqr = q75 - q25
        if iqr < eps:
            # Very uniform image, use percentile-based as fallback
            p_low = np.percentile(image, 1.0)
            p_high = np.percentile(image, 99.0)
            if p_high - p_low < eps:
                # Completely uniform, return zeros or ones
                return np.zeros_like(image) if median < 0.5 else np.ones_like(image)
            normalized = (image - p_low) / (p_high - p_low + eps)
        else:
            # Use IQR-based normalization
            # Scale IQR to approximate standard deviation (IQR ≈ 1.35 * std for normal distribution)
            robust_std = iqr / 1.35
            lower_bound = median - n_sigma * robust_std
            upper_bound = median + n_sigma * robust_std
            normalized = (image - lower_bound) / (upper_bound - lower_bound + eps)
    else:
        # Use MAD-based normalization
        # Scale MAD to approximate standard deviation (MAD ≈ 0.6745 * std for normal distribution)
        robust_std = mad / 0.6745
        lower_bound = median - n_sigma * robust_std
        upper_bound = median + n_sigma * robust_std
        normalized = (image - lower_bound) / (upper_bound - lower_bound + eps)
    
    # Clip to [0, 1] range
    normalized = np.clip(normalized, 0, 1)
    
    return normalized


def create_mask_from_polygons(image_shape: Tuple[int, int], 
                              polygons: list) -> np.ndarray:
    """
    Create a binary mask from a list of polygons.
    
    Args:
        image_shape: (height, width) of the image
        polygons: List of polygons, each as Nx2 array of (x, y) coordinates
        
    Returns:
        Binary mask (1 inside polygons, 0 outside)
    """
    from matplotlib.path import Path as MPLPath
    from matplotlib.patches import PathPatch
    
    mask = np.zeros(image_shape, dtype=bool)
    
    for poly in polygons:
        if len(poly) < 3:
            continue
        
        path = MPLPath(poly)
        y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        inside = path.contains_points(points)
        mask |= inside.reshape(image_shape)
    
    return mask.astype(np.uint8)


def downsample_image(image: np.ndarray, max_size: int = 2048) -> Tuple[np.ndarray, float]:
    """
    Downsample image if it's too large for display.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Downsampled image and scale factor
    """
    h, w = image.shape
    max_dim = max(h, w)
    
    if max_dim <= max_size:
        return image, 1.0
    
    scale = max_size / max_dim
    new_h, new_w = int(h * scale), int(w * scale)
    
    downsampled = ndimage.zoom(image, (new_h / h, new_w / w), order=1)
    
    return downsampled, scale


def downsample_to_particle_size(image: np.ndarray, pixel_size_angstrom: float,
                                target_pixels_per_particle: float = 6.0,
                                particle_size_angstrom: float = 100.0) -> Tuple[np.ndarray, float, float]:
    """
    Downsample image to standardize resolution relative to particle size (MC-inspired).
    
    Ensures consistent spatial scale by downsampling to a target resolution where
    there are approximately target_pixels_per_particle pixels per particle.
    
    Args:
        image: Input image
        pixel_size_angstrom: Current pixel size in Angstrom per pixel
        target_pixels_per_particle: Target number of pixels per particle (default: 6.0)
        particle_size_angstrom: Particle size in Angstrom (default: 100.0)
        
    Returns:
        Tuple of (downsampled_image, scale_factor, new_pixel_size_angstrom)
        If no downsampling needed, returns (image, 1.0, pixel_size_angstrom)
    """
    # Calculate target pixel size
    # target_pixel_size = particle_size / target_pixels_per_particle
    target_pixel_size = particle_size_angstrom / target_pixels_per_particle
    
    # Calculate downsample factor
    # If current pixel size < target, we need to downsample
    # downsample_factor = target_pixel_size / pixel_size_angstrom
    if pixel_size_angstrom >= target_pixel_size:
        # Already at or below target resolution, no downsampling needed
        return image, 1.0, pixel_size_angstrom
    
    downsample_factor = target_pixel_size / pixel_size_angstrom
    
    # Downsample image
    h, w = image.shape
    new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
    
    # Ensure minimum size (don't downsample to too small)
    min_size = 256  # Minimum dimension size
    if new_h < min_size or new_w < min_size:
        # Limit downsample to maintain minimum size
        max_scale = min(h / min_size, w / min_size)
        downsample_factor = max_scale
        new_h, new_w = int(h / downsample_factor), int(w / downsample_factor)
    
    # Downsample using bilinear interpolation
    downsampled = ndimage.zoom(image, (new_h / h, new_w / w), order=1)
    
    # Calculate new pixel size
    new_pixel_size = pixel_size_angstrom * downsample_factor
    
    return downsampled, downsample_factor, new_pixel_size


def apply_lowpass_filter(image: np.ndarray, cutoff_angstrom: float, 
                         pixel_size_angstrom: float) -> np.ndarray:
    """
    Apply a Gaussian low-pass filter to the image.
    
    Args:
        image: Input image
        cutoff_angstrom: Cutoff frequency in Angstrom (lower = more blur)
        pixel_size_angstrom: Pixel size in Angstrom per pixel
        
    Returns:
        Filtered image
    """
    if cutoff_angstrom <= 0 or cutoff_angstrom > 1000:
        return image  # No filtering if invalid cutoff
    
    # Convert Angstrom cutoff to sigma in pixels
    # For Gaussian filter: sigma ≈ cutoff / (2 * sqrt(2 * ln(2)) * pixel_size)
    # Simplified: sigma ≈ cutoff / (2.35 * pixel_size) for FWHM relationship
    # Or more directly: sigma = cutoff / pixel_size gives reasonable results
    sigma_pixels = cutoff_angstrom / pixel_size_angstrom
    
    # Apply Gaussian filter
    filtered = ndimage.gaussian_filter(image, sigma=sigma_pixels)
    
    return filtered


def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0.0, 
                               contrast: float = 1.0) -> np.ndarray:
    """
    Adjust brightness and contrast of an image.
    
    Args:
        image: Input image (0-1 range)
        brightness: Brightness adjustment (-1 to 1, 0 = no change)
        contrast: Contrast multiplier (0.1 to 3.0, 1.0 = no change)
        
    Returns:
        Adjusted image (clipped to 0-1 range)
    """
    # Apply contrast: center at 0.5, scale, then shift back
    adjusted = (image - 0.5) * contrast + 0.5
    
    # Apply brightness: add offset
    adjusted = adjusted + brightness
    
    # Clip to valid range
    adjusted = np.clip(adjusted, 0, 1)
    
    return adjusted


def compute_power_spectrum(
    image: np.ndarray, 
    normalize: bool = True,
    use_radial_normalization: bool = False,
    clip_percentile: float = 0.75,  # Clip top 0.75% (top 0.5-1% is typical)
    return_stats: bool = False  # If True, return (psd, stats_dict) instead of just psd
) -> np.ndarray:
    """
    Compute power spectrum (Fourier space magnitude) of an image with improved normalization.
    Uses Fourier-space magnitude to complement real-space image information.
    
    The power spectrum is useful for detecting:
    - Crystalline ice patterns
    - Contamination
    - Astigmatism
    - Other periodic artifacts
    
    A) Log-compress + robust normalize PSD:
       - Uses log1p(psd) for log compression
       - Robust normalization using median/IQR (more robust than mean/std)
       - Clips extreme values (top 0.5-1%) to prevent network from cheating on spikes
    
    B) Radial normalization (optional):
       - Divides PSD by its radial average to highlight anisotropy/rings/ice contamination
       - Very common in cryo-EM because absolute power varies wildly
    
    C) Train/eval consistency:
       - Same windowing (caller should apply Hann), same FFT size, same normalization
    
    Args:
        image: Input image (2D numpy array)
        normalize: Whether to normalize the power spectrum to 0-1 range
        use_radial_normalization: If True, divide by radial average profile (highlights anisotropy)
        clip_percentile: Percentile to clip from top (default 0.75 = clip top 0.75%)
        
    Returns:
        Power spectrum (same shape as input, log-compressed and normalized)
    """
    # Apply 2D Hann window to reduce edge artifacts in PSD
    # Without windowing, patch edges inject broadband noise into the spectrum
    h, w = image.shape
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    hann_2d = np.outer(hann_y, hann_x)
    
    # Window the image
    image_windowed = image * hann_2d
    
    # Compute 2D FFT on windowed image
    fft = np.fft.fft2(image_windowed)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_shifted) ** 2
    
    # A) Log-compress.
    # PSD is non-negative (|FFT|^2), so log(PSD + eps) is safe and simple.
    eps = 1e-10
    power_spectrum_log = np.log(power_spectrum + eps)
    # Be explicit: never let NaNs/Infs propagate into percentiles/medians.
    power_spectrum_log = np.nan_to_num(
        power_spectrum_log,
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=float(np.finfo(np.float32).min),
    )
    
    # B) Optional: Radial normalization (divide by radial average profile)
    if use_radial_normalization:
        h, w = power_spectrum_log.shape
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate arrays
        y_coords, x_coords = np.ogrid[:h, :w]
        # Distance from center
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Compute radial average (binned by distance)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        num_bins = int(max_dist) + 1
        # Fast radial mean via bincount (vectorized; avoids Python loops)
        dist_flat = distances.astype(np.int32).ravel()
        dist_flat = np.clip(dist_flat, 0, num_bins - 1)
        psd_flat = power_spectrum_log.ravel()

        radial_sum = np.bincount(dist_flat, weights=psd_flat, minlength=num_bins).astype(np.float64)
        radial_counts = np.bincount(dist_flat, minlength=num_bins).astype(np.float64)
        radial_profile = radial_sum / (radial_counts + eps)
        
        # Interpolate radial profile back to full image
        bin_indices = np.clip(distances.astype(int), 0, num_bins - 1)
        radial_profile_image = radial_profile[bin_indices]
        
        # Divide by radial profile (subtract in log space = divide in linear space)
        power_spectrum_log = power_spectrum_log - radial_profile_image
        power_spectrum_log = np.nan_to_num(
            power_spectrum_log,
            nan=0.0,
            posinf=float(np.finfo(np.float32).max),
            neginf=float(np.finfo(np.float32).min),
        )
    
    if normalize:
        # A) Robust normalization using median/IQR (more robust than mean/std to outliers)
        if power_spectrum_log.size == 0:
            return np.zeros_like(image)
        
        try:
            # Compute robust statistics
            median_val = np.median(power_spectrum_log)
            q25 = np.percentile(power_spectrum_log, 25.0)
            q75 = np.percentile(power_spectrum_log, 75.0)
            iqr = q75 - q25
            
            if iqr < 1e-10:
                # Fallback to percentile-based if IQR is too small
                p_low = np.percentile(power_spectrum_log, 1.0)
                p_high = np.percentile(power_spectrum_log, 99.0)
                if np.isnan(p_low) or np.isnan(p_high) or (p_high - p_low) < 1e-10:
                    return np.zeros_like(image)
                power_spectrum_norm = (power_spectrum_log - p_low) / (p_high - p_low + 1e-10)
            else:
                # Robust normalization: (x - median) / IQR, then shift to [0, 1]
                # Use 1.5*IQR range around median (typical for outlier detection)
                power_spectrum_norm = (power_spectrum_log - median_val) / (1.5 * iqr + eps)
                # Shift to [0, 1] range (assume most values are within ±2 IQR)
                power_spectrum_norm = (power_spectrum_norm + 2.0) / 4.0
            
            # Clip extreme values (top clip_percentile %)
            # This prevents the network from learning to exploit a few extreme spikes
            clip_threshold = np.percentile(power_spectrum_norm, 100.0 - clip_percentile)
            power_spectrum_norm = np.clip(power_spectrum_norm, 0.0, clip_threshold)
            
            # Re-normalize to [0, 1] after clipping
            p_min = power_spectrum_norm.min()
            p_max = power_spectrum_norm.max()
            if (p_max - p_min) > 1e-10:
                power_spectrum_norm = (power_spectrum_norm - p_min) / (p_max - p_min + eps)
            
            power_spectrum_norm = np.clip(power_spectrum_norm, 0, 1)
            
            if return_stats:
                # Compute fingerprint statistics for verification
                stats = {
                    'min': float(power_spectrum_norm.min()),
                    'max': float(power_spectrum_norm.max()),
                    'mean': float(power_spectrum_norm.mean()),
                    'std': float(power_spectrum_norm.std()),
                    'median': float(np.median(power_spectrum_norm)),
                    'p25': float(np.percentile(power_spectrum_norm, 25.0)),
                    'p75': float(np.percentile(power_spectrum_norm, 75.0)),
                    'p1': float(np.percentile(power_spectrum_norm, 1.0)),
                    'p99': float(np.percentile(power_spectrum_norm, 99.0)),
                    'use_radial_normalization': use_radial_normalization,
                    'clip_percentile': clip_percentile
                }
                return power_spectrum_norm.astype(np.float32), stats
            
            return power_spectrum_norm.astype(np.float32)
            
        except Exception as e:
            # Fallback: return zero array if computation fails
            if return_stats:
                stats = {'error': str(e)}
                return np.zeros_like(image, dtype=np.float32), stats
            return np.zeros_like(image, dtype=np.float32)
    else:
        if return_stats:
            stats = {
                'min': float(power_spectrum_log.min()),
                'max': float(power_spectrum_log.max()),
                'mean': float(power_spectrum_log.mean()),
                'std': float(power_spectrum_log.std()),
                'median': float(np.median(power_spectrum_log)),
                'normalized': False
            }
            return power_spectrum_log.astype(np.float32), stats
        return power_spectrum_log.astype(np.float32)


def compute_radial_psd_profile(
    image: np.ndarray,
    num_bins: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute 1D radial power spectrum profile from image.
    
    This is MORE INFORMATIVE than 2D PSD for contamination detection because:
    - Carbon/contamination has excess LOW-frequency power
    - Ice has flatter spectrum
    - The signal is RADIALLY SYMMETRIC, so 2D PSD dilutes it
    
    Args:
        image: Input 2D image
        num_bins: Number of frequency bins (default 64)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        1D array of shape (num_bins,) with radial power at each frequency
    """
    eps = 1e-10
    h, w = image.shape
    
    # Apply 2D Hann window to reduce edge artifacts in PSD
    # Without windowing, patch edges inject broadband noise into the spectrum
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    hann_2d = np.outer(hann_y, hann_x)
    image_windowed = image * hann_2d
    
    # Compute 2D PSD (on windowed image)
    fft = np.fft.fft2(image_windowed)
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    
    # Log-compress
    psd_log = np.log(psd + eps)
    
    # Compute radial profile
    center_y, center_x = h // 2, w // 2
    
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Bin by distance
    max_dist = np.sqrt(center_y**2 + center_x**2)
    bin_edges = np.linspace(0, max_dist, num_bins + 1)
    
    radial_profile = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
        if mask.sum() > 0:
            radial_profile[i] = psd_log[mask].mean()
    
    if normalize:
        # Normalize to [0, 1]
        p_min, p_max = radial_profile.min(), radial_profile.max()
        if p_max - p_min > eps:
            radial_profile = (radial_profile - p_min) / (p_max - p_min)
    
    return radial_profile.astype(np.float32)


def compute_radial_psd_image(
    image: np.ndarray,
    num_bins: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute radial PSD profile and tile it back to 2D image.
    
    Each pixel gets the radial power at its frequency distance from center.
    This creates a 2D image where the value at each pixel is the average
    power at that spatial frequency - much cleaner than raw 2D PSD!
    
    Args:
        image: Input 2D image
        num_bins: Number of frequency bins
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        2D array same size as input, with radial power values
    """
    eps = 1e-10
    h, w = image.shape
    
    # Apply 2D Hann window to reduce edge artifacts in PSD
    # Without windowing, patch edges inject broadband noise into the spectrum
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    hann_2d = np.outer(hann_y, hann_x)
    image_windowed = image * hann_2d
    
    # Compute 2D PSD (on windowed image)
    fft = np.fft.fft2(image_windowed)
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    
    # Log-compress
    psd_log = np.log(psd + eps)
    
    # Compute radial profile
    center_y, center_x = h // 2, w // 2
    
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Bin by distance
    max_dist = np.sqrt(center_y**2 + center_x**2)
    bin_width = max_dist / num_bins
    
    # Fast binning
    dist_bins = (distances / (bin_width + eps)).astype(np.int32)
    dist_bins = np.clip(dist_bins, 0, num_bins - 1)
    
    # Compute mean in each bin
    radial_sum = np.bincount(dist_bins.ravel(), weights=psd_log.ravel(), minlength=num_bins)
    radial_counts = np.bincount(dist_bins.ravel(), minlength=num_bins)
    radial_profile = radial_sum / (radial_counts + eps)
    
    # Map back to 2D
    radial_image = radial_profile[dist_bins]
    
    if normalize:
        p_min, p_max = radial_image.min(), radial_image.max()
        if p_max - p_min > eps:
            radial_image = (radial_image - p_min) / (p_max - p_min)
    
    return radial_image.astype(np.float32)


def compute_multires_context_psd(
    full_image: np.ndarray,
    patch_center: tuple,
    context_sizes: tuple = (256, 512, 1024),
    bins_per_scale: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute radial PSD at multiple spatial resolutions centered on a patch.
    
    This captures GLOBAL features like carbon edges that span the entire micrograph:
    - Small context (256px): Local particle/contamination texture
    - Medium context (512px): Medium-scale carbon boundaries  
    - Large context (1024px+): Large carbon edges spanning micrograph
    
    The key insight: Carbon edges are 3000+ pixels long, but patches are 256px.
    By computing PSD on larger regions, we capture the low-frequency signature
    of carbon edges that wouldn't be visible in a small patch.
    
    Args:
        full_image: Full micrograph, shape (H, W)
        patch_center: Center coordinates of the patch (y, x)
        context_sizes: Sizes of regions to extract for PSD (e.g., 256, 512, 1024)
        bins_per_scale: Number of radial frequency bins per scale
        normalize: Whether to z-score normalize each profile
        
    Returns:
        Concatenated radial profiles, shape (bins_per_scale * len(context_sizes),)
    """
    eps = 1e-10
    cy, cx = patch_center
    h, w = full_image.shape
    
    all_profiles = []
    
    for size in context_sizes:
        half = size // 2
        
        # Calculate extraction bounds with reflection padding if near edge
        y0 = cy - half
        y1 = cy + half
        x0 = cx - half
        x1 = cx + half
        
        # Handle boundary cases with reflection padding
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - h)
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - w)
        
        # Clip to valid range
        y0_clip = max(0, y0)
        y1_clip = min(h, y1)
        x0_clip = max(0, x0)
        x1_clip = min(w, x1)
        
        region = full_image[y0_clip:y1_clip, x0_clip:x1_clip]
        
        # Pad if needed
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            region = np.pad(region, 
                          ((pad_top, pad_bottom), (pad_left, pad_right)), 
                          mode='reflect')
        
        # Ensure exact size (handle rounding)
        if region.shape[0] != size or region.shape[1] != size:
            # Crop or pad to exact size
            if region.shape[0] > size:
                region = region[:size, :]
            if region.shape[1] > size:
                region = region[:, :size]
            if region.shape[0] < size:
                region = np.pad(region, ((0, size - region.shape[0]), (0, 0)), mode='reflect')
            if region.shape[1] < size:
                region = np.pad(region, ((0, 0), (0, size - region.shape[1])), mode='reflect')
        
        # Apply 2D Hann window to reduce edge artifacts in PSD
        hann_y = np.hanning(size)
        hann_x = np.hanning(size)
        hann_2d = np.outer(hann_y, hann_x)
        region_windowed = region * hann_2d
        
        # Compute radial PSD profile on this windowed region
        fft = np.fft.fft2(region_windowed)
        fft_shifted = np.fft.fftshift(fft)
        psd = np.abs(fft_shifted) ** 2
        psd_log = np.log(psd + eps)
        
        # Radial averaging
        rh, rw = psd_log.shape
        center_y, center_x = rh // 2, rw // 2
        y_coords, x_coords = np.ogrid[:rh, :rw]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        max_dist = np.sqrt(center_y**2 + center_x**2)
        bin_width = max_dist / bins_per_scale
        dist_bins = (distances / (bin_width + eps)).astype(np.int32)
        dist_bins = np.clip(dist_bins, 0, bins_per_scale - 1)
        
        radial_sum = np.bincount(dist_bins.ravel(), weights=psd_log.ravel(), minlength=bins_per_scale)
        radial_counts = np.bincount(dist_bins.ravel(), minlength=bins_per_scale)
        profile = radial_sum / (radial_counts + eps)
        
        if normalize:
            # Z-score normalize each profile independently
            p_mean, p_std = profile.mean(), profile.std()
            if p_std > eps:
                profile = (profile - p_mean) / p_std
        
        all_profiles.append(profile)
    
    # Concatenate all scales
    return np.concatenate(all_profiles).astype(np.float32)


def compute_multires_psd_image(
    full_image: np.ndarray,
    patch_center: tuple,
    patch_size: int = 64,
    context_sizes: tuple = (64, 128, 256),
    bins_per_scale: int = 32,
) -> np.ndarray:
    """
    Compute multi-resolution context PSD and tile back to 2D image.
    
    Creates a 2D image where each pixel encodes the multi-scale radial PSD
    features for that location. Inner pixels get coarse-scale info, outer
    pixels get fine-scale info.
    
    Args:
        full_image: Full micrograph
        patch_center: Center of the patch (y, x)
        patch_size: Size of output image (should match training patch size)
        context_sizes: Context sizes for PSD computation
        bins_per_scale: Bins per scale
        
    Returns:
        2D array of shape (patch_size, patch_size) with multi-res PSD values
    """
    eps = 1e-10
    
    # Get multi-res PSD profile
    profile = compute_multires_context_psd(
        full_image, patch_center, context_sizes, bins_per_scale, normalize=True
    )
    
    # Create output image
    h = w = patch_size
    center_y, center_x = h // 2, w // 2
    
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    
    # Normalize distances to [0, 1]
    norm_dist = distances / (max_dist + eps)
    
    # Map profile to 2D based on radial distance
    n_scales = len(context_sizes)
    total_bins = bins_per_scale * n_scales
    
    # Map normalized distance to profile index
    profile_idx = (norm_dist * (total_bins - 1)).astype(np.int32)
    profile_idx = np.clip(profile_idx, 0, total_bins - 1)
    
    result = profile[profile_idx]
    
    # Normalize to [0, 1]
    p_min, p_max = result.min(), result.max()
    if p_max - p_min > eps:
        result = (result - p_min) / (p_max - p_min)
    
    return result.astype(np.float32)


_MULTISCALE_PSD_PLAN_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _get_multiscale_psd_plan(
    h: int,
    w: int,
    scales: tuple,
    min_pixel_radii: tuple,
    eps: float = 1e-10,
) -> Dict[str, Any]:
    """Cache geometry/binning data for repeated per-patch multiscale PSD calls."""
    scales_key = tuple(int(s) for s in scales)
    min_radii_key = tuple(int(r) for r in min_pixel_radii)
    key = (int(h), int(w), scales_key, min_radii_key)
    cached = _MULTISCALE_PSD_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    hann_y = np.hanning(h).astype(np.float32)
    hann_x = np.hanning(w).astype(np.float32)
    hann_2d = np.outer(hann_y, hann_x).astype(np.float32)

    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_dist = float(np.sqrt(center_y ** 2 + center_x ** 2))
    norm_dist = (distances / (max_dist + eps)).astype(np.float32)

    adaptive_scales = list(scales_key)
    for min_r in min_radii_key:
        required_bins = int(max_dist / (float(min_r) / 3.0))
        required_bins = min(max(required_bins, 16), 128)
        if required_bins not in adaptive_scales:
            adaptive_scales.append(required_bins)
    adaptive_scales = tuple(sorted(set(adaptive_scales)))

    dist_bins_list = []
    dist_bins_flat_list = []
    radial_counts_list = []
    for num_bins in adaptive_scales:
        bin_width = 1.0 / num_bins
        dist_bins = (norm_dist / (bin_width + eps)).astype(np.int32)
        dist_bins = np.clip(dist_bins, 0, num_bins - 1)
        dist_bins_list.append(dist_bins)
        dist_bins_flat = dist_bins.ravel()
        dist_bins_flat_list.append(dist_bins_flat)
        radial_counts_list.append(np.bincount(dist_bins_flat, minlength=num_bins))

    n_scales = len(adaptive_scales)
    if n_scales == 1:
        blend_weights = [np.ones_like(norm_dist, dtype=np.float32)]
    elif n_scales == 2:
        w_fine = norm_dist.astype(np.float32, copy=False)
        w_coarse = (1.0 - w_fine).astype(np.float32, copy=False)
        blend_weights = [w_coarse, w_fine]
    elif n_scales == 3:
        w_coarse = np.clip(1.0 - norm_dist * 2.5, 0, 1).astype(np.float32)
        w_fine = np.clip((norm_dist - 0.4) * 2.5, 0, 1).astype(np.float32)
        w_medium = np.clip(1.0 - w_coarse - w_fine, 0, 1).astype(np.float32)
        w_sum = (w_coarse + w_medium + w_fine + eps).astype(np.float32)
        blend_weights = [
            (w_coarse / w_sum).astype(np.float32),
            (w_medium / w_sum).astype(np.float32),
            (w_fine / w_sum).astype(np.float32),
        ]
    else:
        weights = []
        for i in range(n_scales):
            center = i / (n_scales - 1)
            sigma = 0.5 / (n_scales - 1)
            w_i = np.exp(-0.5 * ((norm_dist - center) / (sigma + eps)) ** 2).astype(np.float32)
            weights.append(w_i)
        weight_sum = np.sum(np.stack(weights, axis=0), axis=0).astype(np.float32) + np.float32(eps)
        blend_weights = [(w_i / weight_sum).astype(np.float32) for w_i in weights]

    plan = {
        "hann_2d": hann_2d,
        "norm_dist": norm_dist,
        "adaptive_scales": adaptive_scales,
        "dist_bins_list": dist_bins_list,
        "dist_bins_flat_list": dist_bins_flat_list,
        "radial_counts_list": radial_counts_list,
        "blend_weights": blend_weights,
    }
    _MULTISCALE_PSD_PLAN_CACHE[key] = plan
    return plan


def compute_multiscale_radial_psd_image(
    image: np.ndarray,
    scales: tuple = (16, 32, 64),  # frequency bins per scale (coarse to fine)
    normalize: bool = True,
    min_pixel_radii: tuple = (4, 8),  # Minimum pixel radii for high-freq features
    texture_mode: bool = False,  # Focus on fine texture (5-40 Å) for carbon detection
    highpass_cutoff: float = 0.2,  # In texture_mode, ignore frequencies below this (norm_dist)
) -> np.ndarray:
    """
    Compute multi-scale radial PSD image with hybrid frequency binning.
    
    Creates radial PSD at multiple frequency resolutions and combines them.
    This captures contamination signatures at different spatial scales:
    - Coarse (16 bins): Large-scale carbon vs ice (excess low-freq for carbon)
    - Medium (32 bins): Particle structure detection
    - Fine (64 bins): High-freq noise patterns
    
    The 'min_pixel_radii' parameter ensures fine texture is captured even
    at low-resolution datasets. These define explicit high-frequency bands
    at fixed pixel scales (4px, 8px features), preventing loss of detail.
    
    TEXTURE MODE (texture_mode=True):
    Focuses on FINE texture (5-40 Å) where carbon and ice differ most.
    Low-frequency information (80-320 Å) is suppressed because both carbon
    and ice look similar at those scales. This prevents the model from
    learning that PSD is useless and defaulting to "predict high everywhere."
    
    For 64×64 patch at 5 Å/px:
    - 40 Å wavelength → frequency 8 cycles → norm_dist ≈ 0.25
    - 10 Å wavelength → frequency 32 cycles → norm_dist ≈ 1.0
    - highpass_cutoff=0.2 keeps wavelengths < ~50 Å (fine texture only)
    
    The output encodes ALL scales in a single 2D image by weighting
    contributions from each scale based on radial distance.
    
    Args:
        image: Input 2D image
        scales: Tuple of bin counts for each scale (default: 16, 32, 64)
        normalize: Whether to normalize to [0, 1]
        min_pixel_radii: Minimum pixel radii for high-frequency preservation (default: 4, 8)
        texture_mode: If True, focus on high frequencies (fine texture) only
        highpass_cutoff: In texture_mode, ignore frequencies below this (default: 0.2)
        
    Returns:
        2D array same size as input, with multi-scale radial power values
    """
    eps = 1e-10
    h, w = image.shape
    plan = _get_multiscale_psd_plan(h, w, scales=scales, min_pixel_radii=min_pixel_radii, eps=eps)
    
    # Apply 2D Hann window to reduce edge artifacts in PSD
    # Without windowing, patch edges inject broadband noise into the spectrum
    hann_2d = plan["hann_2d"]
    image_windowed = image * hann_2d
    
    # Compute 2D PSD once (on windowed image)
    fft = np.fft.fft2(image_windowed)
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    psd_log = np.log(psd + eps)
    
    # Normalize distances to [0, 1] (cached per patch geometry)
    norm_dist = plan["norm_dist"]
    adaptive_scales = plan["adaptive_scales"]
    dist_bins_list = plan["dist_bins_list"]
    radial_counts_list = plan["radial_counts_list"]
    
    # Compute radial profiles at each scale
    scale_images = []
    for num_bins, dist_bins, radial_counts in zip(adaptive_scales, dist_bins_list, radial_counts_list):
        # Compute mean in each bin
        radial_sum = np.bincount(dist_bins.ravel(), weights=psd_log.ravel(), minlength=num_bins)
        radial_profile = radial_sum / (radial_counts + eps)
        
        # Map back to 2D
        scale_img = radial_profile[dist_bins]
        scale_images.append(scale_img)
    
    # Combine scales: weight by radial distance
    # Inner region (low freq) gets more weight from coarse scale
    # Outer region (high freq) gets more weight from fine scale
    combined = np.zeros_like(psd_log)
    
    # Create smooth blending weights based on distance
    # Generalized to handle any number of scales
    n_scales = len(adaptive_scales)
    if n_scales == 1:
        combined = scale_images[0]
    else:
        # Cached normalized blend weights per patch geometry.
        blend_weights = plan["blend_weights"]
        combined = sum(w * img for w, img in zip(blend_weights, scale_images))
    
    # TEXTURE MODE: Apply high-pass filter to focus on fine texture (5-40 Å)
    # Low-frequency information (80-320 Å) is where carbon and ice look similar,
    # so we suppress it to help the model learn texture differences
    if texture_mode:
        # Create high-pass mask: 1 at high frequencies, 0 at low frequencies
        # Smooth transition to avoid artifacts
        transition_width = 0.1
        highpass_weight = np.clip((norm_dist - highpass_cutoff) / transition_width, 0, 1)
        # Apply weighting: high frequencies kept, low frequencies suppressed
        combined = combined * highpass_weight
        # Fill low-frequency region with mean of high-frequency region to avoid sharp edges
        highpass_region = norm_dist >= highpass_cutoff
        if highpass_region.any():
            mean_highfreq = combined[highpass_region].mean()
            combined = np.where(highpass_region, combined, mean_highfreq * 0.5)
    
    if normalize:
        p_min, p_max = combined.min(), combined.max()
        if p_max - p_min > eps:
            combined = (combined - p_min) / (p_max - p_min)
    
    return combined.astype(np.float32)


def compute_multiscale_radial_psd_image_legacy(
    image: np.ndarray,
    scales: tuple = (16, 32, 64),
    normalize: bool = True,
    min_pixel_radii: tuple = (4, 8),
    texture_mode: bool = False,
    highpass_cutoff: float = 0.2,
) -> np.ndarray:
    """Reproduce the uncached float64 PSD path used by the February panel run."""
    eps = 1e-10
    h, w = image.shape

    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    hann_2d = np.outer(hann_y, hann_x)
    image_windowed = image * hann_2d

    fft = np.fft.fft2(image_windowed)
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    psd_log = np.log(psd + eps)

    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
    norm_dist = distances / (max_dist + eps)

    adaptive_scales = list(scales)
    for min_r in min_pixel_radii:
        required_bins = int(max_dist / (min_r / 3.0))
        required_bins = min(max(required_bins, 16), 128)
        if required_bins not in adaptive_scales:
            adaptive_scales.append(required_bins)
    adaptive_scales = sorted(set(adaptive_scales))

    scale_images = []
    for num_bins in adaptive_scales:
        bin_width = 1.0 / num_bins
        dist_bins = (norm_dist / (bin_width + eps)).astype(np.int32)
        dist_bins = np.clip(dist_bins, 0, num_bins - 1)
        radial_sum = np.bincount(dist_bins.ravel(), weights=psd_log.ravel(), minlength=num_bins)
        radial_counts = np.bincount(dist_bins.ravel(), minlength=num_bins)
        radial_profile = radial_sum / (radial_counts + eps)
        scale_images.append(radial_profile[dist_bins])

    n_scales = len(adaptive_scales)
    if n_scales == 1:
        combined = scale_images[0]
    elif n_scales == 2:
        w_fine = norm_dist
        w_coarse = 1.0 - w_fine
        combined = w_coarse * scale_images[0] + w_fine * scale_images[1]
    elif n_scales == 3:
        w_coarse = np.clip(1.0 - norm_dist * 2.5, 0, 1)
        w_fine = np.clip((norm_dist - 0.4) * 2.5, 0, 1)
        w_medium = np.clip(1.0 - w_coarse - w_fine, 0, 1)
        w_sum = w_coarse + w_medium + w_fine + eps
        combined = (
            w_coarse * scale_images[0]
            + w_medium * scale_images[1]
            + w_fine * scale_images[2]
        ) / w_sum
    else:
        weights = []
        for index, _ in enumerate(adaptive_scales):
            center = index / (n_scales - 1)
            sigma = 0.5 / (n_scales - 1)
            weights.append(np.exp(-0.5 * ((norm_dist - center) / (sigma + eps)) ** 2))
        weight_stack = np.stack(weights, axis=0)
        weight_sum = weight_stack.sum(axis=0) + eps
        combined = sum(weight * scale for weight, scale in zip(weights, scale_images)) / weight_sum

    if texture_mode:
        transition_width = 0.1
        highpass_weight = np.clip((norm_dist - highpass_cutoff) / transition_width, 0, 1)
        combined = combined * highpass_weight
        highpass_region = norm_dist >= highpass_cutoff
        if highpass_region.any():
            mean_highfreq = combined[highpass_region].mean()
            combined = np.where(highpass_region, combined, mean_highfreq * 0.5)

    if normalize:
        p_min, p_max = combined.min(), combined.max()
        if p_max - p_min > eps:
            combined = (combined - p_min) / (p_max - p_min)
    return combined.astype(np.float32)


def compute_multiscale_radial_psd_channels(
    image: np.ndarray,
    scales: tuple = (16, 32, 64),
    normalize: bool = True,
    min_pixel_radii: tuple = (4, 8),
    texture_mode: bool = False,
    highpass_cutoff: float = 0.2,
    use_adaptive_scales: bool = False,
) -> np.ndarray:
    """
    Compute one PSD image per radial scale (no cross-scale fusion).

    This is useful when each radial scale is provided to the model as a separate
    input channel instead of being fused into a single PSD map.

    Args:
        image: Input 2D image.
        scales: Tuple of bin counts for each scale.
        normalize: Whether to normalize each channel independently to [0, 1].
        min_pixel_radii: Optional minimum pixel radii used only when
            use_adaptive_scales=True.
        texture_mode: If True, apply high-pass emphasis on each channel.
        highpass_cutoff: Normalized radial cutoff used in texture_mode.
        use_adaptive_scales: If True, include automatically-derived scales from
            min_pixel_radii. If False (default), returns exactly len(scales)
            channels in the order provided by `scales`.

    Returns:
        Array with shape (C, H, W), float32.
    """
    eps = 1e-10
    h, w = image.shape
    requested_scales = tuple(int(s) for s in scales)
    if len(requested_scales) == 0:
        raise ValueError("scales must contain at least one value")

    plan = _get_multiscale_psd_plan(
        h,
        w,
        scales=requested_scales,
        min_pixel_radii=(min_pixel_radii if use_adaptive_scales else tuple()),
        eps=eps,
    )

    # Use exactly requested scales by default so channel count is stable.
    if use_adaptive_scales:
        active_scales = tuple(int(s) for s in plan["adaptive_scales"])
        dist_bins_list = list(plan["dist_bins_list"])
        radial_counts_list = list(plan["radial_counts_list"])
    else:
        active_scales = requested_scales
        dist_bins_map = {int(s): bins for s, bins in zip(plan["adaptive_scales"], plan["dist_bins_list"])}
        radial_counts_map = {int(s): cnt for s, cnt in zip(plan["adaptive_scales"], plan["radial_counts_list"])}
        dist_bins_list = [dist_bins_map[int(s)] for s in active_scales]
        radial_counts_list = [radial_counts_map[int(s)] for s in active_scales]

    hann_2d = plan["hann_2d"]
    norm_dist = plan["norm_dist"]

    image_windowed = image * hann_2d
    fft = np.fft.fft2(image_windowed)
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    psd_log = np.log(psd + eps)

    scale_images = []
    for num_bins, dist_bins, radial_counts in zip(active_scales, dist_bins_list, radial_counts_list):
        radial_sum = np.bincount(dist_bins.ravel(), weights=psd_log.ravel(), minlength=num_bins)
        radial_profile = radial_sum / (radial_counts + eps)
        scale_img = radial_profile[dist_bins]
        scale_images.append(np.asarray(scale_img, dtype=np.float32))

    channels = np.stack(scale_images, axis=0).astype(np.float32, copy=False)

    if texture_mode:
        transition_width = 0.1
        highpass_weight = np.clip((norm_dist - highpass_cutoff) / transition_width, 0, 1).astype(np.float32)
        channels = channels * highpass_weight[None, :, :]
        highpass_region = norm_dist >= highpass_cutoff
        if highpass_region.any():
            mean_highfreq = channels[:, highpass_region].mean(axis=1)
            channels = np.where(
                highpass_region[None, :, :],
                channels,
                (mean_highfreq * 0.5)[:, None, None],
            )

    if normalize:
        flat = channels.reshape(channels.shape[0], -1)
        p_min = flat.min(axis=1)
        p_max = flat.max(axis=1)
        denom = p_max - p_min
        valid = denom > eps
        if np.any(valid):
            channels_valid = channels[valid]
            channels[valid] = (channels_valid - p_min[valid, None, None]) / (denom[valid, None, None])

    return channels.astype(np.float32, copy=False)


def compute_multiscale_radial_psd_image_batch(
    images: np.ndarray,
    scales: tuple = (16, 32, 64),
    normalize: bool = True,
    min_pixel_radii: tuple = (4, 8),
    texture_mode: bool = False,
    highpass_cutoff: float = 0.2,
) -> np.ndarray:
    """
    Batched equivalent of compute_multiscale_radial_psd_image (same output semantics).

    Args:
        images: Input array of shape (B, H, W)
        scales: Tuple of bin counts for each scale
        normalize: Whether to normalize each sample to [0, 1]
        min_pixel_radii: Minimum pixel radii for high-frequency preservation
        texture_mode: If True, focus on high frequencies only
        highpass_cutoff: In texture_mode, ignore frequencies below this normalized radius

    Returns:
        Array of shape (B, H, W), float32
    """
    eps = 1e-10
    imgs = np.asarray(images, dtype=np.float32)
    if imgs.ndim != 3:
        raise ValueError(f"Expected images shape (B,H,W), got {imgs.shape}")
    if imgs.shape[0] == 0:
        return np.empty_like(imgs, dtype=np.float32)

    b, h, w = imgs.shape
    plan = _get_multiscale_psd_plan(h, w, scales=scales, min_pixel_radii=min_pixel_radii, eps=eps)

    hann_2d = plan["hann_2d"]
    norm_dist = plan["norm_dist"]
    adaptive_scales = plan["adaptive_scales"]
    dist_bins_list = plan["dist_bins_list"]
    dist_bins_flat_list = plan["dist_bins_flat_list"]
    radial_counts_list = plan["radial_counts_list"]

    image_windowed = imgs * hann_2d[None, :, :]

    fft = np.fft.fft2(image_windowed, axes=(-2, -1))
    fft_shifted = np.fft.fftshift(fft, axes=(-2, -1))
    psd = np.abs(fft_shifted) ** 2
    psd_log = np.log(psd + eps)
    psd_log_flat = psd_log.reshape(b, -1)

    n_scales = len(adaptive_scales)
    if n_scales == 1:
        combined = None
    else:
        combined = np.zeros_like(psd_log)

    blend_weights = plan["blend_weights"]
    for i_scale, (num_bins, dist_bins, dist_bins_flat, radial_counts) in enumerate(
        zip(adaptive_scales, dist_bins_list, dist_bins_flat_list, radial_counts_list)
    ):
        radial_sum = np.stack(
            [
                np.bincount(dist_bins_flat, weights=psd_log_flat[i], minlength=num_bins)
                for i in range(b)
            ],
            axis=0,
        )
        radial_profile = radial_sum / (radial_counts[None, :] + eps)
        scale_img = radial_profile[:, dist_bins]

        if n_scales == 1:
            combined = scale_img
        else:
            combined = combined + (blend_weights[i_scale][None, :, :] * scale_img)

    if combined is None:
        combined = np.zeros((b, h, w), dtype=np.float32)

    if texture_mode:
        transition_width = 0.1
        highpass_weight = np.clip((norm_dist - highpass_cutoff) / transition_width, 0, 1)
        combined = combined * highpass_weight[None, :, :]
        highpass_region = norm_dist >= highpass_cutoff
        if highpass_region.any():
            mean_highfreq = combined[:, highpass_region].mean(axis=1)
            combined = np.where(
                highpass_region[None, :, :],
                combined,
                (mean_highfreq * 0.5)[:, None, None],
            )

    if normalize:
        flat = combined.reshape(b, -1)
        p_min = flat.min(axis=1)
        p_max = flat.max(axis=1)
        denom = p_max - p_min
        valid = denom > eps
        if np.any(valid):
            combined_valid = combined[valid]
            combined[valid] = (combined_valid - p_min[valid, None, None]) / (denom[valid, None, None])

    return combined.astype(np.float32)


_FREQUENCY_RADIUS_CACHE: Dict[tuple, np.ndarray] = {}
_WINDOWED_PSD_PLAN_CACHE: Dict[tuple, Dict[str, np.ndarray]] = {}


def _get_windowed_psd_plan(h: int, w: int) -> Dict[str, np.ndarray]:
    """Cache reusable geometry for repeated PSD computations at fixed shape."""
    key = (int(h), int(w))
    cached = _WINDOWED_PSD_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    hann_y = np.hanning(h).astype(np.float32)
    hann_x = np.hanning(w).astype(np.float32)
    hann_2d = np.outer(hann_y, hann_x).astype(np.float32)

    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
    num_bins = int(max_dist) + 1
    dist_bins = np.clip(distances.astype(np.int32), 0, num_bins - 1)
    dist_flat = dist_bins.ravel()
    radial_counts = np.bincount(dist_flat, minlength=num_bins).astype(np.float64)

    plan = {
        "hann_2d": hann_2d,
        "dist_bins": dist_bins,
        "dist_flat": dist_flat,
        "radial_counts": radial_counts,
        "num_bins": np.array([num_bins], dtype=np.int32),
    }
    _WINDOWED_PSD_PLAN_CACHE[key] = plan
    return plan


def _compute_windowed_log_psd(image: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return log power spectrum after Hann windowing."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    h, w = img.shape
    plan = _get_windowed_psd_plan(h, w)
    fft = np.fft.fft2(img * plan["hann_2d"])
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    return np.log(psd + eps)


def _subtract_radial_profile(psd_log: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Radial-whiten a log-PSD image by subtracting its radial average."""
    h, w = psd_log.shape
    plan = _get_windowed_psd_plan(h, w)
    dist_flat = plan["dist_flat"]
    psd_flat = psd_log.ravel()
    num_bins = int(plan["num_bins"][0])
    radial_sum = np.bincount(dist_flat, weights=psd_flat, minlength=num_bins).astype(np.float64)
    radial_profile = radial_sum / (plan["radial_counts"] + eps)
    radial_profile_image = radial_profile[plan["dist_bins"]]
    whitened = psd_log - radial_profile_image
    return np.nan_to_num(
        whitened,
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=float(np.finfo(np.float32).min),
    )


def _robust_normalize_psd_map(
    psd_map: np.ndarray,
    clip_percentile: float = 0.75,
    eps: float = 1e-10,
) -> np.ndarray:
    """Match the repo's PSD normalization: median/IQR + clipping + [0,1] remap."""
    arr = np.asarray(psd_map, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    median_val = np.median(arr)
    q25 = np.percentile(arr, 25.0)
    q75 = np.percentile(arr, 75.0)
    iqr = q75 - q25

    if iqr < eps:
        p_low = np.percentile(arr, 1.0)
        p_high = np.percentile(arr, 99.0)
        if np.isnan(p_low) or np.isnan(p_high) or (p_high - p_low) < eps:
            return np.zeros_like(arr, dtype=np.float32)
        norm = (arr - p_low) / (p_high - p_low + eps)
    else:
        norm = (arr - median_val) / (1.5 * iqr + eps)
        norm = (norm + 2.0) / 4.0

    clip_threshold = np.percentile(norm, 100.0 - clip_percentile)
    norm = np.clip(norm, 0.0, clip_threshold)
    p_min = norm.min()
    p_max = norm.max()
    if (p_max - p_min) > eps:
        norm = (norm - p_min) / (p_max - p_min + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def _robust_normalize_psd_map_batch(
    psd_maps: np.ndarray,
    clip_percentile: float = 0.75,
    eps: float = 1e-10,
) -> np.ndarray:
    """Batch equivalent of _robust_normalize_psd_map for arrays shaped (B,H,W)."""
    arr = np.asarray(psd_maps, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected PSD maps shape (B,H,W), got {arr.shape}")
    if arr.shape[0] == 0:
        return np.asarray(arr, dtype=np.float32)

    b = arr.shape[0]
    flat = arr.reshape(b, -1)
    norm = np.zeros_like(flat, dtype=np.float32)

    median_val = np.median(flat, axis=1)
    q25 = np.percentile(flat, 25.0, axis=1)
    q75 = np.percentile(flat, 75.0, axis=1)
    iqr = q75 - q25
    valid_iqr = iqr >= eps

    if np.any(valid_iqr):
        idx = np.where(valid_iqr)[0]
        norm[idx] = ((flat[idx] - median_val[idx, None]) / (1.5 * iqr[idx, None] + eps)).astype(np.float32)
        norm[idx] = (norm[idx] + 2.0) / 4.0

    invalid_iqr = ~valid_iqr
    if np.any(invalid_iqr):
        idx = np.where(invalid_iqr)[0]
        p_low = np.percentile(flat[idx], 1.0, axis=1)
        p_high = np.percentile(flat[idx], 99.0, axis=1)
        denom = p_high - p_low
        valid_range = denom >= eps
        if np.any(valid_range):
            idx_valid = idx[valid_range]
            norm[idx_valid] = ((flat[idx_valid] - p_low[valid_range, None]) / (denom[valid_range, None] + eps)).astype(np.float32)

    clip_threshold = np.percentile(norm, 100.0 - clip_percentile, axis=1)
    norm = np.clip(norm, 0.0, clip_threshold[:, None])
    p_min = norm.min(axis=1)
    p_max = norm.max(axis=1)
    denom = p_max - p_min
    valid_range = denom >= eps
    if np.any(valid_range):
        idx = np.where(valid_range)[0]
        norm[idx] = ((norm[idx] - p_min[idx, None]) / (denom[idx, None] + eps)).astype(np.float32)
    return np.clip(norm.reshape(arr.shape), 0.0, 1.0).astype(np.float32, copy=False)


def _get_frequency_radius_map(h: int, w: int, pixel_size_angstrom: float) -> np.ndarray:
    """Cache frequency-radius grids in physical units (Å^-1)."""
    px = float(pixel_size_angstrom)
    if not np.isfinite(px) or px <= 0:
        raise ValueError(f"pixel_size_angstrom must be > 0, got {pixel_size_angstrom}")
    key = (int(h), int(w), round(px, 8))
    cached = _FREQUENCY_RADIUS_CACHE.get(key)
    if cached is not None:
        return cached
    fy = np.fft.fftshift(np.fft.fftfreq(int(h), d=px))
    fx = np.fft.fftshift(np.fft.fftfreq(int(w), d=px))
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    freq_radius = np.sqrt(fy_grid ** 2 + fx_grid ** 2).astype(np.float32)
    _FREQUENCY_RADIUS_CACHE[key] = freq_radius
    return freq_radius


def compute_frequency_band_psd_channels(
    image: np.ndarray,
    pixel_size_angstrom: float,
    frequency_bands: tuple,
    normalize: bool = True,
    use_radial_normalization: bool = True,
    include_full_spectrum_channel: bool = False,
    include_anisotropy_channel: bool = False,
    anisotropy_min_frequency: float = 0.0,
    clip_percentile: float = 0.75,
) -> np.ndarray:
    """
    Build physically defined PSD-band channels for one patch.

    Each frequency band contributes one channel whose annulus is filled by the
    mean normalized PSD response inside that band. This preserves per-band
    strength while keeping each channel spatially aligned in Fourier space.

    Optionally append the full normalized PSD and/or one anisotropy channel
    containing the normalized radial-whitened PSD above
    `anisotropy_min_frequency`, which helps capture streaks or other
    directional artifacts.
    """
    bands = []
    for band in tuple(frequency_bands):
        if len(band) != 2:
            raise ValueError(f"Expected (low, high) band pair, got {band!r}")
        low, high = float(band[0]), float(band[1])
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
            raise ValueError(f"Invalid frequency band {band!r}")
        bands.append((low, high))
    if not bands and not include_full_spectrum_channel and not include_anisotropy_channel:
        raise ValueError("At least one frequency band, full-spectrum channel, or anisotropy channel is required")

    eps = 1e-10
    base = _compute_windowed_log_psd(image, eps=eps)
    if use_radial_normalization:
        base = _subtract_radial_profile(base, eps=eps)
    base = np.nan_to_num(
        base,
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=float(np.finfo(np.float32).min),
    )
    base_norm = _robust_normalize_psd_map(
        base,
        clip_percentile=clip_percentile,
        eps=eps,
    ) if normalize else base.astype(np.float32, copy=False)

    h, w = base.shape
    freq_radius = _get_frequency_radius_map(h, w, pixel_size_angstrom)
    channels = []

    if include_full_spectrum_channel:
        channels.append(base_norm.astype(np.float32, copy=False))

    for low, high in bands:
        mask = (freq_radius >= low) & (freq_radius < high)
        channel = np.zeros((h, w), dtype=np.float32)
        if np.any(mask):
            channel[mask] = float(base_norm[mask].mean())
        channels.append(channel)

    if include_anisotropy_channel:
        anis_mask = freq_radius >= float(max(0.0, anisotropy_min_frequency))
        anis = np.zeros((h, w), dtype=np.float32)
        anis[anis_mask] = base_norm[anis_mask]
        channels.append(anis)

    if not channels:
        return np.zeros((0, h, w), dtype=np.float32)
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def compute_frequency_band_psd_channels_batch(
    images: np.ndarray,
    pixel_size_angstrom: float,
    frequency_bands: tuple,
    normalize: bool = True,
    use_radial_normalization: bool = True,
    include_full_spectrum_channel: bool = False,
    include_anisotropy_channel: bool = False,
    anisotropy_min_frequency: float = 0.0,
    clip_percentile: float = 0.75,
) -> np.ndarray:
    """
    Batched equivalent of compute_frequency_band_psd_channels for arrays shaped (B,H,W).
    """
    imgs = np.asarray(images, dtype=np.float32)
    if imgs.ndim != 3:
        raise ValueError(f"Expected images shape (B,H,W), got {imgs.shape}")

    bands = []
    for band in tuple(frequency_bands):
        if len(band) != 2:
            raise ValueError(f"Expected (low, high) band pair, got {band!r}")
        low, high = float(band[0]), float(band[1])
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
            raise ValueError(f"Invalid frequency band {band!r}")
        bands.append((low, high))
    if not bands and not include_full_spectrum_channel and not include_anisotropy_channel:
        raise ValueError("At least one frequency band, full-spectrum channel, or anisotropy channel is required")

    b, h, w = imgs.shape
    if b == 0:
        n_channels = len(bands) + (1 if include_full_spectrum_channel else 0) + (1 if include_anisotropy_channel else 0)
        return np.zeros((0, n_channels, h, w), dtype=np.float32)

    eps = 1e-10
    plan = _get_windowed_psd_plan(h, w)
    fft = np.fft.fft2(imgs * plan["hann_2d"][None, :, :], axes=(-2, -1))
    fft_shifted = np.fft.fftshift(fft, axes=(-2, -1))
    psd = np.abs(fft_shifted) ** 2
    base = np.log(psd + eps)

    if use_radial_normalization:
        base_flat = base.reshape(b, -1)
        num_bins = int(plan["num_bins"][0])
        radial_sum = np.stack(
            [
                np.bincount(plan["dist_flat"], weights=base_flat[i], minlength=num_bins).astype(np.float64)
                for i in range(b)
            ],
            axis=0,
        )
        radial_profile = radial_sum / (plan["radial_counts"][None, :] + eps)
        base = base - radial_profile[:, plan["dist_bins"]]

    base = np.nan_to_num(
        base,
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=float(np.finfo(np.float32).min),
    )
    base_norm = _robust_normalize_psd_map_batch(
        base,
        clip_percentile=clip_percentile,
        eps=eps,
    ) if normalize else base.astype(np.float32, copy=False)

    freq_radius = _get_frequency_radius_map(h, w, pixel_size_angstrom)
    channels = []

    if include_full_spectrum_channel:
        channels.append(base_norm.astype(np.float32, copy=False))

    for low, high in bands:
        mask = (freq_radius >= low) & (freq_radius < high)
        channel = np.zeros((b, h, w), dtype=np.float32)
        if np.any(mask):
            mean_vals = base_norm[:, mask].mean(axis=1).astype(np.float32)
            channel[:, mask] = mean_vals[:, None]
        channels.append(channel)

    if include_anisotropy_channel:
        anis_mask = freq_radius >= float(max(0.0, anisotropy_min_frequency))
        anis = np.zeros((b, h, w), dtype=np.float32)
        anis[:, anis_mask] = base_norm[:, anis_mask]
        channels.append(anis)

    if not channels:
        return np.zeros((b, 0, h, w), dtype=np.float32)
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def compute_edge_magnitude(
    image: np.ndarray,
    sigma: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute edge magnitude using Sobel filters.
    
    This captures SHARP TRANSITIONS like carbon edges that radial PSD misses.
    Carbon interiors are smooth (low edge), but edges are sharp (high edge).
    Particles also have edges but they're more structured/regular.
    
    Args:
        image: Input 2D image (should be normalized)
        sigma: Gaussian smoothing sigma before edge detection (reduces noise)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        2D array of edge magnitudes
    """
    eps = 1e-10
    
    # Optional smoothing to reduce noise
    if sigma > 0:
        smoothed = gaussian_filter(image, sigma=sigma)
    else:
        smoothed = image
    
    # Compute Sobel gradients
    grad_x = sobel(smoothed, axis=1)  # Horizontal edges
    grad_y = sobel(smoothed, axis=0)  # Vertical edges
    
    # Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    if normalize:
        p_min, p_max = magnitude.min(), magnitude.max()
        if p_max - p_min > eps:
            magnitude = (magnitude - p_min) / (p_max - p_min)
    
    return magnitude.astype(np.float32)


def compute_local_variance(
    image: np.ndarray,
    window_size: int = 16,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute local variance map.
    
    Carbon regions have LOW local variance (smooth/uniform).
    Particle regions have HIGH local variance (structured).
    This could help distinguish carbon from particles.
    
    Args:
        image: Input 2D image
        window_size: Size of local window for variance computation
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        2D array of local variance values
    """
    eps = 1e-10
    
    # Use uniform filter for local mean and mean of squares
    kernel_size = window_size
    local_mean = ndimage.uniform_filter(image, size=kernel_size)
    local_sq_mean = ndimage.uniform_filter(image**2, size=kernel_size)
    
    # Variance = E[X^2] - E[X]^2
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  # Numerical stability
    
    if normalize:
        p_min, p_max = local_var.min(), local_var.max()
        if p_max - p_min > eps:
            local_var = (local_var - p_min) / (p_max - p_min)
    
    return local_var.astype(np.float32)


# =============================================================================
# DIRECTIONAL EDGE FEATURES (v2 - replaces complex global features)
# =============================================================================
# Simple 3-scalar approach that matches human perception:
# 1. direction_strength: How aligned are gradients? (carbon edge = high)
# 2. direction_angle: Which way does the edge run?
# 3. edge_proximity: How close is THIS patch to the edge? (patch-specific!)

def compute_global_edge_properties(
    image: np.ndarray,
    target_size: int = 512,
) -> Tuple[float, float]:
    """
    Compute global edge direction and strength for a micrograph.
    
    Carbon edges create strong, consistent directional gradients across the image.
    Clean ice has random/multidirectional gradients.
    
    Args:
        image: Full micrograph (any size, normalized to [0,1])
        target_size: Size to downsample to before analysis
        
    Returns:
        (direction_strength, direction_angle) where:
        - direction_strength: 0-1, how aligned gradients are (carbon ~0.4-0.8, ice ~0.1-0.2)
        - direction_angle: -180 to 180 degrees, dominant gradient direction
    """
    from scipy.ndimage import zoom
    
    # Downsample for efficiency
    h, w = image.shape
    scale = target_size / max(h, w)
    if scale < 1.0:
        img_ds = zoom(image, scale, order=1)
    else:
        img_ds = image.copy()
    
    # Compute Sobel gradients
    grad_x = sobel(img_ds, axis=1)  # Horizontal gradient
    grad_y = sobel(img_ds, axis=0)  # Vertical gradient
    
    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Gradient angles (in radians)
    angles = np.arctan2(grad_y, grad_x)
    
    # Weight by magnitude (strong gradients matter more)
    weights = magnitude.ravel()
    weight_sum = weights.sum() + 1e-10
    
    # Compute circular mean of gradient directions
    # Use double angle to handle opposite directions as same (0° = 180° for edges)
    double_angles = 2 * angles.ravel()
    mean_cos = np.sum(weights * np.cos(double_angles)) / weight_sum
    mean_sin = np.sum(weights * np.sin(double_angles)) / weight_sum
    
    # Direction strength: length of mean vector (0 = random, 1 = perfectly aligned)
    direction_strength = np.sqrt(mean_cos**2 + mean_sin**2)
    direction_strength = float(np.clip(direction_strength, 0, 1))
    
    # Direction angle: dominant gradient direction (divide by 2 to undo doubling)
    direction_angle = np.arctan2(mean_sin, mean_cos) / 2  # Radians
    direction_angle = float(np.degrees(direction_angle))  # Convert to degrees
    
    return direction_strength, direction_angle


def compute_edge_proximity(
    patch_center: Tuple[int, int],
    image_shape: Tuple[int, int],
    direction_angle: float,
    direction_strength: float,
    falloff_sigma: float = 0.25,
) -> float:
    """
    Compute how close a patch is to the global carbon edge.
    
    This is the KEY to spatial localization - patches near the edge get
    high proximity, patches far from the edge get low proximity.
    
    Args:
        patch_center: (y, x) center of patch in image coordinates
        image_shape: (H, W) of the full image
        direction_angle: Dominant gradient direction in degrees (-180 to 180)
        direction_strength: How strong the global edge is (0-1)
        falloff_sigma: Gaussian falloff for proximity (fraction of image size)
        
    Returns:
        edge_proximity: 0-1, how close this patch is to the edge
    """
    if direction_strength < 0.1:
        # No significant edge, proximity is meaningless
        return 0.0
    
    H, W = image_shape
    cy, cx = patch_center
    
    # Normalize coordinates to [-0.5, 0.5]
    nx = (cx / W) - 0.5
    ny = (cy / H) - 0.5
    
    # Convert angle to radians
    angle_rad = np.radians(direction_angle)
    
    # The edge runs PERPENDICULAR to the gradient direction
    # Gradient points from dark to light (across the edge)
    # Edge direction is 90° from gradient
    edge_direction = angle_rad + np.pi / 2
    
    # Project patch position onto edge-perpendicular axis
    # This gives signed distance from image center along the gradient direction
    perp_dist = nx * np.cos(angle_rad) + ny * np.sin(angle_rad)
    
    # For carbon edges, the boundary is typically near the center
    # Patches near the center (perp_dist ≈ 0) are close to the edge
    # Gaussian falloff from the center line
    proximity = np.exp(-0.5 * (perp_dist / falloff_sigma) ** 2)
    
    # Scale by direction_strength (weak edges → low proximity everywhere)
    proximity = float(proximity * direction_strength)
    
    return np.clip(proximity, 0, 1)


def compute_directional_edge_features(
    image: np.ndarray,
    patch_center: Optional[Tuple[int, int]] = None,
    target_size: int = 512,
) -> Dict[str, float]:
    """
    Compute all directional edge features for a micrograph/patch.
    
    If patch_center is None, returns only global features (for caching).
    If patch_center is provided, returns all 3 features including proximity.
    
    Args:
        image: Full micrograph
        patch_center: (y, x) center of patch, or None for global-only
        target_size: Downsample size for edge analysis
        
    Returns:
        Dict with 'direction_strength', 'direction_angle', and optionally 'edge_proximity'
    """
    direction_strength, direction_angle = compute_global_edge_properties(image, target_size)
    
    result = {
        'direction_strength': direction_strength,
        'direction_angle': direction_angle,
    }
    
    if patch_center is not None:
        edge_proximity = compute_edge_proximity(
            patch_center, image.shape, direction_angle, direction_strength
        )
        result['edge_proximity'] = edge_proximity
    
    return result


# =============================================================================
# LEGACY GLOBAL IMAGE-LEVEL FEATURES (kept for compatibility)
# =============================================================================
# These features capture micrograph-wide characteristics that help classify
# patches in uniform regions (like the middle of large carbon areas) that
# lack local distinguishing features.

def compute_global_psd_features(
    image: np.ndarray,
    target_size: int = 512,
    num_bins: int = 32,
) -> np.ndarray:
    """
    Compute global PSD features from the entire micrograph.
    
    Downsample to target_size and compute radial PSD profile.
    Large carbon edges show as excess power in low-frequency bins (0-5).
    
    Args:
        image: Full micrograph (any size)
        target_size: Size to downsample to before PSD
        num_bins: Number of radial frequency bins
        
    Returns:
        Radial PSD profile, shape (num_bins,)
    """
    eps = 1e-10
    
    # Downsample to target size
    from scipy.ndimage import zoom
    h, w = image.shape
    scale = target_size / max(h, w)
    if scale < 1.0:
        img_ds = zoom(image, scale, order=1)
    else:
        img_ds = image
    
    # Pad to square if needed
    h_ds, w_ds = img_ds.shape
    max_dim = max(h_ds, w_ds)
    if h_ds != w_ds:
        padded = np.zeros((max_dim, max_dim), dtype=img_ds.dtype)
        padded[:h_ds, :w_ds] = img_ds
        img_ds = padded
    
    # Apply 2D Hann window to reduce edge artifacts in PSD
    h_psd, w_psd = img_ds.shape
    hann_y = np.hanning(h_psd)
    hann_x = np.hanning(w_psd)
    hann_2d = np.outer(hann_y, hann_x)
    img_windowed = img_ds * hann_2d
    
    # Compute PSD (on windowed image)
    fft = np.fft.fft2(img_windowed)
    fft_shifted = np.fft.fftshift(fft)
    psd = np.abs(fft_shifted) ** 2
    psd_log = np.log(psd + eps)
    
    # Radial profile
    h, w = psd_log.shape
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    max_dist = np.sqrt(center_y**2 + center_x**2)
    bin_width = max_dist / num_bins
    dist_bins = (distances / (bin_width + eps)).astype(np.int32)
    dist_bins = np.clip(dist_bins, 0, num_bins - 1)
    
    radial_sum = np.bincount(dist_bins.ravel(), weights=psd_log.ravel(), minlength=num_bins)
    radial_counts = np.bincount(dist_bins.ravel(), minlength=num_bins)
    profile = radial_sum / (radial_counts + eps)
    
    # Z-score normalize
    p_mean, p_std = profile.mean(), profile.std()
    if p_std > eps:
        profile = (profile - p_mean) / p_std
    
    return profile.astype(np.float32)


def compute_intensity_histogram_features(
    image: np.ndarray,
    num_bins: int = 50,
) -> np.ndarray:
    """
    Compute intensity histogram features from the entire micrograph.
    
    Carbon-contaminated micrographs have bimodal distribution (dark carbon + lighter ice).
    Clean micrographs have unimodal distribution.
    
    Args:
        image: Full micrograph (normalized to [0, 1])
        num_bins: Number of histogram bins
        
    Returns:
        Normalized histogram, shape (num_bins,)
    """
    # Flatten and clip
    pixels = image.ravel()
    pixels = np.clip(pixels, 0, 1)
    
    # Compute histogram
    hist, _ = np.histogram(pixels, bins=num_bins, range=(0, 1), density=True)
    
    # Normalize to sum to 1
    hist = hist / (hist.sum() + 1e-10)
    
    return hist.astype(np.float32)


def compute_intensity_statistics(
    image: np.ndarray,
) -> np.ndarray:
    """
    Compute intensity statistics from the entire micrograph.
    
    Carbon-contaminated images have higher variance and negative skew.
    
    Returns:
        Statistics array (7-dim): [mean, std, median, skewness, kurtosis, p10, p90]
    """
    pixels = image.ravel()
    
    stats = np.array([
        np.mean(pixels),
        np.std(pixels),
        np.median(pixels),
        skew(pixels) if len(pixels) > 10 else 0.0,
        kurtosis(pixels) if len(pixels) > 10 else 0.0,
        np.percentile(pixels, 10),
        np.percentile(pixels, 90),
    ], dtype=np.float32)
    
    return stats


def compute_spatial_variance_grid(
    image: np.ndarray,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute variance in each cell of an 8×8 grid over the micrograph.
    
    Captures spatial distribution: "left half uniform (carbon), right half textured (particles)"
    
    Args:
        image: Full micrograph
        grid_size: Number of cells in each dimension (8 → 64 features)
        
    Returns:
        Flattened grid of variances, shape (grid_size * grid_size,)
    """
    h, w = image.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    variances = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < grid_size - 1 else h
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < grid_size - 1 else w
            
            cell = image[y0:y1, x0:x1]
            variances[i, j] = np.var(cell)
    
    # Normalize
    v_mean, v_std = variances.mean(), variances.std()
    if v_std > 1e-10:
        variances = (variances - v_mean) / v_std
    
    return variances.ravel().astype(np.float32)


def compute_edge_density(
    image: np.ndarray,
    sigma: float = 1.0,
    threshold_percentile: float = 90.0,
) -> float:
    """
    Compute fraction of pixels with strong edges (edge density).
    
    Helps distinguish carbon boundaries from smooth ice.
    
    Args:
        image: Full micrograph
        sigma: Gaussian smoothing before edge detection
        threshold_percentile: Percentile for edge threshold
        
    Returns:
        Edge density (fraction of strong-edge pixels)
    """
    # Smooth
    if sigma > 0:
        smoothed = gaussian_filter(image, sigma=sigma)
    else:
        smoothed = image
    
    # Sobel edge detection
    grad_x = sobel(smoothed, axis=1)
    grad_y = sobel(smoothed, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold
    threshold = np.percentile(magnitude, threshold_percentile)
    edge_mask = magnitude > threshold
    
    # Density
    density = float(edge_mask.sum()) / edge_mask.size
    
    return np.float32(density)


def compute_global_features(
    image: np.ndarray,
    psd_bins: int = 32,
    hist_bins: int = 50,
    grid_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Compute all global features for a micrograph.
    
    Total features: 32 (PSD) + 50 (histogram) + 7 (stats) + 64 (grid) + 1 (edge) = 154
    
    Args:
        image: Full micrograph (should be normalized to [0, 1])
        psd_bins: Number of PSD frequency bins
        hist_bins: Number of histogram bins
        grid_size: Spatial variance grid size
        
    Returns:
        Dictionary with individual feature arrays and concatenated 'all' key
    """
    global_psd = compute_global_psd_features(image, num_bins=psd_bins)
    histogram = compute_intensity_histogram_features(image, num_bins=hist_bins)
    stats = compute_intensity_statistics(image)
    spatial_var = compute_spatial_variance_grid(image, grid_size=grid_size)
    edge_density = np.array([compute_edge_density(image)], dtype=np.float32)
    
    # Concatenate all
    all_features = np.concatenate([
        global_psd,      # 32-dim
        histogram,       # 50-dim
        stats,           # 7-dim
        spatial_var,     # 64-dim
        edge_density,    # 1-dim
    ])  # Total: 154-dim
    
    return {
        'global_psd': global_psd,
        'histogram': histogram,
        'stats': stats,
        'spatial_var': spatial_var,
        'edge_density': edge_density,
        'all': all_features,
    }
