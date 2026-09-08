"""
Post-processing utilities for bad region detection (ASOCEM-inspired).
Provides particle-size based filtering and other post-processing operations.
"""
import numpy as np
from typing import Optional, Tuple
import math
from scipy.ndimage import label, binary_fill_holes, binary_dilation, binary_erosion
from skimage.morphology import disk, binary_closing, binary_opening


def compute_min_area_from_particle_size(
    pixel_size_angstrom: float,
    particle_size_angstrom: float = 100.0
) -> int:
    """
    Compute minimum area (in pixels) for filtering regions smaller than particle size (ASOCEM-inspired).
    
    Filters regions smaller than the expected particle size to reduce false positives.
    Uses circular area: min_area = π * (particle_radius / pixel_size)²
    
    Args:
        pixel_size_angstrom: Pixel size in Angstrom per pixel
        particle_size_angstrom: Particle size in Angstrom (default: 100.0 = minimum particle size)
        
    Returns:
        Minimum area in pixels (rounded to integer)
    """
    # Compute particle radius in Angstrom
    particle_radius = particle_size_angstrom / 2.0
    
    # Compute particle radius in pixels
    particle_radius_pixels = particle_radius / pixel_size_angstrom
    
    # Compute minimum area (circular area: π * r²)
    min_area = math.pi * (particle_radius_pixels ** 2)
    
    # Round to integer (minimum 1 pixel)
    min_area_pixels = max(1, int(round(min_area)))
    
    return min_area_pixels


def remove_small_components(binary_mask: np.ndarray, min_area_pixels: int = 100) -> np.ndarray:
    """
    Remove connected components smaller than min_area_pixels.
    
    Args:
        binary_mask: Binary mask (0/1 or bool)
        min_area_pixels: Minimum area in pixels for components to keep
        
    Returns:
        Binary mask with small components removed
    """
    if min_area_pixels <= 0:
        return binary_mask.astype(np.uint8)
    
    # Label connected components
    structure = np.ones((3, 3), dtype=np.int8)
    labeled_mask, num_components = label(binary_mask.astype(bool), structure=structure)
    
    # Calculate component sizes
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Create mask for valid components (size >= min_area_pixels)
    valid_components = np.zeros(num_components + 1, dtype=bool)
    valid_components[1:] = component_sizes[1:] >= min_area_pixels  # Skip background (index 0)
    
    # Create filtered mask
    filtered_mask = valid_components[labeled_mask].astype(np.uint8)
    
    return filtered_mask


def postprocess_probability_map(
    prob_map: np.ndarray,
    image: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_component_area: int = 100,
    use_morphological: bool = True,
    closing_size: int = 3,
    opening_size: int = 2,
    dilation_size: int = 0,
    use_region_based_filling: bool = False,
    interior_threshold_ratio: float = 0.7,
    use_crf: bool = False,
    crf_iterations: int = 10,
    use_carbon_region_growing: bool = False,
    carbon_confident_threshold: float = 0.6,
    carbon_grow_threshold: float = 0.35,
    carbon_max_distance: int = 50,
    carbon_use_support_constraint: bool = True,
    carbon_support_blur_sigma: float = 10.0,
    carbon_support_quantile: float = 0.35,
    carbon_support_dilate_px: int = 15,
    carbon_support_texture_hp_sigma: Optional[float] = None,
    carbon_support_texture_quantile: Optional[float] = None,
    carbon_support_texture_mode: str = "and",
    # Stage-0: image prior (optional): boost p_bad in dark/textured regions (carbon + ethane-textured carbon)
    carbon_use_image_prior: bool = False,
    carbon_prior_lp_sigma: float = 20.0,
    carbon_prior_dark_q: float = 0.35,
    carbon_prior_texture_hp_sigma: float = 3.0,
    carbon_prior_texture_q: float = 0.60,
    carbon_prior_alpha: float = 1.0,
    carbon_prior_beta: float = 1.0,
    carbon_two_stage: bool = True,
    carbon_grow_threshold_lo: float = 0.40,
    carbon_support_blur_sigma_lo: Optional[float] = None,
    carbon_support_quantile_lo: float = 0.15,
    carbon_support_dilate_px_lo: int = 0,
    carbon_support_texture_hp_sigma_lo: Optional[float] = None,
    carbon_support_texture_quantile_lo: Optional[float] = None,
    carbon_support_texture_mode_lo: str = "and",
    carbon_stage_b_edge_only: bool = False,
    carbon_stage_b_min_mask_hi_area: int = 0,
    carbon_stage_b_min_core_lo_area: int = 0,
    carbon_stage_b_gate_mode: str = "and",  # "and" (default) or "or"
    carbon_stage_b_min_mask_hi_floor: int = 0,  # extra safety floor for OR mode
    carbon_core_expand_from_dark_near_mask_hi: bool = False,
    carbon_core_expand_dilate_px: int = 30,
    carbon_core_expand_min_added_px: int = 5000,
    carbon_seed_threshold_lo: Optional[float] = None,
    carbon_stage_c_support_propagation: bool = False,
    carbon_stage_c_requires_stage_b: bool = True,
    carbon_stage_c_max_distance: int = 50,
    carbon_stage_c_keep_pbad_max: float = 0.15,
    carbon_stage_c_fill_support_components: bool = False,
    carbon_stage_c_fill_dark_components: bool = False,
    carbon_stage_c_dark_component_min_seed_px: int = 2000,
    carbon_stage_c_dark_component_seed_dilate_px: int = 25,
    carbon_stage_c_dark_component_edge_only: bool = False,
    carbon_fft_covariance_refine: bool = False,
    carbon_fft_patch_size: int = 64,
    carbon_fft_stride: int = 16,
    carbon_fft_bins: int = 32,
    carbon_fft_pos_threshold: float = 0.75,
    carbon_fft_neg_threshold: float = 0.05,
    carbon_fft_score_threshold: float = 0.0,
    carbon_fft_ridge: float = 1e-2,
    carbon_fft_lowfreq_core_expand: bool = False,
    carbon_fft_lowfreq_patch_size: int = 96,
    carbon_fft_lowfreq_stride: int = 16,
    carbon_fft_lowfreq_bins: int = 12,
    carbon_fft_lowfreq_local_median_size: int = 129,
    carbon_fft_lowfreq_delta_threshold: float = 0.15,
    fill_enclosed_good_islands: bool = False,
    enclosed_good_island_max_area: int = 20000,
    enclosed_good_island_keep_pbad_quantile: float = 0.99,
    enclosed_good_island_keep_pbad_max: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-process probability map with thresholding, morphological operations, and connected component filtering.
    
    Args:
        prob_map: Probability map (0-1 range)
        image: Optional image for CRF post-processing (not used if use_crf=False)
        threshold: Probability threshold for binary classification
        min_component_area: Minimum area in pixels for connected components
        use_morphological: If True, apply morphological closing/opening
        closing_size: Size of structuring element for morphological closing
        opening_size: Size of structuring element for morphological opening
        dilation_size: Additional dilation size to fill interior gaps (0 = disabled)
        use_region_based_filling: If True, use region-based hole filling
        interior_threshold_ratio: Ratio for interior threshold when using region-based filling
        use_crf: If True, apply CRF post-processing (requires pydensecrf)
        crf_iterations: Number of CRF inference iterations
        
    Returns:
        Tuple of (refined_prob_map, binary_mask)
        - refined_prob_map: Post-processed probability map (may be same as input if no CRF)
        - binary_mask: Binary mask after all post-processing
    """
    refined_prob = prob_map.copy()
    
    # Apply CRF if enabled (before thresholding)
    if use_crf and image is not None:
        try:
            refined_prob = apply_crf_postprocessing(
                refined_prob,
                image,
                num_iterations=crf_iterations
            )
        except ImportError:
            # CRF not available, skip
            pass

    # IMPORTANT SAFETY DESIGN:
    # - `refined_prob` drives the *base* threshold mask (global decision boundary).
    # - Stage-0 "dark+textured" prior should NOT shift that global boundary (it causes FP explosions).
    # So we apply Stage-0 only to a separate probability map used for carbon fill stages (A/B/C).
    carbon_prob = refined_prob

    # Stage-0: optional image prior logit-boost (CARBON-STAGES ONLY)
    # Encodes "carbon is darker by eye" and "ethane-contaminated carbon is textured".
    if carbon_use_image_prior and image is not None:
        try:
            from scipy.ndimage import gaussian_filter
            from scipy.special import logit, expit

            img = np.asarray(image, dtype=np.float32)
            lp = gaussian_filter(img, sigma=float(carbon_prior_lp_sigma))
            t_dark = float(np.quantile(lp, float(carbon_prior_dark_q)))

            # Darkness score in [0,1]: darker-than-threshold -> higher score
            dark = np.clip((t_dark - lp) / (np.std(lp) + 1e-6), 0.0, 3.0) / 3.0

            # Texture energy score in [0,1]: high-pass local energy above quantile
            hp_sigma = float(carbon_prior_texture_hp_sigma)
            tq = float(carbon_prior_texture_q)
            hp = img - gaussian_filter(img, sigma=hp_sigma)
            energy = gaussian_filter(hp * hp, sigma=hp_sigma)
            te = float(np.quantile(energy, tq))
            tex = np.clip((energy - te) / (np.std(energy) + 1e-6), 0.0, 3.0) / 3.0

            # Only apply prior where it's BOTH dark and textured (safety).
            dark_mask = lp <= t_dark
            tex_mask = energy >= te
            support = dark_mask & tex_mask

            p = np.clip(carbon_prob, 1e-5, 1 - 1e-5)
            z = logit(p)
            z = z + (float(carbon_prior_alpha) * dark + float(carbon_prior_beta) * tex) * support.astype(np.float32)
            carbon_prob = expit(z).astype(np.float32)
        except Exception:
            carbon_prob = refined_prob
    
    # Step 1: Base threshold (global decision boundary)
    # NOTE: Carbon stages should only ADD to this mask, never replace it.
    binary_mask = (refined_prob >= threshold).astype(bool)
    
    # Step 2: Remove small components (before growth to avoid growing from tiny noise)
    if min_component_area > 0:
        binary_mask = remove_small_components(binary_mask, min_area_pixels=min_component_area).astype(bool)
    
    # Step 3: Carbon region growing (fast hysteresis-based gap filling)
    # CRITICAL: Do this BEFORE morphological operations to avoid suppressing recall
    # Hysteresis recovers large contiguous regions where P is "pretty high but not confident"
    if use_carbon_region_growing:
        from utils.carbon_region_growing import grow_carbon_regions_hysteresis
        base_mask = binary_mask.copy()
        constraint = None
        constraint_lo = None
        if carbon_use_support_constraint and image is not None:
            # Constrain growth to carbon/support-like regions derived from the image.
            # This prevents the common failure mode where grow_threshold is low enough
            # that the entire micrograph becomes "allowed growth terrain".
            from utils.carbon_region_growing import compute_support_constraint_mask
            constraint, _t = compute_support_constraint_mask(
                image=image,
                blur_sigma=carbon_support_blur_sigma,
                support_quantile=carbon_support_quantile,
                dilate_px=carbon_support_dilate_px,
                texture_hp_sigma=carbon_support_texture_hp_sigma,
                texture_quantile=carbon_support_texture_quantile,
                texture_mode=carbon_support_texture_mode,
            )
            if carbon_two_stage:
                # Tighter "carbon core" mask used only for deep fill.
                # This lets us drop grow_threshold without spilling into ice.
                sigma_lo = carbon_support_blur_sigma if carbon_support_blur_sigma_lo is None else float(carbon_support_blur_sigma_lo)
                constraint_lo, _tlo = compute_support_constraint_mask(
                    image=image,
                    blur_sigma=sigma_lo,
                    support_quantile=carbon_support_quantile_lo,
                    dilate_px=carbon_support_dilate_px_lo,
                    texture_hp_sigma=carbon_support_texture_hp_sigma_lo,
                    texture_quantile=carbon_support_texture_quantile_lo,
                    texture_mode=carbon_support_texture_mode_lo,
                )
        # Use fast hysteresis: seeds at high threshold, grow into lower threshold
        # This is MUCH faster than distance-based growth and recovers large contiguous regions
        # Replace initial mask with hysteresis-grown version
        mask_hi = grow_carbon_regions_hysteresis(
            carbon_prob,
            seed_threshold=carbon_confident_threshold,
            grow_threshold=carbon_grow_threshold,
            fill_holes=True,
            min_component_area=0,  # Keep all components (filtering already done above)
            constraint_mask=constraint,
        ).astype(bool)

        enable_stage_b = bool(carbon_two_stage and constraint_lo is not None)
        carbon_mask = None
        if enable_stage_b:
            mask_hi_px = int(mask_hi.sum())
            core_lo_px = int(np.asarray(constraint_lo, dtype=bool).sum())
            min_hi = int(carbon_stage_b_min_mask_hi_area)
            min_core = int(carbon_stage_b_min_core_lo_area)
            min_floor = int(carbon_stage_b_min_mask_hi_floor)

            mask_ok = True if min_hi <= 0 else (mask_hi_px >= min_hi)
            core_ok = True if min_core <= 0 else (core_lo_px >= min_core)
            floor_ok = True if min_floor <= 0 else (mask_hi_px >= min_floor)

            mode = str(carbon_stage_b_gate_mode).lower().strip()
            if min_hi > 0 and min_core > 0 and mode == "or":
                # OR mode is powerful but dangerous: many images can have a huge core_lo mask from the
                # image-derived support. Require a minimum amount of model evidence (mask_hi floor)
                # before allowing the core_lo path to enable stage-B.
                enable_stage_b = bool(mask_ok or (core_ok and floor_ok))
            else:
                # default (and) behavior; also used when only one gate is active
                enable_stage_b = bool(mask_ok and core_ok)

            # Extra safety gate (no new knobs): only enable stage-B when mask_hi is a *real carbon/ice boundary*.
            #
            # We require BOTH:
            # - **Probability sidedness**: inside band has higher P(bad) than outside band (carbon vs ice)
            # - **Low-frequency (LP) contrast**: inside is darker than a surrounding ring in LP intensity
            #
            # This blocks vignette/corner blobs where P(bad) is elevated but the image doesn't actually have
            # a carbon/ice boundary at that location.
            if enable_stage_b and image is not None and int(mask_hi_px) >= 500:
                try:
                    from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

                    mh = mask_hi.astype(bool)
                    # Narrow band just inside the boundary and just outside it
                    inner = mh & (~binary_erosion(mh, structure=np.ones((7, 7), dtype=bool)))
                    outer = (binary_dilation(mh, structure=np.ones((11, 11), dtype=bool)) & (~mh))

                    if int(inner.sum()) >= 500 and int(outer.sum()) >= 1000:
                        p_in = float(np.median(carbon_prob[inner]))
                        p_out = float(np.median(carbon_prob[outer]))
                        # NOTE: We used to require a large probability sidedness margin (p_in - p_out).
                        # With improved probability maps, p_in and p_out can be close even when the boundary
                        # is real (e.g. large carbon interiors remain moderate-prob), which was disabling
                        # stage-B on true-carbon images and driving large FNs (e.g. 0002/0007).
                        #
                        # We now rely primarily on the *image* cue (LP contrast) as the safety gate.
                        img = np.asarray(image, dtype=np.float32)
                        lp = gaussian_filter(img, sigma=float(carbon_support_blur_sigma))
                        r_outer = max(10, int(carbon_core_expand_dilate_px))
                        r_inner = max(3, r_outer // 3)
                        dil_outer = binary_dilation(mh, structure=np.ones((2 * r_outer + 1, 2 * r_outer + 1), dtype=bool))
                        dil_inner = binary_dilation(mh, structure=np.ones((2 * r_inner + 1, 2 * r_inner + 1), dtype=bool))
                        ring = dil_outer & (~dil_inner)
                        if int(ring.sum()) >= 1000:
                            med_in = float(np.median(lp[mh]))
                            med_out = float(np.median(lp[ring]))
                            delta = float(med_out - med_in)
                            delta_norm = float(delta / (np.std(lp) + 1e-6))
                            if delta_norm < 0.20:
                                enable_stage_b = False
                        else:
                            enable_stage_b = False
                except Exception:
                    pass

        if enable_stage_b:
            # Deep fill stage: lower grow threshold but only inside tight carbon-core constraint.
            #
            # IMPORTANT:
            # Requiring seeds as (P > carbon_confident_threshold) inside the core often fails because
            # the *interior* of carbon can be moderate-probability (e.g. ~0.38–0.45) while only the rim
            # reaches high P. In that case stage-B contributes nothing (exactly what you observed).
            #
            # Fix: seed stage-B from stage-A *inside the core* (mask_hi ∩ core_mask), then propagate
            # through (P > grow_lo) inside the core. This fills interior without letting ice spill in.
            from scipy.ndimage import binary_propagation, binary_fill_holes

            constraint_lo_eff = constraint_lo

            # Extra guard against the "vignette eats the core" failure:
            # If the image-derived core is enormous but only a small fraction of it has high-confidence
            # mask_hi evidence, then stage-B is likely to flood (classic corner/vignette blobs).
            #
            # This uses only existing computed quantities (no new user knobs).
            try:
                core_sz = int(np.asarray(constraint_lo_eff, dtype=bool).sum())
                hi_sz = int(mask_hi_px)
                if core_sz > 0:
                    hi_frac_of_core = float(hi_sz / float(core_sz))
                    # Empirically, true carbon interiors tend to have mask_hi occupying a meaningful
                    # fraction of the core. Vignette cores are huge with comparatively sparse mask_hi.
                    if hi_frac_of_core < 0.25:
                        enable_stage_b = False
            except Exception:
                pass

            # Adaptive core expansion (key for the remaining "big carbon gaps"):
            # If the texture-constrained core is too tight, we add *dark-only* core pixels
            # that are near the model's own stage-A mask_hi. This avoids global flooding because
            # - it's conditional on stage-B being enabled for the image
            # - it's spatially local (near mask_hi)
            # - it's still constrained by the low-frequency darkness mask
            if carbon_core_expand_from_dark_near_mask_hi and image is not None and constraint_lo_eff is not None:
                try:
                    from scipy.ndimage import binary_dilation
                    from scipy.ndimage import gaussian_filter

                    # Dark-only core candidate near mask_hi, using a boundary-contrast adaptive cutoff.
                    # We threshold low-frequency intensity at the midpoint between mask_hi and a
                    # surrounding ring; if the ring contrast is weak, we skip expansion entirely.
                    sigma_lo = carbon_support_blur_sigma if carbon_support_blur_sigma_lo is None else float(carbon_support_blur_sigma_lo)
                    img = np.asarray(image, dtype=np.float32)
                    lp = gaussian_filter(img, sigma=float(sigma_lo))

                    # Prefer an adaptive threshold from mask_hi; fallback to global quantile if mask_hi is tiny.
                    mh = mask_hi.astype(bool)
                    if int(mh.sum()) >= 500:
                        r_outer = max(10, int(carbon_core_expand_dilate_px))
                        r_inner = max(3, r_outer // 3)
                        dil_outer = binary_dilation(mh, structure=np.ones((2 * r_outer + 1, 2 * r_outer + 1), dtype=bool))
                        dil_inner = binary_dilation(mh, structure=np.ones((2 * r_inner + 1, 2 * r_inner + 1), dtype=bool))
                        ring = dil_outer & (~dil_inner)
                        if int(ring.sum()) < 1000:
                            raise RuntimeError("ring_too_small")
                        med_in = float(np.median(lp[mh]))
                        med_out = float(np.median(lp[ring]))
                        delta = (med_out - med_in)
                        delta_norm = float(delta / (np.std(lp) + 1e-6))
                        if delta_norm < 0.30:
                            raise RuntimeError("weak_boundary_contrast")
                        t = 0.5 * (med_in + med_out)  # midpoint cutoff
                    else:
                        t = float(np.quantile(lp, float(carbon_support_quantile_lo)))
                    core_dark = (lp <= t)

                    r = max(0, int(carbon_core_expand_dilate_px))
                    if r > 0:
                        near = binary_dilation(mask_hi.astype(bool), structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool))
                    else:
                        near = mask_hi.astype(bool)

                    candidate = core_dark.astype(bool) & near
                    added_px = int((candidate & ~constraint_lo_eff.astype(bool)).sum())
                    if added_px >= int(carbon_core_expand_min_added_px):
                        constraint_lo_eff = (constraint_lo_eff.astype(bool) | candidate).astype(bool)
                except Exception:
                    pass
            if carbon_stage_b_edge_only:
                # Optional safety prior: only allow deep-fill inside core regions connected to the image border.
                # This helps when the missing carbon is edge-connected (common) and reduces interior spill/FPS.
                labs, n = label(constraint_lo_eff.astype(bool), structure=np.ones((3, 3), dtype=np.int8))
                if n > 0:
                    border = np.zeros_like(labs, dtype=bool)
                    border[0, :] = True
                    border[-1, :] = True
                    border[:, 0] = True
                    border[:, -1] = True
                    border_labels = np.unique(labs[border])
                    keep = np.zeros(n + 1, dtype=bool)
                    keep[border_labels] = True
                    keep[0] = False
                    constraint_lo_eff = keep[labs]

            # FFT low-frequency bandpower-based core expansion (material cue; accuracy-first):
            # Expand the stage-B core (constraint_lo_eff) to include pixels that look support-like in Fourier space.
            # Critically, we only allow expansion near the model's own mask_hi (spatially local) to avoid flooding.
            if carbon_fft_lowfreq_core_expand and image is not None:
                try:
                    from utils.fft_bandpower import LowFreqBandpowerConfig, lowfreq_bandpower_map
                    from scipy.ndimage import binary_dilation

                    cfg = LowFreqBandpowerConfig(
                        patch_size=int(carbon_fft_lowfreq_patch_size),
                        stride=int(carbon_fft_lowfreq_stride),
                        bins=int(carbon_fft_lowfreq_bins),
                        local_median_size=int(carbon_fft_lowfreq_local_median_size),
                    )
                    _bp, delta = lowfreq_bandpower_map(np.asarray(image, dtype=np.float32), cfg)
                    support_fft = (delta >= float(carbon_fft_lowfreq_delta_threshold))

                    # Localize expansion near current mask_hi evidence
                    rloc = max(0, int(carbon_core_expand_dilate_px))
                    if rloc > 0:
                        near = binary_dilation(mask_hi.astype(bool), structure=np.ones((2 * rloc + 1, 2 * rloc + 1), dtype=bool))
                    else:
                        near = mask_hi.astype(bool)

                    candidate = support_fft & near
                    constraint_lo_eff = (np.asarray(constraint_lo_eff, dtype=bool) | candidate).astype(bool)
                except Exception:
                    pass

            # FFT/covariance refinement (accuracy-first):
            # Expand bad mask inside the stage-B core only where Fourier-space features match the
            # model's own high-confidence "bad" seeds, compared against confident-good negatives.
            fft_mask = None
            if carbon_fft_covariance_refine and image is not None:
                try:
                    from utils.fft_covariance_refine import FFTCovarianceConfig, fft_covariance_llr_map, llr_to_mask

                    cfg = FFTCovarianceConfig(
                        patch_size=int(carbon_fft_patch_size),
                        stride=int(carbon_fft_stride),
                        bins=int(carbon_fft_bins),
                        pos_threshold=float(carbon_fft_pos_threshold),
                        neg_threshold=float(carbon_fft_neg_threshold),
                        ridge=float(carbon_fft_ridge),
                        score_threshold=float(carbon_fft_score_threshold),
                    )
                    cand_fft = np.asarray(constraint_lo_eff, dtype=bool)
                    score_map, _info = fft_covariance_llr_map(
                        image=np.asarray(image, dtype=np.float32),
                        prob_map=carbon_prob,
                        candidate_mask=cand_fft,
                        cfg=cfg,
                    )
                    fft_mask = llr_to_mask(score_map, cand_fft, threshold=cfg.score_threshold).astype(bool)
                except Exception:
                    pass

            # By default, seed stage-B from stage-A inside the core (best for avoiding flooding).
            # Optionally, allow additional seeds from moderate probability inside the core to
            # pick up disconnected "islands" the stage-A mask doesn't touch.
            # CRITICAL SAFETY + "fill only on carbon side":
            # Use only boundary-adjacent, high-confidence evidence as seeds, and only grow into pixels
            # that are darker than that boundary in low-frequency intensity.
            #
            # This addresses the observed failure:
            # - Corner/vignette gradients can produce large core masks and elevated P(bad), enabling stage-B.
            # - But they usually do NOT have a sharp carbon/ice boundary. So we require a strong LP gradient
            #   at the boundary before seeding any deep fill.
            seeds_lo = mask_hi & constraint_lo_eff
            lp_fill_mask = None
            if image is not None:
                try:
                    from scipy.ndimage import gaussian_filter

                    # Low-frequency intensity for boundary reasoning (use Stage-A scale; sigma_lo=200 smears the edge)
                    sigma_lo = float(carbon_support_blur_sigma)
                    img = np.asarray(image, dtype=np.float32)
                    lp = gaussian_filter(img, sigma=float(sigma_lo))

                    # Boundary of stage-A mask (1px-ish)
                    mh = mask_hi.astype(bool)
                    boundary = mh & (~binary_erosion(mh, structure=np.ones((3, 3), dtype=bool)))

                    # LP gradient magnitude: strong at real carbon/ice edges, weak in vignette blobs
                    gy, gx = np.gradient(lp)
                    gmag = np.sqrt(gx * gx + gy * gy)

                    # Strong-edge threshold: use a quantile *within the boundary* so we always get
                    # a usable set of seeds on real edges, even if the whole image is noisy/textured.
                    # This avoids the common failure mode where a global 90th percentile is too strict
                    # and produces almost no seeds (=> no fill on 0007/0002).
                    if boundary.any():
                        g_thr = float(np.quantile(gmag[boundary], 0.70))
                    else:
                        g_thr = float(np.quantile(gmag, 0.90))
                    boundary_strong = boundary & (gmag >= g_thr)

                    # Replace seeds with strong boundary only
                    seeds_lo = boundary_strong & constraint_lo_eff

                    # If we have seeds, do a "carbon-side fill" using low-frequency intensity:
                    # - define a carbon-side basin using a cutoff between inside (carbon) and a surrounding ring (ice)
                    # - flood fill through that basin (ignoring interior P(bad)), but still inside the stage-B core
                    if seeds_lo.any():
                        from scipy.ndimage import binary_dilation

                        r_outer = max(10, int(carbon_core_expand_dilate_px))
                        r_inner = max(3, r_outer // 3)
                        dil_outer = binary_dilation(mh, structure=np.ones((2 * r_outer + 1, 2 * r_outer + 1), dtype=bool))
                        dil_inner = binary_dilation(mh, structure=np.ones((2 * r_inner + 1, 2 * r_inner + 1), dtype=bool))
                        ring = dil_outer & (~dil_inner)

                        if int(ring.sum()) >= 1000:
                            med_in = float(np.median(lp[mh]))
                            med_out = float(np.median(lp[ring]))
                            delta = (med_out - med_in)
                            delta_norm = float(delta / (np.std(lp) + 1e-6))

                            if delta_norm >= 0.30:
                                # Carbon-side cutoff: permissive but still blocks bright ice.
                                t_lp = float(med_in + 0.90 * delta)
                                basin = (lp <= t_lp)
                                allowed_lp = basin & constraint_lo_eff.astype(bool)
                                lp_fill_mask = binary_propagation(
                                    seeds_lo.astype(bool),
                                    mask=allowed_lp,
                                    structure=np.ones((3, 3)),
                                )
                                lp_fill_mask = binary_fill_holes(lp_fill_mask)

                except Exception:
                    pass

            # If stage-B yielded no seeds, it can't contribute meaningful fill, and Stage-C should not
            # be allowed to "finish" anything purely due to enable_stage_b being true.
            stage_b_has_seeds = bool(np.asarray(seeds_lo, dtype=bool).any())
            if not stage_b_has_seeds:
                enable_stage_b = False

            if carbon_seed_threshold_lo is not None:
                seeds_lo = seeds_lo | ((carbon_prob >= float(carbon_seed_threshold_lo)) & constraint_lo_eff)
            growth_lo = (carbon_prob >= carbon_grow_threshold_lo) & constraint_lo_eff
            mask_lo = binary_propagation(seeds_lo, mask=growth_lo, structure=np.ones((3, 3)))
            mask_lo = binary_fill_holes(mask_lo)

            # IMPORTANT: union FFT refinement into the final mask (otherwise it gets overwritten).
            if fft_mask is not None:
                carbon_mask = (mask_hi | mask_lo | fft_mask)
            else:
                carbon_mask = (mask_hi | mask_lo)

            # Union carbon-side LP-fill (image-physics-based interior fill).
            if lp_fill_mask is not None:
                carbon_mask = (carbon_mask.astype(bool) | lp_fill_mask.astype(bool)).astype(bool)
        else:
            carbon_mask = mask_hi

        # Carbon stages are additive by design.
        binary_mask = (base_mask.astype(bool) | np.asarray(carbon_mask, dtype=bool)).astype(bool)

        # Stage C: image-prior finishing inside support/core.
        #
        # NOTE: Stage-C has multiple sub-modes (propagation, component fill, dark-component fill).
        # Component-fill used to be incorrectly nested under "support propagation enabled",
        # making it a no-op when propagation was disabled.
        stage_c_enabled = bool(
            carbon_stage_c_support_propagation
            or carbon_stage_c_fill_support_components
            or carbon_stage_c_fill_dark_components
        )
        if stage_c_enabled and image is not None and (not carbon_stage_c_requires_stage_b or enable_stage_b):
            try:
                from scipy.ndimage import binary_propagation, binary_fill_holes, distance_transform_edt

                support_c = None
                if constraint_lo is not None:
                    support_c = constraint_lo_eff.astype(bool) if "constraint_lo_eff" in locals() else constraint_lo.astype(bool)
                elif constraint is not None:
                    support_c = constraint.astype(bool)

                if support_c is not None:
                    seeds_c = binary_mask.astype(bool)
                    allowed = support_c.copy()

                    # Optional: keep very confident-good pixels as good (do not allow propagation)
                    keep_max = float(carbon_stage_c_keep_pbad_max)
                    good_veto = None
                    if keep_max > 0:
                        # Use the *base* probabilities for this veto so the Stage-0 prior can't
                        # override "very confident good" pixels.
                        good_veto = (refined_prob < keep_max)

                    # Stage-C1: support-only propagation (strong prior)
                    if carbon_stage_c_support_propagation:
                        allowed_prop = allowed.copy()

                        # Optional: limit expansion radius from current mask
                        max_d = int(carbon_stage_c_max_distance)
                        if max_d > 0 and seeds_c.any():
                            dist = distance_transform_edt(~seeds_c)  # distance to nearest seed pixel
                            allowed_prop &= (dist <= max_d)

                        if good_veto is not None:
                            allowed_prop &= (~good_veto)

                        grown_c = binary_propagation(seeds_c, mask=allowed_prop, structure=np.ones((3, 3)))
                        grown_c = binary_fill_holes(grown_c)
                        binary_mask = (binary_mask.astype(bool) | grown_c).astype(bool)

                    # Stage-C2: component-fill inside support mask (key for disconnected carbon islands)
                    #
                    # Problem this solves:
                    # If the missing carbon pixels are NOT connected (in the support mask topology) to any
                    # existing seeds, binary_propagation can never reach them.
                    #
                    # Fix:
                    # Label connected components of the support mask and, for any component that contains
                    # at least one seed pixel, fill the entire component (subject to optional "confident-good veto").
                    if carbon_stage_c_fill_support_components:
                        try:
                            labs, n = label(support_c.astype(bool), structure=np.ones((3, 3), dtype=np.int8))
                            if n > 0:
                                # CRITICAL SAFETY:
                                # Don't let a tiny amount of evidence fill an enormous support component.
                                # Use Stage-B seeds (boundary-derived) when available; otherwise fall back to mask_hi.
                                seed_src = None
                                if "seeds_lo" in locals():
                                    seed_src = np.asarray(seeds_lo, dtype=bool)
                                if seed_src is None or not seed_src.any():
                                    seed_src = np.asarray(mask_hi, dtype=bool)

                                if seed_src.any():
                                    # Reuse the existing "min seed px" knob to require meaningful evidence.
                                    # This prevents 0131/0014-style floods where support_c is huge.
                                    min_seed = int(carbon_stage_c_dark_component_min_seed_px)
                                    seed_counts = np.bincount(labs[seed_src].ravel(), minlength=n + 1)
                                    comp_sizes = np.bincount(labs.ravel(), minlength=n + 1)

                                    # Also require evidence to scale with component size:
                                    # huge components need a *lot* of seeds, otherwise we risk flooding a vignette/core blob.
                                    # This is a fixed safety constant (no new knobs): require at least 1% of the component
                                    # to be seeded, in addition to min_seed.
                                    min_frac = 0.01
                                    min_seed_scaled = np.maximum(min_seed, (min_frac * comp_sizes).astype(np.int64))
                                    keep = seed_counts >= min_seed_scaled
                                    keep[0] = False
                                    if np.any(keep):
                                        fill = keep[labs]
                                        if good_veto is not None:
                                            fill &= (~good_veto)
                                        binary_mask = (binary_mask.astype(bool) | fill).astype(bool)
                        except Exception:
                            pass

                    # Stage-C3: fill DARK support components that contain enough model evidence (mask_hi)
                    #
                    # This is the ASOCEM/MC-style "image prior + small user assumption" move:
                    # - Use the image physics (darkness) to define a carbon basin.
                    # - Use the model only to decide which basin(s) are actually bad.
                    #
                    # It fixes the specific remaining failure mode:
                    # disconnected interior carbon islands that are dark-by-eye but have very low P(bad),
                    # and therefore are unreachable by propagation.
                    if carbon_stage_c_fill_dark_components and image is not None:
                        try:
                            from utils.carbon_region_growing import compute_support_constraint_mask
                            from scipy.ndimage import binary_dilation

                            sigma_lo = carbon_support_blur_sigma if carbon_support_blur_sigma_lo is None else float(carbon_support_blur_sigma_lo)
                            core_dark, _ = compute_support_constraint_mask(
                                image=image,
                                blur_sigma=sigma_lo,
                                support_quantile=carbon_support_quantile_lo,
                                dilate_px=carbon_support_dilate_px_lo,
                                texture_hp_sigma=None,
                                texture_quantile=None,
                                texture_mode="and",
                            )
                            labs, n = label(core_dark.astype(bool), structure=np.ones((3, 3), dtype=np.int8))
                            if n > 0:
                                # Only accept components that contain enough stage-A evidence.
                                # This prevents a single stray seed from turning an entire dark open-hole into "bad".
                                min_seed = int(carbon_stage_c_dark_component_min_seed_px)
                                # Allow a bit of spatial tolerance: stage-A evidence is often on carbon rims.
                                r = max(0, int(carbon_stage_c_dark_component_seed_dilate_px))
                                if r > 0:
                                    mh = binary_dilation(mask_hi.astype(bool), structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool))
                                else:
                                    mh = mask_hi.astype(bool)
                                seed_counts = np.bincount(labs[mh].ravel(), minlength=n + 1)
                                keep = seed_counts >= min_seed
                                keep[0] = False

                                if carbon_stage_c_dark_component_edge_only:
                                    # Keep only dark components that touch the image border (common for carbon support).
                                    border = np.zeros_like(labs, dtype=bool)
                                    border[0, :] = True
                                    border[-1, :] = True
                                    border[:, 0] = True
                                    border[:, -1] = True
                                    border_labels = np.unique(labs[border])
                                    edge_keep = np.zeros(n + 1, dtype=bool)
                                    edge_keep[border_labels] = True
                                    edge_keep[0] = False
                                    keep = keep & edge_keep

                                fill = keep[labs]

                                keep_max = float(carbon_stage_c_keep_pbad_max)
                                if keep_max > 0:
                                    fill &= (refined_prob >= keep_max)

                                binary_mask = (binary_mask.astype(bool) | fill).astype(bool)
                        except Exception:
                            pass
            except Exception:
                pass
    elif use_region_based_filling:
        from utils.region_based_thresholding import fill_interior_holes_in_regions
        binary_mask = fill_interior_holes_in_regions(
            binary_mask, refined_prob, interior_threshold_ratio=interior_threshold_ratio
        ).astype(bool)
    elif dilation_size > 0:
        # Simple dilation to fill interior gaps
        se_dilation = disk(dilation_size)
        binary_mask = binary_dilation(binary_mask, structure=se_dilation)
    
    # Step 4: Morphological closing (connect nearby regions, fill small gaps)
    # Do this AFTER hysteresis to connect regions that hysteresis found
    if use_morphological and closing_size > 0:
        se_closing = disk(closing_size)
        binary_mask = binary_closing(binary_mask, se_closing)
    
    # Step 5: Morphological opening (remove small protrusions)
    # WARNING: This can suppress recall by removing marginal positives
    # Only do this if you're okay with potentially losing some true positives
    if use_morphological and opening_size > 0:
        se_opening = disk(opening_size)
        binary_mask = binary_opening(binary_mask, se_opening)

    # Step 6: "Good islands inside bad" prior (hole filling with confidence exception)
    #
    # If a region of predicted GOOD pixels is fully enclosed by predicted BAD pixels,
    # it is usually an under-filled carbon/support interior. We fill these holes unless
    # the model is *very confident* they are good (p_bad extremely low).
    if fill_enclosed_good_islands:
        try:
            mask_bool = binary_mask.astype(bool)
            holes = ~mask_bool
            labs, n = label(holes, structure=np.ones((3, 3), dtype=np.int8))
            if n > 0:
                # components touching border are NOT holes
                border = np.zeros_like(labs, dtype=bool)
                border[0, :] = True
                border[-1, :] = True
                border[:, 0] = True
                border[:, -1] = True
                border_labels = np.unique(labs[border])
                border_set = set(int(x) for x in border_labels.tolist())
                q = float(enclosed_good_island_keep_pbad_quantile)
                keep_max = float(enclosed_good_island_keep_pbad_max)
                max_area = int(enclosed_good_island_max_area)

                for cid in range(1, n + 1):
                    if cid in border_set:
                        continue
                    comp = (labs == cid)
                    area = int(comp.sum())
                    if max_area > 0 and area > max_area:
                        continue
                    # Keep as GOOD only if even the high-quantile p_bad is extremely low
                    pv = refined_prob[comp]
                    if pv.size == 0:
                        continue
                    if float(np.quantile(pv, q)) < keep_max:
                        continue
                    # Otherwise, fill: mark these enclosed "good" pixels as bad
                    mask_bool[comp] = True
                binary_mask = mask_bool.astype(bool)
        except Exception:
            # Never fail postprocessing because of this optional heuristic
            pass
    
    return refined_prob, binary_mask.astype(np.uint8)


def apply_crf_postprocessing(
    prob_map: np.ndarray,
    image: np.ndarray,
    num_iterations: int = 10
) -> np.ndarray:
    """
    Apply CRF (Conditional Random Field) post-processing to probability map.
    
    Requires pydensecrf package. This is optional and may have compilation issues.
    
    Args:
        prob_map: Probability map (0-1 range)
        image: Input image (normalized, 0-1 range)
        num_iterations: Number of CRF inference iterations
        
    Returns:
        Refined probability map after CRF
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        raise ImportError("pydensecrf not available. Install with: pip install pydensecrf")
    
    # Convert probability map to unary potential
    # CRF expects log-probabilities in shape (num_classes, height, width)
    num_classes = 2
    h, w = prob_map.shape
    
    # Create unary potential: [good_prob, bad_prob]
    unary = np.zeros((num_classes, h, w), dtype=np.float32)
    unary[0] = 1.0 - prob_map  # Good class probability
    unary[1] = prob_map  # Bad class probability
    
    # Convert to log-probabilities (negative log-likelihood)
    unary = -np.log(unary + 1e-8)
    
    # Create CRF
    crf = dcrf.DenseCRF2D(w, h, num_classes)
    
    # Set unary potential
    crf.setUnaryEnergy(unary.reshape(num_classes, -1))
    
    # Add pairwise potentials (smoothness and appearance)
    # Smoothness term (encourages similar labels for nearby pixels)
    crf.addPairwiseGaussian(sxy=3, compat=3)
    
    # Appearance term (encourages similar labels for pixels with similar appearance)
    # Convert image to uint8 for CRF
    image_uint8 = (image * 255).astype(np.uint8)
    if len(image_uint8.shape) == 2:
        # Grayscale: add channel dimension
        image_uint8 = image_uint8[..., np.newaxis]
    crf.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_uint8, compat=10)
    
    # Run inference
    Q = crf.inference(num_iterations)
    
    # Get refined probabilities for bad class
    refined_prob = np.array(Q[1]).reshape(h, w)
    
    return refined_prob
