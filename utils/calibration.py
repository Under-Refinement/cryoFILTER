"""
Calibration utilities for correcting class prevalence mismatch between training and inference.
"""
import numpy as np
from typing import Optional


def estimate_true_prevalence(gt_masks: np.ndarray) -> float:
    """
    Estimate true positive (bad pixel) prevalence from ground truth masks.
    
    Args:
        gt_masks: Ground truth masks (can be 2D or flattened), where 1=bad, 0=good
    
    Returns:
        True positive prevalence (fraction of bad pixels)
    """
    gt_flat = gt_masks.flatten() if gt_masks.ndim > 1 else gt_masks
    positive_pixels = (gt_flat >= 0.5).sum()
    total_pixels = len(gt_flat)
    prevalence = positive_pixels / total_pixels if total_pixels > 0 else 0.0
    return float(prevalence)


def compute_logit_adjustment(
    logits: np.ndarray,
    train_prevalence: float,
    true_prevalence: float,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Apply logit adjustment to correct for class prevalence mismatch.
    
    Formula: logit_corrected = logit_model + log(π_true/(1-π_true)) - log(π_train/(1-π_train))
    
    This corrects probabilities when training uses balanced sampling but
    inference has different class prevalence.
    
    Args:
        logits: Raw model logits (before sigmoid)
        train_prevalence: Positive class prevalence during training (π_train)
        true_prevalence: True positive class prevalence at inference (π_true)
        eps: Small epsilon to avoid log(0)
    
    Returns:
        Adjusted logits
    """
    # Clamp prevalences to avoid log(0) or log(inf)
    train_prevalence = np.clip(train_prevalence, eps, 1.0 - eps)
    true_prevalence = np.clip(true_prevalence, eps, 1.0 - eps)
    
    # Log odds: log(p / (1-p))
    log_odds_train = np.log(train_prevalence / (1.0 - train_prevalence))
    log_odds_true = np.log(true_prevalence / (1.0 - true_prevalence))
    
    # Adjustment term
    adjustment = log_odds_true - log_odds_train
    
    # Apply adjustment
    logits_adjusted = logits + adjustment
    
    return logits_adjusted


def apply_prevalence_correction(
    probabilities: np.ndarray,
    train_prevalence: float,
    true_prevalence: float,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Apply prevalence correction to probabilities (via logit adjustment).
    
    This is a convenience function that converts probabilities to logits,
    applies adjustment, and converts back.
    
    Args:
        probabilities: Model probabilities (0-1 range)
        train_prevalence: Positive class prevalence during training (π_train)
        true_prevalence: True positive class prevalence at inference (π_true)
        eps: Small epsilon to avoid log(0) or log(inf)
    
    Returns:
        Corrected probabilities (0-1 range, clipped)
    """
    # Clamp probabilities to avoid log(0) or log(inf)
    probs_clamped = np.clip(probabilities, eps, 1.0 - eps)
    
    # Convert to logits
    logits = np.log(probs_clamped / (1.0 - probs_clamped))
    
    # Apply adjustment
    logits_adjusted = compute_logit_adjustment(
        logits, train_prevalence, true_prevalence, eps
    )
    
    # Convert back to probabilities
    probs_adjusted = 1.0 / (1.0 + np.exp(-logits_adjusted))
    probs_adjusted = np.clip(probs_adjusted, 0.0, 1.0)
    
    return probs_adjusted
