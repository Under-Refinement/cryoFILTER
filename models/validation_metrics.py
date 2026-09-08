"""
Validation metrics for checkpoint selection.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve


def compute_sens_at_spec(
    probs: np.ndarray,
    targets: np.ndarray,
    target_spec: float = 0.99,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, Optional[float]]:
    """
    Compute sensitivity at a fixed specificity target.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_sens = 0.0
    best_spec = 0.0
    best_thresh = None

    for thresh in thresholds:
        preds = (probs >= thresh).astype(float)
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()

        if (tn + fp) == 0:
            continue

        spec = tn / (tn + fp)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if spec >= target_spec and sens > best_sens:
            best_sens = sens
            best_spec = spec
            best_thresh = thresh

    if best_thresh is None:
        max_spec = 0.0
        for thresh in thresholds:
            preds = (probs >= thresh).astype(float)
            fp = ((preds == 1) & (targets == 0)).sum()
            tn = ((preds == 0) & (targets == 0)).sum()
            if (tn + fp) > 0:
                max_spec = max(max_spec, tn / (tn + fp))
        return 0.0, max_spec, None

    return best_sens, best_spec, best_thresh


def compute_pr_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute precision-recall AUC for binary targets.
    """
    try:
        precision, recall, _ = precision_recall_curve(targets, probs)
        return float(auc(recall, precision))
    except Exception:
        return 0.0


def compute_validation_metrics_for_checkpoint(
    model_outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    target_spec: float = 0.99,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute sensitivity-at-specificity and PR-AUC from model outputs.
    """
    del device

    if model_outputs.dim() == 4 and model_outputs.shape[1] == num_classes:
        probs = torch.softmax(model_outputs, dim=1)
        if num_classes == 2:
            p_bad = probs[:, 1].detach().cpu().numpy()
        else:
            p_bad = probs[:, 1:].sum(dim=1).detach().cpu().numpy()
    else:
        p_bad = model_outputs.detach().cpu().numpy()
        if p_bad.ndim == 4:
            p_bad = p_bad[:, 0]

    if targets.dim() == 4 and targets.shape[1] == num_classes:
        if num_classes == 2:
            targets_binary = targets[:, 1].detach().cpu().numpy()
        else:
            targets_binary = targets[:, 1:].sum(dim=1).detach().cpu().numpy()
    else:
        targets_binary = (targets >= 1).float().detach().cpu().numpy()

    probs_flat = p_bad.flatten()
    targets_flat = targets_binary.flatten()
    sens, achieved_spec, thresh = compute_sens_at_spec(
        probs_flat,
        targets_flat,
        target_spec=target_spec,
    )
    pr_auc = compute_pr_auc(probs_flat, targets_flat)

    return {
        "sens_at_spec_99": float(sens),
        "pr_auc": float(pr_auc),
        "achieved_spec": float(achieved_spec),
        "threshold": float(thresh) if thresh is not None else None,
    }
