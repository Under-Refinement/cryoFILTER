"""
Global prior learning from training distribution (MC-inspired).
Computes and stores global statistics from training data for use during inference.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def compute_training_statistics(
    contamination_fractions: List[float],
    intensity_stats: Optional[List[Dict]] = None,
    bad_region_sizes: Optional[List[float]] = None
) -> Dict:
    """
    Compute global statistics from training distribution.
    
    Args:
        contamination_fractions: List of contamination fractions (bad pixel fraction) per micrograph
        intensity_stats: Optional list of intensity statistics per micrograph (mean, std, etc.)
        bad_region_sizes: Optional list of bad region sizes in pixels
        
    Returns:
        Dictionary containing global statistics:
        - contamination_prevalence: Distribution stats (mean, std, percentiles)
        - intensity_statistics: Global intensity stats (if provided)
        - bad_region_size_statistics: Bad region size stats (if provided)
    """
    stats = {}
    
    # Contamination prevalence distribution
    if contamination_fractions:
        contamination_fractions = np.array(contamination_fractions)
        contamination_fractions = contamination_fractions[~np.isnan(contamination_fractions)]
        
        if len(contamination_fractions) > 0:
            stats['contamination_prevalence'] = {
                'mean': float(np.mean(contamination_fractions)),
                'std': float(np.std(contamination_fractions)),
                'median': float(np.median(contamination_fractions)),
                'p25': float(np.percentile(contamination_fractions, 25.0)),
                'p75': float(np.percentile(contamination_fractions, 75.0)),
                'p10': float(np.percentile(contamination_fractions, 10.0)),
                'p90': float(np.percentile(contamination_fractions, 90.0)),
                'min': float(np.min(contamination_fractions)),
                'max': float(np.max(contamination_fractions)),
                'count': int(len(contamination_fractions))
            }
    
    # Intensity statistics (if provided)
    if intensity_stats:
        # Aggregate intensity statistics
        intensity_means = [s.get('mean', 0.0) for s in intensity_stats if 'mean' in s]
        intensity_stds = [s.get('std', 0.0) for s in intensity_stats if 'std' in s]
        
        if intensity_means:
            stats['intensity_statistics'] = {
                'mean': float(np.mean(intensity_means)),
                'std_mean': float(np.std(intensity_means)),
                'mean_std': float(np.mean(intensity_stds)) if intensity_stds else 0.0,
                'count': int(len(intensity_means))
            }
    
    # Bad region size statistics (if provided)
    if bad_region_sizes:
        bad_region_sizes = np.array(bad_region_sizes)
        bad_region_sizes = bad_region_sizes[bad_region_sizes > 0]
        
        if len(bad_region_sizes) > 0:
            stats['bad_region_size_statistics'] = {
                'mean': float(np.mean(bad_region_sizes)),
                'std': float(np.std(bad_region_sizes)),
                'median': float(np.median(bad_region_sizes)),
                'p25': float(np.percentile(bad_region_sizes, 25.0)),
                'p75': float(np.percentile(bad_region_sizes, 75.0)),
                'min': float(np.min(bad_region_sizes)),
                'max': float(np.max(bad_region_sizes)),
                'count': int(len(bad_region_sizes))
            }
    
    return stats


def save_global_prior(stats: Dict, output_path: Path):
    """
    Save global prior statistics to JSON file.
    
    Args:
        stats: Global statistics dictionary
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)


def load_global_prior(input_path: Path) -> Dict:
    """
    Load global prior statistics from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Global statistics dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Global prior file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        stats = json.load(f)
    
    return stats


def get_expected_prevalence(stats: Dict, method: str = 'median') -> float:
    """
    Get expected contamination prevalence from global statistics.
    
    Args:
        stats: Global statistics dictionary
        method: Method to use ('mean', 'median', 'p25', 'p75', 'p10', 'p90')
        
    Returns:
        Expected prevalence (float)
    """
    if 'contamination_prevalence' not in stats:
        return 0.05  # Default
    
    prev_stats = stats['contamination_prevalence']
    
    if method == 'mean':
        return prev_stats.get('mean', 0.05)
    elif method == 'median':
        return prev_stats.get('median', 0.05)
    elif method == 'p25':
        return prev_stats.get('p25', 0.05)
    elif method == 'p75':
        return prev_stats.get('p75', 0.05)
    elif method == 'p10':
        return prev_stats.get('p10', 0.05)
    elif method == 'p90':
        return prev_stats.get('p90', 0.05)
    else:
        return prev_stats.get('mean', 0.05)  # Default to mean
