"""Performance metrics for detector evaluation."""

import numpy as np
from typing import Tuple, Dict


def compute_tpr_far(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Compute True Positive Rate and False Alarm Rate.
    
    Args:
        y_true: Ground truth labels (1=attack, 0=benign)
        y_pred: Predicted labels (1=attack, 0=benign)
        
    Returns:
        (tpr, far) as floats in [0, 1]
        
    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 0, 0, 1])
        >>> tpr, far = compute_tpr_far(y_true, y_pred)
        >>> tpr  # 1/2 attacks detected
        0.5
        >>> far  # 1/2 benign flagged
        0.5
    """
    # Split into attack and benign samples
    attack_mask = y_true == 1
    benign_mask = y_true == 0
    
    # True Positive Rate (recall on attacks)
    if np.sum(attack_mask) > 0:
        tpr = np.sum(y_pred[attack_mask] == 1) / np.sum(attack_mask)
    else:
        tpr = 0.0
    
    # False Alarm Rate (FPR on benign)
    if np.sum(benign_mask) > 0:
        far = np.sum(y_pred[benign_mask] == 1) / np.sum(benign_mask)
    else:
        far = 0.0
    
    return tpr, far


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for proportion.
    
    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default: 0.95)
        
    Returns:
        (lower_bound, upper_bound) as floats in [0, 1]
        
    Reference:
        Wilson, E.B. (1927). "Probable Inference, the Law of Succession,
        and Statistical Inference". Journal of the American Statistical
        Association.
    """
    from scipy import stats
    
    if total == 0:
        return 0.0, 0.0
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    
    return lower, upper


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: float = 0.95
) -> Dict:
    """
    Compute TPR, FAR with Wilson confidence intervals.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidence: Confidence level (default: 0.95)
        
    Returns:
        dict with keys:
        - 'tpr': float
        - 'tpr_ci': (lower, upper)
        - 'far': float
        - 'far_ci': (lower, upper)
    """
    tpr, far = compute_tpr_far(y_true, y_pred)
    
    # Compute CIs
    attack_mask = y_true == 1
    benign_mask = y_true == 0
    
    tp = np.sum(y_pred[attack_mask] == 1)
    n_attacks = np.sum(attack_mask)
    tpr_ci = wilson_confidence_interval(tp, n_attacks, confidence)
    
    fp = np.sum(y_pred[benign_mask] == 1)
    n_benign = np.sum(benign_mask)
    far_ci = wilson_confidence_interval(fp, n_benign, confidence)
    
    return {
        'tpr': tpr,
        'tpr_ci': tpr_ci,
        'far': far,
        'far_ci': far_ci,
    }


# TODO: Import full implementation from phase evaluation scripts
