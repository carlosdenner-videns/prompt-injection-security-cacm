"""Statistical significance tests for comparing detectors."""

import numpy as np
from scipy import stats
from typing import Dict


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> Dict:
    """
    McNemar's test for comparing two classifiers.
    
    Tests whether two detectors have significantly different error rates.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from detector A
        y_pred_b: Predictions from detector B
        
    Returns:
        dict with keys:
        - 'statistic': float, McNemar chi-square statistic
        - 'p_value': float, two-tailed p-value
        - 'significant': bool, True if p < 0.05
        
    Reference:
        McNemar, Q. (1947). "Note on the sampling error of the difference
        between correlated proportions or percentages".
        Psychometrika, 12(2), 153-157.
    """
    # Contingency table
    # A correct, B wrong | A wrong, B correct
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    
    n01 = np.sum(a_correct & ~b_correct)  # A right, B wrong
    n10 = np.sum(~a_correct & b_correct)  # A wrong, B right
    
    # McNemar statistic with continuity correction
    if n01 + n10 == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
        }
    
    statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }


def compute_significance(
    y_true: np.ndarray,
    detectors: Dict[str, np.ndarray]
) -> Dict:
    """
    Compute pairwise McNemar tests for multiple detectors.
    
    Args:
        y_true: Ground truth labels
        detectors: Dict mapping detector name to predictions
        
    Returns:
        dict mapping (detector_a, detector_b) to test results
        
    Example:
        >>> results = compute_significance(
        ...     y_true,
        ...     {'v1': v1_preds, 'v3': v3_preds, 'fusion': fusion_preds}
        ... )
        >>> results[('v1', 'v3')]['significant']
        True
    """
    detector_names = list(detectors.keys())
    results = {}
    
    for i, name_a in enumerate(detector_names):
        for name_b in detector_names[i+1:]:
            test_result = mcnemar_test(
                y_true,
                detectors[name_a],
                detectors[name_b]
            )
            results[(name_a, name_b)] = test_result
    
    return results


# TODO: Import full statistical analysis from phase1/scripts/
