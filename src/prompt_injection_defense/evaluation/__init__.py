"""Evaluation metrics and statistical tests for detector performance."""

from .metrics import compute_tpr_far, wilson_confidence_interval
from .statistical_tests import mcnemar_test, compute_significance

__all__ = [
    "compute_tpr_far",
    "wilson_confidence_interval",
    "mcnemar_test",
    "compute_significance",
]
