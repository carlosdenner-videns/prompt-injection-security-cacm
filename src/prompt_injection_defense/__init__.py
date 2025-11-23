"""
Prompt Injection Defense System
A multi-detector pipeline for detecting prompt injection attacks on LLMs.
"""

__version__ = "1.0.0"
__author__ = "Carlos Denner dos Santos"

from .detectors.normalizer import PromptNormalizer
from .detectors.signature_v1 import SignatureDetectorV1
from .detectors.semantic_v3 import SemanticDetectorV3
from .detectors.fusion import FusionDetector

__all__ = [
    "PromptNormalizer",
    "SignatureDetectorV1", 
    "SemanticDetectorV3",
    "FusionDetector",
]
