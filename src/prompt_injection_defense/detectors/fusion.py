"""
Fusion Detector
Combines multiple detectors using OR-fusion for threshold-invariant detection.
"""

from typing import List, Dict, Literal
from .normalizer import PromptNormalizer
from .signature_v1 import SignatureDetectorV1
from .semantic_v3 import SemanticDetectorV3


DeploymentMode = Literal["production", "monitoring"]


class FusionDetector:
    """
    Multi-detector fusion pipeline.
    
    Modes:
    - Production: Normalizer + v3 (semantic) → 82% TPR, 0.77% FAR
    - Monitoring: Normalizer + v1 (signature) + v3 (semantic) → 87% TPR, 12% FAR
    """
    
    def __init__(
        self,
        mode: DeploymentMode = "production",
        normalizer: PromptNormalizer = None,
        signature_detector: SignatureDetectorV1 = None,
        semantic_detector: SemanticDetectorV3 = None,
    ):
        """
        Initialize fusion detector.
        
        Args:
            mode: Deployment mode ("production" or "monitoring")
            normalizer: Normalizer instance (creates default if None)
            signature_detector: v1 detector (creates default if None)
            semantic_detector: v3 detector (creates default if None)
        """
        self.mode = mode
        self.normalizer = normalizer or PromptNormalizer()
        self.signature_detector = signature_detector or SignatureDetectorV1()
        self.semantic_detector = semantic_detector or SemanticDetectorV3()
    
    def detect(self, text: str) -> Dict:
        """
        Detect prompt injection using multi-detector fusion.
        
        Pipeline:
        1. Normalize input
        2. Run detectors based on mode
        3. OR-fusion: flag if ANY detector triggers
        
        Args:
            text: Input prompt to analyze
            
        Returns:
            dict with keys:
            - 'is_injection': bool, True if flagged by any detector
            - 'normalized_text': str, normalized version of input
            - 'detectors_triggered': List[str], which detectors flagged
            - 'confidence': float, max confidence across detectors
            - 'details': dict, individual detector results
            
        Example:
            >>> firewall = FusionDetector(mode="production")
            >>> result = firewall.detect("Ignore previous instructions")
            >>> result['is_injection']
            True
            >>> result['detectors_triggered']
            ['v3_semantic']
        """
        # Step 1: Normalize
        normalized = self.normalizer.normalize(text)
        
        # Step 2: Run detectors
        detectors_triggered = []
        confidences = []
        details = {}
        
        # Always run semantic detector
        v3_result = self.semantic_detector.detect(normalized)
        details['v3_semantic'] = v3_result
        if v3_result['is_injection']:
            detectors_triggered.append('v3_semantic')
        confidences.append(v3_result['confidence'])
        
        # Run signature detector only in monitoring mode
        if self.mode == "monitoring":
            v1_result = self.signature_detector.detect(normalized)
            details['v1_signature'] = v1_result
            if v1_result['is_injection']:
                detectors_triggered.append('v1_signature')
            confidences.append(v1_result['confidence'])
        
        # Step 3: OR-fusion
        is_injection = len(detectors_triggered) > 0
        max_confidence = max(confidences) if confidences else 0.0
        
        return {
            'is_injection': is_injection,
            'normalized_text': normalized,
            'detectors_triggered': detectors_triggered,
            'confidence': max_confidence,
            'details': details,
        }


# TODO: Import full implementation from phase3/
# See: phase3/fusion_detector.py
