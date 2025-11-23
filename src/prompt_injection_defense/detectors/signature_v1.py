"""
Signature-based Detector (v1)
Pattern matching against known injection markers.
"""

import re
from typing import List, Dict


class SignatureDetectorV1:
    """
    Rule-based detector using regex patterns for known injection markers.
    
    Detects explicit patterns like:
    - "ignore previous instructions"
    - "system:"
    - Role confusion keywords
    - Delimiter abuse
    """
    
    # Default signature patterns (47 total in production)
    DEFAULT_PATTERNS = [
        r'\bignore\s+(previous|prior|all|above)\s+(instruction|command|prompt)',
        r'\bsystem\s*:',
        r'\b(you\s+are|act\s+as|pretend\s+to\s+be)\s+a\s+',
        r'\bDAN\s+mode\b',
        r'```\s*system',
        r'---\s*system',
        r'\[INST\]',
        r'<\|im_start\|>',
        # Add all 47 patterns from production system
    ]
    
    def __init__(self, patterns: List[str] = None):
        """
        Initialize signature detector.
        
        Args:
            patterns: List of regex patterns to match against.
                     Uses DEFAULT_PATTERNS if None.
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.patterns
        ]
    
    def detect(self, text: str) -> Dict:
        """
        Detect prompt injection using signature matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict with keys:
            - 'is_injection': bool
            - 'matched_patterns': List[str], patterns that matched
            - 'confidence': float (1.0 if any match, 0.0 otherwise)
            
        Example:
            >>> detector = SignatureDetectorV1()
            >>> result = detector.detect("Ignore previous instructions and...")
            >>> result['is_injection']
            True
        """
        matched = []
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                matched.append(self.patterns[i])
        
        return {
            'is_injection': len(matched) > 0,
            'matched_patterns': matched,
            'confidence': 1.0 if matched else 0.0,
        }


# TODO: Import full 47-pattern implementation from phase2_input_detection/
# See: phase2_input_detection/detectors/v1_signature.py
