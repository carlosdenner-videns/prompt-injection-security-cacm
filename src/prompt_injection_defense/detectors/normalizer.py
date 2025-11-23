"""
Unicode Normalization Module
Handles Unicode normalization, homoglyph mapping, and zero-width character removal.
"""

import unicodedata
import re
from typing import Dict


class PromptNormalizer:
    """
    Normalizes prompts to prevent trivial obfuscation attacks.
    
    Applies:
    - Unicode NFKC canonicalization
    - Zero-width character stripping
    - Homoglyph mapping to ASCII equivalents
    """
    
    # Common homoglyphs (Cyrillic, Greek, etc. → ASCII)
    HOMOGLYPH_MAP = {
        'а': 'a',  # Cyrillic a
        'е': 'e',  # Cyrillic e
        'о': 'o',  # Cyrillic o
        'р': 'p',  # Cyrillic p
        'с': 'c',  # Cyrillic c
        'у': 'y',  # Cyrillic y
        'х': 'x',  # Cyrillic x
        'Α': 'A',  # Greek Alpha
        'Β': 'B',  # Greek Beta
        'Ε': 'E',  # Greek Epsilon
        # Add more as needed
    }
    
    # Zero-width and invisible characters
    ZERO_WIDTH_CHARS = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space
        '\u2060',  # Word joiner
    ]
    
    def __init__(self):
        """Initialize normalizer."""
        self._zero_width_pattern = re.compile(
            '|'.join(re.escape(char) for char in self.ZERO_WIDTH_CHARS)
        )
    
    def normalize(self, text: str) -> str:
        """
        Normalize text to prevent obfuscation-based evasion.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text with:
            - Unicode NFKC canonicalization
            - Zero-width characters removed
            - Homoglyphs mapped to ASCII
            
        Example:
            >>> normalizer = PromptNormalizer()
            >>> normalizer.normalize("Ignоre prеvious")  # Contains Cyrillic chars
            "Ignore previous"
        """
        # Step 1: Unicode NFKC normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Step 2: Remove zero-width characters
        text = self._zero_width_pattern.sub('', text)
        
        # Step 3: Map homoglyphs
        for homoglyph, ascii_char in self.HOMOGLYPH_MAP.items():
            text = text.replace(homoglyph, ascii_char)
        
        return text
    
    def detect(self, text: str) -> Dict:
        """
        Detect if normalization would change the text (indicates obfuscation).
        
        Args:
            text: Input text
            
        Returns:
            dict with keys:
            - 'is_obfuscated': bool, True if normalization changes text
            - 'original': original text
            - 'normalized': normalized text
        """
        normalized = self.normalize(text)
        return {
            'is_obfuscated': text != normalized,
            'original': text,
            'normalized': normalized,
        }


# TODO: Move implementation from phase folders here
# See: phase2_input_detection/ for full implementation
