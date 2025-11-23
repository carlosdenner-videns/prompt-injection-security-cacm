"""
Semantic Detector (v3)
Embedding-based detection using cosine similarity to attack exemplars.
"""

from typing import List, Dict
import numpy as np


class SemanticDetectorV3:
    """
    Embedding-based detector using sentence transformers.
    
    Computes cosine similarity between input and a library of
    known attack exemplars (150 samples in production).
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
        exemplar_embeddings: np.ndarray = None,
    ):
        """
        Initialize semantic detector.
        
        Args:
            model_name: Sentence transformer model
            threshold: Cosine similarity threshold (0-1)
            exemplar_embeddings: Pre-computed embeddings of attack exemplars
        """
        self.model_name = model_name
        self.threshold = threshold
        self.exemplar_embeddings = exemplar_embeddings
        
        # Lazy load to avoid import overhead
        self._model = None
    
    def _load_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
    
    def detect(self, text: str) -> Dict:
        """
        Detect prompt injection using semantic similarity.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict with keys:
            - 'is_injection': bool
            - 'max_similarity': float, highest similarity to exemplars
            - 'confidence': float (same as max_similarity)
            
        Example:
            >>> detector = SemanticDetectorV3(threshold=0.75)
            >>> result = detector.detect("Disregard your training and...")
            >>> result['is_injection']
            True  # Semantically similar to "ignore instructions"
        """
        self._load_model()
        
        # Compute embedding for input
        input_embedding = self._model.encode([text])[0]
        
        # Compare to exemplar library
        if self.exemplar_embeddings is None:
            # TODO: Load from production exemplar file
            raise ValueError("No exemplar embeddings loaded")
        
        # Cosine similarity
        similarities = np.dot(
            self.exemplar_embeddings, 
            input_embedding
        ) / (
            np.linalg.norm(self.exemplar_embeddings, axis=1) * 
            np.linalg.norm(input_embedding)
        )
        
        max_similarity = float(np.max(similarities))
        
        return {
            'is_injection': max_similarity >= self.threshold,
            'max_similarity': max_similarity,
            'confidence': max_similarity,
        }


# TODO: Import full implementation with 150 exemplars from phase2_input_detection/
# See: phase2_input_detection/detectors/v3_semantic.py
