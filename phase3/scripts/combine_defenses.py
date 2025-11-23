"""
Phase 3: Defense Combination Logic
Implements fusion strategies for multiple defense layers.
"""

from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class FusionStrategy(Enum):
    """Defense fusion strategies."""
    OR = "or"           # Any detector flags = attack
    AND = "and"         # All detectors flag = attack
    MAJORITY = "majority"  # Majority vote
    WEIGHTED = "weighted"  # Weighted by confidence


@dataclass
class CombinedDetectionResult:
    """Result of combined defense detection."""
    is_attack: bool
    confidence: float
    matched: List[str]
    component_results: dict  # {component_name: (is_attack, confidence, matched)}
    fusion_strategy: str


class DefenseCombiner:
    """Combines multiple defense components."""
    
    def __init__(self, strategy: FusionStrategy = FusionStrategy.OR):
        self.strategy = strategy
    
    def combine(self, *detection_results, threshold: float = 0.5) -> CombinedDetectionResult:
        """
        Combine multiple detection results.
        
        Args:
            *detection_results: Detection results from individual detectors
            threshold: Confidence threshold for flagging
        
        Returns:
            CombinedDetectionResult with fused decision
        """
        if not detection_results:
            return CombinedDetectionResult(
                is_attack=False,
                confidence=0.0,
                matched=[],
                component_results={},
                fusion_strategy=self.strategy.value
            )
        
        # Extract component results
        component_results = {}
        all_matched = []
        confidences = []
        attack_votes = 0
        
        for i, result in enumerate(detection_results):
            component_name = getattr(result, 'version', f'component_{i}')
            component_results[component_name] = (
                result.is_attack,
                result.confidence,
                result.matched
            )
            
            if result.is_attack:
                attack_votes += 1
            
            confidences.append(result.confidence)
            all_matched.extend(result.matched)
        
        # Apply fusion strategy
        if self.strategy == FusionStrategy.OR:
            is_attack, confidence = self._fuse_or(detection_results, threshold)
        elif self.strategy == FusionStrategy.AND:
            is_attack, confidence = self._fuse_and(detection_results, threshold)
        elif self.strategy == FusionStrategy.MAJORITY:
            is_attack, confidence = self._fuse_majority(detection_results, len(detection_results), threshold)
        elif self.strategy == FusionStrategy.WEIGHTED:
            is_attack, confidence = self._fuse_weighted(detection_results, threshold)
        else:
            is_attack, confidence = self._fuse_or(detection_results, threshold)
        
        return CombinedDetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            matched=list(set(all_matched)),  # Deduplicate
            component_results=component_results,
            fusion_strategy=self.strategy.value
        )
    
    def _fuse_or(self, results, threshold: float) -> Tuple[bool, float]:
        """OR fusion: Any detector flags = attack (based on is_attack flag only)."""
        max_confidence = max((r.confidence for r in results), default=0.0)
        # Only use is_attack flags - confidence threshold should be applied at detector level
        is_attack = any(r.is_attack for r in results)
        return is_attack, max_confidence
    
    def _fuse_and(self, results, threshold: float) -> Tuple[bool, float]:
        """AND fusion: All detectors must flag = attack (based on is_attack flags only)."""
        min_confidence = min((r.confidence for r in results), default=0.0)
        # Only use is_attack flags - confidence threshold should be applied at detector level
        is_attack = all(r.is_attack for r in results)
        return is_attack, min_confidence
    
    def _fuse_majority(self, results, total: int, threshold: float) -> Tuple[bool, float]:
        """Majority fusion: Majority vote (based on is_attack flags only)."""
        attack_count = sum(1 for r in results if r.is_attack)
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        
        # Only use is_attack flags - confidence threshold should be applied at detector level
        is_attack = attack_count > total / 2
        return is_attack, avg_confidence
    
    def _fuse_weighted(self, results, threshold: float) -> Tuple[bool, float]:
        """Weighted fusion: Weighted by confidence."""
        total_confidence = sum(r.confidence for r in results)
        weighted_confidence = total_confidence / len(results) if results else 0.0
        
        # Weight by both confidence and attack flag
        weighted_score = sum(
            r.confidence * (1.0 if r.is_attack else 0.5)
            for r in results
        ) / len(results) if results else 0.0
        
        is_attack = weighted_score >= threshold
        return is_attack, weighted_score


class DefenseConfiguration:
    """Represents a defense configuration."""
    
    def __init__(self, config_id: str, name: str, components: List[str], 
                 fusion_strategy: FusionStrategy = FusionStrategy.OR):
        self.config_id = config_id
        self.name = name
        self.components = components
        self.fusion_strategy = fusion_strategy
        self.combiner = DefenseCombiner(fusion_strategy)
    
    def __repr__(self):
        return f"Config {self.config_id}: {self.name} ({', '.join(self.components)})"


# Predefined configurations
DEFENSE_CONFIGURATIONS = {
    'A': DefenseConfiguration(
        'A', 'Signature-only', ['v1'],
        FusionStrategy.OR
    ),
    'B': DefenseConfiguration(
        'B', 'Rules-only', ['v2'],
        FusionStrategy.OR
    ),
    'C': DefenseConfiguration(
        'C', 'Classifier-only', ['v3'],
        FusionStrategy.OR
    ),
    'D': DefenseConfiguration(
        'D', 'Signature + Rules', ['v1', 'v2'],
        FusionStrategy.OR
    ),
    'E': DefenseConfiguration(
        'E', 'Signature + Classifier', ['v1', 'v3'],
        FusionStrategy.OR
    ),
    'F': DefenseConfiguration(
        'F', 'Rules + Classifier', ['v2', 'v3'],
        FusionStrategy.OR
    ),
    'G': DefenseConfiguration(
        'G', 'All three combined', ['v1', 'v2', 'v3'],
        FusionStrategy.OR
    ),
}


def get_configuration(config_id: str) -> DefenseConfiguration:
    """Get configuration by ID."""
    if config_id not in DEFENSE_CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {config_id}")
    return DEFENSE_CONFIGURATIONS[config_id]


def list_configurations() -> List[DefenseConfiguration]:
    """List all configurations."""
    return list(DEFENSE_CONFIGURATIONS.values())
