"""
Phase 6c: Evaluate Adversarial Attacks
Tests robustness against attacks designed to evade the system.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2_input_detection" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase5" / "scripts"))

from normalizer import normalize_text
from features import extract_features, get_feature_names, DetectionResult
from input_detectors import get_input_detector

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AdversarialAttackEvaluator:
    """Evaluates robustness against adversarial attacks."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase6c_dir = self.script_dir.parent
        self.phase5_dir = self.phase6c_dir.parent / "phase5"
        self.results_dir = self.phase6c_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.detectors = {}
    
    def load_adversarial_attacks(self):
        """Load adversarial attack dataset."""
        data_file = self.phase6c_dir / "data" / "adversarial_attacks.json"
        
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"✓ Loaded {len(self.data)} adversarial attack samples")
    
    def load_detectors(self):
        """Load v1, v2, v3 detectors."""
        self.detectors['v1'] = get_input_detector("v1")
        self.detectors['v2'] = get_input_detector("v2")
        self.detectors['v3'] = get_input_detector("v3")
        print(f"✓ Loaded detectors: v1, v2, v3")
    
    def evaluate_configuration(self, config_name: str, eval_func) -> Dict:
        """Evaluate a configuration on adversarial attacks."""
        results = []
        attack_type_results = {}
        
        for sample in self.data:
            query = sample["query"]
            attack_type = sample["attack_type"]
            
            # Evaluate
            detected = eval_func(query)
            
            results.append({
                'query': query,
                'attack_type': attack_type,
                'detected': detected,
            })
            
            # Track by attack type
            if attack_type not in attack_type_results:
                attack_type_results[attack_type] = {'tp': 0, 'fn': 0}
            
            if detected:
                attack_type_results[attack_type]['tp'] += 1
            else:
                attack_type_results[attack_type]['fn'] += 1
        
        # Compute overall metrics
        total_tp = sum(1 for r in results if r['detected'])
        total_fn = sum(1 for r in results if not r['detected'])
        total_attacks = total_tp + total_fn
        
        tpr = total_tp / total_attacks if total_attacks > 0 else 0.0
        
        # Wilson CI for TPR
        z = stats.norm.ppf(0.975)  # 95% CI
        p = tpr
        denominator = 1 + z**2 / total_attacks
        center = (p + z**2 / (2 * total_attacks)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total_attacks)) / total_attacks) / denominator
        tpr_ci_low = max(0.0, center - margin)
        tpr_ci_high = min(1.0, center + margin)
        
        return {
            'config': config_name,
            'total_tp': total_tp,
            'total_fn': total_fn,
            'total_attacks': total_attacks,
            'tpr': tpr,
            'tpr_ci_low': tpr_ci_low,
            'tpr_ci_high': tpr_ci_high,
            'attack_type_results': attack_type_results,
        }
    
    def run(self):
        """Run complete evaluation."""
        print("\n" + "="*70)
        print("PHASE 6C: ADVERSARIAL ROBUSTNESS EVALUATION")
        print("="*70)
        
        self.load_adversarial_attacks()
        self.load_detectors()
        
        results = []
        
        # Baseline: v1
        print("\nEvaluating v1...")
        results.append(self.evaluate_configuration(
            'v1',
            lambda text: self.detectors['v1'].classify(text).is_attack
        ))
        
        # Baseline: v3
        print("Evaluating v3...")
        results.append(self.evaluate_configuration(
            'v3',
            lambda text: self.detectors['v3'].classify(text).is_attack
        ))
        
        # Baseline: v1+v3
        print("Evaluating v1+v3...")
        results.append(self.evaluate_configuration(
            'v1+v3',
            lambda text: (
                self.detectors['v1'].classify(text).is_attack or
                self.detectors['v3'].classify(text).is_attack
            )
        ))
        
        # Normalizer + v1
        print("Evaluating Normalizer+v1...")
        results.append(self.evaluate_configuration(
            'Normalizer+v1',
            lambda text: self.detectors['v1'].classify(
                normalize_text(text)['normalized']
            ).is_attack
        ))
        
        # Normalizer + v3
        print("Evaluating Normalizer+v3...")
        results.append(self.evaluate_configuration(
            'Normalizer+v3',
            lambda text: self.detectors['v3'].classify(
                normalize_text(text)['normalized']
            ).is_attack
        ))
        
        # Normalizer + v1+v3
        print("Evaluating Normalizer+v1+v3...")
        results.append(self.evaluate_configuration(
            'Normalizer+v1+v3',
            lambda text: (
                self.detectors['v1'].classify(normalize_text(text)['normalized']).is_attack or
                self.detectors['v3'].classify(normalize_text(text)['normalized']).is_attack
            )
        ))
        
        # Save results
        df_results = pd.DataFrame([{
            'config': r['config'],
            'total_tp': r['total_tp'],
            'total_fn': r['total_fn'],
            'total_attacks': r['total_attacks'],
            'tpr': r['tpr'],
            'tpr_ci_low': r['tpr_ci_low'],
            'tpr_ci_high': r['tpr_ci_high'],
        } for r in results])
        
        df_results.to_csv(
            self.results_dir / "adversarial_attacks_metrics.csv", index=False
        )
        print(f"\n✓ Saved metrics to {self.results_dir / 'adversarial_attacks_metrics.csv'}")
        
        # Print summary
        print("\n" + "="*70)
        print("PHASE 6C EVALUATION SUMMARY")
        print("="*70)
        print(df_results.to_string(index=False))
        
        # Print by attack type
        print("\n" + "-"*70)
        print("TPR BY ATTACK TYPE")
        print("-"*70)
        
        for result in results:
            print(f"\n{result['config']}:")
            for attack_type, counts in sorted(result['attack_type_results'].items()):
                total = counts['tp'] + counts['fn']
                tpr_type = counts['tp'] / total if total > 0 else 0.0
                print(f"  {attack_type}: {tpr_type:.1%} TPR ({counts['tp']}/{total})")
        
        print("\n" + "="*70)
        print("✅ PHASE 6C EVALUATION COMPLETE")
        print("="*70)


def main():
    """Main entry point."""
    evaluator = AdversarialAttackEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
