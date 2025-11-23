"""
Phase 6a: Evaluate Obfuscation-Benign Validation
Tests FAR on benign queries with obfuscation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
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


class ObfuscatedBenignEvaluator:
    """Evaluates FAR on obfuscated benign queries."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase6a_dir = self.script_dir.parent
        self.phase5_dir = self.phase6a_dir.parent / "phase5"
        self.results_dir = self.phase6a_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.detectors = {}
        self.fusion_model = None
        self.fusion_threshold = None
    
    def load_obfuscated_benign(self):
        """Load obfuscated benign dataset."""
        data_file = self.phase6a_dir / "data" / "obfuscated_benign_queries.json"
        
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"✓ Loaded {len(self.data)} obfuscated benign samples")
    
    def load_detectors(self):
        """Load v1, v2, v3 detectors."""
        self.detectors['v1'] = get_input_detector("v1")
        self.detectors['v2'] = get_input_detector("v2")
        self.detectors['v3'] = get_input_detector("v3")
        print(f"✓ Loaded detectors: v1, v2, v3")
    
    def load_fusion_model(self):
        """Load trained fusion model from Phase 5."""
        # Load CV results to get mean threshold
        cv_file = self.phase5_dir / "results" / "fusion_threshold_sweep_cv.csv"
        df_cv = pd.read_csv(cv_file)
        
        # Use mean threshold across folds
        self.fusion_threshold = df_cv['threshold'].mean()
        print(f"✓ Loaded fusion threshold: {self.fusion_threshold:.4f}")
    
    def evaluate_configuration(self, config_name: str, eval_func) -> Dict:
        """Evaluate a configuration on obfuscated benign queries."""
        results = []
        obfuscation_type_results = {}
        
        for sample in self.data:
            query = sample["query"]
            obfuscation_type = sample["obfuscation_type"]
            
            # Evaluate
            detected = eval_func(query)
            
            results.append({
                'query': query,
                'obfuscation_type': obfuscation_type,
                'detected': detected,
            })
            
            # Track by obfuscation type
            if obfuscation_type not in obfuscation_type_results:
                obfuscation_type_results[obfuscation_type] = {'fp': 0, 'tn': 0}
            
            if detected:
                obfuscation_type_results[obfuscation_type]['fp'] += 1
            else:
                obfuscation_type_results[obfuscation_type]['tn'] += 1
        
        # Compute overall metrics
        total_fp = sum(1 for r in results if r['detected'])
        total_tn = sum(1 for r in results if not r['detected'])
        total_benign = total_fp + total_tn
        
        far = total_fp / total_benign if total_benign > 0 else 0.0
        
        # Wilson CI for FAR
        z = stats.norm.ppf(0.975)  # 95% CI
        p = far
        denominator = 1 + z**2 / total_benign
        center = (p + z**2 / (2 * total_benign)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total_benign)) / total_benign) / denominator
        far_ci_low = max(0.0, center - margin)
        far_ci_high = min(1.0, center + margin)
        
        return {
            'config': config_name,
            'total_fp': total_fp,
            'total_tn': total_tn,
            'total_benign': total_benign,
            'far': far,
            'far_ci_low': far_ci_low,
            'far_ci_high': far_ci_high,
            'obfuscation_type_results': obfuscation_type_results,
        }
    
    def run(self):
        """Run complete evaluation."""
        print("\n" + "="*70)
        print("PHASE 6A: OBFUSCATION-BENIGN VALIDATION")
        print("="*70)
        
        self.load_obfuscated_benign()
        self.load_detectors()
        self.load_fusion_model()
        
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
            'total_fp': r['total_fp'],
            'total_tn': r['total_tn'],
            'total_benign': r['total_benign'],
            'far': r['far'],
            'far_ci_low': r['far_ci_low'],
            'far_ci_high': r['far_ci_high'],
        } for r in results])
        
        df_results.to_csv(
            self.results_dir / "obfuscated_benign_metrics.csv", index=False
        )
        print(f"\n✓ Saved metrics to {self.results_dir / 'obfuscated_benign_metrics.csv'}")
        
        # Print summary
        print("\n" + "="*70)
        print("PHASE 6A EVALUATION SUMMARY")
        print("="*70)
        print(df_results.to_string(index=False))
        
        # Print by obfuscation type
        print("\n" + "-"*70)
        print("FAR BY OBFUSCATION TYPE")
        print("-"*70)
        
        for result in results:
            print(f"\n{result['config']}:")
            for obf_type, counts in sorted(result['obfuscation_type_results'].items()):
                total = counts['fp'] + counts['tn']
                far_type = counts['fp'] / total if total > 0 else 0.0
                print(f"  {obf_type}: {far_type:.1%} FAR ({counts['fp']}/{total})")
        
        print("\n" + "="*70)
        print("✅ PHASE 6A EVALUATION COMPLETE")
        print("="*70)


def main():
    """Main entry point."""
    evaluator = ObfuscatedBenignEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
