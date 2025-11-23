"""
Phase 5: Complete Obfuscation-Robust Detection Pipeline
Orchestrates normalizer, feature extraction, learned fusion training, and evaluation.
"""

import sys
from pathlib import Path

# Import pipeline components
from train_learned_fusion import LearnedFusionTrainer
from evaluate_phase5 import Phase5Evaluator
from generate_plots_phase5 import Phase5Plotter


def main():
    """Run complete Phase 5 pipeline."""
    print("\n" + "="*70)
    print("PHASE 5: OBFUSCATION-ROBUST DETECTION WITH LEARNED FUSION")
    print("="*70)
    print("Objective: Harden detection against obfuscation and add learned fusion")
    print("Dataset: Phase 1 Part A (400 samples)")
    print("="*70)
    
    # Step 1: Train learned fusion with FPR-constrained thresholding
    print("\n" + "="*70)
    print("STEP 1: TRAIN LEARNED FUSION")
    print("="*70)
    
    trainer = LearnedFusionTrainer()
    trainer.run()
    
    # Step 2: Evaluate configurations
    print("\n" + "="*70)
    print("STEP 2: EVALUATE CONFIGURATIONS")
    print("="*70)
    
    evaluator = Phase5Evaluator()
    evaluator.run()
    
    # Step 3: Generate visualizations
    print("\n" + "="*70)
    print("STEP 3: GENERATE VISUALIZATIONS")
    print("="*70)
    
    plotter = Phase5Plotter()
    plotter.run()
    
    # Summary
    print("\n" + "="*70)
    print("✅ PHASE 5 COMPLETE!")
    print("="*70)
    print("\n📊 Deliverables:")
    print("  ✓ phase5/results/learned_fusion_cv_metrics.csv")
    print("  ✓ phase5/results/learned_fusion_feature_importance.csv")
    print("  ✓ phase5/results/learned_fusion_thresholds.csv")
    print("  ✓ phase5/results/phase5_comparison_metrics.csv")
    print("  ✓ phase5/plots/feature_importance.png")
    print("  ✓ phase5/plots/cv_metrics.png")
    print("  ✓ phase5/plots/comparison.png")
    print("\n🎯 Key Metrics:")
    print("  • Goal: ≥90% TPR @ ≤1% FAR")
    print("  • Lift on obfuscation: homoglyph, ZWJ, multilingual")
    print("  • No regression: Phase 3 baseline unchanged")
    print("\n📝 Next: Review PHASE5_OBFUSCATION_ROBUST_SUMMARY.md")
    print("="*70)


if __name__ == "__main__":
    main()
