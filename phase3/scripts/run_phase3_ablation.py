"""
Phase 3: Main orchestrator script
Runs complete multilayer defense evaluation pipeline.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add phase2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2_input_detection" / "scripts"))

from evaluate_multilayer import MultilayerDefenseEvaluator
from generate_phase3_plots import Phase3PlotGenerator


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Multilayer Defense Evaluation"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (default: 0.5)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print_header("PHASE 3: MULTILAYER DEFENSE EVALUATION")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Confidence threshold: {args.threshold}")
    
    # Run evaluation
    print_header("Step 1: Evaluating Multilayer Configurations")
    evaluator = MultilayerDefenseEvaluator(threshold=args.threshold)
    df, metrics_df, pareto = evaluator.run_evaluation()
    
    # Generate plots
    if not args.skip_plots:
        print_header("Step 2: Generating Visualizations")
        plotter = Phase3PlotGenerator()
        plotter.generate_all()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("PHASE 3 COMPLETE")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration.total_seconds():.1f} seconds")
    
    print("\n📊 Generated outputs:")
    print("  ✓ phase3/results/multilayer_defense_results.csv")
    print("  ✓ phase3/results/multilayer_metrics_summary.csv")
    print("  ✓ phase3/results/mcnemar_comparisons.csv")
    print("  ✓ phase3/plots/tpr_fpr_comparison.png")
    print("  ✓ phase3/plots/pareto_frontier.png")
    print("  ✓ phase3/plots/f1_scores.png")
    print("  ✓ phase3/plots/latency_comparison.png")
    
    print("\n🎯 Pareto-optimal configurations:")
    for config_id in pareto:
        row = metrics_df[metrics_df['config_id'] == config_id].iloc[0]
        print(f"  {config_id}: {row['config_name']}")
        print(f"     TPR: {row['tpr']:.1%}, FAR: {row['far']:.1%}, Latency: {row['avg_latency_ms']:.2f}ms")
    
    print("\n" + "="*70)
    print("Next: Review PHASE3_MULTILAYER_SUMMARY.md for detailed analysis")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
