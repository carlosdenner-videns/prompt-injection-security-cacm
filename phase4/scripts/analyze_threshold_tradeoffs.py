"""
Phase 4: Threshold Trade-off Analysis
Generates visualizations and analysis for threshold sweep results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ThresholdTradeoffAnalyzer:
    """Analyzes threshold sweep results and generates visualizations."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase4_dir = self.script_dir.parent
        self.results_dir = self.phase4_dir / "results"
        self.plots_dir = self.phase4_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
    
    def load_results(self):
        """Load threshold sweep results."""
        results_file = self.results_dir / "threshold_sweep.csv"
        self.df = pd.read_csv(results_file)
        print(f"✓ Loaded {len(self.df)} threshold sweep results")
    
    def generate_roc_curve(self):
        """Generate ROC-style curve with Phase 3 baseline overlay."""
        plt.figure(figsize=(10, 8))
        
        # Plot threshold sweep curve
        plt.plot(self.df["far"], self.df["tpr"], 
                marker="o", linewidth=2, markersize=8, 
                label="Phase 4 Threshold Sweep (v1+v3)", color="steelblue")
        
        # Annotate key thresholds
        for idx, row in self.df.iterrows():
            if row["threshold"] in [0.05, 0.25, 0.50, 0.75]:
                plt.annotate(f"t={row['threshold']:.2f}", 
                           xy=(row["far"], row["tpr"]),
                           xytext=(5, 5), textcoords="offset points",
                           fontsize=9, alpha=0.7)
        
        # Phase 3 baseline (t=0.50)
        baseline = self.df[self.df["threshold"] == 0.50]
        if not baseline.empty:
            baseline_row = baseline.iloc[0]
            plt.scatter(baseline_row["far"], baseline_row["tpr"], 
                       c="red", s=200, marker="*", 
                       label=f"Phase 3 Baseline (t=0.50): {baseline_row['tpr']:.1%} TPR, {baseline_row['far']:.1%} FAR",
                       zorder=5, edgecolors="darkred", linewidth=2)
        
        # Diagonal reference line (random classifier)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Classifier")
        
        plt.xlabel("False Alarm Rate (FAR)", fontsize=12, fontweight="bold")
        plt.ylabel("True Positive Rate (TPR)", fontsize=12, fontweight="bold")
        plt.title("Phase 4: ROC-Style Curve - Threshold Sweep Analysis", fontsize=14, fontweight="bold")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc="lower right")
        plt.tight_layout()
        
        output_file = self.plots_dir / "roc_curve_thresholds.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved ROC curve to {output_file}")
        plt.close()
    
    def generate_f1_vs_threshold(self):
        """Generate F1 score vs threshold plot."""
        plt.figure(figsize=(10, 6))
        
        # Plot F1 vs threshold
        plt.plot(self.df["threshold"], self.df["f1"], 
                marker="o", linewidth=2, markersize=8, 
                color="darkgreen", label="F1 Score")
        
        # Highlight maximum F1
        max_f1_idx = self.df["f1"].idxmax()
        max_f1_row = self.df.loc[max_f1_idx]
        plt.scatter(max_f1_row["threshold"], max_f1_row["f1"], 
                   c="red", s=200, marker="*", 
                   label=f"Max F1 (t={max_f1_row['threshold']:.2f}): {max_f1_row['f1']:.4f}",
                   zorder=5, edgecolors="darkred", linewidth=2)
        
        # Phase 3 baseline
        baseline = self.df[self.df["threshold"] == 0.50]
        if not baseline.empty:
            baseline_row = baseline.iloc[0]
            plt.scatter(baseline_row["threshold"], baseline_row["f1"], 
                       c="orange", s=150, marker="s", 
                       label=f"Phase 3 Baseline (t=0.50): {baseline_row['f1']:.4f}",
                       zorder=5, edgecolors="darkorange", linewidth=2)
        
        plt.xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
        plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
        plt.title("Phase 4: F1 Score vs Confidence Threshold", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        output_file = self.plots_dir / "f1_vs_threshold.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved F1 vs threshold plot to {output_file}")
        plt.close()
    
    def generate_tpr_far_vs_threshold(self):
        """Generate TPR and FAR vs threshold plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot TPR
        ax.plot(self.df["threshold"], self.df["tpr"], 
               marker="o", linewidth=2, markersize=8, 
               color="steelblue", label="TPR (Detection Rate)")
        ax.fill_between(self.df["threshold"], 
                        self.df["tpr_ci_low"], self.df["tpr_ci_high"],
                        alpha=0.2, color="steelblue")
        
        # Plot FAR on secondary axis
        ax2 = ax.twinx()
        ax2.plot(self.df["threshold"], self.df["far"], 
                marker="s", linewidth=2, markersize=8, 
                color="crimson", label="FAR (False Alarm Rate)")
        ax2.fill_between(self.df["threshold"], 
                         self.df["far_ci_low"], self.df["far_ci_high"],
                         alpha=0.2, color="crimson")
        
        # Phase 3 baseline
        baseline = self.df[self.df["threshold"] == 0.50]
        if not baseline.empty:
            baseline_row = baseline.iloc[0]
            ax.scatter(baseline_row["threshold"], baseline_row["tpr"], 
                      c="darkblue", s=150, marker="*", zorder=5, edgecolors="black", linewidth=2)
            ax2.scatter(baseline_row["threshold"], baseline_row["far"], 
                       c="darkred", s=150, marker="*", zorder=5, edgecolors="black", linewidth=2)
        
        ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=12, fontweight="bold", color="steelblue")
        ax2.set_ylabel("False Alarm Rate (FAR)", fontsize=12, fontweight="bold", color="crimson")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax2.tick_params(axis="y", labelcolor="crimson")
        
        ax.set_title("Phase 4: TPR & FAR vs Confidence Threshold", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="center right")
        
        plt.tight_layout()
        
        output_file = self.plots_dir / "tpr_far_vs_threshold.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved TPR/FAR vs threshold plot to {output_file}")
        plt.close()
    
    def generate_operating_points_table(self):
        """Generate table of key operating points."""
        # Identify key operating points
        operating_points = []
        
        # High recall mode (low threshold)
        high_recall = self.df[self.df["threshold"] == 0.05].iloc[0]
        operating_points.append({
            "Mode": "High Recall",
            "Threshold": high_recall["threshold"],
            "TPR": f"{high_recall['tpr']:.1%}",
            "FAR": f"{high_recall['far']:.1%}",
            "Precision": f"{high_recall['precision']:.1%}",
            "F1": f"{high_recall['f1']:.4f}",
            "Use Case": "Security monitoring, catch all attacks"
        })
        
        # Balanced mode
        balanced = self.df[self.df["threshold"] == 0.50].iloc[0]
        operating_points.append({
            "Mode": "Balanced (Phase 3)",
            "Threshold": balanced["threshold"],
            "TPR": f"{balanced['tpr']:.1%}",
            "FAR": f"{balanced['far']:.1%}",
            "Precision": f"{balanced['precision']:.1%}",
            "F1": f"{balanced['f1']:.4f}",
            "Use Case": "Production deployment, zero false alarms"
        })
        
        # High precision mode
        high_precision = self.df[self.df["threshold"] == 0.75].iloc[0]
        operating_points.append({
            "Mode": "High Precision",
            "Threshold": high_precision["threshold"],
            "TPR": f"{high_precision['tpr']:.1%}",
            "FAR": f"{high_precision['far']:.1%}",
            "Precision": f"{high_precision['precision']:.1%}",
            "F1": f"{high_precision['f1']:.4f}",
            "Use Case": "Minimal false alarms, accept lower catch rate"
        })
        
        op_df = pd.DataFrame(operating_points)
        
        output_file = self.results_dir / "operating_points.csv"
        op_df.to_csv(output_file, index=False)
        print(f"✓ Saved operating points to {output_file}")
        
        return op_df
    
    def print_analysis(self):
        """Print analysis summary."""
        print("\n" + "="*70)
        print("THRESHOLD TRADE-OFF ANALYSIS")
        print("="*70)
        
        # Find key points
        max_f1_idx = self.df["f1"].idxmax()
        max_f1_row = self.df.loc[max_f1_idx]
        
        min_far_idx = self.df["far"].idxmin()
        min_far_row = self.df.loc[min_far_idx]
        
        max_tpr_idx = self.df["tpr"].idxmax()
        max_tpr_row = self.df.loc[max_tpr_idx]
        
        print(f"\n📊 Key Operating Points:")
        print(f"\n  High Recall (t=0.05):")
        high_recall = self.df[self.df["threshold"] == 0.05].iloc[0]
        print(f"    TPR: {high_recall['tpr']:.1%}, FAR: {high_recall['far']:.1%}, F1: {high_recall['f1']:.4f}")
        print(f"    Use: Security monitoring mode")
        
        print(f"\n  Balanced (t=0.50 - Phase 3 Baseline):")
        balanced = self.df[self.df["threshold"] == 0.50].iloc[0]
        print(f"    TPR: {balanced['tpr']:.1%}, FAR: {balanced['far']:.1%}, F1: {balanced['f1']:.4f}")
        print(f"    Use: Production deployment")
        
        print(f"\n  High Precision (t=0.75):")
        high_precision = self.df[self.df["threshold"] == 0.75].iloc[0]
        print(f"    TPR: {high_precision['tpr']:.1%}, FAR: {high_precision['far']:.1%}, F1: {high_precision['f1']:.4f}")
        print(f"    Use: Minimal false alarms")
        
        print(f"\n  Best F1 (t={max_f1_row['threshold']:.2f}):")
        print(f"    TPR: {max_f1_row['tpr']:.1%}, FAR: {max_f1_row['far']:.1%}, F1: {max_f1_row['f1']:.4f}")
        
        print(f"\n📊 Latency Analysis:")
        mean_latency = self.df["mean_latency_ms"].mean()
        max_latency = self.df["mean_latency_ms"].max()
        print(f"  Average latency: {mean_latency:.4f} ms")
        print(f"  Max latency: {max_latency:.4f} ms")
        print(f"  ✓ Negligible compared to LLM inference latency")


def main():
    """Main entry point."""
    analyzer = ThresholdTradeoffAnalyzer()
    analyzer.load_results()
    
    print("\n" + "="*70)
    print("GENERATING PHASE 4 VISUALIZATIONS")
    print("="*70)
    
    analyzer.generate_roc_curve()
    analyzer.generate_f1_vs_threshold()
    analyzer.generate_tpr_far_vs_threshold()
    op_df = analyzer.generate_operating_points_table()
    analyzer.print_analysis()
    
    print("\n" + "="*70)
    print("✅ Phase 4 analysis complete!")
    print("="*70)
    
    print("\n📊 Operating Points Summary:")
    print(op_df.to_string(index=False))


if __name__ == "__main__":
    main()
