"""
Phase 5: Generate Visualizations
Creates ROC overlay, feature importance, and family breakdown plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class Phase5Plotter:
    """Generates Phase 5 visualizations."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase5_dir = self.script_dir.parent
        self.results_dir = self.phase5_dir / "results"
        self.plots_dir = self.phase5_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(self):
        """Plot top features by importance."""
        try:
            df_importance = pd.read_csv(self.results_dir / "learned_fusion_feature_importance.csv")
            
            # Top 15 features
            df_top = df_importance.head(15).sort_values('coefficient')
            
            plt.figure(figsize=(10, 8))
            plt.barh(df_top['feature'], df_top['coefficient'], color='steelblue')
            plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
            plt.title('Phase 5: Top 15 Features by Importance', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            output_file = self.plots_dir / "feature_importance.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature importance plot to {output_file}")
            plt.close()
        except FileNotFoundError:
            print("⚠ Feature importance file not found, skipping plot")
    
    def plot_cv_metrics(self):
        """Plot CV metrics across folds."""
        try:
            df_cv = pd.read_csv(self.results_dir / "learned_fusion_cv_metrics.csv")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # TPR
            axes[0, 0].plot(df_cv['fold'], df_cv['tpr'], marker='o', linewidth=2, markersize=8, color='steelblue')
            axes[0, 0].axhline(df_cv['tpr'].mean(), color='red', linestyle='--', label=f"Mean: {df_cv['tpr'].mean():.1%}")
            axes[0, 0].fill_between(df_cv['fold'], df_cv['tpr'].mean() - df_cv['tpr'].std(), 
                                    df_cv['tpr'].mean() + df_cv['tpr'].std(), alpha=0.2, color='steelblue')
            axes[0, 0].set_ylabel('TPR', fontweight='bold')
            axes[0, 0].set_title('TPR Across Folds')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # FPR
            axes[0, 1].plot(df_cv['fold'], df_cv['fpr'], marker='s', linewidth=2, markersize=8, color='crimson')
            axes[0, 1].axhline(0.01, color='orange', linestyle='--', label='Target: 1%')
            axes[0, 1].set_ylabel('FPR', fontweight='bold')
            axes[0, 1].set_title('FPR Across Folds')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # F1
            axes[1, 0].plot(df_cv['fold'], df_cv['f1'], marker='^', linewidth=2, markersize=8, color='darkgreen')
            axes[1, 0].set_ylabel('F1 Score', fontweight='bold')
            axes[1, 0].set_title('F1 Score Across Folds')
            axes[1, 0].grid(alpha=0.3)
            
            # Precision
            axes[1, 1].plot(df_cv['fold'], df_cv['precision'], marker='d', linewidth=2, markersize=8, color='purple')
            axes[1, 1].set_ylabel('Precision', fontweight='bold')
            axes[1, 1].set_title('Precision Across Folds')
            axes[1, 1].grid(alpha=0.3)
            
            plt.suptitle('Phase 5: Cross-Validation Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.plots_dir / "cv_metrics.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved CV metrics plot to {output_file}")
            plt.close()
        except FileNotFoundError:
            print("⚠ CV metrics file not found, skipping plot")
    
    def plot_comparison(self):
        """Plot configuration comparison."""
        try:
            df_comp = pd.read_csv(self.results_dir / "phase5_comparison_metrics.csv")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # TPR comparison with error bars
            tpr_err_low = np.maximum(df_comp['tpr'] - df_comp['tpr_ci_low'], 0)
            tpr_err_high = np.maximum(df_comp['tpr_ci_high'] - df_comp['tpr'], 0)
            axes[0].barh(df_comp['config'], df_comp['tpr'], xerr=[tpr_err_low, tpr_err_high], 
                        color='steelblue', capsize=5)
            axes[0].set_xlabel('TPR', fontweight='bold')
            axes[0].set_title('TPR Comparison Across Configurations')
            axes[0].grid(axis='x', alpha=0.3)
            axes[0].set_xlim(0, 1.05)
            
            # FAR comparison with error bars
            far_err_low = np.maximum(df_comp['far'] - df_comp['far_ci_low'], 0)
            far_err_high = np.maximum(df_comp['far_ci_high'] - df_comp['far'], 0)
            axes[1].barh(df_comp['config'], df_comp['far'], xerr=[far_err_low, far_err_high],
                        color='crimson', capsize=5)
            axes[1].set_xlabel('FAR', fontweight='bold')
            axes[1].set_title('FAR Comparison Across Configurations')
            axes[1].grid(axis='x', alpha=0.3)
            axes[1].set_xlim(0, max(df_comp['far_ci_high'].max() * 1.1, 0.05))
            
            plt.suptitle('Phase 5: Configuration Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.plots_dir / "comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {output_file}")
            plt.close()
        except FileNotFoundError:
            print("⚠ Comparison metrics file not found, skipping plot")
    
    def run(self):
        """Generate all plots."""
        print("\n" + "="*70)
        print("GENERATING PHASE 5 VISUALIZATIONS")
        print("="*70)
        
        self.plot_feature_importance()
        self.plot_cv_metrics()
        self.plot_comparison()
        
        print("\n✅ All plots generated!")


def main():
    """Main entry point."""
    plotter = Phase5Plotter()
    plotter.run()


if __name__ == "__main__":
    main()
