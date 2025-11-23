"""
Generate visualizations for Phase 2 input-side detection.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class PlotGenerator:
    """Generate plots for Phase 2 input detection."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase2_dir = self.script_dir.parent
        self.results_dir = self.phase2_dir / "results"
        self.plots_dir = self.phase2_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def load_data(self):
        """Load results."""
        self.metrics = pd.read_csv(self.results_dir / "input_detection_metrics.csv")
        self.detailed = pd.read_csv(self.results_dir / "phase2_input_detection_results.csv")
        print("✓ Loaded metrics and detailed results")
    
    def plot_tpr_far_comparison(self):
        """Plot TPR and FAR comparison."""
        print("\n📊 Generating TPR/FAR comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        versions = self.metrics['version'].tolist()
        tpr = self.metrics['tpr'].tolist()
        tpr_low = self.metrics['tpr_ci_low'].tolist()
        tpr_high = self.metrics['tpr_ci_high'].tolist()
        
        far = self.metrics['far'].tolist()
        far_low = self.metrics['far_ci_low'].tolist()
        far_high = self.metrics['far_ci_high'].tolist()
        
        x = np.arange(len(versions))
        
        # TPR plot
        bars1 = ax1.bar(x, [t*100 for t in tpr], color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
        yerr_low = [(tpr[i] - tpr_low[i])*100 for i in range(len(versions))]
        yerr_high = [(tpr_high[i] - tpr[i])*100 for i in range(len(versions))]
        ax1.errorbar(x, [t*100 for t in tpr], yerr=[yerr_low, yerr_high],
                     fmt='none', ecolor='black', capsize=5, alpha=0.6)
        
        ax1.set_ylabel('TPR - Detection Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('True Positive Rate (Successful Attacks)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(versions)
        ax1.set_ylim([0, 105])
        ax1.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{tpr[i]*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # FAR plot
        bars2 = ax2.bar(x, [f*100 for f in far], color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8)
        yerr_low = [max(0, (far[i] - far_low[i])*100) for i in range(len(versions))]
        yerr_high = [max(0, (far_high[i] - far[i])*100) for i in range(len(versions))]
        if any(yerr_low) or any(yerr_high):
            ax2.errorbar(x, [f*100 for f in far], yerr=[yerr_low, yerr_high],
                         fmt='none', ecolor='black', capsize=5, alpha=0.6)
        
        ax2.set_ylabel('FAR - False Alarm Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('False Alarm Rate (Benign Queries)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(versions)
        ax2.set_ylim([0, 2])
        ax2.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{far[i]*100:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_file = self.plots_dir / "tpr_far_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices."""
        print("\n📊 Generating confusion matrices...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, version in enumerate(['v1', 'v2', 'v3']):
            ax = axes[idx]
            row = self.metrics[self.metrics['version'] == version].iloc[0]
            
            cm = np.array([
                [row['tn'], row['fp']],
                [row['fn_successful'], row['tp_successful']]
            ])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted Benign', 'Predicted Attack'],
                       yticklabels=['Actual Benign', 'Actual Attack'],
                       ax=ax, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
            
            f1 = row['f1']
            ax.set_title(f'{version.upper()}\nF1: {f1:.4f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.plots_dir / "confusion_matrices.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def plot_detection_by_evasion_type(self):
        """Plot detection rates by evasion type."""
        print("\n📊 Generating detection by evasion type...")
        
        # Analyze v1 detection by evasion type
        successful = self.detailed[self.detailed['injection_success'] == True]
        
        evasion_types = successful['evasion_type'].unique()
        detection_rates = []
        
        for ev_type in sorted(evasion_types):
            subset = successful[successful['evasion_type'] == ev_type]
            detected = (subset['v1_detected'] == True).sum()
            total = len(subset)
            rate = detected / total if total > 0 else 0
            detection_rates.append({'evasion_type': ev_type, 'rate': rate, 'detected': detected, 'total': total})
        
        df_rates = pd.DataFrame(detection_rates)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(df_rates['evasion_type'], df_rates['rate']*100, color='#3498db', alpha=0.8)
        
        ax.set_xlabel('Detection Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('v1 Detection Rate by Evasion Type', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 105])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, row) in enumerate(zip(bars, df_rates.itertuples())):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                   f'{row.rate*100:.0f}% ({row.detected}/{row.total})',
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        output_file = self.plots_dir / "detection_by_evasion_type.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def plot_metrics_summary(self):
        """Plot summary metrics."""
        print("\n📊 Generating metrics summary...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        versions = self.metrics['version'].tolist()
        metrics_to_plot = ['tpr', 'accuracy', 'precision', 'f1']
        
        x = np.arange(len(versions))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = self.metrics[metric].tolist()
            ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(versions)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.plots_dir / "metrics_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def generate_all(self):
        """Generate all plots."""
        print("\n" + "="*70)
        print("GENERATING PHASE 2 INPUT DETECTION VISUALIZATIONS")
        print("="*70)
        
        self.load_data()
        self.plot_tpr_far_comparison()
        self.plot_confusion_matrices()
        self.plot_detection_by_evasion_type()
        self.plot_metrics_summary()
        
        print(f"\n✅ All plots generated!")
        print(f"Saved to: {self.plots_dir}")


if __name__ == "__main__":
    generator = PlotGenerator()
    generator.generate_all()
