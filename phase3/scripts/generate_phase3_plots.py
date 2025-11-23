"""
Phase 3: Generate visualizations for multilayer defense evaluation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class Phase3PlotGenerator:
    """Generate plots for Phase 3 analysis."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.phase3_dir = self.script_dir.parent
        self.results_dir = self.phase3_dir / "results"
        self.plots_dir = self.phase3_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 7)
    
    def load_data(self):
        """Load results."""
        self.metrics = pd.read_csv(self.results_dir / "multilayer_metrics_summary.csv")
        print("✓ Loaded metrics")
    
    def plot_tpr_fpr_comparison(self):
        """Plot TPR and FAR for all configurations."""
        print("\n📊 Generating TPR/FAR comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        configs = self.metrics['config_id'].tolist()
        tpr = self.metrics['tpr'].tolist()
        tpr_low = self.metrics['tpr_ci_low'].tolist()
        tpr_high = self.metrics['tpr_ci_high'].tolist()
        
        far = self.metrics['far'].tolist()
        far_low = self.metrics['far_ci_low'].tolist()
        far_high = self.metrics['far_ci_high'].tolist()
        
        x = np.arange(len(configs))
        
        # TPR plot
        colors = ['#2ecc71' if len(c.split()) == 1 else '#3498db' if len(c.split()) == 2 else '#9b59b6' 
                 for c in self.metrics['components'].tolist()]
        
        bars1 = ax1.bar(x, [t*100 for t in tpr], color=colors, alpha=0.8)
        yerr_low = [(tpr[i] - tpr_low[i])*100 for i in range(len(configs))]
        yerr_high = [(tpr_high[i] - tpr[i])*100 for i in range(len(configs))]
        ax1.errorbar(x, [t*100 for t in tpr], yerr=[yerr_low, yerr_high],
                     fmt='none', ecolor='black', capsize=5, alpha=0.6)
        
        ax1.set_ylabel('TPR (%)', fontsize=12, fontweight='bold')
        ax1.set_title('True Positive Rate by Configuration', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, fontsize=11, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{tpr[i]*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # FAR plot
        bars2 = ax2.bar(x, [f*100 for f in far], color=colors, alpha=0.8)
        yerr_low = [max(0, (far[i] - far_low[i])*100) for i in range(len(configs))]
        yerr_high = [max(0, (far_high[i] - far[i])*100) for i in range(len(configs))]
        if any(yerr_low) or any(yerr_high):
            ax2.errorbar(x, [f*100 for f in far], yerr=[yerr_low, yerr_high],
                         fmt='none', ecolor='black', capsize=5, alpha=0.6)
        
        ax2.set_ylabel('FAR - False Alarm Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('False Alarm Rate (Benign Queries)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, fontsize=11, fontweight='bold')
        ax2.set_ylim([0, max([f*100 for f in far]) * 1.5 + 1])
        ax2.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{far[i]*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        output_file = self.plots_dir / "tpr_fpr_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def plot_pareto_frontier(self):
        """Plot Pareto frontier (TPR vs FAR)."""
        print("\n📊 Generating Pareto frontier...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = self.metrics['config_id'].tolist()
        tpr = self.metrics['tpr'].tolist()
        far = self.metrics['far'].tolist()
        latency = self.metrics['avg_latency_ms'].tolist()
        
        # Color by number of components
        colors = []
        for comp in self.metrics['components'].tolist():
            n_comp = len(comp.split(', '))
            if n_comp == 1:
                colors.append('#2ecc71')  # Green - single
            elif n_comp == 2:
                colors.append('#3498db')  # Blue - dual
            else:
                colors.append('#9b59b6')  # Purple - triple
        
        # Plot points
        scatter = ax.scatter([f*100 for f in far], [t*100 for t in tpr], 
                            s=[200 + l*20 for l in latency],  # Size by latency
                            c=colors, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Annotate points
        for i, config in enumerate(configs):
            ax.annotate(config, (far[i]*100, tpr[i]*100),
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold', fontsize=11)
        
        ax.set_xlabel('False Alarm Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontier: TPR vs FAR\n(Bubble size = latency)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Single component'),
            Patch(facecolor='#3498db', label='Dual components'),
            Patch(facecolor='#9b59b6', label='Triple components'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        output_file = self.plots_dir / "pareto_frontier.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def plot_f1_scores(self):
        """Plot F1 scores for all configurations."""
        print("\n📊 Generating F1 scores...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = self.metrics['config_id'].tolist()
        f1_scores = self.metrics['f1'].tolist()
        
        colors = ['#2ecc71' if len(c.split()) == 1 else '#3498db' if len(c.split()) == 2 else '#9b59b6'
                 for c in self.metrics['components'].tolist()]
        
        bars = ax.bar(configs, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('F1 Score by Configuration', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{f1_scores[i]:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        output_file = self.plots_dir / "f1_scores.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def plot_latency_comparison(self):
        """Plot average latency for all configurations."""
        print("\n📊 Generating latency comparison...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = self.metrics['config_id'].tolist()
        latencies = self.metrics['avg_latency_ms'].tolist()
        
        colors = ['#2ecc71' if len(c.split()) == 1 else '#3498db' if len(c.split()) == 2 else '#9b59b6'
                 for c in self.metrics['components'].tolist()]
        
        bars = ax.bar(configs, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Detection Latency by Configuration', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{latencies[i]:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        output_file = self.plots_dir / "latency_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {output_file}")
        plt.close()
    
    def generate_all(self):
        """Generate all plots."""
        print("\n" + "="*70)
        print("GENERATING PHASE 3 VISUALIZATIONS")
        print("="*70)
        
        self.load_data()
        self.plot_tpr_fpr_comparison()
        self.plot_pareto_frontier()
        self.plot_f1_scores()
        self.plot_latency_comparison()
        
        print(f"\n✅ All plots generated!")
        print(f"Saved to: {self.plots_dir}")


if __name__ == "__main__":
    generator = Phase3PlotGenerator()
    generator.generate_all()
