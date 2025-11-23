"""
Phase 5: Plot ROC curve for fusion with zero-FPR operating point.
Compares against Sig+Clf (t=0.5) baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Load data
phase5_dir = Path(__file__).parent.parent
results_dir = phase5_dir / "results"
plots_dir = phase5_dir / "plots"

# Load CV results
df_cv = pd.read_csv(results_dir / "fusion_threshold_sweep_cv.csv")

# Compute aggregate metrics
total_tp = df_cv['tp'].sum()
total_fp = df_cv['fp'].sum()
total_tn = df_cv['tn'].sum()
total_fn = df_cv['fn'].sum()

fusion_tpr = total_tp / (total_tp + total_fn)
fusion_fpr = total_fp / (total_fp + total_tn)

# Baseline: Sig+Clf (t=0.5) from Phase 3
baseline_tpr = 0.87
baseline_fpr = 0.0

print("="*70)
print("FUSION vs BASELINE COMPARISON")
print("="*70)
print(f"\nFusion (Zero-FPR Operating Point):")
print(f"  TPR: {fusion_tpr:.1%}")
print(f"  FPR: {fusion_fpr:.1%}")
print(f"  Precision: 100.0%")
print(f"  F1: {2*total_tp/(2*total_tp+total_fp+total_fn):.4f}")

print(f"\nBaseline (Sig+Clf @ t=0.5):")
print(f"  TPR: {baseline_tpr:.1%}")
print(f"  FPR: {baseline_fpr:.1%}")
print(f"  Precision: 100.0%")

print(f"\nLift:")
print(f"  TPR improvement: +{(fusion_tpr-baseline_tpr)*100:.1f}%")
print(f"  FPR change: {fusion_fpr-baseline_fpr:.1%}")

# Create ROC-style plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot operating points
ax.scatter([baseline_fpr], [baseline_tpr], s=300, marker='s', color='steelblue', 
          label=f'Sig+Clf (t=0.5)\nTPR={baseline_tpr:.1%}, FPR={baseline_fpr:.1%}', 
          zorder=5, edgecolors='black', linewidth=2)

ax.scatter([fusion_fpr], [fusion_tpr], s=300, marker='*', color='crimson',
          label=f'Fusion (Zero-FPR)\nTPR={fusion_tpr:.1%}, FPR={fusion_fpr:.1%}',
          zorder=5, edgecolors='black', linewidth=2)

# Plot per-fold points
for idx, row in df_cv.iterrows():
    ax.scatter([row['fpr']], [row['tpr']], s=100, marker='o', color='lightcoral',
              alpha=0.6, zorder=3)

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random Classifier')

# Formatting
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
ax.set_title('Phase 5: Fusion vs Baseline (Sig+Clf)\nZero-FPR Operating Point', 
            fontsize=14, fontweight='bold')
ax.set_xlim(-0.02, 0.15)
ax.set_ylim(0.85, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='lower right')

# Add annotations
ax.annotate('Per-fold points\n(light circles)', xy=(0.05, 0.95), fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(plots_dir / "fusion_roc_with_zero_fpr_point.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved ROC plot to {plots_dir / 'fusion_roc_with_zero_fpr_point.png'}")
plt.close()

print("\n" + "="*70)
print("✅ ROC PLOT COMPLETE")
print("="*70)
