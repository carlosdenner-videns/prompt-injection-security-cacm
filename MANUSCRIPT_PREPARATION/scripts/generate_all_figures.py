#!/usr/bin/env python3
"""
Master Figure Generation Script
Generates all 20 publication-ready figures for manuscript

Usage:
    python generate_all_figures.py

Output:
    All figures saved to: GENERATED_FIGURES/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'GENERATED_FIGURES'
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = SCRIPT_DIR  # Data files are in same directory as script

print("=" * 70)
print("MANUSCRIPT FIGURE GENERATION")
print("=" * 70)

# ============================================================================
# FIGURE 1: Attack Success Rate Comparison
# ============================================================================
print("\n[1/20] Generating Figure 1: Attack Success Rate Comparison...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_1_data.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(data.columns) - 1)
    width = 0.35
    
    llama_data = data.iloc[0, 1:].values.astype(float)
    falcon_data = data.iloc[1, 1:].values.astype(float)
    
    ax.bar(x - width/2, llama_data, width, label='LLaMA-2-7B', color='#1f77b4')
    ax.bar(x + width/2, falcon_data, width, label='Falcon-7B', color='#ff7f0e')
    
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Attack Success Rate Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(data.columns[1:])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1_asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 1: {e}")

# ============================================================================
# FIGURE 2: Evasion Technique Heatmap
# ============================================================================
print("[2/20] Generating Figure 2: Evasion Technique Heatmap...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_2_data.csv')
    pivot_data = data.pivot(index='Evasion Type', columns='Model', values='ASR')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, 
                cbar_kws={'label': 'ASR (%)'}, vmin=0, vmax=100)
    ax.set_title('Figure 2: Evasion Technique Effectiveness Heatmap')
    ax.set_xlabel('Model')
    ax.set_ylabel('Evasion Type')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_2_evasion_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 2: {e}")

# ============================================================================
# FIGURE 3: Schema Smuggling by Tool
# ============================================================================
print("[3/20] Generating Figure 3: Schema Smuggling by Tool...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_3_data.csv')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    x = np.arange(len(data.columns) - 1)
    width = 0.35
    
    llama_data = data.iloc[0, 1:].values.astype(float)
    falcon_data = data.iloc[1, 1:].values.astype(float)
    
    ax.bar(x - width/2, llama_data, width, label='LLaMA-2-7B', color='#1f77b4')
    ax.bar(x + width/2, falcon_data, width, label='Falcon-7B', color='#ff7f0e')
    
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Figure 3: Schema Smuggling Vulnerability by Tool')
    ax.set_xticks(x)
    ax.set_xticklabels(data.columns[1:])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_3_schema_smuggling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 3: {e}")

# ============================================================================
# FIGURE 4: Detector Performance
# ============================================================================
print("[4/20] Generating Figure 4: Detector Performance...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_4_data.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    detectors = data['Detector'].unique()
    x = np.arange(len(detectors))
    width = 0.35
    
    tpr_data = data[data['Metric'] == 'TPR']['Value'].values
    far_data = data[data['Metric'] == 'FAR']['Value'].values
    
    ax.bar(x - width/2, tpr_data, width, label='TPR', color='#2ca02c')
    ax.bar(x + width/2, far_data, width, label='FAR', color='#d62728')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Figure 4: Detector Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_4_detector_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 4: {e}")

# ============================================================================
# FIGURE 5: Fusion Strategy Comparison (Scatter)
# ============================================================================
print("[5/20] Generating Figure 5: Fusion Strategy Comparison...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_5_data.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4' if config != 'v1+v3 (OR)' else '#d62728' for config in data['Configuration']]
    ax.scatter(data['FAR'], data['TPR'], s=200, c=colors, alpha=0.6, edgecolors='black')
    
    for idx, config in enumerate(data['Configuration']):
        ax.annotate(config, (data['FAR'].iloc[idx], data['TPR'].iloc[idx]), 
                   fontsize=8, ha='right')
    
    ax.set_xlabel('FAR (%)')
    ax.set_ylabel('TPR (%)')
    ax.set_title('Figure 5: Fusion Strategy Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_5_fusion_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 5: {e}")

# ============================================================================
# FIGURE 6: Detector Complementarity
# ============================================================================
print("[6/20] Generating Figure 6: Detector Complementarity...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_6_data.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    categories = data['Category']
    counts = data['Count']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Attacks')
    ax.set_title('Figure 6: Detector Complementarity Analysis')
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(counts):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_6_complementarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 6: {e}")

# ============================================================================
# FIGURE 7: Threshold Robustness
# ============================================================================
print("[7/20] Generating Figure 7: Threshold Robustness...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_7_data.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(data['Threshold'], data['TPR'], marker='o', label='TPR', color='#2ca02c', linewidth=2)
    ax.plot(data['Threshold'], data['FAR'], marker='s', label='FAR', color='#d62728', linewidth=2)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Threshold Robustness (Threshold-Invariant Performance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_7_threshold_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 7: {e}")

# ============================================================================
# FIGURE 8: Learned Fusion (Nested CV)
# ============================================================================
print("[8/20] Generating Figure 8: Learned Fusion (Nested CV)...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_8_data.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    folds = data['Fold']
    means = data['Mean_TPR']
    mins = data['Min_TPR']
    maxs = data['Max_TPR']
    
    ax.bar(folds, means, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.errorbar(folds, means, yerr=[means - mins, maxs - means], 
               fmt='none', color='black', capsize=5, capthick=2)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('TPR (%)')
    ax.set_title('Figure 8: Learned Fusion Performance (Nested CV)')
    ax.set_ylim([90, 102])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_8_learned_fusion_cv.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 8 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 8: {e}")

# ============================================================================
# FIGURE 9: Lift Over Baseline
# ============================================================================
print("[9/20] Generating Figure 9: Lift Over Baseline...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_9_data.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    
    configs = data['Configuration']
    tpr = data['TPR']
    colors = ['#1f77b4', '#2ca02c']
    
    bars = ax.bar(configs, tpr, color=colors, alpha=0.7, edgecolor='black')
    
    for i, (bar, val) in enumerate(zip(bars, tpr)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('TPR (%)')
    ax.set_title('Figure 9: Lift Over Baseline (Phase 3 → Phase 5)')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_9_lift_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 9 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 9: {e}")

# ============================================================================
# FIGURE 10: FAR by Configuration & Obfuscation
# ============================================================================
print("[10/20] Generating Figure 10: FAR by Configuration & Obfuscation...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_10_data.csv')
    pivot_data = data.pivot(index='Obfuscation Type', columns='Configuration', values='FAR')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'FAR (%)'}, vmin=0, vmax=25)
    ax.set_title('Figure 10: FAR by Configuration and Obfuscation Type')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Obfuscation Type')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_10_far_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 10 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 10: {e}")

# ============================================================================
# FIGURE 11: TPR by Attack Type
# ============================================================================
print("[11/20] Generating Figure 11: TPR by Attack Type...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_11_data.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    
    attack_types = data['Attack Type']
    tpr = data['TPR']
    colors = ['#2ca02c' if t > 50 else '#ff7f0e' if t > 25 else '#d62728' for t in tpr]
    
    ax.barh(attack_types, tpr, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('TPR (%)')
    ax.set_title('TPR by Attack Type (Novel Attacks)')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(tpr):
        ax.text(v + 1, i, f'{v}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_11_tpr_attack_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 11 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 11: {e}")

# ============================================================================
# FIGURE 12: Coverage Gaps
# ============================================================================
print("[12/20] Generating Figure 12: Coverage Gaps...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_12_data.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    
    attack_types = data['Attack Type']
    detected = data['Detected']
    missed = data['Missed']
    
    x = np.arange(len(attack_types))
    width = 0.6
    
    ax.bar(x, detected, width, label='Detected', color='#2ca02c', alpha=0.7)
    ax.bar(x, missed, width, bottom=detected, label='Missed', color='#d62728', alpha=0.7)
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Figure 12: Coverage Gaps by Attack Type')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_12_coverage_gaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 12 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 12: {e}")

# ============================================================================
# FIGURE 13: Adversarial Techniques
# ============================================================================
print("[13/20] Generating Figure 13: Adversarial Techniques...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_13_data.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    
    techniques = data['Technique']
    evasion = data['Evasion_Rate']
    colors = ['#d62728' if e > 70 else '#ff7f0e' if e > 60 else '#2ca02c' for e in evasion]
    
    ax.barh(techniques, evasion, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Evasion Rate (%)')
    ax.set_title('Figure 13: Adversarial Technique Effectiveness')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(evasion):
        ax.text(v + 1, i, f'{v}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_13_adversarial_techniques.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 13 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 13: {e}")

# ============================================================================
# FIGURE 14: Performance Progression
# ============================================================================
print("[14/20] Generating Figure 14: Performance Progression...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_14_data.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phases = data['Phase']
    tpr = data['TPR']
    
    ax.plot(phases, tpr, marker='o', markersize=8, linewidth=2.5, color='#1f77b4')
    ax.fill_between(range(len(phases)), tpr, alpha=0.3, color='#1f77b4')
    
    ax.set_xlabel('Phase')
    ax.set_ylabel('TPR (%)')
    ax.set_title('Figure 14: Cross-Phase Performance Progression')
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases)
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(tpr):
        ax.text(i, v + 2, f'{v}%', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_14_performance_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 14 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 14: {e}")

# ============================================================================
# FIGURE 15: Generalization Gap
# ============================================================================
print("[15/20] Generating Figure 15: Generalization Gap...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_15_data.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    attack_types = data['Attack Type']
    tpr = data['TPR']
    colors = ['#2ca02c', '#ff7f0e', '#ff7f0e']
    
    bars = ax.bar(attack_types, tpr, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, tpr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('TPR (%)')
    ax.set_title('Figure 15: Generalization Gap Analysis')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_15_generalization_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 15 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 15: {e}")

# ============================================================================
# FIGURE 16: System Architecture (Manual - reference ARCHITECTURE_VISUALIZATION.md)
# ============================================================================
print("[16/20] Figure 16: System Architecture (Manual creation recommended)")
print("       → See ARCHITECTURE_VISUALIZATION.md for ASCII diagram")
print("       → Use draw.io or similar tool for publication-ready version")

# ============================================================================
# FIGURE 17: Confusion Matrices
# ============================================================================
print("[17/20] Generating Figure 17: Confusion Matrices...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_17_data.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    for idx, detector in enumerate(['v1', 'v2', 'v3']):
        detector_data = data[data['Detector'] == detector]
        cm = np.array([[detector_data[detector_data['Type'] == 'TN']['Value'].values[0],
                       detector_data[detector_data['Type'] == 'FP']['Value'].values[0]],
                      [detector_data[detector_data['Type'] == 'FN']['Value'].values[0],
                       detector_data[detector_data['Type'] == 'TP']['Value'].values[0]]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=False, xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        axes[idx].set_title(f'{detector}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    fig.suptitle('Figure 17: Confusion Matrices (Phase 2 Detectors)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_17_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 17 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 17: {e}")

# ============================================================================
# FIGURE 18: Feature Importance
# ============================================================================
print("[18/20] Generating Figure 18: Feature Importance...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_18_data.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    features = data['Feature']
    importance = data['Importance']
    colors = ['#d62728' if imp > 0.3 else '#ff7f0e' if imp > 0.1 else '#2ca02c' for imp in importance]
    
    ax.barh(features, importance, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Figure 18: Feature Importance (Learned Fusion Model)')
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_18_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 18 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 18: {e}")

# ============================================================================
# FIGURE 19: Deployment Comparison (Table)
# ============================================================================
print("[19/20] Generating Figure 19: Deployment Comparison...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_19_data.csv')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in data.iterrows():
        table_data.append([row['Metric'], row['Normalizer+v3'], row['Normalizer+v1+v3']])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Normalizer+v3\n(Production)', 'Normalizer+v1+v3\n(Monitoring)'],
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#1f77b4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Figure 19: Deployment Configuration Comparison', fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / 'figure_19_deployment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 19 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 19: {e}")

# ============================================================================
# FIGURE 20: Execution Timeline
# ============================================================================
print("[20/20] Generating Figure 20: Execution Timeline...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_20_data.csv')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    phases = data['Phase']
    durations = data['Duration']
    colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
    
    ax.barh(phases, durations, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Duration (hours)')
    ax.set_title('Figure 20: Execution Timeline by Phase')
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(durations):
        ax.text(v + 0.1, i, f'{v:.2f}h', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_20_execution_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 20 generated successfully")
except Exception as e:
    print(f"✗ Error generating Figure 20: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE GENERATION COMPLETE")
print("=" * 70)
print(f"\n✓ All figures saved to: {OUTPUT_DIR}/")
print(f"✓ Total figures generated: 19/20 (Figure 16 requires manual creation)")
print("\nNext steps:")
print("1. Review generated figures in GENERATED_FIGURES/")
print("2. Insert figures into manuscript at referenced locations")
print("3. Add figure captions from FIGURES_AND_VISUALIZATIONS_GUIDE.md")
print("4. Create Figure 16 (System Architecture) manually using draw.io")
print("\n" + "=" * 70)
