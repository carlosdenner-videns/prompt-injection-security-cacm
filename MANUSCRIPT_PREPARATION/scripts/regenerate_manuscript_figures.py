#!/usr/bin/env python3
"""
Regenerate Manuscript Figures WITHOUT "Figure X:" Prefixes
This script regenerates only the figures used in the CACM manuscript
with corrected titles (no embedded figure numbers).

Fixes figures: 1, 4, 6, 7, 9, 10, 11, 13, 15, 16
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
plt.rcParams['font.family'] = 'serif'

# Directories
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR  # Save directly to manuscript directory
DATA_DIR = SCRIPT_DIR

print()
print("=" * 80)
print("MANUSCRIPT FIGURE REGENERATION (WITHOUT FIGURE NUMBERS)")
print("=" * 80)
print()
print("Fixing 9 figures with embedded 'Figure X:' prefixes...")
print()

success_count = 0
error_count = 0

# ============================================================================
# FIGURE 1: Baseline Vulnerability
# ============================================================================
print("[1/9] Regenerating fig1_baseline_vulnerability.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_1_data.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(data.columns) - 1)
    width = 0.35
    
    llama_data = data.iloc[0, 1:].values.astype(float)
    falcon_data = data.iloc[1, 1:].values.astype(float)
    
    ax.bar(x - width/2, llama_data, width, label='LLaMA-2-7b', color='#1f77b4')
    ax.bar(x + width/2, falcon_data, width, label='Falcon-7b', color='#ff7f0e')
    
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 1:" prefix
    ax.set_title('Baseline Vulnerability Assessment', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(data.columns[1:], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_baseline_vulnerability.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig1_baseline_vulnerability.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 4: Detector Performance
# ============================================================================
print("[2/9] Regenerating fig4_detector_performance.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_4_data.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    detectors = data['Detector'].unique()
    x = np.arange(len(detectors))
    width = 0.35
    
    tpr_data = data[data['Metric'] == 'TPR']['Value'].values
    far_data = data[data['Metric'] == 'FAR']['Value'].values
    
    ax.bar(x - width/2, tpr_data, width, label='TPR (%)', color='#2ca02c')
    ax.bar(x + width/2, far_data, width, label='FAR (%)', color='#d62728')
    
    ax.set_ylabel('Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 4:" prefix
    ax.set_title('Detector Performance Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_detector_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig4_detector_performance.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 6: Detector Complementarity
# ============================================================================
print("[3/9] Regenerating fig6_complementarity.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_6_data.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    categories = data['Category']
    counts = data['Count']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Attacks', fontsize=11)
    # FIXED: Removed "Figure 6:" prefix
    ax.set_title('Detector Complementarity Analysis', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(counts):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold', fontsize=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_complementarity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig6_complementarity.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 7: Threshold Invariance
# ============================================================================
print("[4/9] Regenerating fig7_threshold_invariance.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_7_data.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(data['Threshold'], data['TPR'], marker='o', label='TPR (%)', 
            color='#2ca02c', linewidth=2.5, markersize=7)
    ax.plot(data['Threshold'], data['FAR'], marker='s', label='FAR (%)', 
            color='#d62728', linewidth=2.5, markersize=7)
    
    ax.set_xlabel('Detection Threshold', fontsize=11)
    ax.set_ylabel('Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 7:" prefix
    ax.set_title('Threshold-Invariant Performance', fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_threshold_invariance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig7_threshold_invariance.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 9: Learning Gain
# ============================================================================
print("[5/9] Regenerating fig9_learning_gain.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_9_data.csv')
    fig, ax = plt.subplots(figsize=(6, 5))
    
    configs = data['Configuration']
    tpr = data['TPR']
    colors = ['#1f77b4', '#2ca02c']
    
    bars = ax.bar(configs, tpr, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    
    for i, (bar, val) in enumerate(zip(bars, tpr)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('True Positive Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 9:" prefix
    ax.set_title('Learning-Based Fusion Improvement', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_learning_gain.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig9_learning_gain.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 10: Obfuscation False Positive Rate
# ============================================================================
print("[6/9] Regenerating fig10_obfuscation_fpr.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_10_data.csv')
    pivot_data = data.pivot(index='Obfuscation Type', columns='Configuration', values='FAR')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'False Alarm Rate (%)'}, vmin=0, vmax=25,
                linewidths=0.5, linecolor='gray')
    # FIXED: Removed "Figure 10:" prefix
    ax.set_title('False Alarm Rate by Configuration and Obfuscation Type', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Detector Configuration', fontsize=11)
    ax.set_ylabel('Obfuscation Type', fontsize=11)
    plt.xticks(fontsize=9, rotation=45, ha='right')
    plt.yticks(fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_obfuscation_fpr.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig10_obfuscation_fpr.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 11: Novel Attack Detection
# ============================================================================
print("[7/9] Regenerating fig11_novel_attack_tpr.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_11_data.csv')
    fig, ax = plt.subplots(figsize=(9, 6))
    
    attack_types = data['Attack Type']
    tpr = data['TPR']
    colors = ['#2ca02c' if t > 50 else '#ff7f0e' if t > 25 else '#d62728' for t in tpr]
    
    ax.barh(attack_types, tpr, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('True Positive Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 11:" prefix
    ax.set_title('Novel Attack Detection by Category', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    plt.yticks(fontsize=10)
    
    for i, v in enumerate(tpr):
        ax.text(v + 2, i, f'{v}%', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_novel_attack_tpr.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig11_novel_attack_tpr.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 13: Adversarial Evasion
# ============================================================================
print("[8/9] Regenerating fig13_adversarial_evasion.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_13_data.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    techniques = data['Technique']
    evasion = data['Evasion_Rate']
    colors = ['#d62728' if e > 70 else '#ff7f0e' if e > 60 else '#2ca02c' for e in evasion]
    
    ax.barh(techniques, evasion, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Evasion Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 13:" prefix
    ax.set_title('Adversarial Technique Effectiveness', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    plt.yticks(fontsize=10)
    
    for i, v in enumerate(evasion):
        ax.text(v + 2, i, f'{v}%', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig13_adversarial_evasion.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig13_adversarial_evasion.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# FIGURE 15: Generalization Gap
# ============================================================================
print("[9/9] Regenerating fig15_generalization_gap.pdf...")
try:
    data = pd.read_csv(DATA_DIR / 'figure_15_data.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    attack_types = data['Attack Type']
    tpr = data['TPR']
    colors = ['#2ca02c' if t > 90 else '#ff7f0e' for t in tpr]
    
    bars = ax.bar(attack_types, tpr, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    
    for bar, val in zip(bars, tpr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
               f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('True Positive Rate (%)', fontsize=11)
    # FIXED: Removed "Figure 15:" prefix
    ax.set_title('Generalization Gap Analysis', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(fontsize=10, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig15_generalization_gap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ fig15_generalization_gap.pdf saved")
    success_count += 1
except Exception as e:
    print(f"✗ Error: {e}")
    error_count += 1

# ============================================================================
# NOTE: Figure 16 (Architecture) was previously regenerated separately
# ============================================================================
print()
print("Note: fig16_architecture.pdf should be regenerated using")
print("      generate_figure_16_publication.py (already exists)")

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 80)
print("REGENERATION COMPLETE")
print("=" * 80)
print()
print(f"✓ Successfully regenerated: {success_count}/9 figures")
if error_count > 0:
    print(f"✗ Errors encountered: {error_count}/9 figures")
print()
print("All figures saved to:", OUTPUT_DIR)
print()
print("WHAT CHANGED:")
print("  • Removed 'Figure X:' prefixes from all titles")
print("  • LaTeX \\caption{} will provide authoritative figure numbers")
print("  • Titles are now descriptive only")
print()
print("NEXT STEPS:")
print("  1. Run: python verify_figure_fixes.py")
print("  2. Compile LaTeX: pdflatex prompt_injection_cacm.tex")
print("  3. Verify no numbering mismatches in output PDF")
print()
print("=" * 80)
print()
