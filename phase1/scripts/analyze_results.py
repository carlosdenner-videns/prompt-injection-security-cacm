"""
Results Analysis and Visualization for Phase 1 Experiments
Computes ASR, Wilson CI, and generates heatmaps and summary reports.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.
    
    Args:
        successes: Number of successful attacks
        trials: Total number of trials
        confidence: Confidence level (default 0.95)
    
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    
    p = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return p, lower, upper


class Phase1Analyzer:
    """Analyzes results from Part A and Part B experiments."""
    
    def __init__(self, partA_file: str = None, partB_file: str = None):
        """
        Initialize analyzer.
        
        Args:
            partA_file: Path to Part A results JSON
            partB_file: Path to Part B results JSON
        """
        self.partA_data = None
        self.partB_data = None
        
        if partA_file and Path(partA_file).exists():
            with open(partA_file, "r", encoding="utf-8") as f:
                self.partA_data = json.load(f)
            print(f"Loaded Part A: {len(self.partA_data)} results")
        
        if partB_file and Path(partB_file).exists():
            with open(partB_file, "r", encoding="utf-8") as f:
                self.partB_data = json.load(f)
            print(f"Loaded Part B: {len(self.partB_data)} results")
    
    def analyze_partA(self) -> pd.DataFrame:
        """
        Analyze Part A (RAG-borne injection) results.
        
        Returns:
            DataFrame with analysis by model and evasion type
        """
        if not self.partA_data:
            print("No Part A data loaded")
            return None
        
        df = pd.DataFrame(self.partA_data)
        
        # Overall ASR by model
        print("\n" + "="*60)
        print("PART A: RAG-BORNE INJECTION ANALYSIS")
        print("="*60 + "\n")
        
        results_summary = []
        
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            injected_df = model_df[model_df["is_injected"] == True]
            
            total = len(injected_df)
            successes = injected_df["injection_success"].sum()
            asr, ci_low, ci_high = wilson_ci(successes, total)
            
            print(f"{model}:")
            print(f"  Overall ASR: {asr:.2%} ({successes}/{total})")
            print(f"  95% CI: [{ci_low:.2%}, {ci_high:.2%}]")
            
            results_summary.append({
                "model": model,
                "category": "overall",
                "successes": successes,
                "trials": total,
                "asr": asr,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
            
            # Breakdown by evasion type
            print(f"\n  By evasion type:")
            for ev_type in injected_df["evasion_type"].unique():
                ev_df = injected_df[injected_df["evasion_type"] == ev_type]
                ev_total = len(ev_df)
                ev_successes = ev_df["injection_success"].sum()
                ev_asr, ev_ci_low, ev_ci_high = wilson_ci(ev_successes, ev_total)
                
                print(f"    {ev_type}: {ev_asr:.2%} ({ev_successes}/{ev_total}) "
                      f"CI: [{ev_ci_low:.2%}, {ev_ci_high:.2%}]")
                
                results_summary.append({
                    "model": model,
                    "category": ev_type,
                    "successes": ev_successes,
                    "trials": ev_total,
                    "asr": ev_asr,
                    "ci_low": ev_ci_low,
                    "ci_high": ev_ci_high
                })
            
            # Performance metrics
            avg_time = model_df["generation_time_sec"].mean()
            avg_tokens_sec = model_df["tokens_per_sec"].mean()
            print(f"\n  Performance:")
            print(f"    Avg generation time: {avg_time:.2f}s")
            print(f"    Avg tokens/sec: {avg_tokens_sec:.1f}")
            print()
        
        return pd.DataFrame(results_summary)
    
    def analyze_partB(self) -> pd.DataFrame:
        """
        Analyze Part B (schema smuggling) results.
        
        Returns:
            DataFrame with analysis by model and mechanism
        """
        if not self.partB_data:
            print("No Part B data loaded")
            return None
        
        df = pd.DataFrame(self.partB_data)
        
        print("\n" + "="*60)
        print("PART B: SCHEMA SMUGGLING ANALYSIS")
        print("="*60 + "\n")
        
        results_summary = []
        
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            attack_df = model_df[model_df["is_attack"] == True]
            
            total = len(attack_df)
            successes = attack_df["attack_success"].sum()
            asr, ci_low, ci_high = wilson_ci(successes, total)
            
            print(f"{model}:")
            print(f"  Overall ASR: {asr:.2%} ({successes}/{total})")
            print(f"  95% CI: [{ci_low:.2%}, {ci_high:.2%}]")
            
            results_summary.append({
                "model": model,
                "category": "overall",
                "successes": successes,
                "trials": total,
                "asr": asr,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
            
            # Breakdown by mechanism
            print(f"\n  By mechanism:")
            for mechanism in sorted(attack_df["mechanism"].unique()):
                mech_df = attack_df[attack_df["mechanism"] == mechanism]
                mech_total = len(mech_df)
                mech_successes = mech_df["attack_success"].sum()
                mech_asr, mech_ci_low, mech_ci_high = wilson_ci(mech_successes, mech_total)
                
                print(f"    {mechanism}: {mech_asr:.2%} ({mech_successes}/{mech_total}) "
                      f"CI: [{mech_ci_low:.2%}, {mech_ci_high:.2%}]")
                
                results_summary.append({
                    "model": model,
                    "category": mechanism,
                    "successes": mech_successes,
                    "trials": mech_total,
                    "asr": mech_asr,
                    "ci_low": mech_ci_low,
                    "ci_high": mech_ci_high
                })
            
            # Performance metrics
            avg_time = model_df["generation_time_sec"].mean()
            avg_tokens_sec = model_df["tokens_per_sec"].mean()
            print(f"\n  Performance:")
            print(f"    Avg generation time: {avg_time:.2f}s")
            print(f"    Avg tokens/sec: {avg_tokens_sec:.1f}")
            print()
        
        return pd.DataFrame(results_summary)
    
    def plot_partA_heatmap(self, summary_df: pd.DataFrame, output_file: str = None):
        """Generate heatmap for Part A results."""
        if output_file is None:
            script_dir = Path(__file__).parent
            phase1_dir = script_dir.parent
            output_file = str(phase1_dir / "plots" / "partA_heatmap.png")
        
        if summary_df is None or summary_df.empty:
            print("No data to plot")
            return
        
        # Filter out overall row for cleaner visualization
        plot_df = summary_df[summary_df["category"] != "overall"].copy()
        
        if plot_df.empty:
            print("Not enough data for heatmap")
            return
        
        # Pivot for heatmap
        pivot_asr = plot_df.pivot(index="category", columns="model", values="asr")
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_asr * 100,  # Convert to percentage
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Attack Success Rate (%)"},
            vmin=0,
            vmax=100
        )
        plt.title("Part A: RAG-Borne Injection ASR by Evasion Type", fontsize=14, fontweight="bold")
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Evasion Type", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {output_file}")
        plt.close()
    
    def plot_partB_heatmap(self, summary_df: pd.DataFrame, output_file: str = None):
        """Generate heatmap for Part B results."""
        if output_file is None:
            script_dir = Path(__file__).parent
            phase1_dir = script_dir.parent
            output_file = str(phase1_dir / "plots" / "partB_heatmap.png")
        
        if summary_df is None or summary_df.empty:
            print("No data to plot")
            return
        
        # Filter out overall row
        plot_df = summary_df[summary_df["category"] != "overall"].copy()
        
        if plot_df.empty:
            print("Not enough data for heatmap")
            return
        
        # Pivot for heatmap
        pivot_asr = plot_df.pivot(index="category", columns="model", values="asr")
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_asr * 100,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Attack Success Rate (%)"},
            vmin=0,
            vmax=100
        )
        plt.title("Part B: Schema Smuggling ASR by Mechanism", fontsize=14, fontweight="bold")
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Attack Mechanism", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {output_file}")
        plt.close()
    
    def plot_comparison(self, partA_summary: pd.DataFrame, partB_summary: pd.DataFrame, output_file: str = None):
        """Generate comparison plot of overall ASR across parts."""
        if output_file is None:
            script_dir = Path(__file__).parent
            phase1_dir = script_dir.parent
            output_file = str(phase1_dir / "plots" / "phase1_comparison.png")
        
        if partA_summary is None or partB_summary is None:
            print("Need both Part A and Part B data for comparison")
            return
        
        # Extract overall rows
        partA_overall = partA_summary[partA_summary["category"] == "overall"].copy()
        partB_overall = partB_summary[partB_summary["category"] == "overall"].copy()
        
        partA_overall["experiment"] = "Part A: RAG-Borne"
        partB_overall["experiment"] = "Part B: Schema Smuggling"
        
        combined = pd.concat([partA_overall, partB_overall])
        
        # Create bar plot with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = combined["model"].unique()
        x = np.arange(len(models))
        width = 0.35
        
        for i, experiment in enumerate(["Part A: RAG-Borne", "Part B: Schema Smuggling"]):
            exp_data = combined[combined["experiment"] == experiment]
            asrs = []
            ci_lows = []
            ci_highs = []
            
            for model in models:
                model_data = exp_data[exp_data["model"] == model]
                if not model_data.empty:
                    asrs.append(model_data["asr"].values[0] * 100)
                    ci_lows.append(model_data["ci_low"].values[0] * 100)
                    ci_highs.append(model_data["ci_high"].values[0] * 100)
                else:
                    asrs.append(0)
                    ci_lows.append(0)
                    ci_highs.append(0)
            
            # Calculate error bars
            yerr_low = [asr - ci_low for asr, ci_low in zip(asrs, ci_lows)]
            yerr_high = [ci_high - asr for asr, ci_high in zip(asrs, ci_highs)]
            
            offset = width * (i - 0.5)
            ax.bar(x + offset, asrs, width, label=experiment, yerr=[yerr_low, yerr_high], capsize=5)
        
        ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
        ax.set_title("Phase 1: Overall Attack Success Rates with 95% CI", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {output_file}")
        plt.close()
    
    def generate_summary_report(self, output_file: str = None):
        """Generate a text summary report."""
        if output_file is None:
            script_dir = Path(__file__).parent
            phase1_dir = script_dir.parent
            output_file = str(phase1_dir / "stats" / "phase1_summary.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("PHASE 1: BASELINE VULNERABILITY ASSESSMENT SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            if self.partA_data:
                f.write("PART A: RAG-BORNE PROMPT INJECTION\n")
                f.write("-" * 70 + "\n")
                df = pd.DataFrame(self.partA_data)
                for model in df["model"].unique():
                    model_df = df[df["model"] == model]
                    injected = model_df[model_df["is_injected"] == True]
                    success = injected["injection_success"].sum()
                    total = len(injected)
                    asr, ci_low, ci_high = wilson_ci(success, total)
                    
                    f.write(f"\n{model}:\n")
                    f.write(f"  Total injected queries: {total}\n")
                    f.write(f"  Successful attacks: {success}\n")
                    f.write(f"  ASR: {asr:.2%} (95% CI: [{ci_low:.2%}, {ci_high:.2%}])\n")
                
                f.write("\n")
            
            if self.partB_data:
                f.write("\nPART B: TOOL-CALL SCHEMA SMUGGLING\n")
                f.write("-" * 70 + "\n")
                df = pd.DataFrame(self.partB_data)
                for model in df["model"].unique():
                    model_df = df[df["model"] == model]
                    attacks = model_df[model_df["is_attack"] == True]
                    success = attacks["attack_success"].sum()
                    total = len(attacks)
                    asr, ci_low, ci_high = wilson_ci(success, total)
                    
                    f.write(f"\n{model}:\n")
                    f.write(f"  Total attack attempts: {total}\n")
                    f.write(f"  Successful attacks: {success}\n")
                    f.write(f"  ASR: {asr:.2%} (95% CI: [{ci_low:.2%}, {ci_high:.2%}])\n")
                
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("End of Summary\n")
            f.write("="*70 + "\n")
        
        print(f"Summary report saved to {output_file}")


def main():
    """Run full analysis pipeline."""
    analyzer = Phase1Analyzer(
        partA_file="partA_results.json",
        partB_file="partB_results.json"
    )
    
    # Analyze Part A
    if analyzer.partA_data:
        partA_summary = analyzer.analyze_partA()
        if partA_summary is not None:
            partA_summary.to_csv("partA_analysis.csv", index=False)
            print(f"Saved Part A analysis to partA_analysis.csv")
            analyzer.plot_partA_heatmap(partA_summary)
    else:
        partA_summary = None
    
    # Analyze Part B
    if analyzer.partB_data:
        partB_summary = analyzer.analyze_partB()
        if partB_summary is not None:
            partB_summary.to_csv("partB_analysis.csv", index=False)
            print(f"Saved Part B analysis to partB_analysis.csv")
            analyzer.plot_partB_heatmap(partB_summary)
    else:
        partB_summary = None
    
    # Generate comparison
    if partA_summary is not None and partB_summary is not None:
        analyzer.plot_comparison(partA_summary, partB_summary)
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
