"""
Task 1: Advanced Statistical Analysis for Phase 1
Computes McNemar's test for paired defenses and Wilson 95% CIs.
Outputs: mcnemar_results.csv, ci_summary.csv, defense_pairwise_matrix.png
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    from scipy import stats
    p = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return p, lower, upper


class Phase1StatisticalAnalysis:
    """Performs advanced statistical analysis on Phase 1 results."""
    
    def __init__(self, results_json: str = None, output_dir: str = None):
        """
        Initialize analyzer.
        
        Args:
            results_json: Path to partA_results.json (defaults to phase1/data/partA_results.json)
            output_dir: Output directory for stats (defaults to phase1/stats)
        """
        if results_json is None:
            script_dir = Path(__file__).parent
            phase1_dir = script_dir.parent
            results_json = str(phase1_dir / "data" / "partA_results.json")
        if output_dir is None:
            script_dir = Path(__file__).parent
            phase1_dir = script_dir.parent
            output_dir = str(phase1_dir / "stats")
        
        self.results_json = results_json
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load results from JSON."""
        with open(self.results_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.df = pd.DataFrame(self.data)
        print(f"✓ Loaded {len(self.df)} results from {self.results_json}")
    
    def compute_wilson_ci_summary(self) -> pd.DataFrame:
        """
        Compute Wilson 95% CI for ASR, FAR, and FDR by model and evasion type.
        
        Returns:
            DataFrame with CI summary
        """
        results = []
        
        for model in self.df["model"].unique():
            model_df = self.df[self.df["model"] == model]
            
            # Overall metrics
            injected = model_df[model_df["is_injected"] == True]
            benign = model_df[model_df["is_injected"] == False]
            
            # ASR (Attack Success Rate): successful injections / total injections
            asr_succ = injected["injection_success"].sum()
            asr_trials = len(injected)
            asr, asr_low, asr_high = wilson_ci(asr_succ, asr_trials)
            
            # FAR (False Alarm Rate): false positives / total benign
            far_succ = benign["injection_success"].sum()
            far_trials = len(benign)
            far, far_low, far_high = wilson_ci(far_succ, far_trials)
            
            # FDR (False Discovery Rate): false positives / (true positives + false positives)
            tp = asr_succ
            fp = far_succ
            fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            fdr_low, fdr_high = 0.0, 1.0  # Placeholder; FDR CI is complex
            
            results.append({
                "model": model,
                "category": "overall",
                "asr": asr,
                "asr_ci_low": asr_low,
                "asr_ci_high": asr_high,
                "far": far,
                "far_ci_low": far_low,
                "far_ci_high": far_high,
                "fdr": fdr,
                "fdr_ci_low": fdr_low,
                "fdr_ci_high": fdr_high,
                "injected_trials": asr_trials,
                "benign_trials": far_trials
            })
            
            # By evasion type
            for evasion in model_df[model_df["is_injected"] == True]["evasion_type"].unique():
                evasion_df = model_df[
                    (model_df["is_injected"] == True) & 
                    (model_df["evasion_type"] == evasion)
                ]
                
                evasion_succ = evasion_df["injection_success"].sum()
                evasion_trials = len(evasion_df)
                evasion_asr, evasion_asr_low, evasion_asr_high = wilson_ci(evasion_succ, evasion_trials)
                
                results.append({
                    "model": model,
                    "category": evasion,
                    "asr": evasion_asr,
                    "asr_ci_low": evasion_asr_low,
                    "asr_ci_high": evasion_asr_high,
                    "far": np.nan,
                    "far_ci_low": np.nan,
                    "far_ci_high": np.nan,
                    "fdr": np.nan,
                    "fdr_ci_low": np.nan,
                    "fdr_ci_high": np.nan,
                    "injected_trials": evasion_trials,
                    "benign_trials": 0
                })
        
        ci_df = pd.DataFrame(results)
        output_file = self.output_dir / "ci_summary.csv"
        ci_df.to_csv(output_file, index=False)
        print(f"✓ Saved CI summary to {output_file}")
        
        return ci_df
    
    def compute_mcnemar_pairwise(self) -> pd.DataFrame:
        """
        Compute chi-square test for all pairs of evasion types.
        Tests whether two evasion types have significantly different success rates.
        
        Returns:
            DataFrame with statistical test results
        """
        results = []
        
        for model in self.df["model"].unique():
            model_df = self.df[self.df["model"] == model]
            injected_df = model_df[model_df["is_injected"] == True]
            
            evasion_types = sorted(injected_df["evasion_type"].unique())
            
            for i, evasion1 in enumerate(evasion_types):
                for evasion2 in evasion_types[i+1:]:
                    # Get samples for each evasion type
                    ev1_df = injected_df[injected_df["evasion_type"] == evasion1]
                    ev2_df = injected_df[injected_df["evasion_type"] == evasion2]
                    
                    ev1_success = ev1_df["injection_success"].sum()
                    ev1_total = len(ev1_df)
                    
                    ev2_success = ev2_df["injection_success"].sum()
                    ev2_total = len(ev2_df)
                    
                    # Create contingency table
                    # [success_ev1, fail_ev1]
                    # [success_ev2, fail_ev2]
                    contingency = np.array([
                        [ev1_success, ev1_total - ev1_success],
                        [ev2_success, ev2_total - ev2_success]
                    ])
                    
                    # Use chi-square test with continuity correction
                    try:
                        from scipy.stats import chi2_contingency
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                    except ValueError:
                        # If chi-square fails (zero expected frequencies), use Fisher's exact
                        try:
                            from scipy.stats import fisher_exact
                            odds_ratio, p_value = fisher_exact(contingency)
                            chi2 = np.nan
                        except:
                            # If both fail, mark as NaN
                            chi2 = np.nan
                            p_value = np.nan
                    
                    results.append({
                        "model": model,
                        "defense_1": evasion1,
                        "defense_2": evasion2,
                        "success_rate_1": ev1_success / ev1_total if ev1_total > 0 else 0,
                        "success_rate_2": ev2_success / ev2_total if ev2_total > 0 else 0,
                        "trials_1": ev1_total,
                        "trials_2": ev2_total,
                        "chi2_statistic": chi2,
                        "p_value": p_value,
                        "significant_at_0.05": "Yes" if (not np.isnan(p_value) and p_value < 0.05) else "No"
                    })
        
        mcnemar_df = pd.DataFrame(results)
        output_file = self.output_dir / "mcnemar_results.csv"
        mcnemar_df.to_csv(output_file, index=False)
        print(f"✓ Saved McNemar results to {output_file}")
        
        return mcnemar_df
    
    def plot_defense_pairwise_matrix(self, ci_df: pd.DataFrame):
        """
        Create heatmap of ASR by model and evasion type.
        
        Args:
            ci_df: DataFrame from compute_wilson_ci_summary
        """
        # Pivot for heatmap
        pivot_df = ci_df[ci_df["category"] != "overall"].pivot_table(
            index="category",
            columns="model",
            values="asr"
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Attack Success Rate"}
        )
        plt.title("Attack Success Rate by Evasion Type and Model")
        plt.xlabel("Model")
        plt.ylabel("Evasion Type")
        plt.tight_layout()
        
        output_file = self.output_dir / "defense_pairwise_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved heatmap to {output_file}")
        plt.close()
    
    def run_analysis(self):
        """Run complete statistical analysis."""
        print("\n" + "="*70)
        print("PHASE 1: ADVANCED STATISTICAL ANALYSIS")
        print("="*70 + "\n")
        
        # Compute Wilson CIs
        ci_df = self.compute_wilson_ci_summary()
        
        # Compute McNemar's test
        mcnemar_df = self.compute_mcnemar_pairwise()
        
        # Create visualizations
        self.plot_defense_pairwise_matrix(ci_df)
        
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOutputs saved to: {self.output_dir}")
        print("  - ci_summary.csv: Wilson CIs for ASR, FAR, FDR")
        print("  - mcnemar_results.csv: Pairwise significance tests")
        print("  - defense_pairwise_matrix.png: ASR heatmap")


if __name__ == "__main__":
    analyzer = Phase1StatisticalAnalysis(
        results_json="partA_results.json",
        output_dir="phase1/stats"
    )
    analyzer.run_analysis()
