"""
Task 2: Phase 1 Folder Refactor
Moves all Phase 1 scripts, data, and plots into dedicated folders.
Adjusts internal path references in scripts.
"""

import shutil
from pathlib import Path
import re


class Phase1Refactor:
    """Refactors Phase 1 folder structure."""
    
    def __init__(self, root_dir: str = "."):
        """
        Initialize refactor.
        
        Args:
            root_dir: Root project directory
        """
        self.root = Path(root_dir)
        self.phase1_dir = self.root / "phase1"
        self.phase1_dir.mkdir(exist_ok=True)
        
        # Define subdirectories
        self.data_dir = self.phase1_dir / "data"
        self.scripts_dir = self.phase1_dir / "scripts"
        self.stats_dir = self.phase1_dir / "stats"
        self.plots_dir = self.phase1_dir / "plots"
        
        for d in [self.data_dir, self.scripts_dir, self.stats_dir, self.plots_dir]:
            d.mkdir(exist_ok=True)
    
    def move_data_files(self):
        """Move data files to phase1/data/."""
        print("\n📁 Moving data files...")
        
        data_patterns = [
            "partA_results.json",
            "partB_results.json",
            "partA_checkpoint.json",
            "partA_kb.jsonl",
            "phase1_output_fixed_results.csv",
            "phase1_output_fixed_full.json",
            "phase1_output_mcnemar.csv"
        ]
        
        for pattern in data_patterns:
            src = self.root / pattern
            if src.exists():
                dst = self.data_dir / pattern
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved {pattern}")
    
    def move_script_files(self):
        """Move Phase 1 scripts to phase1/scripts/."""
        print("\n🐍 Moving scripts...")
        
        script_patterns = [
            "run_phase1.py",
            "generate_kb.py",
            "partA_experiment.py",
            "partB_experiment.py",
            "analyze_results.py",
            "analyze_phase1.py",
            "phase1_statistical_analysis.py",
            "phase1_label_defenses.py"
        ]
        
        for pattern in script_patterns:
            src = self.root / pattern
            if src.exists():
                dst = self.scripts_dir / pattern
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved {pattern}")
    
    def move_plot_files(self):
        """Move plot files to phase1/plots/."""
        print("\n📊 Moving plots...")
        
        plot_patterns = [
            "partA_heatmap.png",
            "partB_heatmap.png",
            "phase1_comparison.png",
            "phase1_comparison_plot.png",
            "phase1_family_heatmap.png"
        ]
        
        for pattern in plot_patterns:
            src = self.root / pattern
            if src.exists():
                dst = self.plots_dir / pattern
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved {pattern}")
    
    def move_analysis_files(self):
        """Move analysis files to phase1/stats/."""
        print("\n📈 Moving analysis files...")
        
        analysis_patterns = [
            "partA_analysis.csv",
            "partB_analysis.csv",
            "phase1_summary.txt",
            "mcnemar_results.csv",
            "ci_summary.csv"
        ]
        
        for pattern in analysis_patterns:
            src = self.root / pattern
            if src.exists():
                dst = self.stats_dir / pattern
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved {pattern}")
    
    def update_path_references(self):
        """Update path references in remaining scripts."""
        print("\n🔧 Updating path references...")
        
        # Scripts that reference Phase 1 files
        scripts_to_update = [
            self.root / "run_phase1.py"
        ]
        
        # Path mappings
        path_replacements = [
            (r'partA_results\.json', r'"phase1/data/partA_results.json"'),
            (r'partB_results\.json', r'"phase1/data/partB_results.json"'),
            (r'partA_kb\.jsonl', r'"phase1/data/partA_kb.jsonl"'),
            (r'partA_analysis\.csv', r'"phase1/stats/partA_analysis.csv"'),
            (r'partB_analysis\.csv', r'"phase1/stats/partB_analysis.csv"'),
            (r'phase1_summary\.txt', r'"phase1/stats/phase1_summary.txt"'),
            (r'partA_heatmap\.png', r'"phase1/plots/partA_heatmap.png"'),
            (r'partB_heatmap\.png', r'"phase1/plots/partB_heatmap.png"'),
            (r'phase1_comparison\.png', r'"phase1/plots/phase1_comparison.png"'),
        ]
        
        for script in scripts_to_update:
            if script.exists():
                with open(script, "r", encoding="utf-8") as f:
                    content = f.read()
                
                original_content = content
                
                for old_pattern, new_pattern in path_replacements:
                    content = re.sub(old_pattern, new_pattern, content)
                
                if content != original_content:
                    with open(script, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"  ✓ Updated {script.name}")
    
    def create_readme(self):
        """Create README for phase1 folder."""
        print("\n📝 Creating README...")
        
        readme_content = """# Phase 1: Baseline Vulnerability Assessment

This folder contains all Phase 1 experiments, data, and analysis for prompt injection attacks.

## Folder Structure

```
phase1/
├── data/              # Raw and processed data
│   ├── partA_results.json
│   ├── partB_results.json
│   ├── phase1_output_annotated.json
│   └── *.csv (individual defense results)
├── scripts/           # Phase 1 scripts
│   ├── run_phase1.py
│   ├── generate_kb.py
│   ├── partA_experiment.py
│   ├── partB_experiment.py
│   ├── analyze_results.py
│   ├── phase1_statistical_analysis.py
│   └── phase1_label_defenses.py
├── stats/             # Statistical analysis outputs
│   ├── partA_analysis.csv
│   ├── partB_analysis.csv
│   ├── mcnemar_results.csv
│   ├── ci_summary.csv
│   └── phase1_summary.txt
├── plots/             # Visualizations
│   ├── partA_heatmap.png
│   ├── partB_heatmap.png
│   ├── phase1_comparison.png
│   └── defense_pairwise_matrix.png
└── README.md          # This file
```

## Running Phase 1

### 1. Generate Knowledge Base and Run Experiments
```bash
python phase1/scripts/run_phase1.py
```

### 2. Run Statistical Analysis
```bash
python phase1_statistical_analysis.py
```

### 3. Label Defense Verdicts
```bash
python phase1_label_defenses.py
```

## Key Outputs

- **partA_results.json**: RAG-borne injection experiment results
- **partB_results.json**: Schema smuggling experiment results
- **phase1_output_annotated.json**: Results with per-sample defense labels
- **mcnemar_results.csv**: Pairwise statistical significance tests
- **ci_summary.csv**: Wilson 95% confidence intervals for ASR, FAR, FDR
- **defense_pairwise_matrix.png**: Heatmap of attack success rates

## Metrics

- **ASR (Attack Success Rate)**: Percentage of injections that succeeded
- **FAR (False Alarm Rate)**: Percentage of benign queries flagged as attacks
- **FDR (False Discovery Rate)**: Proportion of positive predictions that are false

## Next Steps

1. Review statistical analysis results in `stats/`
2. Examine visualizations in `plots/`
3. Analyze annotated data in `data/phase1_output_annotated.json`
4. Proceed to Phase 2 (defense development)
"""
        
        readme_file = self.phase1_dir / "README.md"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print(f"  ✓ Created {readme_file}")
    
    def run_refactor(self):
        """Run complete refactoring."""
        print("\n" + "="*70)
        print("PHASE 1: FOLDER REFACTORING")
        print("="*70)
        
        self.move_data_files()
        self.move_script_files()
        self.move_plot_files()
        self.move_analysis_files()
        self.update_path_references()
        self.create_readme()
        
        print("\n" + "="*70)
        print("REFACTORING COMPLETE")
        print("="*70)
        print(f"\nPhase 1 structure: {self.phase1_dir}")
        print("\nSubdirectories:")
        print(f"  - data/    {self.data_dir}")
        print(f"  - scripts/ {self.scripts_dir}")
        print(f"  - stats/   {self.stats_dir}")
        print(f"  - plots/   {self.plots_dir}")


if __name__ == "__main__":
    refactor = Phase1Refactor()
    refactor.run_refactor()
