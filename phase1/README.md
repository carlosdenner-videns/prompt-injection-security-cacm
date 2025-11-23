# Phase 1: Baseline Vulnerability Assessment

This folder contains all Phase 1 experiments, data, and analysis for prompt injection attacks.

## Folder Structure

```
phase1/
├── data/              # Raw and processed data
│   ├── partA_results.json
│   ├── partB_results.json
│   ├── partA_kb.jsonl
│   ├── phase1_output_annotated.json
│   ├── tool_registry.yaml
│   ├── schema_smuggling_variations.json
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

### Quick Start: Complete Pipeline
```bash
# From root directory
python phase1/scripts/run_phase1.py

# Or from phase1 directory
cd phase1
python scripts/run_phase1.py

# Or from phase1/scripts directory
cd phase1/scripts
python run_phase1.py
```

This orchestrates all steps:
1. Generates knowledge base
2. Runs Part A (RAG-borne injection)
3. Runs Part B (schema smuggling)
4. Analyzes results and generates visualizations

### Individual Steps

#### Step 1: Generate Knowledge Base
```bash
python phase1/scripts/generate_kb.py
```
**Output**: `phase1/data/partA_kb.jsonl` (440 documents)

#### Step 2: Run Part A Experiment
```bash
python phase1/scripts/partA_experiment.py
```
**Output**: `phase1/data/partA_results.json`

#### Step 3: Run Part B Experiment
```bash
python phase1/scripts/partB_experiment.py
```
**Output**: `phase1/data/partB_results.json`

#### Step 4: Analyze Results
```bash
python phase1/scripts/analyze_results.py
```
**Outputs**: 
- `phase1/stats/partA_analysis.csv`
- `phase1/stats/partB_analysis.csv`
- `phase1/plots/partA_heatmap.png`
- `phase1/plots/partB_heatmap.png`
- `phase1/plots/phase1_comparison.png`

#### Step 5: Run Statistical Analysis
```bash
python phase1/scripts/phase1_statistical_analysis.py
```
**Outputs**:
- `phase1/stats/ci_summary.csv` - Wilson 95% CIs
- `phase1/stats/mcnemar_results.csv` - Pairwise significance tests
- `phase1/stats/defense_pairwise_matrix.png` - ASR heatmap

#### Step 6: Label Defense Verdicts
```bash
python phase1/scripts/phase1_label_defenses.py
```
**Output**: `phase1/data/phase1_output_annotated.json`

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

## Path Resolution

All scripts in `phase1/scripts/` use automatic path detection:

```python
script_dir = Path(__file__).parent        # phase1/scripts
phase1_dir = script_dir.parent            # phase1
root_dir = script_dir.parent.parent       # root
```

This means:
- ✅ Scripts work from **any directory**
- ✅ No hardcoded paths needed
- ✅ Data automatically saved to correct subdirectories
- ✅ Config files automatically found in phase1/data/

**You can run scripts from**:
- Root: `python phase1/scripts/run_phase1.py`
- phase1: `cd phase1 && python scripts/run_phase1.py`
- phase1/scripts: `cd phase1/scripts && python run_phase1.py`

## Next Steps

1. Run complete pipeline: `python phase1/scripts/run_phase1.py`
2. Review statistical analysis results in `stats/`
3. Examine visualizations in `plots/`
4. Analyze annotated data in `data/phase1_output_annotated.json`
5. Proceed to Phase 2 (defense development)
