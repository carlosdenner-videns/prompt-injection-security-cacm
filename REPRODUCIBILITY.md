# Reproducibility Guide

Complete step-by-step instructions for reproducing all results from the manuscript.

---

## Environment Setup

### Hardware Requirements
- **GPU**: NVIDIA RTX 4070 Laptop (15GB VRAM) or equivalent
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models and results
- **OS**: Windows 10/11, Linux, or macOS

### Software Requirements
- **Python**: 3.9 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **Git**: For cloning repository

### Installation

```bash
# 1. Clone repository
git clone https://github.com/carlosdenner-videns/prompt-injection-security.git
cd prompt-injection-security

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
.\venv\Scripts\Activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Authenticate with HuggingFace (for LLaMA-2)
huggingface-cli login
# You'll need to:
# - Create account at https://huggingface.co
# - Request access to meta-llama/Llama-2-7b-chat-hf
# - Create access token at https://huggingface.co/settings/tokens
```

### Verify Installation

```bash
python verify_setup.py
```

Expected output:
```
✓ Python 3.9+ detected
✓ CUDA 11.8+ available
✓ PyTorch with GPU support
✓ HuggingFace authentication OK
✓ All dependencies installed
```

---

## Data Preparation

All experimental data is included in the repository under `phase*/data/`. No additional data download required.

### Data Locations

```
phase1/data/
├── partA_kb.jsonl              # 440 documents (400 benign + 40 malicious)
├── partA_results.json          # Phase 1 Part A results
├── partB_results.json          # Phase 1 Part B results
└── phase1_output_annotated.json # Annotated with defense labels

phase2_input_detection/data/
├── train_attacks.json          # Training set (320 samples)
├── test_attacks.json           # Test set (80 samples)
└── benign_queries.json         # 260 benign samples

phase6a/data/
└── obfuscated_benign.json      # 260 obfuscated benign samples

phase6b/data/
└── novel_attacks.json          # 65 novel attack samples

phase6c/data/
└── adversarial_attacks.json    # 30 adversarial samples
```

---

## Reproducing Key Results

### Table 1: Baseline Vulnerability (Phase 1)

**Expected**: LLaMA-2: 65% ASR, Falcon-7B: 5% ASR

```bash
# Run complete Phase 1 pipeline
python phase1/scripts/run_phase1.py

# Or run parts individually:
python phase1/scripts/generate_kb.py
python phase1/scripts/partA_experiment.py
python phase1/scripts/partB_experiment.py
python phase1/scripts/analyze_results.py
```

**Runtime**: ~2-3 hours for both models

**Outputs**:
- `phase1/data/partA_results.json`
- `phase1/data/partB_results.json`
- `phase1/stats/partA_analysis.csv`
- `phase1/plots/partA_heatmap.png`

**Verify**:
```bash
python -c "
import json
with open('phase1/data/partA_results.json') as f:
    data = json.load(f)
llama_attacks = [r for r in data if r['model']=='llama2-7b' and r['is_injected']]
asr = sum(r['injection_success'] for r in llama_attacks) / len(llama_attacks)
print(f'LLaMA-2 ASR: {asr:.1%}')
"
```

Expected: `LLaMA-2 ASR: 65.0%`

---

### Table 2: Detector Performance (Phase 2)

**Expected**: v1: 89% TPR, v3: 82% TPR

```bash
python phase2_input_detection/scripts/run_detection.py
```

**Runtime**: ~30 minutes

**Outputs**:
- `phase2_input_detection/results/input_detection_metrics.csv`

**Verify**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('phase2_input_detection/results/input_detection_metrics.csv')
print(df[['detector', 'tpr', 'far']])
"
```

Expected:
```
  detector   tpr   far
0       v1  0.89  0.00
1       v3  0.82  0.00
```

---

### Table 3: Fusion Results (Phase 3)

**Expected**: OR-fusion (v1+v3): 87% TPR, 0% FAR

```bash
python phase3/scripts/run_fusion.py
```

**Runtime**: ~20 minutes

**Outputs**:
- `phase3/results/fusion_evaluation_results.csv`

**Verify**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('phase3/results/fusion_evaluation_results.csv')
or_fusion = df[df['strategy']=='OR']
print(f\"OR-fusion: TPR={or_fusion['tpr'].values[0]:.2%}, FAR={or_fusion['far'].values[0]:.2%}\")
"
```

Expected: `OR-fusion: TPR=87.00%, FAR=0.00%`

---

### Figure 1: Baseline Vulnerability

**Expected**: Bar chart comparing LLaMA-2 vs Falcon-7B

```bash
python MANUSCRIPT_PREPARATION/scripts/generate_all_figures.py --figures 1
```

**Output**: `MANUSCRIPT_PREPARATION/manuscript/fig1_baseline_vulnerability.pdf`

---

### Figure 4: Detector Performance

**Expected**: Bar chart showing v1: 89%, v3: 82%, OR: 87%

```bash
python MANUSCRIPT_PREPARATION/scripts/generate_all_figures.py --figures 4
```

**Output**: `MANUSCRIPT_PREPARATION/manuscript/fig4_detector_performance.pdf`

---

### Table 6: Production vs Monitoring Modes (Phase 6a)

**Expected**: Production: 0.77% FAR, Monitoring: 12% FAR

```bash
python phase6a/scripts/run_benign_validation.py
```

**Runtime**: ~15 minutes

**Outputs**:
- `phase6a/results/obfuscated_benign_metrics.csv`

**Verify**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('phase6a/results/obfuscated_benign_metrics.csv')
print(df[['configuration', 'far']])
"
```

Expected:
```
   configuration    far
0  normalizer+v3  0.0077
1  normalizer+v1+v3  0.1200
```

---

### Figure 11: Novel Attack Detection (Phase 6b)

**Expected**: 4 categories, overall 49.2% TPR

```bash
python phase6b/scripts/run_novel_attacks.py
python MANUSCRIPT_PREPARATION/scripts/generate_all_figures.py --figures 11
```

**Outputs**:
- `phase6b/results/novel_attacks_metrics.csv`
- `MANUSCRIPT_PREPARATION/manuscript/fig11_novel_attack_tpr.pdf`

**Verify**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('phase6b/results/novel_attacks_metrics.csv')
overall = df[df['category']=='overall']['tpr'].values[0]
print(f'Novel attack TPR: {overall:.1%}')
"
```

Expected: `Novel attack TPR: 49.2%`

---

### Performance Benchmarks (Table 7)

**Expected**: Parallel: 0.63ms, Serial: 0.86ms

```bash
python phase7/scripts/measure_latency.py
```

**Runtime**: ~10 minutes

**Outputs**:
- `phase7/results/latency_measurements.csv`

**Verify**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('phase7/results/latency_measurements.csv')
print(df[['configuration', 'median_ms', 'p95_ms']])
"
```

Expected:
```
   configuration  median_ms  p95_ms
0       parallel       0.63    0.89
1         serial       0.86    1.12
```

---

## Generating All Figures

### All 10 Manuscript Figures

```bash
python MANUSCRIPT_PREPARATION/scripts/generate_all_figures.py
```

**Runtime**: ~5 minutes

**Outputs**: All figures in `MANUSCRIPT_PREPARATION/manuscript/fig*.pdf`

### Individual Figures

```bash
# Figure 16 (System Architecture)
python MANUSCRIPT_PREPARATION/scripts/generate_figure_16_final_optimized.py

# Specific figure by number
python MANUSCRIPT_PREPARATION/scripts/generate_all_figures.py --figures 1,4,11
```

---

## Recompiling Manuscript

### LaTeX Compilation

```bash
cd MANUSCRIPT_PREPARATION/manuscript

# Full compilation (includes bibliography)
pdflatex prompt_injection_cacm.tex
bibtex prompt_injection_cacm
pdflatex prompt_injection_cacm.tex
pdflatex prompt_injection_cacm.tex
```

**Output**: `prompt_injection_cacm.pdf` (21 pages, ~2.8 MB)

### Verify Compilation

Expected output should include:
```
Output written on prompt_injection_cacm.pdf (21 pages, 2847227 bytes).
Transcript written on prompt_injection_cacm.log.
```

Check for errors:
```bash
grep -i error prompt_injection_cacm.log
# Should return nothing
```

---

## Troubleshooting

### GPU Out of Memory

If you encounter OOM errors during Phase 1:

```python
# Edit phase1/scripts/partA_experiment.py
# Enable 8-bit quantization (line ~25):
runner = ModelRunner(model_name, load_in_8bit=True)
```

### Model Download Slow

```bash
# Use HuggingFace mirror
export HF_ENDPOINT="https://hf-mirror.com"  # Linux/Mac
$env:HF_ENDPOINT = "https://hf-mirror.com"  # Windows PowerShell
```

### Import Errors

```bash
# Verify virtual environment is activated
which python  # Linux/Mac
where python  # Windows

# Should show path inside venv/

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Checkpoint Recovery

If experiments are interrupted:

```bash
# Experiments auto-resume from last checkpoint
python phase1/scripts/partA_experiment.py  # Will resume automatically

# To start fresh, delete checkpoint:
rm phase1/data/partA_checkpoint.json
```

---

## Expected Runtimes

| Phase | Task | Runtime (GPU) | Runtime (CPU) |
|-------|------|---------------|---------------|
| 1 | Baseline (both models) | 2-3 hours | 12-15 hours |
| 2 | Detector training | 30 mins | 2 hours |
| 3 | Fusion evaluation | 20 mins | 1 hour |
| 4 | Threshold sweep | 1 hour | 4 hours |
| 5 | Nested CV | 2 hours | 8 hours |
| 6a | Benign validation | 15 mins | 45 mins |
| 6b | Novel attacks | 30 mins | 2 hours |
| 6c | Adversarial attacks | 15 mins | 45 mins |
| 7-8 | Performance | 10 mins | 30 mins |
| **Total** | **~7-8 hours** | **~30-35 hours** |

All experiments include checkpointing and can be interrupted/resumed.

---

## Validation Checklist

After reproducing, verify these key metrics match manuscript:

- [ ] Phase 1: LLaMA-2 ASR = 65% ± 5%
- [ ] Phase 1: Falcon-7B ASR = 5% ± 5%
- [ ] Phase 2: v1 TPR = 89% ± 3%
- [ ] Phase 2: v3 TPR = 82% ± 3%
- [ ] Phase 3: OR-fusion TPR = 87% ± 3%
- [ ] Phase 4: Threshold-invariant (87% across all)
- [ ] Phase 5: Learned fusion TPR = 99% ± 1%
- [ ] Phase 6a: Production FAR ≤ 1%
- [ ] Phase 6a: Monitoring FAR ≈ 12% ± 2%
- [ ] Phase 6b: Novel TPR = 49% ± 5%
- [ ] Phase 7: Latency < 1 ms (GPU)
- [ ] All 10 figures generated without errors
- [ ] Manuscript compiles without errors

---

## Data Availability

To comply with responsible disclosure:
- ✅ All benign queries included
- ✅ Novel and adversarial attacks included
- ⚠️ Some Phase 1 attacks redacted (exfiltration endpoints)

For access to complete unredacted dataset, contact: carlos.denner@videns.ai

---

## Questions or Issues?

1. Check this guide's troubleshooting section
2. Review phase-specific README: `phase*/README.md`
3. Open GitHub issue: https://github.com/carlosdenner-videns/prompt-injection-security/issues
4. Contact author: carlos.denner@videns.ai

---

**Last updated**: November 23, 2025  
**Reproducibility commitment**: All results in manuscript should reproduce within stated error margins
