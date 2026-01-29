# Prompt Injection Demystified: Building an LLM Firewall for Production LLM Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CACM Practice](https://img.shields.io/badge/CACM-Practice%20Article-green.svg)](MANUSCRIPT_PREPARATION/manuscript/prompt_injection_cacm.pdf)

**A Practitioner's Playbook for Building Input-Side Prompt Injection Defenses**

Research implementation and CACM Practice-ready manuscript for systematic evaluation of deployable prompt injection firewalls.

---

## 📄 Publication

**Status**: ✅ Ready for CACM Practice submission  
**Style**: Practitioner-oriented playbook (14 pages, 2 figures, 6 tables)  
**Manuscript**: [`MANUSCRIPT_PREPARATION/manuscript/prompt_injection_cacm.pdf`](MANUSCRIPT_PREPARATION/manuscript/prompt_injection_cacm.pdf)  
**Author**: Carlos Denner dos Santos, PhD (Videns, propelled by Cofomo)

### Key Findings
- **Production Mode**: 82% TPR, 0.77% FAR (Normalizer+v3 semantic)
- **Monitoring Mode**: 87% TPR, 12% FAR (Normalizer+v1+v3 fusion)
- **Novel Attack Generalization**: 49.2% TPR across 4 attack categories
- **Latency**: 0.63-0.86 ms with GPU acceleration
- **Throughput**: ~1,200 queries/second on RTX 4070 Laptop

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/carlosdenner-videns/prompt-injection-security.git
cd prompt-injection-security

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For LLaMA-2 access (gated model)
huggingface-cli login
```

### Basic Usage

```python
from src.prompt_injection_defense import FusionDetector

# Production mode (82% TPR, 0.77% FAR)
firewall = FusionDetector(mode="production")

# Monitoring mode (87% TPR, 12% FAR)
firewall = FusionDetector(mode="monitoring")

# Check a prompt
result = firewall.detect("Ignore previous instructions")
print(f"Attack detected: {result['is_injection']}")
```

### Reproduce Results

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for complete step-by-step instructions to reproduce all manuscript results.

---

## 📁 Repository Structure

```
prompt-injection-security/
├── src/                              # Core library
│   └── prompt_injection_defense/
│       ├── detectors/                # Normalizer, v1, v3, fusion
│       ├── evaluation/               # Metrics, statistical tests
│       ├── models/                   # LLM utilities
│       └── utils/                    # Common utilities
│
├── config/                           # Deployment configurations
│   ├── config.yaml
│   ├── production.yaml
│   └── monitoring.yaml
│
├── phase1/                           # P1: Baseline (400 attacks)
│   ├── data/                         # Raw results
│   ├── results/                      # Processed metrics
│   └── scripts/                      # Experiment code
│
├── phase2_input_detection/           # P2: Detectors (v1, v2, v3)
│   ├── results/                      # Detector metrics
│   └── scripts/                      # Detection code
│
├── phase3/                           # P3: Fusion (OR/AND)
├── phase4/                           # P4: Threshold sweep
├── phase5/                           # P5: Obfuscation
├── phase6a/                          # P6a: Benign validation
├── phase6b/                          # P6b: Novel attacks
├── phase6c/                          # P6c: Adversarial
│
├── MANUSCRIPT_PREPARATION/           # Publication
│   ├── manuscript/                   # PDF, TEX, BIB, figures
│   ├── scripts/                      # Figure generation
│   └── PATENTS/                      # Patent data (Table 1)
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── REPRODUCIBILITY.md                # Step-by-step guide
```

### Manuscript-to-Repository Mapping

Each phase folder corresponds directly to a phase in the manuscript:

| Manuscript Phase | Repository Folder | Key Results |
|------------------|-------------------|-------------|
| P1: Baseline | `phase1/` | 400 attacks, 65% ASR (LLaMA-2) |
| P2: Detectors | `phase2_input_detection/` | v1: 89% TPR, v3: 82% TPR |
| P3: Fusion | `phase3/` | OR-fusion: 87% TPR |
| P4: Threshold | `phase4/` | Threshold-invariant |
| P5: Obfuscation | `phase5/` | 99% TPR with normalization |
| P6a: Benign | `phase6a/` | 0.77% FAR (production) |
| P6b: Novel | `phase6b/` | 49.2% TPR (4 categories) |
| P6c: Adversarial | `phase6c/` | 53.1% TPR |

---

## 🔬 Research Methodology

### Eight-Phase Evaluation Pipeline

#### **Phase 1: Baseline Vulnerability**
Establish attack success rates on undefended LLMs
- **Models**: LLaMA-2-7B, Falcon-7B
- **Attacks**: 400 (200 RAG-borne, 200 schema smuggling)
- **Result**: 65% ASR on LLaMA-2, 5% on Falcon-7B

#### **Phase 2: Input-Side Detection**
Develop and compare three detector variants
- **v1 (Signature)**: 89% TPR, 0% FAR (47 regex patterns)
- **v2 (Rules)**: 44% TPR, 0% FAR (deprecated)
- **v3 (Semantic)**: 82% TPR, 0% FAR (150 exemplars, θ=0.75)

#### **Phase 3: Fusion Optimization**
Combine detectors for complementary coverage
- **OR-fusion (v1+v3)**: 87% TPR, 0% FAR ✅ **Optimal**
- **Threshold-free**: No tuning required

#### **Phase 4: Threshold Robustness**
Validate performance across threshold range
- **Range**: 0.1 to 0.7
- **Result**: Threshold-invariant (87% TPR across all)

#### **Phase 5: Obfuscation Hardening**
Test normalization + learned fusion
- **Nested CV**: 99% TPR (198/200 attacks caught)
- **Lift**: +12 percentage points over Phase 3

#### **Phase 6a: Production Validation**
Measure false alarm rate on benign inputs
- **Production (v3)**: 0.77% FAR ✅ **Target met**
- **Monitoring (v1+v3)**: 12% FAR

#### **Phase 6b: Novel Attack Generalization**
Test on unseen attack patterns
- **Overall**: 49.2% TPR
- **Generalization gap**: -37.8 pp (87% → 49%)
- **Coverage gaps**: Multi-turn (30%), Context-confusion (35%)

#### **Phase 6c: Adversarial Robustness**
Evaluate against adversarially crafted attacks
- **TPR**: 53.1%
- **Most effective evasion**: Multi-step (75%)

---

## 🛠️ Deployment Guide

### Production Mode (Customer-Facing)

```python
from src.prompt_injection_defense import FusionDetector
from src.prompt_injection_defense.utils import load_config

# Load production config
config = load_config('config/production.yaml')

# Initialize firewall
firewall = FusionDetector(mode="production")

# Integrate as middleware
def process_user_query(query: str):
    result = firewall.detect(query)
    
    if result['is_injection']:
        # Block and log
        log_security_event(query, result)
        return "Your request cannot be processed."
    
    # Forward to LLM
    return llm_generate(query)
```

**Characteristics**:
- ✅ 82% TPR on known attacks
- ✅ 0.77% FAR (minimal user frustration)
- ✅ <1 ms latency
- ✅ Production-ready

### Monitoring Mode (Shadow Deployment)

```python
firewall_prod = FusionDetector(mode="production")
firewall_monitor = FusionDetector(mode="monitoring")

def process_with_monitoring(query: str):
    # Production: Real-time blocking
    result_prod = firewall_prod.detect(query)
    
    # Monitoring: Shadow analysis
    result_monitor = firewall_monitor.detect(query)
    
    # Log discrepancies for detector improvement
    if result_monitor['is_injection'] and not result_prod['is_injection']:
        log_potential_false_negative(query, result_monitor)
    
    # Act on production verdict only
    if result_prod['is_injection']:
        return "Blocked"
    return llm_generate(query)
```

**Characteristics**:
- ✅ 87% TPR on known attacks, 49% on novel
- ⚠️ 12% FAR (suitable for logging only)
- ✅ Enables continuous improvement
- ✅ Discovers emerging attack patterns

---

## 📊 Performance Benchmarks

### Latency (GPU Accelerated - RTX 4070 Laptop)

| Configuration | Median Latency | P95 Latency |
|--------------|----------------|-------------|
| Parallel (v1 \|\| v3) | 0.63 ms | 0.89 ms |
| Serial (v1 → v3) | 0.86 ms | 1.12 ms |
| Production (v3 only) | 0.52 ms | 0.71 ms |

### Memory Footprint
- **Total**: 142 MB
- **Normalizer**: <1 MB
- **v1 Signature**: <1 MB
- **v3 Semantic**: ~140 MB (embedding model)

### Throughput
- **Queries/second**: ~1,200 (GPU)
- **GPU utilization**: 18% peak
- **Suitable for**: High-throughput production systems

---

## 📈 Key Results Summary

| Metric | Production Mode | Monitoring Mode |
|--------|-----------------|-----------------|
| **TPR (Known Attacks)** | 82% | 87% |
| **TPR (Novel Attacks)** | — | 49.2% |
| **FAR (Benign Inputs)** | 0.77% | 12% |
| **Latency** | <1 ms | <1 ms |
| **Use Case** | Real-time blocking | Auditing & improvement |

---

## Patent Landscape

Analyzed **31 industry patent filings** (2022-2025) from major technology companies including:
- Cisco, HiddenLayer, Infosys, Microsoft
- Unum Group, Palo Alto Networks, IBM
- Various international filings (US, CN, KR, WO)

**Key themes identified**:
- **Sanitizing middleware**: Input normalization before LLM inference
- **Signature-based detection**: Pattern matching against known attack markers
- **Semantic screening**: Embedding-based similarity to known attacks
- **Signed prompts**: Cryptographic tagging of trusted instructions

Full patent analysis with summaries and citations: [`docs/PATENT_ANALYSIS.md`](docs/PATENT_ANALYSIS.md)

---

## 📚 Documentation

### Manuscript
- **PDF**: [`MANUSCRIPT_PREPARATION/manuscript/prompt_injection_cacm.pdf`](MANUSCRIPT_PREPARATION/manuscript/prompt_injection_cacm.pdf) (14 pages, CACM Practice style)
- **LaTeX Source**: [`MANUSCRIPT_PREPARATION/manuscript/`](MANUSCRIPT_PREPARATION/manuscript/)
- **Figures**: 2 essential visuals (baseline chart + architecture diagram); archived figures in `archived_figures/`

### Reproducibility
- **Full Guide**: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- **Phase READMEs**: Each `phase*/README.md`
- **Source Code**: [`src/prompt_injection_defense/`](src/prompt_injection_defense/)
- **Configuration**: [`config/`](config/)

---

## 🧪 Reproducing Results

### Phase 1: Baseline Assessment

```bash
# Generate knowledge base
python phase1/scripts/generate_kb.py

# Run Part A (RAG-borne attacks)
python phase1/scripts/partA_experiment.py

# Run Part B (schema smuggling)
python phase1/scripts/partB_experiment.py

# Analyze results
python phase1/scripts/analyze_results.py
```

**Expected Output**: `phase1/data/partA_results.json`, `phase1/stats/*.csv`

### Generating Figures

```bash
# Generate baseline vulnerability chart (Figure 1)
python MANUSCRIPT_PREPARATION/scripts/generate_figure_1.py

# Architecture diagram (Figure 2) is created in LaTeX using TikZ
# See manuscript source for clean vector diagram
```

**Output**: High-resolution PDFs in `MANUSCRIPT_PREPARATION/manuscript/`  
**Note**: Additional analysis figures archived in `MANUSCRIPT_PREPARATION/manuscript/archived_figures/`

---

## 🤝 Contributing

This is a research repository. Contributions welcome for:
- ✅ Additional attack patterns
- ✅ New detector implementations
- ✅ Performance optimizations
- ✅ Bug fixes

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 📝 Citation

```bibtex
@article{denner2025llmfirewall,
  title={Prompt Injection Demystified: Building an LLM Firewall for Production LLM Systems},
  author={Denner dos Santos, Carlos},
  journal={Communications of the ACM (Practice Section)},
  year={2025},
  note={Submitted}
}
```

---

## 📄 License

MIT License - See [`LICENSE`](LICENSE) for details

---

## 📞 Contact

**Author**: Carlos Denner dos Santos, PhD  
**Affiliation**: Videns, propelled by Cofomo  
**Email**: carlos.denner@videns.ai  
**GitHub**: [@carlosdenner-videns](https://github.com/carlosdenner-videns)

---

## 🙏 Acknowledgments

- Open-source LLM community for tools and benchmarks
- HuggingFace for model access
- Reviewers and colleagues for feedback

---

**Status**: ✅ Publication-ready | 🚀 Production-deployable | 📊 Fully reproducible

*Last updated: November 23, 2025*
