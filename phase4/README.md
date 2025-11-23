# Phase 4: Threshold Tuning & Sensitivity Analysis

**Objective**: Quantify how confidence thresholds affect TPR/FAR trade-offs for the best-performing configuration from Phase 3.

**Configuration**: v1 + v3 (Signature + Classifier)  
**Threshold Range**: 0.05 to 0.75 in 0.05 increments  
**Dataset**: Phase 1 Part A (400 samples)

---

## Quick Start

### Run Complete Phase 4 Pipeline

```bash
python phase4/scripts/run_phase4_complete.py
```

This will:
1. Perform threshold sweep (15 thresholds)
2. Generate visualizations
3. Produce analysis summary

### Run Individual Steps

**Step 1: Threshold Sweep**
```bash
python phase4/scripts/run_threshold_sweep.py
```
Outputs: `phase4/results/threshold_sweep.csv`

**Step 2: Analysis & Visualization**
```bash
python phase4/scripts/analyze_threshold_tradeoffs.py
```
Outputs: 
- `phase4/plots/roc_curve_thresholds.png`
- `phase4/plots/f1_vs_threshold.png`
- `phase4/plots/tpr_far_vs_threshold.png`
- `phase4/results/operating_points.csv`

---

## Key Results

### Operating Points

| Mode | Threshold | TPR | FAR | F1 | Use Case |
|------|-----------|-----|-----|-----|----------|
| **High Recall** | 0.05-0.45 | 92% | 5% | 0.9355 | Security monitoring |
| **Balanced** | 0.50 | 87% | 0% | 0.9305 | **Production (recommended)** |
| **High Precision** | 0.75 | 87% | 0% | 0.9305 | Conservative systems |

### Key Findings

1. **Phase 3 baseline (t=0.50) is optimal** for production
2. **Two natural operating regions** enable easy threshold selection
3. **Robust to threshold variations** (F1 remains >0.93)
4. **Negligible latency** (<0.01 ms per sample)

---

## Deployment Recommendations

### For Production: Use t=0.50

```python
from phase2_input_detection.scripts.input_detectors import get_input_detector
from phase2_input_detection.scripts.combine_defenses import DefenseCombiner, FusionStrategy

v1 = get_input_detector("v1")
v3 = get_input_detector("v3")

combiner = DefenseCombiner(FusionStrategy.OR)
result = combiner.combine(v1.classify(text), v3.classify(text), threshold=0.50)

if result.is_attack:
    block_query()
```

**Performance**: 87% TPR, 0% FAR, F1=0.9305

### For Security Monitoring: Use t=0.10

```python
result = combiner.combine(v1.classify(text), v3.classify(text), threshold=0.10)
# 92% TPR, 5% FAR - suitable for automated alerting with human review
```

---

## File Structure

```
phase4/
├── scripts/
│   ├── run_threshold_sweep.py          # Threshold sweep evaluation
│   ├── analyze_threshold_tradeoffs.py  # Analysis and visualization
│   └── run_phase4_complete.py          # Orchestrator
├── results/
│   ├── threshold_sweep.csv             # Full sweep data (15 thresholds)
│   └── operating_points.csv            # Key operating points
├── plots/
│   ├── roc_curve_thresholds.png        # ROC-style curve
│   ├── f1_vs_threshold.png             # F1 vs threshold
│   └── tpr_far_vs_threshold.png        # TPR & FAR dual-axis
├── README.md                           # This file
└── PHASE4_THRESHOLD_TUNING_SUMMARY.md  # Detailed analysis
```

---

## Methodology

### Threshold Sweep

For each threshold t ∈ {0.05, 0.10, ..., 0.75}:
1. Classify all 400 samples using v1+v3 with threshold t
2. Compute TPR, FAR, Precision, Recall, F1
3. Calculate 95% Wilson confidence intervals
4. Measure latency

### Metrics

- **TPR**: Detection rate on all injected input (200 samples)
- **FAR**: False alarm rate on benign queries only (200 samples)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of precision and recall

### Visualizations

1. **ROC-style curve**: TPR vs FAR with Phase 3 baseline overlay
2. **F1 vs threshold**: Shows optimal threshold region
3. **TPR & FAR vs threshold**: Dual-axis showing trade-offs

---

## Latency Analysis

| Metric | Value |
|--------|-------|
| Mean Latency | 0.0065 ms |
| Max Latency | 0.0070 ms |
| vs LLM Inference | 15,000-75,000x faster |
| Overhead | <0.01% of total |

**Conclusion**: Detection latency is negligible and not a constraint.

---

## Limitations

1. **Synthetic evaluation**: Uses simulated attack text
2. **Static thresholds**: No adaptive/dynamic threshold tuning
3. **Single configuration**: Only v1+v3 (best from Phase 3)
4. **No adversarial robustness**: Doesn't test adaptive attacks
5. **Limited threshold range**: 0.05-0.75 may not cover all use cases

---

## Connection to Phase 3

- **Reuses**: Phase 3's v1+v3 configuration and fusion logic
- **Extends**: Single threshold (0.50) to full sweep (0.05-0.75)
- **Validates**: Phase 3 baseline as optimal for production
- **Enables**: Informed threshold selection for different use cases

---

## For Publication

### Key Contributions

1. **Threshold tuning framework** for security systems
2. **ROC-style analysis** demonstrating trade-offs
3. **Clear deployment guidance** (t=0.50 for production)
4. **Latency validation** (negligible overhead)

### Recommended Figures

1. ROC curve with Phase 3 baseline overlay
2. F1 vs threshold showing optimality
3. Operating points table
4. Latency comparison chart

---

## Next Steps

1. Review Phase 4 results in `PHASE4_THRESHOLD_TUNING_SUMMARY.md`
2. Validate threshold sweep data in `phase4/results/threshold_sweep.csv`
3. Examine visualizations in `phase4/plots/`
4. Prepare Phase 1-4 summary for IEEE Software submission

---

**Phase 4 Status**: ✅ **COMPLETE**  
**Recommendation**: Deploy Configuration E (v1+v3) with t=0.50  
**Performance**: 87% TPR, 0% FAR, F1=0.9305
