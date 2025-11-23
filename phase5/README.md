# Phase 5: Obfuscation-Robust Detection with Learned Fusion

**Objective**: Harden input-side detection against obfuscation (homoglyphs, zero-width chars, mixed scripts) and add a shallow, auditable learned fusion to catch remaining evasions while maintaining FAR ≈ 0%.

**Configuration**: Normalizer + Learned Fusion (trained on Phase 1 Part A)  
**Dataset**: Phase 1 Part A (400 samples: 200 injected, 200 benign)

---

## Quick Start

### Run Complete Phase 5 Pipeline

```bash
# Train learned fusion with FPR-constrained thresholding
python phase5/scripts/train_learned_fusion.py

# Evaluate baselines vs Phase 5 variants
python phase5/scripts/evaluate_phase5.py

# Generate visualizations
python phase5/scripts/generate_plots_phase5.py
```

---

## Components

### 1. Obfuscation Normalizer (`phase5/scripts/normalizer.py`)

**Defends against**:
- Zero-width characters (ZWJ, ZWNJ, etc.)
- Homoglyphs (Cyrillic/Greek lookalikes)
- Mixed-script obfuscation

**Process**:
1. NFKC Unicode normalization
2. Strip zero-width characters
3. Apply homoglyph mapping (if safe)
4. Emit normalization report

**Safety**:
- Mixed-script detection prevents false positives on real non-Latin text
- Only maps when <15% non-ASCII and >70% ASCII letters

**Output**:
```python
{
    'normalized': str,              # Cleaned text
    'zwj_count': int,               # Zero-width chars removed
    'mapped_confusables': int,      # Homoglyphs mapped
    'mixed_script_ratio': float,    # Non-ASCII ratio
    'symbol_density': float,        # Symbol ratio
    'entropy': float,               # Shannon entropy
    'avg_word_len': float,
    'max_digit_run': int,
    'mapping_applied': bool,
    'notes': [str],                 # Transformation log
}
```

### 2. Feature Extractor (`phase5/scripts/features.py`)

**Combines**:
- Detector outputs (v1/v2/v3 is_attack flags + confidence)
- Normalizer metrics (zwj_count, mapped_confusables, etc.)
- Rule hits (one-hot encoding of matched categories)
- Text statistics (length, uppercase ratio, digit ratio, etc.)

**Total features**: 25 features

### 3. Learned Fusion (`phase5/scripts/train_learned_fusion.py`)

**Training**:
- Stratified 5-fold cross-validation
- Logistic regression (numpy implementation, no sklearn dependency)
- Per-fold threshold calibration to enforce FPR ≤ 1%

**Output**:
- `learned_fusion_cv_metrics.csv` - Per-fold metrics (TPR, FPR, F1)
- `learned_fusion_feature_importance.csv` - Feature coefficients
- `learned_fusion_thresholds.csv` - Per-fold thresholds

### 4. Evaluation (`phase5/scripts/evaluate_phase5.py`)

**Configurations**:
- Baselines: v1, v3, v1+v3 (Phase 3)
- Normalizer ablations: Norm+v1, Norm+v3, Norm+v1+v3
- Final: Norm+LearnedFusion

**Metrics**:
- TPR, FAR, Precision, Recall, F1
- 95% Wilson confidence intervals
- Attack family breakdown

---

## Key Design Decisions

### 1. No sklearn Dependency

Implemented logistic regression in numpy to keep dependencies minimal. Falls back gracefully if sklearn unavailable.

### 2. FPR-Constrained Thresholding

Each CV fold independently calibrates threshold to achieve FPR ≤ 1%. This ensures:
- Predictable false alarm rate
- No surprises in production
- Auditable threshold selection

### 3. Mixed-Script Safeguard

Homoglyph mapping only applied when:
- Mixed-script ratio ≤ 15% (mostly ASCII)
- Not predominantly non-Latin (>40% non-ASCII)

This prevents false positives on legitimate non-Latin text (e.g., Russian, Chinese).

### 4. Explainability

All model decisions are traceable:
- Feature coefficients exported
- Per-fold thresholds saved
- Normalization decisions logged

---

## Expected Results

### Primary Goal
**Achieve TPR ≥ 90% with FPR ≤ 1.0%** on Phase 1 Part A using Norm+LearnedFusion.

### Secondary Goals
- Demonstrate clear lift on obfuscated families (homoglyph, ZWJ, multilingual)
- No regression vs Phase 3 v1+v3 baseline
- Maintain FAR ≈ 0% for production deployment

---

## File Structure

```
phase5/
├── scripts/
│   ├── normalizer.py                    # Obfuscation normalizer
│   ├── features.py                      # Feature extraction
│   ├── train_learned_fusion.py          # CV training with FPR constraint
│   ├── evaluate_phase5.py               # Evaluation harness
│   └── generate_plots_phase5.py         # Visualization
├── results/
│   ├── learned_fusion_cv_metrics.csv    # Per-fold metrics
│   ├── learned_fusion_feature_importance.csv
│   ├── learned_fusion_thresholds.csv
│   └── phase5_comparison_metrics.csv    # Configuration comparison
├── plots/
│   ├── feature_importance.png
│   ├── cv_metrics.png
│   └── comparison.png
├── README.md                            # This file
└── PHASE5_OBFUSCATION_ROBUST_SUMMARY.md # Detailed analysis
```

---

## Reproducibility

All components are seeded for reproducibility:
- CV split: `np.random.seed(42)`
- Logistic regression: Fixed hyperparameters (C=1.0, max_iter=200)
- Feature extraction: Deterministic homoglyph map

To reproduce:
```bash
python phase5/scripts/train_learned_fusion.py
python phase5/scripts/evaluate_phase5.py
python phase5/scripts/generate_plots_phase5.py
```

---

## Deployment Guidance

### Production: Norm+LearnedFusion at FPR ≤ 1%

```python
from phase5.scripts.normalizer import normalize_text
from phase5.scripts.features import extract_features
# Load trained model coefficients from learned_fusion_cv_metrics.csv
# Apply threshold from learned_fusion_thresholds.csv (mean across folds)

norm = normalize_text(text)
features = extract_features(text, norm, v1_result, v2_result, v3_result)
# Predict with learned fusion model
```

### Monitoring Mode: Lower threshold for ≈92-94% TPR

Use per-fold thresholds from `learned_fusion_thresholds.csv` to explore trade-offs.

---

## Limitations

1. **Synthetic evaluation**: Uses simulated attack text, not real RAG contexts
2. **Single dataset**: Evaluated on Phase 1 Part A only
3. **No adversarial robustness**: Doesn't test adaptive attacks targeting normalizer
4. **Limited homoglyph map**: Covers common confusables, not exhaustive

---

## Next Steps

1. Review Phase 5 results in `PHASE5_OBFUSCATION_ROBUST_SUMMARY.md`
2. Validate metrics in `phase5/results/`
3. Examine visualizations in `phase5/plots/`
4. Prepare Phase 1-5 summary for publication

---

**Phase 5 Status**: ✅ **READY FOR EVALUATION**  
**Recommendation**: Deploy Norm+LearnedFusion for production  
**Expected Performance**: ≥90% TPR at ≤1% FAR
