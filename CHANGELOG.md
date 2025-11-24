# Changelog

## [2.0.0] - 2025-11-23 - CACM Practice Transformation

### Major Revision: Research Paper → Practitioner Playbook

Transformed manuscript from academic research paper to CACM Practice article style, focusing on practitioner utility and deployment guidance.

### Changes

#### Title
- **Old**: "Building an LLM Firewall: A Multi-Phase Defense Against Prompt Injection"
- **New**: "Prompt Injection Demystified: Building an LLM Firewall for Production LLM Systems"

#### Manuscript Style
- **Format**: CACM Practice article (practitioner-oriented)
- **Length**: 14 pages (down from 20+)
- **Tone**: Direct, actionable guidance for engineers
- **Structure**: Organized around practitioner questions

#### Visual Footprint Reduced
- **Before**: 10 histogram-style figures
- **After**: 2 essential figures
  - Figure 1: Baseline vulnerability chart (threat baseline)
  - Figure 2: Clean TikZ pipeline diagram (architecture)
- **Archived**: 11 analysis figures moved to `MANUSCRIPT_PREPARATION/manuscript/archived_figures/`
  - All quantitative data preserved in tables and inline text
  - Figures available for presentations/supplements

#### Content Refinements

**Abstract**: Rewritten to foreground practitioner outcomes and deployment scenarios

**Introduction**: 
- Added "Who should read this" and "What you will learn" sections
- Emphasized historical security patterns (WAF, spam filters, SQL injection defenses)

**Section Renaming**: All sections reframed with practitioner questions:
- Section 2: "Why Prompt Injection Matters Operationally"
- Section 3: "The LLM Firewall Architecture and Design Rationale"
- Section 4: "What the Firewall Delivers in Practice"
- Section 5: "How to Deploy the LLM Firewall in Your Stack"
- Section 6: "Lessons for Teams Running LLMs Today"
- Section 7: "Known Gaps and What to Watch Next"

**Deployment Guidance**: 
- Clearer 6-step deployment sequence (Intercept → Normalize → Detect → Fuse → Act)
- Explicit Production vs Monitoring mode configuration guidance
- Added actionable takeaways to every major section

**Kelly Checklist Polish**: Applied CACM editorial standards for:
- **Actionable**: Concrete "do this" guidance in every section
- **Brief**: ~10% word reduction in core sections
- **Clear**: Simplified jargon, added inline explanations
- **Historically aware**: Connected to established security patterns
- **Universal**: Emphasized model-agnostic, vendor-neutral design
- **Well-written**: Engaging practitioner tone

**Figure Simplification**:
- Replaced slide-style architecture PDF with clean TikZ block diagram
- No embedded metrics, legends, or bullet lists in diagrams
- All explanatory text moved to captions and surrounding prose

#### Technical Content
- ✅ **All numerical results preserved**: Every TPR, FAR, latency, throughput metric intact
- ✅ **All tables preserved**: 6 tables with complete data
- ✅ **All claims preserved**: No technical content altered
- ✅ **Section structure preserved**: 8-phase evaluation framework maintained

#### Compilation
- **Status**: Clean compilation (exit code 0)
- **Warnings**: All overfull hbox warnings fixed
- **Output**: 14 pages, 772 KB PDF

### Repository Updates

#### README.md
- Updated title and badges
- Changed "CACM Ready" → "CACM Practice"
- Updated figure generation instructions (2 figures)
- Updated citation with new title

#### REPRODUCIBILITY.md
- Updated validation checklist (2 figures instead of 10)
- Maintained all experimental reproduction steps

#### Archived Figures
- Created `archived_figures/` folder with 11 figures
- Added README documenting what was archived and why
- Preserved all figures for future presentations/supplements

### Files Modified
- `MANUSCRIPT_PREPARATION/manuscript/prompt_injection_cacm.tex`
- `README.md`
- `REPRODUCIBILITY.md`

### Files Created
- `MANUSCRIPT_PREPARATION/manuscript/archived_figures/README.md`
- `CHANGELOG.md` (this file)

### Migration Notes
- All archived figures remain accessible in `archived_figures/`
- Original PDF architecture diagram replaced with LaTeX TikZ version
- Figure numbering automatically updated by LaTeX
- All cross-references validated and working

---

## [1.0.0] - 2025-11-XX - Initial Submission

Initial research paper version with complete 8-phase evaluation framework and comprehensive analysis figures.
