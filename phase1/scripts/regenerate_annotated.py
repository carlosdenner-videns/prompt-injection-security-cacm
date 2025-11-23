"""Regenerate annotated JSON with proper encoding."""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'phase1/scripts')

from phase1_label_defenses import DefenseVerdictLabeler

labeler = DefenseVerdictLabeler(
    results_json="phase1/data/partA_results.json",
    output_dir="phase1/data"
)
labeler.run_labeling()
