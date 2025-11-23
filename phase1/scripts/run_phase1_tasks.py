"""
Master orchestration script for Phase 1 Tasks 1-3.
Runs in sequence:
  1. Statistical Analysis (McNemar, Wilson CI)
  2. Folder Refactoring
  3. Defense Verdict Labeling
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess


def print_header(text):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")


def run_task(task_num, task_name, script_name):
    """
    Run a task script.
    
    Args:
        task_num: Task number
        task_name: Display name
        script_name: Python script to run
    
    Returns:
        bool: True if successful, False otherwise
    """
    print_header(f"TASK {task_num}: {task_name}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Task {task_num} completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: Task {task_num} failed")
        print(f"Error details: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR in Task {task_num}: {e}")
        return False


def main():
    """Run all Phase 1 tasks."""
    
    print_header("PHASE 1: ADVANCED ANALYSIS & REFACTORING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # Task 1: Statistical Analysis
    if not run_task(
        1,
        "Advanced Statistical Analysis",
        "phase1_statistical_analysis.py"
    ):
        print("\n✗ Pipeline aborted at Task 1")
        return 1
    
    # Task 2: Folder Refactoring
    if not run_task(
        2,
        "Phase 1 Folder Refactor",
        "phase1_refactor.py"
    ):
        print("\n✗ Pipeline aborted at Task 2")
        return 1
    
    # Task 3: Defense Verdict Labeling
    if not run_task(
        3,
        "Label Per-Sample Defense Verdicts",
        "phase1_label_defenses.py"
    ):
        print("\n✗ Pipeline aborted at Task 3")
        return 1
    
    # Success!
    pipeline_elapsed = time.time() - pipeline_start
    
    print_header("PHASE 1 TASKS COMPLETE")
    print(f"Total time: {pipeline_elapsed:.1f} seconds ({pipeline_elapsed/60:.1f} minutes)")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Generated outputs:")
    outputs = [
        ("Statistical Analysis", "phase1/stats/ci_summary.csv"),
        ("McNemar Results", "phase1/stats/mcnemar_results.csv"),
        ("Defense Matrix", "phase1/stats/defense_pairwise_matrix.png"),
        ("Annotated Data", "phase1/data/phase1_output_annotated.json"),
        ("Phase 1 README", "phase1/README.md"),
    ]
    
    for name, filepath in outputs:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  ✓ {name}: {filepath} ({size_str})")
        else:
            print(f"  ✗ {name}: {filepath} (MISSING)")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Review phase1/stats/ for statistical results")
    print("  2. Examine phase1/plots/ for visualizations")
    print("  3. Analyze phase1/data/phase1_output_annotated.json")
    print("  4. Check phase1/README.md for documentation")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
