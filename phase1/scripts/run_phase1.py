"""
Master script to run complete Phase 1 pipeline.
Orchestrates KB generation, Part A, Part B, and analysis.
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


def run_step(step_name, script_name, required_files=None):
    """
    Run a pipeline step.
    
    Args:
        step_name: Display name of the step
        script_name: Python script to execute
        required_files: List of files that should exist after step completes
    
    Returns:
        bool: True if successful, False otherwise
    """
    print_header(f"STEP: {step_name}")
    
    start_time = time.time()
    
    try:
        # Run the script from scripts directory
        script_path = Path(__file__).parent / script_name
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed/60:.1f} minutes")
        
        # Verify required files exist (with phase1 prefix)
        if required_files:
            phase1_dir = Path(__file__).parent.parent
            missing = []
            for f in required_files:
                # Check in appropriate subdirectory
                if f.endswith('.jsonl') or f.endswith('.json'):
                    full_path = phase1_dir / "data" / f
                elif f.endswith('.csv') or f.endswith('.txt'):
                    full_path = phase1_dir / "stats" / f
                elif f.endswith('.png'):
                    full_path = phase1_dir / "plots" / f
                else:
                    full_path = phase1_dir / "data" / f
                
                if not full_path.exists():
                    missing.append(str(full_path))
            
            if missing:
                print(f"\n✗ ERROR: Expected files not found: {missing}")
                return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: {step_name} failed")
        print(f"Error details: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR in {step_name}: {e}")
        return False


def main():
    """Run complete Phase 1 pipeline."""
    
    print_header("PHASE 1: BASELINE VULNERABILITY ASSESSMENT")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # Step 1: Generate Knowledge Base
    if not run_step(
        "Generate Knowledge Base (Part A)",
        "generate_kb.py",
        required_files=["partA_kb.jsonl"]
    ):
        print("\n✗ Pipeline aborted at Step 1")
        return 1
    
    # Step 2: Run Part A Experiment
    if not run_step(
        "RAG-Borne Injection Experiment (Part A)",
        "partA_experiment.py",
        required_files=["partA_results.json"]
    ):
        print("\n✗ Pipeline aborted at Step 2")
        print("Note: Part A may take 2-3 hours to complete")
        return 1
    
    # Step 3: Run Part B Experiment
    if not run_step(
        "Schema Smuggling Experiment (Part B)",
        "partB_experiment.py",
        required_files=["partB_results.json"]
    ):
        print("\n✗ Pipeline aborted at Step 3")
        print("Note: Part B may take 1-2 hours to complete")
        return 1
    
    # Step 4: Analyze Results
    if not run_step(
        "Analyze and Visualize Results",
        "analyze_results.py",
        required_files=[
            "partA_analysis.csv",
            "partB_analysis.csv",
            "phase1_summary.txt",
            "partA_heatmap.png",
            "partB_heatmap.png",
            "phase1_comparison.png"
        ]
    ):
        print("\n✗ Pipeline aborted at Step 4")
        return 1
    
    # Success!
    pipeline_elapsed = time.time() - pipeline_start
    
    print_header("PHASE 1 PIPELINE COMPLETE")
    print(f"Total time: {pipeline_elapsed/3600:.2f} hours")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Generated outputs:")
    phase1_dir = Path(__file__).parent.parent
    outputs = [
        ("Knowledge Base", phase1_dir / "data" / "partA_kb.jsonl"),
        ("Part A Results", phase1_dir / "data" / "partA_results.json"),
        ("Part B Results", phase1_dir / "data" / "partB_results.json"),
        ("Part A Analysis", phase1_dir / "stats" / "partA_analysis.csv"),
        ("Part B Analysis", phase1_dir / "stats" / "partB_analysis.csv"),
        ("Part A Heatmap", phase1_dir / "plots" / "partA_heatmap.png"),
        ("Part B Heatmap", phase1_dir / "plots" / "partB_heatmap.png"),
        ("Comparison Plot", phase1_dir / "plots" / "phase1_comparison.png"),
        ("Summary Report", phase1_dir / "stats" / "phase1_summary.txt")
    ]
    
    for name, filepath in outputs:
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"  ✓ {name}: {filepath.relative_to(phase1_dir)} ({size:.1f} KB)")
        else:
            print(f"  ✗ {name}: {filepath.relative_to(phase1_dir)} (MISSING)")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Review phase1_summary.txt for overall results")
    print("  2. Examine heatmaps and comparison plots")
    print("  3. Analyze detailed JSON results for paper")
    print("  4. Proceed to Phase 2 (defense development)")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
