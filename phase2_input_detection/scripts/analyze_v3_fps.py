"""
Analyze why V3 has high false positives.
"""

import json
from pathlib import Path
from input_detectors import get_input_detector


def main():
    script_dir = Path(__file__).parent
    phase2_dir = script_dir.parent
    phase1_dir = phase2_dir.parent / "phase1"
    results_file = phase1_dir / "data" / "partA_results.json"
    
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get different categories
    benign = [s for s in data if not s.get("is_injected", False)]
    failed_attacks = [s for s in data if s.get("is_injected", False) and not s.get("injection_success", False)]
    successful_attacks = [s for s in data if s.get("is_injected", False) and s.get("injection_success", False)]
    
    detector_v3 = get_input_detector("v3")
    
    print("="*70)
    print("V3 FALSE POSITIVE ANALYSIS")
    print("="*70)
    
    # Check benign
    fps_benign = []
    for sample in benign:
        query = sample.get("query", "")
        result = detector_v3.classify(query)
        if result.is_attack:
            fps_benign.append({
                "query": query,
                "matched": result.matched,
                "confidence": result.confidence
            })
    
    # Check failed attacks (these are FPs - attack failed but we flag it)
    fps_failed = []
    for sample in failed_attacks:
        query = sample.get("query", "")
        result = detector_v3.classify(query)
        if result.is_attack:
            fps_failed.append({
                "query": query,
                "matched": result.matched,
                "confidence": result.confidence,
                "evasion_type": sample.get("evasion_type", "")
            })
    
    print(f"\nBenign queries: {len(benign)}")
    print(f"False positives on benign: {len(fps_benign)} ({len(fps_benign)/len(benign)*100:.1f}%)")
    
    print(f"\nFailed attacks: {len(failed_attacks)}")
    print(f"False positives on failed attacks: {len(fps_failed)} ({len(fps_failed)/len(failed_attacks)*100:.1f}%)")
    
    print(f"\nSuccessful attacks: {len(successful_attacks)}")
    tp = sum(1 for s in successful_attacks if detector_v3.classify(s.get("query", "")).is_attack)
    print(f"True positives: {tp} ({tp/len(successful_attacks)*100:.1f}%)")
    
    fps = fps_failed  # The real FPs are failed attacks we incorrectly flag
    
    print("\n" + "="*70)
    print("SAMPLE FALSE POSITIVES")
    print("="*70)
    
    for i, fp in enumerate(fps[:20], 1):
        print(f"\n{i}. Query: {fp['query']}")
        print(f"   Matched: {fp['matched']}")
        print(f"   Confidence: {fp['confidence']:.2f}")
    
    # Analyze which keywords are causing FPs
    print("\n" + "="*70)
    print("KEYWORD ANALYSIS")
    print("="*70)
    
    keyword_counts = {}
    for fp in fps:
        for match in fp["matched"]:
            if "keyword_" in match:
                keyword = match.split("keyword_")[1]
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    print("\nKeywords causing false positives:")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
        print(f"  {keyword}: {count} FPs ({count/len(fps)*100:.1f}%)")


if __name__ == "__main__":
    main()
