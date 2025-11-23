"""
CLI tool for input-side attack detection.
Usage: python detect_input_attack.py --file input.jsonl --model v3
"""

import argparse
import json
import sys
from pathlib import Path
from input_detectors import get_input_detector


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect prompt injection attacks in input text"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Input file (JSONL format, one JSON object per line)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["v1", "v2", "v3"],
        default="v3",
        help="Detector version (default: v3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for flagging attacks (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Load detector
    detector = get_input_detector(args.model)
    print(f"✓ Loaded detector {args.model}", file=sys.stderr)
    
    # Process input file
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"❌ File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    results = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                
                # Detect
                result = detector.classify(text)
                
                # Format output
                output_obj = {
                    "line": line_num,
                    "is_attack": result.is_attack,
                    "confidence": result.confidence,
                    "matched": result.matched,
                    "model": args.model,
                }
                
                # Add original metadata if present
                if "id" in obj:
                    output_obj["id"] = obj["id"]
                if "metadata" in obj:
                    output_obj["metadata"] = obj["metadata"]
                
                results.append(output_obj)
                
                # Print if flagged
                if result.is_attack and result.confidence >= args.threshold:
                    print(json.dumps(output_obj))
            
            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: Invalid JSON - {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"❌ Line {line_num}: Error - {e}", file=sys.stderr)
                continue
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"✓ Saved {len(results)} results to {args.output}", file=sys.stderr)
    
    # Summary
    flagged = sum(1 for r in results if r["is_attack"] and r["confidence"] >= args.threshold)
    print(f"\n✓ Processed {len(results)} inputs", file=sys.stderr)
    print(f"✓ Flagged {flagged} as attacks ({flagged/len(results)*100:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
