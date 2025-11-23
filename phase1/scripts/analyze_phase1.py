import json
from collections import defaultdict

# Load results
with open('partA_results.json', encoding='utf-8') as f:
    partA = json.load(f)

with open('partB_results.json', encoding='utf-8') as f:
    partB = json.load(f)

print("=" * 70)
print("PART A ANALYSIS")
print("=" * 70)

print(f"\nTotal records: {len(partA)}")

# By model
models = defaultdict(list)
for r in partA:
    models[r['model']].append(r)

for model in sorted(models.keys()):
    results = models[model]
    injected = [r for r in results if r['is_injected']]
    successful = [r for r in injected if r['injection_success']]
    asr = len(successful) / len(injected) * 100 if injected else 0
    
    print(f"\n{model}:")
    print(f"  Total queries: {len(results)}")
    print(f"  Injected queries: {len(injected)}")
    print(f"  Successful injections: {len(successful)}")
    print(f"  Attack Success Rate (ASR): {asr:.2f}%")
    
    # By evasion type
    evasion_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in injected:
        ev_type = r['evasion_type']
        evasion_stats[ev_type]['total'] += 1
        if r['injection_success']:
            evasion_stats[ev_type]['success'] += 1
    
    print(f"\n  By evasion type:")
    for ev_type in sorted(evasion_stats.keys()):
        stats = evasion_stats[ev_type]
        ev_asr = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {ev_type}: {ev_asr:.2f}% ({stats['success']}/{stats['total']})")

# Timing
print(f"\n\nTiming Analysis:")
for model in sorted(models.keys()):
    results = models[model]
    avg_time = sum(r['generation_time_sec'] for r in results) / len(results)
    avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in results) / len(results)
    print(f"  {model}:")
    print(f"    Avg generation time: {avg_time:.2f}s")
    print(f"    Avg tokens/sec: {avg_tokens_per_sec:.2f}")

print("\n" + "=" * 70)
print("PART B ANALYSIS")
print("=" * 70)

print(f"\nTotal records: {len(partB)}")

# By model
models_b = defaultdict(list)
for r in partB:
    models_b[r['model']].append(r)

for model in sorted(models_b.keys()):
    results = models_b[model]
    attacks = [r for r in results if r['is_attack']]
    successful = [r for r in attacks if r['attack_success']]
    asr = len(successful) / len(attacks) * 100 if attacks else 0
    
    print(f"\n{model}:")
    print(f"  Total test cases: {len(results)}")
    print(f"  Attack cases: {len(attacks)}")
    print(f"  Successful attacks: {len(successful)}")
    print(f"  Attack Success Rate: {asr:.2f}%")
    
    # By mechanism
    mechanism_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in attacks:
        mech = r['mechanism']
        mechanism_stats[mech]['total'] += 1
        if r['attack_success']:
            mechanism_stats[mech]['success'] += 1
    
    print(f"\n  By mechanism:")
    for mech in sorted(mechanism_stats.keys()):
        stats = mechanism_stats[mech]
        mech_asr = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {mech}: {mech_asr:.2f}% ({stats['success']}/{stats['total']})")
    
    # By tool
    tool_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in attacks:
        tool = r['tool']
        tool_stats[tool]['total'] += 1
        if r['attack_success']:
            tool_stats[tool]['success'] += 1
    
    print(f"\n  By tool:")
    for tool in sorted(tool_stats.keys()):
        stats = tool_stats[tool]
        tool_asr = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {tool}: {tool_asr:.2f}% ({stats['success']}/{stats['total']})")

print("\n" + "=" * 70)
