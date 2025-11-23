"""Setup phase1 folder structure."""
from pathlib import Path

base = Path("c:/Users/carlo/OneDrive - VIDENS ANALYTICS/Prompt Injection Security/phase1")
dirs = [
    base / "data",
    base / "scripts",
    base / "stats",
    base / "plots"
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {d}")

print("\nPhase 1 folder structure ready!")
