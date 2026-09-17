"""Run all seven Stage 2 experiments in numerical order."""

from pathlib import Path
import subprocess
import sys


STAGE2_DIR = Path(__file__).resolve().parent / "stage2"
scripts = sorted(STAGE2_DIR.glob("stage2_[0-9][0-9]_*.py"))

if len(scripts) != 7:
    raise RuntimeError(f"Expected 7 Stage 2 scripts, found {len(scripts)}")

for script in scripts:
    print(f"\n{'=' * 80}\nRUNNING {script.name}\n{'=' * 80}", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)

print("\nAll Stage 2 experiments completed successfully.")
