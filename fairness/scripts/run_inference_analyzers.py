#!/usr/bin/env python3
"""
Script to run fairness analysis for all inference scenarios.

Analyzes fairness across:
- Gender
- Age Groups  
- Pump Models
- Sensor Bands
- Cohorts

For scenarios:
- Inference Only (no training)
- Trained on Standard Data
- Trained on Noisy Data
- Trained on Denoised Data
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Parse arguments
parser = argparse.ArgumentParser(description='Run all inference scenario fairness analyzers')
parser.add_argument('--feature', type=str, default='all',
                   choices=['all', 'gender', 'age', 'pump', 'sensor', 'cohort', 'legendary'],
                   help='Which analyzer to run (default: all)')
args = parser.parse_args()

analyzers = {
    "Gender": "fairness/analyzers/inference_gender_analyzer.py",
    "Age Group": "fairness/analyzers/inference_age_analyzer.py",
    "Pump Model": "fairness/analyzers/inference_pump_analyzer.py",
    "Sensor Band": "fairness/analyzers/inference_sensor_analyzer.py",
    "Cohort": "fairness/analyzers/inference_cohort_analyzer.py",
    "Legendary (All Features)": "fairness/analyzers/inference_legendary_analyzer.py"
}

# Filter analyzers based on argument
if args.feature == 'gender':
    analyzers = {"Gender": analyzers["Gender"]}
elif args.feature == 'age':
    analyzers = {"Age Group": analyzers["Age Group"]}
elif args.feature == 'pump':
    analyzers = {"Pump Model": analyzers["Pump Model"]}
elif args.feature == 'sensor':
    analyzers = {"Sensor Band": analyzers["Sensor Band"]}
elif args.feature == 'cohort':
    analyzers = {"Cohort": analyzers["Cohort"]}
elif args.feature == 'legendary':
    analyzers = {"Legendary (All Features)": analyzers["Legendary (All Features)"]}

# Dynamically get python executable from venv
venv_python = project_root / "venv" / "bin" / "python"
python_cmd = str(venv_python) if venv_python.exists() else sys.executable

print("=" * 80)
print("RUNNING INFERENCE SCENARIOS FAIRNESS ANALYZERS")
print("=" * 80)
print("\nAnalyzing fairness across inference scenarios:")
print("  - Inference Only (no training)")
print("  - Trained on Standard Data")
print("  - Trained on Noisy Data")
print("  - Trained on Denoised Data")
print()

results = {}

for name, script in analyzers.items():
    print(f"\n🔍 Running {name} Analyzer...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [python_cmd, script],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"✅ {name} Analyzer: PASSED")
            # Print stdout to show progress
            if result.stdout:
                print(result.stdout)
            results[name] = "PASSED"
        else:
            print(f"❌ {name} Analyzer: FAILED")
            print(f"Error: {result.stderr[:500]}")
            results[name] = "FAILED"
    except subprocess.TimeoutExpired:
        print(f"⏱️ {name} Analyzer: TIMEOUT")
        results[name] = "TIMEOUT"
    except Exception as e:
        print(f"💥 {name} Analyzer: ERROR - {e}")
        results[name] = "ERROR"

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

for name, status in results.items():
    icon = "✅" if status == "PASSED" else "❌"
    print(f"{icon} {name:30s}: {status}")

passed = sum(1 for s in results.values() if s == "PASSED")
total = len(results)

print("\n" + "=" * 80)
print(f"FINAL RESULT: {passed}/{total} analyzers passed")
print("=" * 80)

sys.exit(0 if passed == total else 1)
