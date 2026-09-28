#!/usr/bin/env python3
"""
Regenerate the comprehensive standardized metrics table using real production data
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from utils.enhanced_data_loader import EnhancedEfficiencyDataLoader
from utils.latex_table_generator import generate_all_tables
import pandas as pd

print("📊 Loading production inference data...")
loader = EnhancedEfficiencyDataLoader(Path('.'))
inference_data = loader.parse_all_data()

print(f"✅ Loaded {len(inference_data)} records")
print(f"📋 Models found: {sorted(inference_data['model_name'].unique())}")

# Filter to inference-only records for cleaner aggregation
if 'mode' in inference_data.columns:
    inference_only = inference_data[inference_data['mode'] == 'inference']
    print(f"🎯 Using {len(inference_only)} inference-mode records")
else:
    inference_only = inference_data
    print(f"⚠️  No mode column found, using all {len(inference_only)} records")

# Generate tables
output_dir = Path.cwd() / "notebooks" / "outputs" / "latex_tables"
results = generate_all_tables(inference_only, output_dir=output_dir)

print(f"\n✅ Table generated: {results['comprehensive_standardized']}")

# Print first 30 lines for verification
with open(results['comprehensive_standardized'], 'r') as f:
    lines = f.readlines()
    print("\n" + "="*80)
    print("First 30 lines of generated table:")
    print("="*80)
    print(''.join(lines[:30]))
