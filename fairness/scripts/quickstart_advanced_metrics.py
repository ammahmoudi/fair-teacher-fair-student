#!/usr/bin/env python3
"""
Quick Start: Run Advanced Fairness Metrics on Your Data
========================================================

This script demonstrates the simplest way to run the new advanced fairness
metrics (DP Gap, EO Gap, FVO) on your experiment results.

Usage:
    # With synthetic demo data
    python3 fairness/quickstart_advanced_metrics.py
    
    # With your own data
    python3 fairness/quickstart_advanced_metrics.py --data-file your_results.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics


def demo_with_synthetic_data():
    """Quick demo with synthetic blood glucose data."""
    print("\n" + "="*80)
    print("QUICK DEMO: Advanced Fairness Metrics")
    print("="*80)
    print("\nGenerating synthetic blood glucose data...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    
    # Create demographics
    gender = np.random.choice(['male', 'female'], n_samples, p=[0.55, 0.45])
    
    # Generate realistic glucose values (mg/dL)
    y_true = np.random.normal(120, 30, n_samples)
    y_true = np.clip(y_true, 40, 300)
    
    # Add some hypoglycemic events (< 70 mg/dL)
    hypo_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_true[hypo_indices] = np.random.uniform(40, 70, len(hypo_indices))
    
    # Generate predictions with slight bias
    y_pred = y_true + np.random.normal(0, 12, n_samples)
    
    # Introduce fairness issue: worse predictions for females in hypoglycemic range
    female_mask = gender == 'female'
    female_hypo_mask = female_mask & (y_true < 70)
    y_pred[female_hypo_mask] += np.random.uniform(5, 15, np.sum(female_hypo_mask))
    
    y_pred = np.clip(y_pred, 40, 300)
    
    print(f"✓ Generated {n_samples} samples")
    print(f"  - Males: {np.sum(gender=='male')}")
    print(f"  - Females: {np.sum(gender=='female')}")
    print(f"  - True hypoglycemic events: {np.sum(y_true < 70)}")
    
    # Run analysis
    run_analysis(y_true, y_pred, gender, 'Gender')


def load_from_csv(csv_path: Path):
    """Load data from CSV file."""
    print(f"\nLoading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows")
    
    # Expected columns: y_true, y_pred, demographic_column
    # Adjust these based on your actual CSV structure
    required = ['y_true', 'y_pred']
    
    for col in required:
        if col not in df.columns:
            print(f"\n❌ Error: Column '{col}' not found in CSV")
            print(f"Available columns: {', '.join(df.columns)}")
            return None
    
    # Find demographic column (anything other than y_true, y_pred)
    demo_cols = [col for col in df.columns if col not in required]
    
    if not demo_cols:
        print("\n❌ Error: No demographic column found")
        print("Your CSV should have columns: y_true, y_pred, <demographic>")
        return None
    
    demo_col = demo_cols[0]
    print(f"Using demographic column: {demo_col}")
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    demographics = df[demo_col].values
    
    run_analysis(y_true, y_pred, demographics, demo_col)


def run_analysis(y_true, y_pred, demographics, demo_name):
    """Run the advanced fairness metrics analysis."""
    print("\n" + "="*80)
    print("RUNNING ADVANCED FAIRNESS ANALYSIS")
    print("="*80)
    
    # Initialize metrics calculator
    afm = AdvancedFairnessMetrics(
        hypoglycemia_threshold=70.0,   # mg/dL
        hyperglycemia_threshold=180.0  # mg/dL
    )
    
    # Calculate all three metrics
    print("\nCalculating metrics...")
    results = afm.calculate_all_advanced_metrics(
        y_true=y_true,
        y_pred=y_pred,
        group_labels=demographics,
        group_attribute=demo_name,
        risk_type='hypoglycemia'
    )
    
    # Print comprehensive report
    afm.print_comprehensive_report(results)
    
    # Save results to JSON
    output_dir = Path(__file__).parent.parent / "analysis_results" / "advanced_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "quickstart_results.json"
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {output_file}")
    
    # Quick summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    
    metrics = results['metrics']
    dp = metrics['demographic_parity_gap']['dp_gap']
    eo = metrics['equal_opportunity_gap']['eo_gap']
    fvo = metrics['fairness_violation_objective']['classification_based']['fvo']
    
    print(f"\n📊 Metric Values:")
    print(f"   DP Gap:  {dp:.4f}  {'✅' if dp < 0.10 else '⚠️' if dp < 0.20 else '❌'}")
    print(f"   EO Gap:  {eo:.4f}  {'✅' if eo < 0.10 else '⚠️' if eo < 0.20 else '❌'}")
    print(f"   FVO:     {fvo:.4f}  {'✅' if fvo < 0.05 else '⚠️' if fvo < 0.10 else '❌'}")
    
    print(f"\n🎯 Overall Status: {results['overall_assessment']['status']}")
    print(f"   {results['overall_assessment']['summary']}")
    
    if results['overall_assessment']['issues']:
        print(f"\n⚠️  Issues Detected:")
        for issue in results['overall_assessment']['issues']:
            print(f"   • {issue}")
    else:
        print(f"\n✅ No significant fairness issues detected!")
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Demographic Parity Gap (DP Gap):
  • Measures if alerts are distributed equally across groups
  • ✅ < 0.10: Good  |  ⚠️ 0.10-0.20: Concerning  |  ❌ > 0.20: Critical
  
Equal Opportunity Gap (EO Gap):
  • Measures if model detects actual events equally well
  • ✅ < 0.10: Good  |  ⚠️ 0.10-0.20: Concerning  |  ❌ > 0.20: Critical
  • **MOST IMPORTANT for patient safety!**
  
Fairness Violation Objective (FVO):
  • Measures maximum accuracy disparity between groups
  • ✅ < 0.05: Good  |  ⚠️ 0.05-0.10: Concerning  |  ❌ > 0.10: Critical
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Quick start for advanced fairness metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  python3 fairness/quickstart_advanced_metrics.py
  
  # Run with your own CSV file
  python3 fairness/quickstart_advanced_metrics.py --data-file results.csv
  
  # Your CSV should have columns: y_true, y_pred, <demographic>
  # Example:
  #   y_true,y_pred,gender
  #   95.2,98.1,male
  #   68.5,75.3,female
  #   ...
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to CSV file with your data'
    )
    
    parser.add_argument(
        '--hypoglycemia-threshold',
        type=float,
        default=70.0,
        help='Hypoglycemia threshold in mg/dL (default: 70.0)'
    )
    
    args = parser.parse_args()
    
    if args.data_file:
        csv_path = Path(args.data_file)
        if not csv_path.exists():
            print(f"\n❌ Error: File not found: {csv_path}")
            return 1
        load_from_csv(csv_path)
    else:
        demo_with_synthetic_data()
    
    print("\n✅ Analysis complete!\n")
    return 0


if __name__ == "__main__":
    exit(main())
