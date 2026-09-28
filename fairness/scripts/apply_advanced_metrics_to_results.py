#!/usr/bin/env python3
"""
Apply Advanced Fairness Metrics to Existing Results
====================================================

This script demonstrates how to integrate advanced fairness metrics (DP Gap, EO Gap, FVO)
with your existing experiment results by loading prediction data.

Usage:
    python fairness/apply_advanced_metrics_to_results.py --experiment-dir <path>
    
Example:
    python fairness/apply_advanced_metrics_to_results.py --experiment-dir distillation_experiments/pipeline_2025-10-18_12-56-50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics
except ImportError as e:
    print(f"❌ Error importing advanced fairness metrics: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class AdvancedMetricsIntegrator:
    """Integrate advanced fairness metrics with existing experiment results."""
    
    def __init__(self, hypoglycemia_threshold=70.0, hyperglycemia_threshold=180.0):
        self.afm = AdvancedFairnessMetrics(
            hypoglycemia_threshold=hypoglycemia_threshold,
            hyperglycemia_threshold=hyperglycemia_threshold
        )
        self.results_dir = Path(__file__).parent.parent / "analysis_results" / "advanced_metrics"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_experiment_directory(self, experiment_dir: Path) -> Dict:
        """
        Analyze an experiment directory with advanced metrics.
        
        Expected structure:
            experiment_dir/
                predictions/
                    patient_001_predictions.csv
                    patient_002_predictions.csv
                    ...
                demographics.json (or metadata with demographic info)
        """
        print(f"\n{'='*80}")
        print(f"Analyzing Experiment: {experiment_dir.name}")
        print(f"{'='*80}\n")
        
        # Look for prediction files
        predictions_dir = experiment_dir / "predictions"
        if not predictions_dir.exists():
            predictions_dir = experiment_dir
        
        # Find CSV files with predictions
        csv_files = list(predictions_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {predictions_dir}")
            print("\nExpected structure:")
            print("  - CSV files with columns: y_true, y_pred")
            print("  - demographics.json with demographic information")
            return {}
        
        print(f"✓ Found {len(csv_files)} prediction files")
        
        # Load all predictions
        all_y_true = []
        all_y_pred = []
        all_demographics = {}
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check for required columns
                if 'y_true' in df.columns and 'y_pred' in df.columns:
                    all_y_true.extend(df['y_true'].values)
                    all_y_pred.extend(df['y_pred'].values)
                    
                    # Extract demographic info if available
                    for col in df.columns:
                        if col not in ['y_true', 'y_pred', 'timestamp', 'index']:
                            if col not in all_demographics:
                                all_demographics[col] = []
                            all_demographics[col].extend(df[col].values)
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load {csv_file.name}: {e}")
        
        if not all_y_true:
            print("❌ No predictions found with y_true and y_pred columns")
            return {}
        
        print(f"✓ Loaded {len(all_y_true)} predictions")
        
        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        
        # If no demographics in CSV, try loading from demographics.json
        if not all_demographics:
            demographics_file = experiment_dir / "demographics.json"
            if demographics_file.exists():
                try:
                    with open(demographics_file, 'r') as f:
                        demo_data = json.load(f)
                    
                    # Convert to arrays matching prediction length
                    for key, values in demo_data.items():
                        if len(values) == len(y_true):
                            all_demographics[key] = np.array(values)
                        elif isinstance(values, (str, int, float)):
                            # Repeat single value for all predictions
                            all_demographics[key] = np.array([values] * len(y_true))
                    
                    print(f"✓ Loaded demographics from {demographics_file.name}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load demographics.json: {e}")
        
        if not all_demographics:
            print("\n⚠️  No demographic information found!")
            print("    Creating synthetic demographics for demonstration...")
            # Create synthetic demographics for demonstration
            n = len(y_true)
            all_demographics = {
                'gender': np.random.choice(['male', 'female'], n),
                'age_group': np.random.choice(['20-40', '40-60', '60-80'], n)
            }
            print("    Note: These are NOT real demographics - for demo only!")
        
        # Run advanced metrics analysis for each demographic attribute
        results = {}
        
        for demo_name, demo_values in all_demographics.items():
            if len(demo_values) != len(y_true):
                print(f"⚠️  Skipping {demo_name}: length mismatch")
                continue
            
            print(f"\n{'─'*80}")
            print(f"Analyzing: {demo_name}")
            print(f"{'─'*80}")
            
            try:
                # Calculate advanced metrics
                demo_results = self.afm.calculate_all_advanced_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    group_labels=np.array(demo_values),
                    group_attribute=demo_name,
                    risk_type='hypoglycemia'
                )
                
                results[demo_name] = demo_results
                
                # Print summary
                self._print_quick_summary(demo_results)
                
            except Exception as e:
                print(f"❌ Error analyzing {demo_name}: {e}")
        
        # Save results
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"{experiment_dir.name}_advanced_metrics_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n{'='*80}")
            print(f"✓ Results saved to: {output_file}")
            print(f"{'='*80}\n")
        
        return results
    
    def _print_quick_summary(self, results: Dict):
        """Print quick summary of results."""
        metrics = results['metrics']
        dp = metrics['demographic_parity_gap']['dp_gap']
        eo = metrics['equal_opportunity_gap']['eo_gap']
        fvo = metrics['fairness_violation_objective']['classification_based']['fvo']
        
        print(f"\n  📊 Quick Summary:")
        print(f"     DP Gap:  {dp:.4f}  {'✅' if dp < 0.10 else '⚠️' if dp < 0.20 else '❌'}")
        print(f"     EO Gap:  {eo:.4f}  {'✅' if eo < 0.10 else '⚠️' if eo < 0.20 else '❌'}")
        print(f"     FVO:     {fvo:.4f}  {'✅' if fvo < 0.05 else '⚠️' if fvo < 0.10 else '❌'}")
        print(f"     Status:  {results['overall_assessment']['status']}")
    
    def analyze_all_experiments(self, base_dir: Path, pattern: str = "pipeline_*") -> Dict:
        """Analyze all experiments matching a pattern."""
        experiment_dirs = sorted(base_dir.glob(pattern))
        
        if not experiment_dirs:
            print(f"❌ No experiments found matching '{pattern}' in {base_dir}")
            return {}
        
        print(f"\n{'='*80}")
        print(f"Found {len(experiment_dirs)} experiments to analyze")
        print(f"{'='*80}")
        
        all_results = {}
        
        for exp_dir in experiment_dirs:
            try:
                results = self.analyze_experiment_directory(exp_dir)
                if results:
                    all_results[exp_dir.name] = results
            except Exception as e:
                print(f"❌ Error analyzing {exp_dir.name}: {e}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Apply advanced fairness metrics to existing experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--experiment-dir',
        type=str,
        help='Path to specific experiment directory'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='distillation_experiments',
        help='Base directory containing multiple experiments'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='pipeline_*',
        help='Pattern to match experiment directories'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all experiments in base directory'
    )
    
    parser.add_argument(
        '--hypoglycemia-threshold',
        type=float,
        default=70.0,
        help='Hypoglycemia threshold in mg/dL'
    )
    
    parser.add_argument(
        '--hyperglycemia-threshold',
        type=float,
        default=180.0,
        help='Hyperglycemia threshold in mg/dL'
    )
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = AdvancedMetricsIntegrator(
        hypoglycemia_threshold=args.hypoglycemia_threshold,
        hyperglycemia_threshold=args.hyperglycemia_threshold
    )
    
    if args.experiment_dir:
        # Analyze single experiment
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            print(f"❌ Directory not found: {exp_dir}")
            return 1
        
        integrator.analyze_experiment_directory(exp_dir)
    
    elif args.all:
        # Analyze all experiments
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f"❌ Base directory not found: {base_dir}")
            return 1
        
        integrator.analyze_all_experiments(base_dir, args.pattern)
    
    else:
        print("❌ Please specify either --experiment-dir or --all")
        print("\nExamples:")
        print("  # Analyze single experiment:")
        print("  python fairness/apply_advanced_metrics_to_results.py --experiment-dir distillation_experiments/pipeline_2025-10-18_12-56-50")
        print("\n  # Analyze all experiments:")
        print("  python fairness/apply_advanced_metrics_to_results.py --all")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
