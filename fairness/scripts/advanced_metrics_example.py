#!/usr/bin/env python3
"""
Example: Using Advanced Fairness Metrics with Fair Teacher–Fair Student Results
==============================================================

This script demonstrates how to integrate the three advanced fairness metrics
(DP Gap, EO Gap, FVO) with your existing Fair Teacher–Fair Student experiment results.

Usage:
    python fairness/advanced_metrics_example.py --experiment-path <path_to_results>
    
Example:
    python fairness/advanced_metrics_example.py --experiment-path distillation_experiments/pipeline_2025-10-18_12-56-50
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from fairness.metrics.advanced_fairness_metrics import AdvancedFairnessMetrics


class AdvancedMetricsAnalyzer:
    """Analyze Fair Teacher–Fair Student experiment results using advanced fairness metrics."""
    
    def __init__(self, 
                 experiment_path: Optional[Path] = None,
                 hypoglycemia_threshold: float = 70.0,
                 hyperglycemia_threshold: float = 180.0):
        """
        Initialize analyzer.
        
        Args:
            experiment_path: Path to experiment results directory
            hypoglycemia_threshold: Threshold for hypoglycemia detection (mg/dL)
            hyperglycemia_threshold: Threshold for hyperglycemia detection (mg/dL)
        """
        self.experiment_path = experiment_path
        self.afm = AdvancedFairnessMetrics(
            hypoglycemia_threshold=hypoglycemia_threshold,
            hyperglycemia_threshold=hyperglycemia_threshold
        )
    
    def load_predictions_from_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load predictions and true values from CSV file."""
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} predictions from {csv_path.name}")
        return df
    
    def analyze_with_demographics(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 demographics: Dict[str, np.ndarray],
                                 risk_type: str = 'hypoglycemia') -> Dict[str, Dict]:
        """
        Analyze fairness across multiple demographic attributes.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            demographics: Dictionary mapping demographic names to group labels
                         e.g., {'gender': array(['male', 'female', ...]),
                                'age_group': array(['20-40', '40-60', ...])}
            risk_type: Type of risk to analyze
            
        Returns:
            Dictionary with results for each demographic attribute
        """
        results = {}
        
        for demo_name, demo_labels in demographics.items():
            print(f"\n{'='*60}")
            print(f"Analyzing: {demo_name}")
            print(f"{'='*60}")
            
            # Calculate all advanced metrics
            demo_results = self.afm.calculate_all_advanced_metrics(
                y_true=y_true,
                y_pred=y_pred,
                group_labels=demo_labels,
                group_attribute=demo_name,
                risk_type=risk_type
            )
            
            results[demo_name] = demo_results
            
            # Print summary
            self._print_summary(demo_results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print concise summary of results."""
        metrics = results['metrics']
        dp_gap = metrics['demographic_parity_gap']['dp_gap']
        eo_gap = metrics['equal_opportunity_gap']['eo_gap']
        fvo = metrics['fairness_violation_objective']['classification_based']['fvo']
        
        print(f"\n📊 Summary for {results['group_attribute']}:")
        print(f"   DP Gap:  {dp_gap:.4f} - {self._interpret_gap(dp_gap, 'dp')}")
        print(f"   EO Gap:  {eo_gap:.4f} - {self._interpret_gap(eo_gap, 'eo')}")
        print(f"   FVO:     {fvo:.4f} - {self._interpret_gap(fvo, 'fvo')}")
        print(f"   Status:  {results['overall_assessment']['status']}")
    
    def _interpret_gap(self, value: float, metric_type: str) -> str:
        """Quick interpretation of gap value."""
        if metric_type in ['dp', 'eo']:
            if value < 0.05:
                return "✅ Excellent"
            elif value < 0.10:
                return "✓ Good"
            elif value < 0.20:
                return "⚠️ Concerning"
            else:
                return "❌ Critical"
        else:  # FVO
            if value < 0.02:
                return "✅ Excellent"
            elif value < 0.05:
                return "✓ Good"
            elif value < 0.10:
                return "⚠️ Concerning"
            else:
                return "❌ Poor"
    
    def compare_scenarios(self,
                         scenarios: Dict[str, Dict],
                         demographic: str = 'gender',
                         output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Compare advanced fairness metrics across different scenarios.
        
        Args:
            scenarios: Dictionary mapping scenario names to results
                      e.g., {'teacher': results1, 'distilled': results2}
            demographic: Demographic attribute to compare
            output_path: Optional path to save comparison plot
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for scenario_name, results in scenarios.items():
            if demographic not in results:
                continue
            
            demo_results = results[demographic]
            metrics = demo_results['metrics']
            
            comparison_data.append({
                'scenario': scenario_name,
                'dp_gap': metrics['demographic_parity_gap']['dp_gap'],
                'eo_gap': metrics['equal_opportunity_gap']['eo_gap'],
                'fvo_classification': metrics['fairness_violation_objective']['classification_based']['fvo'],
                'fvo_regression': metrics['fairness_violation_objective']['regression_based']['fvo'],
                'status': demo_results['overall_assessment']['status']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create visualization
        if output_path is not None:
            self._plot_scenario_comparison(df_comparison, demographic, output_path)
        
        return df_comparison
    
    def _plot_scenario_comparison(self, df: pd.DataFrame, demographic: str, output_path: Path):
        """Create comparison plot across scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        scenarios = df['scenario'].tolist()
        
        # Plot 1: DP Gap
        axes[0].bar(scenarios, df['dp_gap'], color='skyblue', edgecolor='navy')
        axes[0].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Excellent (<0.05)')
        axes[0].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='Good (<0.10)')
        axes[0].axhline(y=0.20, color='red', linestyle='--', alpha=0.5, label='Concerning (<0.20)')
        axes[0].set_ylabel('DP Gap')
        axes[0].set_title('Demographic Parity Gap\n(Lower is Better)')
        axes[0].legend(fontsize=8)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: EO Gap
        axes[1].bar(scenarios, df['eo_gap'], color='lightcoral', edgecolor='darkred')
        axes[1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.10, color='orange', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.20, color='red', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('EO Gap')
        axes[1].set_title('Equal Opportunity Gap\n(Lower is Better)')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: FVO
        axes[2].bar(scenarios, df['fvo_classification'], color='lightgreen', edgecolor='darkgreen')
        axes[2].axhline(y=0.02, color='green', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0.10, color='red', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('FVO')
        axes[2].set_title('Fairness Violation Objective\n(Lower is Better)')
        axes[2].grid(axis='y', alpha=0.3)
        
        for ax in axes:
            ax.set_xlabel('Scenario')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Advanced Fairness Metrics Comparison\nDemographic: {demographic.title()}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot to {output_path}")
        plt.close()
    
    def save_results(self, results: Dict, output_path: Path):
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Saved results to {output_path}")


def example_with_synthetic_data():
    """Example using synthetic blood glucose data."""
    print("\n" + "="*80)
    print("EXAMPLE: Advanced Fairness Metrics with Synthetic Data")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Demographics
    gender = np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4])
    age_group = np.random.choice(['20-40', '40-60', '60-80'], n_samples)
    
    # True glucose values
    y_true = np.random.normal(120, 30, n_samples)
    y_true = np.clip(y_true, 40, 300)
    
    # Add hypoglycemic events
    hypo_mask = np.random.random(n_samples) < 0.15
    y_true[hypo_mask] = np.random.uniform(40, 70, np.sum(hypo_mask))
    
    # Predictions with bias
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    
    # Introduce fairness issue
    female_mask = gender == 'female'
    female_hypo_mask = female_mask & (y_true < 70)
    y_pred[female_hypo_mask] += np.random.uniform(10, 20, np.sum(female_hypo_mask))
    y_pred = np.clip(y_pred, 40, 300)
    
    # Initialize analyzer
    analyzer = AdvancedMetricsAnalyzer()
    
    # Analyze with demographics
    demographics = {
        'gender': gender,
        'age_group': age_group
    }
    
    results = analyzer.analyze_with_demographics(
        y_true=y_true,
        y_pred=y_pred,
        demographics=demographics,
        risk_type='hypoglycemia'
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent / "analysis_results" / "advanced_metrics_examples"
    analyzer.save_results(
        results,
        output_dir / "synthetic_data_advanced_metrics.json"
    )
    
    # Compare scenarios (simulate different models)
    y_pred_improved = y_true + np.random.normal(0, 8, n_samples)
    y_pred_improved = np.clip(y_pred_improved, 40, 300)
    
    results_improved = analyzer.analyze_with_demographics(
        y_true=y_true,
        y_pred=y_pred_improved,
        demographics={'gender': gender},
        risk_type='hypoglycemia'
    )
    
    # Compare
    scenarios = {
        'baseline': results,
        'improved': results_improved
    }
    
    df_comparison = analyzer.compare_scenarios(
        scenarios,
        demographic='gender',
        output_path=output_dir / "scenario_comparison.png"
    )
    
    print("\n" + "="*80)
    print("SCENARIO COMPARISON")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    print("\n✅ Example complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze Fair Teacher–Fair Student results with advanced fairness metrics'
    )
    parser.add_argument(
        '--experiment-path',
        type=str,
        help='Path to experiment results directory'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Run synthetic data example'
    )
    parser.add_argument(
        '--hypoglycemia-threshold',
        type=float,
        default=70.0,
        help='Hypoglycemia threshold in mg/dL (default: 70.0)'
    )
    parser.add_argument(
        '--hyperglycemia-threshold',
        type=float,
        default=180.0,
        help='Hyperglycemia threshold in mg/dL (default: 180.0)'
    )
    
    args = parser.parse_args()
    
    if args.example or not args.experiment_path:
        example_with_synthetic_data()
    else:
        # TODO: Implement loading from actual experiment results
        print(f"Loading from experiment: {args.experiment_path}")
        print("Note: Integration with actual experiment results coming soon!")
        print("For now, run with --example to see synthetic data demonstration.")


if __name__ == "__main__":
    main()
