#!/usr/bin/env python3
"""
Cohort Fairness Analyzer for Inference Scenarios

Analyzes fairness across cohorts (2018, 2020) for:
- Inference only (no training)
- Trained inference on standard data
- Trained inference on noisy data
- Trained inference on denoised data
"""

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from fairness.analyzers.base_inference_analyzer import BaseInferenceAnalyzer


class InferenceCohortAnalyzer(BaseInferenceAnalyzer):
    """Analyze age group fairness across inference scenarios."""
    
    def __init__(self, data_path=None):
        super().__init__(feature_name="Cohort", data_path=data_path or "")
    
    def get_feature_value(self, patient_id: str) -> str:
        """Get age group for a specific patient."""
        return self.patient_data.get(patient_id, {}).get('cohort', 'Unknown')
    
    def analyze(self):
        """Run comprehensive age group fairness analysis across all scenarios."""
        print("\n" + "="*70)
        print("COHORT FAIRNESS ANALYSIS ACROSS INFERENCE SCENARIOS")
        print("="*70)
        
        # Load all scenario results
        all_scenarios = self.load_all_scenarios()
        
        # Analyze each scenario
        all_statistics = {}
        all_fairness = {}
        
        for scenario_key, results in all_scenarios.items():
            if not results:
                print(f"⚠️  Skipping {scenario_key} - no data")
                continue
            
            grouped = self.group_by_feature(results)
            statistics = self.calculate_group_statistics(grouped)
            ratio, level = self.calculate_fairness_ratio(statistics)
            
            all_statistics[scenario_key] = statistics
            all_fairness[scenario_key] = {'ratio': ratio, 'level': level}
            
            self.print_scenario_summary(scenario_key, statistics, ratio, level)
        
        # Compare scenarios
        comparison = self.compare_scenarios(all_statistics)
        
        # Calculate scenario impacts
        scenario_impacts = self._calculate_scenario_impacts(all_statistics)
        
        print(f"\n{'='*70}")
        print("⚖️ FAIRNESS ASSESSMENT BY SCENARIO:")
        print(f"{'='*70}")
        
        for scenario_key, fairness_info in all_fairness.items():
            status = "✅" if scenario_key == comparison['best_scenario'] else "⚠️" if scenario_key == comparison['worst_scenario'] else "  "
            print(f"{status} {scenario_key:20s}: {fairness_info['ratio']:.2f}x ({fairness_info['level']})")
        
        print(f"\n🎯 SCENARIO COMPARISON:")
        print(f"  Best Fairness:  {comparison['best_scenario']} ({all_fairness[comparison['best_scenario']]['ratio']:.2f}x)")
        print(f"  Worst Fairness: {comparison['worst_scenario']} ({all_fairness[comparison['worst_scenario']]['ratio']:.2f}x)")
        
        # Show per-group scenario impacts
        print(f"\n📊 PERFORMANCE BY COHORT ACROSS SCENARIOS:")
        self._print_group_scenario_impacts(all_statistics, scenario_impacts)
        
        # Generate reports
        self._generate_report(all_statistics, all_fairness, comparison, scenario_impacts)
        self._create_visualization(all_statistics, all_fairness, comparison)
        
        print(f"\n📁 All results saved in: {self.results_dir}")
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(all_statistics)} scenarios analyzed")
    
    def _calculate_scenario_impacts(self, all_statistics):
        """Calculate impacts across scenarios."""
        impacts = {}
        
        if 'inference_only' not in all_statistics:
            return {}
        
        baseline = all_statistics['inference_only']
        
        for scenario_key, statistics in all_statistics.items():
            if scenario_key == 'inference_only':
                continue
            
            impacts[scenario_key] = {}
            
            for group_name in baseline.keys():
                if group_name not in statistics:
                    continue
                
                baseline_rmse = baseline[group_name]['rmse_mean']
                scenario_rmse = statistics[group_name]['rmse_mean']
                change = scenario_rmse - baseline_rmse
                percent_change = (change / baseline_rmse * 100) if baseline_rmse > 0 else 0
                
                if change < -0.3:
                    status = "IMPROVED"
                elif change < -0.1:
                    status = "Slightly Improved"
                elif change < 0.1:
                    status = "MAINTAINED"
                elif change < 0.3:
                    status = "Slightly Worse"
                else:
                    status = "WORSE"
                
                impacts[scenario_key][group_name] = {
                    'baseline_rmse': baseline_rmse,
                    'scenario_rmse': scenario_rmse,
                    'rmse_change': change,
                    'percent_change': percent_change,
                    'status': status
                }
        
        return impacts
    
    def _print_group_scenario_impacts(self, all_statistics, scenario_impacts):
        """Print per-group impacts across scenarios."""
        if not all_statistics or 'inference_only' not in all_statistics:
            return
        
        groups = list(all_statistics['inference_only'].keys())
        
        for group_name in sorted(groups):
            print(f"\n  {group_name}:")
            print(f"    Baseline (Inference Only): {all_statistics['inference_only'][group_name]['rmse_mean']:.3f} RMSE")
            
            for scenario_key in ['trained_standard', 'trained_noisy', 'trained_denoised']:
                if scenario_key in scenario_impacts and group_name in scenario_impacts[scenario_key]:
                    impact = scenario_impacts[scenario_key][group_name]
                    print(f"    {scenario_key.replace('_', ' ').title():22s}: {impact['scenario_rmse']:.3f} ({impact['rmse_change']:+.3f}, {impact['percent_change']:+.1f}%) - {impact['status']}")
    
    def _generate_report(self, all_statistics, all_fairness, comparison, scenario_impacts):
        """Generate JSON report."""
        timestamp = self.generate_timestamp()
        
        report = {
            "report_type": "Cohort Fairness Analysis - Inference Scenarios",
            "generated": timestamp.replace('_', ' '),
            "feature": self.feature_name,
            "total_patients": len(self.patient_data),
            "scenario_fairness": {},
            "groups": {},
            "scenario_comparison": {
                "best_scenario": comparison['best_scenario'],
                "worst_scenario": comparison['worst_scenario'],
                "fairness_ratios": {}
            },
            "per_group_impacts": {}
        }
        
        for scenario_key, fairness_info in all_fairness.items():
            report["scenario_fairness"][scenario_key] = {
                "fairness_ratio": round(fairness_info['ratio'], 4),
                "fairness_level": fairness_info['level']
            }
            report["scenario_comparison"]["fairness_ratios"][scenario_key] = round(fairness_info['ratio'], 4)
        
        for scenario_key, statistics in all_statistics.items():
            for group_name, stats in statistics.items():
                if group_name not in report["groups"]:
                    report["groups"][group_name] = {
                        "patient_count": stats['count'],
                        "scenarios": {}
                    }
                
                report["groups"][group_name]["scenarios"][scenario_key] = {
                    "rmse_mean": round(stats['rmse_mean'], 4),
                    "rmse_std": round(stats['rmse_std'], 4),
                    "mae_mean": round(stats['mae_mean'], 4),
                    "mae_std": round(stats['mae_std'], 4)
                }
        
        for scenario_key, scenario_impacts_data in scenario_impacts.items():
            report["per_group_impacts"][scenario_key] = {}
            for group_name, impact in scenario_impacts_data.items():
                report["per_group_impacts"][scenario_key][group_name] = {
                    "baseline_rmse": round(impact['baseline_rmse'], 4),
                    "scenario_rmse": round(impact['scenario_rmse'], 4),
                    "rmse_change": round(impact['rmse_change'], 4),
                    "percent_change": round(impact['percent_change'], 2),
                    "status": impact['status']
                }
        
        report_file = self.results_dir / f"cohort_inference_scenarios_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved: {report_file}")
    
    def _create_visualization(self, all_statistics, all_fairness, comparison):
        """Create comprehensive visualization."""
        timestamp = self.generate_timestamp()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cohort Fairness Analysis - Inference Scenarios Comparison', 
                     fontsize=16, fontweight='bold')
        
        scenarios = list(all_statistics.keys())
        scenario_labels = [self._format_scenario_label(s) for s in scenarios]
        
        # Plot 1: RMSE by scenario and age group
        ax1 = axes[0, 0]
        self._plot_multi_scenario_comparison(ax1, all_statistics, 'rmse_mean', 
                                             'RMSE by Scenario and Cohort', 'RMSE')
        
        # Plot 2: MAE by scenario and age group
        ax2 = axes[0, 1]
        self._plot_multi_scenario_comparison(ax2, all_statistics, 'mae_mean', 
                                             'MAE by Scenario and Cohort', 'MAE')
        
        # Plot 3: Fairness ratios
        ax3 = axes[1, 0]
        ratios = [all_fairness[s]['ratio'] for s in scenarios]
        colors = ['#2ecc71' if r < 1.1 else '#f39c12' if r < 1.25 else '#e74c3c' for r in ratios]
        bars = ax3.bar(scenario_labels, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_title('Fairness Ratios by Scenario', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Fairness Ratio (Worst/Best)', fontweight='bold')
        ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Fairness', alpha=0.7)
        ax3.axhline(y=1.25, color='orange', linestyle='--', linewidth=1.5, label='Acceptable Threshold', alpha=0.7)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, ratio, scenario in zip(bars, ratios, scenarios):
            height = bar.get_height()
            label = f'{ratio:.2f}x'
            if scenario == comparison['best_scenario']:
                label += '\n✓ Best'
            elif scenario == comparison['worst_scenario']:
                label += '\n✗ Worst'
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('Performance Summary by Scenario', fontweight='bold', fontsize=12, pad=20)
        
        # Get all age groups
        all_groups = set()
        for stats in all_statistics.values():
            all_groups.update(stats.keys())
        cohorts = sorted(list(all_groups))
        
        table_data = [['Scenario'] + [f'{g}\nRMSE' for g in cohorts] + ['Fairness\nRatio', 'Level']]
        for scenario_key in scenarios:
            stats = all_statistics[scenario_key]
            row = [self._format_scenario_label(scenario_key)]
            for group in cohorts:
                rmse = stats.get(group, {}).get('rmse_mean', 0)
                row.append(f"{rmse:.2f}")
            ratio = all_fairness[scenario_key]['ratio']
            level = all_fairness[scenario_key]['level']
            row.append(f"{ratio:.2f}x")
            row.append(level)
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0.05, 0.1, 0.9, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4a90e2')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.08)
        
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                table[(i, j)].set_height(0.06)
                
                if j == 0:
                    scenario_key = scenarios[i-1]
                    if scenario_key == comparison['best_scenario']:
                        table[(i, j)].set_facecolor('#d4edda')
                    elif scenario_key == comparison['worst_scenario']:
                        table[(i, j)].set_facecolor('#f8d7da')
        
        plt.tight_layout()
        
        plot_file = self.results_dir / f"cohort_inference_scenarios_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {plot_file}")
    
    def _format_scenario_label(self, scenario_key):
        """Format scenario key for display."""
        labels = {
            'inference_only': 'Inference\nOnly',
            'trained_standard': 'Trained\nStandard',
            'trained_noisy': 'Trained\nNoisy',
            'trained_denoised': 'Trained\nDenoised'
        }
        return labels.get(scenario_key, scenario_key.replace('_', '\n').title())
    
    def _plot_multi_scenario_comparison(self, ax, all_statistics, metric_key, title, ylabel):
        """Plot multi-scenario comparison."""
        scenarios = list(all_statistics.keys())
        scenario_labels = [self._format_scenario_label(s) for s in scenarios]
        
        all_groups = set()
        for stats in all_statistics.values():
            all_groups.update(stats.keys())
        cohorts = sorted(list(all_groups))
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        colors = {'2018': '#3498db', '2020': '#e74c3c'}
        
        for i, group in enumerate(cohorts):
            values = []
            for scenario in scenarios:
                stats = all_statistics[scenario]
                value = stats.get(group, {}).get(metric_key, 0)
                values.append(value)
            
            offset = width * (i - len(cohorts)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=group, 
                         color=colors.get(group, '#95a5a6'),
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', 
                           fontsize=7, fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')


def main():
    """Run age group fairness analysis for inference scenarios."""
    analyzer = InferenceCohortAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
