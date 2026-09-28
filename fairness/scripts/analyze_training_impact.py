#!/usr/bin/env python3
"""
Detailed Training Impact Analysis
Analyzes how training affects fairness for each scenario, feature, and group.
"""

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TrainingImpactAnalyzer:
    def __init__(self):
        self.results_dir = Path(__file__).parent.parent / "analysis_results"
        self.inference_dir = self.results_dir / "inference_scenarios"
        # Match constants from investigate_fairness_issues.py
        self.POOR_RATIO = 1.5  # Fairness ratio above this is POOR
        self.SIGNIFICANT_DEGRADATION = 0.1  # 10% degradation is significant
        
    def find_latest_report(self, directory: Path, pattern: str) -> Path:
        """Find the most recent report file."""
        reports = sorted(directory.glob(pattern))
        if not reports:
            raise FileNotFoundError(f"No reports found matching {pattern}")
        return reports[-1]
    
    def load_inference_data(self) -> Dict:
        """Load inference scenario data."""
        report_file = self.find_latest_report(
            self.inference_dir,
            "legendary_inference_scenarios_report_*.json"
        )
        print(f"Loading: {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def calculate_detailed_impact(self, data: Dict) -> Dict:
        """Calculate detailed training impact for each scenario."""
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        scenarios = ['trained_standard', 'trained_noisy', 'trained_denoised']
        
        impact_analysis = {
            'by_scenario': {},
            'by_feature': {},
            'by_group': []
        }
        
        for scenario in scenarios:
            impact_analysis['by_scenario'][scenario] = {
                'features': {},
                'avg_fairness_degradation': 0,
                'avg_fairness_degradation_significant_only': 0,  # Only features with degradation >= 0.1
                'avg_rmse_change': 0,
                'features_degraded': 0,
                'features_improved': 0,
                'features_significant_degradation': 0
            }
        
        # Analyze each feature
        for display_name, feature_key in feature_map.items():
            if display_name not in data['results']:
                continue
            
            feature_data = data['results'][display_name]
            inference_only = feature_data['inference_only']
            
            impact_analysis['by_feature'][feature_key] = {
                'scenarios': {}
            }
            
            for scenario in scenarios:
                if scenario not in feature_data:
                    continue
                
                scenario_data = feature_data[scenario]
                
                # Calculate fairness ratio change
                # Formula matches investigate_fairness_issues.py line 96 and 199
                fairness_change = scenario_data['fairness_ratio'] - inference_only['fairness_ratio']
                fairness_pct = (fairness_change / inference_only['fairness_ratio']) * 100
                
                # Store feature-level impact
                impact_analysis['by_feature'][feature_key]['scenarios'][scenario] = {
                    'inference_only_ratio': inference_only['fairness_ratio'],
                    'trained_ratio': scenario_data['fairness_ratio'],
                    'fairness_change': fairness_change,
                    'fairness_change_pct': fairness_pct,
                    'status': 'DEGRADED' if fairness_change > 0.05 else 'IMPROVED' if fairness_change < -0.05 else 'STABLE',
                    'groups': {}
                }
                
                # Update scenario-level statistics
                scenario_impact = impact_analysis['by_scenario'][scenario]
                scenario_impact['features'][feature_key] = fairness_pct
                scenario_impact['avg_fairness_degradation'] += fairness_pct
                
                # Track significant degradation (matching investigate_fairness_issues.py logic)
                if fairness_change >= self.SIGNIFICANT_DEGRADATION:
                    scenario_impact['features_significant_degradation'] += 1
                    scenario_impact['avg_fairness_degradation_significant_only'] += fairness_pct
                
                if fairness_change > 0.05:
                    scenario_impact['features_degraded'] += 1
                elif fairness_change < -0.05:
                    scenario_impact['features_improved'] += 1
                
                # Analyze per-group impact
                if 'groups' in inference_only and 'groups' in scenario_data:
                    for group_name in inference_only['groups'].keys():
                        if group_name in scenario_data['groups']:
                            inf_rmse = inference_only['groups'][group_name]['rmse_mean']
                            trained_rmse = scenario_data['groups'][group_name]['rmse_mean']
                            rmse_change = trained_rmse - inf_rmse
                            rmse_pct = (rmse_change / inf_rmse) * 100
                            
                            # Store group-level impact
                            impact_analysis['by_feature'][feature_key]['scenarios'][scenario]['groups'][group_name] = {
                                'inference_only_rmse': inf_rmse,
                                'trained_rmse': trained_rmse,
                                'rmse_change': rmse_change,
                                'rmse_change_pct': rmse_pct
                            }
                            
                            # Add to global group list
                            impact_analysis['by_group'].append({
                                'feature': feature_key,
                                'group': group_name,
                                'scenario': scenario,
                                'rmse_change': rmse_change,
                                'rmse_change_pct': rmse_pct,
                                'fairness_degradation': fairness_pct
                            })
                            
                            scenario_impact['avg_rmse_change'] += rmse_change
        
        # Calculate averages
        num_features = len(feature_map)
        for scenario in scenarios:
            if num_features > 0:
                impact_analysis['by_scenario'][scenario]['avg_fairness_degradation'] /= num_features
            # Calculate average for significant features only (matching chart in investigation report)
            if impact_analysis['by_scenario'][scenario]['features_significant_degradation'] > 0:
                impact_analysis['by_scenario'][scenario]['avg_fairness_degradation_significant_only'] /= \
                    impact_analysis['by_scenario'][scenario]['features_significant_degradation']
        
        return impact_analysis
    
    def print_detailed_report(self, impact: Dict):
        """Print detailed text report."""
        print("\n" + "=" * 80)
        print("DETAILED TRAINING IMPACT ANALYSIS")
        print("=" * 80)
        
        # 1. Scenario-level summary
        print("\n" + "=" * 80)
        print("1. SCENARIO-LEVEL IMPACT SUMMARY")
        print("=" * 80)
        print("\nNOTE: The chart in investigate_fairness_issues.py shows 'Significant Only' average")
        print("      which only includes features with degradation ≥ 0.1 (matches the threshold)")
        print("      for 'training_degradation' list used in the visual report.\n")
        print("      ★ = Feature had significant degradation (≥0.1) and is included in chart average")
        print("=" * 80)
        
        for scenario, data in sorted(impact['by_scenario'].items()):
            print(f"\n{scenario.upper().replace('_', ' ')}:")
            print(f"  Average Fairness Degradation (All Features): {data['avg_fairness_degradation']:+.1f}%")
            print(f"  Average Fairness Degradation (Significant Only, ≥0.1): {data['avg_fairness_degradation_significant_only']:+.1f}%")
            print(f"  Features with Significant Degradation (≥0.1): {data['features_significant_degradation']}")
            print(f"  Features Degraded (>0.05): {data['features_degraded']}")
            print(f"  Features Improved (<-0.05): {data['features_improved']}")
            print(f"  Features Stable: {5 - data['features_degraded'] - data['features_improved']}")
            
            if data['features']:
                print(f"\n  Per-Feature Fairness Impact:")
                # Mark which features had significant degradation
                for feature, impact_pct in sorted(data['features'].items(), 
                                                 key=lambda x: abs(x[1]), reverse=True):
                    # Check if this feature had degradation >= 0.1
                    inference_ratio = None
                    trained_ratio = None
                    for fname, fdata in impact['by_feature'].items():
                        if fname == feature and scenario in fdata['scenarios']:
                            inference_ratio = fdata['scenarios'][scenario]['inference_only_ratio']
                            trained_ratio = fdata['scenarios'][scenario]['trained_ratio']
                            break
                    
                    significant_marker = ""
                    if inference_ratio and trained_ratio:
                        degradation = trained_ratio - inference_ratio
                        if degradation >= self.SIGNIFICANT_DEGRADATION:
                            significant_marker = " ★"  # Mark significant degradation
                    
                    status = "⬆️ WORSE" if impact_pct > 5 else "⬇️ BETTER" if impact_pct < -5 else "≈ STABLE"
                    print(f"    {feature.replace('_', ' ').title():20s}: {impact_pct:+6.1f}%  {status}{significant_marker}")
        
        # 2. Feature-level analysis
        print("\n" + "=" * 80)
        print("2. FEATURE-LEVEL ANALYSIS")
        print("=" * 80)
        
        for feature, data in sorted(impact['by_feature'].items()):
            print(f"\n{feature.upper().replace('_', ' ')}:")
            
            for scenario, scenario_data in sorted(data['scenarios'].items()):
                print(f"\n  {scenario.replace('_', ' ').title()}:")
                print(f"    Fairness Ratio: {scenario_data['inference_only_ratio']:.3f} → {scenario_data['trained_ratio']:.3f}")
                print(f"    Change: {scenario_data['fairness_change']:+.3f} ({scenario_data['fairness_change_pct']:+.1f}%)")
                print(f"    Status: {scenario_data['status']}")
                
                if scenario_data['groups']:
                    print(f"    Per-Group RMSE Impact:")
                    for group, group_data in sorted(scenario_data['groups'].items(),
                                                   key=lambda x: abs(x[1]['rmse_change']), reverse=True):
                        print(f"      {group:15s}: {group_data['inference_only_rmse']:.2f} → {group_data['trained_rmse']:.2f} "
                              f"(Δ {group_data['rmse_change']:+.2f}, {group_data['rmse_change_pct']:+.1f}%)")
        
        # 3. Most affected groups
        print("\n" + "=" * 80)
        print("3. TOP 10 MOST AFFECTED GROUPS")
        print("=" * 80)
        
        # Sort by absolute fairness degradation
        worst_groups = sorted(impact['by_group'], 
                             key=lambda x: abs(x['fairness_degradation']), reverse=True)[:10]
        
        for i, group_data in enumerate(worst_groups, 1):
            print(f"\n{i}. {group_data['feature'].replace('_', ' ').title()} - {group_data['group']}")
            print(f"   Scenario: {group_data['scenario'].replace('_', ' ').title()}")
            print(f"   RMSE Change: {group_data['rmse_change']:+.2f} ({group_data['rmse_change_pct']:+.1f}%)")
            print(f"   Feature Fairness Impact: {group_data['fairness_degradation']:+.1f}%")
    
    def generate_visual_report(self, impact: Dict):
        """Generate visual report."""
        print("\n" + "=" * 80)
        print("GENERATING VISUAL REPORT")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"training_impact_detailed_{timestamp}.png"
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
        
        fig.suptitle('DETAILED TRAINING IMPACT ON FAIRNESS', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Scenario comparison (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_scenario_comparison(ax1, impact)
        
        # 2. Feature impact heatmap (top middle and right, span 2)
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_feature_heatmap(ax2, impact)
        
        # 3. Per-feature breakdown for each scenario (middle row)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_scenario_breakdown(ax3, impact, 'trained_standard')
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_scenario_breakdown(ax4, impact, 'trained_noisy')
        
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_scenario_breakdown(ax5, impact, 'trained_denoised')
        
        # 4. Top affected groups (bottom row, full width)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_top_affected_groups(ax6, impact)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisual report saved: {output_file}")
        return output_file
    
    def _plot_scenario_comparison(self, ax, impact: Dict):
        """Plot scenario comparison - matches investigate_fairness_issues.py logic."""
        scenarios = list(impact['by_scenario'].keys())
        # Use "significant only" average to match the investigation report chart
        avg_degradation = [impact['by_scenario'][s]['avg_fairness_degradation_significant_only'] for s in scenarios]
        degraded = [impact['by_scenario'][s]['features_degraded'] for s in scenarios]
        improved = [impact['by_scenario'][s]['features_improved'] for s in scenarios]
        significant_count = [impact['by_scenario'][s]['features_significant_degradation'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.6
        
        # Main bar
        colors = ['#e74c3c' if d > 20 else '#f39c12' if d > 10 else '#3498db' for d in avg_degradation]
        bars = ax.bar(x, avg_degradation, width, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, avg_degradation):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Avg Fairness Degradation (%)', fontweight='bold')
        ax.set_title('Average Impact by Scenario\n(Features with degradation ≥0.1 only)', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n').title() for s in scenarios], fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add text annotation showing count of significant features
        for i, (deg, imp, sig) in enumerate(zip(degraded, improved, significant_count)):
            ax.text(i, min(avg_degradation) - 5, f'★{sig}', 
                   ha='center', fontsize=9, color='gray', fontweight='bold')
    
    def _plot_feature_heatmap(self, ax, impact: Dict):
        """Plot feature × scenario heatmap."""
        features = list(impact['by_feature'].keys())
        scenarios = ['trained_standard', 'trained_noisy', 'trained_denoised']
        
        # Create matrix
        matrix = np.zeros((len(features), len(scenarios)))
        
        for i, feature in enumerate(features):
            for j, scenario in enumerate(scenarios):
                if scenario in impact['by_feature'][feature]['scenarios']:
                    matrix[i, j] = impact['by_feature'][feature]['scenarios'][scenario]['fairness_change_pct']
        
        # Plot
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-20, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels([s.replace('_', '\n').title() for s in scenarios], fontsize=10)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
        
        # Add values
        for i in range(len(features)):
            for j in range(len(scenarios)):
                text_color = 'white' if abs(matrix[i, j]) > 40 else 'black'
                ax.text(j, i, f'{matrix[i, j]:+.0f}%',
                       ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
        
        ax.set_title('Fairness Degradation by Feature & Scenario (%)', fontweight='bold', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fairness Change (%)', rotation=270, labelpad=20, fontweight='bold')
    
    def _plot_scenario_breakdown(self, ax, impact: Dict, scenario: str):
        """Plot detailed breakdown for one scenario."""
        features = list(impact['by_feature'].keys())
        values = []
        
        for feature in features:
            if scenario in impact['by_feature'][feature]['scenarios']:
                values.append(impact['by_feature'][feature]['scenarios'][scenario]['fairness_change_pct'])
            else:
                values.append(0)
        
        colors = ['#e74c3c' if v > 20 else '#f39c12' if v > 10 else '#2ecc71' if v < -5 else '#3498db' for v in values]
        
        bars = ax.barh([f.replace('_', '\n').title() for f in features], values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            x_pos = width + (1 if width > 0 else -1)
            ha = 'left' if width > 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
                   va='center', ha=ha, fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Fairness Change (%)', fontweight='bold', fontsize=10)
        ax.set_title(scenario.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        max_abs = max(abs(min(values)), abs(max(values)))
        ax.set_xlim(-max_abs*1.3, max_abs*1.3)
    
    def _plot_top_affected_groups(self, ax, impact: Dict):
        """Plot top affected groups."""
        # Get top 15 worst affected
        worst_groups = sorted(impact['by_group'], 
                             key=lambda x: x['fairness_degradation'], reverse=True)[:15]
        
        # Create compact labels: "Feature: Group (Scenario)"
        labels = [f"{g['feature'].replace('_', ' ').title()}: {g['group']} ({g['scenario'].replace('trained_', '').title()})" 
                 for g in worst_groups]
        values = [g['fairness_degradation'] for g in worst_groups]
        rmse_changes = [g['rmse_change'] for g in worst_groups]
        
        # Color by severity
        colors = ['#e74c3c' if v > 50 else '#f39c12' if v > 20 else '#3498db' for v in values]
        
        bars = ax.barh(labels, values, color=colors, alpha=0.8)
        
        # Add RMSE change labels
        for bar, val, rmse in zip(bars, values, rmse_changes):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:+.1f}% (RMSE: {rmse:+.1f})',
                   va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Fairness Degradation (%)', fontweight='bold', fontsize=11)
        ax.set_title('Top 15 Most Affected Groups', fontweight='bold', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Adjust y-axis labels to prevent overlap
        ax.tick_params(axis='y', labelsize=8)
    
    def run(self):
        """Run complete analysis."""
        print("\nDETAILED TRAINING IMPACT ANALYZER")
        print("Analyzing how training affects fairness across scenarios, features, and groups\n")
        
        try:
            # Load data
            data = self.load_inference_data()
            
            # Calculate detailed impact
            print("\nCalculating detailed impact...")
            impact = self.calculate_detailed_impact(data)
            
            # Print detailed report
            self.print_detailed_report(impact)
            
            # Generate visual report
            visual_report = self.generate_visual_report(impact)
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE!")
            print(f"Visual report: {visual_report}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            raise

if __name__ == "__main__":
    analyzer = TrainingImpactAnalyzer()
    analyzer.run()
