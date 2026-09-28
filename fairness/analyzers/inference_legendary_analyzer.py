#!/usr/bin/env python3
"""
🏆 LEGENDARY INFERENCE SCENARIOS ANALYZER 🏆

Comprehensive fairness analysis across ALL demographic features and ALL scenarios:
Features:
- Gender (Male/Female)
- Age Groups (20-40, 40-60, 60-80)
- Pump Models (630G, 530G)
- Sensor Bands (Empatica, Basis)
- Study Cohorts (2020, 2018)

Scenarios:
- Inference Only (no training)
- Trained on Standard Data
- Trained on Noisy Data
- Trained on Denoised Data

Shows which groups perform best/worst across different data conditions.
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from fairness.analyzers.base_inference_analyzer import BaseInferenceAnalyzer


class LegendaryInferenceAnalyzer(BaseInferenceAnalyzer):
    """Analyze fairness across ALL features and ALL scenarios."""
    
    def __init__(self, data_path=None):
        super().__init__(feature_name="ALL FEATURES", data_path=data_path or "")
        self.features = {
            'Gender': self._load_gender_data(),
            'Age Group': self._load_age_data(),
            'Pump Model': self._load_pump_data(),
            'Sensor Band': self._load_sensor_data(),
            'Cohort': self._load_cohort_data()
        }
        print(f"🏆 LEGENDARY INFERENCE ANALYZER LOADED: {len(self.patient_data)} patients")
    
    def _load_gender_data(self) -> Dict:
        return {pid: data['gender'] for pid, data in self.patient_data.items()}
    
    def _load_age_data(self) -> Dict:
        return {pid: data['age'] for pid, data in self.patient_data.items()}
    
    def _load_pump_data(self) -> Dict:
        return {pid: data['pump_model'] for pid, data in self.patient_data.items()}
    
    def _load_sensor_data(self) -> Dict:
        return {pid: data['sensor_band'] for pid, data in self.patient_data.items()}
    
    def _load_cohort_data(self) -> Dict:
        return {pid: data['cohort'] for pid, data in self.patient_data.items()}
    
    def get_feature_value(self, patient_id: str) -> str:
        """Not used in legendary analyzer."""
        return ""
    
    def group_by_custom_feature(self, scenario_results: Dict, feature_data: Dict) -> Dict:
        """Group patients by a custom feature."""
        groups = defaultdict(list)
        
        for patient_id, metrics in scenario_results.items():
            if patient_id in feature_data:
                feature_value = feature_data[patient_id]
                groups[feature_value].append({
                    'patient_id': patient_id,
                    **metrics
                })
        
        return dict(groups)
    
    def analyze(self):
        """Run comprehensive fairness analysis across ALL features and ALL scenarios."""
        print("\n" + "="*80)
        print("🏆 LEGENDARY INFERENCE SCENARIOS FAIRNESS ANALYSIS")
        print("="*80)
        print("Analyzing ALL demographic features across ALL inference scenarios")
        print("="*80)
        
        # Load all scenario results
        all_scenarios = self.load_all_scenarios()
        
        # Store results for all features and scenarios
        mega_results = {}
        
        for feature_name, feature_data in self.features.items():
            print(f"\n{'='*80}")
            print(f"📊 {feature_name.upper()} ANALYSIS")
            print(f"{'='*80}")
            
            feature_results = {}
            
            for scenario_key, results in all_scenarios.items():
                if not results:
                    continue
                
                grouped = self.group_by_custom_feature(results, feature_data)
                statistics = self.calculate_group_statistics(grouped)
                ratio, level = self.calculate_fairness_ratio(statistics)
                
                feature_results[scenario_key] = {
                    'statistics': statistics,
                    'ratio': ratio,
                    'level': level
                }
                
                print(f"\n  {scenario_key.upper().replace('_', ' ')}:")
                print(f"    Fairness Ratio: {ratio:.2f}x ({level})")
                for group_name, stats in statistics.items():
                    print(f"    {group_name}: RMSE={stats['rmse_mean']:.2f}")
            
            mega_results[feature_name] = feature_results
            
            # Find best and worst scenario for this feature
            valid_ratios = {k: v['ratio'] for k, v in feature_results.items() if v['ratio'] > 0}
            if valid_ratios:
                best = min(valid_ratios, key=lambda x: valid_ratios[x])
                worst = max(valid_ratios, key=lambda x: valid_ratios[x])
                print(f"\n  ✅ Best Scenario: {best} ({valid_ratios[best]:.2f}x)")
                print(f"  ⚠️  Worst Scenario: {worst} ({valid_ratios[worst]:.2f}x)")
        
        # Generate comprehensive summary
        self._print_legendary_summary(mega_results)
        
        # Generate reports and visualizations (matching distillation style)
        self._generate_legendary_report(mega_results)
        self._create_legendary_visualization(mega_results)
        self._save_summary_table(mega_results, timestamp=self.generate_timestamp())
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 All results saved in: {self.results_dir}")
    
    def _print_legendary_summary(self, mega_results: Dict):
        """Print comprehensive summary."""
        print(f"\n{'='*80}")
        print("🏆 LEGENDARY SUMMARY: FAIRNESS ACROSS ALL FEATURES & SCENARIOS")
        print(f"{'='*80}")
        
        print(f"\n{'Feature':<15} | {'Inference Only':<12} | {'Trained Std':<12} | {'Trained Noisy':<14} | {'Trained Denoised':<16}")
        print("-" * 80)
        
        for feature_name, results in mega_results.items():
            row = [feature_name[:15]]
            
            for scenario_key in ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']:
                if scenario_key in results and results[scenario_key]['ratio'] > 0:
                    ratio = results[scenario_key]['ratio']
                    level = results[scenario_key]['level']
                    row.append(f"{ratio:.2f}x ({level[:4]})")
                else:
                    row.append("N/A")
            
            print(f"{row[0]:<15} | {row[1]:<12} | {row[2]:<12} | {row[3]:<14} | {row[4]:<16}")
        
        # Overall best/worst scenarios
        print(f"\n{'='*80}")
        print("OVERALL SCENARIO RANKING (Lower = More Fair)")
        print(f"{'='*80}")
        
        scenario_avg_ratios = {}
        for scenario_key in ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']:
            ratios = []
            for feature_results in mega_results.values():
                if scenario_key in feature_results and feature_results[scenario_key]['ratio'] > 0:
                    ratios.append(feature_results[scenario_key]['ratio'])
            
            if ratios:
                scenario_avg_ratios[scenario_key] = np.mean(ratios)
        
        sorted_scenarios = sorted(scenario_avg_ratios.items(), key=lambda x: x[1])
        
        for i, (scenario, avg_ratio) in enumerate(sorted_scenarios, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{icon} {i}. {scenario.replace('_', ' ').title():30s}: {avg_ratio:.3f}x average")
    
    def _generate_legendary_report(self, mega_results: Dict):
        """Generate comprehensive JSON report."""
        timestamp = self.generate_timestamp()
        
        report = {
            "report_type": "Comprehensive Inference Scenarios Fairness Analysis",
            "generated": timestamp.replace('_', ' '),
            "total_patients": len(self.patient_data),
            "features_analyzed": list(self.features.keys()),
            "scenarios_analyzed": list(self.scenarios.keys()),
            "results": {}
        }
        
        for feature_name, results in mega_results.items():
            report["results"][feature_name] = {}
            
            for scenario_key, scenario_data in results.items():
                report["results"][feature_name][scenario_key] = {
                    "fairness_ratio": round(scenario_data['ratio'], 4),
                    "fairness_level": scenario_data['level'],
                    "groups": {}
                }
                
                for group_name, stats in scenario_data['statistics'].items():
                    report["results"][feature_name][scenario_key]["groups"][group_name] = {
                        "count": stats['count'],
                        "rmse_mean": round(stats['rmse_mean'], 4),
                        "rmse_std": round(stats['rmse_std'], 4),
                        "mae_mean": round(stats['mae_mean'], 4),
                        "mae_std": round(stats['mae_std'], 4)
                    }
        
        report_file = self.results_dir / f"legendary_inference_scenarios_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved: {report_file}")
        return report_file
    
    def _create_legendary_visualization(self, mega_results: Dict):
        """Create comprehensive visualization with RMSE comparisons and embedded summary table."""
        timestamp = self.generate_timestamp()
        
        # Create large figure with proper layout: 1 heatmap + 2 rows for RMSE charts + 1 table
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.4, top=0.94, height_ratios=[1, 1, 1, 1.2])
        
        fig.suptitle('📊 COMPREHENSIVE: Fairness Across All Features & Scenarios', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        features = list(mega_results.keys())
        scenarios = ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']
        scenario_labels = ['Inference\nOnly', 'Trained\nStandard', 'Trained\nNoisy', 'Trained\nDenoised']
        
        # Plot 1: Heatmap of fairness ratios (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_fairness_heatmap(ax1, mega_results, features, scenarios, scenario_labels)
        
        # Plots 2-6: RMSE comparisons for each feature (2 rows x 3 columns)
        feature_positions = [
            (1, 0), (1, 1), (1, 2),  # Row 2
            (2, 0), (2, 1)            # Row 3 (only 5 features)
        ]
        
        for idx, feature_name in enumerate(features):
            if idx >= len(feature_positions):
                break
            
            row, col = feature_positions[idx]
            ax = fig.add_subplot(gs[row, col])
            self._plot_rmse_comparison(ax, mega_results[feature_name], feature_name, 
                                      scenarios, scenario_labels)
        
        # Add summary table at the bottom (full width)
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')
        ax_table.set_title('Summary Table: Per-Group RMSE Performance Across Scenarios', 
                          fontweight='bold', fontsize=14, pad=15)
        
        # Prepare table data - show RMSE with color-coded impact indicators
        table_data = []
        impact_colors: list[list[str | None]] = []  # Track colors for each cell
        
        table_data.append(['Feature', 'Group', 'Inf Only\nRMSE', 'Trained Std\nRMSE (Δ)', 
                          'Trained Noisy\nRMSE (Δ)', 'Trained Denoised\nRMSE (Δ)', 'Best Scenario'])
        impact_colors.append([None] * 7)  # Header row has no special colors
        
        for feature_name in features:
            # Get all groups for this feature
            all_groups = set()
            for scenario_data in mega_results[feature_name].values():
                if 'statistics' in scenario_data:
                    all_groups.update(scenario_data['statistics'].keys())
            
            for group_name in sorted(all_groups):
                row = [feature_name, group_name]
                row_colors: list[str | None] = [None, None]  # Feature and Group columns
                rmse_values = {}
                inference_only_rmse = None
                
                # First add inference_only RMSE
                if 'inference_only' in mega_results[feature_name]:
                    stats = mega_results[feature_name]['inference_only'].get('statistics', {})
                    if group_name in stats:
                        inference_only_rmse = stats[group_name]['rmse_mean']
                        row.append(f"{inference_only_rmse:.2f}")
                        row_colors.append(None)  # Baseline - no color
                        rmse_values['inference_only'] = inference_only_rmse
                    else:
                        row.append('N/A')
                        row_colors.append(None)
                else:
                    row.append('N/A')
                    row_colors.append(None)
                
                # Add each training scenario with impact embedded in the text
                training_scenarios = ['trained_standard', 'trained_noisy', 'trained_denoised']
                for scenario in training_scenarios:
                    if scenario in mega_results[feature_name]:
                        stats = mega_results[feature_name][scenario].get('statistics', {})
                        if group_name in stats:
                            rmse = stats[group_name]['rmse_mean']
                            rmse_values[scenario] = rmse
                            
                            # Calculate impact vs inference_only
                            if inference_only_rmse is not None:
                                impact = rmse - inference_only_rmse
                                impact_pct = (impact / inference_only_rmse) * 100
                                
                                # Format cell with RMSE and impact inline
                                if impact < -0.5:
                                    row.append(f"{rmse:.2f} (✓ {impact:.1f})")
                                    row_colors.append('good')  # Green - improvement
                                elif impact > 0.5:
                                    row.append(f"{rmse:.2f} (✗ +{impact:.1f})")
                                    row_colors.append('bad')  # Red - degradation
                                else:
                                    row.append(f"{rmse:.2f} (≈ {impact:+.1f})")
                                    row_colors.append('neutral')  # Yellow - similar
                            else:
                                row.append(f"{rmse:.2f}")
                                row_colors.append(None)
                        else:
                            row.append('N/A')
                            row_colors.append(None)
                    else:
                        row.append('N/A')
                        row_colors.append(None)
                
                # Find best scenario for this group
                if rmse_values:
                    best_scenario = min(rmse_values.keys(), key=lambda k: rmse_values[k])
                    best_rmse = rmse_values[best_scenario]
                    row.append(f"{best_scenario.replace('_', ' ').title()} ({best_rmse:.1f})")
                else:
                    row.append('N/A')
                
                row_colors.append('best')  # Best scenario column
                table_data.append(row)
                impact_colors.append(row_colors)
        
        # Create table
        table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                              bbox=[0.05, 0.1, 0.9, 0.8])  # type: ignore[arg-type]
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        
        # Define color mappings
        color_map = {
            'good': '#c8e6c9',      # Light green - improvement
            'bad': '#ffcdd2',       # Light red - degradation
            'neutral': '#fff9c4',   # Light yellow - similar
            'best': '#a5d6a7',      # Medium green - best scenario
        }
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4a90e2')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=7)
            table[(0, i)].set_height(0.06)
        
        # Style data rows with color coding based on impact
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                # Set base background
                if i % 2 == 0:
                    base_color = '#f8f9fa'
                else:
                    base_color = 'white'
                
                # Apply impact-based coloring
                color_key = impact_colors[i][j]
                if color_key is not None and color_key in color_map:
                    table[(i, j)].set_facecolor(color_map[color_key])
                else:
                    table[(i, j)].set_facecolor(base_color)
                
                table[(i, j)].set_height(0.05)
        
        # Save single combined visualization
        plot_file = self.results_dir / f"legendary_inference_scenarios_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {plot_file}")
    
    def _plot_fairness_heatmap(self, ax, mega_results, features, scenarios, scenario_labels):
        """Plot heatmap of fairness ratios."""
        # Create matrix
        matrix = []
        for feature in features:
            row = []
            for scenario in scenarios:
                if scenario in mega_results[feature]:
                    row.append(mega_results[feature][scenario]['ratio'])
                else:
                    row.append(0)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=1.5)
        
        # Set ticks
        ax.set_xticks(np.arange(len(scenario_labels)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(scenario_labels)
        ax.set_yticklabels(features)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add values in cells
        for i in range(len(features)):
            for j in range(len(scenarios)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Fairness Ratios: Feature × Scenario\n(Lower = More Fair)', 
                    fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fairness Ratio', rotation=270, labelpad=20, fontweight='bold')
    
    def _plot_rmse_comparison(self, ax, feature_results, feature_name, scenarios, scenario_labels):
        """Plot RMSE comparison across groups for all scenarios."""
        # Get all groups for this feature
        all_groups = set()
        for scenario_data in feature_results.values():
            if 'statistics' in scenario_data:
                all_groups.update(scenario_data['statistics'].keys())
        
        groups = sorted(list(all_groups))
        
        # Prepare data
        x = np.arange(len(scenarios))
        width = 0.8 / len(groups) if groups else 0.8
        
        # Define colors for groups
        group_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, group in enumerate(groups):
            rmse_values = []
            for scenario in scenarios:
                if scenario in feature_results and 'statistics' in feature_results[scenario]:
                    stats = feature_results[scenario]['statistics']
                    if group in stats:
                        rmse_values.append(stats[group]['rmse_mean'])
                    else:
                        rmse_values.append(0)
                else:
                    rmse_values.append(0)
            
            offset = width * (idx - len(groups)/2 + 0.5)
            color = group_colors[idx % len(group_colors)]
            bars = ax.bar(x + offset, rmse_values, width, label=group,
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom',
                           fontsize=8, fontweight='bold')
        
        ax.set_title(f'{feature_name} - RMSE Comparison', fontweight='bold', fontsize=12)
        ax.set_ylabel('RMSE', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _save_summary_table(self, mega_results: Dict, timestamp: str):
        """Save summary table with performance across all scenarios (like distillation CSV)."""
        import csv
        
        table_file = self.results_dir / f"legendary_summary_table_inference_{timestamp}.csv"
        
        with open(table_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Feature', 'Group', 'Inference Only RMSE', 'Trained Standard RMSE', 
                           'Trained Noisy RMSE', 'Trained Denoised RMSE', 
                           'Best Scenario', 'Worst Scenario', 'Fairness Range'])
            
            # Collect all rows
            for feature_name, results in mega_results.items():
                # Get all groups for this feature
                all_groups = set()
                for scenario_data in results.values():
                    all_groups.update(scenario_data['statistics'].keys())
                
                for group_name in sorted(all_groups):
                    row = [feature_name, group_name]
                    
                    # Add RMSE for each scenario
                    rmse_values = {}
                    for scenario_key in ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']:
                        if scenario_key in results:
                            stats = results[scenario_key]['statistics']
                            if group_name in stats:
                                rmse = stats[group_name]['rmse_mean']
                                row.append(f"{rmse:.4f}")
                                rmse_values[scenario_key] = rmse
                            else:
                                row.append("N/A")
                        else:
                            row.append("N/A")
                    
                    # Find best and worst for this group
                    if rmse_values:
                        best_scenario = min(rmse_values.keys(), key=lambda k: rmse_values[k])
                        worst_scenario = max(rmse_values.keys(), key=lambda k: rmse_values[k])
                        fairness_range = max(rmse_values.values()) - min(rmse_values.values())
                        
                        row.append(best_scenario)
                        row.append(worst_scenario)
                        row.append(f"{fairness_range:.4f}")
                    else:
                        row.extend(["N/A", "N/A", "N/A"])
                    
                    writer.writerow(row)
        
        print(f"📊 Summary table saved: {table_file}")


def main():
    """Run legendary inference scenarios fairness analysis."""
    analyzer = LegendaryInferenceAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
