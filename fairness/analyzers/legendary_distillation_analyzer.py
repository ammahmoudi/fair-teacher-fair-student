#!/usr/bin/env python3
"""
🏆 LEGENDARY DISTILLATION IMPACT ANALYZER 🏆

Comprehensive analyzer showing distillation impact across ALL demographic features:
- Gender (Male/Female)
- Age Groups (20-40, 40-60, 60-80)
- Pump Models (630G, 530G)
- Sensor Bands (Empatica, Basis)
- Study Cohorts (2020, 2018)

Shows which groups got better or worse after distillation.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add project root to path dynamically
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from fairness.analyzers.base_analyzer import BaseFairnessAnalyzer
from fairness.utils.analyzer_utils import (
    analyze_per_group_distillation_impact,
    format_report_header
)


class LegendaryDistillationAnalyzer(BaseFairnessAnalyzer):
    """Analyze distillation impact across ALL demographic features."""
    
    def __init__(self, data_path=None, experiment_type="per_patient", experiments_folder="distillation_experiments"):
        super().__init__(feature_name="ALL FEATURES", data_path=data_path, 
                        experiment_type=experiment_type, experiments_folder=experiments_folder)
        self.features = {
            'Gender': self._load_gender_data(),
            'Age Group': self._load_age_data(),
            'Pump Model': self._load_pump_data(),
            'Sensor Band': self._load_sensor_data(),
            'Cohort': self._load_cohort_data()
        }
        print(f"🏆 LEGENDARY ANALYZER LOADED: {len(self.patient_data)} patients")
    
    def _load_patient_data(self) -> Dict:
        """Load comprehensive patient data."""
        try:
            df = pd.read_csv(self.data_path)
            patient_mapping = {}
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                patient_mapping[patient_id] = {
                    'gender': row['Gender'],
                    'age': row['Age'],
                    'pump_model': row['Pump Model'],
                    'sensor_band': row['Sensor Band'],
                    'cohort': str(row['Cohort'])
                }
            return patient_mapping
        except Exception as e:
            print(f"⚠️  Using default patient data")
            return self._get_default_patient_data()
    
    def _get_default_patient_data(self) -> Dict:
        """Default OhioT1DM patient data."""
        return {
            '540': {'gender': 'Male', 'age': '40-60', 'pump_model': '630G', 'sensor_band': 'Empatica', 'cohort': '2018'},
            '544': {'gender': 'Male', 'age': '20-40', 'pump_model': '630G', 'sensor_band': 'Empatica', 'cohort': '2018'},
            '552': {'gender': 'Male', 'age': '40-60', 'pump_model': '630G', 'sensor_band': 'Basis', 'cohort': '2018'},
            '559': {'gender': 'Male', 'age': '20-40', 'pump_model': '630G', 'sensor_band': 'Empatica', 'cohort': '2018'},
            '563': {'gender': 'Female', 'age': '40-60', 'pump_model': '630G', 'sensor_band': 'Basis', 'cohort': '2018'},
            '567': {'gender': 'Female', 'age': '60-80', 'pump_model': '530G', 'sensor_band': 'Empatica', 'cohort': '2018'},
            '570': {'gender': 'Male', 'age': '40-60', 'pump_model': '630G', 'sensor_band': 'Basis', 'cohort': '2020'},
            '575': {'gender': 'Female', 'age': '20-40', 'pump_model': '630G', 'sensor_band': 'Empatica', 'cohort': '2020'},
            '584': {'gender': 'Male', 'age': '20-40', 'pump_model': '630G', 'sensor_band': 'Empatica', 'cohort': '2020'},
            '588': {'gender': 'Female', 'age': '60-80', 'pump_model': '630G', 'sensor_band': 'Basis', 'cohort': '2020'},
            '591': {'gender': 'Female', 'age': '60-80', 'pump_model': '630G', 'sensor_band': 'Basis', 'cohort': '2020'},
            '596': {'gender': 'Male', 'age': '60-80', 'pump_model': '530G', 'sensor_band': 'Basis', 'cohort': '2020'}
        }
    
    def _load_gender_data(self) -> Dict:
        """Extract gender mapping."""
        return {pid: data['gender'] for pid, data in self.patient_data.items()}
    
    def _load_age_data(self) -> Dict:
        """Extract age group mapping."""
        return {pid: data['age'] for pid, data in self.patient_data.items()}
    
    def _load_pump_data(self) -> Dict:
        """Extract pump model mapping."""
        return {pid: data['pump_model'] for pid, data in self.patient_data.items()}
    
    def _load_sensor_data(self) -> Dict:
        """Extract sensor band mapping."""
        return {pid: data['sensor_band'] for pid, data in self.patient_data.items()}
    
    def _load_cohort_data(self) -> Dict:
        """Extract cohort mapping."""
        return {pid: data['cohort'] for pid, data in self.patient_data.items()}
    
    def analyze_latest(self):
        """COMPREHENSIVE DISTILLATION IMPACT ANALYSIS."""
        print("\nCOMPREHENSIVE DISTILLATION IMPACT ANALYZER")
        print("=" * 70)
        print("Analyzing distillation impact across ALL demographic features")
        print("=" * 70)
        
        # Find and load latest experiment
        experiment_path = self.find_latest_experiment()
        
        # Load model names for visualization
        self.load_model_names(Path(experiment_path))
        
        patient_results = self.load_patient_results(experiment_path)
        
        # Analyze each feature
        all_feature_results = {}
        all_feature_impacts = {}
        
        for feature_name, feature_data in self.features.items():
            print(f"\n{'='*70}")
            print(f"{feature_name.upper()} ANALYSIS")
            print(f"{'='*70}")
            
            # Group by this feature
            groups = self._group_by_custom_feature(patient_results, feature_data)
            
            # Calculate statistics
            statistics = self.calculate_group_statistics(groups)
            
            # Print multi-phase summary
            from fairness.utils.analyzer_utils import print_multi_phase_summary
            for group_name, stats in statistics.items():
                print_multi_phase_summary(group_name, stats)
            
            # Calculate fairness for each phase
            teacher_ratio, teacher_level = self.calculate_fairness_ratio(statistics, 'teacher')
            distilled_ratio, distilled_level = self.calculate_fairness_ratio(statistics, 'distilled')
            
            print(f"\nFAIRNESS BY PHASE:")
            print(f"  Teacher:   {teacher_ratio:.2f}x ({teacher_level})")
            print(f"  Distilled: {distilled_ratio:.2f}x ({distilled_level})")
            
            # Overall impact
            impact = self.analyze_distillation_impact(statistics)
            print(f"\nOVERALL DISTILLATION IMPACT:")
            print(f"  {impact['conclusion']}")
            print(f"  Change: {impact['change']:+.2f}x ({impact['percent_change']:+.1f}%)")
            
            # Per-group impact
            group_impacts = analyze_per_group_distillation_impact(statistics)
            print(f"\nIMPACT BY {feature_name.upper()}:")
            for group_name, group_impact in group_impacts.items():
                print(f"  {group_name}:")
                print(f"    Teacher → Distilled: {group_impact['teacher_rmse']:.3f} → {group_impact['distilled_rmse']:.3f}")
                print(f"    Change: {group_impact['rmse_change']:+.3f} ({group_impact['percent_change']:+.1f}%)")
                print(f"    Status: {group_impact['status']}")
            
            all_feature_results[feature_name] = {
                'statistics': statistics,
                'impact': impact,
                'teacher_ratio': teacher_ratio,
                'distilled_ratio': distilled_ratio
            }
            all_feature_impacts[feature_name] = group_impacts
        
        # LEGENDARY SUMMARY
        self._print_legendary_summary(all_feature_results, all_feature_impacts)
        
        # Generate comprehensive report
        self._generate_legendary_report(all_feature_results, all_feature_impacts)
        
        # Create mega visualization
        self._create_legendary_visualization(all_feature_results, all_feature_impacts)
        
        print(f"\nAll results saved in: {self.results_dir}")
        print(f"\nANALYSIS COMPLETE!")
    
    def _group_by_custom_feature(self, patient_results: Dict, feature_data: Dict) -> Dict:
        """Group patients by a custom feature."""
        groups = defaultdict(list)
        for patient_id, results in patient_results.items():
            if patient_id in feature_data:
                feature_value = feature_data[patient_id]
                # Store in format expected by base class: list of dicts with patient_id and results
                patient_entry = {
                    'patient_id': patient_id,
                    **results  # Unpack all phase results
                }
                groups[feature_value].append(patient_entry)
        return dict(groups)
    
    def _print_legendary_summary(self, all_results: Dict, all_impacts: Dict):
        """Print comprehensive summary across all features."""
        print(f"\n{'='*70}")
        print("SUMMARY: DISTILLATION IMPACT ACROSS ALL FEATURES")
        print(f"{'='*70}")
        
        print(f"\n{'Feature':<20} | {'Overall Impact':<25} | {'Fairness Change':<15}")
        print("-" * 70)
        
        for feature_name, results in all_results.items():
            impact = results['impact']
            # Simplified status
            if "MAINTAINS" in impact['conclusion']:
                status = "Maintains"
            elif "IMPROVES" in impact['conclusion']:
                status = "Improves"
            else:
                status = "Worsens"
            
            print(f"{feature_name:<20} | {status:<25} | {impact['change']:+.2f}x ({impact['percent_change']:+.1f}%)")
        
        print(f"\n{'='*70}")
        print("GROUPS THAT GOT WORSE:")
        print(f"{'='*70}")
        
        worse_groups = []
        for feature_name, group_impacts in all_impacts.items():
            for group_name, impact in group_impacts.items():
                if impact['rmse_change'] > 0.3:  # Threshold for "worse"
                    worse_groups.append((feature_name, group_name, impact))
        
        if worse_groups:
            for feature_name, group_name, impact in worse_groups:
                print(f"  {feature_name} - {group_name}:")
                print(f"    Change: {impact['rmse_change']:+.3f} RMSE ({impact['percent_change']:+.1f}%)")
                print(f"    Status: {impact['status']}")
        else:
            print("  No groups significantly worsened!")
        
        print(f"\n{'='*70}")
        print("GROUPS THAT IMPROVED:")
        print(f"{'='*70}")
        
        better_groups = []
        for feature_name, group_impacts in all_impacts.items():
            for group_name, impact in group_impacts.items():
                if impact['rmse_change'] < -0.3:  # Threshold for "better"
                    better_groups.append((feature_name, group_name, impact))
        
        if better_groups:
            for feature_name, group_name, impact in better_groups:
                print(f"  {feature_name} - {group_name}:")
                print(f"    Change: {impact['rmse_change']:+.3f} RMSE ({impact['percent_change']:+.1f}%)")
                print(f"    Status: {impact['status']}")
        else:
            print("  No groups significantly improved")
    
    def _generate_legendary_report(self, all_results: Dict, all_impacts: Dict):
        """Generate comprehensive JSON report."""
        timestamp = self.generate_timestamp()
        
        # Build comprehensive JSON structure
        report_data = {
            "report_type": "Comprehensive Distillation Impact Analysis",
            "generated": timestamp.replace('_', ' '),
            "experiment": str(self.find_latest_experiment()),
            "total_patients": len(self.patient_data),
            "feature_analysis": {},
            "per_group_impacts": {},
            "summary": {
                "features_analyzed": list(all_results.keys()),
                "total_groups": 0,
                "groups_improved": [],
                "groups_maintained": [],
                "groups_worsened": []
            }
        }
        
        # Add feature-level analysis
        for feature_name, results in all_results.items():
            report_data["feature_analysis"][feature_name] = {
                "teacher_fairness_ratio": round(results['teacher_ratio'], 4),
                "distilled_fairness_ratio": round(results['distilled_ratio'], 4),
                "overall_impact": results['impact']['conclusion'],
                "fairness_change": round(results['impact']['change'], 4),
                "percent_change": round(results['impact']['percent_change'], 2)
            }
        
        # Add per-group impacts
        for feature_name, group_impacts in all_impacts.items():
            report_data["per_group_impacts"][feature_name] = {}
            
            for group_name, impact in group_impacts.items():
                report_data["per_group_impacts"][feature_name][group_name] = {
                    "teacher_rmse": round(impact['teacher_rmse'], 4),
                    "distilled_rmse": round(impact['distilled_rmse'], 4),
                    "rmse_change": round(impact['rmse_change'], 4),
                    "percent_change": round(impact['percent_change'], 2),
                    "status": impact['status']
                }
                
                # Categorize for summary
                report_data["summary"]["total_groups"] += 1
                if impact['rmse_change'] < -0.1:
                    report_data["summary"]["groups_improved"].append(f"{feature_name} - {group_name}")
                elif abs(impact['rmse_change']) <= 0.1:
                    report_data["summary"]["groups_maintained"].append(f"{feature_name} - {group_name}")
                else:
                    report_data["summary"]["groups_worsened"].append(f"{feature_name} - {group_name}")
        
        # Save as JSON
        report_file = self.results_dir / f"legendary_distillation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nFull report saved to: {report_file}")
        return report_file
    
    def _create_legendary_visualization(self, all_results: Dict, all_impacts: Dict):
        """Create comprehensive visualization with embedded summary table."""
        num_features = len(all_impacts)
        
        # Create figure with proper layout: 1 row for overall + 2 rows for features + 1 row for table
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.4, top=0.94, height_ratios=[1, 1, 1, 1.2])
        
        mode_label = 'All-Patients' if getattr(self, 'experiment_type', 'per_patient') == 'all_patients' else 'Per-Patient'
        fig.suptitle(f'COMPREHENSIVE DISTILLATION IMPACT ANALYSIS ({mode_label} Mode)', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Plot 1: Overall fairness changes by feature (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        features = list(all_results.keys())
        changes = [all_results[f]['impact']['change'] for f in features]
        colors = ['#2ecc71' if c <= 0 else '#e74c3c' for c in changes]
        
        bars = ax1.bar(features, changes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_title('Overall Fairness Change by Feature (Teacher → Distilled)', 
                     fontweight='bold', fontsize=14, pad=10)
        ax1.set_ylabel('Fairness Ratio Change', fontweight='bold', fontsize=12)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        if changes:
            ax1.set_ylim(min(changes) - 0.02, max(changes) + 0.02)
        else:
            ax1.set_ylim(-0.02, 0.02)
        
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            y_offset = 0.005 if height > 0 else -0.005
            ax1.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{change:+.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=11)
        
        # Plots 2-6: Per-group impacts for each feature
        # Properly arrange in 2 rows x 3 columns
        feature_positions = [
            (1, 0), (1, 1), (1, 2),  # Row 2
            (2, 0), (2, 1)            # Row 3 (only 2 plots)
        ]
        
        for idx, (feature_name, group_impacts) in enumerate(all_impacts.items()):
            if idx >= len(feature_positions):
                break
            
            row, col = feature_positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            groups = list(group_impacts.keys())
            group_changes = [group_impacts[g]['rmse_change'] for g in groups]
            
            # Color coding: green=improved, orange=slight worse, red=worse
            group_colors = []
            for c in group_changes:
                if c < -0.1:
                    group_colors.append('#2ecc71')  # Green - improved
                elif c < 0:
                    group_colors.append('#95e1d3')  # Light green - slightly improved
                elif c < 0.3:
                    group_colors.append('#f39c12')  # Orange - slightly worse
                else:
                    group_colors.append('#e74c3c')  # Red - worse
            
            bars = ax.barh(groups, group_changes, color=group_colors, alpha=0.8, 
                          edgecolor='black', linewidth=1)
            ax.set_title(f'{feature_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('RMSE Change (Teacher to Distilled)', fontsize=10)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Set x-axis limits with some padding
            if group_changes:
                x_max = max(abs(min(group_changes)), abs(max(group_changes)))
                ax.set_xlim(-x_max * 1.3, x_max * 1.3)
            
            # Add value labels - position them at the end of bars
            for i, (group, change) in enumerate(zip(groups, group_changes)):
                # Position label at the end of the bar with proper offset
                if abs(change) < 0.01:  # Very small values
                    # Put label slightly away from center for tiny bars
                    x_pos = 0.03 if change >= 0 else -0.03
                    ha = 'left' if change >= 0 else 'right'
                else:
                    # Put label at the end of the bar
                    x_pos = change * 1.05  # 5% beyond bar end
                    ha = 'left' if change > 0 else 'right'
                
                ax.text(x_pos, i, f'{change:+.3f}', 
                       ha=ha, va='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.8))
        
        # Add legend to the empty subplot space
        if num_features < 6:
            ax_legend = fig.add_subplot(gs[2, 2])
            ax_legend.axis('off')
            
            legend_text = [
                "Color Legend:",
                "",
                "Green: Improved (< -0.1)",
                "Light Green: Slightly Improved (< 0)",
                "Orange: Slightly Worse (< 0.3)",
                "Red: Worse (>= 0.3)",
                "",
                "Positive = Worse Performance",
                "Negative = Better Performance"
            ]
            
            for i, text in enumerate(legend_text):
                ax_legend.text(0.1, 0.9 - i * 0.1, text, fontsize=10,
                             verticalalignment='top', family='monospace',
                             fontweight='bold' if i == 0 else 'normal')
        
        # Add summary table at the bottom (full width)
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('off')
        ax_table.set_title('Summary Table: Per-Group Performance & Distillation Impact', 
                          fontweight='bold', fontsize=14, pad=15)
        
        # Prepare table data
        table_data = []
        table_data.append(['Feature', 'Group', 'Teacher\nRMSE', 'Student\nRMSE', 
                          'Distilled\nRMSE', 'Change\n(T→D)', '% Change', 'Status'])
        
        for feature_name in all_results.keys():
            if feature_name not in all_impacts:
                continue
            
            group_impacts = all_impacts[feature_name]
            feature_stats = all_results[feature_name]['statistics']
            
            for group_name, group_impact in group_impacts.items():
                teacher_rmse = group_impact['teacher_rmse']
                distilled_rmse = group_impact['distilled_rmse']
                rmse_change = group_impact['rmse_change']
                percent_change = group_impact['percent_change']
                status = group_impact['status']
                
                # Get student baseline if available
                student_rmse = 'N/A'
                if group_name in feature_stats and 'student_baseline' in feature_stats[group_name]:
                    student_rmse = f"{feature_stats[group_name]['student_baseline']['rmse_mean']:.2f}"
                
                table_data.append([
                    feature_name,
                    group_name,
                    f"{teacher_rmse:.2f}",
                    student_rmse,
                    f"{distilled_rmse:.2f}",
                    f"{rmse_change:+.2f}",
                    f"{percent_change:+.1f}%",
                    status
                ])
        
        # Create table
        table = ax_table.table(cellText=table_data, cellLoc='center',
                              loc='center', bbox=[0.05, 0.0, 0.9, 1.0])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        # Style the table
        for i, cell in enumerate(table.get_celld().values()):
            if i < len(table_data[0]):  # Header row
                cell.set_facecolor('#4a90e2')
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.08)
            else:
                # Color code status column
                row_idx = i // len(table_data[0])
                col_idx = i % len(table_data[0])
                
                if row_idx > 0:  # Not header
                    status_text = table_data[row_idx][7] if row_idx < len(table_data) else ""
                    
                    if col_idx == 7:  # Status column
                        if "IMPROVED" in status_text and "Slightly" not in status_text:
                            cell.set_facecolor('#d4edda')
                        elif "Slightly Improved" in status_text:
                            cell.set_facecolor('#e7f4e4')
                        elif "Slightly Worse" in status_text:
                            cell.set_facecolor('#fff3cd')
                        elif "WORSE" in status_text:
                            cell.set_facecolor('#f8d7da')
                    
                    # Alternate row colors
                    if row_idx % 2 == 0:
                        if col_idx != 7:
                            cell.set_facecolor('#f8f9fa')
                
                cell.set_height(0.06)
        
        # Save
        timestamp = self.generate_timestamp()
        mode_label = getattr(self, 'experiment_type', 'per_patient')
        plot_file = self.results_dir / f"legendary_distillation_analysis_{mode_label}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {plot_file}")
        
        # Generate and save summary table CSV
        self._save_summary_table(all_results, all_impacts, timestamp)
    
    def _save_summary_table(self, all_results: Dict, all_impacts: Dict, timestamp: str):
        """Save summary table with performance and fairness impact for each group."""
        import csv
        
        mode_label = getattr(self, 'experiment_type', 'per_patient')
        table_file = self.results_dir / f"legendary_summary_table_{mode_label}_{timestamp}.csv"
        
        with open(table_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Feature', 'Group', 'Teacher RMSE', 'Student RMSE', 
                           'Distilled RMSE', 'RMSE Change (T->D)', 'Percent Change', 
                           'Status', 'Overall Feature Fairness Change'])
            
            # Collect all rows
            for feature_name in all_results.keys():
                if feature_name not in all_impacts:
                    continue
                
                group_impacts = all_impacts[feature_name]
                feature_fairness_change = all_results[feature_name]['impact']['change']
                
                for group_name, group_impact in group_impacts.items():
                    teacher_rmse = group_impact['teacher_rmse']
                    distilled_rmse = group_impact['distilled_rmse']
                    rmse_change = group_impact['rmse_change']
                    percent_change = group_impact['percent_change']
                    status = group_impact['status']
                    
                    # Get student baseline if available
                    student_rmse = 'N/A'
                    if feature_name in all_results:
                        feature_stats = all_results[feature_name]['statistics']
                        if group_name in feature_stats and 'student_baseline' in feature_stats[group_name]:
                            student_rmse = feature_stats[group_name]['student_baseline']['rmse_mean']
                    
                    writer.writerow([
                        feature_name,
                        group_name,
                        f"{teacher_rmse:.4f}",
                        f"{student_rmse:.4f}" if isinstance(student_rmse, float) else student_rmse,
                        f"{distilled_rmse:.4f}",
                        f"{rmse_change:+.4f}",
                        f"{percent_change:+.2f}%",
                        status,
                        f"{feature_fairness_change:+.4f}"
                    ])
        
        print(f"Summary table saved to: {table_file}")



def main():
    """Run legendary distillation impact analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Legendary Distillation Impact Analyzer')
    parser.add_argument('--experiment-type', type=str, default='per_patient',
                       choices=['per_patient', 'all_patients'],
                       help='Type of experiment to analyze (default: per_patient)')
    parser.add_argument('--experiments-folder', type=str, default='distillation_experiments',
                       help='Name of the experiments folder')
    args = parser.parse_args()
    
    analyzer = LegendaryDistillationAnalyzer(experiment_type=args.experiment_type,
                                            experiments_folder=args.experiments_folder)
    analyzer.analyze_latest()


if __name__ == "__main__":
    main()
