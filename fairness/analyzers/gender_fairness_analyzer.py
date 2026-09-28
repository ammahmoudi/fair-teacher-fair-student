#!/usr/bin/env python3
"""
Gender Fairness Analyzer for OhioT1DM Dataset

Analyzes fairness across genders (20-40, 40-60, 60-80) using actual experiment results.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

# Add project root to path dynamically
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from fairness.analyzers.base_analyzer import BaseFairnessAnalyzer
from fairness.utils.analyzer_utils import (
    extract_feature_from_default_data,
    format_report_header,
    print_group_summary,
    format_fairness_level
)


class GenderFairnessAnalyzer(BaseFairnessAnalyzer):
    """Analyze fairness across genders."""
    
    def __init__(self, data_path=None, experiment_type="per_patient", experiments_folder="distillation_experiments"):
        """
        Initialize gender fairness analyzer.
        
        Args:
            data_path: Optional path to patient data CSV
            experiment_type: Type of experiment ("per_patient" or "all_patients")
            experiments_folder: Name of the experiments folder
        """
        super().__init__(feature_name="Gender", data_path=data_path, 
                        experiment_type=experiment_type, experiments_folder=experiments_folder)
        print(f"📊 Loaded gender data for {len(self.patient_data)} patients")
        print(f"🔬 Experiment type: {experiment_type}")
        print(f"📁 Experiments folder: {experiments_folder}")
    
    def _load_patient_data(self) -> Dict:
        """Load patient gender data from CSV or use defaults."""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Loaded data for {len(df)} patients")
            
            gender_mapping = {}
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                gender = row['Gender']
                gender_mapping[patient_id] = gender
            
            return gender_mapping
        except FileNotFoundError:
            print(f"⚠️  Patient data file not found, using defaults")
            return self._get_default_patient_data()
        except Exception as e:
            print(f"❌ Error loading patient data: {e}")
            return {}
    
    def _get_default_patient_data(self) -> Dict:
        """Get default gender data."""
        return extract_feature_from_default_data('gender')
    
    def analyze_latest(self):
        """Analyze gender fairness for the latest experiment."""
        exp_label = "All-Patients" if self.experiment_type == "all_patients" else "Per-Patient"
        print(f"\n🔍 Starting Gender Fairness Analysis ({exp_label})")
        print("=" * 60)
        
        # Find and load latest experiment
        experiment_path = self.find_latest_experiment()
        
        # Load model names for visualization
        self.load_model_names(Path(experiment_path))
        
        patient_results = self.load_patient_results(Path(experiment_path))
        
        # Group by gender
        groups = self.group_by_feature(patient_results)
        
        # Calculate statistics
        statistics = self.calculate_group_statistics(groups)
        
        # Print results (same multi-phase output for both experiment types)
        exp_type_label = "All-Patients Model" if self.experiment_type == "all_patients" else "Per-Patient Models"
        print(f"\n📊 Gender Distribution ({exp_type_label} - Multi-Phase Results):")
        from fairness.utils.analyzer_utils import print_multi_phase_summary
        for gender, stats in statistics.items():
            print_multi_phase_summary(gender, stats)
        
        # Calculate fairness for each phase
        print(f"\n⚖️ FAIRNESS ASSESSMENT BY PHASE:")
        teacher_ratio, teacher_level = self.calculate_fairness_ratio(statistics, 'teacher')
        student_ratio, student_level = self.calculate_fairness_ratio(statistics, 'student_baseline')
        distilled_ratio, distilled_level = self.calculate_fairness_ratio(statistics, 'distilled')
        
        print(f"  Teacher Model:   {teacher_ratio:.2f}x ({teacher_level})")
        print(f"  Student Model:   {student_ratio:.2f}x ({student_level})")
        print(f"  Distilled Model: {distilled_ratio:.2f}x ({distilled_level})")
        
        # Analyze distillation impact
        impact = self.analyze_distillation_impact(statistics)
        print(f"\n🎯 DISTILLATION IMPACT ON OVERALL FAIRNESS:")
        print(f"  {impact['conclusion']}")
        print(f"  Fairness Change: {impact['change']:+.2f}x ({impact['percent_change']:+.1f}%)")
        print(f"  Severity: {impact['severity']}")
        
        # Show per-group distillation impact
        from fairness.utils.analyzer_utils import analyze_per_group_distillation_impact
        group_impacts = analyze_per_group_distillation_impact(statistics)
        print(f"\n📊 DISTILLATION IMPACT BY GROUP:")
        for group_name, group_impact in group_impacts.items():
            print(f"  {group_name}:")
            print(f"    Teacher → Distilled: {group_impact['teacher_rmse']:.3f} → {group_impact['distilled_rmse']:.3f}")
            print(f"    Change: {group_impact['rmse_change']:+.3f} ({group_impact['percent_change']:+.1f}%)")
            print(f"    Status: {group_impact['status']}")
        
        # Visualize multi-phase
        self.visualize(statistics, distilled_ratio, impact)
        
        # Generate multi-phase report
        self._generate_report(statistics, distilled_ratio, distilled_level, impact)
        
        print(f"\n📁 All results saved in: {self.results_dir}")
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(statistics)} genders analyzed")
    
    def _visualize_single_phase(self, statistics: Dict, fairness_ratio: float, level: str):
        """Visualize single-phase (all-patients) results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not statistics:
            print("⚠️ Cannot create visualizations - insufficient data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gender Fairness Analysis - All-Patients Distilled Model', 
                     fontsize=16, fontweight='bold')
        
        groups = list(statistics.keys())
        
        # 1. RMSE by Gender
        rmse_values = [statistics[g]['distilled']['rmse_mean'] for g in groups if 'distilled' in statistics[g]]
        if rmse_values:
            bars = ax1.bar(groups, rmse_values, color='#2ecc71', alpha=0.8)
            ax1.set_xlabel('Gender Group', fontweight='bold')
            ax1.set_ylabel('RMSE (Lower = Better)', fontweight='bold')
            ax1.set_title('Distilled Model Performance by Gender')
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, rmse_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. MAE by Gender
        mae_values = [statistics[g]['distilled']['mae_mean'] for g in groups if 'distilled' in statistics[g]]
        if mae_values:
            bars = ax2.bar(groups, mae_values, color='#3498db', alpha=0.8)
            ax2.set_xlabel('Gender Group', fontweight='bold')
            ax2.set_ylabel('MAE (Lower = Better)', fontweight='bold')
            ax2.set_title('Mean Absolute Error by Gender')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, mae_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Fairness Ratio
        if fairness_ratio > 0:
            color = 'lightgreen' if fairness_ratio < 1.2 else 'gold' if fairness_ratio < 1.5 else 'salmon'
            ax3.bar(['Fairness Ratio'], [fairness_ratio], color=color, alpha=0.7, width=0.4)
            ax3.set_ylabel('Performance Ratio (Worse/Better)', fontweight='bold')
            ax3.set_title('Gender Fairness Ratio')
            ax3.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect Fairness')
            ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable')
            ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Poor')
            ax3.legend()
            ax3.set_ylim(bottom=0.9)
            ax3.text(0, fairness_ratio + fairness_ratio*0.01,
                    f'{fairness_ratio:.2f}x\n{level}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
        
        # 4. Summary
        ax4.text(0.1, 0.9, 'EXPERIMENT SUMMARY', fontsize=14, fontweight='bold', 
                transform=ax4.transAxes)
        
        summary_text = [
            f"Experiment Type: All-Patients Training",
            f"Total patients: {sum(s['count'] for s in statistics.values())}",
            f"Gender groups: {len(groups)}",
            "",
            "Fairness Assessment:",
            f"Ratio:  {fairness_ratio:.3f}x",
            f"Level:  {level}",
        ]
        
        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.8 - i*0.06, text, fontsize=10, transform=ax4.transAxes,
                    family='monospace')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        timestamp = self.generate_timestamp()
        plot_file = self.results_dir / f"gender_all_patients_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {plot_file}")
    
    def _generate_single_phase_report(self, statistics: Dict, fairness_ratio: float, level: str):
        """Generate JSON report for all-patients experiment."""
        report_data = {
            "report_type": "Gender Fairness Analysis - All-Patients Model",
            "generated": self.generate_timestamp().replace('_', ' '),
            "analysis_type": "gender",
            "experiment_type": "all_patients_training",
            "groups": {},
            "overall_fairness": {
                "ratio": round(fairness_ratio, 4),
                "level": level
            }
        }
        
        for gender, stats in statistics.items():
            report_data["groups"][gender] = {
                "patient_count": stats['count'],
                "patient_ids": stats['patient_ids']
            }
            
            if 'distilled' in stats:
                report_data["groups"][gender]["distilled"] = {
                    "rmse_mean": round(stats['distilled']['rmse_mean'], 4),
                    "rmse_std": round(stats['distilled']['rmse_std'], 4),
                    "mae_mean": round(stats['distilled']['mae_mean'], 4),
                    "mae_std": round(stats['distilled']['mae_std'], 4)
                }
        
        self.save_json_report(report_data, "gender_all_patients")
    
    def _generate_report(self, statistics: Dict, fairness_ratio: float, level: str, impact: Dict = None):
        """Generate JSON report WITH DISTILLATION IMPACT."""
        from fairness.utils.analyzer_utils import analyze_per_group_distillation_impact
        
        report_data = {
            "report_type": "Gender Fairness Analysis - Distillation Impact",
            "generated": self.generate_timestamp().replace('_', ' '),
            "analysis_type": "gender",
            "groups": {},
            "overall_fairness": {
                "ratio": round(fairness_ratio, 4),
                "level": level
            }
        }
        
        # Add multi-phase performance results
        for gender, stats in statistics.items():
            report_data["groups"][gender] = {
                "patient_count": stats['count'],
                "phases": {}
            }
            for phase in ['teacher', 'student_baseline', 'distilled']:
                if phase in stats:
                    report_data["groups"][gender]["phases"][phase] = {
                        "rmse_mean": round(stats[phase]['rmse_mean'], 4),
                        "mae_mean": round(stats[phase]['mae_mean'], 4)
                    }
        
        # Add distillation impact
        if impact:
            report_data["distillation_impact"] = {
                "teacher_fairness_ratio": round(impact['teacher_ratio'], 4),
                "teacher_level": impact['teacher_level'],
                "distilled_fairness_ratio": round(impact['distilled_ratio'], 4),
                "distilled_level": impact['distilled_level'],
                "change": round(impact['change'], 4),
                "percent_change": round(impact['percent_change'], 2),
                "conclusion": impact['conclusion'],
                "severity": impact['severity']
            }
            
            # Add per-group impact
            group_impacts = analyze_per_group_distillation_impact(statistics)
            report_data["per_group_impact"] = {}
            for group_name, group_impact in group_impacts.items():
                report_data["per_group_impact"][group_name] = {
                    "teacher_rmse": round(group_impact['teacher_rmse'], 4),
                    "distilled_rmse": round(group_impact['distilled_rmse'], 4),
                    "rmse_change": round(group_impact['rmse_change'], 4),
                    "percent_change": round(group_impact['percent_change'], 2),
                    "status": group_impact['status']
                }
        
        self.save_json_report(report_data, "gender_fairness")


def main():
    """Run gender fairness analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gender Fairness Analysis')
    parser.add_argument('--experiment-type', type=str, default='per_patient',
                       choices=['per_patient', 'all_patients'],
                       help='Type of experiment to analyze (default: per_patient)')
    parser.add_argument('--experiments-folder', type=str, default='distillation_experiments',
                       help='Name of the experiments folder (default: distillation_experiments)')
    args = parser.parse_args()
    
    analyzer = GenderFairnessAnalyzer(experiment_type=args.experiment_type,
                                     experiments_folder=args.experiments_folder)
    analyzer.analyze_latest()


if __name__ == "__main__":
    main()
