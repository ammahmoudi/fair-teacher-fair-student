#!/usr/bin/env python3
"""
Age Group Fairness Analyzer for OhioT1DM Dataset

Analyzes fairness across age groups (20-40, 40-60, 60-80) using actual experiment results.
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


class AgeFairnessAnalyzer(BaseFairnessAnalyzer):
    """Analyze fairness across age groups."""
    
    def __init__(self, data_path=None, experiment_type='per_patient', experiments_folder='distillation_experiments'):
        super().__init__(feature_name="Age", data_path=data_path, experiment_type=experiment_type, 
                        experiments_folder=experiments_folder)
        print(f"📊 Loaded age data for {len(self.patient_data)} patients")
    
    def _load_patient_data(self) -> Dict:
        """Load patient age group data from CSV or use defaults."""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Loaded data for {len(df)} patients")
            
            age_mapping = {}
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                age_group = row['Age']
                age_mapping[patient_id] = age_group
            
            return age_mapping
        except FileNotFoundError:
            print(f"⚠️  Patient data file not found, using defaults")
            return self._get_default_patient_data()
        except Exception as e:
            print(f"❌ Error loading patient data: {e}")
            return {}
    
    def _get_default_patient_data(self) -> Dict:
        """Get default age group data."""
        return extract_feature_from_default_data('age')
    
    def analyze_latest(self):
        """Analyze age group fairness for the latest experiment WITH DISTILLATION IMPACT."""
        exp_type_label = "All-Patients Model" if self.experiment_type == "all_patients" else "Per-Patient Models"
        print(f"\n🔍 Starting Age Group Fairness Analysis ({exp_type_label} - Multi-Phase)")
        print("=" * 70)
        
        # Find and load latest experiment
        experiment_path = self.find_latest_experiment()
        
        # Load model names for visualization
        self.load_model_names(Path(experiment_path))
        
        patient_results = self.load_patient_results(Path(experiment_path))
        
        # Group by age
        groups = self.group_by_feature(patient_results)
        
        # Calculate statistics (now includes all phases)
        statistics = self.calculate_group_statistics(groups)
        
        # Print results for all phases
        print(f"\n📊 Age Group Distribution ({exp_type_label} - Multi-Phase Results):")
        from fairness.utils.analyzer_utils import print_multi_phase_summary
        for age_group, stats in statistics.items():
            print_multi_phase_summary(age_group, stats)
        
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
        print(f"\n📊 DISTILLATION IMPACT BY AGE GROUP:")
        for group_name, group_impact in group_impacts.items():
            print(f"  {group_name}:")
            print(f"    Teacher → Distilled: {group_impact['teacher_rmse']:.3f} → {group_impact['distilled_rmse']:.3f}")
            print(f"    Change: {group_impact['rmse_change']:+.3f} ({group_impact['percent_change']:+.1f}%)")
            print(f"    Status: {group_impact['status']}")
        
        # Visualize
        self.visualize(statistics, distilled_ratio, impact)
        
        # Generate report
        self._generate_report(statistics, distilled_ratio, distilled_level, impact)
        
        print(f"\n📁 All results saved in: {self.results_dir}")
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(statistics)} age groups analyzed across 3 phases")
    
    def _generate_report(self, statistics: Dict, fairness_ratio: float, level: str, impact: Dict = None):
        """Generate JSON report WITH DISTILLATION IMPACT."""
        from fairness.utils.analyzer_utils import analyze_per_group_distillation_impact
        
        report_data = {
            "report_type": "Age Group Fairness Analysis - Distillation Impact",
            "generated": self.generate_timestamp().replace('_', ' '),
            "analysis_type": "age_group",
            "groups": {},
            "overall_fairness": {
                "ratio": round(fairness_ratio, 4),
                "level": level
            }
        }
        
        # Add multi-phase performance results
        for age_group, stats in statistics.items():
            report_data["groups"][age_group] = {
                "patient_count": stats['count'],
                "phases": {}
            }
            for phase in ['teacher', 'student_baseline', 'distilled']:
                if phase in stats:
                    report_data["groups"][age_group]["phases"][phase] = {
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
        
        self.save_json_report(report_data, "age_group_fairness")


def main():
    """Run age group fairness analysis."""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze age group fairness')
    parser.add_argument('--experiment-type', type=str, default='per_patient',
                       choices=['per_patient', 'all_patients'],
                       help='Type of experiment to analyze')
    parser.add_argument('--experiments-folder', type=str, default='distillation_experiments',
                       help='Name of the experiments folder')
    args = parser.parse_args()
    
    analyzer = AgeFairnessAnalyzer(experiment_type=args.experiment_type,
                                  experiments_folder=args.experiments_folder)
    analyzer.analyze_latest()


if __name__ == "__main__":
    main()
