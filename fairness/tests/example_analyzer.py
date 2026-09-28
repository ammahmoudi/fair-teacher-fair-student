#!/usr/bin/env python3
"""
EXAMPLE: Gender Fairness Analysis with Distillation Impact

This demonstrates the complete analysis showing:
- Teacher model performance
- Student baseline performance  
- Distilled model performance
- How distillation affects fairness

Run this to see the full multi-phase analysis!
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Add project root to path dynamically
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from fairness.analyzers.base_analyzer import BaseFairnessAnalyzer
from fairness.utils.analyzer_utils import (
    extract_feature_from_default_data,
    format_report_header,
    print_multi_phase_summary
)


class DistillationImpactAnalyzer(BaseFairnessAnalyzer):
    """Analyze fairness WITH distillation impact across all 3 phases."""
    
    def __init__(self, data_path=None):
        super().__init__(feature_name="Gender", data_path=data_path)
        print(f"📊 Loaded gender data for {len(self.patient_data)} patients")
    
    def _load_patient_data(self) -> Dict:
        """Load patient gender data."""
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
        """🎯 FULL MULTI-PHASE FAIRNESS ANALYSIS WITH DISTILLATION IMPACT"""
        print("\n" + "="*80)
        print("🎯 DISTILLATION IMPACT FAIRNESS ANALYSIS")
        print("   Analyzing Teacher → Student → Distilled")
        print("="*80)
        
        # Find and load latest experiment (now loads ALL 3 phases!)
        experiment_path = Path(self.find_latest_experiment())
        patient_results = self.load_patient_results(experiment_path)
        
        # Group by gender
        groups = self.group_by_feature(patient_results)
        
        # Calculate statistics for ALL phases
        statistics = self.calculate_group_statistics(groups)
        
        # Print results for ALL phases
        print(f"\n📊 Gender Distribution (ALL 3 PHASES):")
        for gender, stats in statistics.items():
            print_multi_phase_summary(gender, stats)
        
        # Calculate fairness for EACH phase
        print(f"\n⚖️ FAIRNESS ASSESSMENT BY PHASE:")
        teacher_ratio, teacher_level = self.calculate_fairness_ratio(statistics, 'teacher')
        student_ratio, student_level = self.calculate_fairness_ratio(statistics, 'student_baseline')
        distilled_ratio, distilled_level = self.calculate_fairness_ratio(statistics, 'distilled')
        
        print(f"  📚 Teacher Model:   {teacher_ratio:.2f}x ({teacher_level})")
        print(f"  🎓 Student Model:   {student_ratio:.2f}x ({student_level})")
        print(f"  🔬 Distilled Model: {distilled_ratio:.2f}x ({distilled_level})")
        
        # 🎯 ANALYZE DISTILLATION IMPACT!
        impact = self.analyze_distillation_impact(statistics)
        print(f"\n🎯 DISTILLATION IMPACT ON FAIRNESS:")
        print(f"  {impact['conclusion']}")
        print(f"  Fairness Change: {impact['change']:+.2f}x ({impact['percent_change']:+.1f}%)")
        print(f"  Severity: {impact['severity']}")
        
        # Visualize with distillation comparison
        self.visualize_distillation_impact(statistics, impact)
        
        # Generate comprehensive report
        self._generate_report(statistics, impact)
        
        print(f"\n📁 All results saved in: {self.results_dir}")
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(statistics)} groups analyzed across 3 phases")
    
    def visualize_distillation_impact(self, statistics: Dict, impact: Dict):
        """Create visualization showing distillation impact on fairness."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Distillation Impact on Gender Fairness', fontsize=18, fontweight='bold')
        
        genders = list(statistics.keys())
        
        # Plot 1: Performance across all 3 phases
        ax1 = axes[0, 0]
        phases = ['Teacher', 'Student', 'Distilled']
        phase_keys = ['teacher', 'student_baseline', 'distilled']
        
        x = np.arange(len(phases))
        width = 0.35
        
        # Get RMSE for each phase and gender
        male_rmse = [statistics['Male'][pk]['rmse_mean'] if pk in statistics['Male'] else 0 for pk in phase_keys]
        female_rmse = [statistics['Female'][pk]['rmse_mean'] if pk in statistics['Female'] else 0 for pk in phase_keys]
        
        bars1 = ax1.bar(x - width/2, male_rmse, width, label='Male', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, female_rmse, width, label='Female', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Model Phase', fontsize=12, fontweight='bold')
        ax1.set_ylabel('RMSE (Lower = Better)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Across Distillation Pipeline', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(phases)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Fairness Ratio Progression
        ax2 = axes[0, 1]
        fairness_ratios = [
            impact['teacher_ratio'],
            self.calculate_fairness_ratio(statistics, 'student_baseline')[0],
            impact['distilled_ratio']
        ]
        
        ax2.plot(phases, fairness_ratios, 'o-', linewidth=3, markersize=12, color='purple')
        ax2.axhline(y=1.0, color='green', linestyle='-', alpha=0.5, label='Perfect Fairness')
        ax2.axhline(y=1.25, color='orange', linestyle='--', alpha=0.5, label='Good Threshold')
        ax2.set_xlabel('Model Phase', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fairness Ratio (Worse/Better)', fontsize=12, fontweight='bold')
        ax2.set_title('Fairness Ratio Through Distillation', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Annotate points
        for i, (phase, ratio) in enumerate(zip(phases, fairness_ratios)):
            ax2.annotate(f'{ratio:.2f}x', (i, ratio), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=11, fontweight='bold')
        
        # Plot 3: Distillation Impact Bar
        ax3 = axes[1, 0]
        impact_metrics = ['Teacher', 'Distilled', 'Change']
        impact_values = [impact['teacher_ratio'], impact['distilled_ratio'], abs(impact['change'])]
        colors = ['lightblue', 'lightcoral' if impact['makes_worse'] else 'lightgreen', 'gold']
        
        bars = ax3.bar(impact_metrics, impact_values, color=colors, alpha=0.7)
        ax3.set_ylabel('Fairness Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('Distillation Impact Summary', fontsize=14)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, impact_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 4: Summary Text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
DISTILLATION IMPACT ANALYSIS

Sample Sizes:
  Male: {statistics['Male']['count']} patients
  Female: {statistics['Female']['count']} patients

Fairness Metrics:
  Teacher:   {impact['teacher_ratio']:.2f}x ({impact['teacher_level']})
  Distilled: {impact['distilled_ratio']:.2f}x ({impact['distilled_level']})
  Change:    {impact['change']:+.2f}x ({impact['percent_change']:+.1f}%)

CONCLUSION:
  {impact['conclusion']}
  
Severity: {impact['severity']}

{'⚠️  ACTION REQUIRED: Consider fairness-aware distillation' if impact['makes_worse'] and impact['severity'] in ['CRITICAL', 'MODERATE'] else '✅ Fairness maintained through distillation'}
        """
        
        ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                family='monospace', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save
        timestamp = self.generate_timestamp()
        plot_file = self.results_dir / f"distillation_impact_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {plot_file}")
    
    def _generate_report(self, statistics: Dict, impact: Dict):
        """Generate comprehensive report with distillation impact."""
        report = format_report_header("DISTILLATION IMPACT FAIRNESS ANALYSIS")
        report += f"Generated: {self.generate_timestamp().replace('_', ' ')}\n\n"
        
        report += "MULTI-PHASE PERFORMANCE RESULTS:\n"
        report += "="*80 + "\n"
        for gender, stats in statistics.items():
            report += f"\n{gender.upper()} ({stats['count']} patients):\n"
            for phase in ['teacher', 'student_baseline', 'distilled']:
                if phase in stats:
                    phase_label = phase.replace('_', ' ').title()
                    report += f"  {phase_label:20s}: RMSE={stats[phase]['rmse_mean']:.3f}, MAE={stats[phase]['mae_mean']:.3f}\n"
        
        report += "\n\nDISTILLATION IMPACT ANALYSIS:\n"
        report += "="*80 + "\n"
        report += f"Teacher Fairness:   {impact['teacher_ratio']:.2f}x ({impact['teacher_level']})\n"
        report += f"Distilled Fairness: {impact['distilled_ratio']:.2f}x ({impact['distilled_level']})\n"
        report += f"Change:             {impact['change']:+.2f}x ({impact['percent_change']:+.1f}%)\n"
        report += f"\n{impact['conclusion']}\n"
        report += f"Severity: {impact['severity']}\n"
        
        if impact['makes_worse'] and impact['severity'] in ['CRITICAL', 'MODERATE']:
            report += "\n\nRECOMMENDATIONS:\n"
            report += "="*80 + "\n"
            report += "⚠️  IMMEDIATE ACTION REQUIRED:\n"
            report += "  - Implement fairness-aware distillation\n"
            report += "  - Add demographic parity constraints\n"
            report += "  - Balance training data across groups\n"
            report += "  - Monitor fairness metrics during distillation\n"
        else:
            report += "\n\n✅ Fairness is maintained through distillation process.\n"
        
        report += "\n" + "="*80
        
        self.save_report(report, "distillation_impact_fairness")
    
    def visualize(self, statistics: Dict, fairness_ratio: float):
        """Not used - using visualize_distillation_impact instead."""
        pass


def main():
    """Run distillation impact analysis."""
    analyzer = DistillationImpactAnalyzer()
    analyzer.analyze_latest()


if __name__ == "__main__":
    main()
