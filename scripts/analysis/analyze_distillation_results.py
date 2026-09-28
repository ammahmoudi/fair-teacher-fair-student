#!/usr/bin/env python3
"""
Distillation Results Analyzer
==============================

Comprehensive analysis script for comparing distillation experiment results.
Analyzes teacher-student pairs, identifies best performers, and generates insights.

Usage:
    python scripts/analysis/analyze_distillation_results.py --results-dir distillation_pairs_comparison
    python scripts/analysis/analyze_distillation_results.py --results-dir distillation_pairs_comparison --top 10
    python scripts/analysis/analyze_distillation_results.py --results-dir distillation_pairs_comparison --export-json
"""

import pandas as pd
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DistillationAnalyzer:
    """Analyze and compare distillation experiment results."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "pipeline_results.csv"
        self.df = None
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found at: {self.csv_path}")
    
    def load_results(self):
        """Load and prepare results data."""
        print(f"📂 Loading results from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Filter only successful experiments
        self.df = self.df[self.df['pipeline_status'] == 'SUCCESS'].copy()
        
        print(f"✅ Loaded {len(self.df)} successful experiments")
        return self.df
    
    def get_summary_statistics(self):
        """Get overall summary statistics."""
        stats = {
            'total_experiments': len(self.df),
            'unique_teachers': self.df['teacher_model'].nunique(),
            'unique_students': self.df['student_model'].nunique(),
            'total_pairs': self.df.groupby(['teacher_model', 'student_model']).ngroups,
            'avg_teacher_rmse': self.df['teacher_rmse'].mean(),
            'avg_student_rmse': self.df['student_baseline_rmse'].mean(),
            'avg_distilled_rmse': self.df['distilled_rmse'].mean(),
            'avg_total_runtime': self.df['total_runtime'].mean(),
            'successful_distillations': len(self.df[self.df['student_to_distilled_rmse_improvement_pct'] > 0]),
            'failed_distillations': len(self.df[self.df['student_to_distilled_rmse_improvement_pct'] <= 0]),
        }
        return stats
    
    def rank_by_distillation_effectiveness(self, top_n=10):
        """Rank pairs by how much distillation improved student performance."""
        ranked = self.df.sort_values('student_to_distilled_rmse_improvement_pct', ascending=False)
        return ranked[['teacher_model', 'student_model', 
                      'teacher_rmse', 'student_baseline_rmse', 'distilled_rmse',
                      'student_to_distilled_rmse_improvement_pct',
                      'total_runtime']].head(top_n)
    
    def rank_by_absolute_performance(self, top_n=10):
        """Rank pairs by absolute distilled model performance (lowest RMSE)."""
        ranked = self.df.sort_values('distilled_rmse', ascending=True)
        return ranked[['teacher_model', 'student_model',
                      'teacher_rmse', 'student_baseline_rmse', 'distilled_rmse',
                      'student_to_distilled_rmse_improvement_pct',
                      'total_runtime']].head(top_n)
    
    def rank_by_efficiency(self, top_n=10):
        """Rank by performance per training time (RMSE improvement / runtime)."""
        self.df['efficiency_score'] = (
            self.df['student_to_distilled_rmse_improvement_pct'] / 
            (self.df['total_runtime'] / 60)  # Convert to minutes
        )
        ranked = self.df.sort_values('efficiency_score', ascending=False)
        return ranked[['teacher_model', 'student_model',
                      'distilled_rmse', 'student_to_distilled_rmse_improvement_pct',
                      'total_runtime', 'efficiency_score']].head(top_n)
    
    def find_students_beating_teachers(self):
        """Find cases where student baseline beat the teacher."""
        better = self.df[self.df['student_baseline_rmse'] < self.df['teacher_rmse']].copy()
        better['student_advantage_pct'] = (
            (better['teacher_rmse'] - better['student_baseline_rmse']) / 
            better['teacher_rmse'] * 100
        )
        return better.sort_values('student_advantage_pct', ascending=False)[
            ['teacher_model', 'student_model', 'teacher_rmse', 
             'student_baseline_rmse', 'student_advantage_pct']
        ]
    
    def find_negative_distillations(self):
        """Find cases where distillation made performance worse."""
        worse = self.df[self.df['student_to_distilled_rmse_improvement_pct'] < 0].copy()
        return worse.sort_values('student_to_distilled_rmse_improvement_pct', ascending=True)[
            ['teacher_model', 'student_model', 
             'student_baseline_rmse', 'distilled_rmse',
             'student_to_distilled_rmse_improvement_pct']
        ]
    
    def analyze_by_teacher(self):
        """Aggregate results by teacher model."""
        teacher_stats = self.df.groupby('teacher_model').agg({
            'distilled_rmse': ['mean', 'min', 'std'],
            'student_to_distilled_rmse_improvement_pct': ['mean', 'max'],
            'total_runtime': 'mean',
            'teacher_model': 'count'
        }).round(4)
        
        teacher_stats.columns = ['avg_distilled_rmse', 'best_distilled_rmse', 'std_distilled_rmse',
                                 'avg_improvement_pct', 'max_improvement_pct', 'avg_runtime', 'num_experiments']
        return teacher_stats.sort_values('avg_improvement_pct', ascending=False)
    
    def analyze_by_student(self):
        """Aggregate results by student model."""
        student_stats = self.df.groupby('student_model').agg({
            'distilled_rmse': ['mean', 'min', 'std'],
            'student_to_distilled_rmse_improvement_pct': ['mean', 'max'],
            'student_baseline_rmse': 'mean',
            'student_model': 'count'
        }).round(4)
        
        student_stats.columns = ['avg_distilled_rmse', 'best_distilled_rmse', 'std_distilled_rmse',
                                 'avg_improvement_pct', 'max_improvement_pct', 'avg_baseline_rmse', 'num_experiments']
        return student_stats.sort_values('avg_improvement_pct', ascending=False)
    
    def get_best_pairs_matrix(self):
        """Create a matrix showing best improvement for each teacher-student pair."""
        pivot = self.df.pivot_table(
            values='student_to_distilled_rmse_improvement_pct',
            index='teacher_model',
            columns='student_model',
            aggfunc='max'
        ).round(2)
        return pivot
    
    def print_comprehensive_report(self, top_n=10):
        """Print comprehensive analysis report."""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE DISTILLATION ANALYSIS REPORT")
        print("="*80)
        
        # Summary Statistics
        stats = self.get_summary_statistics()
        print("\n📊 OVERALL STATISTICS")
        print("-" * 80)
        print(f"  Total Experiments: {stats['total_experiments']}")
        print(f"  Unique Teachers: {stats['unique_teachers']}")
        print(f"  Unique Students: {stats['unique_students']}")
        print(f"  Total Teacher-Student Pairs: {stats['total_pairs']}")
        print(f"  Successful Distillations (positive improvement): {stats['successful_distillations']} ({stats['successful_distillations']/stats['total_experiments']*100:.1f}%)")
        print(f"  Failed Distillations (negative improvement): {stats['failed_distillations']} ({stats['failed_distillations']/stats['total_experiments']*100:.1f}%)")
        print(f"\n  Average Teacher RMSE: {stats['avg_teacher_rmse']:.4f}")
        print(f"  Average Student Baseline RMSE: {stats['avg_student_rmse']:.4f}")
        print(f"  Average Distilled RMSE: {stats['avg_distilled_rmse']:.4f}")
        print(f"  Average Total Runtime: {stats['avg_total_runtime']/60:.1f} minutes")
        
        # Top Performers by Distillation Effectiveness
        print(f"\n🏆 TOP {top_n} PAIRS BY DISTILLATION EFFECTIVENESS")
        print("-" * 80)
        print("(How much distillation improved student performance)")
        top_effective = self.rank_by_distillation_effectiveness(top_n)
        for idx, row in top_effective.iterrows():
            print(f"\n  {list(top_effective.index).index(idx) + 1}. {row['teacher_model']} → {row['student_model']}")
            print(f"     Teacher RMSE: {row['teacher_rmse']:.4f}")
            print(f"     Student Baseline RMSE: {row['student_baseline_rmse']:.4f}")
            print(f"     Distilled RMSE: {row['distilled_rmse']:.4f}")
            print(f"     ✨ Improvement: {row['student_to_distilled_rmse_improvement_pct']:.2f}%")
            print(f"     ⏱️  Runtime: {row['total_runtime']/60:.1f} minutes")
        
        # Top Performers by Absolute Performance
        print(f"\n🎯 TOP {top_n} PAIRS BY ABSOLUTE PERFORMANCE")
        print("-" * 80)
        print("(Lowest absolute RMSE achieved)")
        top_absolute = self.rank_by_absolute_performance(top_n)
        for idx, row in top_absolute.iterrows():
            print(f"\n  {list(top_absolute.index).index(idx) + 1}. {row['teacher_model']} → {row['student_model']}")
            print(f"     Distilled RMSE: {row['distilled_rmse']:.4f} ⭐")
            print(f"     Teacher RMSE: {row['teacher_rmse']:.4f}")
            print(f"     Improvement: {row['student_to_distilled_rmse_improvement_pct']:.2f}%")
        
        # Top Efficient Pairs
        print(f"\n⚡ TOP {top_n} MOST EFFICIENT PAIRS")
        print("-" * 80)
        print("(Best improvement per minute of training)")
        top_efficient = self.rank_by_efficiency(top_n)
        for idx, row in top_efficient.iterrows():
            print(f"\n  {list(top_efficient.index).index(idx) + 1}. {row['teacher_model']} → {row['student_model']}")
            print(f"     Efficiency Score: {row['efficiency_score']:.4f} (% improvement per minute)")
            print(f"     Improvement: {row['student_to_distilled_rmse_improvement_pct']:.2f}%")
            print(f"     Runtime: {row['total_runtime']/60:.1f} minutes")
            print(f"     Distilled RMSE: {row['distilled_rmse']:.4f}")
        
        # Teacher Analysis
        print("\n👨‍🏫 TEACHER MODEL ANALYSIS")
        print("-" * 80)
        teacher_analysis = self.analyze_by_teacher()
        print(teacher_analysis.to_string())
        
        # Student Analysis
        print("\n👨‍🎓 STUDENT MODEL ANALYSIS")
        print("-" * 80)
        student_analysis = self.analyze_by_student()
        print(student_analysis.to_string())
        
        # Students beating teachers
        students_winning = self.find_students_beating_teachers()
        if len(students_winning) > 0:
            print(f"\n🎉 STUDENTS THAT BEAT THEIR TEACHERS ({len(students_winning)} cases)")
            print("-" * 80)
            for idx, row in students_winning.head(10).iterrows():
                print(f"\n  {row['student_model']} beat {row['teacher_model']}")
                print(f"     Student RMSE: {row['student_baseline_rmse']:.4f}")
                print(f"     Teacher RMSE: {row['teacher_rmse']:.4f}")
                print(f"     Student Advantage: {row['student_advantage_pct']:.2f}% better!")
        
        # Negative distillations
        negative = self.find_negative_distillations()
        if len(negative) > 0:
            print(f"\n⚠️  NEGATIVE DISTILLATIONS ({len(negative)} cases)")
            print("-" * 80)
            print("(Where distillation made performance worse)")
            for idx, row in negative.head(10).iterrows():
                print(f"\n  {row['teacher_model']} → {row['student_model']}")
                print(f"     Student Baseline RMSE: {row['student_baseline_rmse']:.4f}")
                print(f"     Distilled RMSE: {row['distilled_rmse']:.4f}")
                print(f"     Degradation: {row['student_to_distilled_rmse_improvement_pct']:.2f}%")
        
        # Best Pairs Matrix
        print("\n📊 TEACHER-STUDENT IMPROVEMENT MATRIX (%)")
        print("-" * 80)
        matrix = self.get_best_pairs_matrix()
        print(matrix.to_string())
        
        # Recommendations
        print("\n💡 KEY RECOMMENDATIONS")
        print("-" * 80)
        
        best_overall = top_effective.iloc[0]
        print(f"\n  1️⃣  BEST OVERALL PAIR:")
        print(f"     {best_overall['teacher_model']} → {best_overall['student_model']}")
        print(f"     Improvement: {best_overall['student_to_distilled_rmse_improvement_pct']:.2f}%")
        
        best_absolute = top_absolute.iloc[0]
        print(f"\n  2️⃣  BEST ABSOLUTE PERFORMANCE:")
        print(f"     {best_absolute['teacher_model']} → {best_absolute['student_model']}")
        print(f"     RMSE: {best_absolute['distilled_rmse']:.4f}")
        
        best_efficient = top_efficient.iloc[0]
        print(f"\n  3️⃣  MOST EFFICIENT PAIR:")
        print(f"     {best_efficient['teacher_model']} → {best_efficient['student_model']}")
        print(f"     Efficiency: {best_efficient['efficiency_score']:.4f} (% improvement/min)")
        
        best_teacher = teacher_analysis.index[0]
        print(f"\n  4️⃣  BEST TEACHER MODEL:")
        print(f"     {best_teacher}")
        print(f"     Avg Improvement: {teacher_analysis.iloc[0]['avg_improvement_pct']:.2f}%")
        
        best_student = student_analysis.index[0]
        print(f"\n  5️⃣  BEST STUDENT MODEL:")
        print(f"     {best_student}")
        print(f"     Avg Improvement: {student_analysis.iloc[0]['avg_improvement_pct']:.2f}%")
        
        print("\n" + "="*80)
    
    def export_to_json(self, output_path):
        """Export analysis results to JSON."""
        results = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'total_experiments': len(self.df)
            },
            'summary_statistics': self.get_summary_statistics(),
            'top_by_effectiveness': self.rank_by_distillation_effectiveness(10).to_dict('records'),
            'top_by_performance': self.rank_by_absolute_performance(10).to_dict('records'),
            'top_by_efficiency': self.rank_by_efficiency(10).to_dict('records'),
            'teacher_analysis': self.analyze_by_teacher().to_dict(),
            'student_analysis': self.analyze_by_student().to_dict(),
            'students_beating_teachers': self.find_students_beating_teachers().to_dict('records'),
            'negative_distillations': self.find_negative_distillations().to_dict('records'),
            'improvement_matrix': self.get_best_pairs_matrix().to_dict()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results exported to: {output_path}")
    
    def export_summary_csv(self, output_path):
        """Export a simplified summary CSV."""
        summary = self.df[[
            'teacher_model', 'student_model',
            'teacher_rmse', 'student_baseline_rmse', 'distilled_rmse',
            'student_to_distilled_rmse_improvement_pct',
            'total_runtime'
        ]].copy()
        
        summary = summary.sort_values('student_to_distilled_rmse_improvement_pct', ascending=False)
        summary.to_csv(output_path, index=False)
        print(f"✅ Summary CSV exported to: {output_path}")
    
    def export_grouped_report(self, output_dir):
        """Export detailed report of distilled students that don't beat teachers."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Filter: Distilled students that DON'T beat the teacher
        not_beat_teacher = self.df[self.df['distilled_rmse'] >= self.df['teacher_rmse']].copy()
        
        # Group 1: Beat their own student baseline (successful distillation)
        group1 = not_beat_teacher[not_beat_teacher['student_to_distilled_rmse_improvement_pct'] > 0].copy()
        group1 = group1.sort_values('student_to_distilled_rmse_improvement_pct', ascending=False)
        
        # Group 2: Don't beat their own student baseline (failed distillation)
        group2 = not_beat_teacher[not_beat_teacher['student_to_distilled_rmse_improvement_pct'] <= 0].copy()
        group2 = group2.sort_values('student_to_distilled_rmse_improvement_pct', ascending=True)
        
        # Create text report
        report_path = output_dir / f'distilled_not_beating_teacher_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 DISTILLED STUDENTS THAT DON'T BEAT THE TEACHER\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nTotal cases: {len(not_beat_teacher)} out of {len(self.df)} experiments\n")
            f.write(f"Percentage: {len(not_beat_teacher)/len(self.df)*100:.1f}%\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("✅ GROUP 1: BEAT STUDENT BASELINE (Successful Distillation)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Count: {len(group1)} experiments\n")
            f.write("Description: Distillation helped the student improve, but still couldn't match teacher\n\n")
            
            if len(group1) > 0:
                for idx, row in group1.iterrows():
                    f.write(f"\n{list(group1.index).index(idx) + 1}. {row['teacher_model']} → {row['student_model']}\n")
                    f.write(f"   {'─' * 70}\n")
                    f.write(f"   Teacher RMSE:          {row['teacher_rmse']:.4f}\n")
                    f.write(f"   Student Baseline RMSE: {row['student_baseline_rmse']:.4f}\n")
                    f.write(f"   Distilled RMSE:        {row['distilled_rmse']:.4f}\n")
                    f.write(f"   \n")
                    f.write(f"   ✅ Distillation improved student by: {row['student_to_distilled_rmse_improvement_pct']:.2f}%\n")
                    f.write(f"   ❌ Still worse than teacher by:      {((row['distilled_rmse'] - row['teacher_rmse']) / row['teacher_rmse'] * 100):.2f}%\n")
                    f.write(f"   \n")
                    f.write(f"   Absolute gap to teacher: {row['distilled_rmse'] - row['teacher_rmse']:.4f}\n")
                    f.write(f"   Improvement over student: {row['student_baseline_rmse'] - row['distilled_rmse']:.4f}\n")
            else:
                f.write("No experiments in this group.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("❌ GROUP 2: DON'T BEAT STUDENT BASELINE (Failed Distillation)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Count: {len(group2)} experiments\n")
            f.write("Description: Distillation failed - student got worse AND didn't reach teacher\n\n")
            
            if len(group2) > 0:
                for idx, row in group2.iterrows():
                    f.write(f"\n{list(group2.index).index(idx) + 1}. {row['teacher_model']} → {row['student_model']}\n")
                    f.write(f"   {'─' * 70}\n")
                    f.write(f"   Teacher RMSE:          {row['teacher_rmse']:.4f}\n")
                    f.write(f"   Student Baseline RMSE: {row['student_baseline_rmse']:.4f}\n")
                    f.write(f"   Distilled RMSE:        {row['distilled_rmse']:.4f}\n")
                    f.write(f"   \n")
                    f.write(f"   ❌ Distillation degraded student by: {row['student_to_distilled_rmse_improvement_pct']:.2f}%\n")
                    f.write(f"   ❌ Worse than teacher by:            {((row['distilled_rmse'] - row['teacher_rmse']) / row['teacher_rmse'] * 100):.2f}%\n")
                    f.write(f"   \n")
                    f.write(f"   Absolute gap to teacher: {row['distilled_rmse'] - row['teacher_rmse']:.4f}\n")
                    f.write(f"   Degradation from student: {row['distilled_rmse'] - row['student_baseline_rmse']:.4f}\n")
            else:
                f.write("No experiments in this group.\n")
            
            # Summary table
            f.write("\n" + "=" * 80 + "\n")
            f.write("📈 SUMMARY TABLE\n")
            f.write("=" * 80 + "\n\n")
            
            summary_data = []
            for _, row in not_beat_teacher.iterrows():
                beat_student = "✅ Yes" if row['student_to_distilled_rmse_improvement_pct'] > 0 else "❌ No"
                summary_data.append({
                    'Teacher→Student': f"{row['teacher_model'][:20]}→{row['student_model'][:25]}",
                    'Teacher': f"{row['teacher_rmse']:.4f}",
                    'Student': f"{row['student_baseline_rmse']:.4f}",
                    'Distilled': f"{row['distilled_rmse']:.4f}",
                    'Beat Student?': beat_student,
                    'Improvement': f"{row['student_to_distilled_rmse_improvement_pct']:.2f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            f.write(summary_df.to_string(index=False) + "\n")
            
            # Key insights
            f.write("\n" + "=" * 80 + "\n")
            f.write("💡 KEY INSIGHTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"GROUP 1 (Partial Success):\n")
            f.write(f"• {len(group1)} distillations improved student but couldn't reach teacher performance\n")
            f.write(f"• These show knowledge transfer worked, but teacher superiority remained\n")
            if len(group1) > 0:
                f.write(f"• Average improvement over student: {group1['student_to_distilled_rmse_improvement_pct'].mean():.2f}%\n\n")
            else:
                f.write("\n")
            
            f.write(f"GROUP 2 (Complete Failure):\n")
            f.write(f"• {len(group2)} distillations failed to improve student at all\n")
            f.write(f"• Knowledge transfer failed or caused negative transfer\n")
            if len(group2) > 0:
                f.write(f"• Average degradation: {group2['student_to_distilled_rmse_improvement_pct'].mean():.2f}%\n\n")
            else:
                f.write("\n")
            
            f.write(f"OVERALL:\n")
            f.write(f"• Total unsuccessful at beating teacher: {len(not_beat_teacher)} out of {len(self.df)} ({len(not_beat_teacher)/len(self.df)*100:.1f}%)\n")
            f.write(f"• Distillations that DID beat teacher: {len(self.df) - len(not_beat_teacher)} ({(len(self.df) - len(not_beat_teacher))/len(self.df)*100:.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Grouped report exported to: {report_path}")
        
        # Create CSV for easy analysis
        csv_path = output_dir / f'distilled_not_beating_teacher_{timestamp}.csv'
        
        not_beat_teacher['beat_student_baseline'] = not_beat_teacher['student_to_distilled_rmse_improvement_pct'] > 0
        not_beat_teacher['group'] = not_beat_teacher['beat_student_baseline'].map({
            True: 'GROUP1_PartialSuccess',
            False: 'GROUP2_CompleteFail'
        })
        not_beat_teacher['gap_to_teacher'] = not_beat_teacher['distilled_rmse'] - not_beat_teacher['teacher_rmse']
        
        export_cols = ['teacher_model', 'student_model', 'teacher_rmse', 'student_baseline_rmse', 
                       'distilled_rmse', 'student_to_distilled_rmse_improvement_pct', 
                       'gap_to_teacher', 'group', 'beat_student_baseline']
        
        not_beat_teacher[export_cols].to_csv(csv_path, index=False)
        print(f"✅ Grouped CSV exported to: {csv_path}")
        
        return report_path, csv_path
    
    def create_visualizations(self, output_dir):
        """Create comprehensive visualization diagrams."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Generating visualizations...")
        
        # 1. Improvement Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(14, 8))
        df_sorted = self.df.sort_values('student_to_distilled_rmse_improvement_pct', ascending=False)
        pairs = [f"{row['teacher_model'][:20]}\n→ {row['student_model'][:30]}" 
                for _, row in df_sorted.iterrows()]
        improvements = df_sorted['student_to_distilled_rmse_improvement_pct'].values
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        bars = ax.barh(pairs, improvements, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('Distillation Effectiveness: Student Improvement over Baseline', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            label_x = val + (0.02 if val > 0 else -0.02)
            ha = 'left' if val > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                   ha=ha, va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        improvement_path = output_dir / 'distillation_improvement_comparison.png'
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {improvement_path}")
        
        # 2. RMSE Performance Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        df_sorted = self.df.sort_values('distilled_rmse', ascending=True)
        
        x = np.arange(len(df_sorted))
        width = 0.25
        
        pairs_short = [f"{row['teacher_model'][:15]}→{row['student_model'][:15]}" 
                      for _, row in df_sorted.iterrows()]
        
        ax.bar(x - width, df_sorted['teacher_rmse'], width, label='Teacher', alpha=0.8, color='#2E86AB')
        ax.bar(x, df_sorted['student_baseline_rmse'], width, label='Student Baseline', alpha=0.8, color='#A23B72')
        ax.bar(x + width, df_sorted['distilled_rmse'], width, label='Distilled Student', alpha=0.8, color='#F18F01')
        
        ax.set_xlabel('Teacher-Student Pairs', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title('RMSE Performance Comparison: Teacher vs Student vs Distilled', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs_short, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        rmse_path = output_dir / 'rmse_performance_comparison.png'
        plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {rmse_path}")
        
        # 3. Heatmap of Improvement Matrix
        fig, ax = plt.subplots(figsize=(12, 8))
        matrix = self.get_best_pairs_matrix()
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Improvement (%)'}, linewidths=0.5,
                   ax=ax, vmin=-2, vmax=2)
        
        ax.set_title('Teacher-Student Improvement Matrix (%)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Student Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Teacher Model', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        heatmap_path = output_dir / 'improvement_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {heatmap_path}")
        
        # 4. Teacher Model Performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        teacher_stats = self.analyze_by_teacher()
        teachers = teacher_stats.index.tolist()
        avg_improvement = teacher_stats['avg_improvement_pct'].values
        avg_rmse = teacher_stats['avg_distilled_rmse'].values
        
        # Average improvement by teacher
        colors_teacher = ['green' if x > 0 else 'red' for x in avg_improvement]
        bars1 = ax1.barh(teachers, avg_improvement, color=colors_teacher, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Average Improvement (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Teacher Model Performance: Avg Improvement', 
                     fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars1, avg_improvement):
            label_x = val + (0.02 if val > 0 else -0.02)
            ha = 'left' if val > 0 else 'right'
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                    ha=ha, va='center', fontsize=10, fontweight='bold')
        
        # Average RMSE by teacher
        bars2 = ax2.barh(teachers, avg_rmse, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Average Distilled RMSE', fontsize=12, fontweight='bold')
        ax2.set_title('Teacher Model Performance: Avg Final RMSE', 
                     fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars2, avg_rmse):
            ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        teacher_path = output_dir / 'teacher_analysis.png'
        plt.savefig(teacher_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {teacher_path}")
        
        # 5. Student Model Performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        student_stats = self.analyze_by_student()
        students = [s[:40] for s in student_stats.index.tolist()]  # Truncate long names
        avg_improvement_student = student_stats['avg_improvement_pct'].values
        avg_rmse_student = student_stats['avg_distilled_rmse'].values
        
        # Average improvement by student
        colors_student = ['green' if x > 0 else 'red' for x in avg_improvement_student]
        bars1 = ax1.barh(students, avg_improvement_student, color=colors_student, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Average Improvement (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Student Model Performance: Avg Improvement', 
                     fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars1, avg_improvement_student):
            label_x = val + (0.02 if val > 0 else -0.02)
            ha = 'left' if val > 0 else 'right'
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                    ha=ha, va='center', fontsize=10, fontweight='bold')
        
        # Average RMSE by student
        bars2 = ax2.barh(students, avg_rmse_student, color='coral', alpha=0.7)
        ax2.set_xlabel('Average Distilled RMSE', fontsize=12, fontweight='bold')
        ax2.set_title('Student Model Performance: Avg Final RMSE', 
                     fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars2, avg_rmse_student):
            ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        student_path = output_dir / 'student_analysis.png'
        plt.savefig(student_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {student_path}")
        
        # 6. Efficiency Analysis (Improvement per Runtime)
        fig, ax = plt.subplots(figsize=(14, 8))
        df_efficiency = self.df.copy()
        df_efficiency = df_efficiency.sort_values('efficiency_score', ascending=False)
        
        pairs_eff = [f"{row['teacher_model'][:20]}\n→ {row['student_model'][:30]}" 
                    for _, row in df_efficiency.iterrows()]
        efficiency = df_efficiency['efficiency_score'].values
        colors_eff = ['green' if x > 0 else 'red' for x in efficiency]
        
        bars = ax.barh(pairs_eff, efficiency, color=colors_eff, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Efficiency Score (% improvement per minute)', fontsize=12, fontweight='bold')
        ax.set_title('Training Efficiency: Improvement per Minute of Training', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, efficiency):
            label_x = val + (0.001 if val > 0 else -0.001)
            ha = 'left' if val > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                   ha=ha, va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        efficiency_path = output_dir / 'training_efficiency.png'
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {efficiency_path}")
        
        # 7. Success/Failure Pie Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stats = self.get_summary_statistics()
        
        # Distillation success rate
        success_data = [stats['successful_distillations'], stats['failed_distillations']]
        labels_success = ['Positive\nImprovement', 'Negative\nImprovement']
        colors_pie = ['#2ecc71', '#e74c3c']
        
        ax1.pie(success_data, labels=labels_success, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_title('Distillation Success Rate', fontsize=13, fontweight='bold', pad=20)
        
        # Average RMSE comparison
        rmse_data = [stats['avg_teacher_rmse'], stats['avg_student_rmse'], stats['avg_distilled_rmse']]
        labels_rmse = ['Teacher', 'Student\nBaseline', 'Distilled']
        
        bars = ax2.bar(labels_rmse, rmse_data, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax2.set_title('Average RMSE Comparison', fontsize=13, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, rmse_data):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        summary_path = output_dir / 'summary_statistics.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {summary_path}")
        
        # 8. Comprehensive Dashboard - All Key Information in One Image
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # Top section: Title and key statistics
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        
        stats = self.get_summary_statistics()
        title_text = "📊 COMPREHENSIVE DISTILLATION ANALYSIS DASHBOARD\n"
        title_text += f"Total Experiments: {stats['total_experiments']} | "
        title_text += f"Teachers: {stats['unique_teachers']} | Students: {stats['unique_students']} | "
        title_text += f"Success Rate: {stats['successful_distillations']}/{stats['total_experiments']} ({stats['successful_distillations']/stats['total_experiments']*100:.1f}%)"
        
        ax_title.text(0.5, 0.7, title_text, ha='center', va='center', 
                     fontsize=16, fontweight='bold', transform=ax_title.transAxes)
        
        # Row 1: Top 5 pairs by improvement
        ax1 = fig.add_subplot(gs[1, 0])
        top5 = self.rank_by_distillation_effectiveness(5)
        pairs_top5 = [f"{row['teacher_model'][:15]}→\n{row['student_model'][:20]}" 
                      for _, row in top5.iterrows()]
        improvements_top5 = top5['student_to_distilled_rmse_improvement_pct'].values
        colors_top5 = ['green' if x > 0 else 'red' for x in improvements_top5]
        
        bars = ax1.barh(range(len(pairs_top5)), improvements_top5, color=colors_top5, alpha=0.7)
        ax1.set_yticks(range(len(pairs_top5)))
        ax1.set_yticklabels(pairs_top5, fontsize=8)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Improvement (%)', fontsize=9)
        ax1.set_title('🏆 Top 5 Pairs by Improvement', fontsize=11, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, improvements_top5):
            label_x = val + (0.03 if val > 0 else -0.03)
            ha = 'left' if val > 0 else 'right'
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                    ha=ha, va='center', fontsize=8, fontweight='bold')
        
        # Row 1: RMSE comparison
        ax2 = fig.add_subplot(gs[1, 1])
        rmse_labels = ['Teacher\nAvg', 'Student\nBaseline', 'Distilled\nStudent']
        rmse_values = [stats['avg_teacher_rmse'], stats['avg_student_rmse'], stats['avg_distilled_rmse']]
        colors_rmse = ['#2E86AB', '#A23B72', '#F18F01']
        
        bars = ax2.bar(rmse_labels, rmse_values, color=colors_rmse, alpha=0.7)
        ax2.set_ylabel('RMSE', fontsize=9)
        ax2.set_title('📈 Average RMSE Comparison', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Row 1: Success/Failure pie
        ax3 = fig.add_subplot(gs[1, 2])
        success_data = [stats['successful_distillations'], stats['failed_distillations']]
        labels_pie = [f"Success\n({stats['successful_distillations']})", 
                     f"Failed\n({stats['failed_distillations']})"]
        colors_pie = ['#2ecc71', '#e74c3c']
        
        wedges, texts, autotexts = ax3.pie(success_data, labels=labels_pie, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90)
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        ax3.set_title('✅ Success Rate', fontsize=11, fontweight='bold')
        
        # Row 2: Improvement heatmap
        ax4 = fig.add_subplot(gs[2, :])
        matrix = self.get_best_pairs_matrix()
        
        # Truncate long model names for heatmap
        matrix.index = [idx[:30] for idx in matrix.index]
        matrix.columns = [col[:35] for col in matrix.columns]
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Improvement (%)'}, linewidths=0.5,
                   ax=ax4, vmin=-2, vmax=2, annot_kws={'fontsize': 8})
        
        ax4.set_title('🔥 Teacher-Student Improvement Matrix (%)', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Student Model', fontsize=9)
        ax4.set_ylabel('Teacher Model', fontsize=9)
        ax4.tick_params(labelsize=8)
        
        # Row 3: Students beating teachers
        ax5 = fig.add_subplot(gs[3, 0])
        students_winning = self.find_students_beating_teachers()
        
        if len(students_winning) > 0:
            top_wins = students_winning.head(5)
            win_labels = []
            win_advantages = []
            
            for _, row in top_wins.iterrows():
                student_short = row['student_model'].split('/')[-1][:15]
                teacher_short = row['teacher_model'].split('/')[-1][:15]
                win_labels.append(f"{student_short}\nvs {teacher_short}")
                win_advantages.append(row['student_advantage_pct'])
            
            bars = ax5.barh(range(len(win_labels)), win_advantages, color='green', alpha=0.7)
            ax5.set_yticks(range(len(win_labels)))
            ax5.set_yticklabels(win_labels, fontsize=7)
            ax5.set_xlabel('Advantage (%)', fontsize=9)
            ax5.set_title('🎉 Students Beating Teachers (Top 5)', fontsize=10, fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
            
            for bar, val in zip(bars, win_advantages):
                ax5.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                        ha='left', va='center', fontsize=8, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No students beat teachers', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=10)
            ax5.axis('off')
        
        # Row 3: Distilled students beating their baseline (successful distillation)
        ax6 = fig.add_subplot(gs[3, 1])
        successful_distillation = self.df[self.df['student_to_distilled_rmse_improvement_pct'] > 0].copy()
        successful_distillation = successful_distillation.sort_values('student_to_distilled_rmse_improvement_pct', ascending=False).head(5)
        
        if len(successful_distillation) > 0:
            dist_labels = []
            dist_improvements = []
            
            for _, row in successful_distillation.iterrows():
                student_short = row['student_model'].split('/')[-1][:15]
                teacher_short = row['teacher_model'].split('/')[-1][:15]
                dist_labels.append(f"{teacher_short}→\n{student_short}")
                dist_improvements.append(row['student_to_distilled_rmse_improvement_pct'])
            
            bars = ax6.barh(range(len(dist_labels)), dist_improvements, color='orange', alpha=0.7)
            ax6.set_yticks(range(len(dist_labels)))
            ax6.set_yticklabels(dist_labels, fontsize=7)
            ax6.set_xlabel('Improvement (%)', fontsize=9)
            ax6.set_title('✨ Distilled Beating Student Baseline (Top 5)', fontsize=10, fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
            
            for bar, val in zip(bars, dist_improvements):
                ax6.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                        ha='left', va='center', fontsize=8, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No successful\ndistillations', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=10)
            ax6.axis('off')
        
        # Row 3: Best models summary
        ax7 = fig.add_subplot(gs[3, 2])
        ax7.axis('off')
        
        teacher_analysis = self.analyze_by_teacher()
        student_analysis = self.analyze_by_student()
        top_effective = self.rank_by_distillation_effectiveness(1).iloc[0]
        top_absolute = self.rank_by_absolute_performance(1).iloc[0]
        
        best_teacher = teacher_analysis.index[0]
        best_student = student_analysis.index[0]
        
        summary_text = "🎯 KEY RECOMMENDATIONS\n\n"
        summary_text += f"🏆 Best Overall Pair:\n"
        summary_text += f"{top_effective['teacher_model'][:20]}\n→ {top_effective['student_model'][:25]}\n"
        summary_text += f"Improvement: {top_effective['student_to_distilled_rmse_improvement_pct']:.2f}%\n\n"
        
        summary_text += f"⭐ Best Absolute RMSE:\n"
        summary_text += f"{top_absolute['teacher_model'][:20]}\n→ {top_absolute['student_model'][:25]}\n"
        summary_text += f"RMSE: {top_absolute['distilled_rmse']:.4f}\n\n"
        
        summary_text += f"👨‍🏫 Best Teacher:\n{best_teacher[:30]}\n"
        summary_text += f"Avg Improvement: {teacher_analysis.iloc[0]['avg_improvement_pct']:.2f}%\n\n"
        
        summary_text += f"👨‍🎓 Best Student:\n{best_student[:30]}\n"
        summary_text += f"Avg Improvement: {student_analysis.iloc[0]['avg_improvement_pct']:.2f}%"
        
        ax7.text(0.05, 0.95, summary_text, ha='left', va='top',
                transform=ax7.transAxes, fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add footer
        fig.text(0.5, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Dataset: {self.df["dataset_name"].iloc[0]} | Patient: {self.df["patient_ids"].iloc[0]}',
                ha='center', fontsize=8, style='italic')
        
        dashboard_path = output_dir / 'comprehensive_dashboard.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {dashboard_path}")
        
        print(f"\n✅ All visualizations saved to: {output_dir}/")
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare distillation experiment results"
    )
    parser.add_argument(
        '--results-dir',
        default='distillation_pairs_comparison',
        help='Directory containing pipeline_results.csv'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top results to show in each category'
    )
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export analysis results to JSON file'
    )
    parser.add_argument(
        '--export-summary',
        action='store_true',
        help='Export simplified summary CSV'
    )
    parser.add_argument(
        '--export-diagrams',
        action='store_true',
        help='Export visualization diagrams (requires matplotlib)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for exports (default: same as results-dir)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = DistillationAnalyzer(args.results_dir)
        analyzer.load_results()
        
        # Print comprehensive report
        analyzer.print_comprehensive_report(top_n=args.top)
        
        # Export results if requested
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.export_json:
            json_path = output_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            analyzer.export_to_json(json_path)
        
        if args.export_summary:
            csv_path = output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            analyzer.export_summary_csv(csv_path)
        
        # Always export grouped report (distilled not beating teacher)
        analyzer.export_grouped_report(output_dir)
        
        if args.export_diagrams:
            diagrams_dir = output_dir / "diagrams"
            analyzer.create_visualizations(diagrams_dir)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
