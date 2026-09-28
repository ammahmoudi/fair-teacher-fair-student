#!/usr/bin/env python3
"""
Investigate Fairness Issues - Find problematic scenarios and groups
This script analyzes both distillation and inference results to identify
specific groups that develop fairness problems after training/distillation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

class FairnessInvestigator:
    def __init__(self):
        self.results_dir = Path(__file__).parent.parent / "analysis_results"
        self.distillation_dir = self.results_dir / "distillation_all_patients"
        self.inference_dir = self.results_dir / "inference_scenarios"
        
        # Thresholds for problems
        self.POOR_RATIO = 1.5  # Fairness ratio above this is POOR
        self.ACCEPTABLE_RATIO = 1.25  # Above this is ACCEPTABLE or worse
        self.SIGNIFICANT_DEGRADATION = 0.1  # 10% degradation is significant
        
    def find_latest_report(self, directory: Path, pattern: str) -> Path:
        """Find the most recent report file matching the pattern."""
        reports = sorted(directory.glob(pattern))
        if not reports:
            raise FileNotFoundError(f"No reports found matching {pattern} in {directory}")
        return reports[-1]  # Most recent
    
    def analyze_distillation(self) -> Dict:
        """Analyze distillation results to find fairness problems."""
        print("=" * 80)
        print("🔍 ANALYZING DISTILLATION FAIRNESS ISSUES")
        print("=" * 80)
        
        # Load distillation legendary report
        report_file = self.find_latest_report(
            self.distillation_dir, 
            "legendary_distillation_report_*.json"
        )
        print(f"\nLoading: {report_file.name}")
        
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        problems = {
            'poor_fairness': [],      # Groups with poor fairness (>1.5)
            'degraded_fairness': [],  # Groups where fairness got worse
            'worst_cases': []         # Top problematic cases
        }
        
        # Map display names to internal keys
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        if 'feature_analysis' not in data:
            print("⚠️  Warning: No feature_analysis found in report")
            return problems
        
        for display_name, feature_key in feature_map.items():
            if display_name not in data['feature_analysis']:
                continue
            
            feature_data = data['feature_analysis'][display_name]
            teacher_ratio = feature_data['teacher_fairness_ratio']
            distilled_ratio = feature_data['distilled_fairness_ratio']
            degradation = distilled_ratio - teacher_ratio
            
            # Check for poor fairness after distillation
            if distilled_ratio >= self.POOR_RATIO:
                problems['poor_fairness'].append({
                    'feature': feature_key,
                    'ratio': distilled_ratio,
                    'phase': 'distilled',
                    'teacher_ratio': teacher_ratio,
                    'degradation': degradation
                })
            
            # Check for significant degradation
            if degradation >= self.SIGNIFICANT_DEGRADATION:
                problems['degraded_fairness'].append({
                    'feature': feature_key,
                    'teacher_ratio': teacher_ratio,
                    'distilled_ratio': distilled_ratio,
                    'degradation': degradation,
                    'degradation_pct': (degradation / teacher_ratio) * 100
                })
        
        # Get per-group data from per_group_impacts section
        if 'per_group_impacts' in data:
            for display_name, feature_key in feature_map.items():
                if display_name not in data['per_group_impacts']:
                    continue
                
                per_group_data = data['per_group_impacts'][display_name]
                rmse_values = []
                
                for group_name, group_data in per_group_data.items():
                    if 'distilled_rmse' in group_data:
                        rmse_values.append((group_name, group_data['distilled_rmse']))
                
                if rmse_values:
                    worst_group = max(rmse_values, key=lambda x: x[1])
                    best_group = min(rmse_values, key=lambda x: x[1])
                    
                    feature_data = data['feature_analysis'][display_name]
                    
                    problems['worst_cases'].append({
                        'feature': feature_key,
                        'worst_group': worst_group[0],
                        'worst_rmse': worst_group[1],
                        'best_group': best_group[0],
                        'best_rmse': best_group[1],
                        'ratio': feature_data['distilled_fairness_ratio'],
                        'teacher_ratio': feature_data['teacher_fairness_ratio']
                    })
        
        return problems
    
    def analyze_inference(self) -> Dict:
        """Analyze inference scenarios to find fairness problems."""
        print("\n" + "=" * 80)
        print("🔍 ANALYZING INFERENCE SCENARIO FAIRNESS ISSUES")
        print("=" * 80)
        
        # Load inference legendary report
        report_file = self.find_latest_report(
            self.inference_dir, 
            "legendary_inference_scenarios_report_*.json"
        )
        print(f"\nLoading: {report_file.name}")
        
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        problems = {
            'poor_fairness': [],      # Scenarios with poor fairness
            'training_degradation': [],  # Where training made fairness worse
            'worst_cases': []         # Top problematic cases
        }
        
        # Map display names to internal keys
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        if 'results' not in data:
            print("⚠️  Warning: No results found in report")
            return problems
        
        scenarios = ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']
        
        for display_name, feature_key in feature_map.items():
            if display_name not in data['results']:
                continue
            
            feature_data = data['results'][display_name]
            inference_only_ratio = feature_data['inference_only']['fairness_ratio']
            
            for scenario in ['trained_standard', 'trained_noisy', 'trained_denoised']:
                if scenario not in feature_data:
                    continue
                
                scenario_ratio = feature_data[scenario]['fairness_ratio']
                degradation = scenario_ratio - inference_only_ratio
                
                # Check for poor fairness after training
                if scenario_ratio >= self.POOR_RATIO:
                    problems['poor_fairness'].append({
                        'feature': feature_key,
                        'scenario': scenario,
                        'ratio': scenario_ratio,
                        'inference_only_ratio': inference_only_ratio,
                        'degradation': degradation
                    })
                
                # Check for significant degradation from inference-only
                if degradation >= self.SIGNIFICANT_DEGRADATION:
                    problems['training_degradation'].append({
                        'feature': feature_key,
                        'scenario': scenario,
                        'inference_only_ratio': inference_only_ratio,
                        'trained_ratio': scenario_ratio,
                        'degradation': degradation,
                        'degradation_pct': (degradation / inference_only_ratio) * 100
                    })
                
                # Get per-group statistics to find worst groups
                if 'groups' in feature_data[scenario]:
                    groups = feature_data[scenario]['groups']
                    rmse_values = [(group_name, group_data['rmse_mean']) 
                                 for group_name, group_data in groups.items() 
                                 if 'rmse_mean' in group_data]
                    
                    if rmse_values:
                        worst_group = max(rmse_values, key=lambda x: x[1])
                        best_group = min(rmse_values, key=lambda x: x[1])
                        
                        problems['worst_cases'].append({
                            'feature': feature_key,
                            'scenario': scenario,
                            'worst_group': worst_group[0],
                            'worst_rmse': worst_group[1],
                            'best_group': best_group[0],
                            'best_rmse': best_group[1],
                            'ratio': scenario_ratio,
                            'inference_only_ratio': inference_only_ratio
                        })
        
        return problems
    
    def print_distillation_findings(self, problems: Dict):
        """Print distillation findings."""
        print("\n" + "=" * 80)
        print("📊 DISTILLATION FINDINGS")
        print("=" * 80)
        
        # 1. Poor fairness cases
        if problems['poor_fairness']:
            print(f"\n🔴 POOR FAIRNESS (Ratio ≥ {self.POOR_RATIO}):")
            for p in sorted(problems['poor_fairness'], key=lambda x: x['ratio'], reverse=True):
                print(f"  • {p['feature'].upper().replace('_', ' ')}")
                print(f"    Teacher: {p['teacher_ratio']:.3f}x → Distilled: {p['ratio']:.3f}x")
                print(f"    Degradation: {p['degradation']:+.3f}x")
        else:
            print(f"\n✅ No features with POOR fairness (≥{self.POOR_RATIO})")
        
        # 2. Degraded fairness
        if problems['degraded_fairness']:
            print(f"\n⚠️  SIGNIFICANT FAIRNESS DEGRADATION (≥{self.SIGNIFICANT_DEGRADATION}):")
            for p in sorted(problems['degraded_fairness'], key=lambda x: x['degradation'], reverse=True):
                print(f"  • {p['feature'].upper().replace('_', ' ')}")
                print(f"    {p['teacher_ratio']:.3f}x → {p['distilled_ratio']:.3f}x")
                print(f"    Degradation: {p['degradation']:+.3f}x ({p['degradation_pct']:+.1f}%)")
        else:
            print(f"\n✅ No significant fairness degradation")
        
        # 3. Worst performing groups (show top 5 even if not "poor")
        if problems['worst_cases']:
            print(f"\n🎯 TOP 5 FAIRNESS DISPARITIES:")
            for p in sorted(problems['worst_cases'], key=lambda x: x['ratio'], reverse=True)[:5]:
                degraded = p['ratio'] > p['teacher_ratio']
                arrow = "⬆️" if degraded else "⬇️"
                print(f"  {arrow} {p['feature'].upper().replace('_', ' ')}")
                print(f"    Worst Group: {p['worst_group']} (RMSE={p['worst_rmse']:.2f})")
                print(f"    Best Group: {p['best_group']} (RMSE={p['best_rmse']:.2f})")
                print(f"    Fairness Ratio: {p['ratio']:.3f}x (Teacher: {p['teacher_ratio']:.3f}x)")
                print(f"    Gap: {p['worst_rmse'] - p['best_rmse']:.2f} RMSE points")
    
    def print_inference_findings(self, problems: Dict):
        """Print inference findings."""
        print("\n" + "=" * 80)
        print("📊 INFERENCE SCENARIO FINDINGS")
        print("=" * 80)
        
        # 1. Poor fairness cases
        if problems['poor_fairness']:
            print(f"\n🔴 POOR FAIRNESS (Ratio ≥ {self.POOR_RATIO}):")
            for p in sorted(problems['poor_fairness'], key=lambda x: x['ratio'], reverse=True):
                print(f"  • {p['feature'].upper().replace('_', ' ')} - {p['scenario'].replace('_', ' ').title()}")
                print(f"    Inference-only: {p['inference_only_ratio']:.3f}x → Trained: {p['ratio']:.3f}x")
                print(f"    Degradation: {p['degradation']:+.3f}x")
        else:
            print(f"\n✅ No scenarios with POOR fairness (≥{self.POOR_RATIO})")
        
        # 2. Training degradation
        if problems['training_degradation']:
            print(f"\n⚠️  TRAINING-INDUCED FAIRNESS DEGRADATION (≥{self.SIGNIFICANT_DEGRADATION}):")
            for p in sorted(problems['training_degradation'], key=lambda x: x['degradation'], reverse=True):
                print(f"  • {p['feature'].upper().replace('_', ' ')} - {p['scenario'].replace('_', ' ').title()}")
                print(f"    {p['inference_only_ratio']:.3f}x → {p['trained_ratio']:.3f}x")
                print(f"    Degradation: {p['degradation']:+.3f}x ({p['degradation_pct']:+.1f}%)")
        else:
            print(f"\n✅ No significant training-induced degradation")
        
        # 3. Worst performing groups by scenario (show top 10 even if not "poor")
        if problems['worst_cases']:
            print(f"\n🎯 TOP 10 FAIRNESS DISPARITIES BY SCENARIO:")
            for p in sorted(problems['worst_cases'], key=lambda x: x['ratio'], reverse=True)[:10]:
                degraded = p['ratio'] > p['inference_only_ratio']
                arrow = "⬆️" if degraded else "⬇️"
                print(f"  {arrow} {p['feature'].upper().replace('_', ' ')} - {p['scenario'].replace('_', ' ').title()}")
                print(f"    Worst Group: {p['worst_group']} (RMSE={p['worst_rmse']:.2f})")
                print(f"    Best Group: {p['best_group']} (RMSE={p['best_rmse']:.2f})")
                print(f"    Fairness Ratio: {p['ratio']:.3f}x (Inference-only: {p['inference_only_ratio']:.3f}x)")
                print(f"    Gap: {p['worst_rmse'] - p['best_rmse']:.2f} RMSE points")
    
    def recommend_focus_areas(self, distill_problems: Dict, inference_problems: Dict):
        """Recommend specific areas to focus on for fixing fairness."""
        print("\n" + "=" * 80)
        print("🎯 RECOMMENDED FOCUS AREAS FOR FAIRNESS IMPROVEMENT")
        print("=" * 80)
        
        recommendations = []
        
        # Check distillation problems
        for p in distill_problems['poor_fairness']:
            recommendations.append({
                'priority': 'HIGH',
                'context': 'Distillation',
                'feature': p['feature'],
                'issue': f"Poor fairness ratio ({p['ratio']:.3f}x) after distillation",
                'action': f"Focus on {p['feature']} groups in distillation process"
            })
        
        for p in distill_problems['degraded_fairness']:
            recommendations.append({
                'priority': 'MEDIUM',
                'context': 'Distillation',
                'feature': p['feature'],
                'issue': f"Fairness degraded by {p['degradation_pct']:.1f}% during distillation",
                'action': f"Investigate distillation impact on {p['feature']} groups"
            })
        
        # Check inference problems
        for p in inference_problems['poor_fairness']:
            recommendations.append({
                'priority': 'HIGH',
                'context': 'Inference',
                'feature': p['feature'],
                'scenario': p['scenario'],
                'issue': f"Poor fairness ratio ({p['ratio']:.3f}x) in {p['scenario']}",
                'action': f"Focus on {p['feature']} groups in {p['scenario']} scenario"
            })
        
        for p in inference_problems['training_degradation']:
            recommendations.append({
                'priority': 'MEDIUM',
                'context': 'Inference',
                'feature': p['feature'],
                'scenario': p['scenario'],
                'issue': f"Training degraded fairness by {p['degradation_pct']:.1f}%",
                'action': f"Investigate training impact on {p['feature']} in {p['scenario']}"
            })
        
        # Sort by priority and print
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        if not recommendations:
            print("\n✅ No critical fairness issues found!")
            print("All groups show acceptable fairness ratios (< 1.5).")
            print("\nHowever, you can still work on reducing smaller disparities...")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. [{rec['priority']}] {rec['context']} - {rec['feature'].upper().replace('_', ' ')}")
                if 'scenario' in rec:
                    print(f"   Scenario: {rec['scenario'].replace('_', ' ').title()}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Action: {rec['action']}")
        
        # Provide specific group recommendations
        print("\n" + "=" * 80)
        print("🔬 SPECIFIC GROUPS TO TARGET FOR IMPROVEMENT:")
        print("=" * 80)
        
        # Get worst groups from distillation
        if distill_problems['worst_cases']:
            worst_distill = sorted(distill_problems['worst_cases'], 
                                  key=lambda x: x['ratio'], reverse=True)[0]
            print(f"\n📍 DISTILLATION - HIGHEST DISPARITY:")
            print(f"   Feature: {worst_distill['feature'].upper().replace('_', ' ')}")
            print(f"   Disadvantaged Group: {worst_distill['worst_group']} (RMSE={worst_distill['worst_rmse']:.2f})")
            print(f"   Advantaged Group: {worst_distill['best_group']} (RMSE={worst_distill['best_rmse']:.2f})")
            print(f"   Performance Gap: {worst_distill['worst_rmse'] - worst_distill['best_rmse']:.2f} RMSE points")
            print(f"   Fairness Ratio: {worst_distill['ratio']:.3f}x")
            print(f"   Teacher Fairness: {worst_distill['teacher_ratio']:.3f}x")
        
        # Get worst groups from inference
        if inference_problems['worst_cases']:
            worst_inference = sorted(inference_problems['worst_cases'], 
                                    key=lambda x: x['ratio'], reverse=True)[0]
            print(f"\n📍 INFERENCE - HIGHEST DISPARITY:")
            print(f"   Feature: {worst_inference['feature'].upper().replace('_', ' ')}")
            print(f"   Scenario: {worst_inference['scenario'].replace('_', ' ').title()}")
            print(f"   Disadvantaged Group: {worst_inference['worst_group']} (RMSE={worst_inference['worst_rmse']:.2f})")
            print(f"   Advantaged Group: {worst_inference['best_group']} (RMSE={worst_inference['best_rmse']:.2f})")
            print(f"   Performance Gap: {worst_inference['worst_rmse'] - worst_inference['best_rmse']:.2f} RMSE points")
            print(f"   Fairness Ratio: {worst_inference['ratio']:.3f}x")
            print(f"   Inference-only Fairness: {worst_inference['inference_only_ratio']:.3f}x")
    
    def generate_visual_report(self, distill_problems: Dict, inference_problems: Dict):
        """Generate visual report with charts."""
        print("\n" + "=" * 80)
        print("📊 GENERATING VISUAL REPORT")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"fairness_investigation_report_{timestamp}.png"
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, top=0.94, bottom=0.06)
        
        fig.suptitle('FAIRNESS INVESTIGATION REPORT', fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Distillation fairness ratios (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_distillation_ratios(ax1, distill_problems)
        
        # 2. Distillation group gaps (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_distillation_gaps(ax2, distill_problems)
        
        # 3. Distillation status (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_distillation_status(ax3, distill_problems)
        
        # 4. Inference fairness ratios heatmap (middle row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_inference_heatmap(ax4, inference_problems)
        
        # 5. Top 5 problematic scenarios (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_top_problems(ax5, distill_problems, inference_problems)
        
        # 6. Training impact comparison (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_training_impact(ax6, inference_problems)
        
        # 7. Recommendations (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_recommendations(ax7, distill_problems, inference_problems)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visual report saved: {output_file}")
        return output_file
    
    def _plot_distillation_ratios(self, ax, problems: Dict):
        """Plot distillation fairness ratios."""
        if not problems['worst_cases']:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Distillation Fairness Ratios', fontweight='bold')
            ax.axis('off')
            return
        
        features = [p['feature'].replace('_', '\n').title() for p in problems['worst_cases']]
        teacher_ratios = [p['teacher_ratio'] for p in problems['worst_cases']]
        distilled_ratios = [p['ratio'] for p in problems['worst_cases']]
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, teacher_ratios, width, label='Teacher', 
                       color='#4a90e2', alpha=0.8)
        bars2 = ax.bar(x + width/2, distilled_ratios, width, label='Distilled',
                       color='#e74c3c', alpha=0.8)
        
        # Add threshold line
        ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Poor Threshold')
        ax.axhline(y=1.25, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_ylabel('Fairness Ratio', fontweight='bold', fontsize=11)
        ax.set_title('Distillation Fairness Ratios', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(1.0, max(max(distilled_ratios), 1.6))
    
    def _plot_distillation_gaps(self, ax, problems: Dict):
        """Plot RMSE gaps between worst and best groups."""
        if not problems['worst_cases']:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Performance Gaps', fontweight='bold')
            ax.axis('off')
            return
        
        features = [p['feature'].replace('_', '\n').title() for p in problems['worst_cases']]
        gaps = [p['worst_rmse'] - p['best_rmse'] for p in problems['worst_cases']]
        
        # Color by gap size
        colors = ['#e74c3c' if g > 3 else '#f39c12' if g > 2 else '#2ecc71' for g in gaps]
        
        bars = ax.barh(features, gaps, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            ax.text(gap + 0.1, i, f'{gap:.1f}', va='center', fontsize=9)
        
        ax.set_xlabel('RMSE Gap (Worst - Best)', fontweight='bold', fontsize=11)
        ax.set_title('Distillation Performance Gaps', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(gaps) * 1.2)
    
    def _plot_distillation_status(self, ax, problems: Dict):
        """Plot distillation impact status."""
        ax.axis('off')
        
        # Count statuses
        poor_count = len(problems['poor_fairness'])
        degraded_count = len(problems['degraded_fairness'])
        total_features = len(problems['worst_cases'])
        good_count = total_features - poor_count - degraded_count
        
        # Create status summary
        ax.text(0.5, 0.85, 'DISTILLATION STATUS', ha='center', fontsize=12, 
                fontweight='bold', transform=ax.transAxes)
        
        status_text = f"""
Total Features: {total_features}

Good: {good_count}
Degraded: {degraded_count}
Poor: {poor_count}
"""
        
        ax.text(0.5, 0.5, status_text, ha='center', va='center', fontsize=11,
                family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add overall verdict
        if poor_count == 0 and degraded_count == 0:
            verdict = "ALL GOOD"
            color = 'green'
        elif poor_count > 0:
            verdict = "ISSUES FOUND"
            color = 'red'
        else:
            verdict = "NEEDS ATTENTION"
            color = 'orange'
        
        ax.text(0.5, 0.15, verdict, ha='center', fontsize=14, fontweight='bold',
                color=color, transform=ax.transAxes)
    
    def _plot_inference_heatmap(self, ax, problems: Dict):
        """Plot heatmap of inference fairness ratios."""
        if not problems['worst_cases']:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title('Inference Fairness Ratios', fontweight='bold')
            ax.axis('off')
            return
        
        # Organize data by feature and scenario
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        scenarios = ['trained_standard', 'trained_noisy', 'trained_denoised']
        
        # Create matrix
        matrix = np.zeros((len(features), len(scenarios)))
        
        for p in problems['worst_cases']:
            if p['feature'] in features and p['scenario'] in scenarios:
                i = features.index(p['feature'])
                j = scenarios.index(p['scenario'])
                matrix[i, j] = p['ratio']
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.0)
        
        # Set ticks
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels([s.replace('_', '\n').title() for s in scenarios], fontsize=10)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
        
        # Add values in cells
        for i in range(len(features)):
            for j in range(len(scenarios)):
                if matrix[i, j] > 0:
                    text_color = 'white' if matrix[i, j] > 1.4 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
        
        ax.set_title('Inference Scenario Fairness Ratios (by Feature)', fontweight='bold', fontsize=13)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fairness Ratio', rotation=270, labelpad=20, fontweight='bold')
        
        # Add threshold description below the chart (inside the axis)
        ax.text(1.5, 5.3, '< 1.1: Excellent | 1.1-1.25: Good | 1.25-1.5: Acceptable | > 1.5: Poor', 
                fontsize=8, ha='center', va='top')
    
    def _plot_top_problems(self, ax, distill_problems: Dict, inference_problems: Dict):
        """Plot top 5 problematic scenarios."""
        # Combine and rank all problems
        all_problems = []
        
        # Add distillation problems
        for p in distill_problems['worst_cases']:
            all_problems.append({
                'label': f"D: {p['feature'].replace('_', ' ').title()}",
                'ratio': p['ratio'],
                'gap': p['worst_rmse'] - p['best_rmse']
            })
        
        # Add inference problems (top 5 worst)
        worst_inference = sorted(inference_problems['worst_cases'], 
                                key=lambda x: x['ratio'], reverse=True)[:5]
        for p in worst_inference:
            all_problems.append({
                'label': f"I: {p['feature'].replace('_', ' ').title()}\n{p['scenario'].replace('_', ' ').title()}",
                'ratio': p['ratio'],
                'gap': p['worst_rmse'] - p['best_rmse']
            })
        
        # Sort and take top 5
        all_problems.sort(key=lambda x: x['ratio'], reverse=True)
        top5 = all_problems[:5]
        
        if not top5:
            ax.text(0.5, 0.5, 'No Issues Found', ha='center', va='center', fontsize=12)
            ax.set_title('Top 5 Fairness Issues', fontweight='bold')
            ax.axis('off')
            return
        
        labels = [p['label'] for p in top5]
        ratios = [p['ratio'] for p in top5]
        
        # Color based on severity
        colors = ['#e74c3c' if r >= 1.5 else '#f39c12' if r >= 1.25 else '#2ecc71' for r in ratios]
        
        bars = ax.barh(labels, ratios, color=colors, alpha=0.8)
        
        # Add threshold line
        ax.axvline(x=1.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=1.25, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            ax.text(ratio + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{ratio:.2f}x', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Fairness Ratio', fontweight='bold', fontsize=11)
        ax.set_title('Top 5 Fairness Issues', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(1.0, max(ratios) * 1.15)
        
        # Add legend at top right
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Poor (≥1.5)'),
            mpatches.Patch(color='#f39c12', label='Acceptable (≥1.25)'),
            mpatches.Patch(color='#2ecc71', label='Good/Excellent')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_training_impact(self, ax, problems: Dict):
        """Plot training impact on fairness."""
        if not problems['training_degradation']:
            ax.text(0.5, 0.5, 'No Training\nDegradation', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='green')
            ax.set_title('Training Impact Analysis', fontweight='bold')
            ax.axis('off')
            return
        
        # Group by scenario
        scenario_impacts = {}
        for p in problems['training_degradation']:
            scenario = p['scenario']
            if scenario not in scenario_impacts:
                scenario_impacts[scenario] = []
            scenario_impacts[scenario].append(p['degradation_pct'])
        
        scenarios = list(scenario_impacts.keys())
        avg_impacts = [np.mean(scenario_impacts[s]) for s in scenarios]
        
        # Plot
        colors = ['#e74c3c' if i > 30 else '#f39c12' if i > 10 else '#3498db' for i in avg_impacts]
        bars = ax.bar([s.replace('_', '\n').title() for s in scenarios], avg_impacts, 
                      color=colors, alpha=0.8)
        
        # Add value labels
        for bar, impact in zip(bars, avg_impacts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{impact:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Avg Fairness Degradation (%)', fontweight='bold', fontsize=11)
        ax.set_title('Training Impact on Fairness', fontweight='bold', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_recommendations(self, ax, distill_problems: Dict, inference_problems: Dict):
        """Plot recommendations summary."""
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'RECOMMENDATIONS', ha='center', fontsize=12, 
                fontweight='bold', transform=ax.transAxes)
        
        # Find worst cases
        recommendations = []
        
        if distill_problems['worst_cases']:
            worst_distill = sorted(distill_problems['worst_cases'], 
                                  key=lambda x: x['ratio'], reverse=True)[0]
            recommendations.append(f"[D] Distillation:\n   {worst_distill['feature'].replace('_', ' ').title()}")
            recommendations.append(f"   Gap: {worst_distill['worst_rmse'] - worst_distill['best_rmse']:.1f} RMSE")
        
        if inference_problems['worst_cases']:
            worst_inference = sorted(inference_problems['worst_cases'], 
                                    key=lambda x: x['ratio'], reverse=True)[0]
            recommendations.append(f"\n[I] Inference:\n   {worst_inference['feature'].replace('_', ' ').title()}")
            recommendations.append(f"   {worst_inference['scenario'].replace('_', ' ').title()}")
            recommendations.append(f"   Ratio: {worst_inference['ratio']:.2f}x")
        
        if inference_problems['poor_fairness']:
            recommendations.append(f"\n[!] {len(inference_problems['poor_fairness'])} POOR scenarios")
        
        if not recommendations:
            recommendations = ["[OK] No critical issues", "\n   All fairness ratios", "   within acceptable", "   ranges"]
        
        rec_text = '\n'.join(recommendations)
        
        ax.text(0.5, 0.5, rec_text, ha='center', va='center', fontsize=10,
                family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Add focus area
        if inference_problems['poor_fairness']:
            focus = "PRIORITY:\nFix Poor Scenarios"
            color = 'red'
        elif distill_problems['degraded_fairness']:
            focus = "PRIORITY:\nReduce Distillation\nDegradation"
            color = 'orange'
        else:
            focus = "PRIORITY:\nOptimize Small\nDisparities"
            color = 'green'
        
        ax.text(0.5, 0.1, focus, ha='center', fontsize=10, fontweight='bold',
                color=color, transform=ax.transAxes)
    
    def run(self):
        """Run complete investigation."""
        print("\n🔍 FAIRNESS ISSUE INVESTIGATOR")
        print("Finding problematic scenarios and groups for targeted improvement\n")
        
        try:
            # Analyze both contexts
            distill_problems = self.analyze_distillation()
            inference_problems = self.analyze_inference()
            
            # Print findings
            self.print_distillation_findings(distill_problems)
            self.print_inference_findings(inference_problems)
            
            # Provide recommendations
            self.recommend_focus_areas(distill_problems, inference_problems)
            
            # Generate visual report
            visual_report = self.generate_visual_report(distill_problems, inference_problems)
            
            print("\n" + "=" * 80)
            print("✅ Investigation Complete!")
            print(f"📊 Visual report: {visual_report}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during investigation: {e}")
            raise

if __name__ == "__main__":
    investigator = FairnessInvestigator()
    investigator.run()
