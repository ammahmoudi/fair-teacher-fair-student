#!/usr/bin/env python3
"""
🏆 LEGENDARY ALL-IN-ONE FAIRNESS ANALYZER 🏆

Analyzes fairness across ALL demographic features in your OhioT1DM dataset:
- Gender (male/female)
- Age Groups (20-40, 40-60, 60-80) 
- Pump Models (630G, 530G)
- Sensor Brands (Empatica, Basis)
- Study Cohorts (2020, 2018)

This is the ULTIMATE fairness analysis tool.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add project root to path dynamically
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.path_utils import get_project_root

class LegendaryFairnessAnalyzer:
    def __init__(self, data_path=None):
        """Initialize the LEGENDARY analyzer with OhioT1DM patient data"""
        if data_path is None:
            data_path = str(get_project_root() / "data" / "ohiot1dm" / "data.csv")
        self.data_path = data_path
        self.patient_data = self._load_patient_data()
        self.feature_names = {
            'Gender': '👫',
            'Age': '🎂', 
            'Pump Model': '💉',
            'Sensor Band': '📱',
            'Cohort': '📅'
        }
        
    def _load_patient_data(self):
        """Load ALL patient demographic features"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"🚀 LEGENDARY ANALYZER LOADED: {len(df)} patients")
            
            # Create comprehensive patient mapping
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
        except FileNotFoundError:
            print(f"⚠️  Patient data file not found at {self.data_path}")
            print("⚠️  Using default OhioT1DM patient metadata")
            # Create default patient metadata based on OhioT1DM dataset
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
        except Exception as e:
            print(f"💥 Error loading patient data: {e}")
            return {}
    
    def _load_patient_results(self, experiment_path):
        """Load patient results from experiment directory structure"""
        patient_results = {}
        
        # Find patient directories (handle nested structure)
        patient_dirs = []
        for root, dirs, files in os.walk(experiment_path):
            for d in dirs:
                if d.startswith('patient_'):
                    patient_dirs.append(os.path.join(root, d))
        
        for patient_dir in patient_dirs:
            try:
                patient_id = os.path.basename(patient_dir).split('_')[1]
                
                # Try to find distillation summary first, then student summary
                distill_file = os.path.join(patient_dir, "phase_3_distillation", "distillation_summary.json")
                student_file = os.path.join(patient_dir, "phase_2_student", "student_baseline_summary.json")
                teacher_file = os.path.join(patient_dir, "phase_1_teacher", "teacher_training_summary.json")
                
                if os.path.exists(distill_file):
                    with open(distill_file, 'r') as f:
                        data = json.load(f)
                        patient_results[patient_id] = {
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,
                            'mae': data.get('performance_metrics', {}).get('mae', 0),
                            'rmse': data.get('performance_metrics', {}).get('rmse', 0)
                        }
                elif os.path.exists(student_file):
                    with open(student_file, 'r') as f:
                        data = json.load(f)
                        patient_results[patient_id] = {
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,
                            'mae': data.get('performance_metrics', {}).get('mae', 0),
                            'rmse': data.get('performance_metrics', {}).get('rmse', 0)
                        }
                elif os.path.exists(teacher_file):
                    with open(teacher_file, 'r') as f:
                        data = json.load(f)
                        patient_results[patient_id] = {
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,
                            'mae': data.get('performance_metrics', {}).get('mae', 0),
                            'rmse': data.get('performance_metrics', {}).get('rmse', 0)
                        }
                        
            except Exception as e:
                patient_name = os.path.basename(patient_dir)
                print(f"⚠️  Could not load results for {patient_name}: {e}")
                continue
        
        return patient_results
    
    def analyze_latest(self):
        """Analyze ALL fairness metrics for the latest experiment"""
        
        distillation_dir = str(get_project_root() / "distillation_experiments")
        
        if not os.path.exists(distillation_dir):
            raise FileNotFoundError(f"💥 Distillation directory not found: {distillation_dir}")
        
        # Get latest experiment folder
        experiment_dirs = [d for d in os.listdir(distillation_dir) 
                          if os.path.isdir(os.path.join(distillation_dir, d))]
        
        if not experiment_dirs:
            raise FileNotFoundError("💥 No experiment directories found")
        
        latest_experiment = sorted(experiment_dirs)[-1]
        experiment_path = os.path.join(distillation_dir, latest_experiment)
        
        print(f"🎯 ANALYZING EXPERIMENT: {latest_experiment}")
        return self.analyze_experiment(experiment_path)
    
    def analyze_experiment(self, experiment_path):
        """🏆 THE LEGENDARY ANALYSIS FUNCTION"""
        
        # Load patient results from experiment directory
        patient_results = self._load_patient_results(experiment_path)
        
        print(f"\n🏆 LEGENDARY ALL-FEATURE FAIRNESS ANALYSIS")
        print("=" * 60)
        
        # Analyze each feature
        all_fairness_results = {}
        
        features_to_analyze = [
            ('gender', 'Gender'),
            ('age', 'Age Group'),
            ('pump_model', 'Pump Model'),
            ('sensor_band', 'Sensor Brand'),
            ('cohort', 'Cohort')
        ]
        
        for feature_key, feature_name in features_to_analyze:
            print(f"\n{self.feature_names.get(feature_name, '🔍')} {feature_name.upper()} FAIRNESS ANALYSIS")
            print("-" * 50)
            
            fairness_result = self._analyze_single_feature(patient_results, feature_key, feature_name)
            if fairness_result:
                all_fairness_results[feature_key] = fairness_result
        
        # LEGENDARY SUMMARY
        self._print_legendary_summary(all_fairness_results)
        
        return all_fairness_results
    
    def _analyze_single_feature(self, patient_results, feature_key, feature_name):
        """Analyze fairness for a single demographic feature"""
        
        # Group patients by feature
        feature_groups = defaultdict(list)
        
        # Collect results by feature
        for patient_id, patient_data in patient_results.items():
            if patient_id in self.patient_data:
                feature_value = self.patient_data[patient_id][feature_key]
                feature_groups[feature_value].append({
                    'patient_id': patient_id,
                    'mse': patient_data.get('mse', 0),
                    'mae': patient_data.get('mae', 0),
                    'rmse': patient_data.get('rmse', 0)
                })
        
        # Calculate performance for each group
        feature_performance = {}
        for feature_value, patients in feature_groups.items():
            if patients:
                group_mse = np.mean([p['mse'] for p in patients])
                group_mae = np.mean([p['mae'] for p in patients])
                
                feature_performance[feature_value] = {
                    'count': len(patients),
                    'mse': group_mse,
                    'mae': group_mae,
                    'patients': patients
                }
                
                print(f"   {feature_value}: {len(patients)} patients, MSE = {group_mse:.6f}")
        
        # Calculate fairness metrics
        if len(feature_performance) < 2:
            print("   ⚠️  Not enough groups for fairness analysis")
            return None
        
        mse_values = [perf['mse'] for perf in feature_performance.values()]
        mae_values = [perf['mae'] for perf in feature_performance.values()]
        
        # Fairness ratio (worst performing group / best performing group)
        mse_ratio = max(mse_values) / min(mse_values) if min(mse_values) > 0 else float('inf')
        mae_ratio = max(mae_values) / min(mae_values) if min(mae_values) > 0 else float('inf')
        
        # Coefficient of variation
        mse_cv = np.std(mse_values) / np.mean(mse_values) if np.mean(mse_values) > 0 else float('inf')
        
        # Fairness level
        fairness_level = self._classify_fairness_level(mse_ratio)
        
        print(f"   ⚖️  MSE Fairness Ratio: {mse_ratio:.3f}")
        print(f"   🎯 Fairness Level: {fairness_level}")
        
        return {
            'feature_name': feature_name,
            'feature_performance': feature_performance,
            'mse_ratio': mse_ratio,
            'mae_ratio': mae_ratio,
            'mse_cv': mse_cv,
            'fairness_level': fairness_level
        }
    
    def _classify_fairness_level(self, ratio):
        """Classify fairness level based on ratio"""
        if ratio <= 1.10:
            return "🏆 Excellent"
        elif ratio <= 1.25:
            return "👍 Good"
        elif ratio <= 1.50:
            return "⚠️  Acceptable"
        else:
            return "💥 Poor"
    
    def _print_legendary_summary(self, all_results):
        """Print the LEGENDARY summary of all fairness analyses"""
        
        print(f"\n🏆 LEGENDARY FAIRNESS SUMMARY ACROSS ALL FEATURES")
        print("=" * 60)
        
        if not all_results:
            print("💥 No fairness results to summarize")
            return
        
        # Sort by fairness ratio (best to worst)
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1]['mse_ratio'])
        
        print(f"{'Feature':<15} | {'Ratio':<6} | {'Level':<15} | {'Groups':<6}")
        print("-" * 60)
        
        for feature_key, result in sorted_results:
            feature_name = result['feature_name']
            ratio = result['mse_ratio']
            level = result['fairness_level']
            num_groups = len(result['feature_performance'])
            
            print(f"{feature_name:<15} | {ratio:<6.2f} | {level:<15} | {num_groups:<6}")
        
        # Find best and worst features
        best_feature = sorted_results[0]
        worst_feature = sorted_results[-1]
        
        print(f"\n🥇 MOST FAIR FEATURE: {best_feature[1]['feature_name']} (ratio: {best_feature[1]['mse_ratio']:.3f})")
        print(f"🔥 LEAST FAIR FEATURE: {worst_feature[1]['feature_name']} (ratio: {worst_feature[1]['mse_ratio']:.3f})")
        
        # Count fairness levels
        fairness_counts = {}
        for result in all_results.values():
            level = result['fairness_level']
            fairness_counts[level] = fairness_counts.get(level, 0) + 1
        
        print(f"\n📊 FAIRNESS LEVEL DISTRIBUTION:")
        for level, count in fairness_counts.items():
            print(f"   {level}: {count} features")
    
    def create_legendary_visualization(self, all_results):
        """🎨 Create the LEGENDARY visualization of all fairness results"""
        
        if not all_results:
            print("💥 No results to visualize")
            return
        
        # Create mega plot
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('🏆 LEGENDARY ALL-FEATURE FAIRNESS ANALYSIS 🏆', 
                     fontsize=20, fontweight='bold')
        
        # Extract data for plotting
        feature_names = [result['feature_name'] for result in all_results.values()]
        mse_ratios = [result['mse_ratio'] for result in all_results.values()]
        fairness_levels = [result['fairness_level'] for result in all_results.values()]
        
        # Color mapping for fairness levels
        color_map = {
            '🏆 Excellent': '#2ECC71',  # Green
            '👍 Good': '#F39C12',       # Orange  
            '⚠️  Acceptable': '#E67E22', # Dark Orange
            '💥 Poor': '#E74C3C'        # Red
        }
        colors = [color_map.get(level, '#95A5A6') for level in fairness_levels]
        
        # 1. MSE Fairness Ratios by Feature
        axes[0,0].bar(feature_names, mse_ratios, color=colors)
        axes[0,0].set_title('🎯 MSE Fairness Ratios by Feature', fontweight='bold')
        axes[0,0].set_ylabel('Fairness Ratio')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].axhline(y=1.10, color='green', linestyle='--', alpha=0.7, label='Excellent (≤1.10)')
        axes[0,0].axhline(y=1.25, color='orange', linestyle='--', alpha=0.7, label='Good (≤1.25)')
        axes[0,0].axhline(y=1.50, color='red', linestyle='--', alpha=0.7, label='Acceptable (≤1.50)')
        axes[0,0].legend()
        
        # Add value labels
        for i, v in enumerate(mse_ratios):
            axes[0,0].text(i, v + max(mse_ratios)*0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 2. Fairness Level Distribution
        level_counts = {}
        for level in fairness_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        axes[0,1].pie(level_counts.values(), 
                     labels=level_counts.keys(), 
                     autopct='%1.1f%%',
                     colors=[color_map.get(level, '#95A5A6') for level in level_counts.keys()])
        axes[0,1].set_title('📊 Fairness Level Distribution', fontweight='bold')
        
        # 3-6. Individual feature visualizations
        plot_positions = [(1,0), (1,1), (2,0), (2,1)]
        
        for idx, (feature_key, result) in enumerate(list(all_results.items())[:4]):
            if idx >= len(plot_positions):
                break
                
            row, col = plot_positions[idx]
            
            # Get feature performance data
            feature_perf = result['feature_performance']
            groups = list(feature_perf.keys())
            group_mses = [feature_perf[g]['mse'] for g in groups]
            group_counts = [feature_perf[g]['count'] for g in groups]
            
            # Create subplot
            ax = axes[row, col]
            
            # Bar plot of MSE by group
            colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(groups)))
            bars = ax.bar(groups, group_mses, color=colors)
            ax.set_title(f'{self.feature_names.get(result["feature_name"], "🔍")} {result["feature_name"]} MSE by Group', 
                        fontweight='bold')
            ax.set_ylabel('MSE')
            ax.tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar, count in zip(bars, group_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(group_mses)*0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        # Save comprehensive results
        self.results_dir = get_project_root() / "fairness" / "analysis_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        plot_file = self.results_dir / f"legendary_fairness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Legendary visualization saved to: {plot_file}")
    
    def save_legendary_results(self, all_results, filename="legendary_fairness_analysis.json"):
        """💾 Save the LEGENDARY results"""
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for feature_key, result in all_results.items():
            json_result = {}
            for key, value in result.items():
                if key == 'feature_performance':
                    json_result[key] = value
                elif isinstance(value, (float, int, np.number)):
                    json_result[key] = float(value)
                else:
                    json_result[key] = value
            json_results[feature_key] = json_result
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 LEGENDARY RESULTS SAVED: {filename}")

def main():
    """🚀 Main function to run the LEGENDARY fairness analysis"""
    
    analyzer = LegendaryFairnessAnalyzer()
    
    print("🏆 LEGENDARY ALL-FEATURE FAIRNESS ANALYZER")
    print("=" * 50)
    print(f"🚀 Patient data loaded: {len(analyzer.patient_data)} patients")
    
    # Show overall demographics
    print(f"\n📊 DEMOGRAPHIC OVERVIEW:")
    
    # Count by feature
    feature_counts = {
        'Gender': {},
        'Age': {},
        'Pump Model': {},
        'Sensor Band': {},
        'Cohort': {}
    }
    
    for patient_data in analyzer.patient_data.values():
        feature_counts['Gender'][patient_data['gender']] = feature_counts['Gender'].get(patient_data['gender'], 0) + 1
        feature_counts['Age'][patient_data['age']] = feature_counts['Age'].get(patient_data['age'], 0) + 1
        feature_counts['Pump Model'][patient_data['pump_model']] = feature_counts['Pump Model'].get(patient_data['pump_model'], 0) + 1
        feature_counts['Sensor Band'][patient_data['sensor_band']] = feature_counts['Sensor Band'].get(patient_data['sensor_band'], 0) + 1
        feature_counts['Cohort'][patient_data['cohort']] = feature_counts['Cohort'].get(patient_data['cohort'], 0) + 1
    
    for feature_name, counts in feature_counts.items():
        emoji = analyzer.feature_names.get(feature_name, '🔍')
        print(f"{emoji} {feature_name}: {dict(counts)}")
    
    try:
        # Run the LEGENDARY analysis
        all_results = analyzer.analyze_latest()
        
        if all_results:
            # Create LEGENDARY visualization
            analyzer.create_legendary_visualization(all_results)
            
            # Save LEGENDARY results
            analyzer.save_legendary_results(all_results)
            
            print(f"\n🎉 LEGENDARY ANALYSIS COMPLETE!")
            print(f"🏆 Analyzed fairness across {len(all_results)} demographic features")
            
        else:
            print("💥 Could not complete LEGENDARY analysis")
            
    except Exception as e:
        print(f"💥 Error during LEGENDARY analysis: {e}")

if __name__ == "__main__":
    main()