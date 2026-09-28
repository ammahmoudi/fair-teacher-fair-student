#!/usr/bin/env python3
"""
Base class for inference scenario fairness analysis.

Handles loading and analyzing fairness across different inference scenarios:
- Inference only (no training)
- Trained inference on standard data
- Trained inference on noisy data
- Trained inference on denoised data
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.path_utils import get_project_root


class BaseInferenceAnalyzer(ABC):
    """Base class for analyzing fairness across inference scenarios."""
    
    def __init__(self, feature_name: str, data_path: str = None):
        """
        Initialize the analyzer.
        
        Args:
            feature_name: Name of the demographic feature being analyzed
            data_path: Path to patient demographic data CSV
        """
        self.feature_name = feature_name
        self.data_path = data_path or self._get_default_data_path()
        self.patient_data = self._load_patient_data()
        self.results_dir = get_project_root() / "fairness" / "analysis_results" / "inference_scenarios"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define experiment paths
        self.experiments_dir = get_project_root() / "experiments"
        self.scenarios = {
            'inference_only': 'chronos_inference_ohiot1dm',
            'trained_standard': 'chronos_trained_inference_ohiot1dm',
            'trained_noisy': 'chronos_trained_inference_ohiot1dm_noisy',
            'trained_denoised': 'chronos_trained_inference_ohiot1dm_denoised'
        }
    
    def _get_default_data_path(self) -> str:
        """Get default path to patient demographic data."""
        return str(get_project_root() / "data" / "ohiot1dm" / "patient_demographics.csv")
    
    def _load_patient_data(self) -> Dict:
        """Load patient demographic data."""
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
            print(f"⚠️  Error loading patient data: {e}")
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
    
    @abstractmethod
    def get_feature_value(self, patient_id: str) -> str:
        """Get the feature value for a specific patient."""
        pass
    
    def load_scenario_results(self, scenario_key: str) -> Dict:
        """
        Load results for a specific scenario.
        
        Args:
            scenario_key: Key from self.scenarios
            
        Returns:
            Dictionary mapping patient_id to metrics
        """
        scenario_dir = self.experiments_dir / self.scenarios[scenario_key]
        csv_path = scenario_dir / "experiment_results.csv"
        
        if not csv_path.exists():
            print(f"⚠️  Results not found: {csv_path}")
            return {}
        
        try:
            df = pd.read_csv(csv_path)
            
            # Group by patient and average across seeds/models
            patient_results = {}
            for patient_id in df['patient_id'].unique():
                patient_df = df[df['patient_id'] == patient_id]
                
                patient_results[str(patient_id)] = {
                    'rmse': patient_df['rmse'].mean(),
                    'mae': patient_df['mae'].mean(),
                    'mape': patient_df['mape'].mean() if 'mape' in patient_df.columns else 0.0,
                    'count': len(patient_df)
                }
            
            print(f"✅ Loaded {scenario_key}: {len(patient_results)} patients")
            return patient_results
            
        except Exception as e:
            print(f"❌ Error loading {scenario_key}: {e}")
            return {}
    
    def load_all_scenarios(self) -> Dict[str, Dict]:
        """Load results for all scenarios."""
        all_results = {}
        for scenario_key in self.scenarios.keys():
            all_results[scenario_key] = self.load_scenario_results(scenario_key)
        return all_results
    
    def group_by_feature(self, scenario_results: Dict) -> Dict:
        """Group patient results by the demographic feature."""
        groups = defaultdict(list)
        
        for patient_id, metrics in scenario_results.items():
            if patient_id in self.patient_data:
                feature_value = self.get_feature_value(patient_id)
                groups[feature_value].append({
                    'patient_id': patient_id,
                    **metrics
                })
        
        return dict(groups)
    
    def calculate_group_statistics(self, grouped_results: Dict) -> Dict:
        """Calculate statistics for each group."""
        statistics = {}
        
        for group_name, patients in grouped_results.items():
            if not patients:
                continue
            
            rmse_values = [p['rmse'] for p in patients]
            mae_values = [p['mae'] for p in patients]
            
            statistics[group_name] = {
                'count': len(patients),
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values),
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'patients': [p['patient_id'] for p in patients]
            }
        
        return statistics
    
    def calculate_fairness_ratio(self, statistics: Dict) -> Tuple[float, str]:
        """
        Calculate fairness ratio (worst/best group performance).
        
        Returns:
            Tuple of (ratio, fairness_level)
        """
        if not statistics:
            return 0.0, "N/A"
        
        rmse_values = [stats['rmse_mean'] for stats in statistics.values()]
        
        if not rmse_values or min(rmse_values) == 0:
            return 0.0, "N/A"
        
        ratio = max(rmse_values) / min(rmse_values)
        
        # Fairness levels
        if ratio < 1.1:
            level = "EXCELLENT"
        elif ratio < 1.25:
            level = "GOOD"
        elif ratio < 1.5:
            level = "ACCEPTABLE"
        else:
            level = "POOR"
        
        return ratio, level
    
    def compare_scenarios(self, all_statistics: Dict[str, Dict]) -> Dict:
        """
        Compare fairness across scenarios.
        
        Args:
            all_statistics: Dict mapping scenario_key to group statistics
            
        Returns:
            Comparison results
        """
        comparison = {
            'fairness_ratios': {},
            'best_scenario': None,
            'worst_scenario': None
        }
        
        # Calculate fairness ratio for each scenario
        for scenario_key, stats in all_statistics.items():
            ratio, level = self.calculate_fairness_ratio(stats)
            comparison['fairness_ratios'][scenario_key] = {
                'ratio': ratio,
                'level': level
            }
        
        # Find best and worst
        ratios = {k: v['ratio'] for k, v in comparison['fairness_ratios'].items() if v['ratio'] > 0}
        if ratios:
            comparison['best_scenario'] = min(ratios, key=ratios.get)
            comparison['worst_scenario'] = max(ratios, key=ratios.get)
        
        return comparison
    
    def generate_timestamp(self) -> str:
        """Generate timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def print_scenario_summary(self, scenario_name: str, statistics: Dict, fairness_ratio: float, fairness_level: str):
        """Print summary for a single scenario."""
        print(f"\n{'='*70}")
        print(f"{scenario_name.upper().replace('_', ' ')} - {self.feature_name.upper()}")
        print(f"{'='*70}")
        
        for group_name, stats in statistics.items():
            print(f"\n  {group_name} ({stats['count']} patients):")
            print(f"    RMSE: {stats['rmse_mean']:.2f} ± {stats['rmse_std']:.2f}")
            print(f"    MAE:  {stats['mae_mean']:.2f} ± {stats['mae_std']:.2f}")
        
        print(f"\n  Fairness Ratio: {fairness_ratio:.2f}x ({fairness_level})")
    
    @abstractmethod
    def analyze(self):
        """Run the fairness analysis (to be implemented by subclasses)."""
        pass
