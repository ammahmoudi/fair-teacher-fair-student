#!/usr/bin/env python3
"""
Comprehensive Fairness Investigation Across All Scenarios
Analyzes fairness for:
1. Inference experiments (inference_only, trained_standard, trained_noisy, trained_denoised)
2. Distillation (teacher -> student -> distilled)
3. All-patients trained then inference mode

Includes Advanced Fairness Metrics:
- Demographic Parity Gap (DP Gap): Alert distribution equality
- Equal Opportunity Gap (EO Gap): Critical event detection equality
- Fairness Violation Objective (FVO): Maximum accuracy disparity
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from collections import defaultdict

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fairness.metrics.advanced_fairness_metrics import (
        AdvancedFairnessMetrics,
        CALC_MODE_SIMPLE,
        CALC_MODE_TIMELINE_RECONSTRUCTION,
        CALC_MODE_WINDOW_MAJORITY,
        VALID_CALC_MODES
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Advanced fairness metrics not available")
    ADVANCED_METRICS_AVAILABLE = False
    CALC_MODE_SIMPLE = 'simple'
    CALC_MODE_TIMELINE_RECONSTRUCTION = 'timeline_reconstruction'
    CALC_MODE_WINDOW_MAJORITY = 'window_majority'
    VALID_CALC_MODES = [CALC_MODE_SIMPLE, CALC_MODE_TIMELINE_RECONSTRUCTION, CALC_MODE_WINDOW_MAJORITY]

class ComprehensiveFairnessInvestigator:
    def __init__(self, calc_mode: str = CALC_MODE_WINDOW_MAJORITY):
        """
        Initialize the investigator.
        
        Args:
            calc_mode: Calculation mode for advanced fairness metrics
                      - 'simple': Direct calculation on flat arrays
                      - 'timeline_reconstruction': Average overlapping windows, then binarize
                      - 'window_majority': Binarize each window, use majority vote (default)
        """
        self.calc_mode = calc_mode
        self.results_dir = Path(__file__).parent.parent / "analysis_results"
        self.inference_dir = self.results_dir / "inference_scenarios"
        self.distillation_all_patients_dir = self.results_dir / "distillation_all_patients"
        self.distillation_per_patient_dir = self.results_dir / "distillation_per_patient"
        
        # Match constants from investigate_fairness_issues.py
        self.POOR_RATIO = 1.5  # Fairness ratio above this is POOR
        self.SIGNIFICANT_DEGRADATION = 0.1  # 10% degradation is significant
        
        # Initialize advanced fairness metrics if available
        if ADVANCED_METRICS_AVAILABLE:
            self.afm = AdvancedFairnessMetrics(
                hypoglycemia_threshold=70.0,
                hyperglycemia_threshold=180.0
            )
            print(f"✓ Advanced fairness metrics enabled (DP Gap, EO Gap, FVO)")
            print(f"  Calculation mode: {self.calc_mode}")
        else:
            self.afm = None
    
    def find_latest_report(self, directory: Path, pattern: str) -> Path:
        """Find the most recent report file."""
        reports = sorted(directory.glob(pattern))
        if not reports:
            raise FileNotFoundError(f"No reports found matching {pattern} in {directory}")
        return reports[-1]
    
    def load_inference_data(self) -> Dict:
        """Load inference scenario data."""
        report_file = self.find_latest_report(
            self.inference_dir,
            "legendary_inference_scenarios_report_*.json"
        )
        print(f"📂 Loading inference: {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def load_distillation_all_patients_data(self) -> Dict:
        """Load distillation all-patients data."""
        report_file = self.find_latest_report(
            self.distillation_all_patients_dir,
            "legendary_distillation_report_*.json"
        )
        print(f"📂 Loading distillation (all-patients): {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def load_distillation_per_patient_data(self) -> Dict:
        """Load distillation per-patient data."""
        report_file = self.find_latest_report(
            self.distillation_per_patient_dir,
            "legendary_distillation_report_*.json"
        )
        print(f"📂 Loading distillation (per-patient): {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def analyze_all_scenarios(self) -> Dict:
        """Analyze fairness across all scenarios."""
        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE FAIRNESS ANALYSIS")
        print("=" * 80)
        
        results = {
            'inference': {},
            'distillation_all_patients': {},
            'distillation_per_patient': {},
            'all_scenarios': [],
            'poor_fairness': [],
            'significant_degradation': [],
            'top_issues': []
        }
        
        # 1. Analyze inference scenarios
        print("\n📊 Analyzing inference scenarios...")
        inference_data = self.load_inference_data()
        results['inference'] = self._analyze_inference(inference_data)
        
        # 2. Analyze distillation (all-patients)
        print("📊 Analyzing distillation (all-patients)...")
        distillation_all_data = self.load_distillation_all_patients_data()
        results['distillation_all_patients'] = self._analyze_distillation(
            distillation_all_data, 'all_patients'
        )
        
        # 3. Analyze distillation (per-patient)
        print("📊 Analyzing distillation (per-patient)...")
        distillation_per_data = self.load_distillation_per_patient_data()
        results['distillation_per_patient'] = self._analyze_distillation(
            distillation_per_data, 'per_patient'
        )
        
        # 4. Compile all scenarios for comparison
        results['all_scenarios'] = self._compile_all_scenarios(results)
        
        # 5. Find poor fairness cases
        results['poor_fairness'] = [s for s in results['all_scenarios'] 
                                    if s['fairness_ratio'] >= self.POOR_RATIO]
        
        # 6. Find significant degradation cases
        results['significant_degradation'] = [s for s in results['all_scenarios'] 
                                              if s.get('degradation', 0) >= self.SIGNIFICANT_DEGRADATION]
        
        # 7. Get top issues
        results['top_issues'] = sorted(results['all_scenarios'], 
                                       key=lambda x: x['fairness_ratio'], 
                                       reverse=True)[:20]
        
        # 8. Calculate advanced fairness metrics if available
        if self.afm is not None:
            print("\n🔬 Calculating advanced fairness metrics (DP Gap, EO Gap, FVO)...")
            results['advanced_metrics'] = self._calculate_advanced_metrics_summary(
                inference_data,
                distillation_all_data,
                distillation_per_data
            )
        else:
            results['advanced_metrics'] = {}
        
        return results
    
    def _calculate_advanced_metrics_summary(self, 
                                           inference_data: Dict,
                                           distillation_all_data: Dict,
                                           distillation_per_data: Dict) -> Dict:
        """
        Calculate advanced fairness metrics from ACTUAL prediction data for ALL scenarios.
        Loads raw predictions and calculates DP Gap, EO Gap, FVO for:
        - All inference scenarios (inference_only, trained_standard, trained_noisy, trained_denoised)
        - Distillation all-patients (teacher, student, distilled)
        - Distillation per-patient (teacher, student, distilled)
        """
        print("  📊 Loading raw predictions for ALL scenarios...")
        
        advanced_results = {
            'inference': {},
            'distillation_all_patients': {},
            'distillation_per_patient': {}
        }
        
        try:
            from utils.path_utils import get_project_root
            project_root = get_project_root()
            
            # Load patient demographics
            patient_demographics = self._load_patient_demographics(project_root)
            if not patient_demographics:
                print("  ⚠️  No demographics loaded, falling back to ratio-based estimation")
                return self._calculate_metrics_from_ratios(inference_data, distillation_all_data, distillation_per_data)
            
            # 1. Process INFERENCE scenarios
            print("  📂 Processing inference scenarios...")
            inference_metrics = self._load_inference_predictions(project_root, patient_demographics)
            advanced_results['inference'] = inference_metrics
            
            # 2. Process DISTILLATION ALL-PATIENTS
            print("  📂 Processing distillation all-patients...")
            all_patients_metrics = self._load_distillation_all_patients_predictions(project_root, patient_demographics)
            advanced_results['distillation_all_patients'] = all_patients_metrics
            
            # 3. Process DISTILLATION PER-PATIENT
            print("  📂 Processing distillation per-patient...")
            per_patient_metrics = self._load_distillation_per_patient_predictions(project_root, patient_demographics)
            advanced_results['distillation_per_patient'] = per_patient_metrics
            
            total_metrics = sum(len(v) for v in advanced_results.values())
            print(f"  ✅ Calculated {total_metrics} advanced metric sets from ACTUAL predictions across ALL scenarios")
            
            return advanced_results
            
        except Exception as e:
            print(f"  ⚠️  Error loading predictions: {e}")
            import traceback
            traceback.print_exc()
            print("  ℹ️  Falling back to ratio-based estimation")
            return self._calculate_metrics_from_ratios(inference_data, distillation_all_data, distillation_per_data)
    
    def _load_patient_demographics(self, project_root: Path) -> Dict:
        """Load patient demographic data."""
        demographics = {}
        data_file = project_root / "data" / "ohiot1dm" / "data.csv"
        
        try:
            df = pd.read_csv(data_file)
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                demographics[patient_id] = {
                    'gender': row['Gender'].lower(),
                    'age': row['Age'],
                    'pump_model': row['Pump Model'],
                    'sensor_band': row['Sensor Band'],
                    'cohort': str(row['Cohort'])
                }
            print(f"  ✅ Loaded demographics for {len(demographics)} patients")
        except Exception as e:
            print(f"  ⚠️  Could not load demographics: {e}")
        
        return demographics
    
    def _load_inference_predictions(self, project_root: Path, demographics: Dict) -> Dict:
        """Load predictions from Time-LLM inference experiments."""
        inference_metrics = {}
        experiments_dir = project_root / "experiments"
        
        # Map scenario names to Time-LLM experiment directories
        scenarios = {
            'inference_only': 'time_llm_inference_ohiot1dm',
            'trained_standard': 'time_llm_training_inference_ohiot1dm',
            'trained_noisy': 'time_llm_training_inference_ohiot1dm_train_standardized_test_noisy',
            'trained_denoised': 'time_llm_training_inference_ohiot1dm_train_standardized_test_denoised'
        }
        
        for scenario_key, scenario_dir_name in scenarios.items():
            scenario_dir = experiments_dir / scenario_dir_name
            
            if not scenario_dir.exists():
                print(f"    ❌ DATA NOT AVAILABLE: {scenario_dir_name}")
                print(f"       Path checked: {scenario_dir}")
                print(f"       → Please run experiments for this scenario")
                continue
            
            # Load predictions for this scenario
            predictions_data = self._load_predictions_from_directory(scenario_dir, demographics)
            
            if not predictions_data:
                print(f"    ❌ NO PREDICTIONS FOUND: {scenario_key}")
                print(f"       Directory exists but no inference_results.csv files found")
                print(f"       → Check if predictions were generated in: {scenario_dir}")
                continue
            
            print(f"    ✅ Loaded {len(predictions_data)} patients from {scenario_key}")
            
            # Calculate metrics for each demographic feature
            for demo_feature in ['gender', 'age', 'cohort', 'pump_model', 'sensor_band']:
                metrics = self._calculate_metrics_for_feature(predictions_data, demo_feature, demographics)
                if metrics:
                    key = f"{demo_feature}_{scenario_key}"
                    inference_metrics[key] = metrics
                else:
                    print(f"       ⚠️  Insufficient groups for {demo_feature} in {scenario_key}")
        
        print(f"    ✅ Calculated {len(inference_metrics)} inference metric sets")
        return inference_metrics
    
    def _load_distillation_all_patients_predictions(self, project_root: Path, demographics: Dict) -> Dict:
        """Load predictions from distillation all-patients pipeline."""
        all_patients_metrics = {}
        experiments_dir = project_root / "distillation_experiments" / "all_patients_pipeline"
        
        # Find latest pipeline
        pipeline_dirs = sorted([d for d in experiments_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('pipeline_')])
        
        if not pipeline_dirs:
            return {}
        
        latest_pipeline = pipeline_dirs[-1]
        print(f"    Using pipeline: {latest_pipeline.name}")
        
        # Load predictions for each phase
        phases = {
            'teacher': 'phase_1_teacher',
            'student': 'phase_2_student',
            'distilled': 'phase_3_distillation'
        }
        
        for phase_key, phase_dir_name in phases.items():
            phase_dir = latest_pipeline / phase_dir_name / "per_patient_inference"
            
            if not phase_dir.exists():
                continue
            
            predictions_data = self._load_predictions_from_directory(phase_dir, demographics)
            
            if not predictions_data:
                continue
            
            # Calculate metrics for each demographic feature
            for demo_feature in ['gender', 'age', 'cohort', 'pump_model', 'sensor_band']:
                metrics = self._calculate_metrics_for_feature(predictions_data, demo_feature, demographics)
                if metrics:
                    key = f"{demo_feature}_{phase_key}"
                    all_patients_metrics[key] = metrics
        
        print(f"    ✅ Calculated {len(all_patients_metrics)} all-patients metric sets")
        return all_patients_metrics
    
    def _load_distillation_per_patient_predictions(self, project_root: Path, demographics: Dict) -> Dict:
        """Load predictions from distillation per-patient experiments."""
        per_patient_metrics = {}
        experiments_dir = project_root / "distillation_experiments"
        
        # Find latest pipeline (not in all_patients_pipeline)
        pipeline_dirs = sorted([d for d in experiments_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('pipeline_') 
                               and d.name != 'all_patients_pipeline'])
        
        if not pipeline_dirs:
            # Try looking in subdirectories
            for subdir in experiments_dir.iterdir():
                if subdir.is_dir() and subdir.name != 'all_patients_pipeline':
                    pipeline_dirs = sorted([d for d in subdir.iterdir() 
                                           if d.is_dir() and d.name.startswith('pipeline_')])
                    if pipeline_dirs:
                        break
        
        if not pipeline_dirs:
            print("    ⚠️  No per-patient pipelines found")
            return {}
        
        latest_pipeline = pipeline_dirs[-1]
        print(f"    Using pipeline: {latest_pipeline.name}")
        
        # For per-patient, predictions are in patient_XXX directories
        predictions_by_phase = {'teacher': {}, 'student': {}, 'distilled': {}}
        
        # Find all patient directories
        for patient_dir in latest_pipeline.rglob("patient_*"):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name.split('_')[1]
            
            if patient_id not in demographics:
                continue
            
            # Load predictions from each phase for this patient
            # First try CSV files (teacher and student phases)
            for inference_csv in patient_dir.rglob("inference_results.csv"):
                # Determine phase from path
                path_str = str(inference_csv.parent)
                
                phase_key = None
                if 'teacher' in path_str or 'phase_1' in path_str:
                    phase_key = 'teacher'
                elif 'student' in path_str or 'phase_2' in path_str:
                    phase_key = 'student'
                elif 'distill' in path_str or 'phase_3' in path_str:
                    phase_key = 'distilled'
                
                if not phase_key:
                    continue
                
                try:
                    df = pd.read_csv(inference_csv)
                    
                    y_true_list = []
                    y_pred_list = []
                    
                    for _, row in df.iterrows():
                        gt_str = str(row['ground_truth'])
                        pred_str = str(row['predictions'])
                        
                        gt_vals = [float(v.strip()) for v in gt_str.split(',') if v.strip()]
                        pred_vals = [float(v.strip()) for v in pred_str.split(',') if v.strip()]
                        
                        y_true_list.extend(gt_vals)
                        y_pred_list.extend(pred_vals)
                    
                    if y_true_list and y_pred_list:
                        predictions_by_phase[phase_key][patient_id] = {
                            'y_true': np.array(y_true_list),
                            'y_pred': np.array(y_pred_list)
                        }
                
                except Exception as e:
                    continue
            
            # Also try PKL files for distilled phase (phase_3)
            for predictions_pkl in patient_dir.rglob("predictions.pkl"):
                path_str = str(predictions_pkl.parent)
                
                # Only use pkl files from phase_3_distillation
                if 'phase_3' not in path_str and 'distill' not in path_str:
                    continue
                
                targets_pkl = predictions_pkl.parent / "targets.pkl"
                if not targets_pkl.exists():
                    continue
                
                try:
                    import pickle
                    with open(predictions_pkl, 'rb') as f:
                        predictions = pickle.load(f)
                    with open(targets_pkl, 'rb') as f:
                        targets = pickle.load(f)
                    
                    # Flatten the arrays (shape is typically (n_windows, prediction_length, 1))
                    y_pred = predictions.flatten()
                    y_true = targets.flatten()
                    
                    if len(y_pred) > 0 and len(y_true) > 0:
                        predictions_by_phase['distilled'][patient_id] = {
                            'y_true': y_true,
                            'y_pred': y_pred
                        }
                
                except Exception as e:
                    continue
        
        # Calculate metrics for each phase
        for phase_key, phase_predictions in predictions_by_phase.items():
            if not phase_predictions:
                continue
            
            for demo_feature in ['gender', 'age', 'cohort', 'pump_model', 'sensor_band']:
                metrics = self._calculate_metrics_for_feature(phase_predictions, demo_feature, demographics)
                if metrics:
                    key = f"{demo_feature}_{phase_key}"
                    per_patient_metrics[key] = metrics
        
        print(f"    ✅ Calculated {len(per_patient_metrics)} per-patient metric sets")
        return per_patient_metrics
    
    def _load_predictions_from_directory(self, directory: Path, demographics: Dict) -> Dict:
        """Load predictions from inference_results.csv files in a directory."""
        predictions = {}
        
        for inference_csv in directory.rglob("inference_results.csv"):
            # Extract patient_id from path
            path_parts = str(inference_csv.parent).split('/')
            patient_folder = [p for p in path_parts if p.startswith('patient_')]
            
            if not patient_folder:
                continue
            
            patient_id = patient_folder[0].split('_')[1]
            
            if patient_id not in demographics:
                continue
            
            try:
                df = pd.read_csv(inference_csv)
                
                y_true_list = []
                y_pred_list = []
                
                for _, row in df.iterrows():
                    gt_str = str(row['ground_truth'])
                    pred_str = str(row['predictions'])
                    
                    gt_vals = [float(v.strip()) for v in gt_str.split(',') if v.strip()]
                    pred_vals = [float(v.strip()) for v in pred_str.split(',') if v.strip()]
                    
                    y_true_list.extend(gt_vals)
                    y_pred_list.extend(pred_vals)
                
                if y_true_list and y_pred_list:
                    predictions[patient_id] = {
                        'y_true': np.array(y_true_list),
                        'y_pred': np.array(y_pred_list)
                    }
            
            except Exception as e:
                continue
        
        return predictions
    
    def _calculate_metrics_for_feature(self, predictions_data: Dict, demo_feature: str, demographics: Dict) -> Optional[Dict]:
        """Calculate advanced metrics for a specific demographic feature."""
        if not ADVANCED_METRICS_AVAILABLE:
            return None
        
        # Group predictions by demographic feature
        groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})
        
        for patient_id, pred_data in predictions_data.items():
            if patient_id not in demographics:
                continue
            
            demo_value = demographics[patient_id].get(demo_feature)
            if not demo_value:
                continue
            
            groups[demo_value]['y_true'].extend(pred_data['y_true'])
            groups[demo_value]['y_pred'].extend(pred_data['y_pred'])
        
        if len(groups) < 2:
            return None
        
        # Combine all groups
        all_y_true = []
        all_y_pred = []
        all_group_labels = []
        
        for group_name, group_data in groups.items():
            all_y_true.extend(group_data['y_true'])
            all_y_pred.extend(group_data['y_pred'])
            all_group_labels.extend([str(group_name)] * len(group_data['y_true']))
        
        if not all_y_true:
            return None
        
        # Convert to numpy arrays
        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        group_labels_arr = np.array(all_group_labels)
        
        try:
            # Calculate metrics with current calc_mode
            dp_result = self.afm.demographic_parity_gap(
                y_pred_arr, group_labels_arr, 
                calc_mode=self.calc_mode
            )
            eo_result = self.afm.equal_opportunity_gap(
                y_true_arr, y_pred_arr, group_labels_arr,
                calc_mode=self.calc_mode
            )
            fvo_result = self.afm.fairness_violation_objective(
                y_true_arr, y_pred_arr, group_labels_arr,
                calc_mode=self.calc_mode
            )
            
            # Extract values
            dp_gap = dp_result['dp_gap'] if isinstance(dp_result, dict) else dp_result
            eo_gap = eo_result['eo_gap'] if isinstance(eo_result, dict) else eo_result
            fvo = fvo_result['fvo'] if isinstance(fvo_result, dict) else fvo_result
            
            # Determine assessment
            critical_count = sum([dp_gap > 0.20, eo_gap > 0.20, fvo > 0.10])
            
            if critical_count >= 2:
                assessment = '❌ SIGNIFICANT CONCERNS'
            elif eo_gap > 0.20:
                assessment = '❌ CRITICAL (High EO Gap)'
            elif critical_count == 1 or eo_gap > 0.15:
                assessment = '⚠️ MODERATE CONCERNS'
            elif dp_gap > 0.10 or eo_gap > 0.10 or fvo > 0.05:
                assessment = '⚠️ MINOR CONCERNS'
            else:
                assessment = '✅ FAIR'
            
            return {
                'dp_gap': round(float(dp_gap), 4),
                'eo_gap': round(float(eo_gap), 4),
                'fvo': round(float(fvo), 4),
                'assessment': assessment,
                'note': 'Calculated from actual predictions',
                'n_samples': len(y_true_arr),
                'n_groups': len(groups)
            }
        
        except Exception as e:
            print(f"    ⚠️  Error calculating {demo_feature} metrics: {e}")
            return None

    def _load_all_predictions(self, pipeline_dir: Path, demographics: Dict) -> Dict:
        """Load raw predictions from all phases."""
        predictions_by_phase = {
            'phase_1_teacher': {},
            'phase_3_distillation': {}
        }
        
        for phase_name in ['phase_1_teacher', 'phase_3_distillation']:
            phase_dir = pipeline_dir / phase_name / "per_patient_inference"
            
            if not phase_dir.exists():
                continue
            
            # Find all inference_results.csv files
            for inference_csv in phase_dir.rglob("inference_results.csv"):
                # Extract patient_id from path
                path_parts = str(inference_csv.parent).split('/')
                patient_folder = [p for p in path_parts if p.startswith('patient_')]
                
                if not patient_folder:
                    continue
                
                patient_id = patient_folder[0].split('_')[1]
                
                if patient_id not in demographics:
                    continue
                
                try:
                    # Load predictions
                    df = pd.read_csv(inference_csv)
                    
                    y_true_list = []
                    y_pred_list = []
                    
                    for _, row in df.iterrows():
                        # Parse ground_truth and predictions
                        gt_str = str(row['ground_truth'])
                        pred_str = str(row['predictions'])
                        
                        # Parse comma-separated values
                        gt_vals = [float(v.strip()) for v in gt_str.split(',') if v.strip()]
                        pred_vals = [float(v.strip()) for v in pred_str.split(',') if v.strip()]
                        
                        y_true_list.extend(gt_vals)
                        y_pred_list.extend(pred_vals)
                    
                    if y_true_list and y_pred_list:
                        predictions_by_phase[phase_name][patient_id] = {
                            'y_true': np.array(y_true_list),
                            'y_pred': np.array(y_pred_list),
                            'demographics': demographics[patient_id]
                        }
                
                except Exception as e:
                    print(f"  ⚠️  Error loading predictions for patient {patient_id}: {e}")
                    continue
            
            print(f"  ✅ Loaded predictions for {len(predictions_by_phase[phase_name])} patients in {phase_name}")
        
        return predictions_by_phase
    
    def _calculate_metrics_from_predictions(self, predictions_data: Dict) -> Dict:
        """Calculate advanced metrics from actual predictions."""
        if not ADVANCED_METRICS_AVAILABLE:
            print("  ⚠️  Advanced metrics module not available")
            return {'inference': {}, 'distillation_all_patients': {}, 'distillation_per_patient': {}}
        
        advanced_results = {
            'inference': {},
            'distillation_all_patients': {},
            'distillation_per_patient': {}
        }
        
        # Calculate for each phase
        for phase_name, phase_data in predictions_data.items():
            if not phase_data:
                continue
            
            phase_key = 'teacher' if 'teacher' in phase_name else 'distilled'
            
            # Group by demographics
            for demo_feature in ['gender', 'age']:
                groups = defaultdict(lambda: {'y_true': [], 'y_pred': [], 'patient_ids': []})
                
                for patient_id, patient_data in phase_data.items():
                    demo_value = patient_data['demographics'].get(demo_feature)
                    if demo_value:
                        groups[demo_value]['y_true'].extend(patient_data['y_true'])
                        groups[demo_value]['y_pred'].extend(patient_data['y_pred'])
                        groups[demo_value]['patient_ids'].append(patient_id)
                
                if len(groups) < 2:
                    continue
                
                # Combine all groups to calculate overall metrics
                all_y_true = []
                all_y_pred = []
                all_group_labels = []
                
                for group_name, group_data in groups.items():
                    group_y_true = group_data['y_true']
                    group_y_pred = group_data['y_pred']
                    
                    all_y_true.extend(group_y_true)
                    all_y_pred.extend(group_y_pred)
                    all_group_labels.extend([group_name] * len(group_y_true))
                
                if not all_y_true:
                    continue
                
                # Convert to numpy arrays
                y_true_arr = np.array(all_y_true)
                y_pred_arr = np.array(all_y_pred)
                group_labels_arr = np.array(all_group_labels)
                
                try:
                    # Calculate metrics using the actual advanced fairness metrics
                    # These methods return dictionaries with the metric value and metadata
                    dp_result = self.afm.demographic_parity_gap(y_pred_arr, group_labels_arr)
                    eo_result = self.afm.equal_opportunity_gap(y_true_arr, y_pred_arr, group_labels_arr)
                    fvo_result = self.afm.fairness_violation_objective(y_true_arr, y_pred_arr, group_labels_arr)
                    
                    # Extract the numeric values
                    dp_gap = dp_result['dp_gap'] if isinstance(dp_result, dict) else dp_result
                    eo_gap = eo_result['eo_gap'] if isinstance(eo_result, dict) else eo_result
                    fvo = fvo_result['fvo'] if isinstance(fvo_result, dict) else fvo_result
                    
                    # Determine assessment
                    critical_count = sum([dp_gap > 0.20, eo_gap > 0.20, fvo > 0.10])
                    
                    if critical_count >= 2:
                        assessment = '❌ SIGNIFICANT CONCERNS'
                    elif eo_gap > 0.20:
                        assessment = '❌ CRITICAL (High EO Gap)'
                    elif critical_count == 1 or eo_gap > 0.15:
                        assessment = '⚠️ MODERATE CONCERNS'
                    elif dp_gap > 0.10 or eo_gap > 0.10 or fvo > 0.05:
                        assessment = '⚠️ MINOR CONCERNS'
                    else:
                        assessment = '✅ FAIR'
                    
                    key = f"{demo_feature}_{phase_key}"
                    advanced_results['distillation_all_patients'][key] = {
                        'dp_gap': round(float(dp_gap), 4),
                        'eo_gap': round(float(eo_gap), 4),
                        'fvo': round(float(fvo), 4),
                        'assessment': assessment,
                        'note': 'Calculated from actual predictions',
                        'n_samples': len(y_true_arr),
                        'n_groups': len(groups)
                    }
                
                except Exception as e:
                    print(f"    ⚠️  Error calculating {demo_feature} metrics for {phase_key}: {e}")
                    continue
        
        return advanced_results
    
    def _calculate_metrics_from_ratios(self, inference_data: Dict, distillation_all_data: Dict, 
                                      distillation_per_data: Dict) -> Dict:
        """Fallback: Calculate advanced metrics from fairness ratios."""
        advanced_results = {
            'inference': {},
            'distillation_all_patients': {},
            'distillation_per_patient': {}
        }
        
        def calculate_from_ratio(fairness_ratio: float) -> Dict:
            disparity = fairness_ratio - 1.0
            dp_gap = min(0.30, max(0.0, disparity * 0.35))
            eo_gap = min(0.50, max(0.0, disparity * 0.55))
            fvo = min(0.25, max(0.0, disparity * 0.28))
            
            critical_count = sum([dp_gap > 0.20, eo_gap > 0.20, fvo > 0.10])
            if critical_count >= 2:
                assessment = '❌ SIGNIFICANT CONCERNS'
            elif eo_gap > 0.20:
                assessment = '❌ CRITICAL (High EO Gap)'
            elif critical_count == 1 or eo_gap > 0.15:
                assessment = '⚠️ MODERATE CONCERNS'
            elif dp_gap > 0.10 or eo_gap > 0.10 or fvo > 0.05:
                assessment = '⚠️ MINOR CONCERNS'
            else:
                assessment = '✅ FAIR'
            
            return {
                'dp_gap': round(dp_gap, 4),
                'eo_gap': round(eo_gap, 4),
                'fvo': round(fvo, 4),
                'assessment': assessment,
                'fairness_ratio': round(fairness_ratio, 3),
                'note': 'Estimated from RMSE fairness ratios'
            }
        
        # Process all scenarios
        if 'results' in inference_data:
            for feature_name, feature_data in inference_data['results'].items():
                for scenario_name, scenario_data in feature_data.items():
                    if isinstance(scenario_data, dict) and 'fairness_ratio' in scenario_data:
                        key = f"{feature_name.lower().replace(' ', '_')}_{scenario_name}"
                        advanced_results['inference'][key] = calculate_from_ratio(scenario_data['fairness_ratio'])
        
        if 'feature_analysis' in distillation_all_data:
            for feature_name, feature_data in distillation_all_data['feature_analysis'].items():
                for phase in ['teacher', 'distilled']:
                    ratio_key = f'{phase}_fairness_ratio'
                    if ratio_key in feature_data:
                        key = f"{feature_name.lower().replace(' ', '_')}_{phase}"
                        advanced_results['distillation_all_patients'][key] = calculate_from_ratio(feature_data[ratio_key])
        
        if 'feature_analysis' in distillation_per_data:
            for feature_name, feature_data in distillation_per_data['feature_analysis'].items():
                for phase in ['teacher', 'distilled']:
                    ratio_key = f'{phase}_fairness_ratio'
                    if ratio_key in feature_data:
                        key = f"{feature_name.lower().replace(' ', '_')}_{phase}"
                        advanced_results['distillation_per_patient'][key] = calculate_from_ratio(feature_data[ratio_key])
        
        return advanced_results
    
    def _analyze_inference(self, data: Dict) -> Dict:
        """Analyze inference scenarios."""
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        scenarios = ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']
        results = []
        
        for display_name, feature_key in feature_map.items():
            if display_name not in data['results']:
                continue
            
            feature_data = data['results'][display_name]
            
            for scenario in scenarios:
                if scenario not in feature_data:
                    continue
                
                scenario_data = feature_data[scenario]
                
                # Get baseline for comparison
                baseline_ratio = feature_data['inference_only']['fairness_ratio']
                degradation = scenario_data['fairness_ratio'] - baseline_ratio if scenario != 'inference_only' else 0
                
                # Get worst and best groups
                if 'groups' in scenario_data:
                    groups = scenario_data['groups']
                    rmse_values = [(group_name, group_data['rmse_mean']) 
                                 for group_name, group_data in groups.items() 
                                 if 'rmse_mean' in group_data]
                    
                    if rmse_values:
                        worst_group = max(rmse_values, key=lambda x: x[1])
                        best_group = min(rmse_values, key=lambda x: x[1])
                        
                        results.append({
                            'context': 'inference',
                            'scenario': scenario,
                            'feature': feature_key,
                            'fairness_ratio': scenario_data['fairness_ratio'],
                            'baseline_ratio': baseline_ratio,
                            'degradation': degradation,
                            'degradation_pct': (degradation / baseline_ratio * 100) if baseline_ratio > 0 else 0,
                            'worst_group': worst_group[0],
                            'worst_rmse': worst_group[1],
                            'best_group': best_group[0],
                            'best_rmse': best_group[1],
                            'gap': worst_group[1] - best_group[1]
                        })
        
        return {'scenarios': results}
    
    def _analyze_distillation(self, data: Dict, mode: str) -> Dict:
        """Analyze distillation scenarios.
        
        Args:
            data: The distillation data
            mode: Either 'all_patients' or 'per_patient'
        """
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        # Three phases: teacher, student_baseline, distilled
        phases = ['teacher', 'student_baseline', 'distilled']
        results = []
        
        if 'feature_analysis' not in data:
            return {'scenarios': results}
        
        for display_name, feature_key in feature_map.items():
            if display_name not in data['feature_analysis']:
                continue
            
            feature_data = data['feature_analysis'][display_name]
            
            # For each phase, we need to calculate fairness ratios
            # Teacher and distilled are in feature_analysis
            # Student_baseline needs to be calculated from individual reports
            
            for phase in ['teacher', 'distilled']:
                phase_ratio_key = f'{phase}_fairness_ratio'
                if phase_ratio_key not in feature_data:
                    continue
                
                fairness_ratio = feature_data[phase_ratio_key]
                
                # Get baseline (teacher)
                baseline_ratio = feature_data['teacher_fairness_ratio']
                degradation = fairness_ratio - baseline_ratio if phase != 'teacher' else 0
                
                # Get per-group impacts
                worst_group = None
                best_group = None
                worst_rmse = None
                best_rmse = None
                
                if 'per_group_impacts' in data and display_name in data['per_group_impacts']:
                    group_impacts = data['per_group_impacts'][display_name]
                    rmse_key = f'{phase}_rmse'
                    
                    rmse_values = [(group_name, group_data.get(rmse_key, 0)) 
                                 for group_name, group_data in group_impacts.items() 
                                 if rmse_key in group_data]
                    
                    if rmse_values:
                        worst = max(rmse_values, key=lambda x: x[1])
                        best = min(rmse_values, key=lambda x: x[1])
                        worst_group, worst_rmse = worst
                        best_group, best_rmse = best
                
                results.append({
                    'context': f'distillation_{mode}',
                    'scenario': phase,
                    'feature': feature_key,
                    'fairness_ratio': fairness_ratio,
                    'baseline_ratio': baseline_ratio,
                    'degradation': degradation,
                    'degradation_pct': (degradation / baseline_ratio * 100) if baseline_ratio > 0 else 0,
                    'worst_group': worst_group,
                    'worst_rmse': worst_rmse,
                    'best_group': best_group,
                    'best_rmse': best_rmse,
                    'gap': (worst_rmse - best_rmse) if (worst_rmse and best_rmse) else 0
                })
            
            # Calculate student_baseline fairness ratio from groups data
            # This requires loading individual feature report
            student_ratio = self._calculate_student_fairness_ratio(display_name, mode)
            if student_ratio:
                results.append({
                    'context': f'distillation_{mode}',
                    'scenario': 'student_baseline',
                    'feature': feature_key,
                    'fairness_ratio': student_ratio,
                    'baseline_ratio': feature_data['teacher_fairness_ratio'],
                    'degradation': student_ratio - feature_data['teacher_fairness_ratio'],
                    'degradation_pct': ((student_ratio - feature_data['teacher_fairness_ratio']) / feature_data['teacher_fairness_ratio'] * 100),
                    'worst_group': None,
                    'worst_rmse': None,
                    'best_group': None,
                    'best_rmse': None,
                    'gap': 0
                })
        
        return {'scenarios': results}
    
    def _calculate_student_fairness_ratio(self, feature_name: str, mode: str) -> Optional[float]:
        """Calculate student_baseline fairness ratio from groups data."""
        try:
            # Map feature display names to file names
            file_map = {
                'Gender': 'gender',
                'Age Group': 'age_group',
                'Pump Model': 'pump_model',
                'Sensor Band': 'sensor_band',
                'Cohort': 'cohort'
            }
            
            file_name = file_map.get(feature_name)
            if not file_name:
                return None
            
            # Load the individual feature report
            if mode == 'all_patients':
                dir_path = self.distillation_all_patients_dir
            else:
                dir_path = self.distillation_per_patient_dir
            
            pattern = f"{file_name}_fairness_report_{mode}_*.json"
            reports = sorted(dir_path.glob(pattern))
            if not reports:
                return None
            
            with open(reports[-1], 'r') as f:
                data = json.load(f)
            
            # Extract student_baseline RMSE for each group
            if 'groups' not in data:
                return None
            
            student_rmses = []
            for group_name, group_data in data['groups'].items():
                if 'phases' in group_data and 'student_baseline' in group_data['phases']:
                    student_rmse = group_data['phases']['student_baseline'].get('rmse_mean', 0)
                    if student_rmse > 0:
                        student_rmses.append(student_rmse)
            
            if len(student_rmses) >= 2:
                # Fairness ratio = worst / best
                return max(student_rmses) / min(student_rmses)
            
            return None
            
        except Exception as e:
            print(f"  Warning: Could not calculate student fairness for {feature_name} ({mode}): {e}")
            return None
    
    def _compile_all_scenarios(self, results: Dict) -> List[Dict]:
        """Compile all scenarios into one list from inference and both distillation types."""
        all_scenarios = []
        
        # Add inference scenarios
        for s in results['inference']['scenarios']:
            all_scenarios.append(s)
        
        # Add distillation all_patients scenarios
        for s in results['distillation_all_patients']['scenarios']:
            all_scenarios.append(s)
        
        # Add distillation per_patient scenarios
        for s in results['distillation_per_patient']['scenarios']:
            all_scenarios.append(s)
        
        return all_scenarios
    
    def print_summary(self, results: Dict):
        """Print comprehensive summary."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE SUMMARY")
        print("=" * 80)
        
        total_scenarios = len(results['all_scenarios'])
        inference_count = len(results['inference']['scenarios'])
        distillation_all_count = len(results['distillation_all_patients']['scenarios'])
        distillation_per_count = len(results['distillation_per_patient']['scenarios'])
        
        print(f"\n📈 Total Scenarios Analyzed: {total_scenarios}")
        print(f"   - Inference: {inference_count}")
        print(f"   - Distillation (All Patients): {distillation_all_count}")
        print(f"   - Distillation (Per Patient): {distillation_per_count}")
        
        # Poor fairness cases
        print(f"\n🔴 Poor Fairness Cases (≥{self.POOR_RATIO}): {len(results['poor_fairness'])}")
        if results['poor_fairness']:
            for case in sorted(results['poor_fairness'], key=lambda x: x['fairness_ratio'], reverse=True):
                print(f"   • {case['context'].upper()} - {case['feature'].replace('_', ' ').title()}")
                print(f"     Scenario: {case['scenario'].replace('_', ' ').title()}")
                print(f"     Fairness Ratio: {case['fairness_ratio']:.3f}x")
                if case['worst_group'] and case['best_group']:
                    print(f"     {case['worst_group']} vs {case['best_group']}: {case['gap']:.2f} RMSE gap")
        
        # Significant degradation
        print(f"\n⚠️  Significant Degradation (≥{self.SIGNIFICANT_DEGRADATION}): {len(results['significant_degradation'])}")
        if results['significant_degradation']:
            for case in sorted(results['significant_degradation'], key=lambda x: x['degradation'], reverse=True)[:10]:
                print(f"   • {case['context'].upper()} - {case['feature'].replace('_', ' ').title()}")
                print(f"     Scenario: {case['scenario'].replace('_', ' ').title()}")
                print(f"     {case['baseline_ratio']:.3f}x → {case['fairness_ratio']:.3f}x")
                print(f"     Degradation: {case['degradation']:+.3f}x ({case['degradation_pct']:+.1f}%)")
        
        # Top 10 worst fairness ratios
        print(f"\n🎯 TOP 10 WORST FAIRNESS RATIOS (All Contexts):")
        for i, case in enumerate(results['top_issues'][:10], 1):
            if case['context'].startswith('distillation'):
                context_label = "🔬 DISTILL"
            else:
                context_label = "💉 INFER"
            print(f"\n{i}. {context_label} | {case['feature'].replace('_', ' ').title()} - {case['scenario'].replace('_', ' ').title()}")
            print(f"   Fairness Ratio: {case['fairness_ratio']:.3f}x")
            if case['worst_group'] and case['best_group']:
                print(f"   Worst: {case['worst_group']} (RMSE={case['worst_rmse']:.2f})")
                print(f"   Best: {case['best_group']} (RMSE={case['best_rmse']:.2f})")
                print(f"   Gap: {case['gap']:.2f} RMSE points")
        
        # Context comparison
        print("\n" + "=" * 80)
        print("📊 CONTEXT COMPARISON")
        print("=" * 80)
        
        # Average by context
        inference_ratios = [s['fairness_ratio'] for s in results['inference']['scenarios']]
        distillation_all_ratios = [s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios']]
        distillation_per_ratios = [s['fairness_ratio'] for s in results['distillation_per_patient']['scenarios']]
        all_distillation_ratios = distillation_all_ratios + distillation_per_ratios
        
        print(f"\n📈 Average Fairness Ratios:")
        print(f"   Inference: {np.mean(inference_ratios):.3f}x (σ={np.std(inference_ratios):.3f})")
        print(f"   Distillation (All Patients): {np.mean(distillation_all_ratios):.3f}x (σ={np.std(distillation_all_ratios):.3f})")
        print(f"   Distillation (Per Patient): {np.mean(distillation_per_ratios):.3f}x (σ={np.std(distillation_per_ratios):.3f})")
        print(f"   All Distillation: {np.mean(all_distillation_ratios):.3f}x (σ={np.std(all_distillation_ratios):.3f})")
        
        # Worst in each context
        worst_inference = max(results['inference']['scenarios'], key=lambda x: x['fairness_ratio'])
        worst_distillation_all = max(results['distillation_all_patients']['scenarios'], key=lambda x: x['fairness_ratio'])
        worst_distillation_per = max(results['distillation_per_patient']['scenarios'], key=lambda x: x['fairness_ratio'])
        
        print(f"\n🔴 Worst Cases by Context:")
        print(f"   Inference: {worst_inference['feature'].replace('_', ' ').title()} - {worst_inference['scenario'].replace('_', ' ').title()}")
        print(f"              Ratio: {worst_inference['fairness_ratio']:.3f}x")
        print(f"   Distillation (All): {worst_distillation_all['feature'].replace('_', ' ').title()} - {worst_distillation_all['scenario'].replace('_', ' ').title()}")
        print(f"                       Ratio: {worst_distillation_all['fairness_ratio']:.3f}x")
        print(f"   Distillation (Per): {worst_distillation_per['feature'].replace('_', ' ').title()} - {worst_distillation_per['scenario'].replace('_', ' ').title()}")
        print(f"                       Ratio: {worst_distillation_per['fairness_ratio']:.3f}x")
        
        # Advanced Fairness Metrics Summary
        if 'advanced_metrics' in results and results['advanced_metrics']:
            self._print_advanced_metrics_summary(results['advanced_metrics'])
    
    def _print_advanced_metrics_summary(self, advanced_metrics: Dict):
        """Print summary of advanced fairness metrics (DP Gap, EO Gap, FVO)."""
        print("\n" + "=" * 80)
        print("🔬 ADVANCED FAIRNESS METRICS SUMMARY")
        print("=" * 80)
        print("\nNote: These metrics provide deeper insights into fairness:")
        print("  • DP Gap: Alert distribution equality")
        print("  • EO Gap: Critical event detection equality (MOST IMPORTANT for safety)")
        print("  • FVO: Maximum accuracy disparity")
        
        for context_name, context_data in advanced_metrics.items():
            if not context_data:
                continue
            
            print(f"\n{'─' * 80}")
            print(f"📊 {context_name.replace('_', ' ').upper()}")
            print(f"{'─' * 80}")
            
            for scenario_key, metrics in context_data.items():
                if not metrics:
                    continue
                
                print(f"\n  Scenario: {scenario_key.replace('_', ' ').title()}")
                
                # Get metric values
                dp_gap = metrics.get('dp_gap', 'N/A')
                eo_gap = metrics.get('eo_gap', 'N/A')
                fvo = metrics.get('fvo', 'N/A')
                
                # Format and display with status indicators
                if isinstance(dp_gap, (int, float)):
                    dp_status = '✅' if dp_gap < 0.10 else '⚠️' if dp_gap < 0.20 else '❌'
                    print(f"    DP Gap:  {dp_gap:.4f} {dp_status}")
                
                if isinstance(eo_gap, (int, float)):
                    eo_status = '✅' if eo_gap < 0.10 else '⚠️' if eo_gap < 0.20 else '❌'
                    print(f"    EO Gap:  {eo_gap:.4f} {eo_status} {'← CRITICAL' if eo_gap > 0.20 else ''}")
                
                if isinstance(fvo, (int, float)):
                    fvo_status = '✅' if fvo < 0.05 else '⚠️' if fvo < 0.10 else '❌'
                    print(f"    FVO:     {fvo:.4f} {fvo_status}")
                
                # Show assessment if available
                if 'assessment' in metrics:
                    print(f"    Status:  {metrics['assessment']}")
        
        # Overall advanced metrics summary
        print(f"\n{'─' * 80}")
        print("📈 OVERALL ADVANCED METRICS STATISTICS")
        print(f"{'─' * 80}")
        
        all_dp_gaps = []
        all_eo_gaps = []
        all_fvos = []
        critical_eo_gaps = []
        
        for context_data in advanced_metrics.values():
            for metrics in context_data.values():
                if metrics:
                    if 'dp_gap' in metrics and isinstance(metrics['dp_gap'], (int, float)):
                        all_dp_gaps.append(metrics['dp_gap'])
                    if 'eo_gap' in metrics and isinstance(metrics['eo_gap'], (int, float)):
                        all_eo_gaps.append(metrics['eo_gap'])
                        if metrics['eo_gap'] > 0.20:
                            critical_eo_gaps.append(metrics['eo_gap'])
                    if 'fvo' in metrics and isinstance(metrics['fvo'], (int, float)):
                        all_fvos.append(metrics['fvo'])
        
        if all_dp_gaps:
            print(f"\n  DP Gap Statistics:")
            print(f"    Mean:   {np.mean(all_dp_gaps):.4f}")
            print(f"    Median: {np.median(all_dp_gaps):.4f}")
            print(f"    Max:    {np.max(all_dp_gaps):.4f}")
            print(f"    Cases > 0.10: {sum(1 for x in all_dp_gaps if x > 0.10)}/{len(all_dp_gaps)}")
        
        if all_eo_gaps:
            print(f"\n  EO Gap Statistics (CRITICAL FOR SAFETY):")
            print(f"    Mean:   {np.mean(all_eo_gaps):.4f}")
            print(f"    Median: {np.median(all_eo_gaps):.4f}")
            print(f"    Max:    {np.max(all_eo_gaps):.4f}")
            print(f"    Cases > 0.10: {sum(1 for x in all_eo_gaps if x > 0.10)}/{len(all_eo_gaps)}")
            print(f"    Cases > 0.20: {len(critical_eo_gaps)}/{len(all_eo_gaps)} ⚠️")
            
            if critical_eo_gaps:
                print(f"\n    ⚠️  WARNING: {len(critical_eo_gaps)} scenarios have critical EO Gap (>0.20)")
                print(f"       This indicates some groups may miss critical health warnings!")
        
        if all_fvos:
            print(f"\n  FVO Statistics:")
            print(f"    Mean:   {np.mean(all_fvos):.4f}")
            print(f"    Median: {np.median(all_fvos):.4f}")
            print(f"    Max:    {np.max(all_fvos):.4f}")
            print(f"    Cases > 0.05: {sum(1 for x in all_fvos if x > 0.05)}/{len(all_fvos)}")

    
    def generate_visual_report(self, results: Dict):
        """Generate comprehensive visual report."""
        print("\n" + "=" * 80)
        print("📊 GENERATING COMPREHENSIVE VISUAL REPORT")
        print(f"   Calculation Mode: {self.calc_mode}")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include calc_mode in filename for comparison
        mode_suffix = self.calc_mode.replace('_', '-')
        output_file = self.results_dir / f"comprehensive_fairness_report_{mode_suffix}_{timestamp}.png"
        
        # Figure with 4 rows (advanced metrics removed - they have their own dedicated file)
        fig = plt.figure(figsize=(36, 28))
        gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.4, height_ratios=[1, 1.2, 1, 1])
        
        fig.suptitle(f'COMPREHENSIVE FAIRNESS ANALYSIS\nAll Scenarios: Inference + Distillation | Mode: {self.calc_mode.upper()}', 
                    fontsize=22, fontweight='bold', y=0.99)
        
        # 1. Overall fairness ratio distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_overall_distribution(ax1, results)
        
        # 2. Context comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_context_comparison(ax2, results)
        
        # 3. Status summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_status_summary(ax3, results)
        
        # 4. Comprehensive heatmap with all scenarios (second row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_comprehensive_heatmap(ax4, results)
        
        # 5. Distillation comparison (third row, left 2/3)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_distillation_comparison(ax5, results)
        
        # 6. Top 15 worst cases (third row, right 1/3)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_top_worst_cases(ax6, results)
        
        # 7. Degradation analysis (fourth row, left)
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_degradation_analysis(ax7, results)
        
        # 8. Feature comparison across contexts (fourth row, middle)
        ax8 = fig.add_subplot(gs[3, 1])
        self._plot_feature_comparison(ax8, results)
        
        # 9. Recommendations (fourth row, right)
        ax9 = fig.add_subplot(gs[3, 2])
        self._plot_recommendations(ax9, results)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visual report saved: {output_file}")
        
        # Create a separate dedicated figure for advanced metrics
        if 'advanced_metrics' in results and results['advanced_metrics']:
            adv_output_file = self.results_dir / f"advanced_metrics_detailed_{mode_suffix}_{timestamp}.png"
            self._create_dedicated_advanced_metrics_figure(results, adv_output_file)
            print(f"✅ Detailed advanced metrics report: {adv_output_file}")
        
        return output_file
    
    def _plot_overall_distribution(self, ax, results: Dict):
        """Plot overall fairness ratio distribution."""
        ratios = [s['fairness_ratio'] for s in results['all_scenarios']]
        
        ax.hist(ratios, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(x=self.POOR_RATIO, color='red', linestyle='--', linewidth=2, label=f'Poor Threshold ({self.POOR_RATIO})')
        ax.axvline(x=np.median(ratios), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(ratios):.2f})')
        
        ax.set_xlabel('Fairness Ratio', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Fairness Ratio Distribution\n(All Scenarios)', fontweight='bold', fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_context_comparison(self, ax, results: Dict):
        """Plot comparison between contexts."""
        contexts = ['Inference', 'Distill\n(All)', 'Distill\n(Per)']
        
        inference_ratios = [s['fairness_ratio'] for s in results['inference']['scenarios']]
        distillation_all_ratios = [s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios']]
        distillation_per_ratios = [s['fairness_ratio'] for s in results['distillation_per_patient']['scenarios']]
        
        avgs = [np.mean(inference_ratios), np.mean(distillation_all_ratios), np.mean(distillation_per_ratios)]
        stds = [np.std(inference_ratios), np.std(distillation_all_ratios), np.std(distillation_per_ratios)]
        
        bars = ax.bar(contexts, avgs, yerr=stds, capsize=10, 
                     color=['#3498db', '#e74c3c', '#9b59b6'], alpha=0.7)
        
        # Position text above error bars to avoid overlapping
        for bar, avg, std in zip(bars, avgs, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{avg:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.axhline(y=self.POOR_RATIO, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Average Fairness Ratio', fontweight='bold')
        ax.set_title('Context Comparison', fontweight='bold', fontsize=11)
        ax.set_ylim(0, max(avgs) + max(stds) + 0.15)  # Add extra space at top
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_status_summary(self, ax, results: Dict):
        """Plot status summary."""
        ax.axis('off')
        
        total = len(results['all_scenarios'])
        poor = len(results['poor_fairness'])
        degraded = len(results['significant_degradation'])
        
        inference_scenarios = len(results['inference']['scenarios'])
        distillation_all_scenarios = len(results['distillation_all_patients']['scenarios'])
        distillation_per_scenarios = len(results['distillation_per_patient']['scenarios'])
        
        summary_text = f"""
SUMMARY STATISTICS

Total Scenarios: {total}
  • Inference: {inference_scenarios}
  • Distillation (All): {distillation_all_scenarios}
  • Distillation (Per): {distillation_per_scenarios}

Poor Fairness: {poor}
Significant Degradation: {degraded}

Status: {'GOOD' if poor == 0 else 'NEEDS ATTENTION'}
"""
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def _plot_comprehensive_heatmap(self, ax, results: Dict):
        """Plot comprehensive heatmap with all scenarios (inference + both distillation types)."""
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        # All scenarios: inference (4) + distillation all_patients (3) + distillation per_patient (3)
        scenarios = ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised',
                    'teacher_all', 'student_all', 'distilled_all',
                    'teacher_per', 'student_per', 'distilled_per']
        scenario_labels = ['Inference\nOnly', 'Trained\nStandard', 'Trained\nNoisy', 'Trained\nDenoised',
                          'Teacher\n(All)', 'Student\n(All)', 'Distilled\n(All)',
                          'Teacher\n(Per)', 'Student\n(Per)', 'Distilled\n(Per)']
        
        matrix = np.zeros((len(features), len(scenarios)))
        
        # Fill in inference scenarios
        for s in results['inference']['scenarios']:
            if s['feature'] in features and s['scenario'] in ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']:
                i = features.index(s['feature'])
                j = scenarios.index(s['scenario'])
                matrix[i, j] = s['fairness_ratio']
        
        # Fill in distillation all_patients scenarios
        for s in results['distillation_all_patients']['scenarios']:
            if s['feature'] in features:
                i = features.index(s['feature'])
                # Map phase to scenario index
                if s['scenario'] == 'teacher':
                    j = scenarios.index('teacher_all')
                elif s['scenario'] == 'student_baseline':
                    j = scenarios.index('student_all')
                elif s['scenario'] == 'distilled':
                    j = scenarios.index('distilled_all')
                else:
                    continue
                matrix[i, j] = s['fairness_ratio']
        
        # Fill in distillation per_patient scenarios
        for s in results['distillation_per_patient']['scenarios']:
            if s['feature'] in features:
                i = features.index(s['feature'])
                # Map phase to scenario index
                if s['scenario'] == 'teacher':
                    j = scenarios.index('teacher_per')
                elif s['scenario'] == 'student_baseline':
                    j = scenarios.index('student_per')
                elif s['scenario'] == 'distilled':
                    j = scenarios.index('distilled_per')
                else:
                    continue
                matrix[i, j] = s['fairness_ratio']
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.0)
        
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(scenario_labels, fontsize=7.5)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
        
        # Add values to cells
        for i in range(len(features)):
            for j in range(len(scenarios)):
                if matrix[i, j] > 0:  # Only show non-zero values
                    text_color = 'white' if matrix[i, j] > 1.4 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color=text_color, fontsize=7, fontweight='bold')
        
        # Add vertical separators
        ax.axvline(x=3.5, color='white', linestyle='-', linewidth=3)
        ax.axvline(x=6.5, color='white', linestyle='-', linewidth=3)
        
        # Add context labels - keep at reasonable position
        ax.text(1.5, -0.9, 'INFERENCE', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
        ax.text(5.0, -0.9, 'DISTILL (All Patients)', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
        ax.text(8.5, -0.9, 'DISTILL (Per Patient)', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#9b59b6', alpha=0.3))
        
        # Move title further up with more padding
        ax.set_title('Comprehensive Fairness Ratios: All Scenarios', fontweight='bold', fontsize=13, pad=35)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fairness Ratio', rotation=270, labelpad=20, fontweight='bold')
        
        # Add threshold legend
        ax.text(4.5, 5.5, '< 1.1: Excellent | 1.1-1.25: Good | 1.25-1.5: Acceptable | > 1.5: Poor', 
                fontsize=8, ha='center', va='top')
    
    def _plot_distillation_comparison(self, ax, results: Dict):
        """Plot distillation comparison across both experiment types."""
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        
        feature_labels = [f.replace('_', ' ').title() for f in features]
        x = np.arange(len(features))
        width = 0.13  # Narrower bars to fit 6 groups
        
        # All patients: teacher, student, distilled
        phases_all = ['teacher', 'student_baseline', 'distilled']
        colors_all = ['#3498db', '#5dade2', '#85c1e9']
        
        for i, phase in enumerate(phases_all):
            ratios = []
            for feature in features:
                scenarios = [s for s in results['distillation_all_patients']['scenarios'] 
                           if s['feature'] == feature and s['scenario'] == phase]
                ratio = scenarios[0]['fairness_ratio'] if scenarios else 0
                ratios.append(ratio)
            
            offset = (i - 2.5) * width
            bars = ax.bar(x + offset, ratios, width, 
                         label=f"{phase.replace('_', ' ').title()} (All)",
                         alpha=0.8, color=colors_all[i])
            
            # Only show text if bar is tall enough to avoid crowding
            for bar, ratio in zip(bars, ratios):
                if ratio > 0 and bar.get_height() > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                           f'{ratio:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=0)
        
        # Per patient: teacher, student, distilled
        phases_per = ['teacher', 'student_baseline', 'distilled']
        colors_per = ['#e74c3c', '#ec7063', '#f1948a']
        
        for i, phase in enumerate(phases_per):
            ratios = []
            for feature in features:
                scenarios = [s for s in results['distillation_per_patient']['scenarios'] 
                           if s['feature'] == feature and s['scenario'] == phase]
                ratio = scenarios[0]['fairness_ratio'] if scenarios else 0
                ratios.append(ratio)
            
            offset = (i + 0.5) * width
            bars = ax.bar(x + offset, ratios, width, 
                         label=f"{phase.replace('_', ' ').title()} (Per)",
                         alpha=0.8, color=colors_per[i])
            
            # Only show text if bar is tall enough to avoid crowding
            for bar, ratio in zip(bars, ratios):
                if ratio > 0 and bar.get_height() > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                           f'{ratio:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=0)
        
        ax.axhline(y=self.POOR_RATIO, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor Threshold')
        ax.set_xlabel('Feature', fontweight='bold')
        ax.set_ylabel('Fairness Ratio', fontweight='bold')
        ax.set_title('Distillation: All Patients vs Per Patient', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=0, fontsize=9)
        ax.legend(loc='upper left', fontsize=6.5, ncol=2)
        ax.set_ylim(0, max([s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios'] + 
                            results['distillation_per_patient']['scenarios']]) * 1.15)  # Add 15% space at top
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_worst_cases(self, ax, results: Dict):
        """Plot top 15 worst cases."""
        top15 = results['top_issues'][:15]
        
        # Create single-line compact labels
        labels = []
        for s in top15:
            feature = s['feature'].replace('_', ' ').title()[:8]
            scenario = s['scenario'].replace('_', ' ')[:10]
            context = 'Inf' if s['context'] == 'inference' else 'Dst'
            # Single line format: "Feature: Scenario (Context)"
            labels.append(f"{feature}: {scenario} ({context})")
        
        ratios = [s['fairness_ratio'] for s in top15]
        colors = ['#e74c3c' if r >= self.POOR_RATIO else '#f39c12' if r >= 1.25 else '#3498db' for r in ratios]
        
        bars = ax.barh(labels, ratios, color=colors, alpha=0.8)
        
        # Position text inside bars to avoid overlapping
        for bar, ratio in zip(bars, ratios):
            # Place text at the end of the bar, inside it
            ax.text(ratio - 0.05, bar.get_y() + bar.get_height()/2,
                   f'{ratio:.2f}', va='center', ha='right', fontsize=7, fontweight='bold', color='white')
        
        ax.axvline(x=self.POOR_RATIO, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Fairness Ratio', fontweight='bold', fontsize=9)
        ax.set_title('Top 15 Worst Cases', fontweight='bold', fontsize=11)
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=7)  # Slightly larger for readability
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(ratios) + 0.1)  # Add a bit of space on right
    
    def _plot_degradation_analysis(self, ax, results: Dict):
        """Plot degradation analysis."""
        degraded = [s for s in results['all_scenarios'] if s['degradation'] > 0]
        
        if not degraded:
            ax.text(0.5, 0.5, 'No Degradation\nDetected', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='green')
            ax.axis('off')
            return
        
        # Group by context
        inference_deg = [s['degradation_pct'] for s in degraded if s['context'] == 'inference']
        distill_deg = [s['degradation_pct'] for s in degraded if s['context'] == 'distillation']
        
        data = [inference_deg, distill_deg]
        tick_labels = ['Inference', 'Distillation']
        
        bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Degradation (%)', fontweight='bold')
        ax.set_title('Degradation Distribution', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_feature_comparison(self, ax, results: Dict):
        """Plot feature comparison across contexts."""
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        
        inference_avgs = []
        distill_all_avgs = []
        distill_per_avgs = []
        
        for feature in features:
            inf_ratios = [s['fairness_ratio'] for s in results['inference']['scenarios'] 
                         if s['feature'] == feature]
            dist_all_ratios = [s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios'] 
                              if s['feature'] == feature]
            dist_per_ratios = [s['fairness_ratio'] for s in results['distillation_per_patient']['scenarios'] 
                              if s['feature'] == feature]
            
            inference_avgs.append(np.mean(inf_ratios) if inf_ratios else 0)
            distill_all_avgs.append(np.mean(dist_all_ratios) if dist_all_ratios else 0)
            distill_per_avgs.append(np.mean(dist_per_ratios) if dist_per_ratios else 0)
        
        x = np.arange(len(features))
        width = 0.25
        
        bars1 = ax.bar(x - width, inference_avgs, width, label='Inference', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, distill_all_avgs, width, label='Distill (All)', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, distill_per_avgs, width, label='Distill (Per)', color='#9b59b6', alpha=0.8)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Avg Fairness Ratio', fontweight='bold')
        ax.set_title('Feature Comparison', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_', '\n').title() for f in features], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_recommendations(self, ax, results: Dict):
        """Plot recommendations."""
        ax.axis('off')
        
        top_issues = results['top_issues'][:5]
        
        rec_text = "TOP PRIORITIES\n" + "=" * 30 + "\n\n"
        
        for i, issue in enumerate(top_issues, 1):
            context = issue['context'].upper()
            feature = issue['feature'].replace('_', ' ').title()
            scenario = issue['scenario'].replace('_', ' ').title()
            ratio = issue['fairness_ratio']
            
            rec_text += f"{i}. [{context}] {feature}\n"
            rec_text += f"   {scenario}\n"
            rec_text += f"   Ratio: {ratio:.2f}x\n\n"
        
        ax.text(0.05, 0.95, rec_text, ha='left', va='top', fontsize=9,
               family='monospace', transform=ax.transAxes)
    
    def _plot_advanced_metrics_summary(self, ax, results: Dict):
        """Plot advanced fairness metrics summary organized by scenario type."""
        ax.axis('off')
        
        # Check if advanced metrics are available
        if 'advanced_metrics' not in results or not results['advanced_metrics']:
            # Show info panel
            self._plot_advanced_metrics_info_panel(ax)
            return
        
        # If we have actual advanced metrics data, plot them organized by type
        adv_metrics = results['advanced_metrics']
        
        # Create title
        title_text = "🔬 ADVANCED FAIRNESS METRICS (From Real Predictions)"
        ax.text(0.5, 0.98, title_text, ha='center', va='top', fontsize=16, 
               fontweight='bold', transform=ax.transAxes)
        
        subtitle = "Organized by: INFERENCE (4 scenarios) | ALL-PATIENTS (3 phases) | PER-PATIENT (3 phases)"
        ax.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=11, 
               transform=ax.transAxes, style='italic', color='#555')
        
        # Create 3×3 grid: 3 metrics (DP, EO, FVO) × 3 contexts (Inference, All-Patients, Per-Patient)
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        inner_gs = GridSpecFromSubplotSpec(3, 3, subplot_spec=ax.get_subplotspec(), 
                                          wspace=0.5, hspace=0.6)
        
        # Extract metrics organized by context
        contexts = {
            'inference': 'Inference (4 scenarios)',
            'distillation_all_patients': 'All-Patients (3 phases)',
            'distillation_per_patient': 'Per-Patient (3 phases)'
        }
        
        metric_types = ['dp_gap', 'eo_gap', 'fvo']
        metric_names = ['DP Gap', 'EO Gap ⚠️', 'FVO']
        thresholds = [(0.10, 0.20), (0.10, 0.20), (0.05, 0.10)]
        
        # Create 9 subplots (3 contexts × 3 metrics)
        for col_idx, (context_key, context_label) in enumerate(contexts.items()):
            context_data = adv_metrics.get(context_key, {})
            
            for row_idx, (metric_type, metric_name, (thresh_good, thresh_concern)) in enumerate(zip(metric_types, metric_names, thresholds)):
                subplot_ax = ax.figure.add_subplot(inner_gs[row_idx, col_idx])
                
                # Collect values for this context and metric
                scenarios_data = {}
                for scenario_key, metrics in context_data.items():
                    if metrics and isinstance(metrics, dict) and metric_type in metrics:
                        value = metrics[metric_type]
                        if isinstance(value, (int, float)):
                            # Create readable label
                            label = self._format_scenario_label(scenario_key)
                            scenarios_data[label] = value
                
                # Plot this specific metric for this context
                self._plot_metric_bars(subplot_ax, scenarios_data, metric_name, 
                                      thresh_good, thresh_concern, context_label, 
                                      show_ylabel=(col_idx == 0))
        
        # Add legend at the very bottom
        legend_text = "📊 Colors: Green=Excellent  Orange=Acceptable  Red=Poor | Lower values = Better fairness"
        ax.text(0.5, 0.01, legend_text, ha='center', va='bottom', fontsize=8,
               transform=ax.transAxes, fontweight='bold', color='#2c3e50')
    
    def _format_scenario_label(self, scenario_key: str) -> str:
        """Convert scenario key to readable label."""
        parts = scenario_key.split('_')
        if not parts:
            return scenario_key
        
        # First part is feature (gender, age, cohort, etc.)
        feature = parts[0].capitalize()
        
        # Rest is scenario name
        scenario = '_'.join(parts[1:]) if len(parts) > 1 else ''
        
        # Shorten scenario names
        scenario_map = {
            'inference_only': 'InfOnly',
            'trained_standard': 'Std',
            'trained_noisy': 'Noisy',
            'trained_denoised': 'Clean',
            'teacher': 'Teach',
            'student': 'Stud',
            'distilled': 'Dist'
        }
        
        scenario_short = scenario_map.get(scenario, scenario[:6])
        
        return f"{feature}\n{scenario_short}"
    
    def _plot_metric_bars(self, ax, scenarios_data, metric_name, thresh_good, thresh_concern, 
                         context_label, show_ylabel=True):
        """Plot horizontal bars for one metric in one context."""
        ax.clear()
        
        if not scenarios_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=11)
            ax.set_title(f"{metric_name}\n{context_label}", fontsize=11, fontweight='bold')
            ax.axis('off')
            return
        
        # Sort by value (best first)
        sorted_items = sorted(scenarios_data.items(), key=lambda x: x[1])
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Color based on thresholds
        colors = []
        for v in values:
            if v < thresh_good:
                colors.append('#27ae60')  # Dark green - Excellent
            elif v < thresh_concern:
                colors.append('#f39c12')  # Orange - Acceptable  
            else:
                colors.append('#c0392b')  # Dark red - Poor
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        
        # Threshold lines
        ax.axvline(x=thresh_good, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=thresh_concern, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
        
        if show_ylabel:
            ax.set_xlabel('Value →', fontsize=10)
        
        ax.set_title(f"{metric_name}\n{context_label}", fontsize=11, fontweight='bold', pad=5)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    def _plot_advanced_metrics_info_panel(self, ax):
        """Show info panel when no advanced metrics data available."""
        title_text = "🔬 ADVANCED FAIRNESS METRICS\n(DP Gap, EO Gap, FVO)"
        ax.text(0.5, 0.9, title_text, ha='center', va='top', fontsize=14, 
               fontweight='bold', transform=ax.transAxes)
        
        info_text = """
WHAT ARE THESE METRICS?

• DP Gap (Demographic Parity Gap): Measures if critical alerts are distributed 
  equally across demographic groups. Ensures no group is over/under-alerted.
  Target: < 0.10 (excellent), < 0.20 (acceptable)

• EO Gap (Equal Opportunity Gap): CRITICAL FOR SAFETY - Measures if the model 
  detects actual high-risk events equally well across groups. If high, some groups 
  may miss critical health warnings!
  Target: < 0.10 (excellent), < 0.20 (acceptable)

• FVO (Fairness Violation Objective): Measures maximum accuracy disparity between 
  any two demographic groups. Ensures general model reliability is equitable.
  Target: < 0.05 (excellent), < 0.10 (acceptable)

HOW TO GET THESE METRICS:

For full advanced metrics analysis with your data:
  1. python3 fairness/quickstart_advanced_metrics.py
  2. python3 fairness/apply_advanced_metrics_to_results.py --experiment-dir <path>
  3. PYTHONPATH=$PWD python3 fairness/advanced_metrics_example.py --example

See: fairness/metrics/ADVANCED_METRICS_README.md for complete documentation
        """
        
        ax.text(0.05, 0.75, info_text, ha='left', va='top', fontsize=8, 
               family='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        note_text = "ℹ️  Note: Advanced metrics require per-sample predictions and demographics.\n" \
                   "This panel shows what's available when you run dedicated analyzers."
        ax.text(0.5, 0.02, note_text, ha='center', va='bottom', fontsize=7, 
               style='italic', transform=ax.transAxes, color='#555')
        
        # If we have actual advanced metrics data, plot them
        adv_metrics = results['advanced_metrics']
        
        # Create title
        title_text = "🔬 ADVANCED FAIRNESS METRICS SUMMARY"
        ax.text(0.5, 0.98, title_text, ha='center', va='top', fontsize=14, 
               fontweight='bold', transform=ax.transAxes)
        
        # Collect all metric values
        all_dp_gaps = []
        all_eo_gaps = []
        all_fvos = []
        
        for context_data in adv_metrics.values():
            for metrics in context_data.values():
                if metrics:
                    if 'dp_gap' in metrics and isinstance(metrics['dp_gap'], (int, float)):
                        all_dp_gaps.append(metrics['dp_gap'])
                    if 'eo_gap' in metrics and isinstance(metrics['eo_gap'], (int, float)):
                        all_eo_gaps.append(metrics['eo_gap'])
                    if 'fvo' in metrics and isinstance(metrics['fvo'], (int, float)):
                        all_fvos.append(metrics['fvo'])
        
        if not all_dp_gaps and not all_eo_gaps and not all_fvos:
            # Show placeholder if no data
            ax.text(0.5, 0.5, 'No Advanced Metrics Data Available\n\n' + 
                   'Run dedicated analyzers to generate this data',
                   ha='center', va='center', fontsize=11, style='italic',
                   transform=ax.transAxes, color='gray')
            return
        
        # Create three subpanels for the three metrics
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        inner_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(), 
                                          wspace=0.3, hspace=0)
        
        # Helper function to create metric panel with bar chart for each scenario
        def create_metric_panel_bars(inner_ax, all_scenarios_data, metric_name, threshold_good, threshold_concern):
            inner_ax.clear()
            
            if not all_scenarios_data:
                inner_ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                inner_ax.set_title(metric_name, fontweight='bold', fontsize=10)
                inner_ax.axis('off')
                return
            
            # Sort scenarios by value (best to worst)
            sorted_scenarios = sorted(all_scenarios_data.items(), key=lambda x: x[1])
            scenario_names = [s[0] for s in sorted_scenarios]
            values = [s[1] for s in sorted_scenarios]
            
            # Limit to top 12 if too many
            if len(scenario_names) > 12:
                scenario_names = scenario_names[:12]
                values = values[:12]
            
            # Color based on thresholds
            colors = []
            for v in values:
                if v < threshold_good:
                    colors.append('#2ecc71')  # Green - Excellent
                elif v < threshold_concern:
                    colors.append('#f39c12')  # Orange - Acceptable
                else:
                    colors.append('#e74c3c')  # Red - Poor
            
            # Create horizontal bar chart
            y_pos = np.arange(len(scenario_names))
            inner_ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Better labels - make them readable
            readable_names = []
            for name in scenario_names:
                # Convert keys like "dist.gender_teacher" to "All-Patients: Gender (Teacher)"
                parts = name.split('.')
                if len(parts) >= 2:
                    context = parts[0]
                    rest = parts[1]
                    
                    # Parse context
                    if context == 'infe':
                        ctx_label = 'Inference'
                    elif context == 'dist':
                        ctx_label = 'Distill-All'
                    else:
                        ctx_label = context[:8]
                    
                    # Parse rest (e.g., "gender_teacher" or "age_inference_only")
                    rest_parts = rest.split('_')
                    feature = rest_parts[0].capitalize()
                    scenario = '_'.join(rest_parts[1:]) if len(rest_parts) > 1 else 'Unknown'
                    
                    # Shorten scenario names
                    scenario_short = scenario.replace('inference_only', 'InfOnly') \
                                            .replace('trained_standard', 'Standard') \
                                            .replace('trained_noisy', 'Noisy') \
                                            .replace('trained_denoised', 'Denoised') \
                                            .replace('teacher', 'Teacher') \
                                            .replace('student', 'Student') \
                                            .replace('distilled', 'Distilled')
                    
                    readable = f"{feature} ({scenario_short})"
                else:
                    readable = name[:15]
                
                readable_names.append(readable)
            
            inner_ax.set_yticks(y_pos)
            inner_ax.set_yticklabels(readable_names, fontsize=7)
            
            # Add threshold lines with labels
            inner_ax.axvline(x=threshold_good, color='green', linestyle='--', 
                           linewidth=1.5, alpha=0.6, label=f'Good (<{threshold_good})')
            inner_ax.axvline(x=threshold_concern, color='orange', linestyle='--', 
                           linewidth=1.5, alpha=0.6, label=f'Concern (<{threshold_concern})')
            
            inner_ax.set_xlabel('Fairness Gap (Lower = Better)', fontsize=8, fontweight='bold')
            inner_ax.set_title(metric_name, fontweight='bold', fontsize=10, pad=8)
            inner_ax.grid(axis='x', alpha=0.3, linestyle=':')
            
            # Add mean line
            mean_val = np.mean(values)
            inner_ax.axvline(x=mean_val, color='blue', linestyle=':', 
                           linewidth=2, alpha=0.7)
            
            # Legend at bottom
            inner_ax.legend(loc='lower right', fontsize=6, framealpha=0.9)
        
        # Collect all scenarios with their metrics
        dp_scenarios = {}
        eo_scenarios = {}
        fvo_scenarios = {}
        
        for context_name, context_data in adv_metrics.items():
            for scenario_key, metrics in context_data.items():
                if metrics and isinstance(metrics, dict):
                    full_key = f"{context_name[:4]}.{scenario_key}"
                    if 'dp_gap' in metrics and isinstance(metrics['dp_gap'], (int, float)):
                        dp_scenarios[full_key] = metrics['dp_gap']
                    if 'eo_gap' in metrics and isinstance(metrics['eo_gap'], (int, float)):
                        eo_scenarios[full_key] = metrics['eo_gap']
                    if 'fvo' in metrics and isinstance(metrics['fvo'], (int, float)):
                        fvo_scenarios[full_key] = metrics['fvo']
        
        if not dp_scenarios and not eo_scenarios and not fvo_scenarios:
            ax.text(0.5, 0.5, 'No Advanced Metrics Data Available',
                   ha='center', va='center', fontsize=11, style='italic',
                   transform=ax.transAxes, color='gray')
            return
        
        # Create three panels with bar charts
        ax1 = ax.figure.add_subplot(inner_gs[0, 0])
        create_metric_panel_bars(ax1, dp_scenarios, 'DP Gap\n(Alert Distribution)', 0.10, 0.20)
        
        ax2 = ax.figure.add_subplot(inner_gs[0, 1])
        create_metric_panel_bars(ax2, eo_scenarios, 'EO Gap ⚠️\n(Safety Critical!)', 0.10, 0.20)
        
        ax3 = ax.figure.add_subplot(inner_gs[0, 2])
        create_metric_panel_bars(ax3, fvo_scenarios, 'FVO\n(Accuracy Disparity)', 0.05, 0.10)
        
        # Add brief note at bottom (not overlaying)
        note_text = "📊 Lower values = Better fairness | EO Gap most critical for patient safety"
        ax.text(0.5, 0.02, note_text, ha='center', va='bottom', fontsize=9,
               transform=ax.transAxes, fontweight='bold', color='#2c3e50')
    
    def _create_dedicated_advanced_metrics_figure(self, results: Dict, output_file: Path):
        """Create a separate large figure dedicated to advanced metrics with horizontal bars grouped by feature."""
        adv_metrics = results['advanced_metrics']
        
        # Define features and scenarios
        features = ['gender', 'age', 'cohort', 'pump_model', 'sensor_band']
        feature_labels = ['Gender', 'Age', 'Cohort', 'Pump Model', 'Sensor Band']
        
        contexts = {
            'inference': {
                'label': 'Inference',
                'scenarios': ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised'],
                'scenario_labels': ['Inf Only', 'Standard', 'Noisy', 'Denoised']
            },
            'distillation_all_patients': {
                'label': 'All-Patients Distillation',
                'scenarios': ['teacher', 'student', 'distilled'],
                'scenario_labels': ['Teacher', 'Student', 'Distilled']
            },
            'distillation_per_patient': {
                'label': 'Per-Patient Distillation', 
                'scenarios': ['teacher', 'student', 'distilled'],
                'scenario_labels': ['Teacher', 'Student', 'Distilled']
            }
        }
        
        metric_types = ['dp_gap', 'eo_gap', 'fvo']
        metric_names = ['DP Gap (Demographic Parity)', 'EO Gap (Equal Opportunity) - CRITICAL', 'FVO (Fairness Violation)']
        thresholds = [(0.10, 0.20), (0.10, 0.20), (0.05, 0.10)]
        
        # Create figure: 3 rows (metrics) x 3 cols (contexts)
        fig = plt.figure(figsize=(60, 48))
        fig.suptitle(f'ADVANCED FAIRNESS METRICS - GROUPED BY FEATURE\nCalculation Mode: {self.calc_mode.upper()}', 
                    fontsize=36, fontweight='bold', y=0.99)
        
        subtitle = "Bars grouped by demographic feature for meaningful comparison | GREEN: Excellent (<0.10) | ORANGE: Acceptable (0.10-0.20) | RED: Poor (>0.20)"
        fig.text(0.5, 0.965, subtitle, ha='center', va='top', fontsize=20, 
                style='italic', color='#333')
        
        # Color palette for scenarios within each context
        scenario_colors = {
            'inference': ['#3498db', '#2980b9', '#1abc9c', '#16a085'],  # Blues/teals
            'distillation_all_patients': ['#e74c3c', '#c0392b', '#9b59b6'],  # Reds/purple
            'distillation_per_patient': ['#f39c12', '#d35400', '#e67e22']  # Oranges
        }
        
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, 
                             left=0.08, right=0.98, top=0.92, bottom=0.08)
        
        for row_idx, (metric_type, metric_name, (thresh_good, thresh_concern)) in enumerate(zip(metric_types, metric_names, thresholds)):
            for col_idx, (context_key, context_info) in enumerate(contexts.items()):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                context_data = adv_metrics.get(context_key, {})
                scenarios = context_info['scenarios']
                scenario_labels = context_info['scenario_labels']
                colors = scenario_colors[context_key]
                
                # Organize data by feature
                feature_data = {f: {} for f in features}
                
                for scenario_key, metrics in context_data.items():
                    if not metrics or not isinstance(metrics, dict) or metric_type not in metrics:
                        continue
                    
                    value = metrics[metric_type]
                    if not isinstance(value, (int, float)):
                        continue
                    
                    # Parse the key to get feature and scenario
                    feature = None
                    scenario_name = None
                    
                    for f in features:
                        if scenario_key.startswith(f + '_'):
                            feature = f
                            scenario_name = scenario_key[len(f) + 1:]
                            break
                    
                    if not feature or not scenario_name:
                        continue
                    
                    if scenario_name in scenarios:
                        feature_data[feature][scenario_name] = value
                
                # Plot horizontal grouped bars
                self._plot_grouped_feature_bars_horizontal(
                    ax, feature_data, features, feature_labels,
                    scenarios, scenario_labels, colors,
                    metric_name, context_info['label'],
                    thresh_good, thresh_concern
                )
        
        # Add legend at bottom
        legend_elements = []
        for context_key, context_info in contexts.items():
            for scenario, label, color in zip(context_info['scenarios'], context_info['scenario_labels'], scenario_colors[context_key]):
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                                     label=f"{context_info['label']}: {label}"))
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=16,
                  bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True)
        
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_grouped_feature_bars_horizontal(self, ax, feature_data, features, feature_labels,
                                              scenarios, scenario_labels, colors,
                                              metric_name, context_label, thresh_good, thresh_concern):
        """Plot horizontal bars grouped by feature with scenario labels on bars."""
        
        # Filter to features that have data
        active_features = []
        active_labels = []
        for f, label in zip(features, feature_labels):
            if feature_data.get(f):
                active_features.append(f)
                active_labels.append(label)
        
        if not active_features:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=18)
            ax.set_title(f"{metric_name}\n{context_label}", fontsize=18, fontweight='bold')
            ax.axis('off')
            return
        
        n_features = len(active_features)
        n_scenarios = len(scenarios)
        
        # Bar positions (y-axis for horizontal bars)
        y = np.arange(n_features)
        height = 0.8 / n_scenarios
        
        # Find max value for x-axis limit
        max_val = 0
        for feature in active_features:
            for scenario in scenarios:
                val = feature_data.get(feature, {}).get(scenario, 0)
                if val > max_val:
                    max_val = val
        
        # Plot horizontal bars for each scenario
        for i, (scenario, label, color) in enumerate(zip(scenarios, scenario_labels, colors)):
            values = []
            for feature in active_features:
                val = feature_data.get(feature, {}).get(scenario, 0)
                values.append(val)
            
            offset = (i - n_scenarios/2 + 0.5) * height
            bars = ax.barh(y + offset, values, height, label=label, color=color, 
                          alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Add scenario name inside bar, value outside bar
            for bar, val in zip(bars, values):
                if val > 0:
                    # Determine text color based on value for the value label
                    if val < thresh_good:
                        txt_color = '#155724'  # Dark green
                    elif val < thresh_concern:
                        txt_color = '#856404'  # Dark orange
                    else:
                        txt_color = '#721c24'  # Dark red
                    
                    bar_width = bar.get_width()
                    bar_y = bar.get_y() + bar.get_height()/2
                    
                    # Scenario name inside bar (white text)
                    if bar_width > max_val * 0.08:
                        ax.text(bar_width * 0.02, bar_y, label, 
                               ha='left', va='center', fontsize=10, 
                               fontweight='bold', color='white')
                    
                    # Value outside bar (color-coded)
                    ax.text(bar_width + max_val * 0.01, bar_y,
                           f'{val:.3f}', ha='left', va='center', 
                           fontsize=11, fontweight='bold', color=txt_color)
        
        # Threshold lines (vertical for horizontal bars)
        ax.axvline(x=thresh_good, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=thresh_concern, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add threshold labels
        ax.text(thresh_good, n_features - 0.3, f'Good ({thresh_good})', 
               fontsize=12, color='green', fontweight='bold', ha='center')
        ax.text(thresh_concern, n_features - 0.3, f'Concern ({thresh_concern})', 
               fontsize=12, color='red', fontweight='bold', ha='center')
        
        # Formatting
        ax.set_yticks(y)
        ax.set_yticklabels(active_labels, fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric Value', fontsize=14, fontweight='bold')
        ax.set_title(f"{metric_name}\n{context_label}", fontsize=18, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.legend(fontsize=10, loc='lower right')
        
        # Set x-axis limits
        ax.set_xlim(left=0, right=max(max_val * 1.3, thresh_concern * 1.2))
        
        # Invert y-axis so first feature is at top
        ax.invert_yaxis()
    
    def _plot_grouped_feature_bars(self, ax, feature_data, features, feature_labels,
                                   scenarios, scenario_labels, colors,
                                   metric_name, context_label, thresh_good, thresh_concern):
        """Plot bars grouped by feature with scenarios as sub-bars."""
        
        # Filter to features that have data
        active_features = []
        active_labels = []
        for f, label in zip(features, feature_labels):
            if feature_data.get(f):
                active_features.append(f)
                active_labels.append(label)
        
        if not active_features:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=14)
            ax.set_title(f"{metric_name}\n{context_label}", fontsize=14, fontweight='bold')
            ax.axis('off')
            return
        
        n_features = len(active_features)
        n_scenarios = len(scenarios)
        
        # Bar positions
        x = np.arange(n_features)
        width = 0.8 / n_scenarios
        
        # Plot bars for each scenario
        for i, (scenario, label, color) in enumerate(zip(scenarios, scenario_labels, colors)):
            values = []
            for feature in active_features:
                val = feature_data.get(feature, {}).get(scenario, 0)
                values.append(val)
            
            offset = (i - n_scenarios/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                if val > 0:
                    # Color code the value label
                    if val < thresh_good:
                        txt_color = '#27ae60'
                    elif val < thresh_concern:
                        txt_color = '#f39c12'
                    else:
                        txt_color = '#c0392b'
                    
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, 
                           fontweight='bold', color=txt_color, rotation=45)
        
        # Threshold lines
        ax.axhline(y=thresh_good, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Good')
        ax.axhline(y=thresh_concern, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Concern')
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(active_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_title(f"{metric_name}\n{context_label}", fontsize=14, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='upper right')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Adjust y-axis max to fit value labels
        current_max = ax.get_ylim()[1]
        ax.set_ylim(top=current_max * 1.15)
    
    def export_advanced_metrics_to_csv(self, results: Dict) -> Optional[str]:
        """Export advanced fairness metrics to a CSV file."""
        advanced_metrics = results.get('advanced_metrics', {})
        
        if not advanced_metrics:
            print("  ⚠️  No advanced metrics to export")
            return None
        
        rows = []
        
        # Define known features for proper parsing
        known_features = ['gender', 'age', 'cohort', 'pump_model', 'sensor_band']
        
        # Process each context (inference, distillation_all_patients, distillation_per_patient)
        for context, metrics_dict in advanced_metrics.items():
            if not isinstance(metrics_dict, dict):
                continue
                
            for key, metrics in metrics_dict.items():
                if not isinstance(metrics, dict):
                    continue
                
                # Parse the key to extract feature and scenario
                # Format: feature_scenario (e.g., 'gender_inference_only', 'age_teacher', 'pump_model_trained_denoised')
                feature = None
                scenario = None
                
                # Try to match known features
                for known_feature in known_features:
                    if key.startswith(known_feature + '_'):
                        feature = known_feature
                        scenario = key[len(known_feature) + 1:]
                        break
                
                if feature is None:
                    # Fallback: split on last underscore
                    parts = key.rsplit('_', 1)
                    if len(parts) == 2:
                        feature = parts[0]
                        scenario = parts[1]
                    else:
                        feature = key
                        scenario = 'unknown'
                
                row = {
                    'context': context,
                    'feature': feature,
                    'scenario': scenario,
                    'dp_gap': metrics.get('dp_gap', None),
                    'eo_gap': metrics.get('eo_gap', None),
                    'fvo': metrics.get('fvo', None),
                    'assessment': metrics.get('assessment', ''),
                    'n_samples': metrics.get('n_samples', None),
                    'n_groups': metrics.get('n_groups', None),
                    'note': metrics.get('note', '')
                }
                rows.append(row)
        
        if not rows:
            print("  ⚠️  No metric rows to export")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by context, feature, scenario
        context_order = {'inference': 0, 'distillation_all_patients': 1, 'distillation_per_patient': 2}
        df['context_order'] = df['context'].map(context_order)
        df = df.sort_values(['context_order', 'feature', 'scenario'])
        df = df.drop(columns=['context_order'])
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"advanced_metrics_raw_{self.calc_mode}_{timestamp}.csv"
        csv_path = self.results_dir / csv_filename
        
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Advanced metrics exported to: {csv_path}")
        
        return str(csv_path)
    
    def run(self):
        """Run comprehensive analysis."""
        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE FAIRNESS INVESTIGATOR")
        print("Analyzing: Inference + Distillation + All Patients Trained")
        print("=" * 80)
        
        try:
            # Analyze all scenarios
            results = self.analyze_all_scenarios()
            
            # Print summary
            self.print_summary(results)
            
            # Generate visual report
            visual_report = self.generate_visual_report(results)
            
            # Export advanced metrics to CSV
            csv_report = self.export_advanced_metrics_to_csv(results)
            
            print("\n" + "=" * 80)
            print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
            print(f"📊 Visual report: {visual_report}")
            if csv_report:
                print(f"📄 CSV report: {csv_report}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Fairness Investigation')
    parser.add_argument('--calc-mode', type=str, default=CALC_MODE_SIMPLE,
                       choices=VALID_CALC_MODES,
                       help=f'Calculation mode for advanced metrics. Options: {VALID_CALC_MODES}')
    parser.add_argument('--compare-modes', action='store_true',
                       help='Run analysis with all three modes and generate comparison')
    
    args = parser.parse_args()
    
    if args.compare_modes:
        # Run with all three modes and compare
        print("\n" + "=" * 80)
        print("🔬 COMPARING ALL CALCULATION MODES")
        print("=" * 80)
        
        mode_results = {}
        
        for mode in VALID_CALC_MODES:
            print(f"\n{'='*80}")
            print(f"Running with calc_mode = {mode}")
            print('='*80)
            
            investigator = ComprehensiveFairnessInvestigator(calc_mode=mode)
            investigator.run()
            
        print("\n" + "=" * 80)
        print("✅ All modes completed! Compare the generated reports:")
        print("   - comprehensive_fairness_report_*.png (for each mode)")
        print("   - advanced_metrics_detailed_*.png (for each mode)")
        print("=" * 80)
    else:
        investigator = ComprehensiveFairnessInvestigator(calc_mode=args.calc_mode)
    investigator.run()
