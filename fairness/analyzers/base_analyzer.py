#!/usr/bin/env python3
"""
Base Fairness Analyzer Class

Abstract base class providing common functionality for all fairness analyzers.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path dynamically
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.path_utils import get_project_root


class BaseFairnessAnalyzer(ABC):
    """
    Abstract base class for fairness analyzers.
    
    Provides common functionality:
    - Patient data loading
    - Experiment directory management
    - Patient results loading
    - Fairness metric calculations
    - Report generation
    """
    
    def __init__(self, feature_name: str, data_path: Optional[str] = None, 
                 experiment_type: str = "per_patient", experiments_folder: str = "distillation_experiments"):
        """
        Initialize the analyzer.
        
        Args:
            feature_name: Name of the demographic feature (e.g., 'Gender', 'Age')
            data_path: Optional path to patient data CSV
            experiment_type: Type of experiment to analyze:
                           - "per_patient": Per-patient distillation (default)
                           - "all_patients": All-patients training with per-patient inference
            experiments_folder: Name of the experiments folder (default: "distillation_experiments")
        """
        self.feature_name = feature_name
        self.experiment_type = experiment_type
        self.experiments_folder = experiments_folder
        
        if data_path is None:
            data_path = str(get_project_root() / "data" / "ohiot1dm" / "data.csv")
        self.data_path = data_path
        
        # Create results directory based on experiment type and folder
        folder_suffix = "minilm_distillation" if "minilm" in experiments_folder else "distillation"
        if experiment_type == "all_patients":
            self.results_dir = get_project_root() / "fairness" / "analysis_results" / f"{folder_suffix}_all_patients"
        else:
            self.results_dir = get_project_root() / "fairness" / "analysis_results" / f"{folder_suffix}_per_patient"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model names (will be loaded from experiment)
        self.teacher_model = "Teacher Model"
        self.student_model = "Student Model"
        
        # Load patient data
        self.patient_data = self._load_patient_data()
    
    @abstractmethod
    def _load_patient_data(self) -> Dict:
        """
        Load patient demographic data.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary mapping patient_id to demographic value
        """
        pass
    
    @abstractmethod
    def _get_default_patient_data(self) -> Dict:
        """
        Get default patient data when CSV is not available.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary mapping patient_id to demographic value
        """
        pass
    
    def find_latest_experiment(self) -> str:
        """
        Find the latest experiment directory based on experiment_type.
        
        Returns:
            Path to the latest experiment directory
        """
        distillation_dir = get_project_root() / self.experiments_folder
        
        if not distillation_dir.exists():
            raise FileNotFoundError(f"💥 Experiments directory not found: {distillation_dir}")
        
        # Choose subdirectory based on experiment type
        if self.experiment_type == "all_patients":
            search_dir = distillation_dir / "all_patients_pipeline"
            if not search_dir.exists():
                raise FileNotFoundError(f"💥 All patients directory not found: {search_dir}")
        else:
            search_dir = distillation_dir
        
        # Find all pipeline directories
        experiment_dirs = [
            d for d in search_dir.iterdir()
            if d.is_dir() and d.name.startswith('pipeline_')
        ]
        
        if not experiment_dirs:
            raise FileNotFoundError(f"💥 No pipeline experiment directories found in {search_dir}")
        
        # Filter to only complete experiments (with all 3 phases)
        if self.experiment_type == "all_patients":
            complete_dirs = []
            for d in sorted(experiment_dirs, reverse=True):
                required_phases = ['phase_1_teacher', 'phase_2_student', 'phase_3_distillation']
                has_all_phases = all((d / phase).exists() for phase in required_phases)
                
                if has_all_phases:
                    complete_dirs.append(d)
                else:
                    print(f"⚠️  Skipping incomplete experiment: {d.name}")
            
            if not complete_dirs:
                raise FileNotFoundError(f"💥 No complete pipeline experiments found in {search_dir}. Need all 3 phases.")
            
            latest_experiment = complete_dirs[0]
        else:
            # Sort and get latest for per-patient experiments
            latest_experiment = sorted(experiment_dirs)[-1]
        
        exp_type_label = "all-patients" if self.experiment_type == "all_patients" else "per-patient"
        print(f"🔍 Analyzing {exp_type_label} experiment: {latest_experiment.name}")
        return str(latest_experiment)
    
    def load_model_names(self, experiment_path: Path = None):
        """
        Load teacher and student model names from pipeline_results.csv.
        
        Args:
            experiment_path: Path to experiment directory (optional, will find latest if not provided)
        """
        if experiment_path is None:
            experiment_path = Path(self.find_latest_experiment())
        else:
            experiment_path = Path(experiment_path)
        
        # Look for pipeline_results.csv in parent directory
        results_csv = experiment_path.parent.parent / "pipeline_results.csv"
        
        if not results_csv.exists():
            print(f"⚠️  Could not find pipeline_results.csv at {results_csv}")
            return
        
        try:
            df = pd.read_csv(results_csv)
            if not df.empty and 'teacher_model' in df.columns and 'student_model' in df.columns:
                self.teacher_model = df['teacher_model'].iloc[0]
                self.student_model = df['student_model'].iloc[0]
                print(f"✅ Loaded model names:")
                print(f"   Teacher: {self.teacher_model}")
                print(f"   Student: {self.student_model}")
        except Exception as e:
            print(f"⚠️  Could not load model names: {e}")
    
    def load_patient_results(self, experiment_path: Path) -> Dict:
        """
        Load patient results from experiment directory structure.
        Handles both per-patient and all-patients experiment types.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary mapping patient_id to performance metrics
        """
        if self.experiment_type == "all_patients":
            return self._load_all_patients_results(experiment_path)
        else:
            return self._load_per_patient_results(experiment_path)
    
    def _load_all_patients_results(self, experiment_path: Path) -> Dict:
        """
        Load results from all-patients experiment (multi-phase from CSVs).
        Single model trained on all patients, inference on each patient.
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary mapping patient_id to multi-phase results
        """
        experiment_path = Path(experiment_path)
        patient_results = {}
        
        print(f"📊 Loading all-patients multi-phase results...")
        
        # Define phase mappings
        phases = {
            'teacher': 'phase_1_teacher',
            'student_baseline': 'phase_2_student',
            'distilled': 'phase_3_distillation'
        }
        
        # Load results from each phase
        for phase_key, phase_dir in phases.items():
            phase_path = experiment_path / phase_dir / "per_patient_inference"
            
            if not phase_path.exists():
                print(f"⚠️  Phase directory not found: {phase_path}")
                continue
            
            # Find experiment_results.csv
            csv_files = list(phase_path.rglob("experiment_results.csv"))
            
            if not csv_files:
                print(f"⚠️  No experiment_results.csv found in {phase_path}")
                continue
            
            csv_path = csv_files[0]
            
            try:
                df = pd.read_csv(csv_path)
                
                for _, row in df.iterrows():
                    patient_id = str(row['patient_id'])
                    
                    if patient_id not in patient_results:
                        patient_results[patient_id] = {}
                    
                    patient_results[patient_id][phase_key] = {
                        'rmse': float(row['rmse']),
                        'mae': float(row['mae']),
                        'mape': float(row['mape']) if 'mape' in row else 0.0
                    }
                
                print(f"✅ Loaded {phase_key} results for {len(df)} patients")
                
            except Exception as e:
                print(f"❌ Error loading {phase_key} results from {csv_path}: {e}")
        
        return patient_results
    
    def _load_per_patient_results(self, experiment_path: Path) -> Dict:
        """
        Load results from per-patient distillation experiment (multi-phase).
        NOW LOADS ALL 3 PHASES: Teacher, Student, Distilled
        
        Args:
            experiment_path: Path to experiment directory
            
        Returns:
            Dictionary mapping patient_id to multi-phase results
        """
        from fairness.utils.analyzer_utils import load_multi_phase_results
        
        patient_results = {}
        
        # Find patient directories
        patient_dirs = []
        for root, dirs, files in os.walk(experiment_path):
            for d in dirs:
                if d.startswith('patient_'):
                    patient_dirs.append(os.path.join(root, d))
        
        print(f"📊 Loading multi-phase results from {len(patient_dirs)} patients...")
        
        for patient_dir in patient_dirs:
            try:
                patient_id = os.path.basename(patient_dir).split('_')[1]
                
                # Load all three phases
                phase_results = load_multi_phase_results(Path(patient_dir))
                
                if phase_results:
                    patient_results[patient_id] = phase_results
                    
            except Exception as e:
                print(f"⚠️  Could not load results for patient directory {patient_dir}: {e}")
        
        return patient_results
    
    def group_by_feature(self, patient_results: Dict) -> Dict:
        """
        Group patient results by demographic feature.
        NOW HANDLES MULTI-PHASE DATA (teacher, student, distilled)
        
        Args:
            patient_results: Dictionary of patient multi-phase performance metrics
            
        Returns:
            Dictionary mapping feature values to lists of patient results
        """
        groups = defaultdict(list)
        
        for patient_id, phase_metrics in patient_results.items():
            if patient_id in self.patient_data:
                feature_value = self.patient_data[patient_id]
                groups[feature_value].append({
                    'patient_id': patient_id,
                    **phase_metrics  # This now includes teacher, student_baseline, distilled
                })
        
        return dict(groups)
    
    def calculate_group_statistics(self, groups: Dict) -> Dict:
        """
        Calculate statistics for each group.
        Handles both per-patient (multi-phase) and all-patients (single-phase) data.
        
        Args:
            groups: Dictionary mapping feature values to patient results
            
        Returns:
            Dictionary with statistics for each group
        """
        statistics = {}
        
        for group_name, patients in groups.items():
            if not patients:
                continue
            
            group_stats = {
                'count': len(patients),
                'patient_ids': [p['patient_id'] for p in patients]
            }
            
            # Determine which phases are available
            # Both experiment types now support all three phases
            phases = ['teacher', 'student_baseline', 'distilled']
            
            # Calculate stats for each available phase
            for phase in phases:
                phase_patients = [p for p in patients if phase in p]
                
                if phase_patients:
                    rmse_values = [p[phase]['rmse'] for p in phase_patients]
                    mae_values = [p[phase]['mae'] for p in phase_patients]
                    
                    group_stats[phase] = {
                        'rmse_mean': np.mean(rmse_values),
                        'rmse_std': np.std(rmse_values),
                        'mae_mean': np.mean(mae_values),
                        'mae_std': np.std(mae_values),
                        'rmse_values': rmse_values,
                        'mae_values': mae_values,
                        'sample_count': len(phase_patients)
                    }
            
            statistics[group_name] = group_stats
        
        return statistics
    
    def calculate_fairness_ratio(self, statistics: Dict, phase: str = 'distilled') -> Tuple[float, str]:
        """
        Calculate fairness ratio (worst/best performance) for a specific phase.
        
        Args:
            statistics: Group statistics dictionary
            phase: Which phase to calculate for ('teacher', 'student_baseline', or 'distilled')
            
        Returns:
            Tuple of (fairness_ratio, fairness_level)
        """
        if not statistics:
            return 0.0, "UNKNOWN"
        
        # Extract RMSE values for the specified phase
        rmse_values = []
        for stats in statistics.values():
            if phase in stats and 'rmse_mean' in stats[phase]:
                rmse_values.append(stats[phase]['rmse_mean'])
        
        if len(rmse_values) < 2:
            return 1.0, "EXCELLENT"
        
        max_rmse = max(rmse_values)
        min_rmse = min(rmse_values)
        
        if min_rmse == 0:
            return 0.0, "ERROR"
        
        ratio = max_rmse / min_rmse
        
        # Determine fairness level
        if ratio < 1.10:
            level = "EXCELLENT"
        elif ratio < 1.25:
            level = "GOOD"
        elif ratio < 1.50:
            level = "ACCEPTABLE"
        else:
            level = "POOR"
        
        return ratio, level
    
    def analyze_distillation_impact(self, statistics: Dict) -> Dict:
        """
        Analyze how distillation affects fairness compared to teacher model.
        
        Args:
            statistics: Group statistics with multi-phase data
            
        Returns:
            Dictionary with distillation impact analysis
        """
        from fairness.utils.analyzer_utils import calculate_distillation_impact
        
        # Calculate fairness ratios for teacher and distilled phases
        teacher_ratio, teacher_level = self.calculate_fairness_ratio(statistics, 'teacher')
        distilled_ratio, distilled_level = self.calculate_fairness_ratio(statistics, 'distilled')
        
        impact = calculate_distillation_impact(teacher_ratio, distilled_ratio)
        impact['teacher_level'] = teacher_level
        impact['distilled_level'] = distilled_level
        impact['teacher_ratio'] = teacher_ratio
        impact['distilled_ratio'] = distilled_ratio
        
        return impact
    
    def save_report(self, report_text: str, filename_prefix: str):
        """
        Save text report to file.
        
        Args:
            report_text: Report content
            filename_prefix: Prefix for filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_label = getattr(self, 'experiment_type', 'per_patient')
        report_file = self.results_dir / f"{filename_prefix}_report_{mode_label}_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\nFull report saved to: {report_file}")
    
    def save_json_report(self, report_data: Dict, filename_prefix: str):
        """
        Save JSON report to file.
        
        Args:
            report_data: Report data as dictionary
            filename_prefix: Prefix for filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_label = getattr(self, 'experiment_type', 'per_patient')
        report_file = self.results_dir / f"{filename_prefix}_report_{mode_label}_{timestamp}.json"
        # Add mode to report metadata
        if isinstance(report_data, dict):
            report_data['experiment_mode'] = mode_label
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nFull report saved to: {report_file}")
        return report_file
    
    def generate_timestamp(self) -> str:
        """Generate timestamp for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @abstractmethod
    def analyze_latest(self):
        """
        Run fairness analysis on the latest experiment.
        Must be implemented by subclasses.
        """
        pass
    
    def visualize(self, statistics: Dict, fairness_ratio: float, impact: Optional[Dict] = None):
        """
        Create comprehensive 4-panel visualization showing distillation impact.
        
        Args:
            statistics: Group statistics dictionary
            fairness_ratio: Overall fairness ratio
            impact: Distillation impact analysis (optional)
        """
        if not statistics:
            print("⚠️ Cannot create visualizations - insufficient data")
            return
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        mode_label = 'All-Patients' if getattr(self, 'experiment_type', 'per_patient') == 'all_patients' else 'Per-Patient'
        
        # Build title with model names
        title_text = f'{self.feature_name} Fairness Analysis ({mode_label} Mode) - Distillation Impact\n'
        title_text += f'Teacher: {self.teacher_model} | Student: {self.student_model}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
        
        groups = list(statistics.keys())
        
        # Create model name mappings
        model_display_names = {
            'Teacher': f'Teacher\n({self.teacher_model})',
            'Student': f'Student\n({self.student_model})',
            'Distilled': f'Distilled\n({self.student_model})'
        }
        
        # 1. RMSE Comparison by Group (Teacher, Student, Distilled)
        models = ['Teacher', 'Student', 'Distilled']
        group_rmse = {model: [] for model in models}
        
        for model_type in ['teacher', 'student_baseline', 'distilled']:
            model_label = model_type.replace('_baseline', '').replace('_', ' ').title()
            for group in groups:
                if model_type in statistics[group]:
                    group_rmse[model_label].append(statistics[group][model_type]['rmse_mean'])
        
        if any(group_rmse.values()):
            x = np.arange(len(groups))
            width = 0.25
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            
            # Create shorter names for legend
            legend_names = {
                'Teacher': self.teacher_model.split('/')[-1] if '/' in self.teacher_model else self.teacher_model[:20],
                'Student': self.student_model.split('/')[-1] if '/' in self.student_model else self.student_model[:20],
                'Distilled': f"Distilled ({self.student_model.split('/')[-1][:15]})" if '/' in self.student_model else f"Distilled ({self.student_model[:15]})"
            }
            
            for i, (model, rmse_vals) in enumerate(group_rmse.items()):
                if rmse_vals:
                    offset = width * (i - 1)
                    bars = ax1.bar(x + offset, rmse_vals, width, label=legend_names[model], 
                                   color=colors[i], alpha=0.8)
                    
                    # Add value labels
                    for bar, val in zip(bars, rmse_vals):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax1.set_xlabel(f'{self.feature_name} Group', fontweight='bold')
            ax1.set_ylabel('RMSE (Lower = Better)', fontweight='bold')
            ax1.set_title(f'Performance by {self.feature_name} Group')
            ax1.set_xticks(x)
            ax1.set_xticklabels(groups, rotation=15, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Fairness Score Progression
        if impact:
            # Calculate fairness scores for each phase
            fairness_scores = []
            model_names = []
            
            for phase in ['teacher', 'student_baseline', 'distilled']:
                ratio, level = self.calculate_fairness_ratio(statistics, phase)
                if ratio > 0:
                    # Convert ratio to fairness score (coefficient of variation style)
                    fairness_score = (ratio - 1.0) / ratio  # Normalized score
                    fairness_scores.append(fairness_score)
                    model_names.append(phase.replace('_', ' ').title())
            
            if fairness_scores:
                ax2.plot(model_names, fairness_scores, 'o-', linewidth=3, markersize=10, color='purple')
                ax2.set_xlabel('Model Type', fontweight='bold')
                ax2.set_ylabel('Fairness Score (Lower = Better)', fontweight='bold')
                ax2.set_title('Fairness Score Progression')
                ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Good Fairness Threshold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Annotate points
                for i, (model, score) in enumerate(zip(model_names, fairness_scores)):
                    ax2.annotate(f'{score:.3f}', (i, score), textcoords="offset points", 
                                xytext=(0,15), ha='center', fontsize=10, fontweight='bold')
        
        # 3. Performance Ratios
        if impact:
            ratios = []
            labels = []
            
            for phase in ['teacher', 'student_baseline', 'distilled']:
                ratio, level = self.calculate_fairness_ratio(statistics, phase)
                if ratio > 0:
                    ratios.append(ratio)
                    labels.append(phase.replace('_', ' ').title())
            
            if ratios:
                colors_bar = ['lightgreen' if r < 1.2 else 'gold' if r < 1.5 else 'salmon' for r in ratios]
                bars3 = ax3.bar(labels, ratios, color=colors_bar, alpha=0.7)
                ax3.set_xlabel('Model Type', fontweight='bold')
                ax3.set_ylabel('Performance Ratio (Worse/Better)', fontweight='bold')
                ax3.set_title(f'{self.feature_name} Performance Ratios')
                ax3.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect Fairness')
                ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold')
                ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Poor Fairness')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                for bar, ratio in zip(bars3, ratios):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # 4. Experiment Summary
        ax4.text(0.1, 0.9, 'EXPERIMENT SUMMARY', fontsize=14, fontweight='bold', 
                transform=ax4.transAxes)
        
        summary_text = []
        total_patients = sum(stats['count'] for stats in statistics.values())
        summary_text.append(f"Total patients analyzed: {total_patients}")
        summary_text.append(f"{self.feature_name} groups: {len(groups)}")
        summary_text.append("")
        
        if impact:
            summary_text.append("Fairness Impact:")
            summary_text.append(f"Teacher:   {impact['teacher_ratio']:.3f}x ({impact['teacher_level']})")
            summary_text.append(f"Distilled: {impact['distilled_ratio']:.3f}x ({impact['distilled_level']})")
            summary_text.append(f"Change:    {impact['change']:+.3f}x ({impact['percent_change']:+.1f}%)")
            summary_text.append("")
            summary_text.append("CONCLUSION:")
            if impact['change'] > 0.02:
                summary_text.append("🚨 Distillation worsens fairness")
            elif impact['change'] > 0:
                summary_text.append("⚠️ Distillation slightly worsens fairness")
            else:
                summary_text.append("✅ Distillation maintains fairness")
            summary_text.append(f"Severity: {impact['severity']}")
        
        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.8 - i*0.08, text, fontsize=11, transform=ax4.transAxes,
                    family='monospace')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = self.generate_timestamp()
        mode_label = getattr(self, 'experiment_type', 'per_patient')
        plot_file = self.results_dir / f"{self.feature_name.lower().replace(' ', '_')}_fairness_analysis_{mode_label}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Visualization saved to: {plot_file}")
