"""
Training and Distillation Analysis Module

Provides functions for analyzing training performance, distillation results,
and comprehensive model evaluation across different phases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings
from utils.analysis_utils import standardize_model_names
warnings.filterwarnings('ignore')

class TrainingAnalyzer:
    """Analyze training performance and distillation results"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.results_path = self.base_path / 'results'
        self.efficiency_path = self.base_path / 'efficiency_experiments'
        
    def load_distillation_results(self) -> pd.DataFrame:
        """Load distillation pipeline results"""
        # Try multiple possible locations for distillation results
        possible_paths = [
            self.efficiency_path / 'distillation_experiments' / 'pipeline_results.csv',
            self.base_path / 'distillation_experiments' / 'pipeline_results.csv',
            self.base_path / 'efficiency_experiments' / 'distillation_experiments' / 'pipeline_results.csv'
        ]
        
        for pipeline_csv in possible_paths:
            if pipeline_csv.exists():
                print(f"📊 Found distillation results at: {pipeline_csv}")
                return pd.read_csv(pipeline_csv)
        
        print(f"⚠️ Distillation results not found. Searched locations:")
        for path in possible_paths:
            print(f"   • {path}")
        return pd.DataFrame()
    
    def load_training_efficiency_data(self) -> pd.DataFrame:
        """Load training efficiency data from experiments"""
        training_records = []
        
        # Look for training experiment directories in multiple locations
        search_paths = [
            self.efficiency_path / 'experiments',
            self.base_path / 'chronos_training_ohiot1dm',
            self.base_path / 'time_llm_training_ohiot1dm',
            self.base_path / 'efficiency_experiments'
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # Find all training directories
            training_dirs = []
            if 'training' in search_path.name.lower():
                training_dirs.append(search_path)
            else:
                training_dirs.extend([d for d in search_path.rglob('*training*') if d.is_dir()])
            
            for training_dir in training_dirs:
                # Find JSON files in training experiments
                json_files = list(training_dir.rglob('*.json'))
                
                for json_file in json_files:
                    if 'efficiency_report' in json_file.name or 'performance_report' in json_file.name:
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract training metrics
                            record = self._extract_training_metrics(data, json_file)
                            if record:
                                training_records.append(record)
                                
                        except Exception as e:
                            print(f"⚠️ Error parsing training file {json_file}: {e}")
        
        return pd.DataFrame(training_records)
    
    def _extract_training_metrics(self, data: Dict, file_path: Path) -> Optional[Dict]:
        """Extract training metrics from JSON data"""
        record = {
            'experiment_type': 'training',
            'file_path': str(file_path),
            'model_name': self._extract_model_name(file_path),
        }
        
        # Check different data structures
        if 'performance_summary' in data:
            # Comprehensive report format
            if 'training' in data['performance_summary']:
                training_data = data['performance_summary']['training']
                record.update(self._parse_training_section(training_data))
            elif 'training_inference' in data['performance_summary']:
                # Time-LLM format - extract training part
                training_data = data['performance_summary']['training_inference']
                record.update(self._parse_training_section(training_data))
        
        elif 'real_performance_measurements' in data:
            # Regular efficiency report format
            perf_data = data['real_performance_measurements']
            record.update(self._parse_training_section(perf_data))
        
        # Only return record if we found some training data
        return record if any(v is not None for k, v in record.items() if k not in ['experiment_type', 'file_path', 'model_name']) else None
    
    def _parse_training_section(self, training_data: Dict) -> Dict:
        """Parse training performance data"""
        return {
            'training_time_hours': training_data.get('total_training_time_hours'),
            'training_time_minutes': training_data.get('total_training_time_minutes'),
            'epochs_completed': training_data.get('epochs_completed'),
            'final_train_loss': training_data.get('final_train_loss'),
            'final_val_loss': training_data.get('final_validation_loss'),
            'best_train_loss': training_data.get('best_train_loss'),
            'best_val_loss': training_data.get('best_validation_loss'),
            'peak_ram_mb': training_data.get('peak_ram_mb') or training_data.get('process_peak_ram_mb'),
            'avg_power_w': training_data.get('average_power_usage_watts'),
            'peak_power_w': training_data.get('peak_power_usage_watts'),
            'peak_gpu_mb': training_data.get('peak_gpu_allocated_mb'),
            'avg_gpu_util_percent': training_data.get('average_gpu_utilization_percent'),
        }
    
    def _extract_model_name(self, file_path: Path) -> str:
        """Extract model name from file path"""
        path_str = str(file_path)
        
        if 'chronos' in path_str.lower():
            if 't5-tiny' in path_str:
                return 'chronos-t5-tiny'
            elif 't5-base' in path_str:
                return 'chronos-t5-base'
            else:
                return 'chronos-unknown'
        elif 'time_llm' in path_str.lower():
            if 'BERT' in path_str:
                return 'BERT'
            elif 'GPT2' in path_str:
                return 'GPT2'
            elif 'LLAMA' in path_str:
                return 'LLAMA'
        
        return 'unknown'
    
    def analyze_training_efficiency(self) -> pd.DataFrame:
        """Analyze training efficiency across models"""
        training_data = self.load_training_efficiency_data()
        
        if training_data.empty:
            print("⚠️ No training efficiency data found")
            return pd.DataFrame()
        
        # Standardize model names first
        training_data = standardize_model_names(training_data)
        
        # Calculate efficiency metrics
        training_summary = []
        
        for model_name in training_data['model_name'].unique():
            model_data = training_data[training_data['model_name'] == model_name]
            
            summary = {
                'model_name': model_name,
                'avg_training_time_hours': model_data['training_time_hours'].mean(),
                'avg_final_train_loss': model_data['final_train_loss'].mean(),
                'avg_final_val_loss': model_data['final_val_loss'].mean(),
                'avg_peak_ram_mb': model_data['peak_ram_mb'].mean(),
                'avg_power_w': model_data['avg_power_w'].mean(),
                'avg_gpu_util_percent': model_data['avg_gpu_util_percent'].mean(),
                'total_experiments': len(model_data),
            }
            
            # Calculate training efficiency score
            if summary['avg_training_time_hours'] and summary['avg_final_val_loss']:
                summary['efficiency_score'] = summary['avg_final_val_loss'] / summary['avg_training_time_hours']
            
            training_summary.append(summary)
        
        return pd.DataFrame(training_summary)
    
    def analyze_distillation_performance(self) -> pd.DataFrame:
        """Analyze distillation performance results"""
        distill_data = self.load_distillation_results()
        
        if distill_data.empty:
            print("⚠️ No distillation results found")
            return pd.DataFrame()
        
        # Calculate distillation metrics
        distill_summary = []
        
        if 'teacher_model' in distill_data.columns and 'student_model' in distill_data.columns:
            # Group by teacher-student pairs
            for (teacher, student), group in distill_data.groupby(['teacher_model', 'student_model']):
                summary = {
                    'teacher_model': teacher,
                    'student_model': student,
                    'avg_rmse': group['rmse'].mean() if 'rmse' in group.columns else None,
                    'avg_mae': group['mae'].mean() if 'mae' in group.columns else None,
                    'improvement_percent': None,
                    'compression_ratio': None,
                    'total_experiments': len(group),
                }
                
                # Calculate improvement metrics if baseline exists
                if 'baseline_rmse' in group.columns:
                    baseline = group['baseline_rmse'].mean()
                    if baseline > 0:
                        summary['improvement_percent'] = ((baseline - summary['avg_rmse']) / baseline) * 100
                
                distill_summary.append(summary)
        
        return pd.DataFrame(distill_summary)
    
    def create_training_plots(self, training_summary: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Create training performance visualizations"""
        if training_summary.empty:
            print("⚠️ No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🚀 Training Performance Analysis', fontsize=16, fontweight='bold')
        
        # Training time comparison
        if 'avg_training_time_hours' in training_summary.columns:
            valid_time = training_summary.dropna(subset=['avg_training_time_hours'])
            if not valid_time.empty:
                axes[0, 0].bar(valid_time['model_name'], valid_time['avg_training_time_hours'])
                axes[0, 0].set_title('Average Training Time (Hours)')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Loss comparison
        if 'avg_final_val_loss' in training_summary.columns:
            valid_loss = training_summary.dropna(subset=['avg_final_val_loss'])
            if not valid_loss.empty:
                axes[0, 1].bar(valid_loss['model_name'], valid_loss['avg_final_val_loss'])
                axes[0, 1].set_title('Final Validation Loss')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        if 'avg_peak_ram_mb' in training_summary.columns:
            valid_memory = training_summary.dropna(subset=['avg_peak_ram_mb'])
            if not valid_memory.empty:
                axes[1, 0].bar(valid_memory['model_name'], valid_memory['avg_peak_ram_mb'])
                axes[1, 0].set_title('Peak RAM Usage (MB)')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Power consumption
        if 'avg_power_w' in training_summary.columns:
            valid_power = training_summary.dropna(subset=['avg_power_w'])
            if not valid_power.empty:
                axes[1, 1].bar(valid_power['model_name'], valid_power['avg_power_w'])
                axes[1, 1].set_title('Average Power Consumption (W)')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to: {save_path}")
        
        plt.show()
    
    def create_distillation_plots(self, distill_summary: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Create distillation performance visualizations"""
        if distill_summary.empty:
            print("⚠️ No distillation data to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('🎓 Distillation Performance Analysis', fontsize=16, fontweight='bold')
        
        # RMSE comparison
        if 'avg_rmse' in distill_summary.columns:
            valid_rmse = distill_summary.dropna(subset=['avg_rmse'])
            if not valid_rmse.empty:
                pair_names = [f"{row['teacher_model']}\n→{row['student_model']}" for _, row in valid_rmse.iterrows()]
                axes[0].bar(range(len(pair_names)), valid_rmse['avg_rmse'])
                axes[0].set_title('Average RMSE by Teacher-Student Pair')
                axes[0].set_xticks(range(len(pair_names)))
                axes[0].set_xticklabels(pair_names, rotation=45, ha='right')
        
        # Improvement percentage
        if 'improvement_percent' in distill_summary.columns:
            valid_improvement = distill_summary.dropna(subset=['improvement_percent'])
            if not valid_improvement.empty:
                pair_names = [f"{row['teacher_model']}\n→{row['student_model']}" for _, row in valid_improvement.iterrows()]
                bars = axes[1].bar(range(len(pair_names)), valid_improvement['improvement_percent'])
                axes[1].set_title('Performance Improvement (%)')
                axes[1].set_xticks(range(len(pair_names)))
                axes[1].set_xticklabels(pair_names, rotation=45, ha='right')
                axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                
                # Color bars based on improvement
                for i, bar in enumerate(bars):
                    if valid_improvement.iloc[i]['improvement_percent'] > 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Distillation plots saved to: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, inference_summary: pd.DataFrame, 
                                    training_summary: pd.DataFrame, 
                                    distill_summary: pd.DataFrame) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("# 🔍 Comprehensive LLM Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Inference summary
        if not inference_summary.empty:
            report.append("## ⚡ Inference Performance")
            fastest = inference_summary.loc[inference_summary['avg_inference_time_ms'].idxmin()]
            report.append(f"- **Fastest Model**: {fastest['model_name']} ({fastest['avg_inference_time_ms']:.1f}ms)")
            
            if 'inference_peak_ram_mb' in inference_summary.columns:
                most_efficient = inference_summary.loc[inference_summary['inference_peak_ram_mb'].idxmin()]
                report.append(f"- **Memory Efficient**: {most_efficient['model_name']} ({most_efficient['inference_peak_ram_mb']:.1f}MB)")
            report.append("")
        
        # Training summary
        if not training_summary.empty:
            report.append("## 🚀 Training Performance")
            if 'avg_training_time_hours' in training_summary.columns:
                fastest_training = training_summary.loc[training_summary['avg_training_time_hours'].idxmin()]
                report.append(f"- **Fastest Training**: {fastest_training['model_name']} ({fastest_training['avg_training_time_hours']:.1f}h)")
            
            if 'avg_final_val_loss' in training_summary.columns:
                best_loss = training_summary.loc[training_summary['avg_final_val_loss'].idxmin()]
                report.append(f"- **Best Validation Loss**: {best_loss['model_name']} ({best_loss['avg_final_val_loss']:.4f})")
            report.append("")
        
        # Distillation summary
        if not distill_summary.empty:
            report.append("## 🎓 Distillation Results")
            if 'improvement_percent' in distill_summary.columns:
                best_distill = distill_summary.loc[distill_summary['improvement_percent'].idxmax()]
                report.append(f"- **Best Distillation**: {best_distill['teacher_model']} → {best_distill['student_model']}")
                report.append(f"- **Improvement**: {best_distill['improvement_percent']:.1f}%")
            report.append("")
        
        report.append("---")
        report.append("*Generated by Comprehensive LLM Analysis Tool*")
        
        return "\n".join(report)

def analyze_all_phases(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze all phases: inference, training, and distillation"""
    
    # Import inference analysis
    import sys
    sys.path.append(str(project_root))
    from utils.enhanced_data_loader import EnhancedEfficiencyDataLoader
    from utils.analysis_utils import calculate_inference_metrics
    
    # Inference analysis
    loader = EnhancedEfficiencyDataLoader(project_root / 'efficiency_experiments')
    efficiency_df = loader.parse_all_data()
    
    inference_data = efficiency_df[
        (efficiency_df['report_type'].isin(['comprehensive_reports', 'real_performance_reports'])) |
        (efficiency_df['experiment_type'].str.contains('inference', na=False))
    ].copy()
    
    inference_summary = calculate_inference_metrics(inference_data)
    
    # Training and distillation analysis
    analyzer = TrainingAnalyzer(project_root)
    training_summary = analyzer.analyze_training_efficiency()
    distill_summary = analyzer.analyze_distillation_performance()
    
    return inference_summary, training_summary, distill_summary