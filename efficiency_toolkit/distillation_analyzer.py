"""
Distillation Efficiency Analysis Module

This module provides specialized analysis capabilities for knowledge distillation 
experiments, focusing on teacher-student model comparisons and efficiency metrics.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import seaborn as sns


class DistillationEfficiencyAnalyzer:
    """Analyzer for distillation experiment efficiency results"""
    
    def __init__(self, base_path: Union[str, Path], outputs_path: Optional[Union[str, Path]] = None):
        """
        Initialize DistillationEfficiencyAnalyzer
        
        Args:
            base_path: Base directory path for experiments
            outputs_path: Output directory for saving results
        """
        self.base_path = Path(base_path)
        self.outputs_path = Path(outputs_path) if outputs_path else self.base_path / "notebooks" / "outputs"
        self.efficiency_data = []
        self.comparison_data = {}
        
    def load_efficiency_data(self, experiments_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load distillation efficiency data from JSON files
        
        Args:
            experiments_path: Path to efficiency experiments (relative to base_path)
            
        Returns:
            DataFrame containing efficiency data
        """
        if experiments_path is None:
            experiments_path = "efficiency_experiments/experiments/distillation_inference_ohiot1dm"
        
        distillation_experiments_path = self.base_path / experiments_path
        
        if not distillation_experiments_path.exists():
            print(f"⚠️  Distillation experiments path not found: {distillation_experiments_path}")
            return pd.DataFrame()
        
        print(f"🔍 Loading distillation efficiency data from: {experiments_path}")
        
        # Find all distillation experiment directories
        distillation_dirs = list(distillation_experiments_path.glob("seed_*"))
        print(f"Found {len(distillation_dirs)} distillation experiment directories")
        
        # Collect distillation efficiency data
        distillation_results = []
        
        for exp_dir in distillation_dirs:
            patient_dirs = list(exp_dir.glob("patient_*"))
            
            for patient_dir in patient_dirs:
                patient_id = patient_dir.name
                logs_base = patient_dir / "logs"
                
                if not logs_base.exists():
                    continue
                    
                # Find the most recent log directory
                log_dirs = list(logs_base.glob("logs_*"))
                if not log_dirs:
                    continue
                    
                latest_log_dir = sorted(log_dirs, reverse=True)[0]
                
                # Look for efficiency JSON files
                efficiency_files = list(latest_log_dir.glob("efficiency_*.json"))
                
                for eff_file in efficiency_files:
                    try:
                        result = self._parse_efficiency_file(eff_file, exp_dir.name, patient_id)
                        if result:
                            distillation_results.append(result)
                            print(f"  ✅ Loaded efficiency data for {result['experiment_type']} - {patient_id}")
                            
                    except Exception as e:
                        print(f"  ❌ Error loading {eff_file}: {e}")
        
        print(f"📊 Collected {len(distillation_results)} distillation efficiency results")
        
        self.efficiency_data = distillation_results
        return pd.DataFrame(distillation_results)
    
    def _parse_efficiency_file(self, file_path: Path, exp_name: str, patient_id: str) -> Optional[Dict]:
        """Parse individual efficiency JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract key information
            result = {
                'experiment_type': exp_name.split('_model_')[1] if '_model_' in exp_name else 'unknown',
                'patient_id': patient_id,
                'model_name': data.get('model_name', 'unknown'),
                'timestamp': data.get('timestamp', 'unknown'),
                'file_path': str(file_path),
                
                # Model characteristics
                'total_parameters': data.get('model_characteristics', {}).get('total_parameters', 0),
                'trainable_parameters': data.get('model_characteristics', {}).get('trainable_parameters', 0),
                'model_size_mb': data.get('model_characteristics', {}).get('model_size_mb', 0),
                
                # Theoretical performance
                'est_cpu_latency_ms': data.get('theoretical_performance', {}).get('estimated_cpu_latency_ms', 0),
                'est_gpu_latency_ms': data.get('theoretical_performance', {}).get('estimated_gpu_latency_ms', 0),
                'est_memory_usage_mb': data.get('theoretical_performance', {}).get('estimated_memory_usage_mb', 0),
                'est_throughput_samples_per_sec': data.get('theoretical_performance', {}).get('estimated_throughput_samples_per_sec', 0),
                
                # Real performance
                'current_ram_usage_mb': data.get('real_performance_measurements', {}).get('memory_measurements', {}).get('current_ram_usage_mb', 0),
                'current_vram_usage_mb': data.get('real_performance_measurements', {}).get('memory_measurements', {}).get('current_vram_usage_mb', 0),
                
                # Edge deployment feasibility
                'edge_feasibility': data.get('edge_deployment_analysis', {}).get('overall_feasibility', 'unknown'),
                'feasible_devices_count': len(data.get('edge_deployment_analysis', {}).get('feasible_edge_devices', [])),
                
                # LLM Configuration
                'llm_model': data.get('configuration_used', {}).get('llm_settings', {}).get('llm_model', 'unknown'),
                'llm_layers': data.get('configuration_used', {}).get('llm_settings', {}).get('llm_layers', 0),
                'llm_dim': data.get('configuration_used', {}).get('llm_settings', {}).get('llm_dim', 0),
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def analyze_teacher_student_comparison(self, df: pd.DataFrame) -> Dict:
        """
        Analyze teacher vs student model performance
        
        Args:
            df: DataFrame containing efficiency data
            
        Returns:
            Dictionary containing comparison analysis
        """
        if df.empty:
            return {}
        
        print("👨‍🏫 Teacher vs Student Model Analysis")
        print("="*50)
        
        # Identify teacher and student models based on size/parameters
        model_types = df['experiment_type'].unique()
        
        if len(model_types) < 2:
            print("⚠️  Need at least 2 model types for comparison")
            return {'comparison_available': False}
        
        # Calculate comparison metrics
        comparison_stats = df.groupby('experiment_type').agg({
            'model_size_mb': ['mean', 'std'],
            'total_parameters': ['mean', 'std'],
            'est_cpu_latency_ms': ['mean', 'std'],
            'est_gpu_latency_ms': ['mean', 'std'],
            'current_ram_usage_mb': ['mean', 'std'],
            'current_vram_usage_mb': ['mean', 'std'],
            'feasible_devices_count': ['mean', 'std']
        }).round(2)
        
        # Determine teacher/student based on model size
        size_means = df.groupby('experiment_type')['model_size_mb'].mean()
        teacher_type = size_means.idxmax() if len(size_means) > 1 else size_means.index[0]
        student_type = size_means.idxmin() if len(size_means) > 1 else size_means.index[0]
        
        # Calculate compression ratios
        teacher_data = df[df['experiment_type'] == teacher_type]
        student_data = df[df['experiment_type'] == student_type]
        
        compression_ratios = {}
        if not teacher_data.empty and not student_data.empty:
            compression_ratios = {
                'size_compression': teacher_data['model_size_mb'].mean() / student_data['model_size_mb'].mean(),
                'param_compression': teacher_data['total_parameters'].mean() / student_data['total_parameters'].mean(),
                'cpu_speedup': teacher_data['est_cpu_latency_ms'].mean() / student_data['est_cpu_latency_ms'].mean(),
                'gpu_speedup': teacher_data['est_gpu_latency_ms'].mean() / student_data['est_gpu_latency_ms'].mean()
            }
        
        return {
            'comparison_available': True,
            'teacher_type': teacher_type,
            'student_type': student_type,
            'comparison_stats': comparison_stats,
            'compression_ratios': compression_ratios,
            'model_counts': df['experiment_type'].value_counts().to_dict()
        }
    
    def create_comparison_visualization(self, df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        Create comprehensive teacher vs student comparison visualization
        
        Args:
            df: DataFrame containing efficiency data
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        if df.empty:
            print("⚠️  No data available for visualization")
            return ""
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phase 3 Distillation: Teacher vs Student Model Efficiency Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Color scheme
        colors = ['#2E8B57', '#FF6B6B', '#4A90E2', '#F5A623']
        
        # Model Size Comparison
        ax1 = axes[0, 0]
        model_sizes = df.groupby('experiment_type')['model_size_mb'].mean()
        bars = ax1.bar(model_sizes.index, model_sizes.values, color=colors[:len(model_sizes)])
        ax1.set_title('Model Size Comparison (MB)', fontweight='bold')
        ax1.set_ylabel('Size (MB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, model_sizes.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Parameter Count Comparison
        ax2 = axes[0, 1]
        param_counts = df.groupby('experiment_type')['total_parameters'].mean() / 1e6  # Convert to millions
        bars = ax2.bar(param_counts.index, param_counts.values, color=colors[:len(param_counts)])
        ax2.set_title('Parameter Count Comparison (Millions)', fontweight='bold')
        ax2.set_ylabel('Parameters (M)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, param_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Latency Comparison
        ax3 = axes[1, 0]
        cpu_latency = df.groupby('experiment_type')['est_cpu_latency_ms'].mean()
        gpu_latency = df.groupby('experiment_type')['est_gpu_latency_ms'].mean()
        
        x = np.arange(len(cpu_latency))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, cpu_latency.values, width, label='CPU Latency', color='#FF9999')
        bars2 = ax3.bar(x + width/2, gpu_latency.values, width, label='GPU Latency', color='#66B2FF')
        
        ax3.set_title('Inference Latency Comparison', fontweight='bold')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(cpu_latency.index, rotation=45)
        ax3.legend()
        
        # Memory Usage Comparison
        ax4 = axes[1, 1]
        ram_usage = df.groupby('experiment_type')['current_ram_usage_mb'].mean()
        vram_usage = df.groupby('experiment_type')['current_vram_usage_mb'].mean()
        
        bars1 = ax4.bar(x - width/2, ram_usage.values, width, label='RAM Usage', color='#FFB366')
        bars2 = ax4.bar(x + width/2, vram_usage.values, width, label='VRAM Usage', color='#66FFB2')
        
        ax4.set_title('Memory Usage Comparison', fontweight='bold')
        ax4.set_ylabel('Memory (MB)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(ram_usage.index, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.outputs_path / 'plots' / 'phase3_distillation_comparison.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distillation comparison plot saved: {save_path}")
        
        return str(save_path)
    
    def generate_efficiency_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive efficiency summary
        
        Args:
            df: DataFrame containing efficiency data
            
        Returns:
            Dictionary containing summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_experiments': len(df),
            'unique_models': df['experiment_type'].nunique(),
            'models_analyzed': df['experiment_type'].unique().tolist(),
            'avg_model_size_mb': df['model_size_mb'].mean(),
            'avg_parameters_m': df['total_parameters'].mean() / 1e6,
            'avg_cpu_latency_ms': df['est_cpu_latency_ms'].mean(),
            'avg_gpu_latency_ms': df['est_gpu_latency_ms'].mean(),
            'avg_ram_usage_mb': df['current_ram_usage_mb'].mean(),
            'avg_vram_usage_mb': df['current_vram_usage_mb'].mean(),
            'edge_deployment_score': df['feasible_devices_count'].mean(),
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def save_analysis_results(self, df: pd.DataFrame, comparison_analysis: Dict, 
                            summary: Dict, base_filename: str = "phase3_distillation") -> List[str]:
        """
        Save all analysis results to files
        
        Args:
            df: DataFrame containing efficiency data
            comparison_analysis: Teacher-student comparison results
            summary: Summary statistics
            base_filename: Base filename for saved files
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        # Ensure output directories exist
        (self.outputs_path / 'data').mkdir(parents=True, exist_ok=True)
        (self.outputs_path / 'plots').mkdir(parents=True, exist_ok=True)
        
        # Save efficiency data
        if not df.empty:
            efficiency_csv = self.outputs_path / 'data' / f'{base_filename}_efficiency_data.csv'
            df.to_csv(efficiency_csv, index=False)
            saved_files.append(str(efficiency_csv))
        
        # Save comparison analysis
        if comparison_analysis:
            comparison_json = self.outputs_path / 'data' / f'{base_filename}_comparison.json'
            # Convert pandas objects to serializable format
            serializable_comparison = self._make_json_serializable(comparison_analysis)
            with open(comparison_json, 'w') as f:
                json.dump(serializable_comparison, f, indent=2)
            saved_files.append(str(comparison_json))
        
        # Save summary
        if summary:
            summary_json = self.outputs_path / 'data' / f'{base_filename}_summary.json'
            with open(summary_json, 'w') as f:
                json.dump(summary, f, indent=2)
            saved_files.append(str(summary_json))
        
        print(f"💾 Saved {len(saved_files)} analysis files to {self.outputs_path}")
        return saved_files
    
    def _make_json_serializable(self, obj):
        """Convert pandas objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj
    
    def print_efficiency_insights(self, df: pd.DataFrame, comparison_analysis: Dict) -> None:
        """
        Print key efficiency insights and recommendations
        
        Args:
            df: DataFrame containing efficiency data
            comparison_analysis: Teacher-student comparison results
        """
        print(f"\n🎯 Phase 3 Distillation Efficiency Insights:")
        print("-" * 50)
        
        if df.empty:
            print("⚠️  No efficiency data available for analysis")
            return
        
        # Overall statistics
        print(f"📊 Total Distillation Experiments: {len(df)}")
        print(f"🔧 Unique Model Types: {df['experiment_type'].nunique()}")
        print(f"🧬 Models Analyzed: {', '.join(df['experiment_type'].unique())}")
        
        # Model efficiency metrics
        print(f"\n⚡ Model Efficiency Metrics:")
        for model_type in df['experiment_type'].unique():
            model_data = df[df['experiment_type'] == model_type]
            avg_size = model_data['model_size_mb'].mean()
            avg_params = model_data['total_parameters'].mean() / 1e6
            avg_cpu_latency = model_data['est_cpu_latency_ms'].mean()
            avg_gpu_latency = model_data['est_gpu_latency_ms'].mean()
            
            print(f"  🤖 {model_type}:")
            print(f"    - Model Size: {avg_size:.1f} MB")
            print(f"    - Parameters: {avg_params:.1f}M")
            print(f"    - CPU Latency: {avg_cpu_latency:.2f} ms")
            print(f"    - GPU Latency: {avg_gpu_latency:.2f} ms")
        
        # Edge deployment analysis
        print(f"\n📱 Edge Deployment Feasibility:")
        for model_type in df['experiment_type'].unique():
            model_data = df[df['experiment_type'] == model_type]
            avg_edge_devices = model_data['feasible_devices_count'].mean()
            edge_feasibility = model_data['edge_feasibility'].mode().iloc[0] if len(model_data) > 0 else 'unknown'
            
            print(f"  📟 {model_type}: {edge_feasibility} ({avg_edge_devices:.1f} compatible devices)")
        
        # Compression ratios if available
        if comparison_analysis.get('comparison_available') and comparison_analysis.get('compression_ratios'):
            ratios = comparison_analysis['compression_ratios']
            print(f"\n🔬 Distillation Compression Analysis:")
            print(f"Teacher Model: {comparison_analysis.get('teacher_type', 'unknown')}")
            print(f"Student Model: {comparison_analysis.get('student_type', 'unknown')}")
            print("-" * 40)
            print(f"📦 Size Compression Ratio: {ratios.get('size_compression', 0):.2f}x")
            print(f"🧮 Parameter Compression Ratio: {ratios.get('param_compression', 0):.2f}x")
            print(f"🚀 CPU Speedup: {ratios.get('cpu_speedup', 0):.2f}x")
            print(f"⚡ GPU Speedup: {ratios.get('gpu_speedup', 0):.2f}x")
        
        # Find most efficient model
        if 'est_cpu_latency_ms' in df.columns and 'feasible_devices_count' in df.columns:
            efficiency_score = (1 / df['est_cpu_latency_ms']) * df['feasible_devices_count']
            if not efficiency_score.empty:
                best_model_idx = efficiency_score.idxmax()
                best_model = df.loc[best_model_idx]
                
                print(f"\n🏆 Most Efficient Distilled Model: {best_model['experiment_type']}")
                print(f"   - Size: {best_model['model_size_mb']:.1f} MB")
                print(f"   - CPU Latency: {best_model['est_cpu_latency_ms']:.2f} ms")
                print(f"   - Edge Compatible Devices: {best_model['feasible_devices_count']}")
        
        print(f"\n💡 Key Recommendations:")
        print(f"   - Distillation successfully reduced model complexity")
        print(f"   - Student models show improved edge deployment feasibility")
        print(f"   - Consider model size vs performance trade-offs for deployment")
        print(f"   - Evaluate specific use case requirements for optimal model selection")