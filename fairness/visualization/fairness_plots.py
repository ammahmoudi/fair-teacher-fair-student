"""
Fairness Visualization Tools
===========================

This module provides visualization tools for fairness analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path

# Add project root to path dynamically
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.path_utils import get_project_root


class FairnessVisualizer:
    """Create visualizations for fairness analysis."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the fairness visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        if output_dir is None:
            output_dir = str(get_project_root() / "fairness" / "visualizations")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_demographic_distribution(self, 
                                    patient_data: pd.DataFrame, 
                                    attribute: str,
                                    save_path: Optional[str] = None) -> None:
        """Plot demographic distribution of patients.
        
        Args:
            patient_data: DataFrame with patient information
            attribute: Attribute to visualize
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        value_counts = patient_data[attribute].value_counts()
        ax1.bar(value_counts.index, value_counts.values)
        ax1.set_title(f'{attribute} Distribution (Count)')
        ax1.set_xlabel(attribute)
        ax1.set_ylabel('Number of Patients')
        
        # Add count labels on bars
        for i, v in enumerate(value_counts.values):
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # Percentage plot
        percentages = (value_counts / len(patient_data) * 100)
        ax2.pie(percentages.values, labels=percentages.index, autopct='%1.1f%%')
        ax2.set_title(f'{attribute} Distribution (Percentage)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'demographic_distribution_{attribute.lower()}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_group_performance_comparison(self, 
                                        group_metrics: Dict[str, Dict[str, float]], 
                                        metric_name: str = 'mse',
                                        title: str = "Group Performance Comparison",
                                        save_path: Optional[str] = None) -> None:
        """Plot performance comparison across groups.
        
        Args:
            group_metrics: Dictionary with group metrics
            metric_name: Name of the metric to plot
            title: Plot title
            save_path: Path to save the plot
        """
        groups = list(group_metrics.keys())
        values = [group_metrics[group][metric_name] for group in groups]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(groups, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(groups)])
        ax.set_title(title)
        ax.set_xlabel('Groups')
        ax.set_ylabel(f'{metric_name.upper()}')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom')
        
        # Add fairness threshold line (example: within 10% of mean)
        mean_value = np.mean(values)
        threshold = 0.1 * mean_value
        ax.axhline(y=mean_value + threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Fairness threshold (+{threshold:.4f})')
        ax.axhline(y=mean_value - threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Fairness threshold (-{threshold:.4f})')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'group_performance_{metric_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_fairness_metrics_radar(self, 
                                  fairness_metrics: Dict[str, float],
                                  title: str = "Fairness Metrics Overview",
                                  save_path: Optional[str] = None) -> None:
        """Create radar plot for multiple fairness metrics.
        
        Args:
            fairness_metrics: Dictionary with fairness metric values
            title: Plot title
            save_path: Path to save the plot
        """
        # Prepare data
        metrics = list(fairness_metrics.keys())
        values = list(fairness_metrics.values())
        
        # Normalize values to 0-1 scale for radar plot
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]
        
        # Create radar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        normalized_values += normalized_values[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.fill(angles, normalized_values, color='blue', alpha=0.25)
        ax.plot(angles, normalized_values, color='blue', linewidth=2)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, weight='bold', pad=20)
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'fairness_metrics_radar.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_teacher_vs_student_fairness(self, 
                                       teacher_metrics: Dict[str, Dict[str, float]],
                                       student_metrics: Dict[str, Dict[str, float]],
                                       distilled_metrics: Dict[str, Dict[str, float]],
                                       metric_name: str = 'mse',
                                       save_path: Optional[str] = None) -> None:
        """Compare fairness across teacher, student, and distilled models.
        
        Args:
            teacher_metrics: Teacher model group metrics
            student_metrics: Student baseline model group metrics  
            distilled_metrics: Distilled model group metrics
            metric_name: Metric to compare
            save_path: Path to save the plot
        """
        groups = list(teacher_metrics.keys())
        x = np.arange(len(groups))
        width = 0.25
        
        teacher_values = [teacher_metrics[group][metric_name] for group in groups]
        student_values = [student_metrics[group][metric_name] for group in groups]
        distilled_values = [distilled_metrics[group][metric_name] for group in groups]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width, teacher_values, width, label='Teacher', color='skyblue')
        bars2 = ax.bar(x, student_values, width, label='Student (Baseline)', color='lightcoral')
        bars3 = ax.bar(x + width, distilled_values, width, label='Distilled', color='lightgreen')
        
        ax.set_xlabel('Groups')
        ax.set_ylabel(f'{metric_name.upper()}')
        ax.set_title(f'Model Performance Comparison: {metric_name.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'teacher_vs_student_fairness_{metric_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_fairness_dashboard(self, 
                                            experiment_results: Dict,
                                            save_path: Optional[str] = None) -> None:
        """Create interactive dashboard for fairness analysis.
        
        Args:
            experiment_results: Results from fairness experiments
            save_path: Path to save the HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Group Distribution', 'Performance Comparison', 
                          'Fairness Metrics', 'Model Comparison'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Extract data
        groups = experiment_results.get('groups', {})
        if groups:
            # Group distribution pie chart
            group_names = list(groups.keys())
            group_sizes = [len(ids) for ids in groups.values()]
            
            fig.add_trace(
                go.Pie(labels=group_names, values=group_sizes, name="Distribution"),
                row=1, col=1
            )
            
            # Performance comparison (placeholder data)
            metrics = ['MSE', 'MAE', 'R2']
            group1_perf = [0.12, 0.10, 0.85]
            group2_perf = [0.18, 0.15, 0.78]
            
            fig.add_trace(
                go.Bar(x=metrics, y=group1_perf, name=group_names[0] if group_names else 'Group 1'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=metrics, y=group2_perf, name=group_names[1] if len(group_names) > 1 else 'Group 2'),
                row=1, col=2
            )
            
            # Fairness metrics
            fairness_metrics = ['Demographic Parity', 'Equalized Odds', 'Statistical Parity']
            fairness_values = [0.05, 0.08, 0.06]  # Placeholder
            
            fig.add_trace(
                go.Bar(x=fairness_metrics, y=fairness_values, name="Fairness Score",
                      marker_color='orange'),
                row=2, col=1
            )
            
            # Model comparison
            models = ['Teacher', 'Student', 'Distilled']
            model_performance = [0.10, 0.18, 0.14]  # Placeholder
            
            fig.add_trace(
                go.Bar(x=models, y=model_performance, name="Model Performance",
                      marker_color='purple'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Fairness Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            dashboard_path = self.output_dir / 'fairness_dashboard.html'
            fig.write_html(str(dashboard_path))
            print(f"Interactive dashboard saved to: {dashboard_path}")
        
        fig.show()
    
    def plot_intersectional_analysis(self, 
                                   patient_data: pd.DataFrame,
                                   primary_attribute: str = 'Gender',
                                   secondary_attribute: str = 'Age',
                                   save_path: Optional[str] = None) -> None:
        """Plot intersectional analysis across two attributes.
        
        Args:
            patient_data: DataFrame with patient information
            primary_attribute: Primary attribute for analysis
            secondary_attribute: Secondary attribute for analysis
            save_path: Path to save the plot
        """
        # Create crosstab
        crosstab = pd.crosstab(patient_data[primary_attribute], 
                              patient_data[secondary_attribute], 
                              margins=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        sns.heatmap(crosstab.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Intersectional Analysis: {primary_attribute} × {secondary_attribute}')
        ax1.set_xlabel(secondary_attribute)
        ax1.set_ylabel(primary_attribute)
        
        # Stacked bar chart
        crosstab_pct = crosstab.iloc[:-1, :-1].div(crosstab.iloc[:-1, -1], axis=0) * 100
        crosstab_pct.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title(f'Percentage Distribution: {primary_attribute} × {secondary_attribute}')
        ax2.set_xlabel(primary_attribute)
        ax2.set_ylabel('Percentage')
        ax2.legend(title=secondary_attribute)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'intersectional_{primary_attribute}_{secondary_attribute}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_fairness_summary_report(self, 
                                       experiment_results: Dict,
                                       save_path: Optional[str] = None) -> str:
        """Generate visual summary report with key plots.
        
        Args:
            experiment_results: Results from fairness experiments
            save_path: Path to save the summary
            
        Returns:
            Path to generated summary report
        """
        # This would create a comprehensive visual report
        # For now, we'll create a placeholder structure
        
        report_content = f"""
        # Fairness Analysis Summary Report
        
        ## Experiment Overview
        - Attribute Analyzed: {experiment_results.get('attribute', 'N/A')}
        - Groups: {list(experiment_results.get('groups', {}).keys())}
        - Models: Teacher vs Student vs Distilled
        
        ## Key Findings
        1. Demographic distribution analysis
        2. Performance disparities across groups
        3. Impact of distillation on fairness
        4. Recommendations for improvement
        
        ## Visualizations Generated
        - Group distribution plots
        - Performance comparison charts
        - Fairness metrics radar
        - Interactive dashboard
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            return save_path
        else:
            report_path = self.output_dir / 'fairness_summary_report.md'
            with open(report_path, 'w') as f:
                f.write(report_content)
            return str(report_path)


# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = FairnessVisualizer()
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_patient_data = pd.DataFrame({
        'ID': range(1, 13),
        'Gender': ['male', 'female'] * 6,
        'Age': ['20-40', '40-60', '60-80'] * 4
    })
    
    print("Creating sample fairness visualizations...")
    
    # Demographic distribution
    visualizer.plot_demographic_distribution(sample_patient_data, 'Gender')
    
    # Sample group metrics
    sample_group_metrics = {
        'male': {'mse': 0.12, 'mae': 0.10, 'r2': 0.85},
        'female': {'mse': 0.18, 'mae': 0.15, 'r2': 0.78}
    }
    
    # Performance comparison
    visualizer.plot_group_performance_comparison(sample_group_metrics, 'mse')
    
    # Fairness metrics radar
    sample_fairness_metrics = {
        'Demographic Parity': 0.05,
        'Equalized Odds': 0.08,
        'Statistical Parity': 0.06,
        'Fairness Through Awareness': 0.12
    }
    visualizer.plot_fairness_metrics_radar(sample_fairness_metrics)
    
    # Intersectional analysis
    visualizer.plot_intersectional_analysis(sample_patient_data, 'Gender', 'Age')
    
    print(f"Visualizations saved to: {visualizer.output_dir}")
    print("✅ Fairness visualization framework ready!")