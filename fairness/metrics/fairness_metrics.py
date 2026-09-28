"""
Fairness Metrics for Model Evaluation
====================================

This module implements various fairness metrics to evaluate model performance
across different demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


class FairnessMetrics:
    """Calculate fairness metrics for model evaluation across demographic groups."""
    
    def __init__(self):
        """Initialize fairness metrics calculator."""
        pass
    
    def calculate_group_performance(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  group_labels: np.ndarray,
                                  task_type: str = 'regression') -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each demographic group.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            group_labels: Group membership labels (e.g., 'male', 'female')
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with group names as keys and metrics as values
        """
        unique_groups = np.unique(group_labels)
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = group_labels == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) == 0:
                continue
            
            if task_type == 'regression':
                metrics = self._calculate_regression_metrics(group_y_true, group_y_pred)
            elif task_type == 'classification':
                metrics = self._calculate_classification_metrics(group_y_true, group_y_pred)
            else:
                raise ValueError("task_type must be 'regression' or 'classification'")
            
            group_metrics[str(group)] = metrics
        
        return group_metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics for a group."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
            'sample_size': len(y_true)
        }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics for a group."""
        # Convert predictions to binary if needed
        if len(np.unique(y_pred)) > 2:
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = y_pred
        
        return {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, average='binary', zero_division=0),
            'sample_size': len(y_true)
        }
    
    def demographic_parity_difference(self, 
                                    y_pred: np.ndarray, 
                                    group_labels: np.ndarray,
                                    threshold: float = 0.5) -> float:
        """Calculate demographic parity difference.
        
        Measures the maximum difference in positive prediction rates across groups.
        Supports any number of groups (returns max pairwise difference).
        
        Args:
            y_pred: Predicted probabilities or values
            group_labels: Group membership labels
            threshold: Threshold for binary classification
            
        Returns:
            Maximum demographic parity difference across all group pairs (0 = perfect parity)
        """
        unique_groups = np.unique(group_labels)
        if len(unique_groups) < 2:
            return 0.0
        
        positive_rates = []
        for group in unique_groups:
            group_mask = group_labels == group
            group_pred = y_pred[group_mask]
            positive_rate = np.mean(group_pred > threshold)
            positive_rates.append(positive_rate)
        
        return max(positive_rates) - min(positive_rates)
    
    def equalized_odds_difference(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray, 
                                group_labels: np.ndarray,
                                threshold: float = 0.5) -> Dict[str, float]:
        """Calculate equalized odds difference.
        
        Measures maximum differences in TPR and FPR across all group pairs.
        Supports any number of groups.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            group_labels: Group membership labels
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with max TPR and FPR differences across all group pairs
        """
        unique_groups = np.unique(group_labels)
        if len(unique_groups) < 2:
            return {'tpr_difference': 0.0, 'fpr_difference': 0.0, 'max_difference': 0.0}
        
        y_pred_binary = (y_pred > threshold).astype(int)
        
        tpr_rates = []
        fpr_rates = []
        
        for group in unique_groups:
            group_mask = group_labels == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred_binary[group_mask]
            
            # True Positive Rate (Recall)
            tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
            fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_rates.append(tpr)
            
            # False Positive Rate
            fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
            tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_rates.append(fpr)
        
        tpr_diff = max(tpr_rates) - min(tpr_rates)
        fpr_diff = max(fpr_rates) - min(fpr_rates)
        return {
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'max_difference': max(tpr_diff, fpr_diff)
        }
    
    def statistical_parity_difference(self, 
                                    group_metrics: Dict[str, Dict[str, float]], 
                                    metric_name: str = 'mse') -> float:
        """Calculate statistical parity difference for any performance metric.
        
        Returns the maximum pairwise difference in the metric across all groups.
        Supports any number of groups.
        
        Args:
            group_metrics: Dictionary from calculate_group_performance
            metric_name: Name of the metric to compare
            
        Returns:
            Maximum absolute difference in metric across any pair of groups
        """
        groups = list(group_metrics.keys())
        if len(groups) < 2:
            return 0.0
        
        metric_values = [group_metrics[group][metric_name] for group in groups]
        return max(metric_values) - min(metric_values)
    
    def fairness_through_awareness_score(self, 
                                       group_metrics: Dict[str, Dict[str, float]], 
                                       metric_name: str = 'mse') -> float:
        """Calculate fairness through awareness score.
        
        Lower values indicate better fairness (similar performance across groups).
        
        Args:
            group_metrics: Dictionary from calculate_group_performance
            metric_name: Name of the metric to compare
            
        Returns:
            Coefficient of variation across groups (0 = perfect fairness)
        """
        metric_values = [group_metrics[group][metric_name] for group in group_metrics.keys()]
        
        if len(metric_values) < 2:
            return 0.0
        
        mean_metric = np.mean(metric_values)
        std_metric = np.std(metric_values)
        
        # Coefficient of variation
        return std_metric / mean_metric if mean_metric != 0 else 0.0
    
    def calculate_comprehensive_fairness_report(self, 
                                              y_true: np.ndarray,
                                              y_pred: np.ndarray,
                                              group_labels: np.ndarray,
                                              task_type: str = 'regression',
                                              group_attribute: str = 'group') -> Dict:
        """Generate a comprehensive fairness evaluation report.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            group_labels: Group membership labels
            task_type: 'regression' or 'classification'
            group_attribute: Name of the group attribute being analyzed
            
        Returns:
            Comprehensive fairness report dictionary
        """
        # Calculate group-wise performance
        group_metrics = self.calculate_group_performance(y_true, y_pred, group_labels, task_type)
        
        # Overall performance
        if task_type == 'regression':
            overall_metrics = self._calculate_regression_metrics(y_true, y_pred)
            primary_metric = 'mse'
        else:
            overall_metrics = self._calculate_classification_metrics(y_true, y_pred)
            primary_metric = 'accuracy'
        
        report = {
            'group_attribute': group_attribute,
            'task_type': task_type,
            'overall_metrics': overall_metrics,
            'group_metrics': group_metrics,
            'fairness_metrics': {}
        }
        
        # Calculate fairness metrics
        if len(group_metrics) == 2:
            # Demographic parity (for classification-like tasks)
            if task_type == 'classification':
                report['fairness_metrics']['demographic_parity_difference'] = \
                    self.demographic_parity_difference(y_pred, group_labels)
                
                # Convert regression predictions to binary for fairness metrics
                y_true_binary = (y_true > np.median(y_true)).astype(int)
                report['fairness_metrics']['equalized_odds'] = \
                    self.equalized_odds_difference(y_true_binary, y_pred, group_labels)
            
            # Statistical parity for any metric
            report['fairness_metrics']['statistical_parity_difference'] = \
                self.statistical_parity_difference(group_metrics, primary_metric)
        
        # Fairness through awareness (works for any number of groups)
        report['fairness_metrics']['fairness_through_awareness'] = \
            self.fairness_through_awareness_score(group_metrics, primary_metric)
        
        return report
    
    def print_fairness_report(self, report: Dict):
        """Print a formatted fairness evaluation report."""
        print("\n" + "="*80)
        print("FAIRNESS EVALUATION REPORT")
        print("="*80)
        print(f"Group Attribute: {report['group_attribute']}")
        print(f"Task Type: {report['task_type']}")
        
        print(f"\nOverall Performance:")
        for metric, value in report['overall_metrics'].items():
            if metric != 'sample_size':
                print(f"  {metric.upper()}: {value:.4f}")
        print(f"  Total Samples: {report['overall_metrics'].get('sample_size', 'N/A')}")
        
        print(f"\nGroup-wise Performance:")
        for group, metrics in report['group_metrics'].items():
            print(f"  {group}:")
            for metric, value in metrics.items():
                if metric != 'sample_size':
                    print(f"    {metric.upper()}: {value:.4f}")
                else:
                    print(f"    Samples: {value}")
        
        print(f"\nFairness Metrics:")
        for metric, value in report['fairness_metrics'].items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"    {sub_metric}: {sub_value:.4f}")
            else:
                print(f"  {metric}: {value:.4f}")
        
        # Interpretation
        print(f"\nFairness Interpretation:")
        fairness_score = report['fairness_metrics'].get('fairness_through_awareness', 0)
        if fairness_score < 0.1:
            print("  ✅ Model shows good fairness (low performance variation across groups)")
        elif fairness_score < 0.3:
            print("  ⚠️  Model shows moderate fairness concerns")
        else:
            print("  ❌ Model shows significant fairness issues (high performance variation)")


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate patient data with bias
    gender = np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4])
    
    # True values with some bias
    y_true = np.random.normal(0, 1, n_samples)
    # Add bias: males have slightly higher values
    y_true[gender == 'male'] += 0.2
    
    # Predictions with amplified bias (unfair model)
    y_pred = y_true + np.random.normal(0, 0.1, n_samples)
    y_pred[gender == 'male'] += 0.3  # Model is biased toward males
    
    # Initialize fairness metrics
    fairness_metrics = FairnessMetrics()
    
    # Generate comprehensive report
    report = fairness_metrics.calculate_comprehensive_fairness_report(
        y_true=y_true,
        y_pred=y_pred,
        group_labels=gender,
        task_type='regression',
        group_attribute='Gender'
    )
    
    # Print the report
    fairness_metrics.print_fairness_report(report)