"""
Advanced Fairness Metrics for Blood Glucose Prediction
=======================================================

This module implements three advanced fairness metrics specifically designed
for evaluating fairness in blood glucose prediction models:

1. Demographic Parity Gap (DP Gap): Measures if critical alerts are distributed
   equally across demographic groups
2. Equal Opportunity Gap (EO Gap): Ensures the model detects high-risk events
   equally well for all groups
3. Fairness Violation Objective (FVO): Measures maximum accuracy disparity
   between any two demographic subgroups

Calculation Modes:
------------------
1. SIMPLE (default): Direct calculation on raw window predictions
2. TIMELINE_RECONSTRUCTION: Reconstruct timeline by averaging overlapping windows,
   then binarize the averaged points and calculate metrics
3. WINDOW_MAJORITY: Binarize each window, then use majority vote across overlapping
   windows at each timestamp

References:
- Based on fairness metrics from algorithmic fairness literature
- Adapted for continuous blood glucose prediction with critical threshold detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
import warnings

# Calculation mode constants
CALC_MODE_SIMPLE = 'simple'
CALC_MODE_TIMELINE_RECONSTRUCTION = 'timeline_reconstruction'
CALC_MODE_WINDOW_MAJORITY = 'window_majority'
VALID_CALC_MODES = [CALC_MODE_SIMPLE, CALC_MODE_TIMELINE_RECONSTRUCTION, CALC_MODE_WINDOW_MAJORITY]


class AdvancedFairnessMetrics:
    """
    Advanced fairness metrics for blood glucose prediction models.
    
    These metrics are designed to evaluate fairness in critical health predictions
    where false negatives (missing a hypoglycemic event) can have serious consequences.
    """
    
    def __init__(self, 
                 hypoglycemia_threshold: float = 70.0,
                 hyperglycemia_threshold: float = 180.0):
        """
        Initialize advanced fairness metrics calculator.
        
        Args:
            hypoglycemia_threshold: Blood glucose level below which is considered
                                   hypoglycemic (default: 70 mg/dL)
            hyperglycemia_threshold: Blood glucose level above which is considered
                                    hyperglycemic (default: 180 mg/dL)
        """
        self.hypoglycemia_threshold = hypoglycemia_threshold
        self.hyperglycemia_threshold = hyperglycemia_threshold
    
    def _reconstruct_timeline(self,
                             windowed_values: List[np.ndarray],
                             window_start_indices: List[int],
                             total_length: Optional[int] = None,
                             aggregation: str = 'mean') -> np.ndarray:
        """
        Reconstruct a continuous timeline from overlapping windows.
        
        For overlapping windows, each timestamp may be covered by multiple windows.
        This method aggregates values at each timestamp using the specified method.
        
        Args:
            windowed_values: List of arrays, each containing predictions for one window
            window_start_indices: List of starting indices for each window
            total_length: Total length of the reconstructed timeline (optional, inferred if None)
            aggregation: Method to aggregate overlapping values ('mean', 'median', 'max', 'min')
            
        Returns:
            Reconstructed timeline array where each point is the aggregation of all
            overlapping window values at that timestamp
        """
        if not windowed_values or len(windowed_values) == 0:
            raise ValueError("windowed_values cannot be empty")
        
        if len(windowed_values) != len(window_start_indices):
            raise ValueError("windowed_values and window_start_indices must have same length")
        
        # Determine total timeline length
        if total_length is None:
            max_end = 0
            for i, (window, start_idx) in enumerate(zip(windowed_values, window_start_indices)):
                window = np.asarray(window)
                end_idx = start_idx + len(window)
                max_end = max(max_end, end_idx)
            total_length = max_end
        
        # Collect all values at each timestamp
        values_at_timestamp = defaultdict(list)
        
        for window, start_idx in zip(windowed_values, window_start_indices):
            window = np.asarray(window)
            for offset, val in enumerate(window):
                timestamp = start_idx + offset
                if timestamp < total_length:
                    values_at_timestamp[timestamp].append(val)
        
        # Aggregate values at each timestamp
        reconstructed = np.zeros(total_length)
        
        for timestamp in range(total_length):
            if timestamp in values_at_timestamp:
                vals = values_at_timestamp[timestamp]
                if aggregation == 'mean':
                    reconstructed[timestamp] = np.mean(vals)
                elif aggregation == 'median':
                    reconstructed[timestamp] = np.median(vals)
                elif aggregation == 'max':
                    reconstructed[timestamp] = np.max(vals)
                elif aggregation == 'min':
                    reconstructed[timestamp] = np.min(vals)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")
            else:
                # No window covers this timestamp - use NaN
                reconstructed[timestamp] = np.nan
        
        return reconstructed
    
    def _reconstruct_timeline_with_majority_vote(self,
                                                 windowed_values: List[np.ndarray],
                                                 window_start_indices: List[int],
                                                 risk_type: str = 'hypoglycemia',
                                                 total_length: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct timeline using majority vote on binarized window values.
        
        Each window is first binarized (based on thresholds), then overlapping
        binary values at each timestamp are combined using majority vote.
        
        Args:
            windowed_values: List of arrays, each containing predictions for one window
            window_start_indices: List of starting indices for each window
            risk_type: 'hypoglycemia' or 'hyperglycemia' for thresholding
            total_length: Total length of the reconstructed timeline
            
        Returns:
            Binary timeline array where 1 = majority of windows predicted high-risk
        """
        if not windowed_values or len(windowed_values) == 0:
            raise ValueError("windowed_values cannot be empty")
        
        # Determine total timeline length
        if total_length is None:
            max_end = 0
            for window, start_idx in zip(windowed_values, window_start_indices):
                window = np.asarray(window)
                end_idx = start_idx + len(window)
                max_end = max(max_end, end_idx)
            total_length = max_end
        
        # Collect binary votes at each timestamp
        votes_at_timestamp = defaultdict(list)
        
        for window, start_idx in zip(windowed_values, window_start_indices):
            window = np.asarray(window)
            # Binarize this window
            binary_window = self._classify_glucose_level(window, risk_type)
            
            for offset, vote in enumerate(binary_window):
                timestamp = start_idx + offset
                if timestamp < total_length:
                    votes_at_timestamp[timestamp].append(vote)
        
        # Majority vote at each timestamp
        reconstructed = np.zeros(total_length, dtype=int)
        
        for timestamp in range(total_length):
            if timestamp in votes_at_timestamp:
                votes = votes_at_timestamp[timestamp]
                # Majority vote: 1 if more than half voted 1
                reconstructed[timestamp] = 1 if np.mean(votes) > 0.5 else 0
            else:
                reconstructed[timestamp] = 0  # Default to normal if no coverage
        
        return reconstructed
    
    def _process_windowed_data(self,
                               y_values: Union[np.ndarray, List[np.ndarray]],
                               window_start_indices: Optional[List[int]] = None,
                               total_length: Optional[int] = None,
                               calc_mode: str = CALC_MODE_SIMPLE,
                               risk_type: str = 'hypoglycemia') -> Tuple[np.ndarray, np.ndarray]:
        """
        Process input data according to calculation mode.
        
        Args:
            y_values: Either a flat array (simple mode) or list of window arrays
            window_start_indices: Starting indices for each window (required for reconstruction modes)
            total_length: Total timeline length (optional, for reconstruction modes)
            calc_mode: Calculation mode (simple, timeline_reconstruction, window_majority)
            risk_type: Type of risk for binarization
            
        Returns:
            Tuple of (processed_values, binary_values)
            For simple mode: returns original and binarized arrays
            For reconstruction modes: returns reconstructed timeline and its binarization
        """
        if calc_mode == CALC_MODE_SIMPLE:
            # Simple mode: treat input as flat array
            y_array = np.asarray(y_values).flatten()
            y_binary = self._classify_glucose_level(y_array, risk_type)
            return y_array, y_binary
        
        elif calc_mode == CALC_MODE_TIMELINE_RECONSTRUCTION:
            # Reconstruct timeline by averaging overlapping windows, then binarize
            if window_start_indices is None:
                raise ValueError("window_start_indices required for timeline_reconstruction mode")
            
            # Convert to list of arrays if needed
            if isinstance(y_values, np.ndarray) and y_values.ndim == 1:
                # Single flat array - can't reconstruct
                raise ValueError("For timeline_reconstruction mode, y_values must be a list of window arrays")
            
            y_reconstructed = self._reconstruct_timeline(
                windowed_values=y_values,
                window_start_indices=window_start_indices,
                total_length=total_length,
                aggregation='mean'
            )
            # Remove NaN values for metric calculation
            valid_mask = ~np.isnan(y_reconstructed)
            y_reconstructed = y_reconstructed[valid_mask]
            y_binary = self._classify_glucose_level(y_reconstructed, risk_type)
            return y_reconstructed, y_binary
        
        elif calc_mode == CALC_MODE_WINDOW_MAJORITY:
            # Binarize each window, then use majority vote
            if window_start_indices is None:
                raise ValueError("window_start_indices required for window_majority mode")
            
            if isinstance(y_values, np.ndarray) and y_values.ndim == 1:
                raise ValueError("For window_majority mode, y_values must be a list of window arrays")
            
            y_binary = self._reconstruct_timeline_with_majority_vote(
                windowed_values=y_values,
                window_start_indices=window_start_indices,
                risk_type=risk_type,
                total_length=total_length
            )
            # For window_majority, the "continuous" value is synthesized from binary
            # We'll use the binary as the primary output
            y_reconstructed = y_binary.astype(float) * (
                self.hypoglycemia_threshold - 10 if risk_type == 'hypoglycemia' 
                else self.hyperglycemia_threshold + 10
            )
            return y_reconstructed, y_binary
        
        else:
            raise ValueError(f"Unknown calc_mode: {calc_mode}. Must be one of {VALID_CALC_MODES}")
    
    def _classify_glucose_level(self, glucose_values: np.ndarray, 
                                risk_type: str = 'hypoglycemia') -> np.ndarray:
        """
        Convert continuous glucose values to binary classification.
        
        Args:
            glucose_values: Array of glucose values (mg/dL)
            risk_type: 'hypoglycemia' or 'hyperglycemia'
            
        Returns:
            Binary array (1 = high-risk event, 0 = normal)
        """
        if risk_type == 'hypoglycemia':
            # ADA standard: hypoglycemia Level 1 is <= 70 mg/dL
            return (glucose_values <= self.hypoglycemia_threshold).astype(int)
        elif risk_type == 'hyperglycemia':
            # ADA standard: hyperglycemia is >= 180 mg/dL (2h post-meal)
            return (glucose_values >= self.hyperglycemia_threshold).astype(int)
        else:
            raise ValueError("risk_type must be 'hypoglycemia' or 'hyperglycemia'")
    
    def demographic_parity_gap(self,
                              y_pred: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                              group_labels: Union[np.ndarray, Dict[str, np.ndarray]],
                              risk_type: str = 'hypoglycemia',
                              calc_mode: str = CALC_MODE_SIMPLE,
                              window_start_indices: Optional[Dict[str, List[int]]] = None,
                              total_lengths: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Calculate Demographic Parity Gap (DP Gap).
        
        Measures if the model's positive predictions (critical alerts) are independent
        of the protected attribute. Ensures alerts are distributed equally across groups.
        
        Mathematical Definition:
            DP Gap = |P(Ŷ=1 | A=a) - P(Ŷ=1 | A=b)|
        
        Where:
            - Ŷ=1: Predicted positive (critical event)
            - A: Protected attribute (e.g., gender)
            - a, b: Different groups (e.g., male, female)
        
        Calculation Modes:
            - 'simple': Direct calculation on flat arrays (default)
            - 'timeline_reconstruction': Reconstruct timeline by averaging overlapping
              windows, then binarize the averaged values
            - 'window_majority': Binarize each window, use majority vote at each timestamp
        
        Args:
            y_pred: For 'simple' mode: flat array of predictions
                   For other modes: Dict mapping group -> list of window arrays
            group_labels: For 'simple' mode: array of group labels per prediction
                         For other modes: Dict mapping group -> array of group identifiers
            risk_type: Type of critical event to detect ('hypoglycemia' or 'hyperglycemia')
            calc_mode: Calculation mode (simple, timeline_reconstruction, window_majority)
            window_start_indices: For reconstruction modes: Dict mapping group -> list of start indices
            total_lengths: For reconstruction modes: Dict mapping group -> total timeline length
            
        Returns:
            Dictionary containing:
                - dp_gap: Demographic parity gap value
                - group_positive_rates: Positive prediction rate for each group
                - interpretation: Human-readable interpretation
                - calc_mode: The calculation mode used
                - lower_is_better: True (lower values indicate better fairness)
        """
        unique_groups = []
        group_positive_rates = {}
        
        if calc_mode == CALC_MODE_SIMPLE:
            # Simple mode: flat arrays
            y_pred_array = np.asarray(y_pred).flatten()
            group_labels_array = np.asarray(group_labels)
            
            # Convert predictions to binary (critical event or not)
            y_pred_binary = self._classify_glucose_level(y_pred_array, risk_type)
            
            unique_groups = np.unique(group_labels_array)
            
            # Calculate positive prediction rate for each group
            for group in unique_groups:
                group_mask = group_labels_array == group
                group_predictions = y_pred_binary[group_mask]
                
                if len(group_predictions) == 0:
                    warnings.warn(f"No samples for group {group}")
                    continue
                
                # P(Ŷ=1 | A=group)
                positive_rate = np.mean(group_predictions)
                group_positive_rates[str(group)] = positive_rate
        
        else:
            # Reconstruction modes: process each group's windowed data
            if not isinstance(y_pred, dict):
                # Fallback to simple mode if data isn't in windowed format
                warnings.warn(f"{calc_mode} mode requested but data is flat. Falling back to simple mode.")
                return self.demographic_parity_gap(
                    y_pred=y_pred, group_labels=group_labels, risk_type=risk_type,
                    calc_mode=CALC_MODE_SIMPLE
                )
            
            for group, group_windows in y_pred.items():
                group_start_indices = window_start_indices.get(group) if window_start_indices else None
                group_total_length = total_lengths.get(group) if total_lengths else None
                
                try:
                    _, y_binary = self._process_windowed_data(
                        y_values=group_windows,
                        window_start_indices=group_start_indices,
                        total_length=group_total_length,
                        calc_mode=calc_mode,
                        risk_type=risk_type
                    )
                    
                    positive_rate = np.mean(y_binary)
                    group_positive_rates[str(group)] = positive_rate
                    unique_groups.append(group)
                    
                except Exception as e:
                    warnings.warn(f"Error processing group {group}: {e}")
                    continue
        
        # Calculate DP Gap as maximum difference between any two groups
        if len(group_positive_rates) < 2:
            return {
                'dp_gap': 0.0,
                'group_positive_rates': group_positive_rates,
                'calc_mode': calc_mode,
                'interpretation': 'Insufficient groups for comparison',
                'lower_is_better': True
            }
        
        rates = list(group_positive_rates.values())
        dp_gap = max(rates) - min(rates)
        
        # Interpretation
        if dp_gap < 0.05:
            interpretation = f"EXCELLENT: Alerts are distributed fairly (gap={dp_gap:.4f})"
        elif dp_gap < 0.10:
            interpretation = f"GOOD: Minor disparity in alert rates (gap={dp_gap:.4f})"
        elif dp_gap < 0.20:
            interpretation = f"CONCERNING: Moderate disparity in alert rates (gap={dp_gap:.4f})"
        else:
            interpretation = f"POOR: Significant disparity in alert rates (gap={dp_gap:.4f})"
        
        return {
            'dp_gap': dp_gap,
            'group_positive_rates': group_positive_rates,
            'risk_type': risk_type,
            'calc_mode': calc_mode,
            'interpretation': interpretation,
            'lower_is_better': True
        }
    
    def equal_opportunity_gap(self,
                             y_true: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                             y_pred: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                             group_labels: Union[np.ndarray, Dict[str, np.ndarray]],
                             risk_type: str = 'hypoglycemia',
                             calc_mode: str = CALC_MODE_SIMPLE,
                             window_start_indices: Optional[Dict[str, List[int]]] = None,
                             total_lengths: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Calculate Equal Opportunity Gap (EO Gap).
        
        Focuses on equality of True Positive Rates across groups. Ensures the model
        is equally effective at detecting actual high-risk events for all groups.
        This is CRITICAL for health applications where missing a true event can be
        life-threatening.
        
        Mathematical Definition:
            EO Gap = |P(Ŷ=1 | Y=1, A=a) - P(Ŷ=1 | Y=1, A=b)|
                   = |TPR_a - TPR_b|
        
        Where:
            - TPR: True Positive Rate (Sensitivity/Recall)
            - Y=1: Actual positive (true critical event)
            - Ŷ=1: Predicted positive
            - A: Protected attribute
        
        Calculation Modes:
            - 'simple': Direct calculation on flat arrays (default)
            - 'timeline_reconstruction': Reconstruct timeline by averaging overlapping
              windows, then binarize the averaged values
            - 'window_majority': Binarize each window, use majority vote at each timestamp
        
        Args:
            y_true: For 'simple' mode: flat array of true values
                   For other modes: Dict mapping group -> list of window arrays
            y_pred: For 'simple' mode: flat array of predictions
                   For other modes: Dict mapping group -> list of window arrays
            group_labels: For 'simple' mode: array of group labels per sample
                         For other modes: Dict mapping group -> array of identifiers
            risk_type: Type of critical event to detect
            calc_mode: Calculation mode (simple, timeline_reconstruction, window_majority)
            window_start_indices: For reconstruction modes: Dict mapping group -> list of start indices
            total_lengths: For reconstruction modes: Dict mapping group -> total timeline length
            
        Returns:
            Dictionary containing:
                - eo_gap: Equal opportunity gap value
                - group_tpr: TPR for each group
                - group_confusion_matrices: Detailed performance per group
                - calc_mode: The calculation mode used
                - interpretation: Human-readable interpretation
                - lower_is_better: True
        """
        unique_groups = []
        group_tpr = {}
        group_confusion = {}
        
        if calc_mode == CALC_MODE_SIMPLE:
            # Simple mode: flat arrays
            y_true_array = np.asarray(y_true).flatten()
            y_pred_array = np.asarray(y_pred).flatten()
            group_labels_array = np.asarray(group_labels)
            
            # Convert to binary classification
            y_true_binary = self._classify_glucose_level(y_true_array, risk_type)
            y_pred_binary = self._classify_glucose_level(y_pred_array, risk_type)
            
            unique_groups = np.unique(group_labels_array)
            
            # Calculate TPR for each group
            for group in unique_groups:
                group_mask = group_labels_array == group
                group_y_true = y_true_binary[group_mask]
                group_y_pred = y_pred_binary[group_mask]
                
                if len(group_y_true) == 0:
                    warnings.warn(f"No samples for group {group}")
                    continue
                
                # Calculate confusion matrix (labels=[0,1] guarantees a 2x2 matrix)
                cm = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                # Calculate TPR = TP / (TP + FN)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                group_tpr[str(group)] = tpr
                group_confusion[str(group)] = {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'tpr': tpr,
                    'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                    'total_actual_positives': int(tp + fn),
                    'total_actual_negatives': int(fp + tn)
                }
        
        else:
            # Reconstruction modes: process each group's windowed data
            if not isinstance(y_pred, dict) or not isinstance(y_true, dict):
                # Fallback to simple mode if data isn't in windowed format
                warnings.warn(f"{calc_mode} mode requested but data is flat. Falling back to simple mode.")
                return self.equal_opportunity_gap(
                    y_true=y_true, y_pred=y_pred, group_labels=group_labels, 
                    risk_type=risk_type, calc_mode=CALC_MODE_SIMPLE
                )
            
            for group in y_pred.keys():
                if group not in y_true:
                    warnings.warn(f"Group {group} in y_pred but not in y_true")
                    continue
                
                group_pred_windows = y_pred[group]
                group_true_windows = y_true[group]
                group_start_indices = window_start_indices.get(group) if window_start_indices else None
                group_total_length = total_lengths.get(group) if total_lengths else None
                
                try:
                    # Process predictions
                    _, y_pred_binary = self._process_windowed_data(
                        y_values=group_pred_windows,
                        window_start_indices=group_start_indices,
                        total_length=group_total_length,
                        calc_mode=calc_mode,
                        risk_type=risk_type
                    )
                    
                    # Process ground truth
                    _, y_true_binary = self._process_windowed_data(
                        y_values=group_true_windows,
                        window_start_indices=group_start_indices,
                        total_length=group_total_length,
                        calc_mode=calc_mode,
                        risk_type=risk_type
                    )
                    
                    # Ensure same length
                    min_len = min(len(y_pred_binary), len(y_true_binary))
                    y_pred_binary = y_pred_binary[:min_len]
                    y_true_binary = y_true_binary[:min_len]
                    
                    # Calculate confusion matrix (labels=[0,1] guarantees a 2x2 matrix)
                    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Calculate TPR
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    group_tpr[str(group)] = tpr
                    group_confusion[str(group)] = {
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'true_negatives': int(tn),
                        'false_negatives': int(fn),
                        'tpr': tpr,
                        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                        'total_actual_positives': int(tp + fn),
                        'total_actual_negatives': int(fp + tn)
                    }
                    unique_groups.append(group)
                    
                except Exception as e:
                    warnings.warn(f"Error processing group {group}: {e}")
                    continue
        
        # Calculate EO Gap
        if len(group_tpr) < 2:
            return {
                'eo_gap': 0.0,
                'group_tpr': group_tpr,
                'group_confusion_matrices': group_confusion,
                'interpretation': 'Insufficient groups for comparison',
                'lower_is_better': True
            }
        
        tpr_values = list(group_tpr.values())
        eo_gap = max(tpr_values) - min(tpr_values)
        
        # Interpretation - this is CRITICAL for patient safety
        if eo_gap < 0.05:
            interpretation = f"EXCELLENT: Model detects critical events equally well across groups (gap={eo_gap:.4f})"
        elif eo_gap < 0.10:
            interpretation = f"GOOD: Minor difference in detection rates (gap={eo_gap:.4f})"
        elif eo_gap < 0.20:
            interpretation = f"CONCERNING: Some groups may miss critical alerts (gap={eo_gap:.4f})"
        else:
            interpretation = f"CRITICAL SAFETY ISSUE: Significant detection disparity (gap={eo_gap:.4f})"
        
        return {
            'eo_gap': eo_gap,
            'group_tpr': group_tpr,
            'group_confusion_matrices': group_confusion,
            'risk_type': risk_type,
            'calc_mode': calc_mode,
            'interpretation': interpretation,
            'lower_is_better': True
        }
    
    def fairness_violation_objective(self,
                                    y_true: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                                    y_pred: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                                    group_labels: Union[np.ndarray, Dict[str, np.ndarray]],
                                    metric_type: str = 'classification',
                                    calc_mode: str = CALC_MODE_SIMPLE,
                                    risk_type: str = 'hypoglycemia',
                                    window_start_indices: Optional[Dict[str, List[int]]] = None,
                                    total_lengths: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Calculate Fairness Violation Objective (FVO).
        
        Quantifies the maximum disparity in model performance (accuracy) between any
        two demographic subgroups. This is a global metric that ensures general model
        reliability is equitable across all groups.
        
        Mathematical Definition:
            FVO = max|Acc_i - Acc_j| for all pairs (i,j) of groups
        
        Where:
            - Acc_i: Accuracy for group i
            - Can be calculated for classification or regression tasks
        
        Calculation Modes:
            - 'simple': Direct calculation on flat arrays (default)
            - 'timeline_reconstruction': Reconstruct timeline by averaging overlapping
              windows, then calculate accuracy on the reconstructed timeline
            - 'window_majority': Binarize each window, use majority vote at each timestamp
        
        Args:
            y_true: For 'simple' mode: flat array of true values
                   For other modes: Dict mapping group -> list of window arrays
            y_pred: For 'simple' mode: flat array of predictions
                   For other modes: Dict mapping group -> list of window arrays
            group_labels: For 'simple' mode: array of group labels
                         For other modes: Dict mapping group -> array of identifiers
            metric_type: 'classification' (uses accuracy on binary events) or
                        'regression' (uses 1 - normalized RMSE as accuracy proxy)
            calc_mode: Calculation mode (simple, timeline_reconstruction, window_majority)
            risk_type: Type of risk for binarization (used in classification mode)
            window_start_indices: For reconstruction modes: Dict mapping group -> list of start indices
            total_lengths: For reconstruction modes: Dict mapping group -> total timeline length
            
        Returns:
            Dictionary containing:
                - fvo: Fairness violation objective value
                - group_accuracies: Accuracy/performance for each group
                - max_disparity_pair: Groups with maximum disparity
                - calc_mode: The calculation mode used
                - interpretation: Human-readable interpretation
                - lower_is_better: True
        """
        unique_groups = []
        group_accuracies = {}
        
        if calc_mode == CALC_MODE_SIMPLE:
            # Simple mode: flat arrays
            y_true_array = np.asarray(y_true).flatten()
            y_pred_array = np.asarray(y_pred).flatten()
            group_labels_array = np.asarray(group_labels)
            
            # Pre-compute global range for regression proxy (must be consistent across groups)
            global_glucose_range = np.ptp(y_true_array)
            
            unique_groups = np.unique(group_labels_array)
            
            # Calculate accuracy for each group
            for group in unique_groups:
                group_mask = group_labels_array == group
                group_y_true = y_true_array[group_mask]
                group_y_pred = y_pred_array[group_mask]
                
                if len(group_y_true) == 0:
                    warnings.warn(f"No samples for group {group}")
                    continue
                
                if metric_type == 'classification':
                    # For classification: use binary accuracy
                    y_true_binary = self._classify_glucose_level(group_y_true, risk_type)
                    y_pred_binary = self._classify_glucose_level(group_y_pred, risk_type)
                    accuracy = accuracy_score(y_true_binary, y_pred_binary)
                
                elif metric_type == 'regression':
                    # Use global range so the proxy is comparable across groups.
                    # Per-group range would give different scales per group, making
                    # cross-group accuracy comparison meaningless.
                    rmse = np.sqrt(np.mean((group_y_true - group_y_pred) ** 2))
                    if global_glucose_range > 0:
                        accuracy = max(0.0, 1.0 - (rmse / global_glucose_range))
                    else:
                        accuracy = 1.0
                
                else:
                    raise ValueError("metric_type must be 'classification' or 'regression'")
                
                group_accuracies[str(group)] = accuracy
        
        else:
            # Reconstruction modes: process each group's windowed data
            if not isinstance(y_pred, dict) or not isinstance(y_true, dict):
                # Fallback to simple mode if data isn't in windowed format
                warnings.warn(f"{calc_mode} mode requested but data is flat. Falling back to simple mode.")
                return self.fairness_violation_objective(
                    y_true=y_true, y_pred=y_pred, group_labels=group_labels, 
                    risk_type=risk_type, metric_type=metric_type, calc_mode=CALC_MODE_SIMPLE
                )
            
            # Pre-compute global range across all groups for regression proxy
            global_glucose_range = 0.0
            if metric_type == 'regression':
                all_true_values = []
                for group_windows in y_true.values():
                    for w in group_windows:
                        all_true_values.extend(np.asarray(w).flatten().tolist())
                if all_true_values:
                    global_glucose_range = np.ptp(all_true_values)
            
            for group in y_pred.keys():
                if group not in y_true:
                    warnings.warn(f"Group {group} in y_pred but not in y_true")
                    continue
                
                group_pred_windows = y_pred[group]
                group_true_windows = y_true[group]
                group_start_indices = window_start_indices.get(group) if window_start_indices else None
                group_total_length = total_lengths.get(group) if total_lengths else None
                
                try:
                    # Process predictions and ground truth
                    y_pred_processed, y_pred_binary = self._process_windowed_data(
                        y_values=group_pred_windows,
                        window_start_indices=group_start_indices,
                        total_length=group_total_length,
                        calc_mode=calc_mode,
                        risk_type=risk_type
                    )
                    
                    y_true_processed, y_true_binary = self._process_windowed_data(
                        y_values=group_true_windows,
                        window_start_indices=group_start_indices,
                        total_length=group_total_length,
                        calc_mode=calc_mode,
                        risk_type=risk_type
                    )
                    
                    # Ensure same length
                    min_len = min(len(y_pred_processed), len(y_true_processed))
                    y_pred_processed = y_pred_processed[:min_len]
                    y_true_processed = y_true_processed[:min_len]
                    y_pred_binary = y_pred_binary[:min_len]
                    y_true_binary = y_true_binary[:min_len]
                    
                    if metric_type == 'classification':
                        accuracy = accuracy_score(y_true_binary, y_pred_binary)
                    
                    elif metric_type == 'regression':
                        rmse = np.sqrt(np.mean((y_true_processed - y_pred_processed) ** 2))
                        if global_glucose_range > 0:
                            accuracy = max(0.0, 1.0 - (rmse / global_glucose_range))
                        else:
                            accuracy = 1.0
                    
                    else:
                        raise ValueError("metric_type must be 'classification' or 'regression'")
                    
                    group_accuracies[str(group)] = accuracy
                    unique_groups.append(group)
                    
                except Exception as e:
                    warnings.warn(f"Error processing group {group}: {e}")
                    continue
        
        # Calculate FVO as maximum accuracy difference
        if len(group_accuracies) < 2:
            return {
                'fvo': 0.0,
                'group_accuracies': group_accuracies,
                'max_disparity_pair': None,
                'calc_mode': calc_mode,
                'interpretation': 'Insufficient groups for comparison',
                'lower_is_better': True
            }
        
        # Find maximum disparity between any two groups
        max_fvo = 0.0
        max_pair = None
        
        groups = list(group_accuracies.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                disparity = abs(group_accuracies[groups[i]] - group_accuracies[groups[j]])
                if disparity > max_fvo:
                    max_fvo = disparity
                    max_pair = (groups[i], groups[j])
        
        # Interpretation
        if max_fvo < 0.02:
            interpretation = f"EXCELLENT: Model performs equally well across all groups (FVO={max_fvo:.4f})"
        elif max_fvo < 0.05:
            interpretation = f"GOOD: Minor performance variation (FVO={max_fvo:.4f})"
        elif max_fvo < 0.10:
            interpretation = f"CONCERNING: Notable performance disparity (FVO={max_fvo:.4f})"
        else:
            interpretation = f"POOR: Significant reliability concerns for some groups (FVO={max_fvo:.4f})"
        
        return {
            'fvo': max_fvo,
            'group_accuracies': group_accuracies,
            'max_disparity_pair': max_pair,
            'metric_type': metric_type,
            'calc_mode': calc_mode,
            'interpretation': interpretation,
            'lower_is_better': True
        }
    
    def calculate_all_advanced_metrics(self,
                                       y_true: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                                       y_pred: Union[np.ndarray, Dict[str, List[np.ndarray]]],
                                       group_labels: Union[np.ndarray, Dict[str, np.ndarray]],
                                       group_attribute: str = 'demographic_group',
                                       risk_type: str = 'hypoglycemia',
                                       calc_mode: str = CALC_MODE_SIMPLE,
                                       window_start_indices: Optional[Dict[str, List[int]]] = None,
                                       total_lengths: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Calculate all three advanced fairness metrics in one call.
        
        Calculation Modes:
            - 'simple': Direct calculation on flat arrays (default)
            - 'timeline_reconstruction': Reconstruct timeline by averaging overlapping
              windows at each timestamp, then binarize and calculate metrics
            - 'window_majority': Binarize each window, use majority vote at each timestamp
        
        Args:
            y_true: For 'simple' mode: flat array of true values
                   For other modes: Dict mapping group -> list of window arrays
            y_pred: For 'simple' mode: flat array of predictions
                   For other modes: Dict mapping group -> list of window arrays  
            group_labels: For 'simple' mode: array of group labels
                         For other modes: Dict mapping group -> array of identifiers
            group_attribute: Name of the demographic attribute
            risk_type: Type of critical event ('hypoglycemia' or 'hyperglycemia')
            calc_mode: Calculation mode (simple, timeline_reconstruction, window_majority)
            window_start_indices: For reconstruction modes: Dict mapping group -> list of start indices
            total_lengths: For reconstruction modes: Dict mapping group -> total timeline length
            
        Returns:
            Comprehensive dictionary with all metrics and interpretations
        """
        # Determine sample size based on mode
        if calc_mode == CALC_MODE_SIMPLE:
            sample_size = len(np.asarray(y_true).flatten())
            unique_groups_list = list(np.unique(group_labels))
        else:
            sample_size = sum(sum(len(w) for w in windows) for windows in y_true.values())
            unique_groups_list = list(y_true.keys())
        
        results = {
            'group_attribute': group_attribute,
            'risk_type': risk_type,
            'calc_mode': calc_mode,
            'thresholds': {
                'hypoglycemia': self.hypoglycemia_threshold,
                'hyperglycemia': self.hyperglycemia_threshold
            },
            'sample_size': sample_size,
            'unique_groups': unique_groups_list,
            'metrics': {}
        }
        
        # 1. Demographic Parity Gap
        dp_result = self.demographic_parity_gap(
            y_pred, group_labels, risk_type, 
            calc_mode=calc_mode,
            window_start_indices=window_start_indices,
            total_lengths=total_lengths
        )
        results['metrics']['demographic_parity_gap'] = dp_result
        
        # 2. Equal Opportunity Gap
        eo_result = self.equal_opportunity_gap(
            y_true, y_pred, group_labels, risk_type,
            calc_mode=calc_mode,
            window_start_indices=window_start_indices,
            total_lengths=total_lengths
        )
        results['metrics']['equal_opportunity_gap'] = eo_result
        
        # 3. Fairness Violation Objective (both classification and regression)
        fvo_class = self.fairness_violation_objective(
            y_true, y_pred, group_labels, 'classification',
            calc_mode=calc_mode,
            risk_type=risk_type,
            window_start_indices=window_start_indices,
            total_lengths=total_lengths
        )
        fvo_reg = self.fairness_violation_objective(
            y_true, y_pred, group_labels, 'regression',
            calc_mode=calc_mode,
            risk_type=risk_type,
            window_start_indices=window_start_indices,
            total_lengths=total_lengths
        )
        
        results['metrics']['fairness_violation_objective'] = {
            'classification_based': fvo_class,
            'regression_based': fvo_reg
        }
        
        # Overall fairness assessment
        results['overall_assessment'] = self._assess_overall_fairness(
            dp_result, eo_result, fvo_class, fvo_reg
        )
        
        return results
    
    def _assess_overall_fairness(self, dp_result, eo_result, fvo_class, fvo_reg) -> Dict[str, Any]:
        """Generate overall fairness assessment."""
        dp_gap = dp_result['dp_gap']
        eo_gap = eo_result['eo_gap']
        fvo_gap = fvo_class['fvo']
        
        # Count issues
        issues = []
        if dp_gap > 0.10:
            issues.append(f"High DP Gap ({dp_gap:.4f}): Unequal alert distribution")
        if eo_gap > 0.10:
            issues.append(f"High EO Gap ({eo_gap:.4f}): Unequal detection of critical events")
        if fvo_gap > 0.05:
            issues.append(f"High FVO ({fvo_gap:.4f}): Unequal overall performance")
        
        if not issues:
            status = "✅ FAIR"
            summary = "Model demonstrates good fairness across all demographic groups"
        elif len(issues) == 1:
            status = "⚠️ MINOR CONCERNS"
            summary = "Model shows minor fairness issues that should be monitored"
        elif len(issues) == 2:
            status = "⚠️ MODERATE CONCERNS"
            summary = "Model shows moderate fairness issues that should be addressed"
        else:
            status = "❌ SIGNIFICANT CONCERNS"
            summary = "Model shows significant fairness issues requiring immediate attention"
        
        return {
            'status': status,
            'summary': summary,
            'issues': issues,
            'metrics_summary': {
                'dp_gap': dp_gap,
                'eo_gap': eo_gap,
                'fvo': fvo_gap
            }
        }
    
    def print_comprehensive_report(self, results: Dict):
        """Print a formatted comprehensive fairness report."""
        print("\n" + "=" * 80)
        print("ADVANCED FAIRNESS METRICS REPORT - BLOOD GLUCOSE PREDICTION")
        print("=" * 80)
        
        print(f"\nDataset Information:")
        print(f"  Group Attribute: {results['group_attribute']}")
        print(f"  Risk Type Analyzed: {results['risk_type']}")
        print(f"  Sample Size: {results['sample_size']}")
        print(f"  Groups: {', '.join(results['unique_groups'])}")
        print(f"  Hypoglycemia Threshold: {results['thresholds']['hypoglycemia']} mg/dL")
        print(f"  Hyperglycemia Threshold: {results['thresholds']['hyperglycemia']} mg/dL")
        
        # Metric 1: Demographic Parity Gap
        print(f"\n{'─' * 80}")
        print("1. DEMOGRAPHIC PARITY GAP (DP Gap)")
        print(f"{'─' * 80}")
        dp = results['metrics']['demographic_parity_gap']
        print(f"  Value: {dp['dp_gap']:.4f}")
        print(f"  Interpretation: {dp['interpretation']}")
        print(f"\n  Group-wise Positive Prediction Rates:")
        for group, rate in dp['group_positive_rates'].items():
            print(f"    {group}: {rate:.4f} ({rate*100:.2f}% predicted as high-risk)")
        
        # Metric 2: Equal Opportunity Gap
        print(f"\n{'─' * 80}")
        print("2. EQUAL OPPORTUNITY GAP (EO Gap)")
        print(f"{'─' * 80}")
        eo = results['metrics']['equal_opportunity_gap']
        print(f"  Value: {eo['eo_gap']:.4f}")
        print(f"  Interpretation: {eo['interpretation']}")
        print(f"\n  Group-wise True Positive Rates (Detection Rates):")
        for group, tpr in eo['group_tpr'].items():
            confusion = eo['group_confusion_matrices'][group]
            print(f"    {group}: {tpr:.4f} ({tpr*100:.2f}% of actual events detected)")
            print(f"      True Positives: {confusion['true_positives']}, "
                  f"False Negatives: {confusion['false_negatives']} "
                  f"(Total Actual Positives: {confusion['total_actual_positives']})")
        
        # Metric 3: Fairness Violation Objective
        print(f"\n{'─' * 80}")
        print("3. FAIRNESS VIOLATION OBJECTIVE (FVO)")
        print(f"{'─' * 80}")
        fvo_class = results['metrics']['fairness_violation_objective']['classification_based']
        fvo_reg = results['metrics']['fairness_violation_objective']['regression_based']
        
        print(f"  Classification-based FVO: {fvo_class['fvo']:.4f}")
        print(f"  Interpretation: {fvo_class['interpretation']}")
        if fvo_class['max_disparity_pair']:
            print(f"  Max Disparity Between: {fvo_class['max_disparity_pair'][0]} and {fvo_class['max_disparity_pair'][1]}")
        
        print(f"\n  Group-wise Accuracies (Classification):")
        for group, acc in fvo_class['group_accuracies'].items():
            print(f"    {group}: {acc:.4f} ({acc*100:.2f}%)")
        
        print(f"\n  Regression-based FVO: {fvo_reg['fvo']:.4f}")
        print(f"  Group-wise Performance (Regression):")
        for group, acc in fvo_reg['group_accuracies'].items():
            print(f"    {group}: {acc:.4f}")
        
        # Overall Assessment
        print(f"\n{'=' * 80}")
        print("OVERALL FAIRNESS ASSESSMENT")
        print(f"{'=' * 80}")
        assessment = results['overall_assessment']
        print(f"  Status: {assessment['status']}")
        print(f"  Summary: {assessment['summary']}")
        
        if assessment['issues']:
            print(f"\n  Identified Issues:")
            for issue in assessment['issues']:
                print(f"    • {issue}")
        else:
            print(f"\n  ✅ No significant fairness issues detected")
        
        print(f"\n  Metrics Summary:")
        print(f"    DP Gap:  {assessment['metrics_summary']['dp_gap']:.4f} (lower is better)")
        print(f"    EO Gap:  {assessment['metrics_summary']['eo_gap']:.4f} (lower is better)")
        print(f"    FVO:     {assessment['metrics_summary']['fvo']:.4f} (lower is better)")
        
        print("\n" + "=" * 80 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Advanced Fairness Metrics - Example Usage")
    print("=" * 80)
    
    # Generate sample blood glucose data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate demographics
    gender = np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4])
    
    # True glucose values (mg/dL) - normal range with some hypoglycemic events
    y_true = np.random.normal(120, 30, n_samples)
    y_true = np.clip(y_true, 40, 300)  # Realistic glucose range
    
    # Add some actual hypoglycemic events
    hypo_mask = np.random.random(n_samples) < 0.15
    y_true[hypo_mask] = np.random.uniform(40, 70, np.sum(hypo_mask))
    
    # Predictions with some bias
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    
    # Introduce fairness issue: model is less sensitive for females in hypoglycemia range
    female_mask = gender == 'female'
    female_hypo_mask = female_mask & (y_true < 70)
    y_pred[female_hypo_mask] += np.random.uniform(10, 20, np.sum(female_hypo_mask))
    
    y_pred = np.clip(y_pred, 40, 300)
    
    # Initialize advanced fairness metrics
    print("\nInitializing Advanced Fairness Metrics Calculator...")
    afm = AdvancedFairnessMetrics(
        hypoglycemia_threshold=70.0,
        hyperglycemia_threshold=180.0
    )
    
    # ============================================
    # Example 1: SIMPLE MODE (default)
    # ============================================
    print("\n" + "="*80)
    print("EXAMPLE 1: SIMPLE MODE (Direct calculation on flat arrays)")
    print("="*80)
    
    results_simple = afm.calculate_all_advanced_metrics(
        y_true=y_true,
        y_pred=y_pred,
        group_labels=gender,
        group_attribute='Gender',
        risk_type='hypoglycemia',
        calc_mode=CALC_MODE_SIMPLE
    )
    afm.print_comprehensive_report(results_simple)
    
    # ============================================
    # Example 2: TIMELINE RECONSTRUCTION MODE
    # ============================================
    print("\n" + "="*80)
    print("EXAMPLE 2: TIMELINE RECONSTRUCTION MODE")
    print("  - Simulates overlapping windows (window_size=24, stride=6)")
    print("  - Reconstructs timeline by averaging overlapping predictions")
    print("  - Binarizes the averaged timeline, then calculates metrics")
    print("="*80)
    
    # Simulate overlapping windows for each group
    window_size = 24
    stride = 6
    
    # Organize data by group
    y_true_by_group = {}
    y_pred_by_group = {}
    window_starts_by_group = {}
    total_lengths_by_group = {}
    
    for group in ['male', 'female']:
        group_mask = gender == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        # Create overlapping windows
        n_points = len(group_y_true)
        windows_true = []
        windows_pred = []
        start_indices = []
        
        for start in range(0, n_points - window_size + 1, stride):
            windows_true.append(group_y_true[start:start + window_size])
            windows_pred.append(group_y_pred[start:start + window_size])
            start_indices.append(start)
        
        y_true_by_group[group] = windows_true
        y_pred_by_group[group] = windows_pred
        window_starts_by_group[group] = start_indices
        total_lengths_by_group[group] = n_points
    
    results_timeline = afm.calculate_all_advanced_metrics(
        y_true=y_true_by_group,
        y_pred=y_pred_by_group,
        group_labels={g: np.array([g]) for g in ['male', 'female']},
        group_attribute='Gender',
        risk_type='hypoglycemia',
        calc_mode=CALC_MODE_TIMELINE_RECONSTRUCTION,
        window_start_indices=window_starts_by_group,
        total_lengths=total_lengths_by_group
    )
    afm.print_comprehensive_report(results_timeline)
    
    # ============================================
    # Example 3: WINDOW MAJORITY VOTE MODE
    # ============================================
    print("\n" + "="*80)
    print("EXAMPLE 3: WINDOW MAJORITY VOTE MODE")
    print("  - Binarizes each window independently")
    print("  - Uses majority vote across overlapping windows at each timestamp")
    print("  - Calculates metrics on the voted timeline")
    print("="*80)
    
    results_majority = afm.calculate_all_advanced_metrics(
        y_true=y_true_by_group,
        y_pred=y_pred_by_group,
        group_labels={g: np.array([g]) for g in ['male', 'female']},
        group_attribute='Gender',
        risk_type='hypoglycemia',
        calc_mode=CALC_MODE_WINDOW_MAJORITY,
        window_start_indices=window_starts_by_group,
        total_lengths=total_lengths_by_group
    )
    afm.print_comprehensive_report(results_majority)
    
    # ============================================
    # Summary: Compare modes
    # ============================================
    print("\n" + "="*80)
    print("SUMMARY: COMPARISON OF CALCULATION MODES")
    print("="*80)
    print(f"\n{'Mode':<30} {'DP Gap':>10} {'EO Gap':>10} {'FVO':>10}")
    print("-" * 62)
    
    for mode_name, results in [
        ('Simple (default)', results_simple),
        ('Timeline Reconstruction', results_timeline),
        ('Window Majority Vote', results_majority)
    ]:
        dp = results['metrics']['demographic_parity_gap']['dp_gap']
        eo = results['metrics']['equal_opportunity_gap']['eo_gap']
        fvo = results['metrics']['fairness_violation_objective']['classification_based']['fvo']
        print(f"{mode_name:<30} {dp:>10.4f} {eo:>10.4f} {fvo:>10.4f}")
    
    print("\n📝 Notes:")
    print("   - Different modes may produce different metric values")
    print("   - Timeline Reconstruction: Good for continuous monitoring analysis")
    print("   - Window Majority Vote: More robust to noise, conservative alerts")
    print("   - Choose based on your clinical use case and data characteristics")
