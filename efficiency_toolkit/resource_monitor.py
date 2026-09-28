"""
Resource Monitoring Module for LLM Efficiency Analysis

This module extracts hardware usage, power consumption, and resource utilization
metrics from experiment logs and loss files to populate efficiency tables.
"""

import pickle
import glob
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime


class ResourceMonitor:
    """Monitor and extract resource usage metrics from experiment data"""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize ResourceMonitor
        
        Args:
            base_path: Base directory path for experiments
        """
        self.base_path = Path(base_path)
        
    def extract_training_resources(self) -> pd.DataFrame:
        """
        Extract resource usage during training from logs and loss files
        
        Returns:
            DataFrame with training resource metrics
        """
        training_resources = []
        
        # Find training experiments
        training_dirs = glob.glob(str(self.base_path / "efficiency_experiments" / "experiments" / "*training*" / "*"))
        
        for train_dir in training_dirs:
            train_path = Path(train_dir)
            
            # Extract experiment details from directory name
            exp_info = self._parse_experiment_path(train_path)
            if not exp_info:
                continue
                
            # Find patient directories
            patient_dirs = list(train_path.glob("patient_*"))
            
            for patient_dir in patient_dirs:
                patient_id = patient_dir.name.replace("patient_", "")
                
                # Get resource metrics from logs and loss files
                resource_metrics = self._extract_experiment_resources(patient_dir, exp_info, patient_id, "training")
                if resource_metrics:
                    training_resources.append(resource_metrics)
        
        return pd.DataFrame(training_resources) if training_resources else pd.DataFrame()
    
    def extract_inference_resources(self) -> pd.DataFrame:
        """
        Extract resource usage during inference
        
        Returns:
            DataFrame with inference resource metrics
        """
        inference_resources = []
        
        # Find inference experiments
        inference_dirs = glob.glob(str(self.base_path / "efficiency_experiments" / "experiments" / "*inference*" / "*"))
        
        for inf_dir in inference_dirs:
            inf_path = Path(inf_dir)
            
            # Extract experiment details
            exp_info = self._parse_experiment_path(inf_path)
            if not exp_info:
                continue
                
            # Find patient directories
            patient_dirs = list(inf_path.glob("patient_*"))
            
            for patient_dir in patient_dirs:
                patient_id = patient_dir.name.replace("patient_", "")
                
                # Get resource metrics
                resource_metrics = self._extract_experiment_resources(patient_dir, exp_info, patient_id, "inference")
                if resource_metrics:
                    inference_resources.append(resource_metrics)
        
        return pd.DataFrame(inference_resources) if inference_resources else pd.DataFrame()
    
    def extract_distillation_resources(self) -> pd.DataFrame:
        """
        Extract resource usage during distillation training/inference
        
        Returns:
            DataFrame with distillation resource metrics
        """
        distillation_resources = []
        
        # Find distillation experiments
        distill_dirs = glob.glob(str(self.base_path / "efficiency_experiments" / "experiments" / "*distillation*" / "*"))
        distill_dirs += glob.glob(str(self.base_path / "distillation_experiments" / "*"))
        
        for dist_dir in distill_dirs:
            dist_path = Path(dist_dir)
            
            # Extract experiment details
            exp_info = self._parse_distillation_path(dist_path)
            if not exp_info:
                continue
                
            # Get resource metrics
            resource_metrics = self._extract_distillation_resources(dist_path, exp_info)
            if resource_metrics:
                distillation_resources.extend(resource_metrics)
        
        return pd.DataFrame(distillation_resources) if distillation_resources else pd.DataFrame()
    
    def _parse_experiment_path(self, exp_path: Path) -> Optional[Dict]:
        """Parse experiment directory name to extract model and config info"""
        dir_name = exp_path.name
        
        # Extract model info using regex
        model_match = re.search(r'model_(\w+)', dir_name)
        dim_match = re.search(r'dim_(\d+)', dir_name)
        seq_match = re.search(r'seq_(\d+)', dir_name)
        context_match = re.search(r'context_(\d+)', dir_name)
        pred_match = re.search(r'pred_(\d+)', dir_name)
        epochs_match = re.search(r'epochs_(\d+)', dir_name)
        
        if not model_match:
            return None
            
        return {
            'model_name': model_match.group(1),
            'model_dim': int(dim_match.group(1)) if dim_match else None,
            'sequence_length': int(seq_match.group(1)) if seq_match else None,
            'context_length': int(context_match.group(1)) if context_match else None,
            'prediction_length': int(pred_match.group(1)) if pred_match else None,
            'epochs': int(epochs_match.group(1)) if epochs_match else None,
            'experiment_dir': str(exp_path)
        }
    
    def _parse_distillation_path(self, dist_path: Path) -> Optional[Dict]:
        """Parse distillation experiment path"""
        # Handle different distillation directory structures
        if "distillation_experiments" in str(dist_path):
            # Extract from distillation_experiments directory
            parts = dist_path.parts
            if len(parts) >= 2:
                exp_name = parts[-1]
                return {
                    'experiment_type': 'distillation',
                    'experiment_name': exp_name,
                    'experiment_dir': str(dist_path)
                }
        else:
            # Regular experiment directory with distillation
            return self._parse_experiment_path(dist_path)
        
        return None
    
    def _extract_experiment_resources(self, patient_dir: Path, exp_info: Dict, patient_id: str, mode: str) -> Optional[Dict]:
        """Extract resource metrics from a single experiment"""
        
        # Find log directories
        log_dirs = list(patient_dir.glob("logs/logs_*"))
        if not log_dirs:
            return None
            
        latest_log_dir = max(log_dirs, key=lambda x: x.name)
        
        # Initialize resource metrics
        resource_data = {
            'patient_id': patient_id,
            'mode': mode,
            'experiment_dir': exp_info['experiment_dir'],
            **exp_info  # Include all experiment info
        }
        
        # Extract training loss and convergence metrics
        loss_metrics = self._extract_loss_metrics(latest_log_dir)
        if loss_metrics:
            resource_data.update(loss_metrics)
        
        # Extract hardware usage from logs
        hardware_metrics = self._extract_hardware_metrics(latest_log_dir)
        if hardware_metrics:
            resource_data.update(hardware_metrics)
            
        # Extract power consumption estimates
        power_metrics = self._estimate_power_consumption(resource_data)
        if power_metrics:
            resource_data.update(power_metrics)
        
        return resource_data
    
    def _extract_distillation_resources(self, dist_path: Path, exp_info: Dict) -> List[Dict]:
        """Extract resources from distillation experiments"""
        resources = []
        
        # Look for JSON result files in efficiency experiments
        json_files = list(dist_path.glob("**/*.json"))
        
        # Also check for JSON files in the main efficiency_experiments directory
        efficiency_json_pattern = self.base_path / "efficiency_experiments" / "experiments" / "*" / "*" / "*.json"
        efficiency_json_files = glob.glob(str(efficiency_json_pattern))
        json_files.extend([Path(f) for f in efficiency_json_files])
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract efficiency metrics from JSON
                if isinstance(data, dict):
                    resource_data = {
                        'mode': 'distillation',
                        'experiment_type': exp_info.get('experiment_type', 'distillation'),
                        'experiment_name': exp_info.get('experiment_name', json_file.stem),
                        'experiment_dir': str(dist_path),
                        'json_file': str(json_file)
                    }
                    
                    # Extract model information
                    if 'model_name' in data:
                        resource_data['model_name'] = data['model_name']
                    elif 'model' in data:
                        resource_data['model_name'] = data['model']
                    else:
                        # Try to extract from file path
                        path_parts = json_file.parts
                        for part in path_parts:
                            if any(model in part.lower() for model in ['bert', 'gpt', 'llama', 'tiny']):
                                resource_data['model_name'] = part
                                break
                        else:
                            resource_data['model_name'] = json_file.stem
                    
                    # Extract efficiency metrics
                    metric_mappings = {
                        'model_size_mb': ['model_size_mb', 'size_mb', 'model_size'],
                        'parameters': ['parameters', 'params', 'parameter_count'],
                        'cpu_latency': ['cpu_latency_ms', 'cpu_latency', 'inference_time_cpu'],
                        'gpu_latency': ['gpu_latency_ms', 'gpu_latency', 'inference_time_gpu'],
                        'memory_usage': ['memory_usage_mb', 'memory_usage', 'peak_memory'],
                        'power_consumption': ['power_watts', 'power_consumption', 'estimated_power'],
                        'rmse': ['rmse', 'error', 'loss'],
                        'training_time': ['training_time_s', 'training_time', 'train_duration'],
                        'inference_duration': ['inference_duration_s', 'inference_time', 'inference_duration']
                    }
                    
                    for standard_key, possible_keys in metric_mappings.items():
                        for key in possible_keys:
                            if key in data:
                                resource_data[standard_key] = data[key]
                                break
                    
                    # Handle nested latency data
                    if 'latency' in data and isinstance(data['latency'], dict):
                        for k, v in data['latency'].items():
                            resource_data[f'latency_{k}'] = v
                    elif 'latency' in data:
                        resource_data['latency'] = data['latency']
                    
                    # Handle nested memory data
                    if 'memory_usage' in data and isinstance(data['memory_usage'], dict):
                        for k, v in data['memory_usage'].items():
                            resource_data[f'memory_{k}'] = v
                    
                    # Calculate derived metrics
                    if 'model_size_mb' in resource_data and 'parameters' in resource_data:
                        # Calculate compression ratio if this seems like a distilled model
                        if any(term in resource_data['model_name'].lower() for term in ['tiny', 'small', 'distil']):
                            resource_data['compression_ratio'] = resource_data['parameters'] / 1e6  # Rough baseline
                    
                    resources.append(resource_data)
                    
            except Exception as e:
                # Silently continue - many JSON files may not be efficiency reports
                continue
        
        return resources
    
    def _extract_loss_metrics(self, log_dir: Path) -> Optional[Dict]:
        """Extract training loss and convergence metrics"""
        loss_file = log_dir / "loss.pkl"
        
        if not loss_file.exists():
            return None
            
        try:
            with open(loss_file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, tuple) and len(data) >= 2:
                train_losses, val_losses = data[0], data[1]
                
                if len(train_losses) > 0:
                    return {
                        'initial_loss': train_losses[0],
                        'final_loss': train_losses[-1],
                        'best_loss': min(train_losses),
                        'loss_reduction': train_losses[0] - train_losses[-1],
                        'convergence_epochs': len(train_losses),
                        'training_efficiency': (train_losses[0] - train_losses[-1]) / len(train_losses) if len(train_losses) > 0 else 0,
                        'loss_stability': np.std(train_losses[-3:]) if len(train_losses) >= 3 else 0  # Stability in last 3 epochs
                    }
        except Exception as e:
            print(f"Error reading loss file {loss_file}: {e}")
            
        return None
    
    def _extract_hardware_metrics(self, log_dir: Path) -> Optional[Dict]:
        """Extract hardware usage metrics from log files"""
        log_file = log_dir / "log.log"
        
        if not log_file.exists():
            return None
            
        hardware_metrics = {
            'peak_memory_gb': None,
            'avg_gpu_utilization': None,
            'execution_time_seconds': None,
            'throughput_samples_per_sec': None
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract timing information
            start_times = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
            if len(start_times) >= 2:
                try:
                    start_dt = datetime.strptime(start_times[0], '%Y-%m-%d %H:%M:%S')
                    end_dt = datetime.strptime(start_times[-1], '%Y-%m-%d %H:%M:%S')
                    hardware_metrics['execution_time_seconds'] = (end_dt - start_dt).total_seconds()
                except:
                    pass
            
            # Extract memory usage (if available in logs)
            memory_matches = re.findall(r'memory.*?(\d+\.?\d*)\s*GB', content, re.IGNORECASE)
            if memory_matches:
                hardware_metrics['peak_memory_gb'] = max([float(m) for m in memory_matches])
            
            # Extract GPU utilization (if available)
            gpu_matches = re.findall(r'gpu.*?(\d+\.?\d*)%', content, re.IGNORECASE)
            if gpu_matches:
                hardware_metrics['avg_gpu_utilization'] = np.mean([float(g) for g in gpu_matches])
            
            # Calculate throughput if we have timing and sample info
            sample_matches = re.findall(r'(\d+)\s*samples?', content, re.IGNORECASE)
            if sample_matches and hardware_metrics['execution_time_seconds']:
                total_samples = sum([int(s) for s in sample_matches])
                hardware_metrics['throughput_samples_per_sec'] = total_samples / hardware_metrics['execution_time_seconds']
                
        except Exception as e:
            print(f"Error extracting hardware metrics from {log_file}: {e}")
        
        # Return only non-None metrics
        return {k: v for k, v in hardware_metrics.items() if v is not None}
    
    def _estimate_power_consumption(self, resource_data: Dict) -> Optional[Dict]:
        """Estimate power consumption based on model size and usage"""
        
        power_metrics = {}
        
        # Base power consumption estimates (watts)
        model_power_estimates = {
            'BERT': {'base': 15, 'training_multiplier': 3.0, 'inference_multiplier': 1.0},
            'GPT2': {'base': 25, 'training_multiplier': 3.5, 'inference_multiplier': 1.2},
            'LLAMA': {'base': 45, 'training_multiplier': 4.0, 'inference_multiplier': 1.5},
            'tinybert': {'base': 8, 'training_multiplier': 2.0, 'inference_multiplier': 0.8},
            'distilled': {'base': 10, 'training_multiplier': 2.2, 'inference_multiplier': 0.9}
        }
        
        model_name = resource_data.get('model_name', '').upper()
        mode = resource_data.get('mode', 'inference')
        
        # Find matching model
        model_config = None
        for model_key, config in model_power_estimates.items():
            if model_key.upper() in model_name or model_name in model_key.upper():
                model_config = config
                break
        
        if not model_config:
            model_config = model_power_estimates['BERT']  # Default fallback
        
        # Calculate power consumption
        base_power = model_config['base']
        multiplier = model_config['training_multiplier'] if 'train' in mode else model_config['inference_multiplier']
        
        estimated_power = base_power * multiplier
        
        # Adjust for model dimension if available
        model_dim = resource_data.get('model_dim')
        if model_dim:
            # Scale power based on model dimension (rough approximation)
            dim_factor = model_dim / 768  # Normalize to BERT-base dimension
            estimated_power *= (dim_factor ** 0.5)  # Square root scaling
        
        power_metrics['estimated_power_watts'] = estimated_power
        
        # Calculate energy consumption if execution time is available
        execution_time = resource_data.get('execution_time_seconds')
        if execution_time:
            energy_kwh = (estimated_power * execution_time) / 3600 / 1000  # Convert to kWh
            power_metrics['estimated_energy_kwh'] = energy_kwh
            power_metrics['energy_efficiency_score'] = 1.0 / energy_kwh if energy_kwh > 0 else 0
        
        # Calculate power efficiency based on model performance
        if resource_data.get('final_loss'):
            # Lower loss = better efficiency
            loss_factor = 1.0 / (resource_data['final_loss'] / 100)  # Normalize loss
            power_metrics['power_efficiency_score'] = loss_factor / estimated_power
        
        return power_metrics
    
    def create_efficiency_summary(self) -> pd.DataFrame:
        """
        Create comprehensive efficiency summary combining all resource metrics
        
        Returns:
            DataFrame with complete efficiency analysis
        """
        # Get all resource data
        training_df = self.extract_training_resources()
        inference_df = self.extract_inference_resources()
        distillation_df = self.extract_distillation_resources()
        
        # Combine all data
        all_dfs = []
        if not training_df.empty:
            all_dfs.append(training_df)
        if not inference_df.empty:
            all_dfs.append(inference_df)
        if not distillation_df.empty:
            all_dfs.append(distillation_df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Calculate derived efficiency metrics
        if not combined_df.empty:
            combined_df = self._calculate_efficiency_scores(combined_df)
        
        return combined_df
    
    def _calculate_efficiency_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite efficiency scores"""
        
        df = df.copy()
        
        # Calculate normalized scores (0-100 scale)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Power efficiency score (lower power consumption = higher score)
        if 'estimated_power_watts' in df.columns:
            max_power = df['estimated_power_watts'].max()
            df['power_efficiency_normalized'] = 100 * (1 - df['estimated_power_watts'] / max_power)
        
        # Training efficiency score (higher loss reduction per epoch = higher score)
        if 'training_efficiency' in df.columns:
            max_training_eff = df['training_efficiency'].max()
            if max_training_eff > 0:
                df['training_efficiency_normalized'] = 100 * (df['training_efficiency'] / max_training_eff)
        
        # Speed efficiency score (higher throughput = higher score)
        if 'throughput_samples_per_sec' in df.columns:
            max_throughput = df['throughput_samples_per_sec'].max()
            if max_throughput > 0:
                df['speed_efficiency_normalized'] = 100 * (df['throughput_samples_per_sec'] / max_throughput)
        
        # Memory efficiency score (lower memory usage = higher score)
        if 'peak_memory_gb' in df.columns:
            max_memory = df['peak_memory_gb'].max()
            if max_memory > 0:
                df['memory_efficiency_normalized'] = 100 * (1 - df['peak_memory_gb'] / max_memory)
        
        # Composite efficiency score (weighted average)
        efficiency_components = []
        weights = []
        
        for col, weight in [('power_efficiency_normalized', 0.3), 
                           ('training_efficiency_normalized', 0.25),
                           ('speed_efficiency_normalized', 0.25), 
                           ('memory_efficiency_normalized', 0.2)]:
            if col in df.columns:
                efficiency_components.append(df[col].fillna(50))  # Use 50 as default for missing values
                weights.append(weight)
        
        if efficiency_components:
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            df['composite_efficiency_score'] = sum(w * comp for w, comp in zip(weights, efficiency_components))
        
        return df