"""
Log Analysis Module for LLM Efficiency Experiments

This module provides comprehensive log parsing and analysis capabilities for 
efficiency experiments, particularly focusing on distillation training logs.
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import glob


class LogAnalyzer:
    """Analyzer for experiment log files with focus on efficiency metrics"""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize LogAnalyzer
        
        Args:
            base_path: Base directory path for experiments
        """
        self.base_path = Path(base_path)
        self.log_data = []
        
    def find_log_files(self, pattern: Optional[str] = None) -> List[str]:
        """
        Find all log files in efficiency experiments
        
        Args:
            pattern: Optional custom pattern for log files
            
        Returns:
            List of log file paths
        """
        if pattern is None:
            # Default pattern for efficiency experiments
            pattern = str(self.base_path / "efficiency_experiments" / "experiments" / "*" / "*" / "*" / "logs" / "*" / "log.log")
        
        return glob.glob(pattern)
    
    def parse_log_file(self, log_path: str) -> Optional[Dict]:
        """
        Parse a single log file to extract key information
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Dictionary containing extracted information or None if parsing fails
        """
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Extract basic info from path
            path_parts = Path(log_path).parts
            
            # Handle different path structures
            if len(path_parts) >= 6:
                experiment_type = path_parts[-6]  # e.g., time_llm_inference_ohiot1dm
                run_config = path_parts[-5]       # e.g., seed_831363_model_GPT2_...
            else:
                experiment_type = 'unknown'
                run_config = str(Path(log_path).parent.parent.name)
            
            # Parse model name
            model_match = re.search(r'model_([^_]+)', run_config)
            model_name = model_match.group(1) if model_match else 'unknown'
            
            # Parse experiment details
            mode_match = re.search(r'mode_(\w+)', run_config)
            mode = mode_match.group(1) if mode_match else 'unknown'
            
            # Extract timing information
            start_time = self._extract_start_time(content)
            end_time = self._extract_end_time(content)
            duration = self._calculate_duration(start_time, end_time)
            
            # Extract model-specific information
            epochs = self._extract_epochs(content)
            training_time = self._extract_training_time(content)
            memory_usage = self._extract_memory_usage(content)
            dataset_info = self._extract_dataset_info(content)
            
            # Add timing details
            start_hour = start_time.hour if start_time else None
            
            log_info = {
                'log_path': log_path,
                'experiment_type': experiment_type,
                'model_name': model_name,
                'mode': mode,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else None,
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else None,
                'start_hour': start_hour,
                'duration_seconds': duration,
                'epochs': epochs,
                'training_time': training_time,
                'memory_usage': memory_usage,
                'dataset_samples': dataset_info.get('samples', None),
                'sequence_length': dataset_info.get('seq_len', None)
            }
            
            return log_info
            
        except Exception as e:
            print(f"❌ Error parsing {log_path}: {e}")
            return None
    
    def _extract_start_time(self, content: str) -> Optional[datetime]:
        """Extract experiment start time"""
        start_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+', content)
        if start_match:
            try:
                return datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        return None
    
    def _extract_end_time(self, content: str) -> Optional[datetime]:
        """Extract experiment end time (last timestamp)"""
        timestamps = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+', content)
        if timestamps:
            try:
                return datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        return None
    
    def _calculate_duration(self, start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
        """Calculate duration in seconds"""
        if start and end:
            return (end - start).total_seconds()
        return None
    
    def _extract_epochs(self, content: str) -> Optional[int]:
        """Extract number of epochs from log"""
        # Look for epoch information in various formats
        epoch_patterns = [
            r'epochs_(\d+)',
            r'epoch (\d+)/',
            r'Epoch: (\d+)',
            r'Training for (\d+) epochs',
            r'train_epochs.*?(\d+)'
        ]
        
        for pattern in epoch_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
    
    def _extract_training_time(self, content: str) -> Optional[float]:
        """Extract training time information"""
        training_patterns = [
            r'Training completed in ([0-9.]+) seconds',
            r'Total training time: ([0-9.]+)',
            r'training_time.*?([0-9.]+)',
            r'Training took ([0-9.]+) seconds'
        ]
        
        for pattern in training_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _extract_memory_usage(self, content: str) -> Optional[float]:
        """Extract memory usage information"""
        memory_patterns = [
            r'max_memory_mb.*?([0-9.]+)',
            r'Memory usage: ([0-9.]+) MB',
            r'Peak memory: ([0-9.]+)',
            r'GPU memory.*?([0-9.]+)'
        ]
        
        for pattern in memory_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _extract_dataset_info(self, content: str) -> Dict:
        """Extract dataset information"""
        info = {}
        
        # Extract sample count
        samples_patterns = [
            r'Total samples.*?(\d+)',
            r'samples after.*?(\d+)',
            r'Dataset.*?(\d+) samples'
        ]
        
        for pattern in samples_patterns:
            match = re.search(pattern, content)
            if match:
                info['samples'] = int(match.group(1))
                break
        
        # Extract sequence length
        seq_patterns = [
            r'Sequence Length: (\d+)',
            r'seq_len.*?(\d+)',
            r'sequence_length.*?(\d+)'
        ]
        
        for pattern in seq_patterns:
            match = re.search(pattern, content)
            if match:
                info['seq_len'] = int(match.group(1))
                break
        
        return info
    
    def analyze_all_logs(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze all log files and return summary DataFrame
        
        Args:
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            DataFrame containing log analysis results
        """
        log_files = self.find_log_files()
        print(f"🔍 Found {len(log_files)} log files")
        
        if max_files:
            log_files = log_files[:max_files]
            print(f"📊 Processing first {len(log_files)} files")
        
        results = []
        for log_file in log_files:
            result = self.parse_log_file(log_file)
            if result:
                results.append(result)
        
        print(f"✅ Successfully parsed {len(results)} log files")
        self.log_data = results
        return pd.DataFrame(results)
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from analyzed logs"""
        if not self.log_data:
            self.analyze_all_logs()
        
        return pd.DataFrame(self.log_data)
    
    def get_distillation_logs(self) -> List[Dict]:
        """Filter logs for distillation experiments"""
        distill_logs = []
        
        for log in self.log_data:
            # Check if it's a distillation experiment
            if any(keyword in log.get('log_path', '').lower() 
                   for keyword in ['distill', 'phase_3', 'teacher', 'student']):
                log['is_distillation'] = True
                
                # Extract model role information
                log_path = log.get('log_path', '')
                if 'teacher' in log_path.lower():
                    log['model_role'] = 'teacher'
                elif 'student' in log_path.lower() or 'tiny' in log_path.lower():
                    log['model_role'] = 'student'
                elif 'bert_to_tinybert' in log_path.lower():
                    log['distillation_pair'] = 'BERT_to_TinyBERT'
                
                distill_logs.append(log)
        
        return distill_logs
    
    def generate_timeline_analysis(self) -> pd.DataFrame:
        """Generate timeline analysis of experiments"""
        if not self.log_data:
            self.analyze_all_logs()
        
        df = pd.DataFrame(self.log_data)
        
        if 'start_hour' in df.columns:
            timeline = df.groupby(['start_hour', 'model_name']).size().reset_index(name='count')
            return timeline
        
        return pd.DataFrame()
    
    def get_efficiency_summary(self) -> Dict:
        """Generate efficiency summary statistics"""
        if not self.log_data:
            self.analyze_all_logs()
        
        df = pd.DataFrame(self.log_data)
        
        summary = {
            'total_experiments': len(df),
            'unique_models': df['model_name'].nunique() if 'model_name' in df.columns else 0,
            'avg_duration_seconds': df['duration_seconds'].mean() if 'duration_seconds' in df.columns else 0,
            'avg_training_time': df['training_time'].mean() if 'training_time' in df.columns else 0,
            'models_analyzed': df['model_name'].unique().tolist() if 'model_name' in df.columns else [],
        }
        
        return summary


class DistillationLogAnalyzer(LogAnalyzer):
    """Specialized analyzer for distillation training logs"""
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path)
        self.distillation_paths = [
            "distillation_experiments",
            "efficiency_experiments/experiments/distillation_ohiot1dm",
            "efficiency_experiments/experiments/distillation_inference_ohiot1dm"
        ]
    
    def find_distillation_logs(self) -> List[str]:
        """Find logs specifically in distillation experiment directories"""
        all_logs = []
        
        for distill_path in self.distillation_paths:
            full_path = self.base_path / distill_path
            if full_path.exists():
                # Multiple patterns for different directory structures
                patterns = [
                    str(full_path / "**" / "*.log"),
                    str(full_path / "**" / "logs" / "**" / "*.log"),
                    str(full_path / "**" / "phase_3_distillation" / "**" / "*.log")
                ]
                
                for pattern in patterns:
                    logs = glob.glob(pattern, recursive=True)
                    all_logs.extend(logs)
        
        return list(set(all_logs))  # Remove duplicates
    
    def analyze_distillation_efficiency(self) -> Dict:
        """Analyze distillation-specific efficiency metrics"""
        distill_logs = self.find_distillation_logs()
        print(f"🧬 Found {len(distill_logs)} distillation log files")
        
        results = []
        for log_file in distill_logs[:20]:  # Limit to avoid overwhelming
            result = self.parse_log_file(log_file)
            if result:
                # Add distillation-specific metadata
                result['is_distillation'] = True
                result = self._add_distillation_metadata(result)
                results.append(result)
        
        return {
            'total_distillation_logs': len(results),
            'analysis_results': results,
            'summary_stats': self._calculate_distillation_stats(results)
        }
    
    def _add_distillation_metadata(self, log_info: Dict) -> Dict:
        """Add distillation-specific metadata to log information"""
        log_path = log_info.get('log_path', '')
        
        # Determine model role
        if 'teacher' in log_path.lower():
            log_info['model_role'] = 'teacher'
        elif 'student' in log_path.lower() or 'tiny' in log_path.lower():
            log_info['model_role'] = 'student'
        else:
            log_info['model_role'] = 'unknown'
        
        # Extract distillation pair information
        if 'bert_to_tinybert' in log_path.lower():
            log_info['distillation_pair'] = 'BERT_to_TinyBERT'
        elif 'bert' in log_path.lower() and 'tiny' in log_path.lower():
            log_info['distillation_pair'] = 'BERT_family'
        
        # Extract phase information
        if 'phase_3' in log_path.lower():
            log_info['distillation_phase'] = 'phase_3'
        elif 'phase_2' in log_path.lower():
            log_info['distillation_phase'] = 'phase_2'
        elif 'phase_1' in log_path.lower():
            log_info['distillation_phase'] = 'phase_1'
        
        return log_info
    
    def _calculate_distillation_stats(self, results: List[Dict]) -> Dict:
        """Calculate distillation-specific statistics"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        stats = {
            'avg_distillation_duration': df['duration_seconds'].mean() if 'duration_seconds' in df.columns else 0,
            'distillation_phases': df['distillation_phase'].value_counts().to_dict() if 'distillation_phase' in df.columns else {},
            'model_roles': df['model_role'].value_counts().to_dict() if 'model_role' in df.columns else {},
            'distillation_pairs': df['distillation_pair'].value_counts().to_dict() if 'distillation_pair' in df.columns else {}
        }
        
        return stats