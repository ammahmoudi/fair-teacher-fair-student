"""
Enhanced Data Loader for Time-LLM JSON Files

This module provides an enhanced data loader that properly parses
nested Time-LLM JSON structures while maintaining compatibility
with existing Chronos flat JSON format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class EnhancedEfficiencyDataLoader:
    """Enhanced data loader that properly parses nested Time-LLM JSON files"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def scan_all_files(self) -> Dict[str, List[Path]]:
        """Scan for all JSON files in the experiments directory, prioritizing inference files"""
        all_files = {
            'efficiency_reports': [],
            'comprehensive_reports': [],
            'real_performance_reports': []
        }
        
        # Track found files by experiment to prioritize inference over training
        experiment_file_map = {}
        
        # Scan all experiment directories
        experiments_dir = self.base_path / 'efficiency_experiments' / 'experiments'
        if experiments_dir.exists():
            for json_file in experiments_dir.rglob('*.json'):
                filename = json_file.name.lower()
                
                # Create a unique experiment key from the path (exclude logs directory and filename)
                path_parts = json_file.parts
                experiment_key = "/".join(path_parts[:-2])  # Remove logs directory and filename
                
                if experiment_key not in experiment_file_map:
                    experiment_file_map[experiment_key] = {
                        'efficiency_reports': [],
                        'comprehensive_reports': [],
                        'real_performance_reports': []
                    }
                
                # Categorize files and track inference vs training preference
                file_info = {
                    'path': json_file,
                    'is_inference': 'training_inference' in filename,
                    'is_training': 'training_mode' in filename and 'training_inference' not in filename
                }
                
                if 'efficiency_report' in filename:
                    experiment_file_map[experiment_key]['efficiency_reports'].append(file_info)
                elif 'comprehensive_performance_report' in filename:
                    experiment_file_map[experiment_key]['comprehensive_reports'].append(file_info)
                elif 'real_performance_report' in filename:
                    experiment_file_map[experiment_key]['real_performance_reports'].append(file_info)
        
        # Now prioritize inference files over training files for each experiment
        for experiment_key, file_groups in experiment_file_map.items():
            for report_type, file_infos in file_groups.items():
                if not file_infos:
                    continue
                
                # Prioritize inference files over training files
                inference_files = [f for f in file_infos if f['is_inference']]
                training_files = [f for f in file_infos if f['is_training']]
                other_files = [f for f in file_infos if not f['is_inference'] and not f['is_training']]
                
                # Use inference files if available, otherwise use training files, otherwise use other files
                if inference_files:
                    selected_files = inference_files
                    print(f"🎯 Using inference files for {experiment_key}/{report_type}")
                elif training_files:
                    selected_files = training_files
                    print(f"📚 Using training files for {experiment_key}/{report_type}")
                else:
                    selected_files = other_files
                
                # Add selected files to the main collection
                all_files[report_type].extend([f['path'] for f in selected_files])
        
        return all_files
    
    def _extract_model_name_from_path(self, file_path: Path) -> str:
        """Extract model name from file path"""
        path_str = str(file_path)
        
        # Check for Time-LLM models in path
        if 'model_BERT' in path_str:
            return 'BERT'
        elif 'model_GPT2' in path_str:
            return 'GPT2'
        elif 'model_LLAMA' in path_str:
            return 'LLAMA'
        elif 'chronos' in path_str.lower():
            if 't5-tiny' in path_str:
                return 'chronos-t5-tiny'
            elif 't5-base' in path_str:
                return 'chronos-t5-base'
            else:
                return 'chronos-unknown'
        elif 'tinybert' in path_str.lower():
            return 'tinybert'
        
        # Fallback: extract from filename or directory
        parts = file_path.parts
        for part in reversed(parts):
            if any(model in part.upper() for model in ['BERT', 'GPT2', 'LLAMA']):
                if 'BERT' in part.upper():
                    return 'BERT'
                elif 'GPT2' in part.upper():
                    return 'GPT2'
                elif 'LLAMA' in part.upper():
                    return 'LLAMA'
        
        return 'unknown'
    
    def _extract_experiment_type(self, file_path: Path) -> str:
        """Extract experiment type from file path"""
        path_str = str(file_path)
        
        if 'time_llm_inference' in path_str:
            return 'time_llm_inference_ohiot1dm'
        elif 'time_llm_training' in path_str:
            return 'time_llm_training_ohiot1dm'
        elif 'chronos_inference' in path_str:
            return 'chronos_inference_ohiot1dm'
        elif 'chronos_training' in path_str:
            return 'chronos_training_ohiot1dm'
        elif 'distillation_inference' in path_str:
            return 'distillation_inference_ohiot1dm'
        elif 'distillation' in path_str:
            return 'distillation_experiments'
        
        return 'unknown_experiment'
    
    def _extract_best_measurements(self, detailed_data: dict, mode: str) -> dict:
        """Extract the most reliable measurements from detailed_measurements section"""
        best_data = {}
        
        if mode in detailed_data and 'real_performance_measurements' in detailed_data[mode]:
            real_perf = detailed_data[mode]['real_performance_measurements']
            
            # Best latency source: inference_timing section
            if 'inference_timing' in real_perf:
                timing = real_perf['inference_timing']
                best_data.update({
                    'real_avg_latency_ms': timing.get('average_latency_ms'),
                    'real_median_latency_ms': timing.get('median_latency_ms'),
                    'real_min_latency_ms': timing.get('min_latency_ms'),
                    'real_max_latency_ms': timing.get('max_latency_ms'),
                    'real_p95_latency_ms': timing.get('p95_latency_ms'),
                    'total_inferences_measured': timing.get('total_inferences_measured')
                })
            
            # Best memory sources: separate sections for RAM vs VRAM
            if 'memory_usage' in real_perf:
                mem_usage = real_perf['memory_usage']
                best_data.update({
                    'real_process_peak_ram_mb': mem_usage.get('process_peak_ram_mb'),
                    'real_process_avg_ram_mb': mem_usage.get('process_average_ram_mb'),
                    'real_system_peak_ram_mb': mem_usage.get('system_peak_ram_mb'),
                    'ram_measurements_count': mem_usage.get('ram_measurements_count')
                })
            
            # PyTorch GPU memory tracking
            if 'gpu_memory_usage' in real_perf:
                gpu_mem = real_perf['gpu_memory_usage']
                best_data.update({
                    'real_gpu_peak_allocated_mb': gpu_mem.get('peak_allocated_mb'),
                    'real_gpu_avg_allocated_mb': gpu_mem.get('average_allocated_mb'),
                    'real_gpu_peak_reserved_mb': gpu_mem.get('peak_reserved_mb'),
                    'real_gpu_avg_reserved_mb': gpu_mem.get('average_reserved_mb'),
                    'gpu_measurements_count': gpu_mem.get('gpu_measurements_count')
                })
            
            # NVIDIA ML metrics (most reliable for hardware-level measurements)
            if 'nvidia_ml_metrics' in real_perf:
                nvidia_ml = real_perf['nvidia_ml_metrics']
                best_data.update({
                    'nvidia_peak_vram_mb': nvidia_ml.get('peak_used_memory_mb'),
                    'nvidia_avg_vram_mb': nvidia_ml.get('average_used_memory_mb'),
                    'nvidia_min_vram_mb': nvidia_ml.get('min_used_memory_mb'),
                    'nvidia_peak_gpu_util_pct': nvidia_ml.get('peak_gpu_utilization_percent'),
                    'nvidia_avg_gpu_util_pct': nvidia_ml.get('average_gpu_utilization_percent'),
                    'nvidia_peak_mem_util_pct': nvidia_ml.get('peak_memory_utilization_percent'),
                    'nvidia_avg_mem_util_pct': nvidia_ml.get('average_memory_utilization_percent'),
                    'nvidia_peak_power_w': nvidia_ml.get('peak_power_usage_watts'),
                    'nvidia_avg_power_w': nvidia_ml.get('average_power_usage_watts'),
                    'nvidia_peak_temp_c': nvidia_ml.get('peak_temperature_celsius'),
                    'nvidia_avg_temp_c': nvidia_ml.get('average_temperature_celsius'),
                    'nvidia_total_gpu_memory_mb': nvidia_ml.get('total_gpu_memory_mb'),
                    'nvidia_ml_measurements_count': nvidia_ml.get('nvidia_ml_measurements_count')
                })
        
        return best_data
    
    def _extract_report_type(self, file_path: Path) -> str:
        """Extract report type from filename"""
        filename = file_path.name.lower()
        
        if 'efficiency_report' in filename:
            return 'efficiency_reports'
        elif 'comprehensive_performance_report' in filename:
            return 'comprehensive_reports'
        elif 'real_performance_report' in filename:
            return 'real_performance_reports'
        
        return 'unknown_report'
    
    def parse_time_llm_comprehensive(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse Time-LLM comprehensive JSON files with nested structure"""
        results = []
        model_name = self._extract_model_name_from_path(file_path)
        
        # Parse performance_summary section
        if 'performance_summary' in data:
            perf_summary = data['performance_summary']
            
            # Parse training_inference data (key section with inference metrics)
            if 'training_inference' in perf_summary:
                inf_data = perf_summary['training_inference']
                
                # Extract enhanced measurements from detailed section using correct mode
                enhanced_data = {}
                if 'detailed_measurements' in data:
                    enhanced_data = self._extract_best_measurements(data['detailed_measurements'], 'training_inference')
                
                record = {
                    'experiment_type': self._extract_experiment_type(file_path),
                    'model_name': model_name,
                    'mode': 'inference',
                    'report_type': self._extract_report_type(file_path),
                    'file_path': str(file_path),
                    
                    # Model characteristics
                    'total_parameters': inf_data.get('parameters_count'),
                    'model_size_mb': inf_data.get('model_size_on_disk_mb'),
                    
                    # Inference performance (prioritize real measurements over summary)
                    'avg_inference_time_ms': inf_data.get('average_latency_ms'),
                    'median_latency_ms': inf_data.get('median_latency_ms'), 
                    'p95_latency_ms': inf_data.get('p95_latency_ms'),
                    'min_inference_time_ms': inf_data.get('min_latency_ms'),
                    'max_inference_time_ms': inf_data.get('max_latency_ms'),
                    
                    # Memory usage (enhanced with best source priority)
                    'inference_peak_ram_mb': inf_data.get('process_peak_ram_mb'),  # Process-specific (most accurate)
                    'process_average_ram_mb': inf_data.get('process_average_ram_mb'),
                    'system_peak_ram_mb': inf_data.get('system_peak_ram_mb'),      # System-wide for context
                    
                    # GPU Memory - PyTorch tracking (prioritize detailed measurements from training_inference)
                    'inference_peak_gpu_mb': enhanced_data.get('real_gpu_peak_allocated_mb', inf_data.get('peak_gpu_allocated_mb')),
                    'average_gpu_allocated_mb': enhanced_data.get('real_gpu_avg_allocated_mb', inf_data.get('average_gpu_allocated_mb')),
                    'peak_gpu_reserved_mb': enhanced_data.get('real_gpu_peak_reserved_mb', inf_data.get('peak_gpu_reserved_mb')),
                    'average_gpu_reserved_mb': enhanced_data.get('real_gpu_avg_reserved_mb', inf_data.get('average_gpu_reserved_mb')),
                    
                    # VRAM - Use PyTorch model-specific from detailed measurements (NOT system-wide NVIDIA ML)
                    'current_vram_usage_mb': enhanced_data.get('real_gpu_peak_allocated_mb', inf_data.get('peak_gpu_allocated_mb')),
                    'gpu_avg_allocated_mb': enhanced_data.get('real_gpu_avg_allocated_mb', inf_data.get('average_gpu_allocated_mb')),
                    'gpu_reserved_mb': enhanced_data.get('real_gpu_peak_reserved_mb', inf_data.get('peak_gpu_reserved_mb')),
                    
                    # Keep NVIDIA ML for system-wide context (includes all GPU processes)
                    'nvidia_system_vram_mb': inf_data.get('peak_used_memory_mb'),        # NVIDIA ML system-wide
                    'nvidia_avg_system_vram_mb': inf_data.get('average_used_memory_mb'), # NVIDIA ML average
                    
                    # Performance metrics (enhanced with hardware-level measurements)
                    'peak_gpu_utilization_percent': inf_data.get('peak_gpu_utilization_percent'),
                    'average_gpu_utilization_percent': inf_data.get('average_gpu_utilization_percent'),
                    'peak_memory_utilization_percent': inf_data.get('peak_memory_utilization_percent'),
                    'average_memory_utilization_percent': inf_data.get('average_memory_utilization_percent'),
                    'peak_temperature_celsius': inf_data.get('peak_temperature_celsius'),
                    'average_temperature_celsius': inf_data.get('average_temperature_celsius'),
                    
                    # Power measurements (nvidia-ml hardware sensors - most accurate)
                    'inference_avg_power_w': inf_data.get('average_power_usage_watts'),
                    'peak_power_usage_watts': inf_data.get('peak_power_usage_watts'),
                    
                    # Measurement reliability metrics
                    'gpu_measurement_count': inf_data.get('gpu_measurements_count'),
                    'nvidia_ml_measurement_count': inf_data.get('nvidia_ml_measurements_count'),
                    
                    # Calculate throughput
                    'throughput_predictions_per_sec': (1000.0 / inf_data.get('average_latency_ms', float('inf'))) 
                                                    if inf_data.get('average_latency_ms') else None,
                    
                    # Edge deployment
                    'edge_feasibility': inf_data.get('edge_feasibility', 'unknown'),
                    'total_inferences': inf_data.get('total_inferences')
                }
                results.append(record)
        
        return results
    
    def parse_real_performance_report(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse real performance report JSON files (direct inference measurements)"""
        results = []
        model_name = self._extract_model_name_from_path(file_path)
        
        # Extract data from the direct real performance structure
        real_perf = data.get('real_performance_measurements', {})
        model_chars = data.get('model_characteristics', {})
        edge_analysis = data.get('edge_deployment_analysis', {})
        
        # Get timing data
        timing = real_perf.get('inference_timing', {})
        memory = real_perf.get('memory_usage', {})
        gpu_memory = real_perf.get('gpu_memory_usage', {})
        nvidia_ml = real_perf.get('nvidia_ml_metrics', {})
        model_file = real_perf.get('model_file_metrics', {})
        
        record = {
            'experiment_type': self._extract_experiment_type(file_path),
            'model_name': model_name,
            'mode': 'inference',
            'report_type': self._extract_report_type(file_path),
            'file_path': str(file_path),
            
            # Model characteristics (from model_characteristics section)
            'total_parameters': model_chars.get('total_parameters') or model_file.get('parameters_count'),
            'model_size_mb': model_chars.get('model_size_mb') or model_file.get('model_size_on_disk_mb'),
            
            # Inference performance (from inference_timing)
            'avg_inference_time_ms': timing.get('average_latency_ms'),
            'median_latency_ms': timing.get('median_latency_ms'), 
            'p95_latency_ms': timing.get('p95_latency_ms'),
            'min_inference_time_ms': timing.get('min_latency_ms'),
            'max_inference_time_ms': timing.get('max_latency_ms'),
            'total_inferences': timing.get('total_inferences_measured'),
            
            # Memory usage (prioritize process-specific over system-wide)
            'inference_peak_ram_mb': memory.get('process_peak_ram_mb'),
            'process_average_ram_mb': memory.get('process_average_ram_mb'),
            'system_peak_ram_mb': memory.get('system_peak_ram_mb'),
            'system_average_ram_mb': memory.get('system_average_ram_mb'),
            
            # GPU Memory - PyTorch tracking (most accurate for model usage)
            'inference_peak_gpu_mb': gpu_memory.get('peak_allocated_mb'),
            'average_gpu_allocated_mb': gpu_memory.get('average_allocated_mb'),
            'peak_gpu_reserved_mb': gpu_memory.get('peak_reserved_mb'),
            'average_gpu_reserved_mb': gpu_memory.get('average_reserved_mb'),
            
            # VRAM - Use PyTorch model-specific (NOT system-wide NVIDIA ML)
            'current_vram_usage_mb': gpu_memory.get('peak_allocated_mb'),
            'gpu_avg_allocated_mb': gpu_memory.get('average_allocated_mb'),
            'gpu_reserved_mb': gpu_memory.get('peak_reserved_mb'),
            
            # Keep NVIDIA ML for system-wide context (includes all GPU processes)
            'nvidia_system_vram_mb': nvidia_ml.get('peak_used_memory_mb'),
            'nvidia_avg_system_vram_mb': nvidia_ml.get('average_used_memory_mb'),
            
            # Performance metrics (from nvidia_ml_metrics)
            'peak_gpu_utilization_percent': nvidia_ml.get('peak_gpu_utilization_percent'),
            'average_gpu_utilization_percent': nvidia_ml.get('average_gpu_utilization_percent'),
            'peak_memory_utilization_percent': nvidia_ml.get('peak_memory_utilization_percent'),
            'average_memory_utilization_percent': nvidia_ml.get('average_memory_utilization_percent'),
            'peak_temperature_celsius': nvidia_ml.get('peak_temperature_celsius'),
            'average_temperature_celsius': nvidia_ml.get('average_temperature_celsius'),
            
            # Power measurements (nvidia-ml hardware sensors)
            'inference_avg_power_w': nvidia_ml.get('average_power_usage_watts'),
            'peak_power_usage_watts': nvidia_ml.get('peak_power_usage_watts'),
            
            # Measurement reliability metrics
            'gpu_measurement_count': gpu_memory.get('gpu_measurements_count'),
            'nvidia_ml_measurement_count': nvidia_ml.get('nvidia_ml_measurements_count'),
            'ram_measurements_count': memory.get('ram_measurements_count'),
            
            # Calculate throughput
            'throughput_predictions_per_sec': (1000.0 / timing.get('average_latency_ms', float('inf'))) 
                                            if timing.get('average_latency_ms') else None,
            
            # Edge deployment (from edge_deployment_analysis)
            'edge_feasibility': edge_analysis.get('feasibility_assessment', 'unknown'),
        }
        
        results.append(record)
        return results
    
    def parse_chronos_flat(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse Chronos JSON structure (which is actually nested, not flat)"""
        model_name = self._extract_model_name_from_path(file_path)
        
        # Extract nested data from Chronos structure
        timing_data = {}
        memory_data = {}
        model_data = {}
        
        # Extract timing measurements
        if 'real_performance_measurements' in data:
            real_perf = data['real_performance_measurements']
            if 'timing_measurements' in real_perf:
                timing_data = real_perf['timing_measurements']
            if 'memory_measurements' in real_perf:
                memory_data = real_perf['memory_measurements']
        
        # Extract model characteristics
        if 'model_characteristics' in data:
            model_data = data['model_characteristics']
        
        # Extract theoretical performance for additional metrics
        theoretical_data = data.get('theoretical_performance', {})
        
        record = {
            'experiment_type': self._extract_experiment_type(file_path),
            'model_name': model_name,
            'mode': data.get('mode', 'inference'),
            'report_type': self._extract_report_type(file_path),
            'file_path': str(file_path),
            
            # Model characteristics
            'total_parameters': model_data.get('total_parameters'),
            'model_size_mb': model_data.get('model_size_mb') or theoretical_data.get('model_size_on_disk_mb'),
            'model_dtype': model_data.get('model_dtype'),
            'model_architecture': model_data.get('model_architecture'),
            
            # Timing data from real measurements
            'avg_inference_time_ms': timing_data.get('average_inference_time_ms'),
            'min_inference_time_ms': timing_data.get('min_inference_time_ms'),
            'max_inference_time_ms': timing_data.get('max_inference_time_ms'),
            'measurement_runs': timing_data.get('measurement_runs'),
            
            # Memory data with enhanced source priority
            'inference_peak_ram_mb': memory_data.get('process_peak_ram_mb', memory_data.get('current_ram_usage_mb')),
            'system_peak_ram_mb': memory_data.get('system_peak_ram_mb'),
            
            # GPU Memory - PyTorch tracking
            'inference_peak_gpu_mb': memory_data.get('gpu_max_memory_allocated_mb'),
            'peak_gpu_reserved_mb': memory_data.get('gpu_memory_reserved_mb'),
            
            # VRAM - NVIDIA ML (most reliable for hardware-level VRAM)
            'current_vram_usage_mb': memory_data.get('current_vram_usage_mb'),
            'nvidia_peak_vram_mb': memory_data.get('peak_used_memory_mb'),
            'nvidia_avg_vram_mb': memory_data.get('average_used_memory_mb'),
            
            # Theoretical estimates
            'estimated_cpu_latency_ms': theoretical_data.get('estimated_cpu_latency_ms'),
            'estimated_gpu_latency_ms': theoretical_data.get('estimated_gpu_latency_ms'),
            'estimated_memory_usage_mb': theoretical_data.get('estimated_memory_usage_mb'),
            'estimated_throughput_samples_per_sec': theoretical_data.get('estimated_throughput_samples_per_sec'),
            
            # Calculate throughput from timing if available
            'throughput_predictions_per_sec': (1000.0 / timing_data.get('average_inference_time_ms', float('inf'))) 
                                            if timing_data.get('average_inference_time_ms') else None,
            
            # Edge deployment
            'edge_feasibility': data.get('edge_deployment_analysis', {}).get('overall_feasibility', 'unknown'),
            'feasible_edge_devices': data.get('edge_deployment_analysis', {}).get('feasible_edge_devices', [])
        }
        
        return [record]
    
    def parse_chronos_comprehensive(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse Chronos comprehensive JSON files"""
        model_name = self._extract_model_name_from_path(file_path)
        
        results = []
        
        # Parse performance_summary section for Chronos
        if 'performance_summary' in data and 'inference' in data['performance_summary']:
            inf_data = data['performance_summary']['inference']
            
            record = {
                'experiment_type': self._extract_experiment_type(file_path),
                'model_name': model_name,
                'mode': 'inference',
                'report_type': self._extract_report_type(file_path),
                'file_path': str(file_path),
                
                # Model characteristics  
                'total_parameters': inf_data.get('parameters_count'),
                'model_size_mb': inf_data.get('model_size_on_disk_mb'),
                
                # Inference performance
                'avg_inference_time_ms': inf_data.get('average_latency_ms'),
                'median_latency_ms': inf_data.get('median_latency_ms'),
                'p95_latency_ms': inf_data.get('p95_latency_ms'),
                'total_inferences': inf_data.get('total_inferences'),
                
                # Memory usage
                'inference_peak_ram_mb': inf_data.get('process_peak_ram_mb'),
                'process_average_ram_mb': inf_data.get('process_average_ram_mb'),
                'system_peak_ram_mb': inf_data.get('system_peak_ram_mb'),
                'inference_peak_gpu_mb': inf_data.get('peak_gpu_allocated_mb'),
                'average_gpu_allocated_mb': inf_data.get('average_gpu_allocated_mb'),
                'peak_gpu_reserved_mb': inf_data.get('peak_gpu_reserved_mb'),
                
                # Performance metrics
                'peak_gpu_utilization_percent': inf_data.get('peak_gpu_utilization_percent'),
                'average_gpu_utilization_percent': inf_data.get('average_gpu_utilization_percent'),
                'peak_temperature_celsius': inf_data.get('peak_temperature_celsius'),
                'inference_avg_power_w': inf_data.get('average_power_usage_watts'),
                'peak_power_usage_watts': inf_data.get('peak_power_usage_watts'),
                
                # Calculate throughput
                'throughput_predictions_per_sec': (1000.0 / inf_data.get('average_latency_ms', float('inf'))) 
                                                if inf_data.get('average_latency_ms') else None,
                
                # Edge deployment
                'edge_feasibility': inf_data.get('edge_feasibility', 'unknown'),
                'feasible_edge_devices': inf_data.get('feasible_edge_devices', [])
            }
            results.append(record)
        
        return results

    def parse_single_file(self, file_path: Path) -> List[Dict]:
        """Parse a single JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a real_performance_report with direct structure (inference files)
            if (isinstance(data, dict) and 'real_performance_measurements' in data and 
                'inference_timing' in data['real_performance_measurements']):
                return self.parse_real_performance_report(data, file_path)
            
            # Check if this is a comprehensive report with performance_summary
            elif isinstance(data, dict) and 'performance_summary' in data:
                # Check if it's Time-LLM format (has training_inference) or Chronos format (has inference)
                if 'training_inference' in data['performance_summary']:
                    return self.parse_time_llm_comprehensive(data, file_path)
                elif 'inference' in data['performance_summary']:
                    return self.parse_chronos_comprehensive(data, file_path)
                else:
                    # Fallback to regular Time-LLM parsing
                    return self.parse_time_llm_comprehensive(data, file_path)
            else:
                # Regular Chronos efficiency report format
                return self.parse_chronos_flat(data, file_path)
                
        except Exception as e:
            print(f"⚠️ Error parsing {file_path}: {e}")
            return []
    
    def parse_all_data(self) -> pd.DataFrame:
        """Parse all JSON files and return a pandas DataFrame"""
        print("🔍 Scanning for experiment files...")
        all_files = self.scan_all_files()
        
        total_files = sum(len(files) for files in all_files.values())
        print(f"📊 Found {total_files} JSON files")
        
        all_records = []
        
        for report_type, files in all_files.items():
            print(f"📊 Processing {len(files)} {report_type}...")
            for file_path in files:
                records = self.parse_single_file(file_path)
                all_records.extend(records)
        
        df = pd.DataFrame(all_records)
        print(f"✅ Loaded {len(df)} total records")
        
        return df