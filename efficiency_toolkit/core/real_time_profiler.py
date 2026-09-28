"""Real-time performance profiler for capturing actual inference metrics."""

import time
import threading
import json
import os
from typing import Dict, List, Any, Optional
import torch
import psutil
import numpy as np
from datetime import datetime

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    pynvml = None


class RealTimeProfiler:
    """Captures real performance metrics during actual model inference."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.inference_times: List[float] = []
        self.memory_snapshots: List[Dict] = []
        self.gpu_memory_snapshots: List[Dict] = []
        self.nvidia_ml_snapshots: List[Dict] = []  # Enhanced NVIDIA ML monitoring
        self.throughput_measurements: List[Dict] = []
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Storage for all performance data across different modes
        self.all_performance_data = {
            'training': None,
            'inference': None,
            'training_inference': None
        }
        
        # Process-specific monitoring
        self.current_process = psutil.Process()
        self.baseline_memory_mb = None  # To track relative memory usage
        
        # System info
        self.system_info = self._get_system_info()
        
        # Initialize NVIDIA ML if available
        self.nvidia_ml_handle = None
        if NVIDIA_ML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvidia_ml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"Warning: Could not initialize NVIDIA ML: {e}")
                self.nvidia_ml_handle = None
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information once."""
        info = {}
        
        # CPU info
        info["cpu"] = {
            "brand": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency_ghz": psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 0
        }
        
        # Memory info
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_ram_gb": mem.total / (1024**3),
            "available_ram_gb": mem.available / (1024**3)
        }
        
        # GPU info
        info["gpu"] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info["gpu"]["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpu"]["devices"].append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
                
        return info
        
    def start_monitoring(self, interval: float = 0.1):
        """Start background monitoring of system resources."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        # Cleanup NVIDIA ML
        if self.nvidia_ml_handle and NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass  # Ignore cleanup errors
            
    def _monitor_resources(self, interval: float):
        """Background thread for monitoring system resources."""
        while self.is_monitoring:
            try:
                timestamp = time.time()
                
                # System-wide RAM usage
                mem = psutil.virtual_memory()
                
                # Process-specific memory usage
                process_memory = self.current_process.memory_info()
                process_ram_mb = process_memory.rss / (1024**2)  # Resident Set Size
                
                # If this is the first measurement, set baseline
                if self.baseline_memory_mb is None:
                    self.baseline_memory_mb = process_ram_mb
                
                memory_snapshot = {
                    "timestamp": timestamp,
                    # System-wide metrics (includes all processes)
                    "system_ram_used_mb": mem.used / (1024**2),
                    "system_ram_percent": mem.percent,
                    # Process-specific metrics (only this ML process)
                    "process_ram_mb": process_ram_mb,
                    "process_ram_delta_mb": process_ram_mb - self.baseline_memory_mb,
                    # Additional process details
                    "process_ram_peak_mb": process_ram_mb  # Will track peak separately
                }
                self.memory_snapshots.append(memory_snapshot)
                
                # PyTorch GPU memory if available
                if torch.cuda.is_available():
                    gpu_snapshot = {
                        "timestamp": timestamp,
                        "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                        "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                        "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2)
                    }
                    self.gpu_memory_snapshots.append(gpu_snapshot)
                
                # Enhanced NVIDIA ML monitoring
                if self.nvidia_ml_handle:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvidia_ml_handle)
                        util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.nvidia_ml_handle)
                        temperature = pynvml.nvmlDeviceGetTemperature(self.nvidia_ml_handle, pynvml.NVML_TEMPERATURE_GPU)
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(self.nvidia_ml_handle) / 1000.0
                        
                        nvidia_snapshot = {
                            "timestamp": timestamp,
                            "total_memory_mb": memory_info.total / (1024**2),
                            "used_memory_mb": memory_info.used / (1024**2),
                            "free_memory_mb": memory_info.free / (1024**2),
                            "gpu_utilization_percent": util_rates.gpu,
                            "memory_utilization_percent": util_rates.memory,
                            "temperature_celsius": temperature,
                            "power_usage_watts": power_usage
                        }
                        self.nvidia_ml_snapshots.append(nvidia_snapshot)
                    except Exception:
                        pass  # Continue if NVIDIA ML query fails
                
            except Exception:
                pass  # Continue monitoring even if individual snapshots fail
                
            time.sleep(interval)
            
    def measure_inference_call(self, predict_func, *args, **kwargs):
        """Measure a single inference call with real data."""
        # Pre-inference memory state
        pre_inference_time = time.perf_counter()
        pre_ram = psutil.virtual_memory().used / (1024**2)
        pre_gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Measure the actual inference call
        start_time = time.perf_counter()
        result = predict_func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Post-inference measurements
        post_ram = psutil.virtual_memory().used / (1024**2)
        post_gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Record measurements
        inference_time_ms = (end_time - start_time) * 1000
        self.inference_times.append(inference_time_ms)
        
        # Record detailed measurement
        measurement = {
            "timestamp": start_time,
            "inference_time_ms": inference_time_ms,
            "ram_before_mb": pre_ram,
            "ram_after_mb": post_ram,
            "ram_delta_mb": post_ram - pre_ram,
            "gpu_mem_before_mb": pre_gpu_mem,
            "gpu_mem_after_mb": post_gpu_mem,
            "gpu_mem_delta_mb": post_gpu_mem - pre_gpu_mem
        }
        
        self.throughput_measurements.append(measurement)
        
        return result
        
    def get_comprehensive_report(self, model_name: str, model_size_mb: float, model_params: int) -> Dict[str, Any]:
        """Generate comprehensive performance report from real measurements."""
        report = {
            "model_name": model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "system_info": self.system_info,
            "model_characteristics": {
                "total_parameters": model_params,
                "model_size_mb": model_size_mb
            },
            "real_performance_measurements": {}
        }
        
        # Store this report data for later combination
        mode_key = self._determine_mode_from_name(model_name)
        self.all_performance_data[mode_key] = report.copy()
        
        # Process inference timing measurements
        if self.inference_times:
            report["real_performance_measurements"]["inference_timing"] = {
                "total_inferences_measured": len(self.inference_times),
                "average_latency_ms": np.mean(self.inference_times),
                "median_latency_ms": np.median(self.inference_times),
                "min_latency_ms": np.min(self.inference_times),
                "max_latency_ms": np.max(self.inference_times),
                "std_latency_ms": np.std(self.inference_times),
                "p95_latency_ms": np.percentile(self.inference_times, 95),
                "p99_latency_ms": np.percentile(self.inference_times, 99)
            }
            
            # Calculate throughput
            if len(self.inference_times) > 1:
                avg_latency_sec = np.mean(self.inference_times) / 1000
                throughput_samples_per_sec = 1.0 / avg_latency_sec if avg_latency_sec > 0 else 0
                report["real_performance_measurements"]["throughput"] = {
                    "samples_per_second": throughput_samples_per_sec,
                    "samples_per_minute": throughput_samples_per_sec * 60,
                    "based_on_measurements": len(self.inference_times)
                }
        else:
            report["real_performance_measurements"]["inference_timing"] = {
                "note": "No inference calls were measured"
            }
            
        # Process memory measurements
        if self.memory_snapshots:
            # System-wide metrics (includes all processes)
            system_ram_usage = [snap["system_ram_used_mb"] for snap in self.memory_snapshots]
            # Process-specific metrics (only this ML process)
            process_ram_usage = [snap["process_ram_mb"] for snap in self.memory_snapshots]
            process_ram_delta = [snap["process_ram_delta_mb"] for snap in self.memory_snapshots]
            
            report["real_performance_measurements"]["memory_usage"] = {
                # System-wide memory (includes other applications)
                "system_peak_ram_mb": max(system_ram_usage),
                "system_average_ram_mb": np.mean(system_ram_usage),
                "system_min_ram_mb": min(system_ram_usage),
                # Process-specific memory (only this ML model)
                "process_peak_ram_mb": max(process_ram_usage),
                "process_average_ram_mb": np.mean(process_ram_usage),
                "process_baseline_ram_mb": self.baseline_memory_mb,
                "process_peak_delta_mb": max(process_ram_delta),
                "process_average_delta_mb": np.mean(process_ram_delta),
                "ram_measurements_count": len(system_ram_usage),
                "note": "system_* includes all processes, process_* is ML model only"
            }
        
        if self.gpu_memory_snapshots and torch.cuda.is_available():
            gpu_allocated = [snap["gpu_allocated_mb"] for snap in self.gpu_memory_snapshots]
            gpu_reserved = [snap["gpu_reserved_mb"] for snap in self.gpu_memory_snapshots]
            
            report["real_performance_measurements"]["gpu_memory_usage"] = {
                "peak_allocated_mb": max(gpu_allocated) if gpu_allocated else 0,
                "average_allocated_mb": np.mean(gpu_allocated) if gpu_allocated else 0,
                "peak_reserved_mb": max(gpu_reserved) if gpu_reserved else 0,
                "average_reserved_mb": np.mean(gpu_reserved) if gpu_reserved else 0,
                "gpu_measurements_count": len(gpu_allocated)
            }
            
        # Enhanced NVIDIA ML metrics
        if self.nvidia_ml_snapshots:
            used_memory = [snap["used_memory_mb"] for snap in self.nvidia_ml_snapshots]
            gpu_utilization = [snap["gpu_utilization_percent"] for snap in self.nvidia_ml_snapshots]
            memory_utilization = [snap["memory_utilization_percent"] for snap in self.nvidia_ml_snapshots]
            temperature = [snap["temperature_celsius"] for snap in self.nvidia_ml_snapshots]
            power_usage = [snap["power_usage_watts"] for snap in self.nvidia_ml_snapshots]
            
            report["real_performance_measurements"]["nvidia_ml_metrics"] = {
                "peak_used_memory_mb": max(used_memory),
                "average_used_memory_mb": np.mean(used_memory),
                "min_used_memory_mb": min(used_memory),
                "peak_gpu_utilization_percent": max(gpu_utilization),
                "average_gpu_utilization_percent": np.mean(gpu_utilization),
                "peak_memory_utilization_percent": max(memory_utilization),
                "average_memory_utilization_percent": np.mean(memory_utilization),
                "peak_temperature_celsius": max(temperature),
                "average_temperature_celsius": np.mean(temperature),
                "peak_power_usage_watts": max(power_usage),
                "average_power_usage_watts": np.mean(power_usage),
                "nvidia_ml_measurements_count": len(used_memory),
                "total_gpu_memory_mb": self.nvidia_ml_snapshots[0]["total_memory_mb"] if self.nvidia_ml_snapshots else 0
            }
            
        # Process throughput measurements for detailed analysis
        if self.throughput_measurements:
            report["real_performance_measurements"]["detailed_measurements"] = {
                "memory_efficiency": {
                    "avg_ram_delta_per_inference_mb": np.mean([m["ram_delta_mb"] for m in self.throughput_measurements]),
                    "avg_gpu_delta_per_inference_mb": np.mean([m["gpu_mem_delta_mb"] for m in self.throughput_measurements])
                },
                "measurement_count": len(self.throughput_measurements)
            }
        
        # Model file size on disk
        report["real_performance_measurements"]["model_file_metrics"] = {
            "model_size_on_disk_mb": model_size_mb,
            "parameters_count": model_params,
            "mb_per_million_params": model_size_mb / (model_params / 1_000_000) if model_params > 0 else 0
        }
        
        # Edge deployment analysis based on real measurements
        report["edge_deployment_analysis"] = self._analyze_edge_feasibility(report["real_performance_measurements"])
        
        return report
        
    def _analyze_edge_feasibility(self, real_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze edge deployment feasibility based on real measurements."""
        analysis = {
            "feasibility_assessment": "unknown",
            "constraints": {},
            "recommendations": []
        }
        
        # Memory constraints (use process-specific memory for accurate edge analysis)
        if "memory_usage" in real_metrics:
            # Use process-specific memory for edge deployment analysis
            process_peak_ram_mb = real_metrics["memory_usage"].get("process_peak_ram_mb", 
                                                                 real_metrics["memory_usage"].get("peak_ram_mb", 0))
            system_peak_ram_mb = real_metrics["memory_usage"].get("system_peak_ram_mb", 0)
            
            analysis["constraints"]["process_peak_memory_mb"] = process_peak_ram_mb
            analysis["constraints"]["system_peak_memory_mb"] = system_peak_ram_mb
            
            # Edge device memory limits (approximate)
            edge_limits = {
                "raspberry_pi_4": 4000,  # 4GB
                "jetson_nano": 4000,     # 4GB
                "jetson_xavier": 16000,  # 16GB
                "intel_nuc": 8000        # 8GB typical
            }
            
            feasible_devices = []
            for device, limit_mb in edge_limits.items():
                # Use process-specific memory for realistic edge deployment assessment
                if process_peak_ram_mb <= limit_mb * 0.8:  # 80% utilization threshold
                    feasible_devices.append(device)
                    
            analysis["feasible_edge_devices"] = feasible_devices
            
        # Latency constraints
        if "inference_timing" in real_metrics and "average_latency_ms" in real_metrics["inference_timing"]:
            avg_latency = real_metrics["inference_timing"]["average_latency_ms"]
            analysis["constraints"]["average_latency_ms"] = avg_latency
            
            if avg_latency < 100:
                analysis["feasibility_assessment"] = "excellent"
            elif avg_latency < 500:
                analysis["feasibility_assessment"] = "good"
            elif avg_latency < 1000:
                analysis["feasibility_assessment"] = "acceptable"
            else:
                analysis["feasibility_assessment"] = "challenging"
                analysis["recommendations"].append("Consider model optimization for edge deployment")
                
        return analysis
        
    def _determine_mode_from_name(self, model_name: str) -> str:
        """Determine performance mode from model name."""
        if 'training_inference' in model_name:
            return 'training_inference'
        elif 'training' in model_name:
            return 'training'
        else:
            return 'inference'
    
    def save_comprehensive_report(self, base_model_name: str) -> str:
        """Save comprehensive report combining all performance data."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create comprehensive report combining all modes
        comprehensive_report = {
            "model_name": f"{base_model_name}_comprehensive",
            "timestamp": timestamp,
            "system_info": self.system_info,
            "performance_summary": {},
            "detailed_measurements": {}
        }
        
        # Add data from all available modes
        for mode, data in self.all_performance_data.items():
            if data is not None:
                comprehensive_report["detailed_measurements"][mode] = data
                
                # Extract key metrics for summary
                if "real_performance_measurements" in data:
                    perf_data = data["real_performance_measurements"]
                    summary = {}
                    
                    if "inference_timing" in perf_data:
                        summary["average_latency_ms"] = perf_data["inference_timing"]["average_latency_ms"]
                        summary["total_inferences"] = perf_data["inference_timing"]["total_inferences_measured"]
                    
                    if "memory_usage" in perf_data:
                        summary["peak_ram_mb"] = perf_data["memory_usage"]["peak_ram_mb"]
                    
                    if "gpu_memory_usage" in perf_data:
                        summary["peak_gpu_allocated_mb"] = perf_data["gpu_memory_usage"]["peak_allocated_mb"]
                    
                    if "nvidia_ml_metrics" in perf_data:
                        summary["peak_gpu_utilization_percent"] = perf_data["nvidia_ml_metrics"]["peak_gpu_utilization_percent"]
                        summary["peak_temperature_celsius"] = perf_data["nvidia_ml_metrics"]["peak_temperature_celsius"]
                        summary["peak_power_usage_watts"] = perf_data["nvidia_ml_metrics"]["peak_power_usage_watts"]
                    
                    comprehensive_report["performance_summary"][mode] = summary
        
        # Save comprehensive report
        filename = f"comprehensive_performance_report_{base_model_name}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
            
        print(f"📊 Comprehensive performance report saved to: {filepath}")
        return filepath
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save performance report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"real_performance_report_{report['model_name']}_{timestamp}.json"
            
        filepath = os.path.join(self.log_dir, filename)
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        return filepath