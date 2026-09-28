"""
Simple efficiency metrics calculator for LLM time series models.
Automatically calculates and saves efficiency metrics for reviewer requirements.
"""

import os
import json
import time
import psutil
import torch
import logging
from typing import Dict, Any
from pathlib import Path


class EfficiencyCalculator:
    """Simple efficiency calculator that works with any PyTorch model."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize the efficiency calculator."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def calculate_efficiency_metrics(self, model, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive efficiency metrics for any model.
        
        Args:
            model: PyTorch model (can be wrapped or unwrapped)
            model_name: Name of the model for identification
            config: Configuration dictionary with model settings
            
        Returns:
            Dictionary containing all efficiency metrics
        """
        self.logger.info(f"Calculating efficiency metrics for {model_name}")
        
        # Get system information
        system_info = self._get_system_info()
        
        # Get model characteristics
        model_characteristics = self._get_model_characteristics(model, model_name)
        
        # Calculate theoretical performance estimates
        theoretical_metrics = self._calculate_theoretical_metrics(model_characteristics, system_info)
        
        # 🆕 Capture REAL performance metrics
        real_metrics = self._measure_real_performance(model, model_name, config)
        
        # Edge deployment analysis
        edge_analysis = self._analyze_edge_deployment(model_characteristics, theoretical_metrics)
        
        # Compile comprehensive report
        efficiency_report = {
            "model_name": model_name,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "system_info": system_info,
            "model_characteristics": model_characteristics,
            "theoretical_performance": theoretical_metrics,
            "real_performance_measurements": real_metrics,  # 🆕 Added real metrics
            "edge_deployment_analysis": edge_analysis,
            "configuration_used": config,
            "methodology": "Combined theoretical estimates and real performance measurements"
        }
        
        # Save report
        self._save_efficiency_report(efficiency_report, model_name)
        
        return efficiency_report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        system_info = {
            "cpu": {
                "brand": "Unknown",
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "frequency_ghz": 0.0
            },
            "memory": {
                "total_ram_gb": psutil.virtual_memory().total / (1024**3),
                "available_ram_gb": psutil.virtual_memory().available / (1024**3)
            },
            "gpu": {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": []
            },
            "python_version": f"{'.'.join(map(str, __import__('sys').version_info[:3]))}",
            "torch_version": torch.__version__
        }
        
        # Get CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                system_info["cpu"]["frequency_ghz"] = cpu_freq.current / 1000.0
        except:
            pass
            
        # Get CPU brand
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        system_info["cpu"]["brand"] = line.split(':')[1].strip()
                        break
        except:
            pass
        
        # Get GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    system_info["gpu"]["devices"].append({
                        "id": i,
                        "name": props.name,
                        "memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
                except:
                    pass
        
        return system_info
    
    def _get_model_characteristics(self, model, model_name: str) -> Dict[str, Any]:
        """Get model characteristics including size and parameter count."""
        characteristics = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "model_size_mb": 0.0,
            "model_dtype": "unknown",
            "device_location": "unknown",
            "model_architecture": "unknown"
        }
        
        try:
            # Handle different model types
            actual_model = self._extract_pytorch_model(model, model_name)
            
            if actual_model is not None:
                # Count parameters
                total_params = sum(p.numel() for p in actual_model.parameters())
                trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
                
                characteristics["total_parameters"] = total_params
                characteristics["trainable_parameters"] = trainable_params
                characteristics["model_architecture"] = type(actual_model).__name__
                
                # Estimate model size (parameters * bytes_per_parameter)
                # Assuming float32 (4 bytes) by default
                bytes_per_param = 4  # float32
                if hasattr(actual_model, 'dtype'):
                    if actual_model.dtype == torch.float16:
                        bytes_per_param = 2
                    elif actual_model.dtype == torch.bfloat16:
                        bytes_per_param = 2
                
                characteristics["model_size_mb"] = (total_params * bytes_per_param) / (1024**2)
                
                # Get model dtype
                for param in actual_model.parameters():
                    characteristics["model_dtype"] = str(param.dtype)
                    break
                
                # Get device location
                for param in actual_model.parameters():
                    characteristics["device_location"] = str(param.device)
                    break
            else:
                self.logger.warning(f"Could not extract PyTorch model from {type(model).__name__}")
                
        except Exception as e:
            self.logger.warning(f"Could not analyze model characteristics: {e}")
        
        return characteristics
    
    def _extract_pytorch_model(self, model, model_name: str):
        """Extract the actual PyTorch model from various wrappers."""
        
        # For TimeLLM wrapper
        if hasattr(model, 'llm_model'):
            # Check if it's a ChronosPipeline inside TimeLLM wrapper
            if hasattr(model.llm_model, 'model'):
                return model.llm_model.model  # ChronosPipeline.model
            else:
                return model.llm_model
        
        # For direct ChronosPipeline
        if hasattr(model, 'model'):
            return model.model  # ChronosPipeline.model
        
        # For direct ChronosPipeline with tokenizer
        if hasattr(model, 'tokenizer') and hasattr(model, 'model'):
            return model.model
        
        # Check for nested model attributes common in HuggingFace
        if hasattr(model, 'base_model'):
            return model.base_model
        
        # If it's already a PyTorch model
        if hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
            return model
        
        # Last resort - try to find any attribute that looks like a model
        for attr_name in ['model', 'base_model', 'transformer', 'bert', 'llm']:
            if hasattr(model, attr_name):
                attr_model = getattr(model, attr_name)
                if hasattr(attr_model, 'parameters') and callable(getattr(attr_model, 'parameters')):
                    return attr_model
        
        return None
    
    def _calculate_theoretical_metrics(self, model_char: Dict[str, Any], system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate theoretical performance metrics."""
        total_params = model_char["total_parameters"]
        model_size_mb = model_char["model_size_mb"]
        
        # Theoretical latency estimates (based on model complexity)
        # These are rough estimates based on parameter count and hardware
        base_latency_ms = max(1.0, total_params / 1_000_000)  # 1ms per million parameters baseline
        
        # Adjust for hardware
        cpu_cores = system_info["cpu"]["cores_logical"]
        cpu_freq_ghz = system_info["cpu"]["frequency_ghz"]
        
        cpu_latency_ms = base_latency_ms * (2.5 / max(cpu_freq_ghz, 1.0))  # Scale by CPU frequency
        gpu_latency_ms = base_latency_ms * 0.1 if system_info["gpu"]["cuda_available"] else cpu_latency_ms
        
        # Memory requirements
        base_memory_mb = model_size_mb * 1.5  # Model + intermediate activations
        peak_memory_mb = base_memory_mb * 2.0  # Peak during inference
        
        # Throughput estimates
        samples_per_second = max(1.0, 1000.0 / cpu_latency_ms)
        
        return {
            "estimated_cpu_latency_ms": round(cpu_latency_ms, 2),
            "estimated_gpu_latency_ms": round(gpu_latency_ms, 2),
            "estimated_memory_usage_mb": round(base_memory_mb, 2),
            "estimated_peak_memory_mb": round(peak_memory_mb, 2),
            "estimated_throughput_samples_per_sec": round(samples_per_second, 2),
            "model_size_on_disk_mb": round(model_size_mb, 2)
        }
    
    def _analyze_edge_deployment(self, model_char: Dict[str, Any], theoretical: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze edge deployment feasibility."""
        model_size_mb = model_char["model_size_mb"]
        memory_usage_mb = theoretical["estimated_memory_usage_mb"]
        
        # Define edge device categories
        edge_devices = {
            "raspberry_pi_4": {"ram_mb": 8192, "storage_mb": 32000, "gpu": False},
            "jetson_nano": {"ram_mb": 4096, "storage_mb": 16000, "gpu": True},
            "jetson_xavier": {"ram_mb": 32000, "storage_mb": 32000, "gpu": True},
            "intel_nuc": {"ram_mb": 16000, "storage_mb": 512000, "gpu": False}
        }
        
        feasible_devices = []
        for device, specs in edge_devices.items():
            if (model_size_mb < specs["storage_mb"] * 0.8 and  # 80% storage limit
                memory_usage_mb < specs["ram_mb"] * 0.6):      # 60% RAM limit
                feasible_devices.append(device)
        
        overall_feasibility = "feasible" if feasible_devices else "requires_optimization"
        
        recommendations = []
        if not feasible_devices:
            recommendations.append("Consider model quantization (INT8/FP16) to reduce memory footprint")
            recommendations.append("Use model compression techniques")
            recommendations.append("Consider model pruning for edge deployment")
        
        return {
            "overall_feasibility": overall_feasibility,
            "feasible_edge_devices": feasible_devices,
            "recommendations": recommendations,
            "constraints": {
                "model_size_mb": round(model_size_mb, 2),
                "memory_requirement_mb": round(memory_usage_mb, 2)
            }
        }
    
    def _save_efficiency_report(self, report: Dict[str, Any], model_name: str):
        """Save the efficiency report to JSON file."""
        filename = f"efficiency_report_{model_name}_{report['timestamp']}.json"
        filepath = self.log_dir / filename
        
        try:
            # Convert any non-serializable objects to strings
            report_serializable = self._make_json_serializable(report)
            
            with open(filepath, 'w') as f:
                json.dump(report_serializable, f, indent=2)
            
            self.logger.info(f"✅ Efficiency report saved to: {filepath}")
            
            # Also save a latest report for easy access
            latest_path = self.log_dir / f"efficiency_report_{model_name}_latest.json"
            with open(latest_path, 'w') as f:
                json.dump(report_serializable, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save efficiency report: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, 'dtype') and hasattr(obj.dtype, '__str__'):
            return str(obj.dtype)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool)):
            return str(obj)
        else:
            return obj
    
    def _measure_real_performance(self, model, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure actual performance metrics from the loaded model.
        
        Args:
            model: The loaded model (TimeLLM or Chronos wrapper)
            model_name: Name of the model
            config: Configuration dictionary
            
        Returns:
            Dictionary containing real performance measurements
        """
        real_metrics = {
            "measurement_status": "attempted",
            "memory_measurements": {},
            "timing_measurements": {},
            "model_loading_metrics": {},
            "errors": []
        }
        
        try:
            # Memory measurements
            real_metrics["memory_measurements"] = self._measure_memory_usage(model)
            
            # Model loading and initialization timing
            real_metrics["model_loading_metrics"] = self._measure_model_loading_time(model, config)
            
            # Basic inference timing (if possible)
            real_metrics["timing_measurements"] = self._measure_inference_timing(model, model_name, config)
            
            real_metrics["measurement_status"] = "successful"
            
        except Exception as e:
            real_metrics["errors"].append(f"Real performance measurement failed: {str(e)}")
            real_metrics["measurement_status"] = "failed"
            self.logger.warning(f"Could not measure real performance: {e}")
        
        return real_metrics
    
    def _measure_memory_usage(self, model) -> Dict[str, Any]:
        """Measure actual memory usage of the loaded model."""
        memory_metrics = {}
        
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_metrics["current_ram_usage_mb"] = memory_info.rss / (1024**2)
            memory_metrics["current_vram_usage_mb"] = memory_info.vms / (1024**2)
            
            # GPU memory if available
            if torch.cuda.is_available():
                memory_metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                memory_metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
                memory_metrics["gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            
            # Model-specific memory footprint
            if hasattr(model, 'llm_model'):
                actual_model = model.llm_model
            else:
                actual_model = model
                
            # Calculate model parameter memory
            model_memory_mb = 0
            for param in actual_model.parameters():
                model_memory_mb += param.nelement() * param.element_size()
            memory_metrics["model_parameters_memory_mb"] = model_memory_mb / (1024**2)
            
        except Exception as e:
            memory_metrics["error"] = str(e)
        
        return memory_metrics
    
    def _measure_model_loading_time(self, model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure model loading and initialization timing."""
        loading_metrics = {}
        
        try:
            # This measures the time it took to get to this point
            # We can't re-measure loading time, but we can provide context
            loading_metrics["note"] = "Model already loaded - timing captured during initialization"
            loading_metrics["model_type"] = type(model).__name__
            
            # Get model complexity indicators
            if hasattr(model, 'llm_model'):
                actual_model = model.llm_model
            else:
                actual_model = model
                
            total_params = sum(p.numel() for p in actual_model.parameters())
            loading_metrics["total_parameters"] = total_params
            loading_metrics["estimated_loading_time_seconds"] = max(1.0, total_params / 10_000_000)  # Rough estimate
            
        except Exception as e:
            loading_metrics["error"] = str(e)
        
        return loading_metrics
    
    def _measure_inference_timing(self, model, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure actual inference timing with safe dummy data."""
        timing_metrics = {}
        
        try:
            # Create very simple dummy data based on model type
            if model_name == "time_llm":
                timing_metrics.update(self._measure_timellm_inference(model, config))
            elif model_name == "chronos":
                timing_metrics.update(self._measure_chronos_inference(model, config))
            elif model_name in ("time_llm_ecg_classifier", "distillation_ecg_classifier"):
                timing_metrics.update(self._measure_ecg_inference(model, config))
            else:
                timing_metrics["note"] = f"Inference timing not implemented for {model_name}"
                
        except Exception as e:
            timing_metrics["error"] = str(e)
            timing_metrics["note"] = "Inference timing measurement failed - using theoretical estimates"
        
        return timing_metrics
    
    def _measure_timellm_inference(self, model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Safe inference timing for TimeLLM."""
        timing_results = {}
        
        try:
            # Get the actual PyTorch model
            if hasattr(model, 'llm_model'):
                pytorch_model = model.llm_model
            else:
                pytorch_model = model
            
            # Set model to evaluation mode
            pytorch_model.eval()
            
            # Create minimal dummy data
            llm_settings = config.get('llm_settings', {})
            seq_len = llm_settings.get('sequence_length', 6)
            enc_in = llm_settings.get('enc_in', 1)
            
            # Create minimal batch
            batch_size = 1
            dummy_x = torch.randn(batch_size, seq_len, enc_in)
            dummy_x_mark = torch.randn(batch_size, seq_len, 4)  # Time features
            dummy_y = torch.randn(batch_size, seq_len, enc_in)
            dummy_y_mark = torch.randn(batch_size, seq_len, 4)
            
            # Move to device if needed
            device = next(pytorch_model.parameters()).device
            dummy_x = dummy_x.to(device)
            dummy_x_mark = dummy_x_mark.to(device)
            dummy_y = dummy_y.to(device)
            dummy_y_mark = dummy_y_mark.to(device)
            
            # Warm-up run with dtype handling
            with torch.no_grad():
                try:
                    # Check if model uses BFloat16 and convert if needed
                    model_dtype = next(pytorch_model.parameters()).dtype
                    if model_dtype == torch.bfloat16:
                        # Convert inputs to BFloat16 to match model
                        dummy_x = dummy_x.to(dtype=torch.bfloat16)
                        dummy_y = dummy_y.to(dtype=torch.bfloat16)
                    
                    _ = pytorch_model(dummy_x, dummy_x_mark, dummy_y, dummy_y_mark)
                except Exception as e:
                    timing_results["warmup_error"] = str(e)
                    return timing_results
            
            # Measure actual timing with proper dtype handling
            num_runs = 5
            times = []
            
            # Check model dtype for consistency
            model_dtype = next(pytorch_model.parameters()).dtype
            if model_dtype == torch.bfloat16:
                # Ensure inputs match model dtype
                dummy_x = dummy_x.to(dtype=torch.bfloat16)
                dummy_y = dummy_y.to(dtype=torch.bfloat16)
            
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    _ = pytorch_model(dummy_x, dummy_x_mark, dummy_y, dummy_y_mark)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            timing_results["inference_times_ms"] = times
            timing_results["average_inference_time_ms"] = sum(times) / len(times)
            timing_results["min_inference_time_ms"] = min(times)
            timing_results["max_inference_time_ms"] = max(times)
            timing_results["batch_size"] = batch_size
            timing_results["sequence_length"] = seq_len
            timing_results["measurement_runs"] = num_runs
            timing_results["model_dtype"] = str(model_dtype)
            timing_results["device"] = str(device)
            
        except Exception as e:
            timing_results["error"] = str(e)
            timing_results["note"] = "TimeLLM inference timing failed"
        
        return timing_results
    
    def _measure_ecg_inference(self, model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Safe inference timing for the ECG (MIT-BIH) Time-LLM classifier.

        Unlike BG's TimeLLM (which takes x, x_mark, dec_inp, y_mark), the ECG
        classifier's forward signature is just `forward(x)`, so this mirrors
        `_measure_timellm_inference` with a simpler dummy batch.
        """
        timing_results = {}

        try:
            # Prefer whichever attribute actually accepts a raw [B, T, C]
            # tensor: the classifier itself (has `.patch_embedding`) or, if
            # `model.llm_model` is that classifier (as for the plain ECG
            # wrapper), use that instead.
            if hasattr(model, "patch_embedding"):
                pytorch_model = model
            elif hasattr(model, "llm_model") and hasattr(model.llm_model, "patch_embedding"):
                pytorch_model = model.llm_model
            else:
                pytorch_model = model

            pytorch_model.eval()

            llm_settings = config.get('llm_settings', {})
            seq_len = llm_settings.get('sequence_length', 256)
            enc_in = llm_settings.get('enc_in', 1)

            batch_size = 1
            dummy_x = torch.randn(batch_size, seq_len, enc_in)

            device = next(pytorch_model.parameters()).device
            dummy_x = dummy_x.to(device)

            model_dtype = next(pytorch_model.parameters()).dtype
            if model_dtype == torch.bfloat16:
                dummy_x = dummy_x.to(dtype=torch.bfloat16)

            # Warm-up run
            with torch.no_grad():
                try:
                    _ = pytorch_model(dummy_x)
                except Exception as e:
                    timing_results["warmup_error"] = str(e)
                    return timing_results

            # Measure actual timing
            num_runs = 5
            times = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                with torch.no_grad():
                    _ = pytorch_model(dummy_x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms

            timing_results["inference_times_ms"] = times
            timing_results["average_inference_time_ms"] = sum(times) / len(times)
            timing_results["min_inference_time_ms"] = min(times)
            timing_results["max_inference_time_ms"] = max(times)
            timing_results["batch_size"] = batch_size
            timing_results["sequence_length"] = seq_len
            timing_results["measurement_runs"] = num_runs
            timing_results["model_dtype"] = str(model_dtype)
            timing_results["device"] = str(device)

        except Exception as e:
            timing_results["error"] = str(e)
            timing_results["note"] = "ECG classifier inference timing failed"

        return timing_results

    def _measure_chronos_inference(self, model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Safe inference timing for Chronos."""
        timing_results = {}
        
        try:
            # Extract ChronosPipeline
            chronos_pipeline = None
            if hasattr(model, 'llm_model'):
                chronos_pipeline = model.llm_model  # ChronosLLM wrapper
            elif hasattr(model, 'predict'):
                chronos_pipeline = model  # Direct ChronosPipeline
            
            if chronos_pipeline is not None:
                timing_results["model_type"] = "chronos_pipeline"
                timing_results["pipeline_class"] = type(chronos_pipeline).__name__
                
                # Create minimal dummy time series data for timing
                context_length = 64  # Standard context length
                batch_size = 1
                prediction_length = 6  # Small prediction length for timing
                
                # Generate dummy time series (single univariate series)
                dummy_context = torch.randn(batch_size, context_length)
                
                # Warm-up run
                try:
                    with torch.no_grad():
                        _ = chronos_pipeline.predict(
                            context=dummy_context,
                            prediction_length=prediction_length,
                            num_samples=1,
                            limit_prediction_length=True
                        )
                    timing_results["warmup_status"] = "successful"
                except Exception as e:
                    timing_results["warmup_error"] = str(e)
                    return timing_results
                
                # Measure actual timing
                num_runs = 3  # Fewer runs for Chronos to avoid long timing
                times = []
                
                for _ in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        _ = chronos_pipeline.predict(
                            context=dummy_context,
                            prediction_length=prediction_length,
                            num_samples=1,
                            limit_prediction_length=True
                        )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                timing_results["inference_times_ms"] = times
                timing_results["average_inference_time_ms"] = sum(times) / len(times)
                timing_results["min_inference_time_ms"] = min(times)
                timing_results["max_inference_time_ms"] = max(times)
                timing_results["context_length"] = context_length
                timing_results["prediction_length"] = prediction_length
                timing_results["batch_size"] = batch_size
                timing_results["measurement_runs"] = num_runs
                timing_results["inference_method"] = "chronos_pipeline_predict"
                
            else:
                timing_results["error"] = "Could not access Chronos pipeline for timing"
                timing_results["available_attributes"] = [attr for attr in dir(model) if not attr.startswith('_')]
                
        except Exception as e:
            timing_results["error"] = str(e)
            timing_results["note"] = "Chronos inference timing failed"
        
        return timing_results


def auto_calculate_efficiency(model, model_name: str, config: Dict[str, Any], log_dir: str = "./logs"):
    """
    Convenience function to automatically calculate and save efficiency metrics.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        config: Configuration dictionary
        log_dir: Directory to save results
    """
    calculator = EfficiencyCalculator(log_dir)
    return calculator.calculate_efficiency_metrics(model, model_name, config)