#!/usr/bin/env python3
"""
Per-Patient Inference Script
=============================

After training on ALL patients combined, run inference separately on each patient
to analyze individual patient performance.

Usage:
    python distillation/scripts/run_per_patient_inference.py \
        --checkpoint-dir ./results/teacher \
        --model-name bert \
        --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
        --output-dir ./results/per_patient_inference
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root


class PerPatientInferenceRunner:
    """Run inference on individual patients using all-patients checkpoint."""
    
    def __init__(self, checkpoint_dir, model_name, dataset_name="ohiot1dm", output_dir=None, llm_model=None, llm_layers=None, llm_dim=None):
        """Initialize the inference runner.
        
        Args:
            checkpoint_dir: Directory containing trained model checkpoint
            model_name: Name of the model (bert, tinybert, etc.)
            dataset_name: Dataset name (ohiot1dm, d1namo)
            output_dir: Directory to save per-patient inference results
            llm_model: LLM model type (BERT, TinyBERT, etc.) - auto-detected if None
            llm_layers: Number of LLM layers - auto-detected if None
            llm_dim: LLM dimension - auto-detected if None
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_path = f"./data/{dataset_name}"
        
        # Auto-detect model config from model_name if not provided
        if llm_model is None:
            self.llm_model, self.llm_layers, self.llm_dim = self._detect_model_config(model_name)
        else:
            self.llm_model = llm_model
            self.llm_layers = llm_layers
            self.llm_dim = llm_dim
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.checkpoint_dir / "per_patient_inference"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Per-Patient Inference Runner")
        print(f"   Checkpoint: {self.checkpoint_dir}")
        print(f"   Model: {model_name}")
        print(f"   LLM: {self.llm_model} (layers={self.llm_layers}, dim={self.llm_dim})")
        print(f"   Output: {self.output_dir}")
    
    def _detect_model_config(self, model_name):
        """Detect model configuration from model name."""
        # Model configurations (sync with train_students.py)
        model_configs = {
            "tinybert": ("TinyBERT", 4, 312),
            "distilbert": ("DistilBERT", 6, 768),
            "bert-tiny": ("BERT-tiny", 2, 128),
            "bert-mini": ("BERT", 4, 256),
            "bert-small": ("BERT", 4, 512),
            "bert-medium": ("BERT", 8, 512),
            "minilm": ("MiniLM", 6, 384),
            "mobilebert": ("MobileBERT", 24, 512),
            "albert": ("ALBERT", 12, 768),
            "bert": ("BERT", 12, 768),
            "bert-base-uncased": ("BERT", 12, 768),
            "opt-125m": ("OPT-125M", 12, 768),
        }
        
        # Try to extract model key from model_name
        # Handle names like "teacher_bert", "student_baseline_prajjwal1/bert-tiny"
        for key in model_configs:
            if key in model_name.lower():
                return model_configs[key]
        
        # Default to BERT if not found
        print(f"⚠️  Warning: Could not detect model config for '{model_name}', defaulting to BERT")
        return ("BERT", 12, 768)
    
    def find_checkpoint(self):
        """Find the trained model checkpoint."""
        # Look for checkpoint in logs directory - search recursively
        checkpoint_patterns = [
            "**/logs/logs_*/checkpoints/checkpoint.pth",
            "**/checkpoints/checkpoint.pth",
            "checkpoint.pth",
        ]
        
        for pattern in checkpoint_patterns:
            matching = list(self.checkpoint_dir.glob(pattern))
            if matching:
                # Return the most recent checkpoint if multiple found
                return max(matching, key=lambda p: p.stat().st_mtime)
        
        raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")
    
    def generate_inference_config(self, patient_id, checkpoint_path):
        """Generate config for per-patient inference.
        
        Args:
            patient_id: Patient ID to run inference on
            checkpoint_path: Path to trained model checkpoint
            
        Returns:
            Path to generated config file
        """
        # Data paths for this specific patient
        data_dir = f"{self.data_path}/raw_standardized"
        test_file = f"{data_dir}/{patient_id}-ws-testing.csv"
        prompt_file = f"{data_dir}/t1dm_prompt.txt"
        
        # Output directory for this patient
        patient_output = self.output_dir / f"patient_{patient_id}"
        patient_output.mkdir(parents=True, exist_ok=True)
        log_dir = patient_output / "logs"
        
        # Generate config
        config_content = f'''# Per-Patient Inference Config
# Model: {self.model_name}
# Patient: {patient_id}
# Checkpoint: {checkpoint_path}

run.log_dir = "{log_dir}"

run.data_settings = {{
    'path_to_train_data': '',  # Not used for inference-only
    'path_to_test_data': '{test_file}',
    'input_features': ['target'],
    'labels': ['target'],
    'prompt_path': '{prompt_file}',
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'frequency': '5min',
    'percent': 100,
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'inference',
    'method': 'time_llm',
    'llm_model': '{self.llm_model}',
    'llm_layers': {self.llm_layers},
    'llm_dim': {self.llm_dim},
    'num_workers': 1,
    'torch_dtype': 'float32',
    'model_id': '{self.model_name}_patient_{patient_id}',
    'model_comment': '{self.model_name}_patient_{patient_id}_inference',
    'sequence_length': 6,
    'context_length': 6,
    'prediction_length': 9,
    'patch_len': 6,
    'stride': 8,
    'prediction_batch_size': 64,
    'features': 'S',
    'd_model': 32,
    'd_ff': 32,
    'factor': 1,
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'e_layers': 2,
    'd_layers': 1,
    'n_heads': 8,
    'dropout': 0.1,
    'moving_avg': 25,
    'activation': 'gelu',
    'embed': 'timeF',
    'patience': 10,
    'lradj': 'COS',
    'des': 'per_patient_inference',
    'prompt_domain': 0,
    'timeenc': 0,
    'freq': '5min',
    'use_amp': False,
    'restore_from_checkpoint': True,
    'restore_checkpoint_path': '{checkpoint_path}',
    'eval_metrics': ['rmse', 'mae', 'mape', 'mse'],
    'seed': 42
}}
'''
        
        # Save config
        config_path = patient_output / f"inference_config_{patient_id}.gin"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path
    
    def run_inference(self, patient_id, checkpoint_path, dry_run=False):
        """Run inference for a single patient.
        
        Args:
            patient_id: Patient ID
            checkpoint_path: Path to trained checkpoint
            dry_run: If True, only generate config without running
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"🔍 Running inference for Patient {patient_id}")
        print(f"{'='*60}")
        
        # Generate config
        config_path = self.generate_inference_config(patient_id, checkpoint_path)
        print(f"✅ Config generated: {config_path}")
        
        if dry_run:
            print("🔍 DRY RUN - Config generated but not executed")
            return {"status": "dry_run", "config": str(config_path)}
        
        # Run inference
        cmd = ["python", "main.py", "--config_path", str(config_path), "--log_level", "INFO"]
        
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"Running inference...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(get_project_root()),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Inference completed for patient {patient_id}")
                
                # Try to extract metrics from results
                patient_output = self.output_dir / f"patient_{patient_id}"
                metrics = self.extract_metrics(patient_output)
                
                return {
                    "status": "success",
                    "patient_id": patient_id,
                    "config": str(config_path),
                    "metrics": metrics
                }
            else:
                print(f"❌ Inference failed for patient {patient_id}")
                print(f"   Error: {result.stderr[:500]}")
                return {
                    "status": "failed",
                    "patient_id": patient_id,
                    "error": result.stderr[:500]
                }
        
        except subprocess.TimeoutExpired:
            print(f"⏰ Inference timeout for patient {patient_id}")
            return {
                "status": "timeout",
                "patient_id": patient_id
            }
        except Exception as e:
            print(f"💥 Exception during inference for patient {patient_id}: {e}")
            return {
                "status": "exception",
                "patient_id": patient_id,
                "error": str(e)
            }
    
    def extract_metrics(self, patient_output_dir):
        """Extract metrics from inference results.
        
        Args:
            patient_output_dir: Directory containing patient inference results
            
        Returns:
            Dictionary of metrics or None
        """
        # Look for metrics in various possible locations
        metrics_files = [
            patient_output_dir / "logs" / "logs_*" / "inference_results.csv",
            patient_output_dir / "logs" / "inference_results.csv",
            patient_output_dir / "logs" / "logs_*" / "*performance*.json",
        ]
        
        for pattern in metrics_files:
            matching = list(patient_output_dir.glob(str(pattern).replace(str(patient_output_dir) + "/", "")))
            if matching:
                try:
                    if str(matching[0]).endswith('.json'):
                        with open(matching[0], 'r') as f:
                            data = json.load(f)
                            if 'evaluation_metrics' in data:
                                return data['evaluation_metrics']
                    elif str(matching[0]).endswith('.csv'):
                        import pandas as pd
                        import numpy as np
                        df = pd.read_csv(matching[0])
                        if 'ground_truth' in df.columns and 'predictions' in df.columns:
                            # Calculate metrics
                            y_true_list = []
                            y_pred_list = []
                            
                            for _, row in df.iterrows():
                                true_vals = [float(x.strip()) for x in str(row['ground_truth']).split(',')]
                                pred_vals = [float(x.strip()) for x in str(row['predictions']).split(',')]
                                y_true_list.extend(true_vals)
                                y_pred_list.extend(pred_vals)
                            
                            y_true = np.array(y_true_list)
                            y_pred = np.array(y_pred_list)
                            
                            return {
                                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                                "mae": float(np.mean(np.abs(y_true - y_pred))),
                                "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
                            }
                except Exception as e:
                    print(f"⚠️  Could not extract metrics: {e}")
        
        return None
    
    def run_all_patients(self, patient_ids, dry_run=False):
        """Run inference on all specified patients.
        
        Args:
            patient_ids: List of patient IDs or comma-separated string
            dry_run: If True, only generate configs
            
        Returns:
            Dictionary of results per patient
        """
        # Parse patient IDs
        if isinstance(patient_ids, str):
            patient_list = [p.strip() for p in patient_ids.split(',')]
        else:
            patient_list = patient_ids
        
        print(f"\n{'='*60}")
        print(f"🚀 Running Per-Patient Inference")
        print(f"{'='*60}")
        print(f"Total patients: {len(patient_list)}")
        print(f"Patients: {', '.join(patient_list)}")
        print(f"{'='*60}\n")
        
        # Find checkpoint
        try:
            checkpoint_path = self.find_checkpoint()
            print(f"✅ Found checkpoint: {checkpoint_path}\n")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return {}
        
        # Run inference for each patient
        results = {}
        for patient_id in patient_list:
            result = self.run_inference(patient_id, checkpoint_path, dry_run)
            results[patient_id] = result
        
        # Save summary
        self.save_summary(results)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def save_summary(self, results):
        """Save inference summary to JSON and CSV.
        
        Args:
            results: Dictionary of results per patient
        """
        summary_file = self.output_dir / "per_patient_inference_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Summary saved: {summary_file}")
        
        # Also create CSV for easy viewing
        try:
            import pandas as pd
            
            rows = []
            for patient_id, result in results.items():
                row = {
                    'patient_id': patient_id,
                    'status': result.get('status', 'unknown')
                }
                
                if result.get('metrics'):
                    row.update(result['metrics'])
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            csv_file = self.output_dir / "per_patient_inference_summary.csv"
            df.to_csv(csv_file, index=False)
            print(f"✅ CSV saved: {csv_file}")
        except Exception as e:
            print(f"⚠️  Could not create CSV summary: {e}")
    
    def print_summary(self, results):
        """Print summary of per-patient inference results.
        
        Args:
            results: Dictionary of results per patient
        """
        print(f"\n{'='*60}")
        print(f"📊 PER-PATIENT INFERENCE SUMMARY")
        print(f"{'='*60}")
        
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        failed_count = sum(1 for r in results.values() if r.get('status') == 'failed')
        
        print(f"Total patients: {len(results)}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print()
        
        print(f"{'Patient':<15} {'Status':<12} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
        print(f"{'-'*60}")
        
        for patient_id, result in sorted(results.items()):
            status = result.get('status', 'unknown')
            metrics = result.get('metrics', {})
            
            rmse = f"{metrics.get('rmse', 0):.3f}" if metrics else "N/A"
            mae = f"{metrics.get('mae', 0):.3f}" if metrics else "N/A"
            mape = f"{metrics.get('mape', 0):.2f}%" if metrics else "N/A"
            
            print(f"{patient_id:<15} {status:<12} {rmse:<10} {mae:<10} {mape:<10}")
        
        print(f"{'='*60}")
        print(f"Results saved in: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run per-patient inference using all-patients checkpoint")
    
    parser.add_argument("--checkpoint-dir", required=True,
                       help="Directory containing trained model checkpoint")
    parser.add_argument("--model-name", required=True,
                       help="Name of the model (bert, tinybert, etc.)")
    parser.add_argument("--patients", 
                       default="540,544,552,559,563,567,570,575,584,588,591,596",
                       help="Comma-separated patient IDs")
    parser.add_argument("--dataset", default="ohiot1dm",
                       help="Dataset name")
    parser.add_argument("--output-dir",
                       help="Output directory for per-patient results")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate configs but don't run inference")
    
    args = parser.parse_args()
    
    runner = PerPatientInferenceRunner(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )
    
    results = runner.run_all_patients(args.patients, args.dry_run)
    
    # Exit with success if at least one patient succeeded
    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    sys.exit(0 if success_count > 0 else 1)


if __name__ == "__main__":
    main()
