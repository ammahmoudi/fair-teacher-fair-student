#!/usr/bin/env python3
"""
Automated Student Baseline Training Script for Time-LLM Models
Trains student models (TinyBERT, DistilBERT) as independent baselines for comparison.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_configs_path


class StudentTrainer:
    """Class to handle automated training of student baseline models."""
    
    def __init__(self, base_dir=None, output_dir=None, config_dir=None, dataset_name="ohiot1dm", seed=238822):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.data_path = f"./data/{dataset_name}"  # Auto-detect data path
        self.seed = seed
        
        # Model name mappings from HuggingFace names to internal config names
        self.model_name_mapping = {
            # BERT variants
            "bert": "bert",
            "bert-base-uncased": "bert",
            "bert-base-cased": "bert", 
            # DistilBERT variants
            "distilbert": "distilbert",
            "distilbert-base-uncased": "distilbert",
            "distilbert-base-cased": "distilbert",
            # TinyBERT and small models (corrected mappings)
            "tinybert": "tinybert",
            "huawei-noah/TinyBERT_General_4L_312D": "tinybert",
            "prajjwal1/bert-tiny": "bert-tiny",  # Fixed: maps to bert-tiny not tinybert
            "prajjwal1/bert-mini": "bert-mini",  # Fixed: maps to bert-mini not distilbert
            "prajjwal1/bert-small": "bert-small",  # Fixed: maps to bert-small not distilbert
            "prajjwal1/bert-medium": "bert-medium",  # Fixed: maps to bert-medium not distilbert
            # Additional models from Time-LLM
            "gpt2": "gpt2",
            "openai-community/gpt2": "gpt2",
            "llama": "llama",
            "huggyllama/llama-7b": "llama",
            "minilm": "minilm",
            "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large": "minilm",
            "mobilebert": "mobilebert",
            "google/mobilebert-uncased": "mobilebert",
            "albert": "albert",
            "albert/albert-base-v2": "albert",
            "opt-125m": "opt-125m",
            "facebook/opt-125m": "opt-125m",
        }
        
        
        if output_dir:
            self.results_dir = Path(output_dir)
        else:
            self.results_dir = self.base_dir / "results" / "student_baselines"
        
        if config_dir:
            self.configs_dir = Path(config_dir)
        else:
            self.configs_dir = get_configs_path()
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define student model configurations
        self.student_models = {
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "model_comment": "student_TinyBERT_312_6_6_9_6"
            },
            "distilbert": {
                "llm_model": "DistilBERT", 
                "llm_layers": 6,
                "llm_dim": 768,
                "model_comment": "student_DistilBERT_768_6_6_9_6"
            },
            "bert-tiny": {
                "llm_model": "BERT-tiny",
                "llm_layers": 2,
                "llm_dim": 128,
                "model_comment": "student_BERT-tiny_128_6_6_9_6"
            },
            "bert-mini": {
                "llm_model": "BERT",
                "llm_layers": 4,
                "llm_dim": 256,
                "model_comment": "student_BERT_256_6_6_9_6"
            },
            "bert-small": {
                "llm_model": "BERT",
                "llm_layers": 4,
                "llm_dim": 512,
                "model_comment": "student_BERT_512_6_6_9_6"
            },
            "bert-medium": {
                "llm_model": "BERT",
                "llm_layers": 8,
                "llm_dim": 512,
                "model_comment": "student_BERT_512_6_6_9_6"
            },
            "minilm": {
                "llm_model": "MiniLM",
                "llm_layers": 6,
                "llm_dim": 384,
                "model_comment": "student_MiniLM_384_6_6_9_6"
            },
            "mobilebert": {
                "llm_model": "MobileBERT",
                "llm_layers": 24,
                "llm_dim": 512,
                "model_comment": "student_MobileBERT_512_6_6_9_6"
            },
            "albert": {
                "llm_model": "ALBERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_ALBERT_768_6_6_9_6"
            },
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_BERT_768_6_6_9_6"
            },
            "gpt2": {
                "llm_model": "GPT2",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_GPT2_768_6_6_9_6"
            },
            "llama": {
                "llm_model": "LLAMA",
                "llm_layers": 32,
                "llm_dim": 4096,
                "model_comment": "student_LLAMA_4096_6_6_9_6"
            },
            "opt-125m": {
                "llm_model": "OPT-125M",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_OPT-125M_768_6_6_9_6"
            }
        }

    def generate_config(self, model_name, dataset, epochs, lr=0.001, batch_size=32):
        """Generate gin config for student model training."""
        model_config = self.student_models[model_name]
        
        # Build data paths using flexible parameters
        # Check if using all_patients combined data
        if dataset == "all_patients":
            data_dir = f"{self.data_path}/all_patients_combined"
            train_file = f"{data_dir}/all_patients_training.csv"
            test_file = f"{data_dir}/all_patients_testing.csv"
            prompt_file = f"{self.data_path}/raw_standardized/t1dm_prompt.txt"
        else:
            data_dir = f"{self.data_path}/raw_standardized"
            train_file = f"{data_dir}/{dataset}-ws-training.csv"
            test_file = f"{data_dir}/{dataset}-ws-testing.csv"
            prompt_file = f"{data_dir}/t1dm_prompt.txt"
        
        log_dir = f"{self.results_dir}/{model_name}_{dataset}_{epochs}epochs/logs"
        
        config_content = f'''run.log_dir = "{log_dir}"
run.data_settings = {{
    'path_to_train_data': '{train_file}',
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
    'mode': 'training+inference',
    'method': 'time_llm',
    'llm_model': '{model_config["llm_model"]}',
    'llm_layers': {model_config["llm_layers"]},
    'llm_dim': {model_config["llm_dim"]},
    'num_workers': 1,
    'torch_dtype': 'float32',
    'model_id': 'student_baseline',
    'sequence_length': 6,
    'context_length': 6,
    'prediction_length': 9,
    'patch_len': 6,
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': {batch_size},
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
    'learning_rate': {lr},
    'lradj': 'COS',
    'des': 'student_baseline',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': {self.seed},
    'restore_from_checkpoint': False,
    'train_epochs': {epochs},
    'model_comment': '{model_config["model_comment"]}'
}}
'''
        return config_content

    def save_config(self, config_content, model_name, dataset, epochs):
        """Save config content to a gin file."""
        # Sanitize model name for filename
        safe_model_name = self.sanitize_filename(model_name)
        config_filename = f"config_student_{safe_model_name}_{dataset}_{epochs}epochs.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def normalize_model_name(self, model_name):
        """Normalize HuggingFace model names to internal config names."""
        if model_name in self.model_name_mapping:
            return self.model_name_mapping[model_name]
        else:
            # If not in mapping, try to extract base model name
            model_lower = model_name.lower()
            if "tiny" in model_lower:
                return "tinybert"
            elif "distilbert" in model_lower:
                return "distilbert"
            elif "bert" in model_lower:
                return "bert"
            else:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.model_name_mapping.keys())}")

    def sanitize_filename(self, name):
        """Sanitize model names for safe file/directory usage."""
        return name.replace('/', '_').replace('\\', '_')

    def train_model(self, model_name, dataset="570", epochs=1, lr=0.001, batch_size=32, dry_run=False):
        """Train a specific student baseline model."""
        # Normalize the model name
        normalized_name = self.normalize_model_name(model_name)
        print(f"\\n{'='*60}")
        print(f"Training Student Baseline: {model_name} ({normalized_name.upper()})")
        print(f"Dataset: {dataset}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        # Generate and save config using normalized name
        config_content = self.generate_config(normalized_name, dataset, epochs, lr, batch_size)
        config_path = self.save_config(config_content, model_name, dataset, epochs)
        
        print(f"✓ Config generated: {config_path}")
        
        if dry_run:
            print("DRY RUN: Would execute training command")
            return config_path
        
        # Execute training command
        cmd = [
            "python", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Change to base directory for execution
        original_dir = os.getcwd()
        try:
            os.chdir(self.base_dir)
            
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully trained {model_name} student baseline")
                
                # Create training summary
                self.create_training_summary(model_name, dataset, epochs, config_path, lr, normalized_name)
                
                return config_path
            else:
                print(f"✗ Training failed with return code: {result.returncode}")
                return None
        
        except Exception as e:
            print(f"✗ Error during training: {str(e)}")
            return None
        finally:
            os.chdir(original_dir)

    def create_training_summary(self, model_name, dataset, epochs, config_path, lr, normalized_name):
        """Create a comprehensive summary of the student baseline training run."""
        # Find the latest model directory and results
        model_dir = None
        performance_metrics = None
        training_info = None
        
        # Look for model directory pattern: {model}_{dataset}_{epochs}epochs
        # Use normalized model name for directory pattern matching (as that's what's actually created)
        model_pattern = f"{normalized_name}_{dataset}_{epochs}epochs"
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.startswith(model_pattern):
                model_dir = item
                break
        
        if model_dir:
            # Look for logs directory
            logs_dirs = list(model_dir.glob("logs/logs_*"))
            if logs_dirs:
                latest_log_dir = max(logs_dirs, key=lambda x: x.name)
                
                # Try to extract performance metrics from various sources
                # 1. Look for performance reports
                perf_reports = list(latest_log_dir.glob("*performance_report*.json"))
                if perf_reports:
                    try:
                        with open(perf_reports[0], 'r') as f:
                            perf_data = json.load(f)
                            if 'evaluation_metrics' in perf_data:
                                performance_metrics = perf_data['evaluation_metrics']
                    except:
                        pass
                
                # 2. Look for inference results
                if not performance_metrics:
                    inference_file = latest_log_dir / "inference_results.csv"
                    if inference_file.exists():
                        try:
                            import pandas as pd
                            import numpy as np
                            df = pd.read_csv(inference_file)
                            if 'ground_truth' in df.columns and 'predictions' in df.columns:
                                # Calculate basic metrics from flattened arrays
                                y_true_list = []
                                y_pred_list = []
                                
                                for _, row in df.iterrows():
                                    # Parse comma-separated values
                                    true_vals = [float(x.strip()) for x in str(row['ground_truth']).split(',')]
                                    pred_vals = [float(x.strip()) for x in str(row['predictions']).split(',')]
                                    y_true_list.extend(true_vals)
                                    y_pred_list.extend(pred_vals)
                                
                                y_true = np.array(y_true_list)
                                y_pred = np.array(y_pred_list)
                                
                                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                                mae = np.mean(np.abs(y_true - y_pred))
                                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                                
                                performance_metrics = {
                                    "rmse": float(rmse),
                                    "mae": float(mae), 
                                    "mape": float(mape)
                                }
                        except Exception as e:
                            print(f"Warning: Could not calculate metrics from inference results: {e}")
                
                # Extract training information
                training_info = {
                    "model_directory": str(model_dir),
                    "logs_directory": str(latest_log_dir),
                    "checkpoint_available": (latest_log_dir / "checkpoints" / "checkpoint.pth").exists()
                }
        
        summary = {
            "model_name": model_name,
            "model_type": "student_baseline",
            "dataset": dataset,
            "epochs": epochs,
            "learning_rate": lr,
            "config_path": str(config_path),
            "model_config": self.student_models[normalized_name],
            "training_info": training_info,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Sanitize model name for filename
        safe_model_name = self.sanitize_filename(model_name)
        summary_file = self.results_dir / f"{safe_model_name}_{dataset}_{epochs}epochs_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save with standardized name for pipeline CSV logging
        standard_summary_file = self.results_dir / "student_baseline_summary.json"
        with open(standard_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Enhanced student baseline summary saved: {summary_file}")
        print(f"✓ Standardized summary saved: {standard_summary_file}")
        if performance_metrics:
            print(f"  📊 Baseline Performance: RMSE={performance_metrics.get('rmse', 'N/A'):.3f}, MAE={performance_metrics.get('mae', 'N/A'):.3f}, MAPE={performance_metrics.get('mape', 'N/A'):.3f}%")


def main():
    parser = argparse.ArgumentParser(description="Train student baseline models for knowledge distillation")
    
    # Model and data configuration
    parser.add_argument("--model", choices=["bert", "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", 
                                           "distilbert", "distilbert-base-uncased", "distilbert-base-cased", 
                                           "tinybert", "huawei-noah/TinyBERT_General_4L_312D",
                                           "prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small", "prajjwal1/bert-medium",
                                           "minilm", "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large",
                                           "mobilebert", "google/mobilebert-uncased", 
                                           "albert", "albert/albert-base-v2",
                                           "opt-125m", "facebook/opt-125m"], 
                       required=True, help="Student model to train")
    parser.add_argument("--patients", default="570", help="Patient IDs (comma-separated or single)")
    parser.add_argument("--all-patients", action="store_true", help="Train on ALL patients combined data (overrides --patients)")
    parser.add_argument("--dataset", default="ohiot1dm", help="Dataset name (ohiot1dm, d1namo)")
    parser.add_argument("--seed", type=int, default=238822, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    # Directory configuration  
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--config-dir", help="Directory to save configs")
    
    # Options
    parser.add_argument("--dry-run", action="store_true", help="Generate config but don't train")
    
    args = parser.parse_args()
    
    trainer = StudentTrainer(
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        dataset_name=args.dataset,
        seed=args.seed
    )
    
    # If --all-patients flag is set, use "all_patients" as the patient ID
    if args.all_patients:
        print("🌍 Training on ALL PATIENTS COMBINED")
        patient_id = "all_patients"
    else:
        patient_id = args.patients
    
    # Train the specified model
    config_path = trainer.train_model(
        args.model, 
        patient_id, 
        args.epochs, 
        args.lr, 
        args.batch_size,
        args.dry_run
    )
    
    if config_path:
        print(f"\\n✅ Student baseline training completed successfully!")
        print(f"📁 Config: {config_path}")
        print(f"📁 Results: {trainer.results_dir}")
    else:
        print(f"\\n❌ Student baseline training failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())