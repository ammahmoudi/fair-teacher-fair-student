#!/usr/bin/env python3
"""
Knowledge Distillation Driver
Based on the original working implementation
"""

import os
import subprocess
import pandas as pd
import argparse
import datetime
import gin
import json
from shutil import copyfile
import ast
import re
import logging

# Removed gin.configurable to avoid conflicts with main.py run function

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode('utf-8').strip())
        process.stdout.close()
    return process.wait()

def extract_metrics_from_log(log_path):
    """Extract evaluation metrics from the log file"""
    metrics = {}
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Metric results:" in line or "Student Model Evaluation Metrics:" in line:
                    # Extract the part after ':'
                    match = re.search(r':\s*(\{.*\})', line)
                    if match:
                        metrics_str = match.group(1)
                        # Replace np.float32(x) with x
                        metrics_str = re.sub(r'np\.float32\(([^\)]+)\)', r'\1', metrics_str)
                        # Safely evaluate dict
                        metrics_dict = ast.literal_eval(metrics_str)
                        return metrics_dict
    except Exception as e:
        print(f"Error extracting metrics from log: {e}")
    return metrics

def create_test_config(template_path, output_path, model_params, is_student=False):
    """
    Create a test config file from template with proper model parameters
    """
    with open(template_path, 'r') as f:
        config_content = f.read()

    # --- Update run.log_dir ---
    if is_student:
        model_name = model_params['model_type'].lower()
        new_log_dir = f"./experiment_configs_distil/student_{model_name}_dim_{model_params['dim']}_layers_{model_params['layers']}_test/logs"

        config_content = re.sub(
            r'run\.log_dir\s*=\s*"([^"]+)"',
            f'run.log_dir = "{new_log_dir}"',
            config_content
        )

    # --- Update run.llm_settings ---
    llm_settings_pattern = r'run\.llm_settings\s*=\s*({.*?})'

    match = re.search(llm_settings_pattern, config_content, re.DOTALL)

    if not match:
        print("Could not find run.llm_settings block in template config.")
        return

    llm_settings_str = match.group(1)
    llm_settings_dict = ast.literal_eval(llm_settings_str)

    # Update the relevant keys
    llm_settings_dict['restore_checkpoint_path'] = model_params['checkpoint_path']
    llm_settings_dict['llm_model'] = model_params['model_type']
    llm_settings_dict['llm_layers'] = model_params['layers']
    llm_settings_dict['llm_dim'] = model_params['dim']
    
    if 'd_ff' in model_params:
        llm_settings_dict['d_ff'] = model_params['d_ff']

    # Convert dict back to nicely formatted string
    updated_llm_settings_str = "{\n"
    for key, value in llm_settings_dict.items():
        if isinstance(value, str):
            updated_llm_settings_str += f"    '{key}': '{value}',\n"
        else:
            updated_llm_settings_str += f"    '{key}': {value},\n"
    updated_llm_settings_str += "}\n\n"

    # Replace old block with updated one
    config_content = re.sub(
        llm_settings_pattern,
        f'run.llm_settings = {updated_llm_settings_str}',
        config_content,
        flags=re.DOTALL
    )

    # --- Write updated config ---
    with open(output_path, 'w') as f:
        f.write(config_content)

    print(f"Created test config at {output_path}")

def update_train_config_with_student_params(config_path, output_path, student_params, distill_log_dir):
    """Update the training config with student model parameters"""
    with open(config_path, 'r') as f:
        config_lines = f.readlines()

    llm_settings_start = None
    llm_settings_end = None

    # Step 1: Find where run.llm_settings starts
    for i, line in enumerate(config_lines):
        if 'run.llm_settings' in line and '=' in line and '{' in line:
            llm_settings_start = i
            break

    if llm_settings_start is None:
        print("Could not find run.llm_settings in config")
        return config_path

    # Step 2: Find where run.llm_settings block ends (matching '}')
    brace_count = 0
    for i in range(llm_settings_start, len(config_lines)):
        line = config_lines[i]
        brace_count += line.count('{')
        brace_count -= line.count('}')
        if brace_count == 0:
            llm_settings_end = i
            break

    if llm_settings_end is None:
        print("Could not find end of run.llm_settings block")
        return config_path

    # Check if line before '}' ends with comma, if not add it
    prev_line_index = llm_settings_end - 1
    if not config_lines[prev_line_index].strip().endswith(','):
        config_lines[prev_line_index] = config_lines[prev_line_index].rstrip() + ',\n'

    # Update run.log_dir = "logs/distill_xxx"
    for i, line in enumerate(config_lines):
        if line.strip().startswith("run.log_dir"):
            config_lines[i] = f'run.log_dir = "{distill_log_dir}"\n'
            print(f"✅ Forced run.log_dir = {distill_log_dir}")
            break

    student_params_str = [
        f"    'student_model': '{student_params['model']}',\n",
        f"    'student_layers': {student_params['layers']},\n",
        f"    'student_dim': {student_params['dim']},\n",
        f"    'student_d_ff': {student_params['d_ff']},\n"
    ]

    # Insert the lines
    config_lines = config_lines[:llm_settings_end] + student_params_str + config_lines[llm_settings_end:]

    # Save updated config
    with open(output_path, 'w') as f:
        f.writelines(config_lines)

    print(f"Updated training config at {output_path} with student parameters")
    return output_path

class DistillationDriver:
    def __init__(self, settings, data_settings, log_dir="./logs", teacher_checkpoint_path=None):
        """Initialize distillation driver with working parameters"""
        self.settings = settings
        self.data_settings = data_settings
        self.log_dir = log_dir
        self.teacher_checkpoint_path = teacher_checkpoint_path
        
        # Extract student model parameters from settings
        self.student_params = {
            'model': settings.get('llm_model', 'TinyBERT'),
            'layers': settings.get('llm_layers', 4), 
            'dim': settings.get('llm_dim', 312),
            'd_ff': settings.get('d_ff', 32)
        }
        
        # Use the provided log directory instead of creating a hardcoded one
        self.distill_log_dir = log_dir
        os.makedirs(self.distill_log_dir, exist_ok=True)
        
    def distill_knowledge(self, train_loader, val_loader=None, epochs=1):
        """Main distillation method using the working implementation"""
        logging.info(f"🎓 Starting Knowledge Distillation for {epochs} epochs...")
        
        # Create training config for distillation (first config example)
        train_config_path = self._create_training_config(epochs)
        
        # 1. Run the distillation training
        logging.info("🔥 Step 1: Training Student Model with Distillation...")
        distill_command = f"python main.py --config_path {train_config_path}"
        success = run_command(distill_command) == 0
        
        if not success:
            raise RuntimeError("Distillation training failed")
            
        # Find the actual checkpoint file and copy it to expected location
        checkpoints_dir = os.path.join(self.distill_log_dir, "checkpoints")
        actual_checkpoint = os.path.join(checkpoints_dir, "checkpoint.pth")
        checkpoint_path = os.path.join(self.distill_log_dir, "student_distilled.pth")
        
        if os.path.exists(actual_checkpoint):
            copyfile(actual_checkpoint, checkpoint_path)
        else:
            # Find any .pth file in the logs subdirectory
            for root, dirs, files in os.walk(self.distill_log_dir):
                for file in files:
                    if file.endswith(".pth") and "checkpoint" in file:
                        actual_path = os.path.join(root, file)
                        copyfile(actual_path, checkpoint_path)
                        break
                if os.path.exists(checkpoint_path):
                    break
        
        logging.info(f"✅ Knowledge Distillation completed!")
        logging.info(f"📁 Student checkpoint saved to: {checkpoint_path}")
        
        return checkpoint_path, 350.0, None
        
    def _create_training_config(self, epochs):
        """Create training config based on your first config example"""
        config_path = f"{self.distill_log_dir}/training_config.gin"
        
        # Extract patient number from data settings (e.g., 570 from path)
        train_data_path = self.data_settings.get('path_to_train_data', './data/standardized/570-ws-training.csv')
        patient = train_data_path.split('/')[-1].split('-')[0]  # Extract patient number
        
        config_content = f'''run.log_dir = "{self.distill_log_dir}"
run.data_settings = {{
    'path_to_train_data': '{train_data_path}',
    'path_to_test_data': '{self.data_settings.get("path_to_test_data", "./data/standardized/570-ws-testing.csv")}',
    'input_features': {self.data_settings.get("input_features", ["target"])},
    'labels': {self.data_settings.get("labels", ["target"])},
    'prompt_path': '{self.data_settings.get("prompt_path", "./data/standardized/t1dm_prompt.txt")}',
    'preprocessing_method': '{self.data_settings.get("preprocessing_method", "min_max")}',
    'preprocess_input_features': {str(self.data_settings.get("preprocess_input_features", False))},
    'preprocess_label': {str(self.data_settings.get("preprocess_label", False))},
    'frequency': '{self.data_settings.get("frequency", "5min")}',
    'percent': {self.data_settings.get("percent", 100)},
    'val_split': {self.data_settings.get("val_split", 0)}
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'training+inference',
    'method': 'time_llm',
    'teacher_checkpoint_path': '{self.teacher_checkpoint_path}',
    'restore_from_checkpoint': False,
    'llm_model': 'BERT',
    'llm_layers': 12,
    'llm_dim': 768,
    'num_workers': 1,
    'torch_dtype': 'float32',
    'model_id': 'test',
    'sequence_length': {self.settings.get("sequence_length", 6)},
    'context_length': {self.settings.get("context_length", 6)},
    'prediction_length': {self.settings.get("prediction_length", 9)},
    'patch_len': {self.settings.get("patch_len", 6)},
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': {self.settings.get("learning_rate", 0.0005)},
    'train_epochs': {epochs},
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
    'des': 'test',
    'model_comment': 'time_llm_BERT_768_6_6_9_6',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': {self.settings.get("seed", 238822)},
    'student_model': '{self.student_params["model"]}',
    'student_layers': {self.student_params["layers"]},
    'student_dim': {self.student_params["dim"]},
    'student_d_ff': {self.student_params["d_ff"]},
}}'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        logging.info(f"Created training config: {config_path}")
        return config_path
        
    def predict(self, test_loader, output_dir=None):
        """Prediction using the trained student model"""
        logging.info("🔮 Running inference with distilled student model...")
        
        # Create inference config (second config example)
        inference_config_path = self._create_inference_config()
        
        # Run inference
        inference_command = f"./run_main.sh --config_path {inference_config_path} --log_level INFO --remove_checkpoints False"
        run_command(inference_command)
        
        logging.info("✅ Inference completed with distilled student model!")
        return None, None, {}
        
    def _create_inference_config(self):
        """Create inference config based on your second config example"""
        config_path = f"{self.distill_log_dir}/inference_config.gin"
        
        student_model_path = os.path.join(self.distill_log_dir, "student_distilled.pth")
        
        config_content = f'''run.log_dir = "{self.distill_log_dir}/inference_logs"
run.data_settings = {{
    'path_to_train_data': '{self.data_settings.get("path_to_train_data", "./data/standardized/570-ws-training.csv")}',
    'path_to_test_data': '{self.data_settings.get("path_to_test_data", "./data/standardized/570-ws-testing.csv")}',
    'input_features': {self.data_settings.get("input_features", ["target"])},
    'labels': {self.data_settings.get("labels", ["target"])},
    'prompt_path': '{self.data_settings.get("prompt_path", "./data/standardized/t1dm_prompt.txt")}',
    'preprocessing_method': '{self.data_settings.get("preprocessing_method", "min_max")}',
    'preprocess_input_features': {str(self.data_settings.get("preprocess_input_features", False))},
    'preprocess_label': {str(self.data_settings.get("preprocess_label", False))},
    'frequency': '{self.data_settings.get("frequency", "5min")}',
    'percent': {self.data_settings.get("percent", 100)},
    'val_split': {self.data_settings.get("val_split", 0)}
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'inference',
    'method': 'time_llm',
    'restore_from_checkpoint': True,
    'restore_checkpoint_path': '{student_model_path}',
    'llm_model': '{self.student_params["model"]}',
    'llm_layers': {self.student_params["layers"]},
    'llm_dim': {self.student_params["dim"]},
    'num_workers': 1,
    'torch_dtype': 'float32',
    'model_id': 'test',
    'sequence_length': {self.settings.get("sequence_length", 6)},
    'context_length': {self.settings.get("context_length", 6)},
    'prediction_length': {self.settings.get("prediction_length", 9)},
    'patch_len': {self.settings.get("patch_len", 6)},
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.001,
    'train_epochs': 20,
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
    'des': 'test',
    'model_comment': 'time_llm_BERT_768_6_6_9_6',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': {self.settings.get("seed", 238822)}
}}'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        logging.info(f"Created inference config: {config_path}")
        return config_path
