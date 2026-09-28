#!/usr/bin/env python3
"""
Generate Comprehensive Time-LLM Configurations
This script generates Time-LLM configurations for all patients and scenarios across both datasets.
The experiments folder will be created in the root of the project.
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure we're running from the Fair Teacher–Fair Student root directory
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent  # Go up from scripts/time_llm to root

# Change to project root
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

# Path to the unified config generator
config_generator = "scripts/time_llm/config_generator.py"

def run_config_generation(cmd_args, description):
    """Run the config generator with given arguments"""
    print(f"\n🔧 {description}")
    print(f"Command: python {config_generator} {' '.join(cmd_args)}")
    
    try:
        result = subprocess.run(
            ["python", config_generator] + cmd_args,
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    # Import seeds from utilities
    sys.path.append(os.path.join(project_root, "scripts"))
    from utilities.seeds import fixed_seeds
    
    print("🚀 Generating Comprehensive Time-LLM Configurations")
    print("=" * 60)
    print(f"🎲 Using seeds from utilities/seeds.py: {fixed_seeds[:3]}")
    
    # Dataset configurations
    datasets = {
        "ohiot1dm": {
            "patients": ["540", "544", "552", "559", "563", "567", "570", "575", "584", "588", "591", "596"],
            "scenarios": ["raw", "missing_periodic", "missing_random", "noisy", "denoised"]
        },
        "d1namo": {
            "patients": ["001", "002", "003", "004", "005", "006", "007"],
            "scenarios": ["raw", "missing_periodic", "missing_random", "noisy"]
        }
    }
    
    # Use first 3 seeds from utilities/seeds.py as default
    default_seeds = ",".join(map(str, fixed_seeds[:3]))
    
    # Common parameters - using comprehensive model set
    common_params = [
        "--llm_models", "BERT,GPT2,LLAMA,DistilBERT,TinyBERT,BERT-tiny,MiniLM,MobileBERT,ALBERT,OPT-125M", 
        "--seeds", default_seeds  # Use seeds from utilities/seeds.py
    ]
    
    success_count = 0
    total_count = 0
    
    # Generate configurations for each dataset and scenario
    for dataset, config in datasets.items():
        patients_str = ",".join(config["patients"])
        
        print(f"\n📊 Processing Dataset: {dataset.upper()}")
        print(f"Patients: {patients_str}")
        print(f"Scenarios: {config['scenarios']}")
        
        for scenario in config["scenarios"]:
            # Map raw to standardized for the data_scenario parameter
            data_scenario = "standardized" if scenario == "raw" else scenario
            
            total_count += 1
            
            # 1. Training configurations
            train_args = common_params + [
                "--mode", "train",
                "--dataset", dataset,
                "--data_scenario", data_scenario,
                "--patients", patients_str,
                "--epochs", "10"
            ]
            
            success = run_config_generation(
                train_args,
                f"Generating {dataset} {scenario} training configs"
            )
            if success:
                success_count += 1
            
            total_count += 1
            
            # 2. Inference configurations  
            inference_args = common_params + [
                "--mode", "inference",
                "--dataset", dataset,
                "--data_scenario", data_scenario,
                "--patients", patients_str,
                "--epochs", "0"
            ]
            
            success = run_config_generation(
                inference_args,
                f"Generating {dataset} {scenario} inference configs"
            )
            if success:
                success_count += 1
            
            total_count += 1
            
            # 3. Training + Inference configurations
            train_inference_args = common_params + [
                "--mode", "train_inference", 
                "--dataset", dataset,
                "--data_scenario", data_scenario,
                "--patients", patients_str,
                "--epochs", "10"
            ]
            
            success = run_config_generation(
                train_inference_args,
                f"Generating {dataset} {scenario} training+inference configs"
            )
            if success:
                success_count += 1
    
    # Generate cross-scenario configurations (train on clean, test on others)
    print(f"\n🔄 Generating Cross-Scenario Configurations")
    
    for dataset, config in datasets.items():
        patients_str = ",".join(config["patients"])
        
        # Skip raw scenario for cross-scenario (use standardized as training baseline)
        test_scenarios = [s for s in config["scenarios"] if s != "raw"]
        
        for test_scenario in test_scenarios:
            total_count += 1
            
            cross_args = common_params + [
                "--mode", "train_inference",
                "--dataset", dataset,
                "--data_scenario", test_scenario,
                "--train_data_scenario", "standardized",
                "--patients", patients_str,
                "--epochs", "10"
            ]
            
            success = run_config_generation(
                cross_args,
                f"Generating {dataset} cross-scenario: train on clean, test on {test_scenario}"
            )
            if success:
                success_count += 1
    
    # Summary
    print(f"\n✅ Configuration Generation Complete!")
    print(f"📈 Success: {success_count}/{total_count} operations")
    print(f"📁 Experiments folder: {project_root}/experiments/")
    
    # Check what was created
    experiments_dir = project_root / "experiments"
    if experiments_dir.exists():
        experiment_folders = [f.name for f in experiments_dir.iterdir() if f.is_dir() and f.name.startswith("time_llm")]
        print(f"📂 Created {len(experiment_folders)} experiment directories:")
        for folder in sorted(experiment_folders)[:5]:  # Show first 5
            print(f"   - {folder}")
        if len(experiment_folders) > 5:
            print(f"   ... and {len(experiment_folders) - 5} more")

if __name__ == "__main__":
    main()