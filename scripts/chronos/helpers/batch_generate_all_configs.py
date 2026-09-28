#!/usr/bin/env python3
"""
Comprehensive Chronos Configuration Generator

This script generates ALL possible Chronos experiment configurations across:
- All datasets (d1namo, ohiot1dm)  
- All scenarios (raw, missing_periodic, missing_random, noisy, denoised)
- All patients per dataset
- Multiple models and seeds
- Training and inference modes
- Cross-scenario experiments (train on clean, test on corrupted data)

NOTE: This script generates training and inference configs only.
For trained_inference configs (which require existing checkpoints from completed training),
use batch_generate_trained_inference.py after training is complete.

Usage:
    python helpers/batch_generate_all_configs.py

The script will automatically:
1. Generate training configs for all scenarios
2. Generate inference configs for all scenarios  
3. Generate cross-scenario inference configs (test on corrupted data with pretrained models)
4. Use proper seeds from utilities/seeds.py
5. Create organized experiment folders in ./experiments/
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_config_generation(command_args, description):
    """Run a config generation command with error handling."""
    print(f"\n🔧 {description}")
    command = ["python", "scripts/chronos/config_generator.py"] + command_args
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def main():
    """Generate comprehensive Chronos configurations for all scenarios."""
    
    # Import seeds here to ensure it's available when main() is called
    from utilities.seeds import fixed_seeds
    print(f"✅ Using seeds from utilities/seeds.py: {fixed_seeds}")
    
    # Configuration parameters
    chronos_models = "amazon/chronos-t5-tiny,amazon/chronos-t5-base"
    seeds_str = ",".join(map(str, fixed_seeds))
    
    # Dataset configurations
    datasets = {
        "d1namo": "001,002,003,004,005,006,007",
        "ohiot1dm": "540,544,552,559,563,567,570,575,584,588,591,596"
    }
    
    # Data scenarios
    scenarios = ["standardized", "missing_periodic", "missing_random", "noisy", "denoised"]
    
    # Track success/failure
    success_count = 0
    total_operations = 0
    
    print("🚀 Generating Comprehensive Chronos Configurations")
    print("=" * 60)
    print(f"🤖 Models: {chronos_models}")
    print(f"🎲 Seeds: {seeds_str}")
    print(f"📊 Datasets: {list(datasets.keys())}")
    print(f"🔬 Scenarios: {scenarios}")
    print("=" * 60)
    
    # 1. Generate training configurations for all datasets and scenarios
    print("\n" + "="*60)
    print("PHASE 1: CHRONOS TRAINING CONFIGURATIONS")
    print("="*60)
    
    for dataset, patients in datasets.items():
        for scenario in scenarios:
            total_operations += 1
            description = f"Generating {dataset} training configs for {scenario} scenario"
            
            command_args = [
                "--mode", "train",
                "--dataset", dataset,
                "--data_scenario", scenario,
                "--patients", patients,
                "--models", chronos_models,
                "--seeds", seeds_str
            ]
            
            if run_config_generation(command_args, description):
                success_count += 1
    
    # 2. Generate inference configurations for all datasets and scenarios
    print("\n" + "="*60)
    print("PHASE 2: CHRONOS INFERENCE CONFIGURATIONS")
    print("="*60)
    
    for dataset, patients in datasets.items():
        for scenario in scenarios:
            total_operations += 1
            description = f"Generating {dataset} inference configs for {scenario} scenario"
            
            command_args = [
                "--mode", "inference", 
                "--dataset", dataset,
                "--data_scenario", scenario,
                "--patients", patients,
                "--models", chronos_models,
                "--seeds", seeds_str
            ]
            
            if run_config_generation(command_args, description):
                success_count += 1
    
    # 3. Generate cross-scenario experiments (train on standardized, test on corrupted scenarios)
    # Note: trained_inference configs should be generated AFTER training is complete and checkpoints exist
    print("\n" + "="*60) 
    print("PHASE 3: CHRONOS CROSS-SCENARIO EXPERIMENTS")
    print("="*60)
    
    # Cross-scenario: Train on clean/standardized data, test on corrupted scenarios
    corrupted_scenarios = ["missing_periodic", "missing_random", "noisy", "denoised"]
    
    for dataset, patients in datasets.items():
        for test_scenario in corrupted_scenarios:
            # Generate training configs for standardized data (if not already done above)
            # Skip since we already generated standardized training configs above
            
            # Generate inference configs that use standardized training but test on corrupted data
            total_operations += 1
            description = f"Generating {dataset} cross-scenario: train on standardized, test on {test_scenario}"
            
            # Generate inference configs for cross-scenario testing
            # These will use pretrained models, not trained checkpoints (which don't exist yet)
            command_args = [
                "--mode", "inference",
                "--dataset", dataset,
                "--data_scenario", test_scenario,  # Test on corrupted scenario
                "--patients", patients,
                "--models", chronos_models, 
                "--seeds", seeds_str,
                "--output_dir", f"./experiments/chronos_cross_scenario_{dataset}_train_standardized_test_{test_scenario}/"
            ]
            
            if run_config_generation(command_args, description):
                success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("CHRONOS CONFIGURATION GENERATION COMPLETE!")
    print("="*60)
    print(f"📈 Success: {success_count}/{total_operations} operations")
    print(f"📁 Experiments folder: {os.path.abspath('./experiments/')}")
    
    # Count generated experiment directories
    experiments_path = Path("./experiments/")
    if experiments_path.exists():
        chronos_dirs = [d for d in experiments_path.iterdir() if d.is_dir() and "chronos" in d.name.lower()]
        print(f"📂 Created {len(chronos_dirs)} Chronos experiment directories:")
        
        # Group directories by type for better organization
        training_dirs = [d.name for d in chronos_dirs if "training" in d.name and "trained_inference" not in d.name]
        inference_dirs = [d.name for d in chronos_dirs if "inference" in d.name and "training" not in d.name and "cross_scenario" not in d.name]
        cross_scenario_dirs = [d.name for d in chronos_dirs if "cross_scenario" in d.name]
        
        if training_dirs:
            print(f"   🏋️  Training: {len(training_dirs)} directories")
            for d in sorted(training_dirs)[:3]:  # Show first 3
                print(f"      - {d}")
            if len(training_dirs) > 3:
                print(f"      ... and {len(training_dirs) - 3} more")
        
        if inference_dirs:
            print(f"   🔮 Inference: {len(inference_dirs)} directories") 
            for d in sorted(inference_dirs)[:3]:  # Show first 3
                print(f"      - {d}")
            if len(inference_dirs) > 3:
                print(f"      ... and {len(inference_dirs) - 3} more")
                
        if cross_scenario_dirs:
            print(f"   🔄 Cross-scenario: {len(cross_scenario_dirs)} directories")
            for d in sorted(cross_scenario_dirs)[:3]:  # Show first 3
                print(f"      - {d}")
            if len(cross_scenario_dirs) > 3:
                print(f"      ... and {len(cross_scenario_dirs) - 3} more")
    
    print("\n🎉 All Chronos configurations generated successfully!")
    
    if success_count < total_operations:
        print(f"\n⚠️  Note: {total_operations - success_count} operations had issues. Check logs above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    # Change to project root directory
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)
    
    exit_code = main()
    sys.exit(exit_code)
