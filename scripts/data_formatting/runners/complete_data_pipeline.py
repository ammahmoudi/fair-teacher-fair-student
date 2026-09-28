#!/usr/bin/env python3
"""
Complete Data Processing Pipeline for Fair Teacher–Fair Student Project

This script provides a complete data processing pipeline that:
1. Standardizes data files (converts to item_id, timestamp, target format)
2. Formats data with window configurations (6_6 and 6_9)
3. Converts formatted data to Arrow format for training

Supports all datasets (ohiot1dm, d1namo) and scenarios (raw, missing_periodic, 
missing_random, noisy, denoised).

Usage:
    # Process all datasets and scenarios
    python complete_data_pipeline.py --all

    # Process specific dataset
    python complete_data_pipeline.py --dataset ohiot1dm --scenarios all

    # Process specific dataset and scenarios
    python complete_data_pipeline.py --dataset d1namo --scenarios raw,noisy

    # Dry run (show what would be processed)
    python complete_data_pipeline.py --dataset ohiot1dm --dry-run

    # Skip specific steps
    python complete_data_pipeline.py --dataset ohiot1dm --skip-standardize
    python complete_data_pipeline.py --dataset ohiot1dm --skip-format --skip-arrow
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_data_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define available datasets and scenarios
DATASETS = ['ohiot1dm', 'd1namo']
SCENARIOS = ['raw', 'missing_periodic', 'missing_random', 'noisy', 'denoised']
WINDOW_CONFIGS = ['6,6', '6,9']

class DataProcessingPipeline:
    """Complete data processing pipeline manager."""
    
    def __init__(self, dry_run=False):
        self.project_root = get_project_root()
        self.core_scripts_dir = Path(__file__).parent.parent / "core"  # ../core folder
        self.dry_run = dry_run
        
        # Paths to core processing scripts
        self.standardizer_script = self.core_scripts_dir / "standardize_data.py"
        self.formatter_script = self.core_scripts_dir / "format_data.py"
        self.arrow_converter_script = self.core_scripts_dir / "convert_to_arrow.py"
        
        # Verify scripts exist
        self._verify_scripts()
    
    def _verify_scripts(self):
        """Verify that all required processing scripts exist."""
        scripts = [
            self.standardizer_script,
            self.formatter_script, 
            self.arrow_converter_script
        ]
        
        for script in scripts:
            if not script.exists():
                raise FileNotFoundError(f"Required script not found: {script}")
        
        logger.info("✅ All processing scripts found")
    
    def get_available_scenarios(self, dataset: str) -> List[str]:
        """Get available scenarios for a dataset."""
        dataset_path = get_data_path(dataset)
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return []
        
        available_scenarios = []
        for scenario in SCENARIOS:
            scenario_path = dataset_path / scenario
            if scenario_path.exists():
                available_scenarios.append(scenario)
        
        return available_scenarios
    
    def get_processing_summary(self, datasets: List[str], scenarios: List[str]) -> dict:
        """Get a summary of what will be processed."""
        summary = {
            'datasets': {},
            'total_combinations': 0,
            'total_window_configs': len(WINDOW_CONFIGS)
        }
        
        for dataset in datasets:
            available_scenarios = self.get_available_scenarios(dataset)
            dataset_scenarios = [s for s in scenarios if s in available_scenarios]
            
            summary['datasets'][dataset] = {
                'scenarios': dataset_scenarios,
                'available_scenarios': available_scenarios,
                'combinations': len(dataset_scenarios)
            }
            summary['total_combinations'] += len(dataset_scenarios)
        
        return summary
    
    def run_standardization(self, datasets: List[str], scenarios: List[str]) -> bool:
        """Run data standardization step."""
        logger.info("🔧 Starting data standardization...")
        
        success = True
        for dataset in datasets:
            available_scenarios = self.get_available_scenarios(dataset)
            dataset_scenarios = [s for s in scenarios if s in available_scenarios]
            
            for scenario in dataset_scenarios:
                logger.info(f"Standardizing {dataset}/{scenario}")
                
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would standardize {dataset}/{scenario}")
                    continue
                
                cmd = [
                    'python', str(self.standardizer_script),
                    '--dataset', dataset,
                    '--scenario', scenario
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                    if result.returncode == 0:
                        logger.info(f"✅ Standardized {dataset}/{scenario}")
                    else:
                        logger.error(f"❌ Failed to standardize {dataset}/{scenario}")
                        logger.error(f"Error: {result.stderr}")
                        success = False
                except Exception as e:
                    logger.error(f"❌ Error standardizing {dataset}/{scenario}: {e}")
                    success = False
        
        return success
    
    def run_formatting(self, datasets: List[str], scenarios: List[str]) -> bool:
        """Run data formatting step for both window configurations."""
        logger.info("📐 Starting data formatting...")
        
        success = True
        for dataset in datasets:
            available_scenarios = self.get_available_scenarios(dataset)
            dataset_scenarios = [s for s in scenarios if s in available_scenarios]
            
            for scenario in dataset_scenarios:
                for window_config in WINDOW_CONFIGS:
                    logger.info(f"Formatting {dataset}/{scenario} with window {window_config}")
                    
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would format {dataset}/{scenario} with {window_config}")
                        continue
                    
                    cmd = [
                        'python', str(self.formatter_script),
                        '--dataset', dataset,
                        '--scenario', scenario,
                        '--windows', window_config
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                        if result.returncode == 0:
                            logger.info(f"✅ Formatted {dataset}/{scenario} with {window_config}")
                        else:
                            logger.error(f"❌ Failed to format {dataset}/{scenario} with {window_config}")
                            logger.error(f"Error: {result.stderr}")
                            success = False
                    except Exception as e:
                        logger.error(f"❌ Error formatting {dataset}/{scenario} with {window_config}: {e}")
                        success = False
        
        return success
    
    def run_arrow_conversion(self, datasets: List[str], scenarios: List[str]) -> bool:
        """Run Arrow format conversion step."""
        logger.info("🏹 Starting Arrow conversion...")
        
        success = True
        for dataset in datasets:
            available_scenarios = self.get_available_scenarios(dataset)
            dataset_scenarios = [s for s in scenarios if s in available_scenarios]
            
            for scenario in dataset_scenarios:
                # Convert standardized scenario to arrow
                standardized_scenario = scenario if scenario == 'raw' else f"{scenario}_standardized"
                logger.info(f"Converting {dataset}/{standardized_scenario} to Arrow format")
                
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would convert {dataset}/{standardized_scenario} to Arrow")
                    continue
                
                cmd = [
                    'python', str(self.arrow_converter_script),
                    '--dataset', dataset,
                    '--scenario', standardized_scenario
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                    if result.returncode == 0:
                        logger.info(f"✅ Converted {dataset}/{standardized_scenario} to Arrow")
                    else:
                        logger.error(f"❌ Failed to convert {dataset}/{standardized_scenario} to Arrow")
                        logger.error(f"Error: {result.stderr}")
                        success = False
                except Exception as e:
                    logger.error(f"❌ Error converting {dataset}/{standardized_scenario} to Arrow: {e}")
                    success = False
        
        return success
    
    def run_complete_pipeline(self, datasets: List[str], scenarios: List[str], 
                            skip_standardize=False, skip_format=False, skip_arrow=False) -> bool:
        """Run the complete data processing pipeline."""
        logger.info("🚀 Starting complete data processing pipeline")
        
        # Show processing summary
        summary = self.get_processing_summary(datasets, scenarios)
        logger.info("=" * 60)
        logger.info("📋 Processing Summary:")
        logger.info(f"Total datasets: {len(datasets)}")
        logger.info(f"Total combinations: {summary['total_combinations']}")
        logger.info(f"Window configurations: {WINDOW_CONFIGS}")
        
        for dataset, info in summary['datasets'].items():
            logger.info(f"\n{dataset}:")
            logger.info(f"  Available scenarios: {info['available_scenarios']}")
            logger.info(f"  Will process: {info['scenarios']}")
        
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("🔍 DRY RUN MODE - showing what would be processed")
            logger.info("=" * 60)
        
        # Run pipeline steps
        overall_success = True
        
        # Step 1: Standardization
        if not skip_standardize:
            if not self.run_standardization(datasets, scenarios):
                logger.error("❌ Standardization step failed")
                overall_success = False
        else:
            logger.info("⏭️  Skipping standardization step")
        
        # Step 2: Formatting
        if not skip_format:
            if not self.run_formatting(datasets, scenarios):
                logger.error("❌ Formatting step failed") 
                overall_success = False
        else:
            logger.info("⏭️  Skipping formatting step")
        
        # Step 3: Arrow conversion
        if not skip_arrow:
            if not self.run_arrow_conversion(datasets, scenarios):
                logger.error("❌ Arrow conversion step failed")
                overall_success = False
        else:
            logger.info("⏭️  Skipping Arrow conversion step")
        
        # Final summary
        logger.info("=" * 60)
        if overall_success:
            logger.info("🎉 Complete data processing pipeline finished successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Check the data directories for processed files")
            logger.info("2. Use the formatted data for training configurations")
            logger.info("3. Use the Arrow files for Chronos training")
        else:
            logger.error("❌ Pipeline completed with errors. Check logs above.")
        
        return overall_success


def parse_scenarios(scenarios_str: str) -> List[str]:
    """Parse comma-separated scenarios string."""
    if scenarios_str.lower() == 'all':
        return SCENARIOS
    return [s.strip() for s in scenarios_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Complete Data Processing Pipeline for Fair Teacher–Fair Student",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets and scenarios
  python complete_data_pipeline.py --all

  # Process specific dataset with all scenarios
  python complete_data_pipeline.py --dataset ohiot1dm

  # Process specific scenarios
  python complete_data_pipeline.py --dataset d1namo --scenarios raw,noisy

  # Dry run to see what would be processed
  python complete_data_pipeline.py --dataset ohiot1dm --dry-run

  # Skip specific steps
  python complete_data_pipeline.py --dataset ohiot1dm --skip-format
        """
    )
    
    # Main arguments
    parser.add_argument('--all', action='store_true',
                       help='Process all datasets and scenarios')
    parser.add_argument('--dataset', choices=DATASETS + ['all'],
                       help='Dataset to process')
    parser.add_argument('--scenarios', default='all',
                       help='Comma-separated list of scenarios or "all"')
    
    # Control arguments
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually running')
    parser.add_argument('--skip-standardize', action='store_true',
                       help='Skip the standardization step')
    parser.add_argument('--skip-format', action='store_true',
                       help='Skip the formatting step')
    parser.add_argument('--skip-arrow', action='store_true',
                       help='Skip the Arrow conversion step')
    
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.all or args.dataset == 'all':
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Must specify --dataset or --all")
    
    # Parse scenarios
    scenarios = parse_scenarios(args.scenarios)
    
    # Validate scenarios
    invalid_scenarios = [s for s in scenarios if s not in SCENARIOS]
    if invalid_scenarios:
        parser.error(f"Invalid scenarios: {invalid_scenarios}. Valid: {SCENARIOS}")
    
    # Create and run pipeline
    try:
        pipeline = DataProcessingPipeline(dry_run=args.dry_run)
        success = pipeline.run_complete_pipeline(
            datasets=datasets,
            scenarios=scenarios,
            skip_standardize=args.skip_standardize,
            skip_format=args.skip_format,
            skip_arrow=args.skip_arrow
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()