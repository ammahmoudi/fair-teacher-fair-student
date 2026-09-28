"""
Combine All Patient Data Script
================================

This script combines all training and testing files from raw_standardized folder
into a single unified dataset for training teacher and student models on all patients.

Usage:
    python data_processing/combine_all_patients.py
"""

import pandas as pd
import os
from pathlib import Path
import sys
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.path_utils import get_project_root


class AllPatientsDataCombiner:
    """Combines all patient data into unified training dataset."""
    
    def __init__(self, source_folder: str = None, output_folder: str = None):
        """Initialize the combiner.
        
        Args:
            source_folder: Path to raw_standardized folder
            output_folder: Path to save combined datasets
        """
        if source_folder is None:
            source_folder = str(get_project_root() / "data" / "ohiot1dm" / "raw_standardized")
        if output_folder is None:
            output_folder = str(get_project_root() / "data" / "ohiot1dm" / "all_patients_combined")
        
        self.source_folder = Path(source_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Source folder: {self.source_folder}")
        print(f"Output folder: {self.output_folder}")
    
    def get_all_csv_files(self) -> Tuple[List[Path], List[Path]]:
        """Get all training and testing CSV files.
        
        Returns:
            Tuple of (training_files, testing_files)
        """
        all_files = list(self.source_folder.glob("*.csv"))
        # Filter out Zone.Identifier files
        all_files = [f for f in all_files if "Zone.Identifier" not in f.name]
        
        training_files = sorted([f for f in all_files if "training" in f.name])
        testing_files = sorted([f for f in all_files if "testing" in f.name])
        
        print(f"\nFound {len(training_files)} training files")
        print(f"Found {len(testing_files)} testing files")
        
        return training_files, testing_files
    
    def combine_files(self, files: List[Path], output_name: str) -> pd.DataFrame:
        """Combine multiple CSV files into one DataFrame.
        
        Args:
            files: List of file paths to combine
            output_name: Name for the output file (e.g., 'all_training.csv')
            
        Returns:
            Combined DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Combining files into: {output_name}")
        print(f"{'='*60}")
        
        dataframes = []
        total_rows = 0
        
        for file_path in files:
            df = pd.read_csv(file_path)
            rows = len(df)
            patient_id = df['item_id'].iloc[0] if 'item_id' in df.columns else 'unknown'
            
            print(f"  {file_path.name:40s} -> {rows:6d} rows (Patient {patient_id})")
            
            dataframes.append(df)
            total_rows += rows
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"Combined Total: {len(combined_df):,} rows from {len(files)} files")
        print(f"{'='*60}")
        
        # Verify columns
        print(f"\nColumns: {list(combined_df.columns)}")
        print(f"\nPatients included: {sorted(combined_df['item_id'].unique().tolist())}")
        print(f"Total unique patients: {combined_df['item_id'].nunique()}")
        
        # Show data distribution
        print(f"\nData distribution per patient:")
        patient_counts = combined_df['item_id'].value_counts().sort_index()
        for patient_id, count in patient_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  Patient {patient_id}: {count:6d} rows ({percentage:5.2f}%)")
        
        return combined_df
    
    def save_combined_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save combined DataFrame to CSV and Arrow formats.
        
        Args:
            df: DataFrame to save
            filename: Base filename (without extension)
            
        Returns:
            Path to saved CSV file
        """
        # Save as CSV
        csv_path = self.output_folder / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved CSV: {csv_path}")
        print(f"   Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Save as Arrow (for faster loading)
        try:
            import pyarrow as pa
            import pyarrow.csv as pa_csv
            
            arrow_path = self.output_folder / f"{filename}.arrow"
            table = pa.Table.from_pandas(df)
            
            with pa.OSFile(str(arrow_path), 'wb') as sink:
                writer = pa.RecordBatchFileWriter(sink, table.schema)
                writer.write_table(table)
                writer.close()
            
            print(f"✅ Saved Arrow: {arrow_path}")
            print(f"   Size: {arrow_path.stat().st_size / (1024*1024):.2f} MB")
        except ImportError:
            print("⚠️  PyArrow not available, skipping Arrow format")
        
        return csv_path
    
    def create_combined_datasets(self):
        """Main method to create all combined datasets."""
        print("\n" + "="*60)
        print("COMBINING ALL PATIENT DATA")
        print("="*60)
        
        # Get all files
        training_files, testing_files = self.get_all_csv_files()
        
        # Combine training files
        print("\n[1/3] Combining all TRAINING files...")
        training_df = self.combine_files(training_files, "all_patients_training.csv")
        training_path = self.save_combined_data(training_df, "all_patients_training")
        
        # Combine testing files
        print("\n[2/3] Combining all TESTING files...")
        testing_df = self.combine_files(testing_files, "all_patients_testing.csv")
        testing_path = self.save_combined_data(testing_df, "all_patients_testing")
        
        # Combine EVERYTHING (training + testing)
        print("\n[3/3] Combining ALL files (training + testing)...")
        all_files = training_files + testing_files
        all_df = self.combine_files(all_files, "all_patients_complete.csv")
        complete_path = self.save_combined_data(all_df, "all_patients_complete")
        
        # Summary
        print("\n" + "="*60)
        print("✅ COMBINATION COMPLETE!")
        print("="*60)
        print(f"\nCreated datasets:")
        print(f"  1. all_patients_training.csv  -> {len(training_df):,} rows ({len(training_files)} patients)")
        print(f"  2. all_patients_testing.csv   -> {len(testing_df):,} rows ({len(testing_files)} patients)")
        print(f"  3. all_patients_complete.csv  -> {len(all_df):,} rows (ALL data)")
        print(f"\nOutput location: {self.output_folder}")
        print(f"\nNext steps:")
        print(f"  - Use 'all_patients_training.csv' to train teacher model")
        print(f"  - Use 'all_patients_testing.csv' to evaluate both models")
        print(f"  - Use 'all_patients_complete.csv' if you want to train on everything")
        print("="*60)
        
        return {
            'training': training_path,
            'testing': testing_path,
            'complete': complete_path
        }


def main():
    """Main execution function."""
    combiner = AllPatientsDataCombiner()
    paths = combiner.create_combined_datasets()
    
    print("\n🎯 Ready for distillation pipeline with all patients!")
    return paths


if __name__ == "__main__":
    paths = main()
