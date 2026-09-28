#!/usr/bin/env python3
"""
LLM Efficiency Analysis Runner

This script runs the clean efficiency analysis notebook programmatically.
Use this to generate reports without opening Jupyter.
"""

import subprocess
import sys
from pathlib import Path


def run_efficiency_analysis():
    """Run the efficiency analysis notebook"""
    
    notebook_path = Path("notebooks/efficiency_analysis_clean.ipynb")
    output_notebook = Path("notebooks/outputs/efficiency_analysis_results.ipynb")
    
    # Ensure output directory exists
    output_notebook.parent.mkdir(exist_ok=True)
    
    print("🚀 Running LLM Efficiency Analysis...")
    print(f"📖 Notebook: {notebook_path}")
    print(f"📄 Output: {output_notebook}")
    
    try:
        # Run the notebook using jupyter nbconvert
        cmd = [
            "jupyter", "nbconvert", 
            "--to", "notebook",
            "--execute",
            "--output", str(output_notebook),
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {output_notebook}")
        print("\n📁 Check the notebooks/outputs/ folder for:")
        print("  • CSV data files")
        print("  • Analysis visualizations") 
        print("  • Efficiency rankings")
        print("  • Deployment recommendations")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Jupyter not found. Please install jupyter:")
        print("   pip install jupyter nbconvert")
        return False
    
    return True


if __name__ == "__main__":
    success = run_efficiency_analysis()
    sys.exit(0 if success else 1)