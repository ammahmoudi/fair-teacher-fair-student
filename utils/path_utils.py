#!/usr/bin/env python3
"""
Path Utilities for Fair Teacher–Fair Student Project

This module provides centralized path management utilities to make the project
portable across different installations and users. It automatically detects
the project root and provides consistent path handling.

Usage:
    from utils.path_utils import get_project_root, get_data_path, get_models_path
    
    project_root = get_project_root()
    data_path = get_data_path("ohiot1dm", "raw_standardized")
    models_path = get_models_path()
"""

import os
from pathlib import Path
from typing import Optional, Union


def get_project_root() -> Path:
    """
    Get the project root directory dynamically.
    
    This function searches for the Fair Teacher–Fair Student project root by looking for
    characteristic files/directories like README.md, requirements.txt, etc.
    
    Returns:
        Path: The project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be determined
    """
    # Start from the current file's directory
    current_path = Path(__file__).resolve()
    
    # Look for characteristic files that indicate project root
    root_indicators = [
        'README.md',
        'requirements.txt',
        'main.py',
        'configs',
        'data',
        'models'
    ]
    
    # Traverse up the directory tree
    for parent in [current_path] + list(current_path.parents):
        # Check if this directory contains enough indicators
        indicators_found = sum(1 for indicator in root_indicators 
                             if (parent / indicator).exists())
        
        # If we find at least 3 indicators, consider this the root
        if indicators_found >= 3:
            return parent
    
    # If we can't find the root automatically, check environment variables
    # Support both new FAIR_TEACHER_ROOT and legacy LLM_TIME_ROOT
    if 'FAIR_TEACHER_ROOT' in os.environ:
        return Path(os.environ['FAIR_TEACHER_ROOT'])
    if 'LLM_TIME_ROOT' in os.environ:
        return Path(os.environ['LLM_TIME_ROOT'])
    
    # Fallback: assume current working directory if it looks like project root
    cwd = Path.cwd()
    if (cwd / 'README.md').exists() and (cwd / 'main.py').exists():
        return cwd
    
    raise FileNotFoundError(
        "Could not determine Fair Teacher–Fair Student project root. "
        "Please set the FAIR_TEACHER_ROOT or LLM_TIME_ROOT environment variable or "
        "run from within the project directory."
    )


def get_data_path(*subdirs: str) -> Path:
    """
    Get path to data directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectories to append to data path
        
    Returns:
        Path: The complete data path
        
    Examples:
        get_data_path() -> /path/to/fair-teacher-fair-student/data
        get_data_path("ohiot1dm") -> /path/to/fair-teacher-fair-student/data/ohiot1dm
        get_data_path("ohiot1dm", "raw_standardized") -> /path/to/fair-teacher-fair-student/data/ohiot1dm/raw_standardized
    """
    data_path = get_project_root() / "data"
    for subdir in subdirs:
        data_path = data_path / subdir
    return data_path


def get_models_path(*subdirs: str) -> Path:
    """
    Get path to models directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectories to append to models path
        
    Returns:
        Path: The complete models path
    """
    models_path = get_project_root() / "models"
    for subdir in subdirs:
        models_path = models_path / subdir
    return models_path


def get_configs_path(*subdirs: str) -> Path:
    """
    Get path to configs directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectories to append to configs path
        
    Returns:
        Path: The complete configs path
    """
    configs_path = get_project_root() / "configs"
    for subdir in subdirs:
        configs_path = configs_path / subdir
    return configs_path


def get_scripts_path(*subdirs: str) -> Path:
    """
    Get path to scripts directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectories to append to scripts path
        
    Returns:
        Path: The complete scripts path
    """
    scripts_path = get_project_root() / "scripts"
    for subdir in subdirs:
        scripts_path = scripts_path / subdir
    return scripts_path


def get_distillation_path(*subdirs: str) -> Path:
    """
    Get path to distillation directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectories to append to distillation path
        
    Returns:
        Path: The complete distillation path
    """
    distillation_path = get_project_root() / "distillation"
    for subdir in subdirs:
        distillation_path = distillation_path / subdir
    return distillation_path


def get_logs_path(*subdirs: str) -> Path:
    """
    Get path to logs directory with optional subdirectories.
    Creates the directory if it doesn't exist.
    
    Args:
        *subdirs: Optional subdirectories to append to logs path
        
    Returns:
        Path: The complete logs path
    """
    logs_path = get_project_root() / "logs"
    for subdir in subdirs:
        logs_path = logs_path / subdir
    
    # Create logs directory if it doesn't exist
    logs_path.mkdir(parents=True, exist_ok=True)
    return logs_path


def get_results_path(*subdirs: str) -> Path:
    """
    Get path to results directory with optional subdirectories.
    Creates the directory if it doesn't exist.
    
    Args:
        *subdirs: Optional subdirectories to append to results path
        
    Returns:
        Path: The complete results path
    """
    results_path = get_project_root() / "results"
    for subdir in subdirs:
        results_path = results_path / subdir
    
    # Create results directory if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def ensure_path_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a path exists by creating it if necessary.
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path: The path as a Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(full_path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> str:
    """
    Get a path relative to the project root or specified directory.
    
    Args:
        full_path: The full path to make relative
        relative_to: Directory to make path relative to (default: project root)
        
    Returns:
        str: The relative path as a string
    """
    full_path = Path(full_path)
    if relative_to is None:
        relative_to = get_project_root()
    else:
        relative_to = Path(relative_to)
    
    try:
        return str(full_path.relative_to(relative_to))
    except ValueError:
        # If path is not relative to the specified directory, return absolute path
        return str(full_path)


# Environment variable support for backwards compatibility
def get_project_root_env() -> Optional[str]:
    """
    Get project root from environment variable if set.
    Supports both FAIR_TEACHER_ROOT and legacy LLM_TIME_ROOT.
    
    Returns:
        Optional[str]: Project root path from environment or None
    """
    return os.environ.get('FAIR_TEACHER_ROOT') or os.environ.get('LLM_TIME_ROOT')


def set_project_root_env(path: Union[str, Path]) -> None:
    """
    Set the project root environment variable.
    Sets both FAIR_TEACHER_ROOT and LLM_TIME_ROOT for compatibility.
    
    Args:
        path: Project root path to set
    """
    resolved_path = str(Path(path).resolve())
    os.environ['FAIR_TEACHER_ROOT'] = resolved_path
    os.environ['LLM_TIME_ROOT'] = resolved_path  # Legacy support


# Legacy support functions for common patterns
def get_arrow_data_path(dataset: str, scenario: str = "raw_standardized", 
                       patient_id: Optional[str] = None) -> Path:
    """
    Get path to arrow data files following project conventions.
    
    Args:
        dataset: Dataset name (e.g., "ohiot1dm", "d1namo")
        scenario: Data scenario (e.g., "raw_standardized", "missing_periodic")
        patient_id: Optional patient ID for specific file
        
    Returns:
        Path: Path to arrow data
    """
    if scenario == "standardized":
        # Handle legacy "standardized" scenario
        path = get_data_path(dataset, "raw_standardized")
    elif dataset == "ohiot1dm" and scenario != "raw_standardized":
        # Special case for ohiot1dm non-standardized scenarios
        path = get_data_path(scenario)
    else:
        path = get_data_path(dataset, scenario)
    
    if patient_id:
        path = path / f"{patient_id}-ws-training.arrow"
    
    return path


def get_formatted_data_path(context_length: int, prediction_length: int) -> str:
    """
    Get relative path to formatted data for inference.
    
    Args:
        context_length: Context window length
        prediction_length: Prediction window length
        
    Returns:
        str: Relative path to formatted data
    """
    return f"./data/formatted/{context_length}_{prediction_length}"


if __name__ == "__main__":
    # Test the utilities
    try:
        print(f"Project root: {get_project_root()}")
        print(f"Data path: {get_data_path()}")
        print(f"Models path: {get_models_path()}")
        print(f"OhioT1DM raw data: {get_data_path('ohiot1dm', 'raw_standardized')}")
        print(f"Arrow file example: {get_arrow_data_path('ohiot1dm', 'raw_standardized', '570')}")
    except Exception as e:
        print(f"Error testing path utilities: {e}")