"""
Efficiency Toolkit for LLM Analysis

This module provides comprehensive tools for analyzing LLM efficiency experiments,
including log parsing, distillation analysis, resource monitoring, and performance metrics.
"""

from .log_analyzer import LogAnalyzer, DistillationLogAnalyzer
from .distillation_analyzer import DistillationEfficiencyAnalyzer
from .resource_monitor import ResourceMonitor

__version__ = "1.0.0"
__author__ = "Fair Teacher–Fair Student Team"

# Core imports
__all__ = [
    'LogAnalyzer',
    'DistillationLogAnalyzer', 
    'DistillationEfficiencyAnalyzer',
    'ResourceMonitor'
]

# Try to import additional components if available
try:
    from .core.comprehensive_efficiency_runner import ComprehensiveEfficiencyRunner
    from .core.efficiency_calculator import EfficiencyCalculator
    from .core.real_time_profiler import RealTimeProfiler
    
    __all__.extend([
        'ComprehensiveEfficiencyRunner',
        'EfficiencyCalculator', 
        'RealTimeProfiler'
    ])
except ImportError:
    # Allow imports to fail gracefully during setup
    pass