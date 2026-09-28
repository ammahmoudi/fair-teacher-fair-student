#!/usr/bin/env python3
"""
Import utility to help with importing from utilities directory.
This should be imported first in scripts that need access to utilities.
"""

import sys
import os

# Add the scripts directory to Python path for utilities import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)