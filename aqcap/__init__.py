"""
AQCAP: Adaptive Quantization and Context-Aware Pruning
A framework for edge-deployed LLM vulnerability detection via sensitivity-guided mixed precision.
"""

import os
# Prevent Windows multiple OpenMP runtime initialization conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "0.1.0"
