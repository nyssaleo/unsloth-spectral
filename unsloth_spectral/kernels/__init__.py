"""
Triton Kernels for Spectral Attention

This module provides optimized Triton kernels for computing attention
directly in compressed spectral space without full reconstruction.

Kernels:
--------
- spectral_score_kernel: Computes Q @ K_spectral^T with RoPE
- spectral_value_kernel: Computes attn @ V_spectral

High-Level API:
--------------
- spectral_attention_decode: Full spectral attention for decode (single token)
- TritonSpectralConfig: Configuration for kernel tuning

Usage:
------
```python
from unsloth_spectral.kernels import spectral_attention_decode, TRITON_AVAILABLE

if TRITON_AVAILABLE:
    output = spectral_attention_decode(Q, cache, cos, sin, query_pos, scale)
else:
    # Fallback to PyTorch
    from unsloth_spectral.spectral_attention import spectral_attention_forward
    output = spectral_attention_forward(Q, cache, cos, sin, query_pos)
```
"""

import logging

# Setup module logger
logger = logging.getLogger("unsloth_spectral.kernels")
logger.setLevel(logging.DEBUG)

# Check Triton availability
TRITON_AVAILABLE = False
TRITON_VERSION = None

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    TRITON_VERSION = triton.__version__
    logger.info(f"Triton v{TRITON_VERSION} available - GPU kernels enabled")
except ImportError as e:
    logger.warning(f"Triton not available: {e}. Using PyTorch fallback.")

# Import kernel functions (only if Triton available)
if TRITON_AVAILABLE:
    from .spectral_attention import (
        spectral_attention_decode,
        spectral_score_forward,
        spectral_value_forward,
        TritonSpectralConfig,
    )
    __all__ = [
        "spectral_attention_decode",
        "spectral_score_forward", 
        "spectral_value_forward",
        "TritonSpectralConfig",
        "TRITON_AVAILABLE",
        "TRITON_VERSION",
    ]
else:
    __all__ = ["TRITON_AVAILABLE", "TRITON_VERSION"]
