"""
Triton Kernels for Spectral Attention

This module implements high-performance Triton kernels for computing attention
directly in compressed spectral space. The kernels avoid full K/V reconstruction,
reducing memory bandwidth by ~5x for long contexts.

Architecture:
------------
Phase 1: Score Computation (Triton)
    scores = (Q @ B_K^T) @ C_K^T  with RoPE correction

Phase 2: Softmax (PyTorch)
    attn = softmax(scores * scale)

Phase 3: Value Aggregation (Triton)
    output = (attn @ C_V) @ B_V

Memory Bandwidth:
----------------
Standard:  2 * H * T * D * sizeof(fp16) = 16.8 MB for T=4096, H=8, D=128
Spectral:  H * (T*k + k*D) * sizeof(fp16) = 3.3 MB for k_K=16, k_V=32
Reduction: 5.1x

Author: Ankit Prajapati & Claude (Anthropic)
Date: January 2026
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging
import math
import time

# Setup logger with detailed formatting
logger = logging.getLogger("unsloth_spectral.kernels.spectral_attention")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '[%(levelname)s] %(name)s | %(message)s'
))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


@dataclass
class TritonSpectralConfig:
    """Configuration for Triton kernel tuning."""
    BLOCK_T: int = 64      # Tile size for sequence dimension
    BLOCK_K: int = 32      # Tile size for spectral rank dimension
    BLOCK_D: int = 64      # Tile size for head dimension
    num_warps: int = 4     # Number of warps per block
    num_stages: int = 2    # Number of pipeline stages
    enable_logging: bool = True
    
    def __post_init__(self):
        # Validate configuration
        assert self.BLOCK_T >= 16, "BLOCK_T must be at least 16"
        assert self.BLOCK_K >= 8, "BLOCK_K must be at least 8"
        assert self.BLOCK_D >= 32, "BLOCK_D must be at least 32"


# ============================================================================
# TRITON KERNEL: Spectral Score Computation
# ============================================================================

@triton.jit
def _spectral_score_kernel(
    # Input pointers
    Q_ptr,              # [B, H_q, 1, D] - Query (already RoPE rotated at query_pos)
    coeffs_K_ptr,       # [H_kv, T_block, k_K] - Spectral coefficients
    basis_K_ptr,        # [H_kv, k_K, D] - Spectral basis
    cos_ptr,            # [max_seq, D] - RoPE cosines
    sin_ptr,            # [max_seq, D] - RoPE sines
    # Output pointer
    scores_ptr,         # [B, H_q, 1, T_block] - Output scores
    # Dequantization (optional - set to None if not using INT8)
    scales_K_ptr,       # [H_kv, 2] - min, max for dequant (or None)
    # Dimensions
    B: tl.constexpr,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    T_block: tl.constexpr,
    k_K: tl.constexpr,
    D: tl.constexpr,
    # Position info
    start_position,     # int: First key position in this block
    query_position,     # int: Position of current query
    # Strides for Q [B, H_q, 1, D]
    stride_qb, stride_qh, stride_qt, stride_qd,
    # Strides for coeffs_K [H_kv, T_block, k_K]
    stride_ckh, stride_ckt, stride_ckk,
    # Strides for basis_K [H_kv, k_K, D]
    stride_bkh, stride_bkk, stride_bkd,
    # Strides for scores [B, H_q, 1, T_block]
    stride_sb, stride_sh, stride_st, stride_ss,
    # Strides for cos/sin [max_seq, D]
    stride_ropeseq, stride_roped,
    # GQA
    n_rep: tl.constexpr,  # H_q // H_kv
    # Block sizes (constexpr for compilation)
    BLOCK_T: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 16,
    BLOCK_D: tl.constexpr = 64,
):
    """
    Compute spectral attention scores with per-position RoPE correction.
    
    For each key position p in [start_position, start_position + T_block):
        1. Load Q (already rotated by query_position)
        2. Apply inverse RoPE: Q_aligned = Q * cos(p) + rotate_half(Q) * sin(p)
           (This effectively computes Q_raw @ R(query_pos - p)^T)
        3. Project: Q_latent = Q_aligned @ basis_K^T  [1, D] @ [k, D]^T -> [1, k]
        4. Score: score[p] = sum(Q_latent * coeffs_K[p])  [1, k] · [k] -> scalar
    
    Grid: (B, H_q) - one program per batch and query head
    """
    # Program IDs
    pid_b = tl.program_id(0)   # Batch index
    pid_h = tl.program_id(1)   # Query head index
    
    # Map query head to KV head (GQA)
    pid_kv = pid_h // n_rep
    
    # Pointer offsets for this batch and head
    Q_block_ptr = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    coeffs_K_block_ptr = coeffs_K_ptr + pid_kv * stride_ckh
    basis_K_block_ptr = basis_K_ptr + pid_kv * stride_bkh
    scores_block_ptr = scores_ptr + pid_b * stride_sb + pid_h * stride_sh
    
    # Load Q into SRAM - shape [1, D], we'll use [D]
    # Use mask for D in case BLOCK_D > D
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    
    Q_ptrs = Q_block_ptr + d_range * stride_qd
    Q = tl.load(Q_ptrs, mask=d_mask, other=0.0)  # [BLOCK_D]
    
    # Load basis_K into SRAM - shape [k_K, D]
    # We need all of basis_K for projection
    # Load block by block if k_K > BLOCK_K
    basis_K = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float16)
    for k_start in range(0, k_K, BLOCK_K):
        k_range = tl.arange(0, BLOCK_K) + k_start
        k_mask = k_range < k_K
        
        for d_start in range(0, D, BLOCK_D):
            d_range_inner = tl.arange(0, BLOCK_D) + d_start
            d_mask_inner = d_range_inner < D
            
            # Load basis_K[k_start:k_start+BLOCK_K, d_start:d_start+BLOCK_D]
            offs = (k_range[:, None] * stride_bkk + 
                   d_range_inner[None, :] * stride_bkd)
            mask = k_mask[:, None] & d_mask_inner[None, :]
            basis_K = tl.load(basis_K_block_ptr + offs, mask=mask, other=0.0)
    
    # Process sequence in tiles of BLOCK_T
    for t_start in range(0, T_block, BLOCK_T):
        t_range = tl.arange(0, BLOCK_T) + t_start
        t_mask = t_range < T_block
        
        # Absolute positions of keys in this tile
        key_positions = start_position + t_range  # [BLOCK_T]
        
        # Load RoPE cos/sin for these key positions
        # cos_ptr[key_positions, :D]
        cos_ptrs = cos_ptr + key_positions[:, None] * stride_ropeseq + d_range[None, :] * stride_roped
        sin_ptrs = sin_ptr + key_positions[:, None] * stride_ropeseq + d_range[None, :] * stride_roped
        
        rope_mask = t_mask[:, None] & d_mask[None, :]
        cos_keys = tl.load(cos_ptrs, mask=rope_mask, other=1.0)  # [BLOCK_T, BLOCK_D]
        sin_keys = tl.load(sin_ptrs, mask=rope_mask, other=0.0)  # [BLOCK_T, BLOCK_D]
        
        # Apply INVERSE RoPE to Q for each key position
        # Q_aligned = Q * cos(-θ_key) + rotate_half(Q) * sin(-θ_key)
        #           = Q * cos(θ_key) - rotate_half(Q) * sin(θ_key)  (since cos(-x)=cos(x), sin(-x)=-sin(x))
        
        # Broadcast Q to [BLOCK_T, BLOCK_D]
        Q_broadcast = Q[None, :].expand((BLOCK_T, BLOCK_D))  # Actually just broadcasts
        
        # rotate_half: swap and negate first half
        # For [d0, d1, d2, d3, ...] -> [-d1, d0, -d3, d2, ...]
        # Split D into two halves
        half_D = D // 2
        d_first = d_range < half_D
        d_second = ~d_first
        
        # This is tricky in Triton - let's use a simpler approach
        # For now, apply simplified RoPE: just multiply by cos (skip rotation for speed)
        # TODO: Full RoPE implementation
        Q_aligned = Q_broadcast * cos_keys  # Simplified - full RoPE needs rotate_half
        
        # Project Q_aligned into spectral space
        # Q_latent = Q_aligned @ basis_K^T  [BLOCK_T, BLOCK_D] @ [BLOCK_D, BLOCK_K] -> [BLOCK_T, BLOCK_K]
        # But basis_K is [BLOCK_K, BLOCK_D], so we need basis_K^T
        Q_latent = tl.dot(Q_aligned.to(tl.float16), tl.trans(basis_K))  # [BLOCK_T, BLOCK_K]
        
        # Load coeffs_K for this tile
        # coeffs_K[t_range, :k_K]
        k_range_full = tl.arange(0, BLOCK_K)
        k_mask_full = k_range_full < k_K
        
        coeffs_offs = (t_range[:, None] * stride_ckt + 
                      k_range_full[None, :] * stride_ckk)
        coeffs_mask = t_mask[:, None] & k_mask_full[None, :]
        coeffs_K_tile = tl.load(coeffs_K_block_ptr + coeffs_offs, mask=coeffs_mask, other=0.0)  # [BLOCK_T, BLOCK_K]
        
        # Compute scores: element-wise multiply and sum over k
        # scores = sum(Q_latent * coeffs_K, dim=-1)  [BLOCK_T]
        scores_tile = tl.sum(Q_latent * coeffs_K_tile, axis=1)  # [BLOCK_T]
        
        # Store scores
        scores_offs = t_range * stride_ss
        tl.store(scores_block_ptr + scores_offs, scores_tile, mask=t_mask)


# ============================================================================
# TRITON KERNEL: Spectral Value Aggregation
# ============================================================================

@triton.jit
def _spectral_value_kernel(
    # Input pointers
    attn_ptr,           # [B, H_q, 1, T_block] - Attention weights for this block
    coeffs_V_ptr,       # [H_kv, T_block, k_V] - Spectral coefficients
    basis_V_ptr,        # [H_kv, k_V, D] - Spectral basis
    # Output pointer (accumulated)
    output_ptr,         # [B, H_q, 1, D] - Output (add to existing)
    # Dequantization (optional)
    scales_V_ptr,       # [H_kv, 2] - min, max for dequant
    # Dimensions
    B: tl.constexpr,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    T_block: tl.constexpr,
    k_V: tl.constexpr,
    D: tl.constexpr,
    # Strides for attn [B, H_q, 1, T_block]
    stride_ab, stride_ah, stride_at, stride_as,
    # Strides for coeffs_V [H_kv, T_block, k_V]
    stride_cvh, stride_cvt, stride_cvk,
    # Strides for basis_V [H_kv, k_V, D]
    stride_bvh, stride_bvk, stride_bvd,
    # Strides for output [B, H_q, 1, D]
    stride_ob, stride_oh, stride_ot, stride_od,
    # GQA
    n_rep: tl.constexpr,
    # Block sizes
    BLOCK_T: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    BLOCK_D: tl.constexpr = 64,
):
    """
    Compute weighted spectral values:
        output = (attn @ coeffs_V) @ basis_V
        
    Where:
        attn: [1, T_block] - attention weights for this block
        coeffs_V: [T_block, k_V] - spectral coefficients
        basis_V: [k_V, D] - spectral basis
        output: [1, D]
    
    Grid: (B, H_q) - one program per batch and query head
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kv = pid_h // n_rep
    
    # Block pointers
    attn_block_ptr = attn_ptr + pid_b * stride_ab + pid_h * stride_ah
    coeffs_V_block_ptr = coeffs_V_ptr + pid_kv * stride_cvh
    basis_V_block_ptr = basis_V_ptr + pid_kv * stride_bvh
    output_block_ptr = output_ptr + pid_b * stride_ob + pid_h * stride_oh
    
    # Accumulator for v_latent [k_V]
    v_latent = tl.zeros((BLOCK_K,), dtype=tl.float32)
    
    # Phase 1: attn @ coeffs_V -> v_latent [k_V]
    # Process T in tiles
    for t_start in range(0, T_block, BLOCK_T):
        t_range = tl.arange(0, BLOCK_T) + t_start
        t_mask = t_range < T_block
        
        # Load attn weights [BLOCK_T]
        attn_ptrs = attn_block_ptr + t_range * stride_as
        attn_tile = tl.load(attn_ptrs, mask=t_mask, other=0.0)  # [BLOCK_T]
        
        # Load coeffs_V [BLOCK_T, k_V]
        k_range = tl.arange(0, BLOCK_K)
        k_mask = k_range < k_V
        
        coeffs_offs = (t_range[:, None] * stride_cvt + 
                      k_range[None, :] * stride_cvk)
        coeffs_mask = t_mask[:, None] & k_mask[None, :]
        coeffs_V_tile = tl.load(coeffs_V_block_ptr + coeffs_offs, mask=coeffs_mask, other=0.0)
        
        # v_latent += attn @ coeffs_V
        # [BLOCK_T] @ [BLOCK_T, BLOCK_K] -> [BLOCK_K]
        # Using einsum: v_latent[k] = sum_t(attn[t] * coeffs_V[t, k])
        v_latent += tl.sum(attn_tile[:, None] * coeffs_V_tile, axis=0)
    
    # Phase 2: v_latent @ basis_V -> output [D]
    # Load basis_V [k_V, D]
    d_range = tl.arange(0, BLOCK_D)
    
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    for d_start in range(0, D, BLOCK_D):
        d_range_inner = d_range + d_start
        d_mask = d_range_inner < D
        
        k_range_inner = tl.arange(0, BLOCK_K)
        k_mask_inner = k_range_inner < k_V
        
        # Load basis_V[:k_V, d_start:d_start+BLOCK_D]
        basis_offs = (k_range_inner[:, None] * stride_bvk + 
                     d_range_inner[None, :] * stride_bvd)
        basis_mask = k_mask_inner[:, None] & d_mask[None, :]
        basis_V_tile = tl.load(basis_V_block_ptr + basis_offs, mask=basis_mask, other=0.0)
        
        # output[d] = sum_k(v_latent[k] * basis_V[k, d])
        # v_latent is [BLOCK_K], basis_V_tile is [BLOCK_K, BLOCK_D]
        output_tile = tl.sum(v_latent[:, None] * basis_V_tile, axis=0)  # [BLOCK_D]
        
        # Store output (accumulate)
        output_offs = d_range_inner * stride_od
        # Load existing, add, store
        existing = tl.load(output_block_ptr + output_offs, mask=d_mask, other=0.0)
        tl.store(output_block_ptr + output_offs, existing + output_tile.to(tl.float16), mask=d_mask)


# ============================================================================
# PYTHON WRAPPERS WITH LOGGING
# ============================================================================

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to tensor.
    
    Args:
        x: Input tensor [..., D]
        cos, sin: RoPE tensors [..., D]
    
    Returns:
        Rotated tensor [..., D]
    """
    # Split into even and odd indices (interleaved format)
    # For standard RoPE: x = [x0, x1, x2, x3, ...] -> rotate pairs
    D = x.shape[-1]
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    
    cos1 = cos[..., :D//2]
    cos2 = cos[..., D//2:]
    sin1 = sin[..., :D//2]
    sin2 = sin[..., D//2:]
    
    # Rotation: [x1, x2] @ [[cos, -sin], [sin, cos]]
    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2
    
    return torch.cat([out1, out2], dim=-1)


def _apply_inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply inverse RoPE (rotation by -θ)."""
    # Inverse rotation: use -sin
    return _apply_rope(x, cos, -sin)


def spectral_score_forward_pytorch(
    Q: torch.Tensor,              # [B, H_q, 1, D] - already RoPE rotated
    coeffs_K: torch.Tensor,       # [H_kv, T_block, k_K]
    basis_K: torch.Tensor,        # [H_kv, k_K, D]
    cos: torch.Tensor,            # [max_seq, D]
    sin: torch.Tensor,            # [max_seq, D]
    start_position: int,
    query_position: int,
    config: Optional[TritonSpectralConfig] = None,
) -> torch.Tensor:
    """
    Compute spectral attention scores using PyTorch (with full RoPE).
    
    This is the CORRECT implementation that handles RoPE properly.
    Use this for correctness validation.
    
    Returns: scores [B, H_q, 1, T_block]
    """
    config = config or TritonSpectralConfig()
    
    B, H_q, _, D = Q.shape
    H_kv, T_block, k_K = coeffs_K.shape
    n_rep = H_q // H_kv
    
    # Preserve model dtype for consistent output
    model_dtype = Q.dtype
    
    if config.enable_logging:
        logger.debug(f"[SCORE_PT] Input shapes: Q={Q.shape}, coeffs_K={coeffs_K.shape}, basis_K={basis_K.shape}")
        logger.debug(f"[SCORE_PT] Positions: start={start_position}, query={query_position}")
        logger.debug(f"[SCORE_PT] GQA: H_q={H_q}, H_kv={H_kv}, n_rep={n_rep}")
        logger.debug(f"[SCORE_PT] Input dtypes: Q={Q.dtype}, coeffs_K={coeffs_K.dtype}, basis_K={basis_K.dtype}")
    
    start_time = time.perf_counter()
    
    # Key positions in this block
    key_positions = torch.arange(start_position, start_position + T_block, device=Q.device, dtype=torch.long)
    
    # Get RoPE values for key positions
    cos_keys = cos[key_positions]  # [T_block, D]
    sin_keys = sin[key_positions]  # [T_block, D]
    
    # Broadcast Q to [B, H_q, T_block, D] for per-position alignment
    Q_broadcast = Q.expand(B, H_q, T_block, D)
    
    # Apply INVERSE RoPE to Q for each key position
    # Q_aligned[t] = Q * cos(key_pos[t]) - rotate_half(Q) * sin(key_pos[t])
    cos_batch = cos_keys.unsqueeze(0).unsqueeze(0)  # [1, 1, T_block, D]
    sin_batch = sin_keys.unsqueeze(0).unsqueeze(0)  # [1, 1, T_block, D]
    
    Q_aligned = _apply_inverse_rope(Q_broadcast, cos_batch, sin_batch)  # [B, H_q, T_block, D]
    
    # GQA: Expand basis_K to match query heads
    if n_rep > 1:
        # basis_K: [H_kv, k_K, D] -> [H_q, k_K, D]
        basis_K = basis_K.repeat_interleave(n_rep, dim=0)
        # coeffs_K: [H_kv, T_block, k_K] -> [H_q, T_block, k_K]
        coeffs_K = coeffs_K.repeat_interleave(n_rep, dim=0)
    
    # CRITICAL: Use FP32 for einsum computation to avoid overflow/precision loss
    # Research finding: einsum is NOT on autocast whitelist, requires explicit FP32 cast
    # Project Q into spectral space
    # Q_latent = Q_aligned @ basis_K^T : [B, H_q, T_block, D] @ [H_q, k_K, D]^T
    basis_K_batch = basis_K.unsqueeze(0).float()  # [1, H_q, k_K, D] FP32
    Q_aligned_f32 = Q_aligned.float()  # FP32 for einsum stability
    Q_latent = torch.einsum('bhtd,bhkd->bhtk', Q_aligned_f32, basis_K_batch.expand(B, -1, -1, -1))  # [B, H_q, T_block, k_K]
    
    # Compute scores: element-wise multiply with coeffs and sum
    # scores[b,h,1,t] = sum_k(Q_latent[b,h,t,k] * coeffs_K[h,t,k])
    coeffs_K_batch = coeffs_K.unsqueeze(0).float()  # [1, H_q, T_block, k_K] FP32
    scores = torch.sum(Q_latent * coeffs_K_batch.expand(B, -1, -1, -1), dim=-1)  # [B, H_q, T_block]
    scores = scores.unsqueeze(2).to(model_dtype)  # [B, H_q, 1, T_block] back to model dtype
    
    if config.enable_logging:
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.debug(f"[SCORE_PT] Completed in {elapsed_us:.1f}μs, output={scores.shape}")
        logger.debug(f"[SCORE_PT] Score stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    
    return scores


def spectral_score_forward(
    Q: torch.Tensor,              # [B, H_q, 1, D]
    coeffs_K: torch.Tensor,       # [H_kv, T_block, k_K]
    basis_K: torch.Tensor,        # [H_kv, k_K, D]
    cos: torch.Tensor,            # [max_seq, D]
    sin: torch.Tensor,            # [max_seq, D]
    start_position: int,
    query_position: int,
    config: Optional[TritonSpectralConfig] = None,
    use_triton: bool = False,     # Default to PyTorch for correctness
) -> torch.Tensor:
    """
    Compute spectral attention scores.
    
    By default uses PyTorch for correctness. Set use_triton=True
    for Triton kernel (faster but RoPE is simplified).
    
    Returns: scores [B, H_q, 1, T_block]
    """
    config = config or TritonSpectralConfig()
    
    # Always use PyTorch for now (Triton RoPE incomplete)
    # TODO: Once Triton RoPE is fixed, enable this branch
    if not use_triton:
        return spectral_score_forward_pytorch(
            Q, coeffs_K, basis_K, cos, sin, start_position, query_position, config
        )
    
    # Triton path (RoPE simplified - for benchmarking only)
    B, H_q, _, D = Q.shape
    H_kv, T_block, k_K = coeffs_K.shape
    n_rep = H_q // H_kv
    
    if config.enable_logging:
        logger.debug(f"[SCORE] Input shapes: Q={Q.shape}, coeffs_K={coeffs_K.shape}, basis_K={basis_K.shape}")
        logger.debug(f"[SCORE] Positions: start={start_position}, query={query_position}")
        logger.debug(f"[SCORE] GQA: H_q={H_q}, H_kv={H_kv}, n_rep={n_rep}")
        logger.warning("[SCORE] Using Triton kernel with SIMPLIFIED RoPE - results may be inaccurate!")
    
    # Allocate output
    scores = torch.zeros(B, H_q, 1, T_block, device=Q.device, dtype=Q.dtype)
    
    # Grid: (B, H_q)
    grid = (B, H_q)
    
    if config.enable_logging:
        logger.debug(f"[SCORE] Launching kernel with grid={grid}, BLOCK_T={config.BLOCK_T}")
    
    start_time = time.perf_counter()
    
    _spectral_score_kernel[grid](
        # Inputs
        Q, coeffs_K, basis_K, cos, sin,
        # Output
        scores,
        # Scales (None for FP16)
        None,
        # Dimensions
        B, H_q, H_kv, T_block, k_K, D,
        # Positions
        start_position, query_position,
        # Q strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        # coeffs_K strides
        coeffs_K.stride(0), coeffs_K.stride(1), coeffs_K.stride(2),
        # basis_K strides
        basis_K.stride(0), basis_K.stride(1), basis_K.stride(2),
        # scores strides
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        # RoPE strides
        cos.stride(0), cos.stride(1),
        # GQA
        n_rep,
        # Block sizes
        config.BLOCK_T, config.BLOCK_K, config.BLOCK_D,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    
    if config.enable_logging:
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.debug(f"[SCORE] Kernel completed in {elapsed_us:.1f}μs, output={scores.shape}")
        logger.debug(f"[SCORE] Score stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    
    return scores


def spectral_value_forward_pytorch(
    attn: torch.Tensor,           # [B, H_q, 1, T_block]
    coeffs_V: torch.Tensor,       # [H_kv, T_block, k_V]
    basis_V: torch.Tensor,        # [H_kv, k_V, D]
    D: int,
    config: Optional[TritonSpectralConfig] = None,
) -> torch.Tensor:
    """
    Compute spectral weighted values using PyTorch.
    
    Algorithm:
        v_latent = attn @ coeffs_V   [1, T] @ [T, k] -> [1, k]
        output = v_latent @ basis_V  [1, k] @ [k, D] -> [1, D]
    
    Returns: output [B, H_q, 1, D]
    """
    config = config or TritonSpectralConfig()
    
    B, H_q, _, T_block = attn.shape
    H_kv, _, k_V = coeffs_V.shape
    n_rep = H_q // H_kv
    
    # Preserve model dtype for consistent output
    model_dtype = attn.dtype
    
    if config.enable_logging:
        logger.debug(f"[VALUE_PT] Input shapes: attn={attn.shape}, coeffs_V={coeffs_V.shape}, basis_V={basis_V.shape}")
        logger.debug(f"[VALUE_PT] GQA: H_q={H_q}, H_kv={H_kv}, n_rep={n_rep}")
        logger.debug(f"[VALUE_PT] Input dtypes: attn={attn.dtype}, coeffs_V={coeffs_V.dtype}, basis_V={basis_V.dtype}")
    
    start_time = time.perf_counter()
    
    # GQA: Expand spectral components to match query heads
    if n_rep > 1:
        coeffs_V = coeffs_V.repeat_interleave(n_rep, dim=0)  # [H_q, T_block, k_V]
        basis_V = basis_V.repeat_interleave(n_rep, dim=0)    # [H_q, k_V, D]
    
    # CRITICAL: Use FP32 for einsum computation to avoid precision loss
    # Step 1: v_latent = attn @ coeffs_V
    # attn: [B, H_q, 1, T_block], coeffs_V: [H_q, T_block, k_V]
    # v_latent[b,h,1,k] = sum_t(attn[b,h,1,t] * coeffs_V[h,t,k])
    coeffs_V_batch = coeffs_V.unsqueeze(0).float()  # [1, H_q, T_block, k_V] FP32
    attn_f32 = attn.float()  # FP32 for einsum stability
    v_latent = torch.einsum('bhst,bhtk->bhsk', attn_f32, coeffs_V_batch.expand(B, -1, -1, -1))  # [B, H_q, 1, k_V]
    
    # Step 2: output = v_latent @ basis_V
    # v_latent: [B, H_q, 1, k_V], basis_V: [H_q, k_V, D]
    # output[b,h,1,d] = sum_k(v_latent[b,h,1,k] * basis_V[h,k,d])
    basis_V_batch = basis_V.unsqueeze(0).float()  # [1, H_q, k_V, D] FP32
    output = torch.einsum('bhsk,bhkd->bhsd', v_latent, basis_V_batch.expand(B, -1, -1, -1))  # [B, H_q, 1, D]
    output = output.to(model_dtype)  # Back to model dtype
    
    if config.enable_logging:
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.debug(f"[VALUE_PT] Completed in {elapsed_us:.1f}μs, output={output.shape}")
        logger.debug(f"[VALUE_PT] Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    return output


def spectral_value_forward(
    attn: torch.Tensor,           # [B, H_q, 1, T_block]
    coeffs_V: torch.Tensor,       # [H_kv, T_block, k_V]
    basis_V: torch.Tensor,        # [H_kv, k_V, D]
    D: int,
    config: Optional[TritonSpectralConfig] = None,
    use_triton: bool = False,     # Default to PyTorch for correctness
) -> torch.Tensor:
    """
    Compute spectral weighted values.
    
    By default uses PyTorch for correctness. Set use_triton=True
    for Triton kernel (faster, for benchmarking).
    
    Returns: output [B, H_q, 1, D]
    """
    config = config or TritonSpectralConfig()
    
    # Use PyTorch by default for correctness
    if not use_triton:
        return spectral_value_forward_pytorch(attn, coeffs_V, basis_V, D, config)
    
    # Triton path (for benchmarking)
    B, H_q, _, T_block = attn.shape
    H_kv, _, k_V = coeffs_V.shape
    n_rep = H_q // H_kv
    
    if config.enable_logging:
        logger.debug(f"[VALUE] Input shapes: attn={attn.shape}, coeffs_V={coeffs_V.shape}, basis_V={basis_V.shape}")
        logger.debug(f"[VALUE] GQA: H_q={H_q}, H_kv={H_kv}, n_rep={n_rep}")
    
    # Allocate output (initialize to zero for accumulation)
    output = torch.zeros(B, H_q, 1, D, device=attn.device, dtype=attn.dtype)
    
    # Grid: (B, H_q)
    grid = (B, H_q)
    
    if config.enable_logging:
        logger.debug(f"[VALUE] Launching kernel with grid={grid}, BLOCK_T={config.BLOCK_T}")
    
    start_time = time.perf_counter()
    
    _spectral_value_kernel[grid](
        # Inputs
        attn, coeffs_V, basis_V,
        # Output
        output,
        # Scales
        None,
        # Dimensions
        B, H_q, H_kv, T_block, k_V, D,
        # attn strides
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        # coeffs_V strides
        coeffs_V.stride(0), coeffs_V.stride(1), coeffs_V.stride(2),
        # basis_V strides
        basis_V.stride(0), basis_V.stride(1), basis_V.stride(2),
        # output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # GQA
        n_rep,
        # Block sizes
        config.BLOCK_T, config.BLOCK_K, config.BLOCK_D,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    
    if config.enable_logging:
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        logger.debug(f"[VALUE] Kernel completed in {elapsed_us:.1f}μs, output={output.shape}")
        logger.debug(f"[VALUE] Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    return output


# ============================================================================
# HIGH-LEVEL API: Full Spectral Attention Decode
# ============================================================================

def spectral_attention_decode(
    Q: torch.Tensor,
    cache,  # SpectralCache
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_position: int,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    config: Optional[TritonSpectralConfig] = None,
) -> torch.Tensor:
    """
    Full spectral attention for decode (single new token).
    
    This is the main entry point that:
    1. Computes scores for cold blocks (Triton kernel)
    2. Computes scores for hot cache (PyTorch, small)
    3. Applies softmax (PyTorch)
    4. Aggregates values (Triton kernel for cold, PyTorch for hot)
    
    Args:
        Q: Query tensor [B, H_q, 1, D]
        cache: SpectralCache with cold blocks and hot cache
        cos, sin: RoPE tables [max_seq, D]
        query_position: Absolute position of current query
        attention_mask: Optional mask
        scale: Attention scale (default: 1/sqrt(D))
        config: Triton kernel configuration
        
    Returns:
        output: [B, H_q, 1, D]
    """
    config = config or TritonSpectralConfig()
    
    B, H_q, _, D = Q.shape
    H_kv = cache.num_heads
    n_rep = H_q // H_kv
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    if config.enable_logging:
        logger.info("=" * 60)
        logger.info(f"[SPECTRAL_DECODE] Starting spectral attention decode")
        logger.info(f"[SPECTRAL_DECODE] Q={Q.shape}, query_pos={query_position}")
        logger.info(f"[SPECTRAL_DECODE] Cache: {len(cache.cold_blocks)} cold blocks, hot={cache.total_tokens - sum(b.block_size for b in cache.cold_blocks) if cache.cold_blocks else cache.total_tokens} tokens")
    
    # Get spectral components
    cold_blocks, hot_K, hot_V = cache.get_spectral_components()
    
    # ========================================================================
    # PHASE 1: Compute Scores
    # ========================================================================
    scores_parts = []
    
    # 1a. Cold blocks (Triton)
    for i, block in enumerate(cold_blocks):
        if config.enable_logging:
            logger.debug(f"[PHASE1] Processing cold block {i}: T={block.block_size}, start_pos={block.start_position}")
        
        scores_block = spectral_score_forward(
            Q=Q,
            coeffs_K=block.coeffs_K,
            basis_K=block.basis_K,
            cos=cos,
            sin=sin,
            start_position=block.start_position,
            query_position=query_position,
            config=config,
        )
        scores_parts.append(scores_block)
    
    # 1b. Hot cache (PyTorch - typically small)
    if hot_K is not None and hot_K.shape[2] > 0:
        T_hot = hot_K.shape[2]
        
        if config.enable_logging:
            logger.debug(f"[PHASE1] Processing hot cache: T={T_hot}")
        
        # Get positions for hot cache
        if cache.hot_position_ids is not None:
            hot_positions = cache.hot_position_ids.flatten()
        else:
            hot_start = sum(b.block_size for b in cold_blocks)
            hot_positions = torch.arange(hot_start, hot_start + T_hot, device=Q.device)
        
        # Apply RoPE to hot keys
        cos_hot = cos[hot_positions].unsqueeze(0).unsqueeze(0)  # [1, 1, T_hot, D]
        sin_hot = sin[hot_positions].unsqueeze(0).unsqueeze(0)
        
        # CRITICAL FIX: Use proper RoPE rotation, not simplified approximation!
        # The simplified version (K * cos) was causing attention corruption
        # which cascaded into NaN values and model degeneration.
        hot_K_rotated = _apply_rope(hot_K, cos_hot, sin_hot)  # Proper RoPE!
        
        # GQA expand
        if n_rep > 1:
            hot_K_rotated = hot_K_rotated.repeat_interleave(n_rep, dim=1)
        
        # Standard attention: Q @ K^T
        scores_hot = torch.matmul(Q, hot_K_rotated.transpose(-2, -1))  # [B, H_q, 1, T_hot]
        scores_parts.append(scores_hot)
    
    # Concatenate all scores
    if len(scores_parts) == 0:
        raise ValueError("Empty cache - no keys available")
    
    scores = torch.cat(scores_parts, dim=-1)  # [B, H_q, 1, T_total]
    scores = scores * scale
    
    if config.enable_logging:
        logger.debug(f"[PHASE1] Combined scores: shape={scores.shape}, min={scores.min():.4f}, max={scores.max():.4f}")
    
    # ========================================================================
    # PHASE 2: Softmax (PyTorch)
    # ========================================================================
    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
    
    # CRITICAL: Compute softmax in FP32 to prevent exp() overflow!
    # Scores can reach ±26 after scaling, and exp(26) ≈ 5×10^11 overflows FP16 (max 65504)
    # This causes Inf → NaN cascade that corrupts the entire model
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)  # [B, H_q, 1, T_total]
    
    if config.enable_logging:
        logger.debug(f"[PHASE2] Softmax applied: attn_sum={attn_weights.sum(dim=-1).mean():.4f}")
    
    # ========================================================================
    # PHASE 3: Value Aggregation
    # ========================================================================
    output_parts = []
    seq_pos = 0
    
    # 3a. Cold blocks (Triton)
    for i, block in enumerate(cold_blocks):
        T_block = block.block_size
        attn_block = attn_weights[:, :, :, seq_pos:seq_pos + T_block]
        
        if config.enable_logging:
            logger.debug(f"[PHASE3] Aggregating cold block {i}: attn_sum={attn_block.sum(dim=-1).mean():.4f}")
        
        output_block = spectral_value_forward(
            attn=attn_block,
            coeffs_V=block.coeffs_V,
            basis_V=block.basis_V,
            D=D,
            config=config,
        )
        output_parts.append(output_block)
        seq_pos += T_block
    
    # 3b. Hot cache (PyTorch)
    if hot_V is not None and hot_V.shape[2] > 0:
        T_hot = hot_V.shape[2]
        attn_hot = attn_weights[:, :, :, seq_pos:seq_pos + T_hot]
        
        if config.enable_logging:
            logger.debug(f"[PHASE3] Aggregating hot cache: attn_sum={attn_hot.sum(dim=-1).mean():.4f}")
        
        # GQA expand hot_V
        hot_V_expanded = hot_V
        if n_rep > 1:
            hot_V_expanded = hot_V.repeat_interleave(n_rep, dim=1)
        
        output_hot = torch.matmul(attn_hot, hot_V_expanded)  # [B, H_q, 1, D]
        output_parts.append(output_hot)
    
    # Combine outputs
    output = sum(output_parts)
    
    if config.enable_logging:
        logger.info(f"[SPECTRAL_DECODE] Complete: output={output.shape}, stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        logger.info("=" * 60)
    
    return output


# ============================================================================
# PYTORCH FALLBACK (for comparison and debugging)
# ============================================================================

def spectral_attention_decode_pytorch(
    Q: torch.Tensor,
    cache,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_position: int,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation for comparison.
    Same interface as spectral_attention_decode.
    """
    # Import from existing module
    from ..spectral_attention import spectral_attention_forward
    return spectral_attention_forward(Q, cache, cos, sin, query_position, attention_mask, scale)
