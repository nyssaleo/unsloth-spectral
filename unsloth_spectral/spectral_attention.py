"""
Spectral Attention: Direct attention computation in compressed space.

This module implements the core innovation: computing attention scores and
weighted values WITHOUT reconstructing the full K/V matrices.

Algorithm:
---------
Standard Attention:
    scores = Q @ K^T                    # O(T×D) memory bandwidth
    attn = softmax(scores)
    output = attn @ V                   # O(T×D) memory bandwidth

Spectral Attention (Dual Projection):
    # Phase 1: Compute scores
    scores_cold = (Q @ B_K^T) @ C_K^T   # O(k×D + k×T) - k<<D
    scores_hot = Q @ K_hot^T            # Standard for recent tokens
    scores = concat([scores_cold, scores_hot])
    
    # Phase 2: Weighted values
    attn = softmax(scores)
    attn_cold, attn_hot = split(attn)
    v_cold = (attn_cold @ C_V) @ B_V    # O(k×T + k×D)
    v_hot = attn_hot @ V_hot            # Standard
    output = v_cold + v_hot

Memory Bandwidth: O(k×(T+D)) vs O(T×D), where k=16-32, T=4096+
Speedup: ~8x for long contexts (k=16, D=128, T>>D)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .spectral_cache import SpectralCache, SpectralBlock
import math


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads for Grouped Query Attention (GQA).
    Handles standard [B, H_kv, T, D] tensors.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_spectral(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat spectral components for Grouped Query Attention (GQA).
    Handles [H_kv, ...] shaped tensors (no batch dimension at start).
    
    Args:
        x: Spectral tensor [H_kv, ...] where ... can be [T, k] or [k, D]
        n_rep: Number of repetitions (num_q_heads // num_kv_heads)
        
    Returns:
        Expanded tensor [H_q, ...] where H_q = H_kv * n_rep
        
    Example:
        coeffs_K: [8, 256, 16] -> [32, 256, 16] with n_rep=4
        basis_K:  [8, 16, 128] -> [32, 16, 128] with n_rep=4
    """
    if n_rep == 1:
        return x
    
    num_kv_heads = x.shape[0]
    # Insert dimension and expand: [8, ...] -> [8, 1, ...] -> [8, 4, ...]
    x = x.unsqueeze(1)
    sizes = list(x.shape)
    sizes[1] = n_rep
    x = x.expand(*sizes)
    
    # Reshape: [8, 4, ...] -> [32, ...]
    return x.reshape(num_kv_heads * n_rep, *x.shape[2:])


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input tensor.
    
    This implements the standard RoPE transformation where each consecutive
    pair of dimensions is rotated by an angle that depends on the position.
    
    Args:
        x: Input tensor [..., D] where D is head dimension
        cos: Cosine values [..., D] matching x's shape
        sin: Sine values [..., D] matching x's shape
        
    Returns:
        Rotated tensor with same shape as x
        
    Math:
        For each pair (x[2i], x[2i+1]):
        out[2i]   = x[2i]   * cos[2i] - x[2i+1] * sin[2i]
        out[2i+1] = x[2i+1] * cos[2i] + x[2i]   * sin[2i]
    """
    # Split into first and second half of dimensions
    d = x.shape[-1]
    x1 = x[..., :d//2]  # First half
    x2 = x[..., d//2:]  # Second half
    
    # Extract corresponding cos/sin values
    cos1 = cos[..., :d//2]
    sin1 = sin[..., :d//2]
    
    # Rotate: [x1, x2] * [cos, sin] = [x1*cos - x2*sin, x2*cos + x1*sin]
    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos1 + x1 * sin1
    
    # Concatenate back
    return torch.cat([out1, out2], dim=-1)


def spectral_attention_forward(
    Q: torch.Tensor,
    cache: SpectralCache,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_position: int,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention using spectral cache with RoPE-aware projection.
    
    This implements the "Latent-Space Relative Rotation" technique:
    Instead of reconstructing full keys and applying RoPE, we:
    1. Apply INVERSE RoPE to the query for each key position
    2. Project the aligned queries into the spectral basis
    3. Compute scores in the latent space
    
    This avoids the memory bandwidth cost of reconstructing full keys while
    correctly handling position-dependent RoPE rotations.
    
    Args:
        Q: Query tensor [B, H, 1, D] (already rotated by query_position)
        cache: SpectralCache containing PRE-RoPE spectral blocks + hot buffer  
        cos: RoPE cosine table [MaxLen, D]
        sin: RoPE sine table [MaxLen, D]
        query_position: Absolute position of the query token (for relative rotation)
        attention_mask: Optional mask [B, H, 1, T_total]
        scale: Attention scale factor (default: 1/sqrt(D))
        
    Returns:
        output: Attention output [B, H, 1, D]
    """
    B, H_q, _, D = Q.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Get spectral components
    cold_blocks, hot_K, hot_V = cache.get_spectral_components()
    
    # Calculate GQA repetition factor
    # cache.num_heads is num_key_value_heads (e.g., 8 for Mistral)
    # H_q is num_attention_heads (e.g., 32 for Mistral)
    H_kv = cache.num_heads
    n_rep = H_q // H_kv
    
    # PHASE 1: Compute Attention Scores
    # ===================================
    
    scores_parts = []
    
    # 1a. Spectral attention scores (cold cache) with RoPE correction
    # ================================================================
    # Implement "Latent-Space Relative Rotation" technique
    for block in cold_blocks:
        # Extract spectral components for this block
        coeffs_K = block.coeffs_K  # [H_kv, T_block, k]
        basis_K = block.basis_K    # [H_kv, k, D]
        T_block = block.block_size
        start_pos = block.start_position
        
        # GQA: Expand spectral components to match query heads
        # [H_kv, ...] -> [H_q, ...]
        if n_rep > 1:
            coeffs_K = repeat_kv_spectral(coeffs_K, n_rep)  # [H_q, T_block, k]
            basis_K = repeat_kv_spectral(basis_K, n_rep)    # [H_q, k, D]
        
        # Step 1: Identify ABSOLUTE positions of keys in this block
        # Key positions: [start_pos, start_pos+1, ..., start_pos+T_block-1]
        key_positions = torch.arange(start_pos, start_pos + T_block, device=Q.device, dtype=torch.long)
        
        # Step 2: Extract RoPE cos/sin for KEY'S ABSOLUTE POSITIONS
        # CRITICAL: We use key_positions directly, NOT distances!
        # This applies R(-θ_n) to Q, transforming it to align with unrotated keys
        # Net effect: R(-θ_n) @ R(θ_m) @ Q_raw = R(θ_m - θ_n) @ Q_raw ✅
        #
        # Previous bug: Used distances = query_position - key_positions
        # This applied R(-θ_{m-n}), resulting in R(θ_n) - WRONG! Destroyed relative info.
        
        # Ensure we don't index out of bounds
        max_idx = cos.shape[0]
        if (key_positions >= max_idx).any():
            # Safety clamp (shouldn't happen if RoPE cache managed correctly)
            key_positions = key_positions.clamp(max=max_idx-1)
        
        cos_keys = cos[key_positions]  # [T_block, D] - cos(θ_n) for each key
        sin_keys = sin[key_positions]  # [T_block, D] - sin(θ_n) for each key
        
        # Step 3: Apply INVERSE RoPE using key's absolute position
        # Q is [B, H, 1, D] - already rotated by θ_query_position
        # We rotate it by -θ_key_position to align with unrotated keys
        
        # Broadcast Q to [B, H_q, T_block, D]
        Q_broadcast = Q.expand(B, H_q, T_block, D)
        
        # Broadcast cos/sin to [B, H, T_block, D]
        cos_batch = cos_keys.unsqueeze(0).unsqueeze(0)  # [1, 1, T_block, D]
        sin_batch = sin_keys.unsqueeze(0).unsqueeze(0)  # [1, 1, T_block, D]
        
        # Apply inverse rotation (note: -sin for inverse)
        Q_aligned = apply_rope(Q_broadcast, cos_batch, -sin_batch)  # [B, H, T_block, D]
        
        # Step 4: Project aligned queries into spectral space
        # Q_aligned @ B_K^T: [B, H, T_block, D] @ [H, k, D]^T -> [B, H, T_block, k]
        basis_K_batched = basis_K.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, k, D]
        
        # CRITICAL: Use FP32 for einsum (NOT on autocast whitelist)
        # Research finding: einsum with mixed FP16/FP32 can overflow or produce incorrect results
        # Compute in FP32 for numerical stability, then convert back to model dtype
        model_dtype = Q.dtype
        Q_aligned_fp32 = Q_aligned.float()
        basis_K_batched_fp32 = basis_K_batched.float()
        
        # Use einsum for clarity: bhTd,bhkd->bhTk
        Q_latent = torch.einsum('bhtd,bhkd->bhtk', Q_aligned_fp32, basis_K_batched_fp32)  # [B, H, T_block, k]
        
        # Step 5: Compute scores via element-wise dot product with coefficients
        # Q_latent * C_K: [B, H, T_block, k] * [H, T_block, k] -> sum over k -> [B, H, T_block]
        coeffs_K_batched = coeffs_K.unsqueeze(0).expand(B, -1, -1, -1).float()  # [B, H, T_block, k] in FP32
        scores_block = torch.sum(Q_latent * coeffs_K_batched, dim=-1)  # [B, H, T_block]
        scores_block = scores_block.to(model_dtype)  # Convert back to model dtype
        scores_block = scores_block.unsqueeze(2)  # [B, H, 1, T_block] for concatenation
        
        scores_parts.append(scores_block)
    
    # 1b. Standard attention scores (hot cache with RoPE)
    # ====================================================
    # Hot cache stores PRE-RoPE keys, so we need to apply RoPE first
    if hot_K is not None:
        T_hot = hot_K.shape[2]
        
        # Get positions for hot cache keys
        if cache.hot_position_ids is not None:
            hot_positions = cache.hot_position_ids.flatten()  # [T_hot]
        else:
            # Fallback: assume hot cache is at the end
            hot_start = sum(block.block_size for block in cold_blocks)
            hot_positions = torch.arange(hot_start, hot_start + T_hot, device=Q.device, dtype=torch.long)
        
        # Apply RoPE to hot keys at their positions
        # Extract cos/sin for hot positions
        cos_hot = cos[hot_positions]  # [T_hot, D]
        sin_hot = sin[hot_positions]  # [T_hot, D]
        
        # Broadcast for batch and heads: [B, H, T_hot, D]
        cos_hot_batch = cos_hot.unsqueeze(0).unsqueeze(0)
        sin_hot_batch = sin_hot.unsqueeze(0).unsqueeze(0)
        
        # Apply RoPE to hot keys
        hot_K_rotated = apply_rope(hot_K, cos_hot_batch, sin_hot_batch)  # [B, H_kv, T_hot, D]
        
        # GQA: Expand hot keys to match query heads
        if n_rep > 1:
            hot_K_rotated = repeat_kv(hot_K_rotated, n_rep)  # [B, H_q, T_hot, D]
        
        # Standard attention: Q @ K^T
        scores_hot = torch.matmul(Q, hot_K_rotated.transpose(-2, -1))  # [B, H_q, 1, T_hot]
        scores_parts.append(scores_hot)
    
    # Concatenate all scores
    if len(scores_parts) == 0:
        raise ValueError("Empty cache - no keys available")
    
    scores = torch.cat(scores_parts, dim=-1)  # [B, H, 1, T_total]
    scores = scores * scale
    
    # Apply attention mask if provided
    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, T_total]
    
    # PHASE 2: Compute Weighted Values
    # ==================================
    
    output_parts = []
    seq_pos = 0
    
    # 2a. Spectral weighted values (cold cache)
    for block in cold_blocks:
        # Extract spectral components for this block
        coeffs_V = block.coeffs_V  # [H_kv, T_block, k]
        basis_V = block.basis_V    # [H_kv, k, D]
        T_block = block.block_size
        
        # GQA: Expand spectral components to match query heads
        if n_rep > 1:
            coeffs_V = repeat_kv_spectral(coeffs_V, n_rep)  # [H_q, T_block, k]
            basis_V = repeat_kv_spectral(basis_V, n_rep)    # [H_q, k, D]
        
        # Extract attention weights for this block
        attn_block = attn_weights[:, :, :, seq_pos:seq_pos+T_block]  # [B, H_q, 1, T_block]
        seq_pos += T_block
        
        # Dual projection: (attn @ C_V) @ B_V
        # attn: [B, H, 1, T_block]
        # C_V: [H, T_block, k]
        # B_V: [H, k, D]
        
        # CRITICAL: Use FP32 for matmul to avoid numerical issues
        # Research finding: FP16 dot products can overflow for large dimensions
        model_dtype = Q.dtype
        
        # Step 1: Aggregate temporal coefficients
        # attn @ C_V: [B, H, 1, T_block] @ [H, T_block, k] -> [B, H, 1, k]
        coeffs_V_batched = coeffs_V.unsqueeze(0).expand(B, -1, -1, -1).float()  # [B, H, T_block, k] FP32
        attn_block_fp32 = attn_block.float()
        v_proj = torch.matmul(attn_block_fp32, coeffs_V_batched)  # [B, H, 1, k]
        
        # Step 2: Project back to feature space
        # v_proj @ B_V: [B, H, 1, k] @ [H, k, D] -> [B, H, 1, D]
        basis_V_batched = basis_V.unsqueeze(0).expand(B, -1, -1, -1).float()  # [B, H, k, D] FP32
        output_block = torch.matmul(v_proj, basis_V_batched)  # [B, H, 1, D]
        output_block = output_block.to(model_dtype)  # Convert back to model dtype
        
        output_parts.append(output_block)
    
    # 2b. Standard weighted values (hot cache)
    if hot_V is not None:
        T_hot = hot_V.shape[2]
        attn_hot = attn_weights[:, :, :, seq_pos:seq_pos+T_hot]  # [B, H_q, 1, T_hot]
        
        # GQA: Expand hot values to match query heads
        hot_V_expanded = hot_V
        if n_rep > 1:
            hot_V_expanded = repeat_kv(hot_V, n_rep)  # [B, H_q, T_hot, D]
        
        # Standard: attn @ V
        output_hot = torch.matmul(attn_hot, hot_V_expanded)  # [B, H_q, 1, D]
        output_parts.append(output_hot)
    
    # Sum all outputs (spectral + hot)
    output = sum(output_parts)  # [B, H, 1, D]
    
    return output


def validate_spectral_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cache: SpectralCache,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Tuple[bool, dict]:
    """
    Validate that spectral attention produces same results as standard attention.
    
    This is a critical correctness test: we compare the spectral attention output
    against the standard (Q @ K^T) @ V computation.
    
    Args:
        Q: Query [B, H, 1, D]
        K: Full keys [B, H, T, D] (for reference)
        V: Full values [B, H, T, D] (for reference)
        cache: SpectralCache (should contain compressed K, V)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        is_valid: True if outputs match within tolerance
        metrics: Dictionary of error metrics
    """
    B, H, _, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Standard attention (reference)
    scores_ref = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, 1, T]
    attn_ref = F.softmax(scores_ref, dim=-1)
    output_ref = torch.matmul(attn_ref, V)  # [B, H, 1, D]
    
    # Spectral attention (test)
    output_spectral = spectral_attention_forward(Q, cache, scale=scale)
    
    # Compute errors
    abs_error = torch.abs(output_ref - output_spectral).max().item()
    rel_error = (abs_error / (torch.abs(output_ref).max().item() + 1e-8))
    mean_abs_error = torch.abs(output_ref - output_spectral).mean().item()
    
    # Attention score correlation
    # Recompute scores for comparison
    cold_blocks, hot_K, hot_V = cache.get_spectral_components()
    K_recon, _ = cache.get_kv()
    scores_spectral = torch.matmul(Q, K_recon.transpose(-2, -1)) * scale
    
    scores_ref_flat = scores_ref.flatten()
    scores_spectral_flat = scores_spectral.flatten()
    correlation = torch.corrcoef(torch.stack([scores_ref_flat, scores_spectral_flat]))[0, 1].item()
    
    is_valid = (abs_error < atol) or (rel_error < rtol)
    
    metrics = {
        "max_abs_error": abs_error,
        "max_rel_error": rel_error,
        "mean_abs_error": mean_abs_error,
        "attention_correlation": correlation,
        "is_valid": is_valid,
    }
    
    return is_valid, metrics


def test_spectral_attention():
    """Unit test for spectral attention correctness."""
    print("="*60)
    print("Testing Spectral Attention")
    print("="*60)
    
    # Setup
    B, H, T, D = 1, 8, 1024, 128
    device = "cpu"
    
    # Create cache and populate it
    cache = SpectralCache(
        num_heads=H,
        head_dim=D,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        device=device,
        dtype=torch.float32,
    )
    
    # Generate synthetic K, V
    K_full = torch.randn(B, H, T, D, device=device)
    V_full = torch.randn(B, H, T, D, device=device)
    
    # Populate cache in blocks
    for i in range(0, T, 256):
        end = min(i + 256, T)
        cache.append(K_full[:, :, i:end, :], V_full[:, :, i:end, :])
    
    print(f"Cache populated: {cache}")
    
    # Test queries at different positions
    test_positions = [T//4, T//2, 3*T//4, T-1]
    
    for pos in test_positions:
        Q = K_full[:, :, pos:pos+1, :]  # Use a key as query for testing
        
        is_valid, metrics = validate_spectral_attention(Q, K_full, V_full, cache)
        
        print(f"\nQuery position {pos}/{T}:")
        print(f"  Attention correlation: {metrics['attention_correlation']:.6f}")
        print(f"  Max abs error: {metrics['max_abs_error']:.6f}")
        print(f"  Mean abs error: {metrics['mean_abs_error']:.6f}")
        print(f"  Valid: {'✅' if is_valid else '❌'}")
    
    print("\n" + "="*60)
    print("Spectral Attention Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_spectral_attention()

