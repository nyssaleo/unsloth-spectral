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


def spectral_attention_forward(
    Q: torch.Tensor,
    cache: SpectralCache,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention using spectral cache (no reconstruction).
    
    This is the PyTorch reference implementation of dual-spectral attention.
    In production, this would be replaced by a Triton kernel for maximum efficiency.
    
    Args:
        Q: Query tensor [B, H, 1, D] (single token decode)
        cache: SpectralCache containing spectral blocks + hot buffer
        attention_mask: Optional mask [B, H, 1, T_total]
        scale: Attention scale factor (default: 1/sqrt(D))
        
    Returns:
        output: Attention output [B, H, 1, D]
    """
    B, H, _, D = Q.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Get spectral components
    cold_blocks, hot_K, hot_V = cache.get_spectral_components()
    
    # PHASE 1: Compute Attention Scores
    # ===================================
    
    scores_parts = []
    
    # 1a. Spectral attention scores (cold cache)
    for block in cold_blocks:
        # Extract spectral components for this block
        coeffs_K = block.coeffs_K  # [H, T_block, k]
        basis_K = block.basis_K    # [H, k, D]
        
        # Dual projection: (Q @ B_K^T) @ C_K^T
        # Q: [B, H, 1, D]
        # B_K: [H, k, D]
        # C_K: [H, T_block, k]
        
        # Step 1: Project query to spectral space
        # Q @ B_K^T: [B, H, 1, D] @ [H, k, D]^T -> [B, H, 1, k]
        # We need to add batch dim to basis_K
        basis_K_batched = basis_K.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, k, D]
        Q_proj = torch.matmul(Q, basis_K_batched.transpose(-2, -1))  # [B, H, 1, k]
        
        # Step 2: Correlate with temporal coefficients
        # Q_proj @ C_K^T: [B, H, 1, k] @ [H, T_block, k]^T -> [B, H, 1, T_block]
        coeffs_K_batched = coeffs_K.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, T_block, k]
        scores_block = torch.matmul(Q_proj, coeffs_K_batched.transpose(-2, -1))  # [B, H, 1, T_block]
        
        scores_parts.append(scores_block)
    
    # 1b. Standard attention scores (hot cache)
    if hot_K is not None:
        # Standard: Q @ K^T
        scores_hot = torch.matmul(Q, hot_K.transpose(-2, -1))  # [B, H, 1, T_hot]
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
        coeffs_V = block.coeffs_V  # [H, T_block, k]
        basis_V = block.basis_V    # [H, k, D]
        T_block = block.block_size
        
        # Extract attention weights for this block
        attn_block = attn_weights[:, :, :, seq_pos:seq_pos+T_block]  # [B, H, 1, T_block]
        seq_pos += T_block
        
        # Dual projection: (attn @ C_V) @ B_V
        # attn: [B, H, 1, T_block]
        # C_V: [H, T_block, k]
        # B_V: [H, k, D]
        
        # Step 1: Aggregate temporal coefficients
        # attn @ C_V: [B, H, 1, T_block] @ [H, T_block, k] -> [B, H, 1, k]
        coeffs_V_batched = coeffs_V.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, T_block, k]
        v_proj = torch.matmul(attn_block, coeffs_V_batched)  # [B, H, 1, k]
        
        # Step 2: Project back to feature space
        # v_proj @ B_V: [B, H, 1, k] @ [H, k, D] -> [B, H, 1, D]
        basis_V_batched = basis_V.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, k, D]
        output_block = torch.matmul(v_proj, basis_V_batched)  # [B, H, 1, D]
        
        output_parts.append(output_block)
    
    # 2b. Standard weighted values (hot cache)
    if hot_V is not None:
        T_hot = hot_V.shape[2]
        attn_hot = attn_weights[:, :, :, seq_pos:seq_pos+T_hot]  # [B, H, 1, T_hot]
        
        # Standard: attn @ V
        output_hot = torch.matmul(attn_hot, hot_V)  # [B, H, 1, D]
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

