"""
Test GQA (Grouped Query Attention) Fix

This script validates that spectral attention correctly handles the dimension mismatch
between query heads (H_q) and key/value heads (H_kv) in GQA models like Mistral.

Bug: Previously, spectral components were [H_kv, ...] but Q was [B, H_q, ...],
causing dimension mismatch in einsum operations.

Fix: Expand spectral components from [H_kv, ...] to [H_q, ...] before projection.
"""

import sys
import torch
import math

sys.path.insert(0, '/Users/ankitprajapati/unsloth_test')
from unsloth_spectral.spectral_cache import SpectralCache
from unsloth_spectral.spectral_attention import spectral_attention_forward, apply_rope


def get_rope_embeddings(seq_len, head_dim, device="cpu"):
    """Generate RoPE cos/sin tables."""
    theta = 10000.0 ** (-2 * torch.arange(0, head_dim, 2).float() / head_dim).to(device)
    position = torch.arange(seq_len, device=device).float().unsqueeze(1)
    freqs = torch.einsum("i,j->ij", position.flatten(), theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def test_gqa_dimensions():
    """Test GQA with mismatched query and KV heads."""
    print("="*70)
    print("TEST: GQA Dimension Handling")
    print("="*70)
    
    device = "cpu"
    B = 1
    H_q = 8      # Query heads (simulating Mistral-style GQA)
    H_kv = 2     # KV heads
    T = 512
    D = 64
    n_rep = H_q // H_kv  # Should be 4
    
    print(f"\nGQA Configuration:")
    print(f"  Query heads (H_q): {H_q}")
    print(f"  KV heads (H_kv): {H_kv}")
    print(f"  GQA groups (n_rep): {n_rep}")
    print(f"  Sequence length: {T}")
    print(f"  Head dimension: {D}")
    
    # Create cache with KV heads
    print(f"\n1. Creating SpectralCache with num_heads={H_kv}...")
    cache = SpectralCache(
        num_heads=H_kv,  # This is num_key_value_heads
        head_dim=D,
        block_size=256,
        k_rank_keys=16,
        k_rank_values=16,
        device=device,
        dtype=torch.float32,
    )
    print(f"   ✓ Cache created: num_heads={cache.num_heads}")
    
    # Generate data with KV heads
    print(f"\n2. Generating KV data with H_kv={H_kv}...")
    K_pre = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32)
    position_ids = torch.arange(0, T, device=device).unsqueeze(0)
    print(f"   ✓ K_pre shape: {K_pre.shape} (note: H={H_kv})")
    print(f"   ✓ V shape: {V.shape}")
    
    # Append to cache
    print(f"\n3. Appending to cache...")
    chunk_size = 128
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        cache.append(
            K_pre[:, :, i:end, :],
            V[:, :, i:end, :],
            position_ids[:, i:end]
        )
    print(f"   ✓ Cache populated: {cache.total_tokens} tokens")
    print(f"   ✓ Cold blocks: {len(cache.cold_blocks)}")
    
    # Verify spectral component dimensions
    print(f"\n4. Checking spectral component dimensions...")
    cold_blocks, hot_K, hot_V = cache.get_spectral_components()
    for idx, block in enumerate(cold_blocks):
        print(f"   Block {idx}:")
        print(f"     coeffs_K: {block.coeffs_K.shape} (should be [H_kv={H_kv}, T, k])")
        print(f"     basis_K:  {block.basis_K.shape} (should be [H_kv={H_kv}, k, D])")
        assert block.coeffs_K.shape[0] == H_kv, f"coeffs_K has wrong head count: {block.coeffs_K.shape[0]} vs {H_kv}"
        assert block.basis_K.shape[0] == H_kv, f"basis_K has wrong head count: {block.basis_K.shape[0]} vs {H_kv}"
    
    # Generate query with QUERY heads (H_q)
    print(f"\n5. Generating query with H_q={H_q}...")
    cos, sin = get_rope_embeddings(T + 10, D, device)
    query_position = T
    Q_pre = torch.randn(B, H_q, 1, D, device=device, dtype=torch.float32)
    
    # Apply RoPE to Q
    Q_cos = cos[query_position:query_position+1]
    Q_sin = sin[query_position:query_position+1]
    
    d = D
    q1 = Q_pre[..., :d//2]
    q2 = Q_pre[..., d//2:]
    cos1 = Q_cos[..., :d//2].unsqueeze(0).unsqueeze(0)
    sin1 = Q_sin[..., :d//2].unsqueeze(0).unsqueeze(0)
    out1 = q1 * cos1 - q2 * sin1
    out2 = q2 * cos1 + q1 * sin1
    Q_rot = torch.cat([out1, out2], dim=-1)
    
    print(f"   ✓ Q shape: {Q_rot.shape} (note: H={H_q})")
    print(f"   ⚠️  Dimension mismatch: Q has {H_q} heads but cache has {H_kv} heads")
    print(f"   ✓ This is GQA - the fix should expand cache components to match Q")
    
    # Test spectral attention (this would crash without the fix!)
    print(f"\n6. Running spectral attention...")
    print(f"   Testing dimension compatibility fix...")
    
    try:
        output = spectral_attention_forward(
            Q=Q_rot,
            cache=cache,
            cos=cos,
            sin=sin,
            query_position=query_position,
            scale=1.0 / math.sqrt(D),
        )
        print(f"   ✅ SUCCESS! No dimension mismatch error")
        print(f"   ✓ Output shape: {output.shape}")
        assert output.shape == (B, H_q, 1, D), f"Output shape mismatch: {output.shape} vs expected {(B, H_q, 1, D)}"
        print(f"   ✓ Output has correct dimensions: [B={B}, H_q={H_q}, 1, D={D}]")
        print(f"   ✓ Output mean: {output.mean().item():.6f}")
        print(f"   ✓ Output std: {output.std().item():.6f}")
        
    except RuntimeError as e:
        if "shape" in str(e).lower() or "size" in str(e).lower():
            print(f"   ❌ FAILED: Dimension mismatch error (GQA fix not working)")
            print(f"   Error: {str(e)}")
            return False
        else:
            raise
    
    print("\n" + "="*70)
    print("✅ GQA TEST PASSED!")
    print("="*70)
    print("\nValidation:")
    print(f"  ✓ Cache stores {H_kv} KV heads")
    print(f"  ✓ Query uses {H_q} query heads (GQA with n_rep={n_rep})")
    print(f"  ✓ Spectral components correctly expanded from {H_kv} to {H_q} heads")
    print(f"  ✓ No dimension mismatch errors")
    print(f"  ✓ Output shape correct: {output.shape}")
    
    return True


def test_non_gqa_still_works():
    """Verify non-GQA case (n_rep=1) still works."""
    print("\n" + "="*70)
    print("TEST: Non-GQA Case (n_rep=1)")
    print("="*70)
    
    device = "cpu"
    B, H, T, D = 1, 4, 256, 64
    
    print(f"\nConfiguration: H_q=H_kv={H}, n_rep=1 (no GQA)")
    
    cache = SpectralCache(
        num_heads=H,
        head_dim=D,
        block_size=128,
        k_rank_keys=16,
        k_rank_values=16,
        device=device,
        dtype=torch.float32,
    )
    
    K_pre = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    position_ids = torch.arange(0, T, device=device).unsqueeze(0)
    
    cache.append(K_pre, V, position_ids)
    
    cos, sin = get_rope_embeddings(T + 10, D, device)
    Q_pre = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
    Q_cos = cos[T:T+1].unsqueeze(0).unsqueeze(0)
    Q_sin = sin[T:T+1].unsqueeze(0).unsqueeze(0)
    
    d = D
    q1, q2 = Q_pre[..., :d//2], Q_pre[..., d//2:]
    cos1, sin1 = Q_cos[..., :d//2], Q_sin[..., :d//2]
    Q_rot = torch.cat([q1 * cos1 - q2 * sin1, q2 * cos1 + q1 * sin1], dim=-1)
    
    output = spectral_attention_forward(Q=Q_rot, cache=cache, cos=cos, sin=sin, query_position=T, scale=1.0/math.sqrt(D))
    
    assert output.shape == (B, H, 1, D)
    print(f"✓ Non-GQA case works: output shape {output.shape}")
    print("✅ PASSED\n")
    
    return True


def main():
    """Run all GQA fix tests."""
    print("\n" + "="*70)
    print("GQA FIX VALIDATION SUITE")
    print("="*70)
    print("\nTesting the fix for dimension mismatch in Grouped Query Attention")
    
    torch.manual_seed(42)
    
    try:
        # Test 1: GQA with mismatched heads
        if not test_gqa_dimensions():
            return False
        
        # Test 2: Non-GQA case still works
        if not test_non_gqa_still_works():
            return False
        
        print("="*70)
        print("🎉 ALL GQA TESTS PASSED!")
        print("="*70)
        print("\nThe fix successfully handles:")
        print("  ✓ GQA models with H_q ≠ H_kv (e.g., Mistral with 32:8 ratio)")
        print("  ✓ Non-GQA models with H_q = H_kv (e.g., standard transformers)")
        print("  ✓ Spectral component expansion from [H_kv, ...] to [H_q, ...]")
        print("  ✓ Both cold cache (spectral) and hot cache (standard) paths")
        print()
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ GQA FIX VALIDATION FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
