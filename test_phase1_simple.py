"""
Simple standalone test for Phase 1 - no unsloth imports
"""

import sys
import torch
import math

# Direct imports to avoid __init__.py issues
sys.path.insert(0, '/Users/ankitprajapati/unsloth_test')
from unsloth_spectral.spectral_cache import SpectralCache, SpectralBlock
from unsloth_spectral.spectral_attention import spectral_attention_forward, apply_rope


def get_rope_embeddings(seq_len, head_dim, device="cpu"):
    """Generate RoPE cos/sin tables."""
    theta = 10000.0 ** (-2 * torch.arange(0, head_dim, 2).float() / head_dim).to(device)
    position = torch.arange(seq_len, device=device).float().unsqueeze(1)
    freqs = torch.einsum("i,j->ij", position.flatten(), theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def test_basic():
    """Basic functionality test."""
    print("="*70)
    print("TEST: Basic Spectral Cache with RoPE")
    print("="*70)
    
    device = "cpu"  # Use CPU to avoid CUDA issues
    B, H, T, D = 1, 4, 512, 64  # Smaller for CPU
    
    print(f"\nSetup: B={B}, H={H}, T={T}, D={D}, device={device}")
    
    # Create cache
    print("\n1. Creating SpectralCache...")
    cache = SpectralCache(
        num_heads=H,
        head_dim=D,
        block_size=256,
        k_rank_keys=16,
        k_rank_values=16,
        device=device,
        dtype=torch.float32,
    )
    print(f"   ✓ Cache created: {cache}")
    
    # Generate data
    print("\n2. Generating test data...")
    K_pre = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
    position_ids = torch.arange(0, T, device=device).unsqueeze(0)
    print(f"   ✓ K_pre: {K_pre.shape}")
    print(f"   ✓ V: {V.shape}")
    print(f"   ✓ position_ids: {position_ids.shape}")
    
    # Append to cache
    print("\n3. Appending to cache...")
    chunk_size = 128
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        cache.append(
            K_pre[:, :, i:end, :],
            V[:, :, i:end, :],
            position_ids[:, i:end]
        )
        print(f"   ✓ Appended tokens {i}-{end}, cache now: {cache.total_tokens} tokens")
    
    print(f"\n4. Cache state:")
    print(f"   - Total tokens: {cache.total_tokens}")
    print(f"   - Cold blocks: {len(cache.cold_blocks)}")
    print(f"   - Hot tokens: {cache.hot_K.shape[2] if cache.hot_K is not None else 0}")
    
    # Check positions
    print(f"\n5. Position tracking:")
    all_pos = cache.get_all_positions()
    print(f"   - All positions shape: {all_pos.shape}")
    print(f"   - Position range: [{all_pos.min().item()}, {all_pos.max().item()}]")
    for idx, block in enumerate(cache.cold_blocks):
        print(f"   - Block {idx}: start_pos={block.start_position}, size={block.block_size}")
    
    # Test spectral attention
    print(f"\n6. Testing spectral attention...")
    cos, sin = get_rope_embeddings(T + 10, D, device)
    query_position = T
    Q_pre = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
    
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
    
    print(f"   - Query shape: {Q_rot.shape}")
    print(f"   - Query position: {query_position}")
    
    try:
        output = spectral_attention_forward(
            Q=Q_rot,
            cache=cache,
            cos=cos,
            sin=sin,
            query_position=query_position,
            scale=1.0 / math.sqrt(D),
        )
        print(f"   ✓ Attention output shape: {output.shape}")
        print(f"   ✓ Output mean: {output.mean().item():.6f}")
        print(f"   ✓ Output std: {output.std().item():.6f}")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ Attention forward failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic()
    exit(0 if success else 1)

