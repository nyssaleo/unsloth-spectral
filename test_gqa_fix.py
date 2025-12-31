#!/usr/bin/env python3
"""
Test Suite for Spectral Cache GQA Fix

This test validates that:
1. SpectralCache stores K/V with num_kv_heads (8), not num_heads (32)
2. Cache.__getitem__() returns correct shapes for Unsloth compatibility
3. GQA expansion (repeat_kv) happens AFTER cache storage
4. Integration with Unsloth's inference path works correctly

Run this BEFORE pushing any changes!
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unsloth_spectral.spectral_cache import SpectralCache


def test_spectral_cache_shapes():
    """Test 1: Verify SpectralCache stores and returns correct shapes."""
    print("="*70)
    print("TEST 1: SpectralCache Shape Validation")
    print("="*70)
    
    # Mistral 7B config
    B, num_kv_heads, T, D = 1, 8, 64, 128
    
    # Create cache with KV heads (NOT query heads!)
    cache = SpectralCache(
        num_heads=num_kv_heads,  # CRITICAL: Must be 8, not 32
        head_dim=D,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        device="cpu",
        dtype=torch.float32,
    )
    
    # Append K/V with KV head count
    K = torch.randn(B, num_kv_heads, T, D)
    V = torch.randn(B, num_kv_heads, T, D)
    
    print(f"\n📥 Appending to cache:")
    print(f"   K shape: {K.shape} (num_kv_heads={num_kv_heads})")
    print(f"   V shape: {V.shape}")
    
    cache.append(K, V)
    
    # Retrieve via __getitem__ (what Unsloth uses)
    K_retrieved = cache[0]
    V_retrieved = cache[1]
    
    print(f"\n📤 Retrieved from cache:")
    print(f"   K shape: {K_retrieved.shape}")
    print(f"   V shape: {V_retrieved.shape}")
    
    # Validate shapes
    assert K_retrieved.shape == (B, num_kv_heads, T, D), \
        f"Wrong K shape: {K_retrieved.shape}, expected {(B, num_kv_heads, T, D)}"
    assert V_retrieved.shape == (B, num_kv_heads, T, D), \
        f"Wrong V shape: {V_retrieved.shape}, expected {(B, num_kv_heads, T, D)}"
    
    print("\n✅ PASS: Cache stores and returns correct KV head count (8)")
    
    # Test tuple unpacking
    K_unpack, V_unpack = cache
    assert K_unpack.shape == K_retrieved.shape, "Unpacking failed"
    assert V_unpack.shape == V_retrieved.shape, "Unpacking failed"
    
    print("✅ PASS: Tuple unpacking works correctly")
    
    return True


def test_gqa_simulation():
    """Test 2: Simulate Unsloth's GQA flow."""
    print("\n" + "="*70)
    print("TEST 2: GQA Flow Simulation")
    print("="*70)
    
    # Mistral config
    B, num_heads, num_kv_heads, T, D = 1, 32, 8, 17, 128
    num_kv_groups = num_heads // num_kv_heads  # 32 // 8 = 4
    
    print(f"\n🔧 Config:")
    print(f"   num_heads (Q): {num_heads}")
    print(f"   num_kv_heads (K/V): {num_kv_heads}")
    print(f"   num_kv_groups: {num_kv_groups}")
    
    # Simulate projection output
    Q = torch.randn(B, num_heads, T, D)
    K = torch.randn(B, num_kv_heads, T, D)  # GQA: fewer KV heads
    V = torch.randn(B, num_kv_heads, T, D)
    
    print(f"\n📊 After QKV projection:")
    print(f"   Q: {Q.shape}")
    print(f"   K: {K.shape} (KV heads)")
    print(f"   V: {V.shape} (KV heads)")
    
    # STEP 1: Store in cache BEFORE repeat_kv (THE FIX!)
    cache = SpectralCache(
        num_heads=num_kv_heads,  # Use KV heads
        head_dim=D,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        device="cpu",
        dtype=torch.float32,
    )
    
    cache.append(K, V)
    print(f"\n💾 Stored in cache (BEFORE repeat_kv):")
    print(f"   K: {K.shape}")
    print(f"   V: {V.shape}")
    
    # STEP 2: Apply repeat_kv for attention (AFTER cache storage)
    def repeat_kv(x, n_rep):
        b, n_kv_heads, slen, head_dim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(b, n_kv_heads, n_rep, slen, head_dim)
        return x.reshape(b, n_kv_heads * n_rep, slen, head_dim)
    
    K_expanded = repeat_kv(K, num_kv_groups)
    V_expanded = repeat_kv(V, num_kv_groups)
    
    print(f"\n🔁 After repeat_kv (for attention):")
    print(f"   K_expanded: {K_expanded.shape} (matches Q)")
    print(f"   V_expanded: {V_expanded.shape}")
    
    # Validate attention can proceed
    assert K_expanded.shape[1] == num_heads, "K heads don't match Q heads"
    assert V_expanded.shape[1] == num_heads, "V heads don't match Q heads"
    
    print("\n✅ PASS: Can compute attention with expanded K/V")
    
    # STEP 3: Simulate Unsloth retrieving from cache
    K_from_cache = cache[0]
    V_from_cache = cache[1]
    
    print(f"\n🔍 Unsloth retrieves from cache:")
    print(f"   cache[0]: {K_from_cache.shape}")
    print(f"   cache[1]: {V_from_cache.shape}")
    
    # This is what Unsloth expects
    expected_shape = (B, num_kv_heads, T, D)
    assert K_from_cache.shape == expected_shape, \
        f"Unsloth expects {expected_shape}, got {K_from_cache.shape}"
    assert V_from_cache.shape == expected_shape, \
        f"Unsloth expects {expected_shape}, got {V_from_cache.shape}"
    
    print(f"\n✅ PASS: Cache returns shape {expected_shape} (num_kv_heads={num_kv_heads})")
    print("✅ PASS: Compatible with Unsloth's paged attention indexing")
    
    return True


def test_unsloth_compatibility():
    """Test 3: Simulate Unsloth's exact access pattern."""
    print("\n" + "="*70)
    print("TEST 3: Unsloth Compatibility Simulation")
    print("="*70)
    
    # Simulate what Unsloth does
    # In LlamaModel_fast_forward_inference_custom:309
    # self.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
    
    B, num_kv_heads, seq_len, D = 1, 8, 17, 128
    
    # Create a cache
    cache = SpectralCache(
        num_heads=num_kv_heads,
        head_dim=D,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        device="cpu",
        dtype=torch.float32,
    )
    
    # Append K/V
    K = torch.randn(B, num_kv_heads, seq_len, D)
    V = torch.randn(B, num_kv_heads, seq_len, D)
    cache.append(K, V)
    
    # Simulate Unsloth accessing cache
    # past_key_values is a list of caches: [cache_layer0, cache_layer1, ...]
    # Unsloth does: seq_len = past_key_values[0][0].shape[-2]
    past_key_values = [cache]  # Simulate list of layer caches
    
    print(f"\n🔍 Simulating Unsloth's access pattern:")
    print(f"   past_key_values[0][0].shape = {past_key_values[0][0].shape}")
    
    try:
        # This is line 1174 in llama.py that was failing
        retrieved_seq_len = past_key_values[0][0].shape[-2]
        print(f"   Retrieved seq_len: {retrieved_seq_len}")
        
        # This is line 309 that was failing with dimension mismatch
        K1 = past_key_values[0][0]  # [B, num_kv_heads, T, D]
        K1_permuted = K1.permute(2, 0, 1, 3)  # [T, B, num_kv_heads, D]
        
        print(f"   K1 shape: {K1.shape}")
        print(f"   K1_permuted shape: {K1_permuted.shape}")
        
        # Simulate paged_attention_K (pre-allocated tensor)
        paged_attention_K = torch.zeros(seq_len, B, num_kv_heads, D)
        print(f"   paged_attention_K shape: {paged_attention_K.shape}")
        
        # This was failing before (dimension mismatch)
        paged_attention_K[:seq_len] = K1_permuted
        
        print("\n✅ PASS: Unsloth's cache access and permutation works!")
        print("✅ PASS: No dimension mismatch errors")
        
    except Exception as e:
        print(f"\n❌ FAIL: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "SPECTRAL CACHE GQA FIX TEST SUITE" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("SpectralCache Shape Validation", test_spectral_cache_shapes),
        ("GQA Flow Simulation", test_gqa_simulation),
        ("Unsloth Compatibility", test_unsloth_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "🎉"*35)
        print("ALL TESTS PASSED! Safe to push changes.")
        print("🎉"*35)
        return 0
    else:
        print("\n" + "❌"*35)
        print("SOME TESTS FAILED! DO NOT PUSH!")
        print("❌"*35)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

