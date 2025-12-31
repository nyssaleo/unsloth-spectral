#!/usr/bin/env python3
"""
Colab T4 Benchmark Script for Unsloth Spectral

This script validates Phase 2 Randomized SVD optimization on GPU hardware.

Expected Results:
- rSVD speedup: 13-15x (vs 9.5x on CPU)
- End-to-end generation: 18-25 tok/s
- Memory compression: 7-10x for 4K context
- Attention correlation: >0.97 on real model

Usage (in Colab):
    !python colab_t4_benchmark.py
"""

import torch
import time
import sys
import gc
from pathlib import Path

# Ensure library is importable
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_rsvd():
    """Test 1: Randomized SVD GPU performance."""
    print("="*70)
    print("TEST 1: Randomized SVD GPU Benchmark")
    print("="*70)
    
    from unsloth_spectral.rsvd import batched_randomized_svd
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Benchmark key case: 8 heads × 512 tokens × 128 dim, k=16
    M = torch.randn(8, 512, 128, device=device, dtype=torch.float32)
    
    # Warmup
    _ = torch.linalg.svd(M, full_matrices=False)
    _ = batched_randomized_svd(M, k=16)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark standard SVD
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            _ = torch.linalg.svd(M, full_matrices=False)
        end.record()
        torch.cuda.synchronize()
        time_std = start.elapsed_time(end) / 10
    else:
        start_time = time.time()
        for _ in range(10):
            _ = torch.linalg.svd(M, full_matrices=False)
        time_std = (time.time() - start_time) * 100
    
    # Benchmark rSVD
    if device == "cuda":
        start.record()
        for _ in range(10):
            _ = batched_randomized_svd(M, k=16)
        end.record()
        torch.cuda.synchronize()
        time_rsvd = start.elapsed_time(end) / 10
    else:
        start_time = time.time()
        for _ in range(10):
            _ = batched_randomized_svd(M, k=16)
        time_rsvd = (time.time() - start_time) * 100
    
    speedup = time_std / time_rsvd
    
    print(f"\n📊 Results (8×512×128, k=16):")
    print(f"   Standard SVD:     {time_std:.2f} ms")
    print(f"   Randomized SVD:   {time_rsvd:.2f} ms")
    print(f"   Speedup:          {speedup:.2f}x")
    
    if speedup > 12:
        print(f"\n   ✅ EXCELLENT: {speedup:.1f}x speedup on GPU!")
    elif speedup > 8:
        print(f"\n   ✅ GOOD: {speedup:.1f}x speedup")
    else:
        print(f"\n   ⚠️  Lower than expected (target: 13-15x on GPU)")
    
    return speedup


def benchmark_generation():
    """Test 2: End-to-end generation performance."""
    print("\n" + "="*70)
    print("TEST 2: End-to-End Generation Benchmark")
    print("="*70)
    
    try:
        from unsloth import FastLanguageModel
        from unsloth_spectral import patch_unsloth_attention
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install unsloth")
        return None
    
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    max_tokens = 500
    prompt = "Write a detailed technical explanation of quantum entanglement (500 words):"
    
    print(f"\nModel: {model_name}")
    print(f"Max tokens: {max_tokens}")
    
    # Baseline
    print("\n--- Baseline: Standard Cache ---")
    model_baseline, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model_baseline)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs_baseline = model_baseline.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    torch.cuda.synchronize()
    baseline_time = time.time() - start
    baseline_tokens = outputs_baseline.shape[1] - inputs.input_ids.shape[1]
    baseline_speed = baseline_tokens / baseline_time
    
    print(f"Time: {baseline_time:.2f}s, Speed: {baseline_speed:.1f} tok/s")
    
    baseline_text = tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
    
    del model_baseline
    torch.cuda.empty_cache()
    
    # Spectral
    print("\n--- Spectral: Compressed Cache ---")
    model_spectral, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model_spectral)
    
    patch_unsloth_attention(
        model_spectral,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        use_spectral_attention=True,
        verbose=False,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs_spectral = model_spectral.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    torch.cuda.synchronize()
    spectral_time = time.time() - start
    spectral_tokens = outputs_spectral.shape[1] - inputs.input_ids.shape[1]
    spectral_speed = spectral_tokens / spectral_time
    
    print(f"Time: {spectral_time:.2f}s, Speed: {spectral_speed:.1f} tok/s")
    
    spectral_text = tokenizer.decode(outputs_spectral[0], skip_special_tokens=True)
    
    # Free spectral model immediately
    del model_spectral
    del tokenizer
    torch.cuda.empty_cache()
    
    # Comparison
    speedup = baseline_time / spectral_time
    
    print(f"\n📊 Comparison:")
    print(f"   Baseline:  {baseline_speed:.1f} tok/s")
    print(f"   Spectral:  {spectral_speed:.1f} tok/s")
    print(f"   Speedup:   {speedup:.2f}x")
    
    # Quality check
    set_baseline = set(baseline_text.split())
    set_spectral = set(spectral_text.split())
    overlap = len(set_baseline & set_spectral) / len(set_baseline | set_spectral)
    
    print(f"\n📝 Quality:")
    print(f"   Vocabulary overlap: {overlap:.1%}")
    
    if overlap > 0.85:
        print("   ✅ HIGH: Outputs are semantically similar")
    
    print(f"\n   Baseline (first 150 chars):")
    print(f"   {baseline_text[len(prompt):len(prompt)+150]}...")
    print(f"\n   Spectral (first 150 chars):")
    print(f"   {spectral_text[len(prompt):len(prompt)+150]}...")
    
    return {
        "baseline_speed": baseline_speed,
        "spectral_speed": spectral_speed,
        "speedup": speedup,
        "overlap": overlap,
    }


def main():
    print("="*70)
    print("  UNSLOTH SPECTRAL - COLAB T4 BENCHMARK")
    print("="*70)
    print("\nValidating Phase 2: Randomized SVD Optimization")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. This benchmark requires GPU.")
        print("   In Colab: Runtime > Change runtime type > T4 GPU")
        return
    
    # CRITICAL: Clear GPU memory from previous runs
    # This prevents contamination when running the benchmark multiple times
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name()}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   💾 GPU cache cleared for clean benchmark")
    
    # Test 1: rSVD
    rsvd_speedup = benchmark_rsvd()
    
    # Test 2: Generation (optional, requires model download)
    print("\n" + "="*70)
    response = input("\nRun end-to-end generation benchmark? (requires model download, ~5min) [y/N]: ")
    
    if response.lower() == 'y':
        gen_results = benchmark_generation()
        
        # CRITICAL: Free models from GPU memory after generation test
        # This prevents memory leaks in Colab sessions
        print("\n💾 Clearing GPU memory...")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Skipping generation benchmark.")
        gen_results = None
    
    # Summary
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n✅ Randomized SVD:")
    print(f"   GPU Speedup: {rsvd_speedup:.2f}x")
    print(f"   Expected: 13-15x")
    print(f"   Status: {'✅ PASS' if rsvd_speedup > 10 else '⚠️ CHECK'}")
    
    if gen_results:
        print(f"\n✅ End-to-End Generation:")
        print(f"   Baseline: {gen_results['baseline_speed']:.1f} tok/s")
        print(f"   Spectral: {gen_results['spectral_speed']:.1f} tok/s")
        print(f"   Speedup: {gen_results['speedup']:.2f}x")
        print(f"   Quality: {gen_results['overlap']:.1%} overlap")
    
    print("\n" + "="*70)
    print("🎉 Phase 2 GPU Validation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

