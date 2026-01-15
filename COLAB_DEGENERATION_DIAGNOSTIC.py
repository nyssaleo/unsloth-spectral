"""
DEGENERATION DIAGNOSTIC: Spectral Cache vs Baseline Comparison
===============================================================

This diagnostic investigates why the model generates "The The The The..."
at 3500 tokens, even though Mistral-7B supports 8192 tokens.

Key Questions:
1. Does baseline (no spectral) also fail? → Prompt/model issue
2. Does only spectral fail? → Our implementation issue
3. At what context length does failure start?
4. What's different in attention patterns?

Run in Google Colab with T4 GPU. Copy this entire file into a single code cell.

Author: Ankit Prajapati & Claude (Anthropic)
Date: January 2026
"""

# =============================================================================
# SETUP
# =============================================================================
print("=" * 70)
print("DEGENERATION DIAGNOSTIC: Spectral vs Baseline")
print("=" * 70)

import subprocess
import sys
import os

def install_packages():
    packages = [
        ("triton", "triton"),
        ("unsloth", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"),
        ("unsloth_spectral", "git+https://github.com/nyssaleo/unsloth-spectral.git"),
    ]
    for name, spec in packages:
        print(f"📦 Installing {name}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", spec], 
                      capture_output=True, timeout=300)
        print(f"  ✅ {name} installed")

install_packages()

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import time
import gc
from typing import Optional, Dict, List, Tuple

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from unsloth import FastLanguageModel

# Suppress spectral banner
os.environ["UNSLOTH_SPECTRAL_QUIET"] = "1"
from unsloth_spectral import (
    SpectralCache,
    patch_unsloth_attention,
    get_cache_stats,
)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 8192

# =============================================================================
# TEST PROMPTS - Carefully designed
# =============================================================================

def create_test_prompt(target_tokens: int, needle: str, needle_position: float = 0.5) -> str:
    """
    Create a test prompt with a needle at a specific position.
    
    Args:
        target_tokens: Approximate number of tokens
        needle: The fact to embed
        needle_position: 0.0 = start, 0.5 = middle, 1.0 = end
    """
    # Filler text (varied to avoid repetition patterns)
    filler_sentences = [
        "Machine learning algorithms analyze patterns in data to make predictions.",
        "Neural networks consist of layers of interconnected nodes that process information.",
        "Deep learning has revolutionized image recognition and natural language processing.",
        "Transformers use attention mechanisms to weigh the importance of different inputs.",
        "Gradient descent optimizes model parameters by following the direction of steepest descent.",
        "Backpropagation calculates gradients efficiently using the chain rule of calculus.",
        "Regularization techniques like dropout prevent neural networks from overfitting.",
        "Batch normalization stabilizes training by normalizing layer inputs.",
        "Convolutional neural networks excel at processing grid-structured data like images.",
        "Recurrent neural networks handle sequential data through feedback connections.",
    ]
    
    # Build filler text
    target_chars = target_tokens * 4  # Rough estimate
    filler = ""
    i = 0
    while len(filler) < target_chars:
        filler += filler_sentences[i % len(filler_sentences)] + " "
        i += 1
    
    # Insert needle at specified position
    needle_char_pos = int(len(filler) * needle_position)
    needle_text = f"\n\n[IMPORTANT FACT: {needle}]\n\n"
    
    prompt = filler[:needle_char_pos] + needle_text + filler[needle_char_pos:]
    return prompt[:target_chars]  # Trim to target


def create_recall_question(needle: str) -> str:
    """Create a question to test recall of the needle."""
    if "password" in needle.lower():
        return "\n\nQuestion: What is the password mentioned above?\nAnswer:"
    elif "codename" in needle.lower():
        return "\n\nQuestion: What is the codename mentioned above?\nAnswer:"
    elif "coordinates" in needle.lower():
        return "\n\nQuestion: What are the coordinates mentioned above?\nAnswer:"
    else:
        return "\n\nQuestion: What important fact was mentioned above?\nAnswer:"


# =============================================================================
# TEST 1: BASELINE VS SPECTRAL - SAME PROMPT
# =============================================================================
def test_baseline_vs_spectral():
    """
    Compare generation on the SAME prompt with and without spectral cache.
    This isolates whether the issue is our implementation or the prompt/model.
    """
    print("\n" + "=" * 70)
    print("TEST 1: BASELINE vs SPECTRAL - Same Prompt")
    print("=" * 70)
    
    # Create the problematic prompt (3500 tokens)
    needle = "The secret project codename is PHOENIX-OMEGA-7749."
    base_prompt = create_test_prompt(3500, needle, needle_position=0.5)
    question = create_recall_question(needle)
    full_prompt = base_prompt + question
    
    results = {}
    
    # --- TEST A: BASELINE (No Spectral Cache) ---
    print("\n📥 Loading model for BASELINE test (no spectral)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    # NO patching - pure Unsloth
    
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=4000, truncation=True).to("cuda")
    prompt_length = inputs["input_ids"].shape[1]
    print(f"Prompt length: {prompt_length} tokens")
    
    print("\n🔄 BASELINE generation...")
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    baseline_time = time.time() - start
    baseline_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    print(f"  Time: {baseline_time:.2f}s")
    print(f"  Output: {baseline_output[:150]}...")
    
    # Check for degeneration
    baseline_degeneration = baseline_output.count("The The") > 3 or len(set(baseline_output.split())) < 5
    
    results["baseline"] = {
        "output": baseline_output,
        "time": baseline_time,
        "degenerated": baseline_degeneration,
        "prompt_length": prompt_length,
    }
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- TEST B: SPECTRAL CACHE ---
    print("\n📥 Loading model for SPECTRAL test...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    # Apply spectral cache
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
        debug_logging=False,
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=4000, truncation=True).to("cuda")
    
    print("\n🔄 SPECTRAL generation...")
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    spectral_time = time.time() - start
    spectral_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    print(f"  Time: {spectral_time:.2f}s")
    print(f"  Output: {spectral_output[:150]}...")
    
    # Get cache stats
    stats = get_cache_stats(model)
    
    # Check for degeneration
    spectral_degeneration = spectral_output.count("The The") > 3 or len(set(spectral_output.split())) < 5
    
    results["spectral"] = {
        "output": spectral_output,
        "time": spectral_time,
        "degenerated": spectral_degeneration,
        "compression": stats["compression_ratio"],
        "blocks": stats["total_blocks"],
    }
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- COMPARISON ---
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nPrompt length: {results['baseline']['prompt_length']} tokens")
    print(f"Needle: {needle}")
    print(f"\nBASELINE (No Spectral):")
    print(f"  Output: {results['baseline']['output'][:100]}...")
    print(f"  Degenerated: {'❌ YES' if results['baseline']['degenerated'] else '✅ NO'}")
    print(f"  Recall: {'✅' if 'PHOENIX' in results['baseline']['output'] or '7749' in results['baseline']['output'] else '❌'}")
    
    print(f"\nSPECTRAL:")
    print(f"  Output: {results['spectral']['output'][:100]}...")
    print(f"  Degenerated: {'❌ YES' if results['spectral']['degenerated'] else '✅ NO'}")
    print(f"  Recall: {'✅' if 'PHOENIX' in results['spectral']['output'] or '7749' in results['spectral']['output'] else '❌'}")
    print(f"  Compression: {results['spectral']['compression']:.2f}x")
    print(f"  Blocks: {results['spectral']['blocks']}")
    
    # Diagnosis
    print("\n" + "-" * 70)
    print("DIAGNOSIS:")
    if results['baseline']['degenerated'] and results['spectral']['degenerated']:
        print("  🔴 BOTH degenerate → Issue is PROMPT or MODEL, not spectral cache!")
    elif results['spectral']['degenerated'] and not results['baseline']['degenerated']:
        print("  🟡 Only SPECTRAL degenerates → Issue is in OUR IMPLEMENTATION!")
    elif not results['baseline']['degenerated'] and not results['spectral']['degenerated']:
        print("  🟢 NEITHER degenerate → Previous test may have had different conditions")
    else:
        print("  🟣 Only BASELINE degenerates → Unexpected! Spectral may be stabilizing")
    
    return results


# =============================================================================
# TEST 2: PROGRESSIVE CONTEXT LENGTH
# =============================================================================
def test_progressive_length():
    """
    Test at increasing context lengths to find the failure point.
    """
    print("\n" + "=" * 70)
    print("TEST 2: PROGRESSIVE CONTEXT LENGTH")
    print("=" * 70)
    
    # Load model once with spectral
    print("\n📥 Loading model with spectral cache...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
        debug_logging=False,
    )
    
    # Helper to reset caches
    def reset_caches():
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_spectral_cache'):
                cache = layer.self_attn._spectral_cache
                if hasattr(cache, 'reset'):
                    cache.reset()
    
    # Test configurations
    test_lengths = [512, 1024, 1536, 2048, 2560, 3072, 3500]
    
    results = []
    
    for target_tokens in test_lengths:
        print(f"\n--- Testing {target_tokens} tokens ---")
        reset_caches()
        
        # Create prompt
        needle = f"SECRET-{target_tokens}-CODE"
        prompt = create_test_prompt(target_tokens, needle, needle_position=0.5)
        prompt += f"\n\nWhat is the secret code?\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=target_tokens + 100, truncation=True).to("cuda")
        actual_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        output_text = tokenizer.decode(outputs[0][actual_length:], skip_special_tokens=True)
        
        # Get stats
        stats = get_cache_stats(model)
        
        # Check quality
        degenerated = output_text.count("The The") > 2 or len(set(output_text.split())) < 5
        recalled = needle in output_text or str(target_tokens) in output_text
        
        result = {
            "target": target_tokens,
            "actual": actual_length,
            "blocks": stats["total_blocks"] // 32 if stats["total_blocks"] > 0 else 0,
            "compression": stats["compression_ratio"],
            "output": output_text[:60],
            "degenerated": degenerated,
            "recalled": recalled,
        }
        results.append(result)
        
        status = "❌ DEGEN" if degenerated else ("✅ RECALL" if recalled else "⚠️ NO RECALL")
        print(f"  Actual: {actual_length} tok | Blocks: {result['blocks']}/layer | {status}")
        print(f"  Output: {output_text[:50]}...")
    
    # Summary
    print("\n" + "=" * 70)
    print("PROGRESSIVE LENGTH SUMMARY")
    print("=" * 70)
    print(f"{'Tokens':<10} {'Blocks':<10} {'Compression':<12} {'Status':<15} {'Output Preview':<30}")
    print("-" * 70)
    
    failure_point = None
    for r in results:
        if r["degenerated"]:
            status = "❌ DEGENERATED"
            if failure_point is None:
                failure_point = r["target"]
        elif r["recalled"]:
            status = "✅ RECALLED"
        else:
            status = "⚠️ NO RECALL"
        
        print(f"{r['actual']:<10} {r['blocks']:<10} {r['compression']:.2f}x{'':<7} {status:<15} {r['output'][:25]}...")
    
    if failure_point:
        print(f"\n🔴 Degeneration starts at ~{failure_point} tokens")
    else:
        print(f"\n🟢 No degeneration detected up to {test_lengths[-1]} tokens")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# =============================================================================
# TEST 3: ATTENTION STATISTICS COMPARISON
# =============================================================================
def test_attention_statistics():
    """
    Compare attention statistics between spectral and standard attention.
    """
    print("\n" + "=" * 70)
    print("TEST 3: ATTENTION STATISTICS")
    print("=" * 70)
    
    # This test requires instrumenting the attention layers
    # For now, we'll use the spectral cache's internal statistics
    
    print("\n📥 Loading model with spectral cache...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
        debug_logging=True,  # Enable debug logging!
    )
    
    # Create a moderate-length prompt
    prompt = create_test_prompt(2000, "TEST-DATA-12345", needle_position=0.5)
    prompt += "\n\nWhat is the test data?\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2200, truncation=True).to("cuda")
    print(f"\nPrompt length: {inputs['input_ids'].shape[1]} tokens")
    
    print("\n🔄 Generating with debug logging enabled...")
    print("(Watch for attention statistics in the logs)\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nOutput: {output_text}")
    
    # Get detailed cache stats
    print("\n📊 Cache Statistics per Layer:")
    for i, layer in enumerate(model.model.layers[:5]):  # First 5 layers
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_spectral_cache'):
            cache = layer.self_attn._spectral_cache
            stats = cache.get_memory_stats()
            print(f"  Layer {i}: {stats['num_blocks']} blocks, {stats['compression_ratio']:.2f}x compression")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# TEST 4: RECONSTRUCTION QUALITY PER BLOCK
# =============================================================================
def test_reconstruction_quality():
    """
    Test the reconstruction quality of each compressed block.
    """
    print("\n" + "=" * 70)
    print("TEST 4: RECONSTRUCTION QUALITY PER BLOCK")
    print("=" * 70)
    
    # Create a cache and test reconstruction error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Create test data - simulate real K/V from a model
    print("\n🔬 Testing reconstruction error accumulation...")
    
    H_kv = 8
    D = 128
    block_size = 512
    k_K = 16
    k_V = 32
    
    cache = SpectralCache(
        num_heads=H_kv,
        head_dim=D,
        block_size=block_size,
        k_rank_keys=k_K,
        k_rank_values=k_V,
        device=device,
        dtype=dtype,
    )
    
    # Add multiple blocks worth of data
    total_tokens = 3500
    original_K_list = []
    original_V_list = []
    
    # Generate random data in chunks (simulating prefill)
    remaining = total_tokens
    while remaining > 0:
        chunk_size = min(512, remaining)
        K_chunk = torch.randn(1, H_kv, chunk_size, D, device=device, dtype=dtype)
        V_chunk = torch.randn(1, H_kv, chunk_size, D, device=device, dtype=dtype)
        
        original_K_list.append(K_chunk.clone())
        original_V_list.append(V_chunk.clone())
        
        cache.append(K_chunk, V_chunk)
        remaining -= chunk_size
    
    # Concatenate original tensors
    original_K = torch.cat(original_K_list, dim=2)
    original_V = torch.cat(original_V_list, dim=2)
    
    # Get reconstructed tensors
    reconstructed_K, reconstructed_V = cache.get_kv()
    
    # Compute reconstruction error
    K_error = (original_K - reconstructed_K).abs()
    V_error = (original_V - reconstructed_V).abs()
    
    # Per-block analysis
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Cold blocks: {len(cache.cold_blocks)}")
    print(f"Hot tokens: {cache.hot_K.shape[2] if cache.hot_K is not None else 0}")
    
    print(f"\n📊 Overall Reconstruction Error:")
    print(f"  K: mean={K_error.mean():.6f}, max={K_error.max():.6f}")
    print(f"  V: mean={V_error.mean():.6f}, max={V_error.max():.6f}")
    
    print(f"\n📊 Per-Block Error (K):")
    for i, block in enumerate(cache.cold_blocks):
        start = i * block_size
        end = start + block_size
        block_K_error = K_error[:, :, start:end, :].mean()
        print(f"  Block {i} (tokens {start}-{end}): {block_K_error:.6f}")
    
    # Check hot cache (should have zero error)
    if cache.hot_K is not None:
        hot_start = len(cache.cold_blocks) * block_size
        hot_K_error = K_error[:, :, hot_start:, :].mean()
        print(f"  Hot cache (tokens {hot_start}+): {hot_K_error:.6f}")
    
    # Compute relative error
    K_relative_error = K_error.mean() / original_K.abs().mean()
    V_relative_error = V_error.mean() / original_V.abs().mean()
    
    print(f"\n📊 Relative Error:")
    print(f"  K: {K_relative_error*100:.4f}%")
    print(f"  V: {V_relative_error*100:.4f}%")
    
    # Memory stats
    stats = cache.get_memory_stats()
    print(f"\n📊 Compression Stats:")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Original: {stats['original_bytes'] / 1024:.1f} KB")
    print(f"  Compressed: {stats['compressed_bytes'] / 1024:.1f} KB")
    
    return {
        "K_relative_error": K_relative_error.item(),
        "V_relative_error": V_relative_error.item(),
        "compression_ratio": stats["compression_ratio"],
    }


# =============================================================================
# TEST 5: SIMPLE GENERATION TEST (No Needle)
# =============================================================================
def test_simple_generation():
    """
    Test simple generation at different lengths without needle recall.
    This isolates whether the issue is needle recall or general generation.
    """
    print("\n" + "=" * 70)
    print("TEST 5: SIMPLE GENERATION (No Needle Recall)")
    print("=" * 70)
    
    print("\n📥 Loading model with spectral cache...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
        debug_logging=False,
    )
    
    def reset_caches():
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_spectral_cache'):
                cache = layer.self_attn._spectral_cache
                if hasattr(cache, 'reset'):
                    cache.reset()
    
    # Simple continuation tests (no question-answering)
    test_cases = [
        ("Short context", "Write a poem about", 500),
        ("Medium context", "Explain quantum computing in detail. Start with the basics: ", 1500),
        ("Long context", "Tell me everything about machine learning, neural networks, and AI. ", 3000),
    ]
    
    # Pad with filler to reach target length
    filler = "This is additional context to extend the prompt. Neural networks learn patterns. "
    
    print("\n📊 Generation Quality at Different Lengths:")
    print("-" * 70)
    
    for name, prompt_start, target_length in test_cases:
        reset_caches()
        
        # Build prompt to target length
        prompt = prompt_start
        while len(tokenizer.encode(prompt)) < target_length:
            prompt = filler + prompt
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=target_length, truncation=True).to("cuda")
        actual_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        output_text = tokenizer.decode(outputs[0][actual_length:], skip_special_tokens=True)
        
        # Check quality
        unique_words = len(set(output_text.lower().split()))
        repeated = output_text.count("The The") + output_text.count("the the")
        degenerated = repeated > 3 or unique_words < 5
        
        stats = get_cache_stats(model)
        
        status = "❌ DEGENERATED" if degenerated else "✅ OK"
        print(f"\n{name} ({actual_length} tokens, {stats['total_blocks']//32} blocks/layer):")
        print(f"  Status: {status}")
        print(f"  Unique words: {unique_words}")
        print(f"  Output: {output_text[:80]}...")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("🔬 RUNNING DEGENERATION DIAGNOSTICS")
    print("=" * 70)
    
    results = {}
    
    # Test 1: The most important - baseline vs spectral
    print("\n" + "=" * 70)
    results["baseline_vs_spectral"] = test_baseline_vs_spectral()
    
    # Test 2: Progressive length
    print("\n" + "=" * 70)
    results["progressive_length"] = test_progressive_length()
    
    # Test 3: Reconstruction quality
    print("\n" + "=" * 70)
    results["reconstruction"] = test_reconstruction_quality()
    
    # Test 4: Simple generation
    print("\n" + "=" * 70)
    test_simple_generation()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    if results.get("baseline_vs_spectral"):
        bvs = results["baseline_vs_spectral"]
        print(f"\n1. Baseline vs Spectral:")
        print(f"   Baseline degenerated: {bvs['baseline']['degenerated']}")
        print(f"   Spectral degenerated: {bvs['spectral']['degenerated']}")
        
        if bvs['baseline']['degenerated'] and bvs['spectral']['degenerated']:
            print("   → Issue is NOT specific to spectral cache!")
        elif bvs['spectral']['degenerated'] and not bvs['baseline']['degenerated']:
            print("   → Issue IS specific to spectral cache!")
    
    if results.get("reconstruction"):
        recon = results["reconstruction"]
        print(f"\n2. Reconstruction Quality:")
        print(f"   K relative error: {recon['K_relative_error']*100:.4f}%")
        print(f"   V relative error: {recon['V_relative_error']*100:.4f}%")
        
        if recon['K_relative_error'] > 0.1 or recon['V_relative_error'] > 0.1:
            print("   → High reconstruction error may cause quality issues!")
        else:
            print("   → Reconstruction error is acceptable")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


# Run diagnostics
if __name__ == "__main__":
    run_diagnostics()
else:
    run_diagnostics()
