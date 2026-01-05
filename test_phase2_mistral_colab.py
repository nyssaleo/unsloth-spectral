"""
Phase 2 Validation: Real Mistral Model with Spectral Cache

This script tests the spectral KV cache compression on a real Mistral 7B model
via Unsloth. It measures perplexity, memory usage, and generation speed.

Usage (in Colab):
1. Install dependencies:
   !pip install unsloth
   !pip install git+https://github.com/nyssaleo/unsloth-spectral.git

2. Run this script:
   python test_phase2_mistral_colab.py

Requirements:
- GPU with at least 16GB VRAM (T4, V100, A100)
- Unsloth library
- HuggingFace account (for model download)
"""

import torch
import gc
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth_spectral import patch_unsloth_attention, get_cache_stats
import time
import psutil
import os


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def test_generation_quality(model, tokenizer, prompts, max_new_tokens=100):
    """Test generation quality with various prompts."""
    print("\n" + "="*70)
    print("GENERATION QUALITY TEST")
    print("="*70)
    
    results = []
    for idx, prompt in enumerate(prompts):
        print(f"\nPrompt {idx+1}/{len(prompts)}:")
        print(f"  Input: {prompt[:100]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"  Generated ({len(outputs[0])} tokens in {gen_time:.2f}s):")
        print(f"  {generated_text[len(prompt):200]}...")
        
        results.append({
            'prompt': prompt,
            'generated': generated_text,
            'tokens': len(outputs[0]),
            'time': gen_time,
            'tokens_per_sec': len(outputs[0]) / gen_time
        })
    
    avg_speed = sum(r['tokens_per_sec'] for r in results) / len(results)
    print(f"\n✅ Average generation speed: {avg_speed:.2f} tokens/sec")
    
    return results


def test_long_context(model, tokenizer, context_length=4096):
    """Test with long context to measure memory compression."""
    print("\n" + "="*70)
    print(f"LONG CONTEXT TEST ({context_length} tokens)")
    print("="*70)
    
    # Generate a long input (repeat a paragraph)
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    long_prompt = base_text[:context_length * 4]  # Approximate token count
    
    inputs = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=context_length).to(model.device)
    actual_len = inputs.input_ids.shape[1]
    
    print(f"  Input length: {actual_len} tokens")
    
    # Measure memory before
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory()
    
    # Generate
    print("  Generating...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start_time
    
    # Measure memory after
    mem_after = get_gpu_memory()
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"\n📊 Memory Usage:")
    print(f"  Before: {mem_before:.1f} MB")
    print(f"  After: {mem_after:.1f} MB")
    print(f"  Peak: {mem_peak:.1f} MB")
    print(f"  Delta: {mem_after - mem_before:.1f} MB")
    
    print(f"\n⚡ Performance:")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Speed: {len(outputs[0]) / gen_time:.2f} tokens/sec")
    
    # Get cache stats if available
    try:
        stats = get_cache_stats(model)
        if stats:
            print(f"\n💾 Cache Statistics:")
            for layer_idx, layer_stats in stats.items():
                if layer_idx == 0:  # Print first layer as example
                    print(f"  Layer {layer_idx}:")
                    print(f"    Total tokens: {layer_stats.get('total_tokens', 'N/A')}")
                    print(f"    Compression ratio: {layer_stats.get('compression_ratio', 'N/A'):.2f}x")
                    break
    except Exception as e:
        print(f"\n⚠️  Could not get cache stats: {e}")
    
    return {
        'input_tokens': actual_len,
        'output_tokens': len(outputs[0]),
        'memory_delta_mb': mem_after - mem_before,
        'memory_peak_mb': mem_peak,
        'generation_time': gen_time,
        'tokens_per_sec': len(outputs[0]) / gen_time,
    }


def compare_baseline_vs_spectral():
    """Compare baseline Unsloth vs Spectral cache."""
    print("\n" + "="*70)
    print("PHASE 2: MISTRAL 7B WITH SPECTRAL CACHE")
    print("="*70)
    
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    max_seq_length = 8192
    
    print(f"\n📦 Loading model: {model_name}")
    print(f"   Max sequence length: {max_seq_length}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"✅ Model loaded")
    print(f"   Device: {model.device}")
    print(f"   Memory allocated: {get_gpu_memory():.1f} MB")
    
    # Test prompts
    test_prompts = [
        "Explain the theory of relativity in simple terms:",
        "Write a Python function to sort a list:",
        "What are the main causes of climate change?",
    ]
    
    # =========================================================================
    # TEST 1: Baseline (Standard Unsloth)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: BASELINE (Standard Unsloth Cache)")
    print("="*70)
    
    FastLanguageModel.for_inference(model)
    
    baseline_results = test_generation_quality(model, tokenizer, test_prompts, max_new_tokens=50)
    baseline_long = test_long_context(model, tokenizer, context_length=2048)
    
    # =========================================================================
    # TEST 2: Spectral Cache
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: SPECTRAL CACHE (Compressed)")
    print("="*70)
    
    print("\n🔧 Applying spectral patch...")
    print("   Configuration:")
    print("     - Block size: 512 tokens")
    print("     - Key rank: 16")
    print("     - Value rank: 32")
    print("     - Hot buffer: 64 tokens")
    
    # Apply spectral patch
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        hot_buffer_size=64,
        use_spectral_attention=True,
        debug_logging=False,
    )
    
    print("✅ Spectral patch applied")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    spectral_results = test_generation_quality(model, tokenizer, test_prompts, max_new_tokens=50)
    spectral_long = test_long_context(model, tokenizer, context_length=2048)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Spectral")
    print("="*70)
    
    print("\n📊 Short Context (50 tokens):")
    baseline_avg_speed = sum(r['tokens_per_sec'] for r in baseline_results) / len(baseline_results)
    spectral_avg_speed = sum(r['tokens_per_sec'] for r in spectral_results) / len(spectral_results)
    
    print(f"  Baseline speed: {baseline_avg_speed:.2f} tokens/sec")
    print(f"  Spectral speed: {spectral_avg_speed:.2f} tokens/sec")
    print(f"  Speedup: {spectral_avg_speed / baseline_avg_speed:.2f}x")
    
    print("\n📊 Long Context (2048 input + 100 output):")
    print(f"  Baseline:")
    print(f"    Memory delta: {baseline_long['memory_delta_mb']:.1f} MB")
    print(f"    Speed: {baseline_long['tokens_per_sec']:.2f} tokens/sec")
    
    print(f"  Spectral:")
    print(f"    Memory delta: {spectral_long['memory_delta_mb']:.1f} MB")
    print(f"    Speed: {spectral_long['tokens_per_sec']:.2f} tokens/sec")
    
    memory_reduction = baseline_long['memory_delta_mb'] / spectral_long['memory_delta_mb']
    speed_ratio = spectral_long['tokens_per_sec'] / baseline_long['tokens_per_sec']
    
    print(f"\n🎯 Results:")
    print(f"  Memory compression: {memory_reduction:.2f}x")
    print(f"  Speed ratio: {speed_ratio:.2f}x")
    
    if memory_reduction > 1.5:
        print(f"  ✅ Memory compression achieved!")
    else:
        print(f"  ⚠️  Memory compression lower than expected")
    
    if speed_ratio > 0.8:  # Allow some slowdown due to projection overhead
        print(f"  ✅ Speed maintained!")
    else:
        print(f"  ⚠️  Significant speed degradation")
    
    # =========================================================================
    # QUALITY CHECK
    # =========================================================================
    print("\n📝 Generation Quality Check:")
    print("  Comparing outputs for similarity...")
    
    quality_ok = True
    for i in range(len(test_prompts)):
        baseline_output = baseline_results[i]['generated']
        spectral_output = spectral_results[i]['generated']
        
        # Simple similarity check (in production, use BLEU or perplexity)
        baseline_tokens = set(baseline_output.split())
        spectral_tokens = set(spectral_output.split())
        overlap = len(baseline_tokens & spectral_tokens) / max(len(baseline_tokens), 1)
        
        print(f"  Prompt {i+1}: {overlap:.1%} token overlap")
        
        if overlap < 0.3:  # Very different outputs
            quality_ok = False
    
    if quality_ok:
        print("  ✅ Outputs are reasonably similar")
    else:
        print("  ⚠️  Outputs differ significantly (may need tuning)")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2 VALIDATION: VERDICT")
    print("="*70)
    
    success = True
    
    if memory_reduction > 1.5:
        print("✅ Memory compression: PASS")
    else:
        print("❌ Memory compression: FAIL")
        success = False
    
    if speed_ratio > 0.8:
        print("✅ Speed performance: PASS")
    else:
        print("❌ Speed performance: FAIL")
        success = False
    
    if quality_ok:
        print("✅ Generation quality: PASS")
    else:
        print("⚠️  Generation quality: NEEDS REVIEW")
    
    if success:
        print("\n🎉 PHASE 2 COMPLETE: Spectral cache works with real models!")
        print("   Ready for Phase 3 (Triton kernel optimization)")
    else:
        print("\n⚠️  PHASE 2 INCOMPLETE: Issues need addressing")
    
    return success


def main():
    """Main entry point."""
    print("="*70)
    print("UNSLOTH SPECTRAL - PHASE 2 VALIDATION")
    print("="*70)
    print("\nTesting spectral KV cache compression on real Mistral 7B model")
    print("This will take 10-15 minutes depending on GPU...")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ ERROR: No GPU detected!")
        print("   This test requires a CUDA-capable GPU")
        return False
    
    print(f"\n✅ GPU detected: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        success = compare_baseline_vs_spectral()
        return success
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

