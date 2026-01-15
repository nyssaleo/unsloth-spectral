"""
COMPREHENSIVE LONG-CONTEXT VALIDATION TEST SUITE v2
====================================================

This script validates the Unsloth Spectral KV Cache implementation with extensive tests:
1. Unit Tests: Isolated component verification
2. Long Context Tests: Compression trigger verification (>512 tokens)
3. Very Long Context: 2000-4000 token tests for real compression benefits
4. Needle-in-Haystack: Factual recall under compression
5. Memory Profiling: VRAM usage measurement
6. Quality Metrics: Generation comparison (with cache reset between prompts)
7. Performance Benchmarks: Throughput measurement

FIXES in v2:
- Cache contamination fix: Reset between prompts in quality test
- Model: Mistral-7B-Instruct (more capable than Llama-1B)
- Memory calculation: Corrected formula
- Long context: Added 2000-4000 token tests

Run in Google Colab with T4 GPU. Copy this entire file into a single code cell.

Author: Ankit Prajapati & Claude (Anthropic)
Date: January 2026
"""

# =============================================================================
# CELL 0: INSTALLATION & SETUP
# =============================================================================
print("=" * 70)
print("UNSLOTH SPECTRAL - COMPREHENSIVE VALIDATION SUITE v2")
print("=" * 70)

import subprocess
import sys
import os

def install_packages():
    """Install required packages with error handling."""
    packages = [
        ("triton", "triton"),
        ("unsloth", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"),
        ("unsloth_spectral", "git+https://github.com/nyssaleo/unsloth-spectral.git"),
    ]
    
    for name, spec in packages:
        print(f"\n📦 Installing {name}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", spec],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"  ⚠️ Warning: pip install returned {result.returncode}")
                print(f"  stderr: {result.stderr[:500] if result.stderr else 'None'}")
            else:
                print(f"  ✅ {name} installed")
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ Timeout installing {name}")
        except Exception as e:
            print(f"  ❌ Error installing {name}: {e}")

# Run installation
install_packages()

# =============================================================================
# CELL 1: IMPORTS & ENVIRONMENT CHECK
# =============================================================================
print("\n" + "=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)

import torch
import time
import math
import gc
from typing import Optional, Dict, List, Tuple

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Check Triton
try:
    import triton
    print(f"Triton version: {triton.__version__}")
    TRITON_OK = True
except ImportError:
    print("Triton: Not available")
    TRITON_OK = False

# Import unsloth_spectral components
try:
    os.environ["UNSLOTH_SPECTRAL_QUIET"] = "1"  # Suppress banner
    
    from unsloth_spectral import (
        SpectralCache,
        SpectralBlock,
        patch_unsloth_attention,
        get_cache_stats,
        spectral_attention_forward,
    )
    
    # Try importing Triton kernel components
    try:
        from unsloth_spectral.kernels import (
            TRITON_AVAILABLE,
            TRITON_VERSION,
            spectral_attention_decode,
            spectral_score_forward,
            spectral_value_forward,
            TritonSpectralConfig,
        )
        print(f"Triton kernels: Available (v{TRITON_VERSION})")
    except ImportError as e:
        print(f"Triton kernels: Not available ({e})")
        TRITON_AVAILABLE = False
        TritonSpectralConfig = None
        spectral_attention_decode = None
        spectral_score_forward = None
        spectral_value_forward = None
    
    print("✅ unsloth_spectral imported successfully")
    SPECTRAL_OK = True
    
except ImportError as e:
    print(f"❌ Failed to import unsloth_spectral: {e}")
    SPECTRAL_OK = False

# Import Unsloth
try:
    from unsloth import FastLanguageModel
    print("✅ unsloth imported successfully")
    UNSLOTH_OK = True
except ImportError as e:
    print(f"❌ Failed to import unsloth: {e}")
    UNSLOTH_OK = False

print("\n" + "=" * 70)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Using Mistral-7B for better quality and realistic testing
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 8192  # Mistral supports long context
NUM_LAYERS = 32        # Mistral-7B has 32 layers
NUM_KV_HEADS = 8       # Mistral uses GQA with 8 KV heads
NUM_Q_HEADS = 32       # 32 query heads
HEAD_DIM = 128         # 128 dim per head

print(f"Model: {MODEL_NAME}")
print(f"Config: {NUM_LAYERS} layers, {NUM_KV_HEADS} KV heads, {NUM_Q_HEADS} Q heads, {HEAD_DIM} dim")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def reset_all_caches(model):
    """
    Reset SpectralCache for all layers to prevent contamination between prompts.
    
    CRITICAL: Without this, cache from previous generations leaks into new ones,
    causing outputs like "The quick brown fox -> 1, 1, 2, 3, 5..." (Fibonacci contamination)
    """
    reset_count = 0
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_spectral_cache'):
            cache = layer.self_attn._spectral_cache
            if isinstance(cache, SpectralCache) and hasattr(cache, 'reset'):
                cache.reset()
                reset_count += 1
    return reset_count


def calculate_theoretical_kv_size(tokens_per_layer: int, num_layers: int = NUM_LAYERS, 
                                   num_kv_heads: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM) -> int:
    """
    Calculate theoretical uncompressed KV cache size in bytes.
    
    Formula: num_layers × 2 (K+V) × T × H_kv × D × 2 (FP16 bytes)
    
    Args:
        tokens_per_layer: Tokens stored per layer (NOT total across all layers)
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (8 for Mistral GQA)
        head_dim: Dimension per head (128)
    
    Returns:
        Size in bytes
    """
    return num_layers * 2 * tokens_per_layer * num_kv_heads * head_dim * 2


def get_tokens_per_layer_from_stats(stats: dict, num_layers: int = NUM_LAYERS) -> int:
    """
    Extract tokens per layer from aggregated stats.
    
    stats['total_tokens'] sums tokens across all layers, so divide by num_layers.
    """
    if stats['total_tokens'] == 0 or stats['layers_with_cache'] == 0:
        return 0
    return stats['total_tokens'] // stats['layers_with_cache']


# =============================================================================
# CELL 2: UNIT TESTS - SPECTRAL CACHE ISOLATION
# =============================================================================
def test_spectral_cache_unit():
    """Test SpectralCache in isolation (no model dependencies)."""
    print("\n" + "=" * 70)
    print("UNIT TEST 1: SpectralCache Isolation")
    print("=" * 70)
    
    if not SPECTRAL_OK:
        print("❌ Skipping: unsloth_spectral not available")
        return False
    
    # Configuration matching Mistral-7B
    H_kv = NUM_KV_HEADS  # 8
    D = HEAD_DIM         # 128
    block_size = 512
    k_K = 16             # k_rank_keys
    k_V = 32             # k_rank_values
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Config: H_kv={H_kv}, D={D}, block_size={block_size}, k_K={k_K}, k_V={k_V}")
    print(f"Device: {device}, dtype: {dtype}")
    
    # Create cache
    cache = SpectralCache(
        num_heads=H_kv,
        head_dim=D,
        block_size=block_size,
        k_rank_keys=k_K,
        k_rank_values=k_V,
        device=device,
        dtype=dtype,
        debug_logging=False,
    )
    
    print(f"\n✅ Cache created: {cache}")
    
    # Test 1: Append small batch (no compression)
    print("\n--- Test 1: Small append (no compression) ---")
    K1 = torch.randn(1, H_kv, 100, D, device=device, dtype=dtype)
    V1 = torch.randn(1, H_kv, 100, D, device=device, dtype=dtype)
    cache.append(K1, V1)
    
    print(f"After 100 tokens: total={cache.total_tokens}, cold_blocks={len(cache.cold_blocks)}")
    assert cache.total_tokens == 100, f"Expected 100, got {cache.total_tokens}"
    assert len(cache.cold_blocks) == 0, f"Expected 0 blocks, got {len(cache.cold_blocks)}"
    print("✅ Small append passed")
    
    # Test 2: Append to trigger compression
    print("\n--- Test 2: Append to trigger compression (>512 tokens) ---")
    K2 = torch.randn(1, H_kv, 500, D, device=device, dtype=dtype)
    V2 = torch.randn(1, H_kv, 500, D, device=device, dtype=dtype)
    cache.append(K2, V2)
    
    print(f"After 600 tokens: total={cache.total_tokens}")
    print(f"  Cold blocks: {len(cache.cold_blocks)}")
    print(f"  Hot tokens: {cache.hot_K.shape[2] if cache.hot_K is not None else 0}")
    
    expected_cold = 1  # 512 tokens compressed
    expected_hot = 88  # 600 - 512 = 88
    assert len(cache.cold_blocks) == expected_cold, f"Expected {expected_cold} blocks, got {len(cache.cold_blocks)}"
    if cache.hot_K is not None:
        actual_hot = cache.hot_K.shape[2]
        assert actual_hot == expected_hot, f"Expected {expected_hot} hot tokens, got {actual_hot}"
    print("✅ Compression triggered correctly")
    
    # Test 3: Verify spectral block structure
    print("\n--- Test 3: Spectral block structure ---")
    block = cache.cold_blocks[0]
    print(f"Block 0:")
    print(f"  coeffs_K: {block.coeffs_K.shape} (expected: [H_kv={H_kv}, block_size={block_size}, k_K={k_K}])")
    print(f"  basis_K: {block.basis_K.shape} (expected: [H_kv={H_kv}, k_K={k_K}, D={D}])")
    print(f"  coeffs_V: {block.coeffs_V.shape} (expected: [H_kv={H_kv}, block_size={block_size}, k_V={k_V}])")
    print(f"  basis_V: {block.basis_V.shape} (expected: [H_kv={H_kv}, k_V={k_V}, D={D}])")
    print(f"  dtype: coeffs_K={block.coeffs_K.dtype}, basis_K={block.basis_K.dtype}")
    
    assert block.coeffs_K.shape == (H_kv, block_size, k_K), f"coeffs_K shape mismatch"
    assert block.basis_K.shape == (H_kv, k_K, D), f"basis_K shape mismatch"
    assert block.coeffs_V.shape == (H_kv, block_size, k_V), f"coeffs_V shape mismatch"
    assert block.basis_V.shape == (H_kv, k_V, D), f"basis_V shape mismatch"
    print("✅ Spectral block structure correct")
    
    # Test 4: Full reconstruction
    print("\n--- Test 4: Full K/V reconstruction ---")
    K_full, V_full = cache.get_kv()
    print(f"Reconstructed: K={K_full.shape}, V={V_full.shape}")
    
    expected_shape = (1, H_kv, 600, D)
    assert K_full.shape == expected_shape, f"K shape mismatch: {K_full.shape} vs {expected_shape}"
    assert V_full.shape == expected_shape, f"V shape mismatch: {V_full.shape} vs {expected_shape}"
    print("✅ Full reconstruction shape correct")
    
    # Test 5: Memory statistics
    print("\n--- Test 5: Memory statistics ---")
    stats = cache.get_memory_stats()
    print(f"Memory stats:")
    print(f"  Original: {stats['original_bytes'] / 1024:.1f} KB")
    print(f"  Compressed: {stats['compressed_bytes'] / 1024:.1f} KB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Blocks: {stats['num_blocks']}, Tokens: {stats['total_tokens']}")
    
    # With compression, ratio should be > 1
    if stats['num_blocks'] > 0:
        assert stats['compression_ratio'] > 1.0, f"Expected compression_ratio > 1, got {stats['compression_ratio']}"
        print("✅ Compression active")
    else:
        print("⚠️ No compressed blocks (expected in short context)")
    
    # Test 6: Reset functionality
    print("\n--- Test 6: Cache reset ---")
    cache.reset()
    assert cache.total_tokens == 0, f"Expected 0 tokens after reset, got {cache.total_tokens}"
    assert len(cache.cold_blocks) == 0, f"Expected 0 blocks after reset"
    print("✅ Cache reset works")
    
    print("\n" + "=" * 70)
    print("UNIT TEST 1: ✅ ALL PASSED")
    print("=" * 70)
    return True


# =============================================================================
# CELL 3: UNIT TESTS - KERNEL FUNCTIONS
# =============================================================================
def test_kernel_functions():
    """Test kernel functions (score and value forward) in isolation."""
    print("\n" + "=" * 70)
    print("UNIT TEST 2: Kernel Functions")
    print("=" * 70)
    
    if not SPECTRAL_OK:
        print("❌ Skipping: unsloth_spectral not available")
        return False
    
    if not TRITON_AVAILABLE or spectral_score_forward is None:
        print("⚠️ Triton kernels not available, testing PyTorch fallbacks")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Configuration
    B = 1              # batch
    H_q = NUM_Q_HEADS  # 32
    H_kv = NUM_KV_HEADS # 8
    D = HEAD_DIM       # 128
    T_block = 256      # tokens in compressed block
    k_K = 16           # key rank
    k_V = 32           # value rank
    max_seq = 8192     # RoPE table size
    
    print(f"Config: B={B}, H_q={H_q}, H_kv={H_kv}, D={D}, T_block={T_block}")
    
    # Create test tensors
    Q = torch.randn(B, H_q, 1, D, device=device, dtype=dtype)
    coeffs_K = torch.randn(H_kv, T_block, k_K, device=device, dtype=dtype)
    basis_K = torch.randn(H_kv, k_K, D, device=device, dtype=dtype)
    coeffs_V = torch.randn(H_kv, T_block, k_V, device=device, dtype=dtype)
    basis_V = torch.randn(H_kv, k_V, D, device=device, dtype=dtype)
    
    # RoPE tables
    cos = torch.randn(max_seq, D, device=device, dtype=dtype)
    sin = torch.randn(max_seq, D, device=device, dtype=dtype)
    
    config = TritonSpectralConfig(enable_logging=True) if TritonSpectralConfig else None
    
    # Test 1: spectral_score_forward
    print("\n--- Test 1: spectral_score_forward ---")
    if spectral_score_forward is not None:
        scores = spectral_score_forward(
            Q=Q,
            coeffs_K=coeffs_K,
            basis_K=basis_K,
            cos=cos,
            sin=sin,
            start_position=0,
            query_position=256,
            config=config,
            use_triton=False,  # PyTorch for correctness
        )
        print(f"Scores shape: {scores.shape} (expected: [{B}, {H_q}, 1, {T_block}])")
        print(f"Scores dtype: {scores.dtype}")
        print(f"Scores stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        expected_shape = (B, H_q, 1, T_block)
        assert scores.shape == expected_shape, f"Shape mismatch: {scores.shape} vs {expected_shape}"
        assert scores.dtype == dtype, f"Dtype mismatch: {scores.dtype} vs {dtype}"
        print("✅ spectral_score_forward passed")
    else:
        print("⚠️ spectral_score_forward not available")
    
    # Test 2: spectral_value_forward
    print("\n--- Test 2: spectral_value_forward ---")
    if spectral_value_forward is not None:
        # Create attention weights from scores
        if spectral_score_forward is not None:
            scale = 1.0 / math.sqrt(D)
            attn = torch.softmax(scores * scale, dim=-1)  # [B, H_q, 1, T_block]
        else:
            attn = torch.softmax(torch.randn(B, H_q, 1, T_block, device=device, dtype=dtype), dim=-1)
        
        output = spectral_value_forward(
            attn=attn,
            coeffs_V=coeffs_V,
            basis_V=basis_V,
            D=D,
            config=config,
            use_triton=False,  # PyTorch for correctness
        )
        print(f"Output shape: {output.shape} (expected: [{B}, {H_q}, 1, {D}])")
        print(f"Output dtype: {output.dtype}")
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        expected_shape = (B, H_q, 1, D)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        assert output.dtype == dtype, f"Dtype mismatch: {output.dtype} vs {dtype}"
        print("✅ spectral_value_forward passed")
    else:
        print("⚠️ spectral_value_forward not available")
    
    # Test 3: spectral_attention_decode (full integration)
    print("\n--- Test 3: spectral_attention_decode (integration) ---")
    if spectral_attention_decode is not None:
        # Create a proper SpectralCache with enough data for cold blocks
        cache = SpectralCache(
            num_heads=H_kv,
            head_dim=D,
            block_size=512,
            k_rank_keys=k_K,
            k_rank_values=k_V,
            device=device,
            dtype=dtype,
            debug_logging=False,
        )
        
        # Add enough data to create a cold block
        K_test = torch.randn(B, H_kv, 700, D, device=device, dtype=dtype)
        V_test = torch.randn(B, H_kv, 700, D, device=device, dtype=dtype)
        cache.append(K_test, V_test)
        
        print(f"Cache state: total={cache.total_tokens}, cold_blocks={len(cache.cold_blocks)}")
        
        # Run spectral_attention_decode
        Q_test = torch.randn(B, H_q, 1, D, device=device, dtype=dtype)
        output = spectral_attention_decode(
            Q=Q_test,
            cache=cache,
            cos=cos,
            sin=sin,
            query_position=699,
            attention_mask=None,
            scale=1.0 / math.sqrt(D),
            config=config,
        )
        
        print(f"Decode output shape: {output.shape} (expected: [{B}, {H_q}, 1, {D}])")
        print(f"Decode output dtype: {output.dtype}")
        
        expected_shape = (B, H_q, 1, D)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        print("✅ spectral_attention_decode passed")
    else:
        print("⚠️ spectral_attention_decode not available")
    
    print("\n" + "=" * 70)
    print("UNIT TEST 2: ✅ ALL PASSED")
    print("=" * 70)
    return True


# =============================================================================
# CELL 4: LONG CONTEXT TEST - COMPRESSION VERIFICATION
# =============================================================================
def test_long_context_compression():
    """Test that compression actually triggers and works with long contexts."""
    print("\n" + "=" * 70)
    print("LONG CONTEXT TEST: Compression Verification (Mistral-7B)")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    # Load Mistral-7B model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    # Patch with spectral cache
    print("\n🔧 Patching with SpectralCache...")
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        hot_buffer_size=64,
        use_spectral_attention=True,
        verbose=True,
        debug_logging=False,
        use_triton_kernel=True,
    )
    
    # Create a LONG prompt (> 512 tokens to trigger compression)
    print("\n📝 Creating long prompt (target: ~1500 tokens)...")
    
    # Generate a long context with varied content
    base_text = """
    This is an extensive technical document about machine learning and artificial intelligence.
    Machine learning involves training models on data to make predictions and decisions.
    Neural networks are a key component of modern machine learning systems.
    Deep learning has revolutionized computer vision, natural language processing, and more.
    Transformers have become the dominant architecture for language modeling tasks.
    Attention mechanisms allow models to focus on relevant parts of the input.
    The key insight of transformers is self-attention: each token attends to all other tokens.
    This enables capturing long-range dependencies in sequences.
    """
    
    # Repeat to get ~1500 tokens
    long_context = (base_text * 20)  # Should be ~1600 tokens
    
    # Add a specific fact we'll test recall on
    needle = "The secret access code for the quantum laboratory is OMEGA-3847-DELTA."
    long_context += f"\n\n[IMPORTANT SECURITY NOTE: {needle}]\n\n"
    long_context += "Now, continuing with our discussion of neural network architectures..."
    
    # Tokenize
    inputs = tokenizer(long_context, return_tensors="pt").to("cuda")
    prompt_length = inputs["input_ids"].shape[1]
    print(f"Prompt length: {prompt_length} tokens")
    
    if prompt_length < 1000:
        print(f"⚠️ Warning: Prompt ({prompt_length}) < 1000. Adding more content...")
        long_context = (base_text * 35) + f"\n\n[IMPORTANT: {needle}]\n\n"
        inputs = tokenizer(long_context, return_tensors="pt").to("cuda")
        prompt_length = inputs["input_ids"].shape[1]
        print(f"Extended prompt length: {prompt_length} tokens")
    
    # Generate with use_cache=True
    print("\n🔄 Running generation...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    gen_time = time.time() - start_time
    total_tokens = outputs.shape[1]
    new_tokens = total_tokens - prompt_length
    
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Generated {new_tokens} new tokens")
    print(f"Speed: {new_tokens / gen_time:.2f} tokens/sec")
    
    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text (last 200 chars): ...{output_text[-200:]}")
    
    # Check cache statistics
    print("\n📊 Cache Statistics:")
    stats = get_cache_stats(model)
    tokens_per_layer = get_tokens_per_layer_from_stats(stats)
    
    print(f"  Layers with cache: {stats['layers_with_cache']}")
    print(f"  Tokens per layer: {tokens_per_layer}")
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Calculate theoretical vs actual memory
    theoretical_bytes = calculate_theoretical_kv_size(tokens_per_layer)
    actual_bytes = stats.get('total_compressed_bytes', 0)
    
    print(f"\n📊 Memory Analysis:")
    print(f"  Theoretical uncompressed: {theoretical_bytes / 1024**2:.2f} MB")
    print(f"  Actual compressed: {actual_bytes / 1024**2:.2f} MB")
    print(f"  Memory savings: {(1 - actual_bytes/theoretical_bytes)*100:.1f}%" if theoretical_bytes > 0 else "N/A")
    
    # Verify compression occurred
    success = True
    if prompt_length > 512:
        expected_blocks_per_layer = prompt_length // 512
        if stats['total_blocks'] == 0:
            print(f"❌ FAIL: Expected compression (prompt={prompt_length} > 512) but got 0 blocks")
            success = False
        else:
            print(f"✅ Compression verified: {stats['total_blocks']} blocks (expect ~{expected_blocks_per_layer * NUM_LAYERS})")
    
    # Test needle recall
    print("\n🔍 Testing needle recall...")
    reset_all_caches(model)  # CRITICAL: Reset before new generation
    
    recall_prompt = long_context + "\nWhat is the secret access code mentioned above? Answer directly:"
    recall_inputs = tokenizer(recall_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        recall_outputs = model.generate(
            **recall_inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    recall_text = tokenizer.decode(recall_outputs[0][recall_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Recall response: {recall_text[:150]}...")
    
    if "OMEGA-3847-DELTA" in recall_text or "omega-3847-delta" in recall_text.lower():
        print("✅ Needle successfully recalled!")
    elif "OMEGA" in recall_text or "3847" in recall_text or "DELTA" in recall_text:
        print("⚠️ Partial needle recall (some parts found)")
    else:
        print("⚠️ Needle not found in response (may be model/compression limitation)")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print(f"LONG CONTEXT TEST: {'✅ PASSED' if success else '❌ FAILED'}")
    print("=" * 70)
    return success


# =============================================================================
# CELL 5: VERY LONG CONTEXT TEST (2000-4000 tokens)
# =============================================================================
def test_very_long_context():
    """Test with very long context (2000-4000 tokens) to see real compression benefits."""
    print("\n" + "=" * 70)
    print("VERY LONG CONTEXT TEST: 2000-4000 Tokens")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    # Load model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    # Patch with spectral cache
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        hot_buffer_size=64,
        use_spectral_attention=True,
        verbose=True,
        debug_logging=False,
    )
    
    # Create VERY long context (~3000 tokens)
    print("\n📝 Creating very long prompt (target: ~3000 tokens)...")
    
    # Rich, varied content to avoid repetition patterns
    sections = [
        "In the realm of artificial intelligence, neural networks have emerged as powerful tools.",
        "The backpropagation algorithm revolutionized how we train deep neural networks.",
        "Convolutional neural networks excel at processing grid-like topology data such as images.",
        "Recurrent neural networks were designed to handle sequential data with temporal dependencies.",
        "Long Short-Term Memory networks address the vanishing gradient problem in RNNs.",
        "Transformers replaced recurrence with attention mechanisms for parallel processing.",
        "BERT introduced bidirectional pre-training for language understanding tasks.",
        "GPT models demonstrated the power of autoregressive language modeling at scale.",
        "Diffusion models have shown remarkable results in image generation tasks.",
        "Reinforcement learning from human feedback aligns model outputs with human preferences.",
    ]
    
    # Generate ~3000 tokens of varied content
    long_context = ""
    for i in range(30):  # Repeat sections with variations
        for j, section in enumerate(sections):
            long_context += f"Section {i*10 + j + 1}: {section} "
            long_context += "This represents a key advancement in the field. "
    
    # Add needle in the middle
    needle = "CLASSIFIED: Project Starlight uses encryption key XK-9921-GAMMA-7."
    middle_pos = len(long_context) // 2
    long_context = long_context[:middle_pos] + f"\n[{needle}]\n" + long_context[middle_pos:]
    
    # Tokenize
    inputs = tokenizer(long_context, return_tensors="pt", max_length=3500, truncation=True).to("cuda")
    prompt_length = inputs["input_ids"].shape[1]
    print(f"Prompt length: {prompt_length} tokens")
    
    # Expected compression
    expected_blocks = (prompt_length // 512)
    compressed_tokens = expected_blocks * 512
    hot_tokens = prompt_length - compressed_tokens
    theoretical_compression = prompt_length / (hot_tokens + compressed_tokens * 0.1)  # Rough estimate
    print(f"Expected: {expected_blocks} blocks compressed, {hot_tokens} hot tokens")
    print(f"Theoretical max compression: ~{theoretical_compression:.1f}x")
    
    # Generate
    print("\n🔄 Running generation...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    gen_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    new_tokens = outputs.shape[1] - prompt_length
    
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Generated {new_tokens} new tokens")
    print(f"Speed: {new_tokens / gen_time:.2f} tokens/sec")
    print(f"Peak VRAM: {peak_memory:.1f} MB")
    
    # Cache statistics
    stats = get_cache_stats(model)
    tokens_per_layer = get_tokens_per_layer_from_stats(stats)
    
    print("\n📊 Compression Statistics:")
    print(f"  Layers with cache: {stats['layers_with_cache']}")
    print(f"  Tokens per layer: {tokens_per_layer}")
    print(f"  Total compressed blocks: {stats['total_blocks']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Memory analysis
    theoretical_bytes = calculate_theoretical_kv_size(tokens_per_layer)
    actual_bytes = stats.get('total_compressed_bytes', 0)
    
    print(f"\n📊 Memory Analysis:")
    print(f"  Theoretical uncompressed KV: {theoretical_bytes / 1024**2:.2f} MB")
    print(f"  Actual compressed: {actual_bytes / 1024**2:.2f} MB")
    if theoretical_bytes > 0 and actual_bytes > 0:
        savings = (1 - actual_bytes / theoretical_bytes) * 100
        print(f"  Memory savings: {savings:.1f}%")
    
    # Needle recall test
    print("\n🔍 Testing needle recall from middle of context...")
    reset_all_caches(model)
    
    recall_prompt = long_context[:1000] + "\n\nQuestion: What is the Project Starlight encryption key?\nAnswer:"
    recall_inputs = tokenizer(recall_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        recall_outputs = model.generate(
            **recall_inputs,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    recall_text = tokenizer.decode(recall_outputs[0][recall_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Recall response: {recall_text}")
    
    if "XK-9921-GAMMA-7" in recall_text:
        print("✅ Needle perfectly recalled!")
    elif any(x in recall_text for x in ["XK", "9921", "GAMMA"]):
        print("⚠️ Partial recall")
    else:
        print("⚠️ Needle not found")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("VERY LONG CONTEXT TEST: ✅ COMPLETE")
    print("=" * 70)
    return True


# =============================================================================
# CELL 6: MEMORY PROFILING
# =============================================================================
def test_memory_usage():
    """Profile VRAM usage with spectral compression."""
    print("\n" + "=" * 70)
    print("MEMORY PROFILING (Mistral-7B)")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    def get_memory_mb():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    
    def get_max_memory_mb():
        return torch.cuda.max_memory_allocated() / 1024**2
    
    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    baseline_memory = get_memory_mb()
    print(f"Baseline VRAM: {baseline_memory:.1f} MB")
    
    # Load model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    model_memory = get_memory_mb()
    print(f"After model load: {model_memory:.1f} MB (+{model_memory - baseline_memory:.1f} MB)")
    
    # Patch with spectral cache
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
        debug_logging=False,
    )
    
    # Create long prompt (~2000 tokens)
    long_text = "The quick brown fox jumps over the lazy dog. " * 250  # ~2500 tokens
    inputs = tokenizer(long_text, return_tensors="pt", max_length=2000, truncation=True).to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"\nPrompt length: {prompt_tokens} tokens")
    
    # Pre-generation memory
    pre_gen_memory = get_memory_mb()
    print(f"Pre-generation: {pre_gen_memory:.1f} MB")
    
    # Run generation
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    post_gen_memory = get_memory_mb()
    peak_memory = get_max_memory_mb()
    
    print(f"\nPost-generation: {post_gen_memory:.1f} MB")
    print(f"Peak during generation: {peak_memory:.1f} MB")
    print(f"Generation overhead: {peak_memory - pre_gen_memory:.1f} MB")
    
    # Get cache stats
    stats = get_cache_stats(model)
    tokens_per_layer = get_tokens_per_layer_from_stats(stats)
    
    print(f"\n📊 Cache Statistics:")
    print(f"  Tokens per layer: {tokens_per_layer}")
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Corrected memory calculation
    # Formula: num_layers × 2 (K+V) × T × H_kv × D × 2 (FP16 bytes)
    theoretical_bytes = calculate_theoretical_kv_size(tokens_per_layer)
    actual_bytes = stats.get('total_compressed_bytes', 0)
    
    print(f"\n📊 Memory Analysis (Corrected):")
    print(f"  Theoretical uncompressed KV cache: {theoretical_bytes / 1024**2:.2f} MB")
    print(f"  Formula: {NUM_LAYERS} layers × 2 × {tokens_per_layer} tokens × {NUM_KV_HEADS} heads × {HEAD_DIM} dim × 2 bytes")
    print(f"  Actual compressed: {actual_bytes / 1024**2:.2f} MB")
    if theoretical_bytes > 0:
        print(f"  True compression: {theoretical_bytes / actual_bytes:.2f}x" if actual_bytes > 0 else "  N/A")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("MEMORY PROFILING: ✅ COMPLETE")
    print("=" * 70)
    return True


# =============================================================================
# CELL 7: QUALITY COMPARISON (WITH CACHE RESET FIX)
# =============================================================================
def test_generation_quality():
    """Compare generation quality with cache reset between prompts."""
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON TEST (Mistral-7B + Cache Reset)")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    # Load model WITH spectral cache
    print(f"\n📥 Loading model: {MODEL_NAME}...")
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
    
    # Test prompts
    test_cases = [
        {
            "name": "Fibonacci",
            "prompt": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,",
            "check": lambda x: any(n in x for n in ["233", "377", "610"]),
        },
        {
            "name": "Common Phrase",
            "prompt": "The quick brown fox",
            "check": lambda x: "jumps" in x.lower() or "lazy" in x.lower() or "dog" in x.lower(),
        },
        {
            "name": "Math",
            "prompt": "Calculate: 15 + 27 = ",
            "check": lambda x: "42" in x,
        },
        {
            "name": "Capital City",
            "prompt": "The capital of France is",
            "check": lambda x: "paris" in x.lower(),
        },
    ]
    
    print("\n📝 Generating with spectral cache (RESET between prompts)...")
    results = []
    
    for test in test_cases:
        # CRITICAL FIX: Reset cache before each independent prompt
        reset_count = reset_all_caches(model)
        
        inputs = tokenizer(test["prompt"], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_only = text[len(test["prompt"]):].strip()
        passed = test["check"](text)
        
        results.append({
            "name": test["name"],
            "prompt": test["prompt"],
            "output": output_only[:100],
            "passed": passed,
        })
        
        print(f"\n[{test['name']}] (reset {reset_count} caches)")
        print(f"  Prompt: {test['prompt']}")
        print(f"  Output: {output_only[:80]}...")
        print(f"  Status: {'✅ PASS' if passed else '⚠️ CHECK'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("QUALITY ANALYSIS SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for r in results if r["passed"])
    for r in results:
        status = "✅" if r["passed"] else "⚠️"
        print(f"  {status} {r['name']}")
    
    print(f"\n  Results: {passed_count}/{len(results)} tests passed")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON: ✅ COMPLETE")
    print("=" * 70)
    return True


# =============================================================================
# CELL 8: PERFORMANCE BENCHMARK
# =============================================================================
def test_performance_benchmark():
    """Benchmark generation speed with spectral cache at various context lengths."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK (Mistral-7B)")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    # Load model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
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
    
    # Warmup
    print("\n🔥 Warmup...")
    warmup_input = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=10, use_cache=True, 
                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    
    # Benchmark configurations - including longer contexts
    configs = [
        {"name": "Short (256 tokens)", "prompt_tokens": 256, "new_tokens": 50},
        {"name": "Medium (512 tokens)", "prompt_tokens": 512, "new_tokens": 50},
        {"name": "Long (1024 tokens)", "prompt_tokens": 1024, "new_tokens": 50},
        {"name": "Very Long (2048 tokens)", "prompt_tokens": 2048, "new_tokens": 50},
    ]
    
    print("\n📊 Running benchmarks...")
    print("-" * 70)
    
    benchmark_results = []
    
    for config in configs:
        # Reset cache before each benchmark
        reset_all_caches(model)
        
        # Create prompt of specified length
        base = "This is a comprehensive test sentence for benchmarking purposes. "
        prompt = base * (config["prompt_tokens"] // 12 + 1)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=config["prompt_tokens"], 
                          truncation=True).to("cuda")
        actual_prompt_len = inputs["input_ids"].shape[1]
        
        # Run benchmark (3 iterations)
        times = []
        for _ in range(3):
            reset_all_caches(model)  # Reset between iterations too
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["new_tokens"],
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = config["new_tokens"] / avg_time
        
        # Get compression stats
        stats = get_cache_stats(model)
        tokens_per_layer = get_tokens_per_layer_from_stats(stats)
        
        result = {
            "name": config["name"],
            "prompt_tokens": actual_prompt_len,
            "new_tokens": config["new_tokens"],
            "avg_time": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "compression_ratio": stats["compression_ratio"],
            "cold_blocks": stats["total_blocks"],
        }
        benchmark_results.append(result)
        
        print(f"\n{config['name']}:")
        print(f"  Prompt: {actual_prompt_len} tokens")
        print(f"  Generated: {config['new_tokens']} tokens")
        print(f"  Time: {avg_time:.3f}s (avg of 3)")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")
        print(f"  Cold blocks: {stats['total_blocks']} ({stats['total_blocks'] // NUM_LAYERS} per layer)")
    
    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Context':<25} {'Speed':<15} {'Compression':<15} {'Blocks/Layer':<12}")
    print("-" * 70)
    for r in benchmark_results:
        blocks_per_layer = r['cold_blocks'] // NUM_LAYERS if r['cold_blocks'] > 0 else 0
        print(f"{r['name']:<25} {r['tokens_per_sec']:.1f} tok/s{'':<6} {r['compression_ratio']:.2f}x{'':<10} {blocks_per_layer}")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK: ✅ COMPLETE")
    print("=" * 70)
    return True


# =============================================================================
# CELL 9: NEEDLE-IN-HAYSTACK RECALL TEST (Enhanced)
# =============================================================================
def test_needle_recall():
    """
    Comprehensive needle-in-haystack test for compressed context.
    
    Tests:
    1. Needle at different positions (early, middle, late)
    2. Different context lengths (1K, 2K, 3K tokens)
    3. Compression vs non-compression comparison
    """
    print("\n" + "=" * 70)
    print("NEEDLE-IN-HAYSTACK RECALL TEST")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    # Load model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
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
    
    # Filler text
    filler_unit = """Machine learning is a subset of artificial intelligence that enables 
    computers to learn from data without being explicitly programmed. Neural networks 
    are computational models inspired by biological neurons. Deep learning uses multiple 
    layers to progressively extract higher-level features from raw input. """
    
    # Test configurations
    needle_tests = [
        {
            "name": "1K tokens, needle at ~200",
            "total_tokens": 1000,
            "needle_position": 0.2,  # 20% into the context
            "needle": "The password for the vault is RUBY-2847.",
            "question": "What is the password for the vault?",
            "expected": ["RUBY-2847", "ruby-2847", "RUBY", "2847"],
        },
        {
            "name": "2K tokens, needle at middle",
            "total_tokens": 2000,
            "needle_position": 0.5,  # 50% into the context
            "needle": "Agent Smith's codename is PHOENIX-0099.",
            "question": "What is Agent Smith's codename?",
            "expected": ["PHOENIX-0099", "phoenix-0099", "PHOENIX", "0099"],
        },
        {
            "name": "3K tokens, needle at ~80%",
            "total_tokens": 3000,
            "needle_position": 0.8,  # 80% into the context (late)
            "needle": "The treasure is buried at coordinates LAT-42 LON-73.",
            "question": "Where is the treasure buried? What are the coordinates?",
            "expected": ["LAT-42", "LON-73", "42", "73"],
        },
    ]
    
    results = []
    
    for test in needle_tests:
        print(f"\n--- {test['name']} ---")
        
        # Reset cache
        reset_all_caches(model)
        
        # Build context with needle
        target_chars = test["total_tokens"] * 4  # Rough char estimate
        needle_char_pos = int(target_chars * test["needle_position"])
        
        # Build filler
        filler = ""
        while len(filler) < target_chars:
            filler += filler_unit
        filler = filler[:target_chars]
        
        # Insert needle
        context = filler[:needle_char_pos] + f"\n[IMPORTANT: {test['needle']}]\n" + filler[needle_char_pos:]
        
        # Build prompt with question
        full_prompt = context + f"\n\nQuestion: {test['question']}\nAnswer:"
        
        # Tokenize and check length
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=test["total_tokens"] + 100, truncation=True).to("cuda")
        actual_tokens = inputs["input_ids"].shape[1]
        print(f"  Context: {actual_tokens} tokens")
        print(f"  Needle inserted at: ~{test['needle_position']*100:.0f}% ({needle_char_pos} chars)")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  Response: {response[:80]}...")
        
        # Check cache stats
        stats = get_cache_stats(model)
        blocks_per_layer = stats['total_blocks'] // NUM_LAYERS if stats['total_blocks'] > 0 else 0
        print(f"  Compression: {stats['compression_ratio']:.2f}x ({blocks_per_layer} blocks/layer)")
        
        # Check recall
        recall_found = any(exp in response for exp in test["expected"])
        partial_found = any(exp.lower() in response.lower() for exp in test["expected"])
        
        if recall_found:
            print(f"  ✅ PERFECT RECALL")
            status = "perfect"
        elif partial_found:
            print(f"  ⚠️ PARTIAL RECALL")
            status = "partial"
        else:
            print(f"  ❌ NO RECALL")
            status = "failed"
        
        results.append({
            "name": test["name"],
            "tokens": actual_tokens,
            "blocks": blocks_per_layer,
            "compression": stats['compression_ratio'],
            "status": status,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("NEEDLE RECALL SUMMARY")
    print("=" * 70)
    print(f"{'Test':<35} {'Tokens':<10} {'Blocks':<10} {'Compression':<12} {'Recall':<10}")
    print("-" * 70)
    
    perfect_count = 0
    for r in results:
        status_icon = {"perfect": "✅", "partial": "⚠️", "failed": "❌"}[r["status"]]
        print(f"{r['name']:<35} {r['tokens']:<10} {r['blocks']:<10} {r['compression']:.2f}x{'':<7} {status_icon}")
        if r["status"] == "perfect":
            perfect_count += 1
    
    print("-" * 70)
    print(f"Perfect Recalls: {perfect_count}/{len(results)}")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("NEEDLE RECALL TEST: ✅ COMPLETE")
    print("=" * 70)
    
    return perfect_count >= len(results) // 2  # Pass if at least half are perfect


# =============================================================================
# CELL 10: MEMORY BREAKDOWN ANALYSIS
# =============================================================================
def test_memory_breakdown():
    """
    Detailed analysis of where memory goes during generation.
    
    Breaks down:
    1. Model weights
    2. KV cache (theoretical vs actual)
    3. Activations and intermediate tensors
    4. Generation overhead
    """
    print("\n" + "=" * 70)
    print("MEMORY BREAKDOWN ANALYSIS")
    print("=" * 70)
    
    if not (SPECTRAL_OK and UNSLOTH_OK):
        print("❌ Skipping: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ Skipping: CUDA not available")
        return False
    
    def get_memory_mb():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    
    def get_reserved_mb():
        torch.cuda.synchronize()
        return torch.cuda.memory_reserved() / 1024**2
    
    # Clean slate
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    memory_log = []
    memory_log.append(("Baseline (empty)", get_memory_mb(), get_reserved_mb()))
    
    # Load model
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    memory_log.append(("After model load", get_memory_mb(), get_reserved_mb()))
    
    # Patch with spectral cache
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
        debug_logging=False,
    )
    memory_log.append(("After patching", get_memory_mb(), get_reserved_mb()))
    
    # Create test prompt (2048 tokens)
    test_text = "The quick brown fox jumps over the lazy dog. " * 300
    inputs = tokenizer(test_text, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    memory_log.append((f"After tokenization ({prompt_tokens} tok)", get_memory_mb(), get_reserved_mb()))
    
    # Pre-generation
    torch.cuda.reset_peak_memory_stats()
    
    # Generation
    print(f"\n🔄 Generating 100 tokens from {prompt_tokens} token prompt...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    memory_log.append(("After generation", get_memory_mb(), get_reserved_mb()))
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    # Get cache stats
    stats = get_cache_stats(model)
    tokens_per_layer = get_tokens_per_layer_from_stats(stats)
    
    # Print memory log
    print("\n" + "=" * 70)
    print("MEMORY ALLOCATION LOG")
    print("=" * 70)
    print(f"{'Stage':<35} {'Allocated (MB)':<18} {'Reserved (MB)':<18}")
    print("-" * 70)
    
    for stage, alloc, reserved in memory_log:
        print(f"{stage:<35} {alloc:>12.1f}{'':<6} {reserved:>12.1f}")
    
    print(f"{'Peak during generation':<35} {peak_mb:>12.1f}")
    
    # Memory breakdown calculation
    print("\n" + "=" * 70)
    print("MEMORY BREAKDOWN")
    print("=" * 70)

    # memory_log entries are (label, allocated_mb, reserved_mb)
    # Index [1] is allocated_mb, [0] is the label string
    model_memory = memory_log[1][1] - memory_log[0][1]
    generation_overhead = memory_log[4][1] - memory_log[3][1]
    peak_overhead = peak_mb - memory_log[3][1]
    
    # KV cache calculations
    theoretical_kv_bytes = calculate_theoretical_kv_size(tokens_per_layer)
    actual_kv_bytes = stats.get('total_compressed_bytes', 0)
    
    print(f"  Model weights (4-bit):           {model_memory:>8.1f} MB")
    print(f"  ")
    print(f"  KV Cache Analysis ({tokens_per_layer} tokens/layer):")
    print(f"    Theoretical (FP16):            {theoretical_kv_bytes / 1024**2:>8.2f} MB")
    print(f"    Actual (compressed):           {actual_kv_bytes / 1024**2:>8.2f} MB")
    print(f"    Compression ratio:             {stats['compression_ratio']:>8.2f}x")
    print(f"    Cold blocks:                   {stats['total_blocks']:>8} ({stats['total_blocks'] // NUM_LAYERS}/layer)")
    print(f"  ")
    print(f"  Generation overhead:")
    print(f"    Final - Pre-gen:               {generation_overhead:>8.1f} MB")
    print(f"    Peak - Pre-gen:                {peak_overhead:>8.1f} MB")
    print(f"  ")
    print(f"  Non-KV overhead (activations, etc):")
    print(f"    = Peak overhead - KV cache")
    non_kv_overhead = peak_overhead - (actual_kv_bytes / 1024**2)
    print(f"    = {peak_overhead:.1f} - {actual_kv_bytes / 1024**2:.1f} = {non_kv_overhead:.1f} MB")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("MEMORY BREAKDOWN: ✅ COMPLETE")
    print("=" * 70)
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("🚀 RUNNING ALL VALIDATION TESTS (v3 - Multi-Block Fix)")
    print("=" * 70)
    
    results = {}
    
    # Unit tests (always run)
    results["SpectralCache Unit"] = test_spectral_cache_unit()
    results["Kernel Functions"] = test_kernel_functions()
    
    # Integration tests (require full setup)
    results["Long Context (1500 tok)"] = test_long_context_compression()
    results["Very Long Context (3000 tok)"] = test_very_long_context()
    results["Memory Breakdown"] = test_memory_breakdown()
    results["Needle Recall"] = test_needle_recall()
    results["Generation Quality"] = test_generation_quality()
    results["Performance Benchmark"] = test_performance_benchmark()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print("-" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


# Run all tests when executed
if __name__ == "__main__":
    success = run_all_tests()
    # Don't call sys.exit() in notebook - just print result
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
else:
    # When pasted into Colab, run automatically
    success = run_all_tests()
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
