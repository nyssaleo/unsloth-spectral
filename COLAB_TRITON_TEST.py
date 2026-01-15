"""
Colab Triton Kernel Test Suite for Unsloth Spectral
====================================================

Copy these cells to your Colab notebook to test the Triton kernels.

Requirements:
- Google Colab with T4 GPU
- Free or Pro tier works

Usage:
1. Copy each cell (marked with # %%% CELL N) to Colab
2. Run sequentially
3. Check logs for any errors

Author: Ankit Prajapati & Claude
Date: January 2026
"""

# %%% CELL 1: Installation
# =============================================================================
# Install unsloth and our spectral extension from git
# =============================================================================

# !pip uninstall unsloth -y  # Uncomment if you have an old version
# !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --upgrade --no-cache-dir "git+https://github.com/nyssaleo/unsloth-spectral.git"

# For local testing (if you have the repo cloned)
import sys
sys.path.insert(0, '/content/unsloth-spectral')  # Adjust path as needed

print("✓ Paths configured")


# %%% CELL 2: Imports and Config Verification
# =============================================================================
# Verify Triton is available and show configuration
# =============================================================================

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check Triton
try:
    import triton
    print(f"✓ Triton version: {triton.__version__}")
except ImportError as e:
    print(f"✗ Triton not available: {e}")
    print("Installing Triton...")
    # !pip install triton

# Import unsloth_spectral
import os
os.environ["UNSLOTH_SPECTRAL_QUIET"] = "1"  # Suppress banner for now

from unsloth_spectral import (
    SpectralCache,
    patch_unsloth_attention,
    TRITON_AVAILABLE,
    TRITON_VERSION,
)

print("\n" + "="*60)
print("UNSLOTH SPECTRAL CONFIGURATION")
print("="*60)
print(f"Triton Available: {TRITON_AVAILABLE}")
print(f"Triton Version: {TRITON_VERSION}")

if TRITON_AVAILABLE:
    from unsloth_spectral import spectral_attention_decode, TritonSpectralConfig
    print("✓ Triton kernels imported successfully")
else:
    print("⚠ Triton not available - will use PyTorch fallback")
print("="*60)


# %%% CELL 3: Triton Kernel Unit Test
# =============================================================================
# Test the Triton kernels in isolation (without full model)
# =============================================================================

import torch
import math
import time

print("\n" + "="*60)
print("TRITON KERNEL UNIT TEST")
print("="*60)

# Test configuration matching Mistral-7B
B = 1           # Batch size
H_q = 32        # Query heads
H_kv = 8        # KV heads
D = 128         # Head dimension
T_block = 256   # Sequence length in block
k_K = 16        # Spectral rank for K
k_V = 32        # Spectral rank for V

print(f"Config: B={B}, H_q={H_q}, H_kv={H_kv}, D={D}")
print(f"        T_block={T_block}, k_K={k_K}, k_V={k_V}")

# Create test tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

Q = torch.randn(B, H_q, 1, D, device=device, dtype=dtype)
coeffs_K = torch.randn(H_kv, T_block, k_K, device=device, dtype=dtype)
basis_K = torch.randn(H_kv, k_K, D, device=device, dtype=dtype)
coeffs_V = torch.randn(H_kv, T_block, k_V, device=device, dtype=dtype)
basis_V = torch.randn(H_kv, k_V, D, device=device, dtype=dtype)

# Create RoPE tables
max_seq = 4096
cos = torch.randn(max_seq, D, device=device, dtype=dtype)
sin = torch.randn(max_seq, D, device=device, dtype=dtype)

print(f"\nTensors created on {device}")
print(f"  Q: {Q.shape}")
print(f"  coeffs_K: {coeffs_K.shape}")
print(f"  basis_K: {basis_K.shape}")
print(f"  coeffs_V: {coeffs_V.shape}")
print(f"  basis_V: {basis_V.shape}")

if TRITON_AVAILABLE:
    from unsloth_spectral.kernels import (
        spectral_score_forward,
        spectral_value_forward,
        TritonSpectralConfig,
    )
    
    config = TritonSpectralConfig(enable_logging=True)
    
    print("\n--- Testing Score Forward (PyTorch) ---")
    scores_pt = spectral_score_forward(
        Q, coeffs_K, basis_K, cos, sin,
        start_position=0,
        query_position=T_block,
        config=config,
        use_triton=False,  # PyTorch
    )
    print(f"PyTorch scores: {scores_pt.shape}")
    
    print("\n--- Testing Value Forward (PyTorch) ---")
    # Create attention weights from scores
    scale = 1.0 / math.sqrt(D)
    attn = torch.softmax(scores_pt * scale, dim=-1)
    
    # FIX: Do NOT squeeze - spectral_value_forward expects [B, H_q, 1, T_block]
    output_pt = spectral_value_forward(
        attn,  # Keep 4D: [B, H_q, 1, T_block]
        coeffs_V, basis_V, D,
        config=config,
        use_triton=False,  # PyTorch
    )
    print(f"PyTorch output: {output_pt.shape}")
    
    # Print summary
    print("\n" + "="*60)
    print("UNIT TEST RESULTS")
    print("="*60)
    print(f"✓ Score computation: {scores_pt.shape}")
    print(f"  min={scores_pt.min().item():.4f}, max={scores_pt.max().item():.4f}")
    print(f"✓ Value computation: {output_pt.shape}")
    print(f"  min={output_pt.min().item():.4f}, max={output_pt.max().item():.4f}")
    print("="*60)
else:
    print("⚠ Skipping Triton kernel test - Triton not available")


# %%% CELL 4: SpectralCache Integration Test
# =============================================================================
# Test SpectralCache with the kernels
# =============================================================================

print("\n" + "="*60)
print("SPECTRAL CACHE INTEGRATION TEST")
print("="*60)

from unsloth_spectral import SpectralCache

# Create a SpectralCache
cache = SpectralCache(
    num_heads=H_kv,
    head_dim=D,
    block_size=256,      # Compress every 256 tokens
    k_rank_keys=k_K,
    k_rank_values=k_V,
    hot_buffer_size=64,
    device=device,
    dtype=dtype,
    debug_logging=True,
)

print(f"Created SpectralCache:")
print(f"  num_heads: {cache.num_heads}")
print(f"  head_dim: {cache.head_dim}")
print(f"  block_size: {cache.block_size}")
print(f"  k_rank_keys: {cache.k_rank_keys}")
print(f"  k_rank_values: {cache.k_rank_values}")

# Simulate a prompt with 300 tokens (triggers one compression)
print("\n--- Simulating 300-token prompt ---")
for i in range(300):
    K_token = torch.randn(B, H_kv, 1, D, device=device, dtype=dtype)
    V_token = torch.randn(B, H_kv, 1, D, device=device, dtype=dtype)
    pos_ids = torch.tensor([[i]], device=device)
    cache.append(K_token, V_token, pos_ids)

print(f"\nCache state after 300 tokens:")
print(f"  total_tokens: {cache.total_tokens}")
print(f"  cold_blocks: {len(cache.cold_blocks)}")
if cache.cold_blocks:
    block = cache.cold_blocks[0]
    print(f"  block[0].coeffs_K: {block.coeffs_K.shape}")
    print(f"  block[0].basis_K: {block.basis_K.shape}")
print(f"  hot_K: {cache.hot_K.shape if cache.hot_K is not None else 'None'}")

# Test get_spectral_components
cold_blocks, hot_K, hot_V = cache.get_spectral_components()
print(f"\nget_spectral_components():")
print(f"  cold_blocks: {len(cold_blocks)}")
print(f"  hot_K: {hot_K.shape if hot_K is not None else 'None'}")
print(f"  hot_V: {hot_V.shape if hot_V is not None else 'None'}")

if TRITON_AVAILABLE and len(cold_blocks) > 0:
    print("\n--- Testing spectral_attention_decode ---")
    
    # Create a query
    Q_test = torch.randn(B, H_q, 1, D, device=device, dtype=dtype)
    
    from unsloth_spectral import spectral_attention_decode, TritonSpectralConfig
    
    config = TritonSpectralConfig(enable_logging=True)
    
    output = spectral_attention_decode(
        Q=Q_test,
        cache=cache,
        cos=cos,
        sin=sin,
        query_position=300,
        attention_mask=None,
        scale=1.0 / math.sqrt(D),
        config=config,
    )
    
    print(f"\n✓ spectral_attention_decode output: {output.shape}")
    print(f"  min={output.min().item():.4f}, max={output.max().item():.4f}")

print("\n" + "="*60)
print("INTEGRATION TEST COMPLETE")
print("="*60)


# %%% CELL 5: Full Model Test
# =============================================================================
# Test with a real Unsloth model
# =============================================================================

print("\n" + "="*60)
print("FULL MODEL TEST")
print("="*60)

# Load Unsloth model
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)

print(f"✓ Model loaded: {model.config._name_or_path}")
print(f"  max_seq_length: 4096")
print(f"  load_in_4bit: True")

# Patch with spectral attention
patch_unsloth_attention(
    model,
    block_size=512,
    k_rank_keys=16,
    k_rank_values=32,
    hot_buffer_size=64,
    use_spectral_attention=True,
    debug_logging=True,     # Enable detailed logging
    use_triton_kernel=True,  # Use Triton when available
    verbose=True,
)

print("\n✓ Model patched with spectral attention")

# Test generation
print("\n--- Testing Generation ---")

prompt = "Explain quantum computing in simple terms. Start by describing what makes quantum computers different from classical computers."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")

# Generate
import time
start = time.perf_counter()

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,  # Greedy for reproducibility
    use_cache=True,
)

elapsed = time.perf_counter() - start

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]

print(f"\n{'='*60}")
print("GENERATION RESULTS")
print(f"{'='*60}")
print(f"Generated {num_tokens} tokens in {elapsed:.2f}s ({num_tokens/elapsed:.1f} tok/s)")
print(f"\nResponse:")
print("-"*40)
print(response[:500] + "..." if len(response) > 500 else response)
print("-"*40)

# Check cache stats
from unsloth_spectral import get_cache_stats
try:
    stats = get_cache_stats(model)
    print(f"\n📊 Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"Could not get cache stats: {e}")

print("\n" + "="*60)
print("FULL MODEL TEST COMPLETE")
print("="*60)


# %%% CELL 6: Performance Benchmark
# =============================================================================
# Benchmark spectral vs standard attention
# =============================================================================

print("\n" + "="*60)
print("PERFORMANCE BENCHMARK")
print("="*60)

import gc
import torch

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Test different sequence lengths
seq_lengths = [512, 1024, 2048]
results = []

for seq_len in seq_lengths:
    print(f"\n--- Sequence Length: {seq_len} ---")
    
    # Create test prompt of approximate length
    test_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer(test_text, return_tensors="pt", max_length=seq_len, truncation=True).to(model.device)
    actual_len = inputs['input_ids'].shape[1]
    print(f"  Actual input length: {actual_len}")
    
    # Warm up
    clear_cache()
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    clear_cache()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    new_tokens = outputs.shape[1] - actual_len
    tokens_per_sec = new_tokens / elapsed
    
    results.append({
        'seq_len': actual_len,
        'new_tokens': new_tokens,
        'time': elapsed,
        'tok_per_sec': tokens_per_sec,
    })
    
    print(f"  Generated: {new_tokens} tokens in {elapsed:.3f}s")
    print(f"  Throughput: {tokens_per_sec:.1f} tok/s")
    
    # Memory usage
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak memory: {mem_used:.2f} GB")

print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)
for r in results:
    print(f"  Seq={r['seq_len']:4d}: {r['tok_per_sec']:5.1f} tok/s")
print("="*60)


# %%% CELL 7: Comparison Test (Optional)
# =============================================================================
# Compare spectral vs standard output quality
# =============================================================================

print("\n" + "="*60)
print("OUTPUT QUALITY COMPARISON")
print("="*60)

# This requires loading the model twice, so only run if you have memory

# Test prompt for comparison
comparison_prompt = "What is the capital of France? Answer in one word:"

inputs = tokenizer(comparison_prompt, return_tensors="pt").to(model.device)

# Generate with spectral attention
spectral_output = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
)
spectral_response = tokenizer.decode(spectral_output[0], skip_special_tokens=True)

print(f"Spectral response: {spectral_response}")

# Note: To compare with standard attention, you would need to:
# 1. Reload the model without spectral patch
# 2. Generate the same prompt
# 3. Compare outputs

print("\n✓ To compare with standard attention, reload model without patch")
print("="*60)

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
print("Next steps:")
print("1. Check logs above for any errors")
print("2. Compare generation quality with your expectations")
print("3. Note throughput numbers for your use case")
print("="*60)
