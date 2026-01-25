"""
SPECTRAL CACHE TRANSPARENCY TEST
=================================

This script shows EXACTLY what's happening with the spectral cache:
- When caches are created vs reused
- Cache state evolution (hot/cold/compression)
- Per-layer breakdown
- Baseline vs Spectral comparison
- Token-by-token output analysis

Run this to SEE the internals and identify any flaws.
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages."""
    packages = [
        ("triton", "triton"),
        ("unsloth", "unsloth"),
    ]
    
    for name, pkg in packages:
        print(f"📦 Installing {name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        print(f"  ✅ {name} installed")
    
    # Install unsloth_spectral from GitHub
    print("📦 Installing unsloth_spectral...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/nyssaleo/unsloth-spectral.git"
    ])
    print("  ✅ unsloth_spectral installed")

# Run installation
install_packages()

import torch
import gc
import time
from typing import Dict, List, Tuple
from collections import defaultdict

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

from unsloth import FastLanguageModel

# Import spectral (but don't patch yet)
os.environ["UNSLOTH_SPECTRAL_QUIET"] = "1"
from unsloth_spectral import patch_unsloth_attention, get_cache_stats

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 8192

# Test prompt (medium length to trigger compression)
TEST_PROMPT = """You are a helpful AI assistant. Please answer the following question.

Context: In the field of quantum computing, qubits are the fundamental units of information. 
Unlike classical bits which can only be 0 or 1, qubits can exist in superposition states. 
This property, combined with entanglement, gives quantum computers their power. The secret 
research code for our quantum project is QUANTUM-ALPHA-9527. This code must be remembered 
for later reference.

Additional context: Quantum algorithms like Shor's algorithm and Grover's algorithm demonstrate 
exponential speedups over classical algorithms. The development of error correction techniques 
is crucial for building practical quantum computers. Current quantum computers are in the NISQ 
(Noisy Intermediate-Scale Quantum) era.

Question: What is the secret research code mentioned in the context above?
Answer:"""

# =============================================================================
# MONITORING INFRASTRUCTURE
# =============================================================================

class CacheMonitor:
    """Monitor cache state changes during generation."""
    
    def __init__(self, model, name=""):
        self.model = model
        self.name = name
        self.snapshots = []
        
    def snapshot(self, step: int, description: str = ""):
        """Capture current cache state."""
        snapshot = {
            "step": step,
            "description": description,
            "layers": {}
        }
        
        # Iterate through all layers
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.self_attn, '_spectral_cache'):
                cache = layer.self_attn._spectral_cache
                if cache is not None:
                    snapshot["layers"][layer_idx] = {
                        "cache_id": id(cache),
                        "total_tokens": cache.total_tokens,
                        "num_blocks": len(cache.cold_blocks),
                        "hot_tokens": cache.hot_K.shape[2] if cache.hot_K is not None else 0,
                        "compression_count": cache.compression_count,
                    }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def print_snapshot(self, snapshot: dict):
        """Print a snapshot in readable format."""
        print(f"\n{'='*70}")
        print(f"📸 SNAPSHOT: Step {snapshot['step']} - {snapshot['description']}")
        print(f"{'='*70}")
        
        if not snapshot["layers"]:
            print("  (No caches found)")
            return
        
        # Show stats for first, middle, last layer
        key_layers = [0, 15, 31]
        for layer_idx in key_layers:
            if layer_idx in snapshot["layers"]:
                info = snapshot["layers"][layer_idx]
                print(f"\nLayer {layer_idx}:")
                print(f"  Cache ID:        {info['cache_id']}")
                print(f"  Total tokens:    {info['total_tokens']}")
                print(f"  Cold blocks:     {info['num_blocks']}")
                print(f"  Hot tokens:      {info['hot_tokens']}")
                print(f"  Compressions:    {info['compression_count']}")
                
                if info['total_tokens'] > 0:
                    compression = (info['num_blocks'] * 512 + info['hot_tokens']) / info['total_tokens']
                    print(f"  Est. compression: {1.0/compression:.2f}x" if compression > 0 else "  (no compression)")
    
    def print_summary(self):
        """Print summary of all snapshots."""
        print(f"\n{'='*70}")
        print(f"📊 CACHE EVOLUTION SUMMARY - {self.name}")
        print(f"{'='*70}")
        
        if not self.snapshots:
            print("  (No snapshots)")
            return
        
        # Track cache ID persistence (Layer 15 as example)
        layer_15_ids = []
        for snapshot in self.snapshots:
            if 15 in snapshot["layers"]:
                layer_15_ids.append(snapshot["layers"][15]["cache_id"])
        
        if layer_15_ids:
            unique_ids = len(set(layer_15_ids))
            if unique_ids == 1:
                print(f"✅ Cache persistence: GOOD (same cache object reused)")
            else:
                print(f"❌ Cache persistence: BAD (cache recreated {unique_ids} times!)")
                print(f"   Cache IDs: {[hex(id) for id in layer_15_ids]}")
        
        # Show token growth
        print(f"\n📈 Token Growth (Layer 15):")
        for snapshot in self.snapshots:
            if 15 in snapshot["layers"]:
                info = snapshot["layers"][15]
                print(f"  Step {snapshot['step']:2d}: {info['total_tokens']:4d} tokens "
                      f"({info['num_blocks']} blocks + {info['hot_tokens']} hot)")

# =============================================================================
# PART 1: BASELINE (NO PATCHING)
# =============================================================================

def test_baseline():
    """Test baseline Unsloth without spectral cache."""
    print("\n" + "="*70)
    print("PART 1: BASELINE (No Spectral Cache)")
    print("="*70)
    
    # Load model
    print("\n📥 Loading model (baseline)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Prepare input
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]
    print(f"  Prompt length: {prompt_tokens} tokens")
    
    # Generate with cache inspection
    print("\n🔄 Generating (baseline)...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=False,
        )
    
    elapsed = time.time() - start_time
    
    # Decode output
    output_text = tokenizer.decode(
        outputs.sequences[0][prompt_tokens:],
        skip_special_tokens=True
    )
    
    print(f"\n⏱️  Time: {elapsed:.2f}s")
    print(f"📝 Output: {output_text}")
    
    # Inspect past_key_values structure
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        past_kv = outputs.past_key_values
        print(f"\n🔍 Cache Structure:")
        print(f"  Type: {type(past_kv)}")
        print(f"  Layers: {len(past_kv)}")
        
        # Inspect first layer
        if past_kv[0] is not None:
            K, V = past_kv[0]
            print(f"  Layer 0 K shape: {K.shape}  (B, H, T, D)")
            print(f"  Layer 0 V shape: {V.shape}")
            print(f"  Storage: Raw FP16 tensors")
            
            # Calculate memory
            memory_bytes = K.numel() * 2 + V.numel() * 2  # FP16 = 2 bytes
            memory_mb = memory_bytes / (1024 * 1024) * len(past_kv)
            print(f"  Estimated memory: {memory_mb:.1f} MB")
    else:
        print(f"\n⚠️  No past_key_values returned!")
    
    # Cleanup
    result = {
        "output": output_text,
        "time": elapsed,
        "prompt_tokens": prompt_tokens,
    }
    
    del model, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

# =============================================================================
# PART 2: SPECTRAL (WITH PATCHING)
# =============================================================================

def test_spectral(config: dict):
    """Test with spectral cache, full transparency."""
    print("\n" + "="*70)
    print(f"PART 2: SPECTRAL CACHE (k_K={config['k_K']}, k_V={config['k_V']})")
    print("="*70)
    
    # Load model
    print("\n📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Patch with VERBOSE logging
    print("\n🔧 Patching with spectral cache...")
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=config['k_K'],
        k_rank_values=config['k_V'],
        hot_buffer_size=64,
        use_spectral_attention=True,
        verbose=True,
        debug_logging=False,  # Keep False for now, we have our own logging
        landmark_count=config.get('landmark_count', 0),
    )
    
    # Create monitor
    monitor = CacheMonitor(model, name=f"Spectral k_K={config['k_K']}")
    
    # Prepare input
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]
    print(f"  Prompt length: {prompt_tokens} tokens")
    
    # Snapshot 0: Before generation
    monitor.snapshot(0, "Before generation")
    
    # Generate with monitoring
    print("\n🔄 Generating...")
    
    # Snapshot 0: Before generation
    snap = monitor.snapshot(0, "Before generation")
    monitor.print_snapshot(snap)
    
    start_time = time.time()
    
    # Use standard generate() - simpler and avoids attention_mask issues
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    elapsed = time.time() - start_time
    
    # Snapshot after generation
    snap = monitor.snapshot(1, "After generation complete")
    monitor.print_snapshot(snap)
    
    # Print summary (shows cache ID persistence)
    monitor.print_summary()
    
    # Decode output
    output_text = tokenizer.decode(
        outputs[0][prompt_tokens:],
        skip_special_tokens=True
    )
    
    print(f"\n⏱️  Time: {elapsed:.2f}s")
    print(f"📝 Output: {output_text}")
    
    # Get compression stats
    print(f"\n📊 COMPRESSION STATS:")
    stats = get_cache_stats(model)
    if stats:
        print(f"  Total tokens:      {stats.get('total_tokens', 0)}")
        print(f"  Total blocks:      {stats.get('total_blocks', 0)}")
        print(f"  Avg compression:   {stats.get('avg_compression', 0):.2f}x")
        print(f"  Original size:     {stats.get('original_bytes', 0) / 1024:.1f} KB")
        print(f"  Compressed size:   {stats.get('compressed_bytes', 0) / 1024:.1f} KB")
    
    # Per-layer breakdown
    print(f"\n📊 PER-LAYER BREAKDOWN:")
    print(f"{'Layer':<8} {'Tokens':<10} {'Blocks':<8} {'Hot':<8} {'Compression':<12}")
    print("-" * 60)
    
    for layer_idx in [0, 7, 15, 23, 31]:  # Sample layers
        layer = model.model.layers[layer_idx]
        if hasattr(layer.self_attn, '_spectral_cache'):
            cache = layer.self_attn._spectral_cache
            if cache is not None:
                total = cache.total_tokens
                blocks = len(cache.cold_blocks)
                hot = cache.hot_K.shape[2] if cache.hot_K is not None else 0
                
                if total > 0:
                    # Calculate compression
                    original_size = total * cache.num_heads * cache.head_dim * 2 * 2  # K and V, FP16
                    compressed_size = 0
                    for block in cache.cold_blocks:
                        compressed_size += block.coeffs_K.numel() * 2  # Simulated INT8 but stored FP16
                        compressed_size += block.basis_K.numel() * 2
                        compressed_size += block.coeffs_V.numel() * 2
                        compressed_size += block.basis_V.numel() * 2
                    if cache.hot_K is not None:
                        compressed_size += cache.hot_K.numel() * 2
                        compressed_size += cache.hot_V.numel() * 2
                    
                    compression = original_size / compressed_size if compressed_size > 0 else 1.0
                    print(f"{layer_idx:<8} {total:<10} {blocks:<8} {hot:<8} {compression:.2f}x")
    
    # Cleanup
    result = {
        "output": output_text,
        "time": elapsed,
        "prompt_tokens": prompt_tokens,
        "monitor": monitor,
    }
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

# =============================================================================
# PART 3: COMPARISON
# =============================================================================

def compare_results(baseline: dict, spectral: dict):
    """Compare baseline and spectral results."""
    print("\n" + "="*70)
    print("PART 3: BASELINE vs SPECTRAL COMPARISON")
    print("="*70)
    
    # Output comparison
    print("\n📝 OUTPUT COMPARISON:")
    print(f"\nBaseline: {baseline['output']}")
    print(f"\nSpectral: {spectral['output']}")
    
    # Token-by-token
    baseline_tokens = baseline['output'].split()
    spectral_tokens = spectral['output'].split()
    
    max_len = max(len(baseline_tokens), len(spectral_tokens))
    
    print(f"\n📊 TOKEN-BY-TOKEN COMPARISON:")
    print(f"{'Position':<10} {'Baseline':<20} {'Spectral':<20} {'Match':<10}")
    print("-" * 70)
    
    matches = 0
    for i in range(max_len):
        base_tok = baseline_tokens[i] if i < len(baseline_tokens) else "(none)"
        spec_tok = spectral_tokens[i] if i < len(spectral_tokens) else "(none)"
        match = "✅" if base_tok == spec_tok else "❌"
        if base_tok == spec_tok:
            matches += 1
        print(f"{i:<10} {base_tok:<20} {spec_tok:<20} {match:<10}")
    
    # Overlap
    overlap = matches / max_len * 100 if max_len > 0 else 0
    print(f"\n📊 VOCABULARY OVERLAP: {overlap:.1f}%")
    
    # Performance
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Baseline time: {baseline['time']:.2f}s")
    print(f"  Spectral time: {spectral['time']:.2f}s")
    print(f"  Slowdown:      {spectral['time'] / baseline['time']:.2f}x")
    
    # Recall check
    needle = "QUANTUM-ALPHA-9527"
    baseline_recall = needle in baseline['output']
    spectral_recall = needle in spectral['output']
    
    print(f"\n🎯 NEEDLE RECALL ('{needle}'):")
    print(f"  Baseline: {'✅ Found' if baseline_recall else '❌ Not found'}")
    print(f"  Spectral: {'✅ Found' if spectral_recall else '❌ Not found'}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🔬 SPECTRAL CACHE TRANSPARENCY TEST")
    print("="*70)
    print("\nThis test will show you EXACTLY what's happening:")
    print("1. Baseline generation (no spectral)")
    print("2. Spectral generation with step-by-step monitoring")
    print("3. Detailed comparison")
    
    # Part 1: Baseline
    baseline = test_baseline()
    
    # Part 2: Spectral (default config)
    spectral_16 = test_spectral({
        'k_K': 16,
        'k_V': 32,
    })
    
    # Part 3: Compare
    compare_results(baseline, spectral_16)
    
    # Optional: Test with higher ranks
    print("\n" + "="*70)
    print("🔬 BONUS: Testing Higher Ranks (k_K=32, k_V=64)")
    print("="*70)
    
    spectral_32 = test_spectral({
        'k_K': 32,
        'k_V': 64,
    })
    
    compare_results(baseline, spectral_32)
    
    print("\n" + "="*70)
    print("✅ TRANSPARENCY TEST COMPLETE")
    print("="*70)
    print("\nKey Insights:")
    print("1. Check 'Cache persistence' - should be GOOD (same ID reused)")
    print("2. Check 'Token Growth' - should increase smoothly")
    print("3. Check 'Vocabulary Overlap' - should be 50-100%")
    print("4. Check 'Needle Recall' - this shows compression quality")

if __name__ == "__main__":
    main()
