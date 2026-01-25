"""
GROUND TRUTH LOGGER - GRANULAR INSTRUMENTATION
===============================================

This script instruments EVERYTHING to understand the exact difference between:
- Pure baseline Unsloth (no imports, no patches)
- Baseline after importing unsloth_spectral (contaminated?)
- Spectral with patches applied

Tracks:
1. Token-by-token generation
2. Cache state at each step
3. Attention computations
4. RoPE application
5. Module patching state
6. Function call traces
"""

import subprocess
import sys
import os

def install_packages():
    packages = [("triton", "triton"), ("unsloth", "unsloth")]
    for name, pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    # Install unsloth_spectral
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/nyssaleo/unsloth-spectral.git"
    ])

install_packages()

import torch
import gc
import json
import types
from typing import List, Dict, Any
from collections import defaultdict

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

from unsloth import FastLanguageModel

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 8192

# Short prompt to keep logging manageable
TEST_PROMPT = """Context: The secret code is ALPHA-123. 
Question: What is the secret code?
Answer:"""

# =============================================================================
# INSTRUMENTATION INFRASTRUCTURE
# =============================================================================

class GenerationLogger:
    """Logs every detail of generation process."""
    
    def __init__(self, name: str):
        self.name = name
        self.logs = []
        self.step = 0
        
    def log(self, event: str, data: Dict[str, Any]):
        """Log an event with data."""
        entry = {
            "step": self.step,
            "event": event,
            "data": data,
        }
        self.logs.append(entry)
        
    def log_token(self, token_id: int, token_text: str, logits_top5: List[tuple]):
        """Log a generated token."""
        self.log("token_generated", {
            "token_id": token_id,
            "token_text": token_text,
            "top5_logits": logits_top5,
        })
        self.step += 1
        
    def log_cache_state(self, layer_idx: int, cache_info: Dict):
        """Log cache state for a layer."""
        self.log("cache_state", {
            "layer": layer_idx,
            **cache_info
        })
    
    def log_attention(self, layer_idx: int, attn_info: Dict):
        """Log attention computation."""
        self.log("attention", {
            "layer": layer_idx,
            **attn_info
        })
    
    def print_summary(self):
        """Print readable summary."""
        print(f"\n{'='*70}")
        print(f"📊 {self.name} - GENERATION LOG")
        print(f"{'='*70}")
        
        # Group by event type
        token_events = [e for e in self.logs if e['event'] == 'token_generated']
        cache_events = [e for e in self.logs if e['event'] == 'cache_state']
        
        print(f"\n🔢 Token Generation ({len(token_events)} tokens):")
        for e in token_events[:10]:  # First 10 tokens
            data = e['data']
            print(f"  Step {e['step']}: '{data['token_text']}' (ID: {data['token_id']})")
            top5 = data['top5_logits']
            print(f"    Top-5: {top5}")
        
        if len(token_events) > 10:
            print(f"  ... ({len(token_events) - 10} more tokens)")
        
        # Final output
        full_output = "".join([e['data']['token_text'] for e in token_events])
        print(f"\n📝 Full Output: {full_output}")
        print(f"🎯 Needle 'ALPHA-123': {'✅ Found' if 'ALPHA-123' in full_output else '❌ Not found'}")
        
        return full_output

def hook_model_for_logging(model, logger: GenerationLogger):
    """Add hooks to model to log everything."""
    
    # Hook each attention layer
    for layer_idx, layer in enumerate(model.model.layers):
        # Only hook a few representative layers
        if layer_idx not in [0, 15, 31]:
            continue
        
        original_forward = layer.self_attn.forward
        
        def make_logged_forward(orig_fn, l_idx):
            def logged_forward(self, *args, **kwargs):
                # Log input
                hidden_states = args[0] if args else kwargs.get('hidden_states')
                if hidden_states is not None:
                    logger.log("layer_input", {
                        "layer": l_idx,
                        "shape": list(hidden_states.shape),
                        "mean": float(hidden_states.mean()),
                        "std": float(hidden_states.std()),
                    })
                
                # Call original
                output = orig_fn(*args, **kwargs)
                
                # Log output
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output
                
                logger.log("layer_output", {
                    "layer": l_idx,
                    "shape": list(attn_output.shape),
                    "mean": float(attn_output.mean()),
                    "std": float(attn_output.std()),
                })
                
                return output
            
            return logged_forward
        
        layer.self_attn.forward = types.MethodType(
            make_logged_forward(original_forward, layer_idx),
            layer.self_attn
        )

# =============================================================================
# TEST SCENARIOS
# =============================================================================

def test_pure_baseline():
    """Test 1: Pure baseline with NO unsloth_spectral import at all."""
    print("\n" + "="*70)
    print("TEST 1: PURE BASELINE (No spectral imports)")
    print("="*70)
    print("This is the GROUND TRUTH - pure Unsloth behavior")
    
    logger = GenerationLogger("Pure Baseline")
    
    # Set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Instrument model
    hook_model_for_logging(model, logger)
    
    # Prepare input
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]
    print(f"Prompt: {TEST_PROMPT}")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Generate token by token
    print("\n🔄 Generating...")
    input_ids = inputs.input_ids
    
    for step in range(20):  # Generate 20 tokens
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, -1:] if step > 0 else input_ids,
                use_cache=True,
                return_dict=True,
            )
        
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Get top-5 logits
        top5_values, top5_indices = torch.topk(logits[0], 5)
        top5_tokens = [(tokenizer.decode([idx.item()]), val.item()) 
                       for idx, val in zip(top5_indices, top5_values)]
        
        # Log token
        token_text = tokenizer.decode(next_token[0])
        logger.log_token(next_token[0].item(), token_text, top5_tokens)
        
        # Continue
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop at EOS
        if next_token[0].item() == tokenizer.eos_token_id:
            break
    
    # Print summary
    output = logger.print_summary()
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output, logger.logs

def test_contaminated_baseline():
    """Test 2: Baseline AFTER importing unsloth_spectral (but not patching)."""
    print("\n" + "="*70)
    print("TEST 2: CONTAMINATED BASELINE (Import but no patch)")
    print("="*70)
    print("Import unsloth_spectral, but don't call patch_unsloth_attention()")
    
    # NOW import unsloth_spectral for the first time
    os.environ["UNSLOTH_SPECTRAL_QUIET"] = "1"
    from unsloth_spectral import patch_unsloth_attention
    
    print("✅ Imported unsloth_spectral (but NOT patching)")
    
    # Check if anything changed in unsloth.models.llama
    import unsloth.models.llama as llama_module
    print(f"\n🔍 Checking unsloth.models.llama state:")
    print(f"  LlamaAttention_fast_forward_inference: {llama_module.LlamaAttention_fast_forward_inference}")
    print(f"  LlamaModel_fast_forward_inference: {llama_module.LlamaModel_fast_forward_inference}")
    
    logger = GenerationLogger("Contaminated Baseline")
    
    # Set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Instrument model
    hook_model_for_logging(model, logger)
    
    # Prepare input
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]
    print(f"Prompt: {TEST_PROMPT}")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Generate token by token
    print("\n🔄 Generating...")
    input_ids = inputs.input_ids
    
    for step in range(20):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, -1:] if step > 0 else input_ids,
                use_cache=True,
                return_dict=True,
            )
        
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Get top-5 logits
        top5_values, top5_indices = torch.topk(logits[0], 5)
        top5_tokens = [(tokenizer.decode([idx.item()]), val.item()) 
                       for idx, val in zip(top5_indices, top5_values)]
        
        # Log token
        token_text = tokenizer.decode(next_token[0])
        logger.log_token(next_token[0].item(), token_text, top5_tokens)
        
        # Continue
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token[0].item() == tokenizer.eos_token_id:
            break
    
    # Print summary
    output = logger.print_summary()
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output, logger.logs

def test_spectral_patched():
    """Test 3: Spectral with patches applied."""
    print("\n" + "="*70)
    print("TEST 3: SPECTRAL PATCHED")
    print("="*70)
    
    # Import already done in test 2
    from unsloth_spectral import patch_unsloth_attention
    
    logger = GenerationLogger("Spectral Patched")
    
    # Set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # PATCH
    print("🔧 Patching with spectral cache...")
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        verbose=True,
    )
    
    # Instrument model
    hook_model_for_logging(model, logger)
    
    # Prepare input
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]
    print(f"Prompt: {TEST_PROMPT}")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Generate token by token
    print("\n🔄 Generating...")
    input_ids = inputs.input_ids
    
    for step in range(20):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[:, -1:] if step > 0 else input_ids,
                use_cache=True,
                return_dict=True,
            )
        
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Get top-5 logits
        top5_values, top5_indices = torch.topk(logits[0], 5)
        top5_tokens = [(tokenizer.decode([idx.item()]), val.item()) 
                       for idx, val in zip(top5_indices, top5_values)]
        
        # Log token
        token_text = tokenizer.decode(next_token[0])
        logger.log_token(next_token[0].item(), token_text, top5_tokens)
        
        # Continue
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token[0].item() == tokenizer.eos_token_id:
            break
    
    # Print summary
    output = logger.print_summary()
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output, logger.logs

# =============================================================================
# COMPARISON & ANALYSIS
# =============================================================================

def compare_logs(log1: List[Dict], log2: List[Dict], name1: str, name2: str):
    """Compare two generation logs to find where they diverge."""
    print(f"\n{'='*70}")
    print(f"🔍 COMPARING: {name1} vs {name2}")
    print(f"{'='*70}")
    
    # Extract token sequences
    tokens1 = [e['data'] for e in log1 if e['event'] == 'token_generated']
    tokens2 = [e['data'] for e in log2 if e['event'] == 'token_generated']
    
    print(f"\n📊 Token-by-Token Comparison:")
    print(f"{'Step':<6} {name1:<30} {name2:<30} {'Match':<10}")
    print("-" * 80)
    
    max_len = max(len(tokens1), len(tokens2))
    divergence_step = None
    
    for i in range(max_len):
        tok1 = tokens1[i] if i < len(tokens1) else None
        tok2 = tokens2[i] if i < len(tokens2) else None
        
        if tok1 and tok2:
            text1 = tok1['token_text']
            text2 = tok2['token_text']
            match = "✅" if text1 == text2 else "❌"
            
            if text1 != text2 and divergence_step is None:
                divergence_step = i
                print(f"{i:<6} {text1:<30} {text2:<30} {match} ← DIVERGENCE!")
            elif i < 10 or (divergence_step and i < divergence_step + 5):
                print(f"{i:<6} {text1:<30} {text2:<30} {match}")
        elif tok1:
            print(f"{i:<6} {tok1['token_text']:<30} {'(none)':<30} ❌")
        elif tok2:
            print(f"{i:<6} {'(none)':<30} {tok2['token_text']:<30} ❌")
    
    if divergence_step is not None:
        print(f"\n🚨 DIVERGENCE AT STEP {divergence_step}")
        
        # Show logits at divergence point
        if divergence_step < len(tokens1) and divergence_step < len(tokens2):
            print(f"\n📊 Logits at divergence ({name1}):")
            for tok, logit in tokens1[divergence_step]['top5_logits'][:5]:
                print(f"  {tok}: {logit:.4f}")
            
            print(f"\n📊 Logits at divergence ({name2}):")
            for tok, logit in tokens2[divergence_step]['top5_logits'][:5]:
                print(f"  {tok}: {logit:.4f}")
    else:
        print(f"\n✅ No divergence - outputs are IDENTICAL")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🔬 GROUND TRUTH LOGGER - GRANULAR INSTRUMENTATION")
    print("="*70)
    print("\nThis test will show EXACTLY where behavior diverges.")
    print("We'll run 3 scenarios and compare token-by-token.")
    
    # Test 1: Pure baseline
    output1, logs1 = test_pure_baseline()
    
    # Test 2: Contaminated baseline (after import)
    output2, logs2 = test_contaminated_baseline()
    
    # Test 3: Spectral patched
    output3, logs3 = test_spectral_patched()
    
    # Comparisons
    print("\n" + "="*70)
    print("📋 FINAL ANALYSIS")
    print("="*70)
    
    # Compare 1 vs 2 (does import contaminate?)
    compare_logs(logs1, logs2, "Pure Baseline", "Contaminated Baseline")
    
    # Compare 2 vs 3 (does patching improve?)
    compare_logs(logs2, logs3, "Contaminated Baseline", "Spectral Patched")
    
    # Compare 1 vs 3 (overall difference)
    compare_logs(logs1, logs3, "Pure Baseline", "Spectral Patched")
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 NEEDLE RECALL SUMMARY")
    print(f"{'='*70}")
    print(f"Pure Baseline:         {'✅ Found' if 'ALPHA-123' in output1 else '❌ Not found'}")
    print(f"Contaminated Baseline: {'✅ Found' if 'ALPHA-123' in output2 else '❌ Not found'}")
    print(f"Spectral Patched:      {'✅ Found' if 'ALPHA-123' in output3 else '❌ Not found'}")

if __name__ == "__main__":
    main()
