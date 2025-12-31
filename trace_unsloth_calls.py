"""
Trace Unsloth's function calls during generation to find the decode path.

This script instruments the model to log every method call, helping us
identify which function Unsloth uses for decode steps (q_len=1).
"""

import torch
import torch.nn
from unsloth import FastLanguageModel
import sys

# Monkey-patch to trace all calls
original_forward_calls = []

def trace_wrapper(original_fn, name):
    """Wrap a function to log when it's called."""
    def wrapper(*args, **kwargs):
        # Log the call
        if len(args) > 1 and isinstance(args[1], torch.Tensor):
            q_len = args[1].shape[1] if len(args[1].shape) >= 2 else "?"
            original_forward_calls.append((name, q_len))
            print(f"🔍 TRACE: {name} called with q_len={q_len}")
        else:
            original_forward_calls.append((name, "unknown"))
            print(f"🔍 TRACE: {name} called")
        
        # Call original
        return original_fn(*args, **kwargs)
    return wrapper

def instrument_model(model):
    """Patch all attention methods to trace calls."""
    print("="*70)
    print("INSTRUMENTING MODEL FOR TRACING")
    print("="*70)
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # Find all callable methods (but skip nn.Module submodules)
            for attr_name in dir(attn):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(attn, attr_name)
                
                # Skip if it's an nn.Module (like q_proj, k_proj, etc.)
                if isinstance(attr, torch.nn.Module):
                    continue
                
                if callable(attr) and not isinstance(attr, type):
                    # Only wrap if it's a bound method, not a module
                    try:
                        wrapped = trace_wrapper(attr, f"Layer{layer_idx}.{attr_name}")
                        setattr(attn, attr_name, wrapped)
                    except TypeError:
                        # Skip if we can't wrap it
                        pass
    
    print(f"✅ Instrumented all attention layers")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("  UNSLOTH DECODE PATH TRACER")
    print("="*70)
    
    # Load model
    print("\n📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Instrument
    instrument_model(model)
    
    # Test generation
    prompt = "Hello, how are"  # Short prompt (5 tokens)
    max_new_tokens = 5  # Generate just 5 tokens
    
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"🎯 Generating {max_new_tokens} new tokens...")
    print("\n" + "="*70)
    print("GENERATION START - WATCH FOR PATTERNS")
    print("="*70 + "\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,  # CRITICAL: This is what triggers the fast path
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    print(f"\n📤 Generated: '{text}'")
    
    # Analyze calls
    print("\n" + "="*70)
    print("CALL ANALYSIS")
    print("="*70)
    
    # Group by function name
    from collections import Counter
    call_counts = Counter([name for name, q_len in original_forward_calls])
    
    print("\n📊 Function Call Counts:")
    for name, count in call_counts.most_common(10):
        print(f"  {name}: {count} calls")
    
    # Look for patterns
    print("\n🔍 Call Sequence Pattern:")
    print("  (showing first 10 calls)")
    for i, (name, q_len) in enumerate(original_forward_calls[:10]):
        print(f"  {i+1}. {name} (q_len={q_len})")
    
    print("\n💡 INTERPRETATION:")
    print("  - If you see the SAME function called repeatedly during decode,")
    print("    that's likely the decode-specific path we need to patch!")
    print("  - Look for functions called with q_len=1 (decode steps)")
    print("="*70)

if __name__ == "__main__":
    main()

