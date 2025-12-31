"""
Trace ONLY the forward() method to see if it's called during decode.

This is a simplified version that focuses on the key question:
Is our patched forward() being called during generation?
"""

import torch
from unsloth import FastLanguageModel

# Track all forward calls
forward_calls = []

def trace_forward_wrapper(original_fn, layer_idx):
    """Wrap forward() to log when it's called."""
    def wrapper(self, hidden_states, *args, **kwargs):
        # Log the call
        q_len = hidden_states.shape[1] if len(hidden_states.shape) >= 2 else "?"
        forward_calls.append((layer_idx, q_len))
        print(f"🔍 Layer {layer_idx}: forward() called with q_len={q_len}")
        
        # Call original
        return original_fn(hidden_states, *args, **kwargs)
    return wrapper

def instrument_model(model):
    """Patch only the forward() method on all layers."""
    print("="*70)
    print("INSTRUMENTING MODEL - FORWARD() ONLY")
    print("="*70)
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # Save original
            original_forward = attn.forward
            
            # Wrap it
            wrapped = trace_forward_wrapper(original_forward, layer_idx)
            
            # Bind to instance
            attn.forward = wrapped.__get__(attn, type(attn))
    
    print(f"✅ Instrumented forward() on 32 layers")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("  FORWARD() CALL TRACER")
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
    prompt = "Hello, how are"  # Short prompt
    max_new_tokens = 5  # Generate just 5 tokens
    
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"🎯 Generating {max_new_tokens} new tokens...")
    print("\n" + "="*70)
    print("GENERATION START")
    print("="*70 + "\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs['input_ids'].shape[1]
    print(f"📏 Prompt length: {prompt_length} tokens\n")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    print(f"\n📤 Generated: '{text}'")
    
    # Analyze calls
    print("\n" + "="*70)
    print("ANALYSIS: WAS forward() CALLED?")
    print("="*70)
    
    if len(forward_calls) == 0:
        print("\n❌ CRITICAL: forward() was NEVER called!")
        print("   This means Unsloth is bypassing forward() entirely.")
        print("   We need to patch a DIFFERENT entry point!")
    else:
        # Count by q_len
        prefill_calls = [call for call in forward_calls if call[1] == prompt_length]
        decode_calls = [call for call in forward_calls if call[1] == 1]
        other_calls = [call for call in forward_calls if call[1] not in [prompt_length, 1]]
        
        print(f"\n✅ forward() WAS called: {len(forward_calls)} times total")
        print(f"   📊 Prefill (q_len={prompt_length}): {len(prefill_calls)} calls")
        print(f"   📊 Decode (q_len=1): {len(decode_calls)} calls")
        if other_calls:
            print(f"   📊 Other q_len: {len(other_calls)} calls")
        
        if len(decode_calls) == 0:
            print("\n⚠️  WARNING: forward() called for prefill, but NOT for decode!")
            print("   Expected: 32 layers × 5 tokens = 160 decode calls")
            print("   Actual: 0 decode calls")
            print("\n   This confirms: Unsloth switches to a different path for decode!")
        else:
            print(f"\n✅ forward() called for BOTH prefill and decode")
            print(f"   Expected decode calls: {32 * max_new_tokens} (32 layers × {max_new_tokens} tokens)")
            print(f"   Actual decode calls: {len(decode_calls)}")
            
            if len(decode_calls) < 32 * max_new_tokens:
                print("\n⚠️  Fewer decode calls than expected!")
    
    print("\n" + "="*70)
    print("CALL SEQUENCE (first 20):")
    print("="*70)
    for i, (layer_idx, q_len) in enumerate(forward_calls[:20]):
        print(f"  {i+1}. Layer {layer_idx}: q_len={q_len}")
    
    if len(forward_calls) > 20:
        print(f"  ... ({len(forward_calls) - 20} more calls)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

