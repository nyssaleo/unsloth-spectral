"""
Test if SpectralCache is being passed correctly to Unsloth's decode path.

This will help us understand HOW Unsloth accesses the cache during decode.
"""

import torch
from unsloth import FastLanguageModel
from unsloth_spectral import patch_unsloth_attention

def main():
    print("="*70)
    print("  CACHE COMPATIBILITY TEST")
    print("="*70)
    
    # Load model
    print("\n📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Patch with spectral cache
    print("\n🔧 Patching with SpectralCache...")
    patch_unsloth_attention(
        model,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        use_spectral_attention=True,
        verbose=True,
        debug_logging=True,  # Enable detailed logging
    )
    
    # Test 1: Short generation WITH cache
    print("\n" + "="*70)
    print("TEST 1: Generation WITH use_cache=True (default)")
    print("="*70)
    
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"\n📝 Generating with use_cache=True...")
    outputs1 = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,  # This is what triggers the fast path
    )
    text1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
    print(f"✅ Generated: '{text1}'")
    
    # Test 2: Short generation WITHOUT cache
    print("\n" + "="*70)
    print("TEST 2: Generation WITH use_cache=False")
    print("="*70)
    
    print(f"\n📝 Generating with use_cache=False...")
    outputs2 = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        use_cache=False,  # This should force Python forward() every time
    )
    text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    print(f"✅ Generated: '{text2}'")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\n📊 Results:")
    print(f"  WITH cache:    '{text1}'")
    print(f"  WITHOUT cache: '{text2}'")
    print(f"  Match: {text1 == text2}")
    
    print("\n💡 INTERPRETATION:")
    if text1 == text2:
        print("  ✅ Both paths produce same output")
        print("  → SpectralCache is working correctly")
    else:
        print("  ❌ Different outputs!")
        print("  → use_cache=True bypasses our SpectralCache")
        print("  → use_cache=False goes through our forward()")
        print("\n  🔧 SOLUTION: We need to disable Unsloth's fast decode path")
        print("     or make SpectralCache compatible with it.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

