"""
Inspect what methods actually exist on Unsloth's attention layers.

This will show us all the functions Unsloth has that we could potentially patch.
"""

from unsloth import FastLanguageModel
import inspect

def main():
    print("="*70)
    print("  UNSLOTH ATTENTION LAYER METHOD INSPECTOR")
    print("="*70)
    
    # Load model
    print("\n📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Inspect first attention layer
    layer_0_attn = model.model.layers[0].self_attn
    
    print("\n" + "="*70)
    print("LAYER 0 ATTENTION METHODS")
    print("="*70)
    
    print(f"\n📦 Class: {type(layer_0_attn).__name__}")
    print(f"📦 Module: {type(layer_0_attn).__module__}")
    
    # Get all methods
    all_methods = []
    for attr_name in dir(layer_0_attn):
        attr = getattr(layer_0_attn, attr_name)
        if callable(attr) and not isinstance(attr, type):
            # Check if it's a method (not a built-in)
            if not attr_name.startswith('__'):
                all_methods.append(attr_name)
    
    print(f"\n🔧 Found {len(all_methods)} callable methods:")
    for method_name in sorted(all_methods):
        method = getattr(layer_0_attn, method_name)
        
        # Try to get signature
        try:
            sig = inspect.signature(method)
            params = str(sig)
            # Truncate if too long
            if len(params) > 80:
                params = params[:77] + "..."
        except:
            params = "(...)"
        
        # Check if it's defined in Unsloth
        try:
            module = inspect.getmodule(method)
            if module and 'unsloth' in module.__name__:
                marker = "🎯 UNSLOTH"
            else:
                marker = "   (inherited)"
        except:
            marker = "   (?)"
        
        print(f"  {marker} {method_name}{params}")
    
    # Look for specific patterns
    print("\n" + "="*70)
    print("SUSPICIOUS METHODS (likely decode paths)")
    print("="*70)
    
    suspects = [
        'forward',
        'forward_inference', 
        'fast_forward',
        'fast_forward_inference',
        '_forward_impl',
        'decode',
        'decode_step',
        'generate_step',
    ]
    
    print("\n🔍 Looking for these patterns:")
    for pattern in suspects:
        matching = [m for m in all_methods if pattern.lower() in m.lower()]
        if matching:
            print(f"  ✅ {pattern}: {', '.join(matching)}")
        else:
            print(f"  ❌ {pattern}: not found")
    
    # Check what forward actually is
    print("\n" + "="*70)
    print("FORWARD METHOD DETAILS")
    print("="*70)
    
    forward_fn = layer_0_attn.forward
    print(f"\n📌 forward is: {forward_fn}")
    print(f"📌 Type: {type(forward_fn)}")
    
    try:
        print(f"📌 Defined in: {inspect.getfile(forward_fn)}")
        print(f"📌 Line: {inspect.getsourcelines(forward_fn)[1]}")
    except:
        print(f"📌 Source: (C extension or wrapped)")
    
    # Check for __call__
    if hasattr(layer_0_attn, '__call__'):
        print(f"\n📌 __call__ exists: {layer_0_attn.__call__}")
        print(f"📌 Same as forward? {layer_0_attn.__call__ == forward_fn}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    print("""
    Based on the methods found above, we should patch:
    
    1. PRIMARY: The 'forward' method (what we're doing now)
       - This handles prefill (q_len > 1)
    
    2. SECONDARY: Any 'fast_forward_inference' or similar
       - This likely handles decode (q_len = 1)
       - Need to find and patch this too!
    
    Next step: Run trace_unsloth_calls.py to see which function
    is ACTUALLY called during generation.
    """)
    
    print("="*70)

if __name__ == "__main__":
    main()

