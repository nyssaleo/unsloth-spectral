# 🔴 **CRITICAL BUG: Unsloth Decode Path Not Patched**

## 📊 **EVIDENCE FROM LOGS (Dec 31, 2025)**

### ✅ **What Works (Prefill):**
```
[SpectralForward] NEW FORWARD PASS - Layer 0
  q_len: 17 (1=decode, >1=prefill)
  past_key_value type: <class 'NoneType'>
  ❌ Cache is None (will create fresh cache)
  
[SpectralCache.__init__] Created cache:
  num_heads=8 (KV heads)
  ...
  
[SpectralForward] Appending to cache (BEFORE repeat_kv):
  K shape: torch.Size([1, 8, 17, 128])
  ...
  
[SpectralForward] Forward pass complete:
  🔍 Returning cache type: <class 'unsloth_spectral.spectral_cache.SpectralCache'>
  🔍 use_cache flag: True
```

**This happens for ALL 32 layers during the initial 17-token prompt processing.**

---

### ❌ **What's Broken (Decode):**

**After the prefill, the model generated 50 tokens (18-67) at 18.6 tok/s:**
- ✅ Generation happened (text was produced)
- ❌ ZERO logs from `[SpectralForward]`
- ❌ NO new forward pass markers
- ❌ NO `q_len=1` logged anywhere
- ❌ NO cache append calls
- ❌ Cache stayed frozen at 17 tokens

**This is IMPOSSIBLE if our `spectral_forward` was being called.**

---

## 🧠 **ROOT CAUSE ANALYSIS**

### **The Problem:**
```python
# In integration.py line 414:
layer.self_attn.forward = spectral_fwd.__get__(layer.self_attn, type(layer.self_attn))
```

We're monkey-patching the **standard HuggingFace `forward()` method**.

### **Why This Fails:**

Unsloth uses **TWO SEPARATE CODE PATHS**:

1. **Prefill Path** (q_len > 1):
   - Uses: `MistralAttention.forward()`
   - ✅ **We patched this**
   - ✅ Called for initial multi-token processing

2. **Decode Path** (q_len = 1):
   - Uses: `???` (Some optimized inference function)
   - ❌ **We did NOT patch this**
   - ❌ Bypasses our `forward()` entirely

During `model.generate()`, Unsloth internally switches from path #1 to path #2 after the prefill is complete.

---

## 🔍 **HYPOTHESES ON THE DECODE PATH**

Based on Unsloth's architecture, the decode function is likely one of these:

### **Hypothesis 1: Separate Inference Forward**
```python
# Unsloth likely has something like:
class MistralAttention:
    def forward(self, ...):          # ← Prefill (what we patched)
        ...
    
    def forward_inference(self, ...):  # ← Decode (NOT patched!)
        # Optimized single-token path
        ...
```

### **Hypothesis 2: Kernel-Level Bypass**
```python
# During generation, Unsloth might directly call:
unsloth.kernels.fast_decode(...)  # ← Completely bypasses attention.forward()
```

### **Hypothesis 3: Dynamic Method Swapping**
```python
# Unsloth might swap the forward method at runtime:
if use_cache and q_len == 1:
    layer.self_attn.forward = fast_inference_forward  # ← Overwrites our patch!
else:
    layer.self_attn.forward = standard_forward
```

---

## 🛠️ **INVESTIGATION PLAN**

### **Step 1: Identify All Attention Methods**

Run: `inspect_unsloth_methods.py`

This will list ALL methods on `layer.self_attn`, including:
- `forward`
- `forward_inference`? 
- `fast_forward`?
- `fast_forward_inference`?
- `_forward_impl`?
- Any Unsloth-specific methods

### **Step 2: Trace Actual Execution**

Run: `trace_unsloth_calls.py`

This instruments EVERY method to log when it's called during `model.generate()`.

Expected output:
```
🔍 TRACE: Layer0.forward called with q_len=5          ← Prefill
🔍 TRACE: Layer1.forward called with q_len=5
...
🔍 TRACE: Layer31.forward called with q_len=5
🔍 TRACE: Layer0.fast_forward_inference called with q_len=1  ← Decode!
🔍 TRACE: Layer1.fast_forward_inference called with q_len=1
...
```

### **Step 3: Patch the Correct Function**

Once we identify the decode function (let's call it `X`), we need to:

```python
def patch_unsloth_attention(model, ...):
    for layer in model.model.layers:
        # Patch prefill path (already done)
        layer.self_attn.forward = spectral_fwd_prefill
        
        # Patch decode path (NEW!)
        layer.self_attn.X = spectral_fwd_decode  # ← X = discovered function name
```

### **Step 4: Specialized Decode Forward**

The decode path needs special handling:

```python
def spectral_fwd_decode(self, hidden_states, past_key_value, ...):
    """
    Optimized for single-token decode (q_len=1).
    
    Key differences from prefill:
    - Always appends exactly 1 token
    - Can use spectral attention (no reconstruction)
    - Expects past_key_value to be SpectralCache
    """
    bsz, q_len, _ = hidden_states.shape
    assert q_len == 1, "Decode path expects single token!"
    
    # ... QKV projection ...
    # ... RoPE ...
    
    # Append to cache
    if not isinstance(past_key_value, SpectralCache):
        raise TypeError("Decode expects SpectralCache from prefill!")
    
    past_key_value.append(key_states, value_states)
    
    # Use spectral attention (no reconstruction needed!)
    if past_key_value.total_tokens > block_size:
        attn_output = spectral_attention_forward(query_states, past_key_value, ...)
    else:
        # Standard attention for short sequences
        K_full, V_full = past_key_value.get_kv()
        K_full = repeat_kv(K_full, num_key_value_groups)
        V_full = repeat_kv(V_full, num_key_value_groups)
        attn_output = torch.matmul(query_states, K_full.transpose(2, 3)) / sqrt(head_dim)
        # ... softmax, matmul with V ...
    
    return attn_output, None, past_key_value
```

---

## 🎯 **EXPECTED OUTCOME**

After patching the decode path correctly, we should see:

```
=== PREFILL ===
[SpectralForward] NEW FORWARD PASS - Layer 0
  q_len: 17
  ...

=== DECODE STEP 1 ===
[SpectralForwardDecode] NEW DECODE PASS - Layer 0
  q_len: 1
  past_key_value type: <class 'SpectralCache'>
  ✓ Cache IS SpectralCache, total_tokens=17
  
[SpectralCache.append] Incoming K/V:
  Before append: total_tokens=17
  After append: total_tokens=18  ← GROWING!
  
=== DECODE STEP 2 ===
[SpectralForwardDecode] NEW DECODE PASS - Layer 0
  q_len: 1
  ✓ Cache IS SpectralCache, total_tokens=18
  
[SpectralCache.append] Incoming K/V:
  Before append: total_tokens=18
  After append: total_tokens=19  ← GROWING!

... and so on for all 50 tokens ...
```

---

## 📋 **ACTION ITEMS**

1. ✅ Push investigation tools to GitHub
2. ⏳ Run `inspect_unsloth_methods.py` in Colab
3. ⏳ Run `trace_unsloth_calls.py` in Colab
4. ⏳ Identify the decode function name
5. ⏳ Patch the decode function
6. ⏳ Re-run benchmark
7. ⏳ Verify cache grows: 17 → 18 → 19 → ... → 67

---

## 💡 **WHY THIS EXPLAINS EVERYTHING**

1. **Low Vocabulary Overlap (51.9%):**
   - Baseline: Uses proper KV cache (all 67 tokens)
   - Spectral: Stuck with only 17 tokens (prompt only)
   - Model "forgets" everything after initial prompt!

2. **Speedup Still Observed (1.49x):**
   - Even with broken cache, there's a speedup
   - Likely because Unsloth's decode path is naturally faster
   - The 17-token cache is smaller → less memory traffic

3. **No Compression Happening:**
   - Never reaches 512 tokens
   - Never creates cold blocks
   - All tokens stay in hot buffer

---

## 🚀 **NEXT STEPS: RUN IN COLAB**

```bash
# Pull latest code
!cd /content/unsloth-spectral && git pull origin main

# Inspect methods
!cd /content/unsloth-spectral && python inspect_unsloth_methods.py

# Trace execution
!cd /content/unsloth-spectral && python trace_unsloth_calls.py
```

The output from these scripts will tell us **exactly** which function to patch for decode! 🎯

