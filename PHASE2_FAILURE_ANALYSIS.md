# Phase 2 Failure Analysis

## 🔴 Critical Failure: Gibberish Outputs

### Observed Results

**Memory:** 1.35x compression (⚠️ below target)  
**Speed:** 0.67x of baseline (❌ 33% slower)  
**Quality:** 19-35% token overlap (❌ CATASTROPHIC)

### The Smoking Gun 🔍

The spectral outputs contain **"The quick brown fox jumps over the lazy dog"** - this is the EXACT text from the `test_long_context()` function!

```python
# From test script line 48:
base_text = "The quick brown fox jumps over the lazy dog. " * 100
```

**Spectral Output Examples:**
- Prompt 1: "terms and conditions" instead of relativity explanation
- Prompt 2: "The quick brown fox..." **repeated** instead of Python code
- Prompt 3: "The lazy dog riddle" instead of climate change

**This means:** The model is attending to completely wrong tokens - specifically, tokens from a previous unrelated generation!

---

## 🐛 Root Cause Analysis

### Hypothesis 1: Cache Contamination (MOST LIKELY) ⭐

**The Problem:**  
The SpectralCache is persisting across different `generate()` calls when it should be fresh for each generation.

**Evidence:**
- Outputs contain text from **previous** test (long context baseline)
- This can only happen if the cache has tokens from that previous generation
- Position tracking increments: Gen1 (0-50), Gen2 (50-100), Gen3 (100-150)
- But each generation should start at position 0!

**The Bug:**  
In `spectral_cache.py` lines 119-136:

```python
if position_ids is None:
    position_ids = torch.arange(
        self.current_position,  # ← PERSISTS across generations!
        self.current_position + T_new,
        ...
    )
self.current_position += T_new  # ← Keeps incrementing!
```

If the cache object persists (shouldn't happen but apparently does), then:
- Generation 1: positions 0-62 ✅
- Generation 2: positions 62-122 ❌ (should be 0-60)
- Generation 3: positions 122-182 ❌ (should be 0-60)

With wrong positions:
1. RoPE relative rotation uses wrong offsets
2. Queries attend to wrong keys
3. Model retrieves tokens from previous generations
4. Outputs are gibberish

---

### Hypothesis 2: HuggingFace Cache Reuse

**The Problem:**  
`model.generate()` might be reusing `past_key_values` across calls instead of starting fresh.

**Standard Behavior:**
- Each `generate()` call should start with `past_key_values=None`
- Cache is built during generation, then discarded
- Next generation starts fresh

**Possible Bug:**
- Maybe Unsloth/HF is caching at model level?
- Maybe our patch is storing cache reference somewhere persistent?

---

### Hypothesis 3: Position IDs from HuggingFace are Wrong

**The Problem:**  
HuggingFace auto-generates position_ids, but maybe they're continuing from previous generation?

**Less Likely Because:**
- HF typically resets position_ids for each generation
- This would be a fundamental HF bug (unlikely)

---

## 🔬 Diagnostic Strategy

### Step 1: Add Debug Logging

Modify test to enable debug logging:

```python
patch_unsloth_attention(
    model,
    debug_logging=True,  # ← Enable
    ...
)
```

**Look for:**
- "Creating new SpectralCache" message for EACH generation
- Position values in each forward pass
- Whether cache total_tokens resets between generations

### Step 2: Check Cache Lifecycle

Add diagnostics in `spectral_cache.py`:

```python
def __init__(self, ...):
    self.cache_id = id(self)  # Unique ID
    print(f"[SpectralCache] Created new cache {self.cache_id}")
    self.current_position = 0
    
def append(self, ...):
    print(f"[SpectralCache {self.cache_id}] Appending at position {self.current_position}")
    # ... rest of code
```

**Expected:**
- Different cache_id for each generation
- current_position=0 at start of each generation

**If Bug:**
- Same cache_id across generations
- current_position keeps incrementing

### Step 3: Verify Position IDs

Log what position_ids are being passed:

```python
def append(self, K_new, V_new, position_ids):
    if position_ids is not None:
        print(f"[DEBUG] Received position_ids: {position_ids.flatten()[:10]}")
    else:
        print(f"[DEBUG] No position_ids, auto-generating from {self.current_position}")
```

---

## 🛠️ Proposed Fixes

### Fix 1: Force Cache Reset (Quick Fix)

In `spectral_cache.py`, add a reset method:

```python
def reset(self):
    """Reset cache for new generation."""
    self.hot_K = None
    self.hot_V = None
    self.hot_position_ids = None
    self.cold_blocks = []
    self.current_position = 0  # ← Reset!
    self.total_tokens = 0
```

Call this at the start of each generation in `integration.py`:

```python
if not isinstance(past_key_value, SpectralCache):
    cache = SpectralCache(...)
else:
    # Reusing cache from previous step in SAME generation
    cache = past_key_value
    
    # Check if this is a NEW generation (position_ids restart at 0)
    if position_ids is not None and position_ids.min().item() == 0:
        cache.reset()  # ← New generation detected!
```

### Fix 2: Never Reuse Cache (Safer)

Always create fresh cache, even if one exists:

```python
# In integration.py, ALWAYS create new cache:
cache = SpectralCache(
    num_heads=num_key_value_heads,
    ...
)

# Don't check if past_key_value is SpectralCache
# Always start fresh!
```

**Downside:** Loses cache from previous tokens in same generation  
**Upside:** Guaranteed fresh state each forward pass

### Fix 3: Detect Generation Boundary (Robust)

Use position_ids to detect when a new generation starts:

```python
def append(self, K_new, V_new, position_ids):
    # Detect if this is a new generation
    if position_ids is not None:
        first_pos = position_ids[0, 0].item()
        if first_pos < self.current_position:
            # Position went backwards = new generation!
            self.reset()
    
    # ... rest of append logic
```

---

## 🎯 Immediate Action Plan

1. **Enable debug logging** in Phase 2 test
2. **Run test again** and examine logs for:
   - Cache creation messages
   - Position values
   - Cache total_tokens between generations
3. **Implement Fix 1** (cache reset on new generation)
4. **Re-run test** and verify outputs are coherent

---

## 📊 Expected Results After Fix

**Quality:**
- ✅ Coherent outputs matching prompts
- ✅ Token overlap >70%
- ✅ No cross-contamination between generations

**Memory:**
- Should remain ~1.35x (may improve with correct attention)
- Memory compression depends on sequence length (longer = better)

**Speed:**
- May improve slightly with correct attention
- 0.67x might be inherent overhead (projection cost)
- Should improve at longer contexts (>4K tokens)

---

## 🔍 Why This Wasn't Caught in Phase 1

**Phase 1 Tests:**
- Used single forward passes, not full generation
- Created cache manually (no HF generate() involvement)
- No multi-generation scenarios
- All tests passed because single-generation logic worked

**Phase 2 Exposed:**
- Multi-generation workflow
- HuggingFace generate() integration
- Cache lifecycle across generations
- Real model interaction

---

## 📝 Lessons Learned

1. **Integration testing is critical** - Unit tests passed but integration failed
2. **Cache lifecycle needs explicit management** - Can't rely on garbage collection
3. **Position tracking must be bulletproof** - Off-by-one errors = gibberish
4. **Debug logging should be default** - Would have caught this immediately

---

## 🚀 Next Steps

1. **Implement Fix 1** (cache reset detection)
2. **Add comprehensive logging** to cache operations
3. **Re-run Phase 2 test** in Colab
4. **If still fails:** Try Fix 2 (always fresh cache)
5. **Once passing:** Measure proper perplexity (not just token overlap)

---

## 🎓 Technical Deep Dive

### Why Wrong Positions Cause Gibberish

**The RoPE Relative Rotation Logic:**

```python
relative_positions = key_positions - query_position
cos_relative = cos[relative_positions]
sin_relative = sin[relative_positions]
Q_aligned = apply_rope(Q_broadcast, cos_relative, -sin_relative)
```

**With Correct Positions:**
- Query at position 5, keys at [0,1,2,3,4]
- Relative offsets: [-5,-4,-3,-2,-1]
- RoPE correctly aligns query to each key
- Attention retrieves relevant context

**With Wrong Positions (cache contaminated):**
- Query at position 5, keys at [50,51,52,...] (from previous gen!)
- Relative offsets: [45,46,47,...]
- RoPE applies HUGE rotations
- Attention retrieves random tokens (or cached "quick brown fox" text)
- Output is gibberish

---

## ✅ Confidence Level

**High confidence (90%)** that this is a cache persistence issue.

**Evidence:**
1. ✅ Outputs contain text from previous test
2. ✅ Position tracking logic has no reset mechanism
3. ✅ Symptoms match exactly what wrong positions would cause

**Fix complexity:** Low (add reset logic)  
**Risk:** Low (only affects cache initialization)  
**Expected outcome:** Should resolve quality issue completely

---

## 🔧 CRITICAL UPDATE: RoPE Indexing Bug (Jan 5, 2026)

### **Root Cause #2: Negative Indexing in RoPE Table**

After implementing the cache contamination fix, a **second critical bug** was identified in the RoPE implementation:

**The Bug:**
```python
# spectral_attention.py (Line 189 - BEFORE FIX)
relative_positions = key_positions - query_position  # Produces NEGATIVE values!
cos_relative = cos[relative_positions]  # PyTorch wraps around!
```

**Example:**
- Query at position 1200
- Keys at positions [0, 1, 2, ...]
- `relative_positions = [0-1200, 1-1200, 2-1200, ...] = [-1200, -1199, -1198, ...]`
- `cos[-1200]` → PyTorch interprets as `cos[8192-1200]` → `cos[6992]` ❌

**Impact:** Applies rotations for positions 6992, 6993, 6994... instead of distances 1200, 1199, 1198... → **Completely scrambles attention scores** → Gibberish output.

**The Fix:**
```python
# spectral_attention.py (Line 195 - AFTER FIX)
distances = (query_position - key_positions).clamp(min=0).long()  # POSITIVE values!
cos_relative = cos[distances]  # Correct indexing ✅
```

**Why This Works:**
- RoPE property: R(θ_m) · R(θ_n) = R(θ_m - θ_n)
- Distance d = query_position - key_position (always positive)
- `cos[d]` and `sin[d]` give the correct rotation for distance `d`
- Inverse RoPE uses (cos, -sin) to align query with pre-RoPE keys

**Status:** ✅ Fixed in commit (Jan 5, 2026)

---

## 📞 Support

If fix doesn't work, provide:
1. Full debug logs from patched test
2. Cache lifecycle (creation/reuse messages)
3. Position values for first 3 generations

**Debug command:**
```python
patch_unsloth_attention(model, debug_logging=True, ...)
```

Then share output.

