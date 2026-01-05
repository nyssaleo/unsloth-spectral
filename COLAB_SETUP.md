# Colab Setup Guide: Unsloth Spectral (Phase 2)

## 🎯 Objective
Test spectral KV cache compression on real Mistral 7B model using Unsloth in Google Colab.

---

## 📋 Prerequisites

- **Google Colab** with GPU runtime (T4, V100, or A100)
- **HuggingFace account** (for model download)
- **~10-15 minutes** for complete test

---

## 🚀 Step-by-Step Setup

### Step 1: Enable GPU Runtime

1. In Colab: `Runtime` → `Change runtime type`
2. Select `T4 GPU` or higher
3. Click `Save`

### Step 2: Install Dependencies

```python
# Cell 1: Install Unsloth
!pip install -q unsloth

# Cell 2: Install Unsloth Spectral
!pip install -q git+https://github.com/nyssaleo/unsloth-spectral.git
```

### Step 3: Verify Installation

```python
# Cell 3: Test imports
import torch
from unsloth import FastLanguageModel
from unsloth_spectral import patch_unsloth_attention

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
```

### Step 4: Run Phase 2 Test

```python
# Cell 4: Download and run test script
!wget https://raw.githubusercontent.com/nyssaleo/unsloth-spectral/main/test_phase2_mistral_colab.py
!python test_phase2_mistral_colab.py
```

**Or run inline:**

```python
# Cell 4 (Alternative): Run test inline
!python -c "
from test_phase2_mistral_colab import main
main()
"
```

---

## 📊 Expected Output

The test will:

1. **Load Mistral 7B** (4-bit quantized via Unsloth)
2. **Test baseline** (standard Unsloth cache)
   - Generate text with 3 prompts
   - Test long context (2K tokens)
3. **Apply spectral patch** (compressed cache)
   - Generate same prompts
   - Test same long context
4. **Compare results**:
   - Memory compression ratio
   - Speed performance
   - Generation quality

### Success Criteria

✅ **Memory compression**: >1.5x reduction  
✅ **Speed**: >0.8x of baseline (some slowdown acceptable)  
✅ **Quality**: Outputs remain coherent and similar

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Use shorter context or smaller model

```python
# Modify test script
context_length = 1024  # Instead of 2048
```

### Error: "No module named 'unsloth_spectral'"

**Solution**: Reinstall from GitHub

```bash
!pip uninstall -y unsloth-spectral
!pip install -q git+https://github.com/nyssaleo/unsloth-spectral.git@main
```

### Error: "Model download failed"

**Solution**: Login to HuggingFace

```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

---

## 📈 Understanding the Results

### Memory Compression

The test measures GPU memory delta (change in allocation):

- **Baseline**: Standard KV cache (FP16)
- **Spectral**: Compressed cache (rank 16/32 + INT8)

Expected compression: **4-8x** for sequences >2K tokens

### Speed Performance

Spectral cache trades compute for memory:

- **Prefill**: May be slightly slower (1-10%)
- **Decode**: Should be similar or faster for long contexts
- **Overall**: 0.8-1.2x of baseline

### Quality Check

Simple token overlap metric (production would use perplexity):

- **>70% overlap**: Excellent
- **50-70% overlap**: Good
- **<50% overlap**: Needs tuning

---

## 🎓 What to Look For

### Good Signs ✅

- No crashes or errors
- Memory delta significantly lower with spectral
- Generated text is coherent
- Speed within 20% of baseline

### Red Flags ⚠️

- Crashes during generation
- Memory usage **increases** (bug!)
- Gibberish output (numerical issues)
- Speed <50% of baseline (too slow)

---

## 🔬 Advanced Testing

### Custom Prompts

```python
test_prompts = [
    "Your custom prompt here",
    "Another test case",
]

results = test_generation_quality(model, tokenizer, test_prompts)
```

### Longer Contexts

```python
# Test up to 8K tokens (model's max)
test_long_context(model, tokenizer, context_length=8192)
```

### Debug Mode

```python
patch_unsloth_attention(
    model,
    debug_logging=True,  # Enable verbose output
    # ... other params ...
)
```

---

## 📁 Files in Repository

- `test_phase2_mistral_colab.py` - Main test script
- `unsloth_spectral/` - Library code
  - `integration.py` - Unsloth patching
  - `spectral_cache.py` - Cache implementation
  - `spectral_attention.py` - RoPE-aware attention
- `test_phase1_simple.py` - Unit tests (CPU)
- `test_gqa_fix.py` - GQA validation

---

## 🚀 Next Steps After Phase 2

Once Phase 2 passes:

1. **Measure perplexity** on Wikitext-2 (proper quality metric)
2. **Profile memory** across various sequence lengths
3. **Benchmark speed** vs other compression methods
4. **Phase 3**: Optimize with Triton kernel for T4

---

## 📞 Support

Issues? Check:
- [GitHub Issues](https://github.com/nyssaleo/unsloth-spectral/issues)
- Verify GPU is enabled in Colab
- Try restarting runtime

---

## 📜 License

MIT License - See repository for details

---

**Ready to test! 🎉**

Just run the cells in order and watch the magic happen.

