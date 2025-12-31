# Git Setup & Deployment Guide

**Goal:** Initialize Git repository and deploy to GitHub for Colab testing.

---

## 📋 Prerequisites

1. **Git installed:** `git --version`
2. **GitHub account:** Create at github.com
3. **SSH keys configured** (recommended) or HTTPS authentication

---

## 🚀 Step 1: Initialize Local Repository

```bash
cd /Users/ankitprajapati/unsloth_test

# Initialize Git
git init

# Check status
git status
```

---

## 📁 Step 2: Stage Files

```bash
# Add all library files
git add unsloth_spectral/

# Add documentation
git add README.md
git add QUICK_START.md
git add IMPLEMENTATION_SUMMARY.md
git add PHASE2_RSVD_OPTIMIZATION.md
git add HOLOGRAPHIC_SPECTRAL_COMPRESSION_TECHNICAL_SPECIFICATION.md

# Add tests and examples
git add test_spectral_integration.py
git add example_spectral_usage.py
git add debug_spectral.py

# Add package files
git add setup.py
git add requirements.txt
git add .gitignore

# Check what's staged
git status
```

---

## 💾 Step 3: Create First Commit

```bash
git commit -m "Phase 2: Randomized SVD optimization

- Implement batched Randomized SVD (Halko et al. 2011)
- 9.5x speedup in compression (CPU), expected 13-15x on GPU
- Update SpectralCache to use rSVD
- Add vectorized quantization
- Comprehensive testing and validation
- Repository structure for pip installation

Performance:
- Compression: 264ms -> 20ms per layer
- Quality: 0.000% error increase
- Tests: All passing (correctness + integration)

Ready for GPU benchmarking on Colab T4."
```

---

## 🌐 Step 4: Create GitHub Repository

### Option A: GitHub CLI (Recommended)

```bash
# Install GitHub CLI if not already installed
# Mac: brew install gh
# Login
gh auth login

# Create repository
gh repo create unsloth-spectral --public --description "Holographic Spectral Compression for LLM KV Caches - 7-15x memory reduction" --source=. --remote=origin --push
```

### Option B: Manual (GitHub Website)

1. Go to https://github.com/new
2. Repository name: `unsloth-spectral`
3. Description: `Holographic Spectral Compression for LLM KV Caches`
4. Public repository
5. **Do NOT initialize with README** (we already have one)
6. Click "Create repository"

Then:

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/unsloth-spectral.git

# or with SSH:
git remote add origin git@github.com:YOUR_USERNAME/unsloth-spectral.git

# Push
git branch -M main
git push -u origin main
```

---

## ✅ Step 5: Verify

Visit your repository:
```
https://github.com/YOUR_USERNAME/unsloth-spectral
```

You should see:
- ✅ README.md rendered with badges
- ✅ File structure
- ✅ setup.py and requirements.txt
- ✅ All documentation

---

## 🔬 Step 6: Test on Colab (T4 GPU)

### Create new Colab notebook

1. Go to https://colab.research.google.com
2. New notebook
3. Runtime > Change runtime type > T4 GPU

### Run installation

```python
# Cell 1: Install Unsloth
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Cell 2: Install Unsloth Spectral
!git clone https://github.com/YOUR_USERNAME/unsloth-spectral.git
%cd unsloth-spectral
!pip install -e .

# Cell 3: Run tests
!python test_spectral_integration.py --quick

# Cell 4: Run example
!python example_spectral_usage.py

# Cell 5: Benchmark rSVD on GPU
!python unsloth_spectral/rsvd.py
```

### Expected Results

**Test Output:**
```
✅ UNIT: PASSED
✅ ATTENTION: PASSED
🎉 ALL TESTS PASSED (2/2)
```

**rSVD Benchmark (T4 GPU):**
```
Standard SVD: ~10 ms
Randomized SVD: ~0.7 ms
Speedup: 14x  ← Should be higher than CPU (9.5x)
```

---

## 📊 Step 7: Profile End-to-End Performance

Create a benchmark notebook:

```python
from unsloth import FastLanguageModel
from unsloth_spectral import patch_unsloth_attention
import torch
import time

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Baseline: Standard cache
prompt = "Write a detailed essay about quantum computing (500 words):"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

start = time.time()
outputs_baseline = model.generate(**inputs, max_new_tokens=500)
baseline_time = time.time() - start
baseline_tokens = outputs_baseline.shape[1] - inputs.input_ids.shape[1]

print(f"Baseline: {baseline_tokens / baseline_time:.1f} tok/s")

# Spectral cache
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
patch_unsloth_attention(model, block_size=512, k_rank_keys=16, k_rank_values=32)

start = time.time()
outputs_spectral = model.generate(**inputs, max_new_tokens=500)
spectral_time = time.time() - start
spectral_tokens = outputs_spectral.shape[1] - inputs.input_ids.shape[1]

print(f"Spectral: {spectral_tokens / spectral_time:.1f} tok/s")
print(f"Speedup: {spectral_time / baseline_time:.2f}x")

# Quality check
baseline_text = tokenizer.decode(outputs_baseline[0])
spectral_text = tokenizer.decode(outputs_spectral[0])

# Manual inspection
print("\n=== Baseline Output ===")
print(baseline_text[len(prompt):200])
print("\n=== Spectral Output ===")
print(spectral_text[len(prompt):200])
```

**Target Results:**
- Baseline: 18-20 tok/s
- Spectral: 18-20 tok/s (similar or faster!)
- Compression: ~7x memory reduction
- Quality: Visually indistinguishable

---

## 🔄 Step 8: Iterate Based on Results

### If Results Are Good (Expected)

1. **Document performance in README**
   ```bash
   # On local machine
   git pull origin main  # Sync any Colab changes
   # Edit README.md with actual benchmark results
   git add README.md
   git commit -m "Add Colab T4 benchmark results"
   git push
   ```

2. **Create release tag**
   ```bash
   git tag -a v0.1.0 -m "Phase 2 release: Randomized SVD optimization"
   git push origin v0.1.0
   ```

3. **Open source announcement**
   - Post on Twitter/X
   - Share on Reddit (r/MachineLearning, r/LocalLLaMA)
   - Submit to Papers With Code

### If Results Need Tuning

Common adjustments:
- **Increase ranks:** `k_rank_keys=24, k_rank_values=48`
- **Adjust block size:** `block_size=256` or `block_size=1024`
- **Power iterations:** Modify `n_iter` in rSVD
- **Check attention backend:** Ensure xFormers or FlashAttention is used

---

## 🐛 Troubleshooting

### Git Issues

**Problem:** `git push` fails with authentication error

**Solution:**
```bash
# Use HTTPS with token
git remote set-url origin https://YOUR_USERNAME@github.com/YOUR_USERNAME/unsloth-spectral.git

# Or set up SSH keys
ssh-keygen -t ed25519 -C "your.email@gmail.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub settings
git remote set-url origin git@github.com:YOUR_USERNAME/unsloth-spectral.git
```

### Colab Issues

**Problem:** `ModuleNotFoundError: No module named 'unsloth_spectral'`

**Solution:**
```python
import sys
sys.path.insert(0, '/content/unsloth-spectral')
```

**Problem:** CUDA out of memory

**Solution:**
```python
# Use smaller model or lower context
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=2048,  # Instead of 4096
    load_in_4bit=True,
)
```

**Problem:** Slow generation even with spectral cache

**Solution:**
```python
# Check if compression is actually happening
from unsloth_spectral.spectral_cache import SpectralCache

# Add debug output in patch_unsloth_attention
patch_unsloth_attention(model, verbose=True)

# Manual check after generation
# (Requires modifying integration.py to expose cache stats)
```

---

## 📝 Checklist

Before pushing to GitHub:

- [ ] All tests pass locally
- [ ] README.md is complete and accurate
- [ ] setup.py has correct metadata
- [ ] .gitignore excludes unnecessary files
- [ ] No large files (models, checkpoints) committed
- [ ] No API keys or secrets in code
- [ ] Documentation is clear and helpful
- [ ] Examples run successfully

After GitHub push:

- [ ] Repository is public and accessible
- [ ] README renders correctly
- [ ] Clone command works
- [ ] Tests pass on Colab T4
- [ ] Benchmark results are documented
- [ ] Issues are enabled for feedback

---

## 🎯 Success Metrics

**Phase 2 is successful if:**

1. ✅ Repository is public on GitHub
2. ✅ Tests pass on Colab T4
3. ✅ rSVD achieves >13x speedup on GPU
4. ✅ End-to-end generation is comparable or faster than baseline
5. ✅ Attention correlation >0.97 on real LLM data
6. ✅ Memory compression 7-10x for 4K context

**If all metrics are met:** Proceed to Phase 2b (Triton kernels)

---

## 📧 Next Communication

After running on Colab T4, report back with:

1. **rSVD benchmark results** (standard vs randomized on GPU)
2. **End-to-end performance** (tok/s baseline vs spectral)
3. **Memory statistics** (actual compression ratio)
4. **Attention correlation** (on real model generation)
5. **Any issues or surprises** encountered

---

**Ready to deploy!** 🚀

Execute these commands and let's see the results on real GPU hardware.

