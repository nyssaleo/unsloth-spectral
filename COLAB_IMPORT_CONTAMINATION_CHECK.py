"""
IMPORT CONTAMINATION CHECK
===========================

Simple script to check if importing unsloth_spectral changes Unsloth's state.

This will definitively answer: Does the mere ACT of importing our library
change Unsloth's behavior, even before calling patch_unsloth_attention()?
"""

import subprocess
import sys

def install_packages():
    packages = [("triton", "triton"), ("unsloth", "unsloth")]
    for name, pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/nyssaleo/unsloth-spectral.git"
    ])

install_packages()

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

print("\n" + "="*70)
print("🔬 IMPORT CONTAMINATION CHECK")
print("="*70)

# =============================================================================
# STEP 1: Capture BEFORE state
# =============================================================================

print("\n📸 STEP 1: Capturing Unsloth state BEFORE import...")

import unsloth.models.llama as llama_module

before_state = {
    "LlamaAttention_fast_forward": id(llama_module.LlamaAttention_fast_forward),
    "LlamaAttention_fast_forward_inference": id(llama_module.LlamaAttention_fast_forward_inference),
    "LlamaModel_fast_forward_inference": id(llama_module.LlamaModel_fast_forward_inference),
}

print("✅ Captured state")
for name, obj_id in before_state.items():
    print(f"  {name}: {obj_id}")

# Check if _LlamaModel_fast_forward_inference exists
if hasattr(llama_module, '_LlamaModel_fast_forward_inference'):
    before_state["_LlamaModel_fast_forward_inference"] = id(llama_module._LlamaModel_fast_forward_inference)
    print(f"  _LlamaModel_fast_forward_inference: {before_state['_LlamaModel_fast_forward_inference']}")

# =============================================================================
# STEP 2: Import unsloth_spectral
# =============================================================================

print("\n📦 STEP 2: Importing unsloth_spectral...")
print("  (NOT calling patch_unsloth_attention, just importing)")

import os
os.environ["UNSLOTH_SPECTRAL_QUIET"] = "1"

from unsloth_spectral import patch_unsloth_attention

print("✅ Import complete")

# =============================================================================
# STEP 3: Capture AFTER state
# =============================================================================

print("\n📸 STEP 3: Capturing Unsloth state AFTER import...")

after_state = {
    "LlamaAttention_fast_forward": id(llama_module.LlamaAttention_fast_forward),
    "LlamaAttention_fast_forward_inference": id(llama_module.LlamaAttention_fast_forward_inference),
    "LlamaModel_fast_forward_inference": id(llama_module.LlamaModel_fast_forward_inference),
}

print("✅ Captured state")
for name, obj_id in after_state.items():
    print(f"  {name}: {obj_id}")

if hasattr(llama_module, '_LlamaModel_fast_forward_inference'):
    after_state["_LlamaModel_fast_forward_inference"] = id(llama_module._LlamaModel_fast_forward_inference)
    print(f"  _LlamaModel_fast_forward_inference: {after_state['_LlamaModel_fast_forward_inference']}")

# =============================================================================
# STEP 4: Compare
# =============================================================================

print("\n" + "="*70)
print("📊 COMPARISON")
print("="*70)

contaminated = False

for name in before_state:
    before_id = before_state[name]
    after_id = after_state.get(name)
    
    if before_id != after_id:
        print(f"❌ {name}: CHANGED!")
        print(f"   Before: {before_id}")
        print(f"   After:  {after_id}")
        contaminated = True
    else:
        print(f"✅ {name}: unchanged")

print("\n" + "="*70)
if contaminated:
    print("🚨 CONTAMINATION DETECTED!")
    print("   Importing unsloth_spectral CHANGES Unsloth's state!")
    print("   This explains why baseline fails in Test 3.")
else:
    print("✅ NO CONTAMINATION")
    print("   Importing unsloth_spectral does NOT change Unsloth.")
    print("   The Test 3 failure must have another cause.")

# =============================================================================
# STEP 5: Check sys.modules
# =============================================================================

print("\n" + "="*70)
print("📦 sys.modules CHECK")
print("="*70)

import sys

print("\nUnsloth-related modules:")
for name in sorted(sys.modules.keys()):
    if 'unsloth' in name.lower():
        print(f"  {name}")

print("\nSpectral-related modules:")
for name in sorted(sys.modules.keys()):
    if 'spectral' in name.lower():
        print(f"  {name}")

# =============================================================================
# STEP 6: Functional test
# =============================================================================

print("\n" + "="*70)
print("🧪 FUNCTIONAL TEST")
print("="*70)
print("Testing actual generation with contaminated Unsloth...")

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
TEST_PROMPT = "Context: The code is ALPHA-123. Question: What is the code? Answer:"

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
prompt_tokens = inputs.input_ids.shape[1]

print(f"Prompt: {TEST_PROMPT}")
print(f"Prompt tokens: {prompt_tokens}")
print("Generating...")

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

output = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True)

print(f"\nOutput: {output}")
print(f"Needle: {'✅ Found' if 'ALPHA-123' in output else '❌ Not found'}")

print("\n" + "="*70)
print("📋 FINAL VERDICT")
print("="*70)

if contaminated:
    print("🚨 Our import CONTAMINATES Unsloth!")
    print("   We need to find what in our __init__.py or imports causes this.")
else:
    if 'ALPHA-123' in output:
        print("✅ No contamination, baseline works fine")
    else:
        print("⚠️ No contamination detected, but baseline still fails")
        print("   The issue must be in the test setup, not our library")
