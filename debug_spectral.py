#!/usr/bin/env python3
"""
Debug script to understand spectral attention issues.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from unsloth_spectral import SpectralCache
import math


def debug_spectral_math():
    """Debug the spectral compression and attention math."""
    print("="*70)
    print("DEBUGGING SPECTRAL MATH")
    print("="*70)
    
    # Simple test case: 1 head, small dimensions
    B, H, T, D, k = 1, 1, 128, 32, 8
    
    # Create synthetic K, V with known structure
    torch.manual_seed(42)
    K_orig = torch.randn(B, H, T, D)
    V_orig = torch.randn(B, H, T, D)
    
    print(f"\nOriginal K shape: {K_orig.shape}")
    
    # Manual compression (mimic SpectralCache logic)
    K_h = K_orig.squeeze(0).squeeze(0)  # [T, D]
    print(f"K_h shape: {K_h.shape}")
    
    # SVD
    U, S, Vh = torch.linalg.svd(K_h, full_matrices=False)
    print(f"SVD shapes: U={U.shape}, S={S.shape}, Vh={Vh.shape}")
    
    # Truncate
    U_k = U[:, :k]  # [T, k]
    S_k = S[:k]  # [k]
    Vh_k = Vh[:k, :]  # [k, D]
    
    # Coefficients and basis
    coeffs = U_k * S_k.unsqueeze(0)  # [T, k]
    basis = Vh_k  # [k, D]
    
    print(f"\nCompressed shapes:")
    print(f"  coeffs: {coeffs.shape}")
    print(f"  basis: {basis.shape}")
    
    # Reconstruct
    K_recon = torch.matmul(coeffs, basis)  # [T, D]
    print(f"\nReconstructed K shape: {K_recon.shape}")
    
    # Check reconstruction error
    recon_error = torch.norm(K_h - K_recon) / torch.norm(K_h)
    print(f"Reconstruction error: {recon_error:.6f}")
    
    # Now test attention scores
    Q = K_orig[:, :, -1:, :]  # [B, H, 1, D] - use last token as query
    print(f"\nQuery Q shape: {Q.shape}")
    
    # Standard attention scores
    scores_standard = torch.matmul(Q, K_orig.transpose(-2, -1))  # [B, H, 1, T]
    print(f"Standard scores shape: {scores_standard.shape}")
    
    # Spectral attention scores (method 1: via reconstruction)
    K_recon_full = K_recon.unsqueeze(0).unsqueeze(0)  # [B, H, T, D]
    scores_recon = torch.matmul(Q, K_recon_full.transpose(-2, -1))  # [B, H, 1, T]
    print(f"Reconstructed scores shape: {scores_recon.shape}")
    
    # Spectral attention scores (method 2: direct computation)
    # Q @ K^T = Q @ (coeffs @ basis)^T = Q @ basis^T @ coeffs^T
    # Q: [B, H, 1, D]
    # basis: [k, D]
    # coeffs: [T, k]
    
    basis_batched = basis.unsqueeze(0).unsqueeze(0)  # [1, 1, k, D]
    Q_proj = torch.matmul(Q, basis_batched.transpose(-2, -1))  # [1, 1, 1, k]
    print(f"Q projected to spectral space: {Q_proj.shape}")
    
    coeffs_batched = coeffs.unsqueeze(0).unsqueeze(0)  # [1, 1, T, k]
    scores_spectral = torch.matmul(Q_proj, coeffs_batched.transpose(-2, -1))  # [1, 1, 1, T]
    print(f"Spectral scores shape: {scores_spectral.shape}")
    
    # Compare
    print(f"\n{'Method':<20} {'Mean Score':<15} {'Std Score':<15}")
    print("-"*50)
    print(f"{'Standard':<20} {scores_standard.mean():.6f}      {scores_standard.std():.6f}")
    print(f"{'Via Reconstruction':<20} {scores_recon.mean():.6f}      {scores_recon.std():.6f}")
    print(f"{'Direct Spectral':<20} {scores_spectral.mean():.6f}      {scores_spectral.std():.6f}")
    
    # Correlation
    s_std = scores_standard.flatten()
    s_rec = scores_recon.flatten()
    s_spec = scores_spectral.flatten()
    
    corr_rec = torch.corrcoef(torch.stack([s_std, s_rec]))[0, 1]
    corr_spec = torch.corrcoef(torch.stack([s_std, s_spec]))[0, 1]
    
    print(f"\n{'Comparison':<25} {'Correlation':<15}")
    print("-"*40)
    print(f"{'Standard vs Recon':<25} {corr_rec:.6f}")
    print(f"{'Standard vs Spectral':<25} {corr_spec:.6f}")
    
    # Check if reconstruction and spectral give same result
    corr_methods = torch.corrcoef(torch.stack([s_rec, s_spec]))[0, 1]
    print(f"{'Recon vs Spectral':<25} {corr_methods:.6f}")
    print(f"\n{'Expected: 1.000000 (should be identical methods)'}")
    
    if corr_methods < 0.999:
        print("\n❌ ERROR: Spectral method != Reconstruction method!")
        print("   This indicates a bug in the spectral attention implementation.")
    else:
        print("\n✅ Spectral method matches reconstruction (math is correct)")
        
        if corr_rec < 0.95:
            print(f"⚠️  But correlation with original is only {corr_rec:.4f}")
            print(f"   This means rank k={k} is too aggressive for this data.")
            print(f"   Try increasing k or check variance explained.")


if __name__ == "__main__":
    debug_spectral_math()

