"""
Randomized SVD (Halko et al. 2011)

Efficient truncated SVD for low-rank approximation with batching support.

Reference:
    Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).
    Finding structure with randomness: Probabilistic algorithms for 
    constructing approximate matrix decompositions.
    SIAM review, 53(2), 217-288.

Key Innovation:
    Instead of computing full SVD (O(mn^2)), we:
    1. Sample the column space with a random matrix (O(mnk))
    2. Compute SVD on a much smaller matrix (O(k^2n))
    Total: O(mnk) where k << min(m,n)

Batching:
    Process multiple matrices simultaneously (e.g., all attention heads)
    Input: [Batch, M, N]
    Output: U [Batch, M, k], S [Batch, k], Vh [Batch, k, N]
"""

import torch
import math


def batched_randomized_svd(
    M: torch.Tensor,
    k: int,
    n_iter: int = 2,
    oversampling: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute truncated SVD using randomized algorithm (batched).
    
    This is significantly faster than torch.linalg.svd for:
    - Low-rank approximations (k << min(m, n))
    - Large matrices (m, n > 100)
    - Batched operations (multiple matrices at once)
    
    Algorithm:
    ----------
    1. Random sampling: Y = M @ Omega (capture column space)
    2. Power iteration: Y = (MM^T)^q @ Y (enhance spectral gap)
    3. Orthogonalization: Q = qr(Y) (find orthonormal basis)
    4. Projection: B = Q^T @ M (project to low-dim space)
    5. Small SVD: U_tilde, S, Vh = svd(B)
    6. Lift back: U = Q @ U_tilde
    
    Args:
        M: Input matrices [Batch, M, N]
        k: Target rank (number of singular values to compute)
        n_iter: Number of power iterations (default: 2)
                Higher = more accurate but slower
                0 = single-pass, 2 = good balance, 4+ = high accuracy
        oversampling: Extra dimensions for stability (default: 5)
                      Ensures k-th singular value is captured accurately
                      
    Returns:
        U: Left singular vectors [Batch, M, k]
        S: Singular values [Batch, k]
        Vh: Right singular vectors [Batch, k, N]
        
    Approximation Quality:
        ||M - U @ diag(S) @ Vh|| ≈ σ_{k+1}
        where σ_{k+1} is the (k+1)-th singular value
        
    Speedup:
        Standard SVD: O(mn^2)
        Randomized SVD: O(mnk)
        For k=16, m=512, n=128: ~10-15x faster
    """
    B, m, n = M.shape
    r = min(k + oversampling, min(m, n))  # Oversampled rank
    
    # Edge case: if k is too large, fall back to standard SVD
    if r >= min(m, n) - 1:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, :, :k], S[:, :k], Vh[:, :k, :]
    
    # =========================================================================
    # STAGE 1: Random Sampling (Capture Column Space)
    # =========================================================================
    
    # Generate random test matrix Omega: [B, n, r]
    # We use Gaussian random projection (most common choice)
    Omega = torch.randn(B, n, r, device=M.device, dtype=M.dtype)
    
    # Compute sampling matrix: Y = M @ Omega
    # This captures the action of M on random vectors
    # Shape: [B, m, r]
    Y = torch.bmm(M, Omega)
    
    # =========================================================================
    # STAGE 2: Power Iteration (Enhance Spectral Gap)
    # =========================================================================
    
    # The power iteration scheme: Y = (M @ M^T)^q @ Y
    # This amplifies the dominant singular values
    # After q iterations: σ_i → σ_i^(2q+1)
    # This makes the top-k subspace more distinguishable
    
    for _ in range(n_iter):
        # Y = M @ (M^T @ Y)
        Y = torch.bmm(M, torch.bmm(M.transpose(1, 2), Y))
    
    # =========================================================================
    # STAGE 3: Orthogonalization (Find Basis for Range(M))
    # =========================================================================
    
    # QR decomposition: Y = Q @ R
    # Q is an orthonormal basis for the column space of M
    # Shape: Q [B, m, r]
    Q, _ = torch.linalg.qr(Y)
    
    # =========================================================================
    # STAGE 4: Projection (Dimensionality Reduction)
    # =========================================================================
    
    # Project M into the low-dimensional subspace
    # B = Q^T @ M
    # This is a small matrix [B, r, n] instead of [B, m, n]
    # Shape: [B, r, n]
    B_proj = torch.bmm(Q.transpose(1, 2), M)
    
    # =========================================================================
    # STAGE 5: Small SVD (Exact Decomposition of Projected Matrix)
    # =========================================================================
    
    # Compute full SVD of the small matrix B_proj
    # This is fast because r << m
    # U_tilde: [B, r, r], S: [B, r], Vh: [B, r, n]
    U_tilde, S, Vh = torch.linalg.svd(B_proj, full_matrices=False)
    
    # =========================================================================
    # STAGE 6: Lift Back (Recover Full-Space Singular Vectors)
    # =========================================================================
    
    # Project U back to the original space: U = Q @ U_tilde
    # Shape: [B, m, r]
    U = torch.bmm(Q, U_tilde)
    
    # =========================================================================
    # STAGE 7: Truncation (Keep Only k Components)
    # =========================================================================
    
    # Truncate to exact rank k (discard oversampled dimensions)
    return U[:, :, :k], S[:, :k], Vh[:, :k, :]


def test_randomized_svd_correctness():
    """Test: rSVD approximation quality vs standard SVD."""
    print("="*70)
    print("TEST: Randomized SVD Correctness")
    print("="*70)
    
    # Test configuration
    B, m, n, k = 8, 512, 128, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nTest case: Batch={B}, M={m}, N={n}, k={k}")
    print(f"Device: {device}")
    
    # Generate test matrices (with some low-rank structure)
    torch.manual_seed(42)
    M_low_rank = torch.randn(B, m, k, device=device) @ torch.randn(B, k, n, device=device)
    M_noise = 0.1 * torch.randn(B, m, n, device=device)
    M = M_low_rank + M_noise
    
    # Standard SVD (reference)
    print("\n⏱️  Running standard SVD...")
    U_std, S_std, Vh_std = torch.linalg.svd(M, full_matrices=False)
    U_std, S_std, Vh_std = U_std[:, :, :k], S_std[:, :k], Vh_std[:, :k, :]
    
    # Randomized SVD
    print("⏱️  Running randomized SVD...")
    U_rsvd, S_rsvd, Vh_rsvd = batched_randomized_svd(M, k, n_iter=2)
    
    # Reconstruction error
    M_recon_std = torch.bmm(U_std * S_std.unsqueeze(1), Vh_std)
    M_recon_rsvd = torch.bmm(U_rsvd * S_rsvd.unsqueeze(1), Vh_rsvd)
    
    err_std = torch.norm(M - M_recon_std) / torch.norm(M)
    err_rsvd = torch.norm(M - M_recon_rsvd) / torch.norm(M)
    
    print(f"\n📊 Reconstruction Errors:")
    print(f"   Standard SVD:    {err_std:.6f}")
    print(f"   Randomized SVD:  {err_rsvd:.6f}")
    print(f"   Difference:      {abs(err_std - err_rsvd):.6f}")
    
    # Singular value comparison
    s_diff = torch.abs(S_std - S_rsvd).mean()
    s_rel_diff = (torch.abs(S_std - S_rsvd) / (S_std + 1e-8)).mean()
    
    print(f"\n📊 Singular Value Errors:")
    print(f"   Mean absolute diff:  {s_diff:.6f}")
    print(f"   Mean relative diff:  {s_rel_diff:.6%}")
    
    # Verdict
    # Check if rSVD is comparable to standard SVD
    # Allow up to 5% degradation in reconstruction quality
    error_increase = err_rsvd / err_std
    success = error_increase < 1.05 and s_rel_diff < 0.01
    
    if success:
        print("\n✅ PASSED: Randomized SVD is accurate!")
        print(f"   Error increase: {(error_increase - 1) * 100:.2f}%")
    else:
        print(f"\n❌ FAILED: Error increase {(error_increase - 1) * 100:.1f}% > 5%")
    
    return success


def benchmark_randomized_svd():
    """Benchmark: Speed comparison."""
    print("\n" + "="*70)
    print("BENCHMARK: Randomized SVD Speed")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("\n⚠️  Running on CPU - speedup will be less pronounced")
    else:
        print(f"\n✅ Running on GPU: {torch.cuda.get_device_name()}")
    
    test_cases = [
        (8, 128, 128, 16),   # Small
        (8, 512, 128, 16),   # Medium (KV cache size)
        (8, 1024, 128, 16),  # Large
        (8, 512, 128, 32),   # Higher rank
    ]
    
    print(f"\n{'Batch':<6} {'M':<6} {'N':<6} {'k':<6} {'Standard (ms)':<15} {'rSVD (ms)':<15} {'Speedup'}")
    print("-"*70)
    
    for B, m, n, k in test_cases:
        # Generate test matrix
        M = torch.randn(B, m, n, device=device, dtype=torch.float32)
        
        # Warm-up
        _ = torch.linalg.svd(M[0], full_matrices=False)
        _ = batched_randomized_svd(M, k)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark standard SVD
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(10):
                _ = torch.linalg.svd(M, full_matrices=False)
            end.record()
            torch.cuda.synchronize()
            time_std = start.elapsed_time(end) / 10
        else:
            import time
            start = time.time()
            for _ in range(10):
                _ = torch.linalg.svd(M, full_matrices=False)
            time_std = (time.time() - start) * 100  # to ms
        
        # Benchmark rSVD
        if device == "cuda":
            start.record()
            for _ in range(10):
                _ = batched_randomized_svd(M, k)
            end.record()
            torch.cuda.synchronize()
            time_rsvd = start.elapsed_time(end) / 10
        else:
            start = time.time()
            for _ in range(10):
                _ = batched_randomized_svd(M, k)
            time_rsvd = (time.time() - start) * 100
        
        speedup = time_std / time_rsvd
        
        print(f"{B:<6} {m:<6} {n:<6} {k:<6} {time_std:>12.2f}   {time_rsvd:>12.2f}   {speedup:>6.2f}x")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    # Run tests
    success = test_randomized_svd_correctness()
    
    if success:
        benchmark_randomized_svd()
    else:
        print("\n⚠️  Skipping benchmark due to correctness failure")

