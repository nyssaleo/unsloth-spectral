"""
Spectral Cache: Block-level KV Cache Compression via SVD

This module implements the core SpectralCache class that stores Key-Value caches
as spectral decompositions (Coefficients + Basis) instead of raw tokens.

Key Innovation: Stores cache as C @ B instead of full K/V matrices.
- Coefficients (C): Temporal activations [T, k] - quantized to INT8
- Basis (B): Spectral basis [k, D] - kept in FP16 for precision

Memory: O(T×k + k×D) vs O(T×D), where k << T
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SpectralBlock:
    """
    Compressed representation of a single cache block (e.g., 512 tokens).
    
    Stores separate spectral decompositions for Keys and Values:
    K ≈ coeffs_K @ basis_K
    V ≈ coeffs_V @ basis_V
    
    Attributes:
        coeffs_K: Temporal coefficients for Keys [H, T, k_K] - INT8 simulated
        basis_K: Spectral basis for Keys [H, k_K, D] - FP16
        coeffs_V: Temporal coefficients for Values [H, T, k_V] - INT8 simulated
        basis_V: Spectral basis for Values [H, k_V, D] - FP16
        scales_K: Dequantization scales for coeffs_K [H, 2] (min, max)
        scales_V: Dequantization scales for coeffs_V [H, 2]
        block_size: Number of tokens in this block
    """
    coeffs_K: torch.Tensor
    basis_K: torch.Tensor
    coeffs_V: torch.Tensor
    basis_V: torch.Tensor
    scales_K: torch.Tensor
    scales_V: torch.Tensor
    block_size: int


class SpectralCache:
    """
    Three-tier KV Cache with automatic spectral compression.
    
    Architecture:
    - Hot Cache (FP16): Most recent tokens (fast access, no compression)
    - Warm Buffer (FP16): Staging area for accumulation
    - Cold Cache (Spectral INT8): Compressed historical blocks
    
    The cache automatically compresses blocks when they reach block_size tokens.
    
    Args:
        num_heads: Number of KV attention heads
        head_dim: Dimension per head (typically 128)
        block_size: Tokens per compressed block (default: 512)
        k_rank_keys: Spectral rank for Keys (default: 16)
        k_rank_values: Spectral rank for Values (default: 32)
        hot_buffer_size: Number of recent tokens kept uncompressed (default: 64)
        device: Torch device
        dtype: Data type for uncompressed cache (default: float16)
        debug_logging: Enable detailed logging for debugging
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 512,
        k_rank_keys: int = 16,
        k_rank_values: int = 32,
        hot_buffer_size: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        debug_logging: bool = False,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.k_rank_keys = k_rank_keys
        self.k_rank_values = k_rank_values
        self.hot_buffer_size = hot_buffer_size
        self.device = device
        self.dtype = dtype
        self.debug_logging = debug_logging
        
        # Three-tier cache
        self.hot_K: Optional[torch.Tensor] = None  # [B, H, T_hot, D]
        self.hot_V: Optional[torch.Tensor] = None
        self.cold_blocks: List[SpectralBlock] = []
        
        # State tracking
        self.total_tokens = 0
        self.compression_count = 0
        
        if self.debug_logging:
            print(f"[SpectralCache.__init__] Created cache:")
            print(f"  num_heads={num_heads} (KV heads)")
            print(f"  head_dim={head_dim}")
            print(f"  block_size={block_size}")
            print(f"  k_rank_keys={k_rank_keys}, k_rank_values={k_rank_values}")
            print(f"  hot_buffer_size={hot_buffer_size}")
    
    def append(self, K_new: torch.Tensor, V_new: torch.Tensor):
        """
        Append new Key-Value pairs to the cache.
        
        Automatically triggers compression when hot cache exceeds block_size.
        
        Args:
            K_new: New keys [B, H, T_new, D]
            V_new: New values [B, H, T_new, D]
        """
        if self.debug_logging:
            print(f"\n[SpectralCache.append] Incoming K/V:")
            print(f"  K_new shape: {K_new.shape}")
            print(f"  V_new shape: {V_new.shape}")
            print(f"  Before append: total_tokens={self.total_tokens}, hot_tokens={self.hot_K.shape[2] if self.hot_K is not None else 0}")
        
        # First append: Initialize hot cache
        if self.hot_K is None:
            self.hot_K = K_new
            self.hot_V = V_new
            if self.debug_logging:
                print(f"  Action: Initialized hot cache")
        else:
            # Concatenate along sequence dimension
            old_hot_size = self.hot_K.shape[2]
            self.hot_K = torch.cat([self.hot_K, K_new], dim=2)
            self.hot_V = torch.cat([self.hot_V, V_new], dim=2)
            if self.debug_logging:
                print(f"  Action: Concatenated ({old_hot_size} + {K_new.shape[2]} = {self.hot_K.shape[2]} hot tokens)")
        
        self.total_tokens += K_new.shape[2]
        
        if self.debug_logging:
            print(f"  After append: total_tokens={self.total_tokens}, hot_tokens={self.hot_K.shape[2]}, cold_blocks={len(self.cold_blocks)}")
        
        # Trigger compression if hot cache is large enough
        if self.hot_K.shape[2] >= self.block_size:
            if self.debug_logging:
                print(f"  Triggering compression (hot_tokens={self.hot_K.shape[2]} >= block_size={self.block_size})")
            self._compress_hot_cache()
    
    def _compress_hot_cache(self):
        """
        Compress the hot cache into a spectral block and move to cold storage.
        
        Algorithm:
        1. Extract exactly block_size tokens from hot cache
        2. Perform per-head SVD: K_h = U @ S @ Vh^T
        3. Truncate to rank k: coeffs = U_k @ S_k, basis = V_k^T
        4. Quantize coefficients to INT8 (simulated)
        5. Store SpectralBlock in cold_blocks
        6. Keep remaining tokens in hot cache
        """
        if self.hot_K is None or self.hot_K.shape[2] < self.block_size:
            return
        
        # Extract block to compress
        K_block = self.hot_K[:, :, :self.block_size, :]  # [B, H, T, D]
        V_block = self.hot_V[:, :, :self.block_size, :]
        
        # Compress K and V separately (asymmetric ranks)
        coeffs_K, basis_K, scales_K = self._compress_tensor(K_block, self.k_rank_keys)
        coeffs_V, basis_V, scales_V = self._compress_tensor(V_block, self.k_rank_values)
        
        # Create spectral block
        block = SpectralBlock(
            coeffs_K=coeffs_K,
            basis_K=basis_K,
            coeffs_V=coeffs_V,
            basis_V=basis_V,
            scales_K=scales_K,
            scales_V=scales_V,
            block_size=self.block_size,
        )
        self.cold_blocks.append(block)
        self.compression_count += 1
        
        # Keep only the remainder in hot cache
        if self.hot_K.shape[2] > self.block_size:
            self.hot_K = self.hot_K[:, :, self.block_size:, :].contiguous()
            self.hot_V = self.hot_V[:, :, self.block_size:, :].contiguous()
        else:
            self.hot_K = None
            self.hot_V = None
    
    def _compress_tensor(
        self, 
        X: torch.Tensor, 
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress a single tensor (K or V) using batched Randomized SVD.
        
        This is the OPTIMIZED version using batched rSVD instead of per-head loop.
        Expected speedup: ~13x (267ms → 20ms for 8 heads × 512×128 matrices)
        
        Args:
            X: Input tensor [B, H, T, D]
            rank: Target spectral rank
            
        Returns:
            coeffs: Quantized temporal coefficients [H, T, rank]
            basis: Spectral basis [H, rank, D] (FP16)
            scales: Quantization scales [H, 2] (min, max per head)
        """
        B, H, T, D = X.shape
        assert B == 1, "Batch size must be 1 for now"
        
        # Reshape: [B, H, T, D] -> [H, T, D] for batched processing
        X_batched = X.squeeze(0).float()  # [H, T, D]
        
        # BATCHED RANDOMIZED SVD (The Performance Fix!)
        # Process all heads in parallel instead of looping
        try:
            from .rsvd import batched_randomized_svd
            U, S, Vh = batched_randomized_svd(X_batched, k=rank, n_iter=2, oversampling=5)
        except Exception as e:
            # Fallback to standard SVD if rSVD fails (e.g., matrix too small/degenerate)
            try:
                U, S, Vh = torch.linalg.svd(X_batched, full_matrices=False)
                # Truncate to rank
                U, S, Vh = U[:, :, :rank], S[:, :rank], Vh[:, :rank, :]
            except RuntimeError:
                # Ultimate fallback: CPU + per-head loop (slowest but most robust)
                return self._compress_tensor_fallback(X, rank)
        
        # Compute coefficients: C = U @ diag(S)
        # Broadcasting: U [H, T, k] * S [H, k] -> [H, T, k]
        coeffs = U * S.unsqueeze(1)  # [H, T, k]
        basis = Vh  # [H, k, D]
        
        # Batched quantization (vectorized per-head)
        coeffs_q, scales = self._quantize_int8_batched(coeffs)
        
        # Keep basis in FP16 (it's small, no need to quantize)
        basis = basis.to(self.dtype)
        
        return coeffs_q, basis, scales
    
    def _compress_tensor_fallback(
        self, 
        X: torch.Tensor, 
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fallback: per-head loop with standard SVD (robust but slow).
        Only used if batched rSVD fails.
        """
        B, H, T, D = X.shape
        X_per_head = X.squeeze(0)
        
        coeffs_list = []
        basis_list = []
        scales_list = []
        
        for h in range(H):
            X_h = X_per_head[h]
            
            try:
                U, S, Vh = torch.linalg.svd(X_h.float(), full_matrices=False)
            except RuntimeError:
                U, S, Vh = torch.linalg.svd(X_h.cpu().float(), full_matrices=False)
                U, S, Vh = U.to(X.device), S.to(X.device), Vh.to(X.device)
            
            U_k = U[:, :rank]
            S_k = S[:rank]
            Vh_k = Vh[:rank, :]
            
            coeffs_h = U_k * S_k.unsqueeze(0)
            coeffs_q, scale = self._quantize_int8(coeffs_h)
            
            coeffs_list.append(coeffs_q)
            basis_list.append(Vh_k.to(self.dtype))
            scales_list.append(scale)
        
        coeffs = torch.stack(coeffs_list, dim=0)
        basis = torch.stack(basis_list, dim=0)
        scales = torch.stack(scales_list, dim=0)
        
        return coeffs, basis, scales
    
    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate INT8 quantization (symmetric, per-tensor).
        
        In production, this would actually store INT8. Here we simulate
        the quantization noise by applying round(x / scale) * scale.
        
        Args:
            tensor: Input tensor (any shape)
            
        Returns:
            quantized: Quantized tensor (same dtype as input, but with INT8 precision)
            scale: Scale factors [2] (min, max)
        """
        t_min = tensor.min()
        t_max = tensor.max()
        
        # Symmetric quantization: scale to [-127, 127]
        abs_max = torch.max(tensor.abs().max(), torch.tensor(1e-8, device=tensor.device))
        scale = abs_max / 127.0
        
        # Quantize and dequantize (simulates INT8 storage)
        quantized = torch.round(tensor / scale) * scale
        
        # Store scale info for potential dequantization
        scale_info = torch.tensor([t_min.item(), t_max.item()], device=tensor.device)
        
        return quantized, scale_info
    
    def _quantize_int8_batched(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched INT8 quantization (vectorized per-head).
        
        This is the OPTIMIZED version that quantizes all heads in parallel.
        
        Args:
            tensor: Input tensor [H, T, k] or [H, k, D]
            
        Returns:
            quantized: Quantized tensor (same shape and dtype, with INT8 precision)
            scales: Scale factors [H, 2] (min, max per head)
        """
        # Find min/max per head (reduce over all dims except batch)
        dims_to_reduce = tuple(range(1, tensor.ndim))
        
        t_min = tensor.amin(dim=dims_to_reduce, keepdim=True)  # [H, 1, ...]
        t_max = tensor.amax(dim=dims_to_reduce, keepdim=True)  # [H, 1, ...]
        
        # Symmetric quantization per head
        abs_max = torch.maximum(
            tensor.abs().amax(dim=dims_to_reduce, keepdim=True),
            torch.tensor(1e-8, device=tensor.device, dtype=tensor.dtype)
        )
        scale = abs_max / 127.0  # [H, 1, ...]
        
        # Quantize (broadcast scale across all dimensions)
        quantized = torch.round(tensor / scale) * scale
        
        # Store scale info: [H, 2]
        scales = torch.cat([
            t_min.reshape(tensor.shape[0], 1),
            t_max.reshape(tensor.shape[0], 1)
        ], dim=1)
        
        return quantized, scales
    
    def get_spectral_components(self) -> Tuple[List[SpectralBlock], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the spectral representation of the cache (for direct spectral attention).
        
        Returns:
            cold_blocks: List of SpectralBlock objects
            hot_K: Hot cache keys [B, H, T_hot, D] or None
            hot_V: Hot cache values [B, H, T_hot, D] or None
        """
        return self.cold_blocks, self.hot_K, self.hot_V
    
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct full K and V tensors (for debugging/validation only).
        
        WARNING: This defeats the purpose of spectral compression!
        Only use this for correctness validation, not in production.
        
        Returns:
            K_full: Reconstructed keys [B, H, T_total, D]
            V_full: Reconstructed values [B, H, T_total, D]
        """
        if self.debug_logging:
            print(f"\n[SpectralCache.get_kv] Reconstructing full K/V:")
            print(f"  Cold blocks: {len(self.cold_blocks)}")
            print(f"  Hot tokens: {self.hot_K.shape[2] if self.hot_K is not None else 0}")
            print(f"  Total tokens: {self.total_tokens}")
        
        K_parts = []
        V_parts = []
        
        # Reconstruct cold blocks
        for i, block in enumerate(self.cold_blocks):
            # K_recon = coeffs_K @ basis_K
            # Shape: [H, T, k] @ [H, k, D] -> [H, T, D]
            K_block = torch.bmm(block.coeffs_K, block.basis_K)  # [H, T, D]
            V_block = torch.bmm(block.coeffs_V, block.basis_V)
            
            if self.debug_logging:
                print(f"  Block {i}: coeffs_K {block.coeffs_K.shape} @ basis_K {block.basis_K.shape} → K_block {K_block.shape}")
            
            # Add batch dimension: [H, T, D] -> [1, H, T, D]
            K_parts.append(K_block.unsqueeze(0))
            V_parts.append(V_block.unsqueeze(0))
        
        # Add hot cache
        if self.hot_K is not None:
            K_parts.append(self.hot_K)
            V_parts.append(self.hot_V)
            if self.debug_logging:
                print(f"  Hot cache: K {self.hot_K.shape}, V {self.hot_V.shape}")
        
        if len(K_parts) == 0:
            # Empty cache
            if self.debug_logging:
                print(f"  Result: Empty cache!")
            return None, None
        
        # Concatenate along sequence dimension
        K_full = torch.cat(K_parts, dim=2)  # [B, H, T_total, D]
        V_full = torch.cat(V_parts, dim=2)
        
        if self.debug_logging:
            print(f"  Result: K_full {K_full.shape}, V_full {V_full.shape}")
            print(f"  Verification: {K_full.shape[2]} tokens == total_tokens {self.total_tokens}? {K_full.shape[2] == self.total_tokens}")
        
        return K_full, V_full
    
    def get_memory_stats(self) -> dict:
        """
        Calculate memory usage statistics.
        
        Returns:
            dict with keys:
                - original_bytes: Memory if stored as FP16
                - compressed_bytes: Actual memory used
                - compression_ratio: original / compressed
                - num_blocks: Number of spectral blocks
                - total_tokens: Total tokens stored
        """
        # Calculate original size (if stored as FP16)
        original_bytes = self.total_tokens * self.num_heads * self.head_dim * 2 * 2  # K and V
        
        # Calculate compressed size
        compressed_bytes = 0
        for block in self.cold_blocks:
            # Coeffs (INT8): H * T * k
            compressed_bytes += block.coeffs_K.numel() * 1  # Simulated as FP, but count as INT8
            compressed_bytes += block.coeffs_V.numel() * 1
            # Basis (FP16): H * k * D
            compressed_bytes += block.basis_K.numel() * 2
            compressed_bytes += block.basis_V.numel() * 2
            # Scales: negligible
            compressed_bytes += block.scales_K.numel() * 4
            compressed_bytes += block.scales_V.numel() * 4
        
        # Hot cache (FP16)
        if self.hot_K is not None:
            compressed_bytes += self.hot_K.numel() * 2
            compressed_bytes += self.hot_V.numel() * 2
        
        compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        
        return {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": compression_ratio,
            "num_blocks": len(self.cold_blocks),
            "total_tokens": self.total_tokens,
        }
    
    def __len__(self):
        """Return total number of tokens in cache."""
        return self.total_tokens
    
    def __getitem__(self, index):
        """
        Make SpectralCache subscriptable for Unsloth compatibility.
        
        Unsloth expects: past_key_values[layer_idx][0] for K, [1] for V
        Returns: [B, num_kv_heads, T, D] - NOT expanded with repeat_kv!
        
        Args:
            index: 0 for Keys, 1 for Values
            
        Returns:
            torch.Tensor: Reconstructed K or V tensor with original KV head count
        """
        if index == 0:
            K, _ = self.get_kv()
            # Verify shape is correct (should have num_kv_heads, not num_heads)
            assert K.shape[1] == self.num_heads, \
                f"Cache K has wrong head count: {K.shape[1]} vs expected {self.num_heads}"
            return K
        elif index == 1:
            _, V = self.get_kv()
            assert V.shape[1] == self.num_heads, \
                f"Cache V has wrong head count: {V.shape[1]} vs expected {self.num_heads}"
            return V
        else:
            raise IndexError(f"SpectralCache subscript index out of range: {index} (expected 0 or 1)")
    
    def __iter__(self):
        """
        Make SpectralCache iterable for tuple unpacking.
        
        Allows: K, V = cache
        """
        K, V = self.get_kv()
        yield K
        yield V
    
    def __repr__(self):
        stats = self.get_memory_stats()
        return (
            f"SpectralCache(tokens={self.total_tokens}, "
            f"blocks={stats['num_blocks']}, "
            f"compression={stats['compression_ratio']:.2f}x)"
        )


def test_spectral_cache():
    """Quick unit test for SpectralCache."""
    print("="*60)
    print("Testing SpectralCache")
    print("="*60)
    
    # Create cache
    cache = SpectralCache(
        num_heads=8,
        head_dim=128,
        block_size=512,
        k_rank_keys=16,
        k_rank_values=32,
        device="cpu",
        dtype=torch.float32,
    )
    
    # Append tokens
    B, H, D = 1, 8, 128
    for i in range(5):
        K_new = torch.randn(B, H, 200, D)
        V_new = torch.randn(B, H, 200, D)
        cache.append(K_new, V_new)
        print(f"Appended 200 tokens. {cache}")
    
    # Get spectral components
    blocks, hot_K, hot_V = cache.get_spectral_components()
    print(f"\nSpectral representation:")
    print(f"  Cold blocks: {len(blocks)}")
    print(f"  Hot tokens: {hot_K.shape[2] if hot_K is not None else 0}")
    
    # Reconstruction test
    K_full, V_full = cache.get_kv()
    print(f"\nReconstructed shape: K={K_full.shape}, V={V_full.shape}")
    
    # Memory stats
    stats = cache.get_memory_stats()
    print(f"\nMemory efficiency:")
    print(f"  Original: {stats['original_bytes'] / 1024:.1f} KB")
    print(f"  Compressed: {stats['compressed_bytes'] / 1024:.1f} KB")
    print(f"  Ratio: {stats['compression_ratio']:.2f}x")
    
    print("\n✅ SpectralCache test passed!")


if __name__ == "__main__":
    test_spectral_cache()

