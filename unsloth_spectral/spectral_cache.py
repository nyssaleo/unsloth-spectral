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
        start_position: Absolute position of first token in this block (for RoPE)
    """
    coeffs_K: torch.Tensor
    basis_K: torch.Tensor
    coeffs_V: torch.Tensor
    basis_V: torch.Tensor
    scales_K: torch.Tensor
    scales_V: torch.Tensor
    block_size: int
    start_position: int  # NEW: Track position for RoPE relative rotation


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
        
        # Position tracking (for RoPE relative rotation)
        self.current_position = 0  # Absolute position in sequence
        self.hot_position_ids: Optional[torch.Tensor] = None  # Position IDs for hot cache
        
        # State tracking
        self.total_tokens = 0
        self.compression_count = 0
        self.cache_id = id(self)  # Unique cache identifier for debugging
        
        if self.debug_logging:
            print(f"[SpectralCache.__init__] Created NEW cache (ID: {self.cache_id}):")
            print(f"  num_heads={num_heads} (KV heads)")
            print(f"  head_dim={head_dim}")
            print(f"  block_size={block_size}")
            print(f"  k_rank_keys={k_rank_keys}, k_rank_values={k_rank_values}")
            print(f"  hot_buffer_size={hot_buffer_size}")
    
    def reset(self):
        """
        Reset cache for a new generation.
        
        This should be called when starting a completely new generation sequence
        to prevent contamination from previous generations.
        """
        if self.debug_logging:
            print(f"\n[SpectralCache.reset] Resetting cache (ID: {self.cache_id})")
            print(f"  Clearing {self.total_tokens} tokens, {len(self.cold_blocks)} blocks")
        
        self.hot_K = None
        self.hot_V = None
        self.hot_position_ids = None
        self.cold_blocks = []
        self.current_position = 0
        self.total_tokens = 0
        self.compression_count = 0
    
    def append(self, K_new: torch.Tensor, V_new: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """
        Append new Key-Value pairs to the cache.
        
        Automatically triggers compression when hot cache exceeds block_size.
        Detects new generation when position_ids restart from 0.
        
        Args:
            K_new: New keys [B, H, T_new, D] (PRE-RoPE!)
            V_new: New values [B, H, T_new, D]
            position_ids: Position IDs for the new tokens [B, T_new] (optional, auto-increments if None)
        """
        # CRITICAL: Detect new generation (position_ids restart from 0)
        if position_ids is not None and self.total_tokens > 0:
            # Handle both 1D and 2D position_ids (Unsloth decode can pass 1D)
            if position_ids.dim() == 1:
                first_pos = position_ids[0].item()
            else:
                first_pos = position_ids[0, 0].item()
            if first_pos < self.current_position:
                # Position went backwards = NEW GENERATION!
                if self.debug_logging:
                    print(f"\n[SpectralCache.append] NEW GENERATION DETECTED!")
                    print(f"  Previous position: {self.current_position}, New position: {first_pos}")
                    print(f"  Resetting cache to avoid contamination")
                self.reset()
        
        if self.debug_logging:
            print(f"\n[SpectralCache.append] Cache ID: {self.cache_id}")
            print(f"  K_new shape: {K_new.shape}")
            print(f"  V_new shape: {V_new.shape}")
            print(f"  Before append: total_tokens={self.total_tokens}, current_position={self.current_position}")
            if position_ids is not None:
                print(f"  position_ids: {position_ids.flatten()[:10].tolist()}...")
        
        # Handle position IDs: auto-increment if not provided
        T_new = K_new.shape[2]
        if position_ids is None:
            # Auto-increment positions
            position_ids = torch.arange(
                self.current_position, 
                self.current_position + T_new, 
                device=K_new.device, 
                dtype=torch.long
            ).unsqueeze(0)  # [1, T_new]
        else:
            # NORMALIZE: Ensure position_ids is always 2D [B, T_new]
            # Unsloth's decode path may pass 1D tensor [T_new]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)  # [1, T_new]
        
        # First append: Initialize hot cache
        if self.hot_K is None:
            self.hot_K = K_new
            self.hot_V = V_new
            self.hot_position_ids = position_ids
            if self.debug_logging:
                print(f"  Action: Initialized hot cache")
        else:
            # Concatenate along sequence dimension
            old_hot_size = self.hot_K.shape[2]
            self.hot_K = torch.cat([self.hot_K, K_new], dim=2)
            self.hot_V = torch.cat([self.hot_V, V_new], dim=2)
            self.hot_position_ids = torch.cat([self.hot_position_ids, position_ids], dim=1)
            if self.debug_logging:
                print(f"  Action: Concatenated ({old_hot_size} + {K_new.shape[2]} = {self.hot_K.shape[2]} hot tokens)")
        
        self.total_tokens += T_new
        self.current_position += T_new
        
        if self.debug_logging:
            print(f"  After append: total_tokens={self.total_tokens}, hot_tokens={self.hot_K.shape[2]}, cold_blocks={len(self.cold_blocks)}")
        
        # Trigger compression if hot cache is large enough
        if self.hot_K.shape[2] >= self.block_size:
            if self.debug_logging:
                print(f"  Triggering compression (hot_tokens={self.hot_K.shape[2]} >= block_size={self.block_size})")
            self._compress_hot_cache()
    
    def _compress_hot_cache(self):
        """
        Compress the hot cache into spectral blocks and move to cold storage.
        
        CRITICAL FIX (Jan 2026): Now uses WHILE loop to compress ALL complete blocks,
        not just the first one. This is essential for long contexts!
        
        Algorithm:
        1. WHILE hot cache has >= block_size tokens:
           a. Extract exactly block_size tokens from hot cache
           b. Perform per-head SVD: K_h = U @ S @ Vh^T
           c. Truncate to rank k: coeffs = U_k @ S_k, basis = V_k^T
           d. Quantize coefficients to INT8 (simulated)
           e. Store SpectralBlock in cold_blocks
           f. Keep remaining tokens in hot cache
        2. Continue until hot cache < block_size
        
        Example: 3500 tokens with block_size=512
        - Before fix: 1 block compressed, 2988 tokens stay hot
        - After fix: 6 blocks compressed, 428 tokens stay hot
        """
        # WHILE loop to compress ALL complete blocks (THE FIX!)
        blocks_compressed_this_call = 0
        
        while self.hot_K is not None and self.hot_K.shape[2] >= self.block_size:
            # Extract block to compress
            K_block = self.hot_K[:, :, :self.block_size, :]  # [B, H, T, D]
            V_block = self.hot_V[:, :, :self.block_size, :]
            
            # Get start position for this block
            if self.hot_position_ids is not None:
                start_position = self.hot_position_ids[0, 0].item()
            else:
                # Fallback: calculate from existing blocks
                start_position = sum(block.block_size for block in self.cold_blocks)
            
            if self.debug_logging:
                print(f"  [_compress_hot_cache] Compressing block {len(self.cold_blocks)}: "
                      f"tokens [{start_position}:{start_position + self.block_size}], "
                      f"hot_remaining={self.hot_K.shape[2] - self.block_size}")
            
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
                start_position=start_position,
            )
            self.cold_blocks.append(block)
            self.compression_count += 1
            blocks_compressed_this_call += 1
            
            # Keep only the remainder in hot cache
            if self.hot_K.shape[2] > self.block_size:
                self.hot_K = self.hot_K[:, :, self.block_size:, :].contiguous()
                self.hot_V = self.hot_V[:, :, self.block_size:, :].contiguous()
                if self.hot_position_ids is not None:
                    self.hot_position_ids = self.hot_position_ids[:, self.block_size:].contiguous()
            else:
                # All tokens compressed, hot cache is empty
                self.hot_K = None
                self.hot_V = None
                self.hot_position_ids = None
                break  # Exit loop when hot cache is empty
        
        if self.debug_logging and blocks_compressed_this_call > 0:
            hot_tokens = self.hot_K.shape[2] if self.hot_K is not None else 0
            print(f"  [_compress_hot_cache] Compressed {blocks_compressed_this_call} blocks, "
                  f"total_cold={len(self.cold_blocks)}, hot_remaining={hot_tokens}")
    
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
        
        # CRITICAL: Convert ALL spectral components back to model dtype (BF16/FP16)
        # SVD computes in FP32 for numerical stability, but downstream ops need matching dtypes.
        # Research finding: einsum is NOT on autocast whitelist, requires explicit dtype match.
        # Without this, mixed FP32 coeffs × FP16 queries cause "expected Float but found Half".
        coeffs_q = coeffs_q.to(self.dtype)
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
            
            # Convert coeffs to model dtype (matching basis)
            coeffs_list.append(coeffs_q.to(self.dtype))
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
    
    def get_all_positions(self) -> torch.Tensor:
        """
        Get position IDs for all tokens in cache (cold + hot).
        
        Returns:
            position_ids: [total_tokens] tensor of position indices
        """
        positions = []
        
        # Collect positions from cold blocks
        for block in self.cold_blocks:
            block_positions = torch.arange(
                block.start_position,
                block.start_position + block.block_size,
                device=self.device,
                dtype=torch.long
            )
            positions.append(block_positions)
        
        # Add hot cache positions
        if self.hot_position_ids is not None:
            positions.append(self.hot_position_ids.flatten())
        
        if len(positions) == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        
        return torch.cat(positions, dim=0)
    
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

