"""
Unsloth Integration: Monkey-patching for Spectral Cache

This module provides seamless integration with Unsloth's optimized Mistral implementation.
It monkey-patches the MistralAttention_fast_forward method to use SpectralCache instead
of standard tuple-based KV cache.

Key Design: The patched forward method is a drop-in replacement that:
1. Detects if past_key_value is a SpectralCache
2. Converts tuple caches to SpectralCache automatically
3. Uses spectral_attention_forward for computation
4. Returns compatible outputs for model.generate()
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from .spectral_cache import SpectralCache
from .spectral_attention import spectral_attention_forward
import math


def create_spectral_forward(
    original_forward,
    block_size: int = 512,
    k_rank_keys: int = 16,
    k_rank_values: int = 32,
    hot_buffer_size: int = 64,
    use_spectral_attention: bool = True,
    debug_logging: bool = False,
):
    """
    Factory function that creates a spectral-enabled forward method.
    
    This wraps the original MistralAttention forward to inject spectral cache logic.
    
    Args:
        original_forward: Original self.forward method
        block_size: Tokens per compressed block
        k_rank_keys: Spectral rank for Keys
        k_rank_values: Spectral rank for Values
        hot_buffer_size: Number of recent tokens kept uncompressed
        use_spectral_attention: If True, use direct spectral attention (no reconstruction)
        
    Returns:
        spectral_forward: Modified forward method with spectral cache
    """
    
    def spectral_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Modified forward pass with spectral cache integration.
        
        This method:
        1. Computes Q, K, V projections (standard)
        2. Applies RoPE to Q, K (standard)
        3. Manages SpectralCache (new)
        4. Computes attention using spectral method (new)
        5. Returns output in Unsloth-compatible format (standard)
        """
        
        # Get architecture parameters (compatible with Unsloth/Mistral)
        # Mistral stores these in config, not directly on attention module
        if hasattr(self, 'config'):
            num_heads = self.config.num_attention_heads
            num_key_value_heads = self.config.num_key_value_heads
            head_dim = self.config.hidden_size // num_heads
            hidden_size = self.config.hidden_size
            num_key_value_groups = num_heads // num_key_value_heads
        else:
            # Fallback for other architectures
            num_heads = self.num_heads
            num_key_value_heads = self.num_key_value_heads
            head_dim = self.head_dim
            hidden_size = self.hidden_size
            num_key_value_groups = self.num_key_value_groups
        
        bsz, q_len, _ = hidden_states.size()
        
        # 🔍 DIAGNOSTIC: What type is past_key_value when it arrives?
        if debug_logging:
            print(f"\n{'='*80}")
            print(f"[SpectralForward] NEW FORWARD PASS - Layer {getattr(self, 'layer_idx', 'N/A')}")
            print(f"  q_len: {q_len} (1=decode, >1=prefill)")
            print(f"  past_key_value type: {type(past_key_value)}")
            if past_key_value is not None:
                if isinstance(past_key_value, SpectralCache):
                    print(f"  ✓ Cache IS SpectralCache, total_tokens={past_key_value.total_tokens}")
                elif isinstance(past_key_value, (tuple, list)):
                    print(f"  ⚠️  Cache is tuple/list, len={len(past_key_value)}")
                    if len(past_key_value) > 0:
                        print(f"      First element type: {type(past_key_value[0])}")
                else:
                    print(f"  ❌ Cache is unexpected type: {type(past_key_value)}")
            else:
                print(f"  ❌ Cache is None (will create fresh cache)")
        
        # 1. QKV Projection (Standard Unsloth path)
        # ============================================
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [B, H, T, D]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        if debug_logging:
            print(f"[SpectralForward] After QKV projection:")
            print(f"  Q shape: {query_states.shape} (num_heads={num_heads})")
            print(f"  K shape: {key_states.shape} (num_kv_heads={num_key_value_heads})")
            print(f"  V shape: {value_states.shape} (num_kv_heads={num_key_value_heads})")
        
        # 2. RoPE (Rotary Positional Embeddings) - Unsloth's method
        # ===========================================================
        # Calculate total sequence length (current + past cache)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, SpectralCache):
                kv_seq_len += past_key_value.total_tokens
            elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                kv_seq_len += past_key_value[0].shape[-2]
        
        if debug_logging:
            print(f"[SpectralForward] RoPE: kv_seq_len={kv_seq_len}")
        
        # Extend RoPE embedding cache if needed
        if hasattr(self, 'rotary_emb') and hasattr(self.rotary_emb, 'extend_rope_embedding'):
            # Unsloth's optimized RoPE
            self.rotary_emb.extend_rope_embedding(value_states, seq_len=kv_seq_len)
            cos, sin = self.rotary_emb.get_cached(kv_seq_len, query_states.device.index)
            
            # Use Unsloth's fast_rope_embedding if available
            try:
                from unsloth.models.llama import fast_rope_embedding
                query_states, key_states = fast_rope_embedding(query_states, key_states, cos, sin, position_ids)
            except ImportError:
                # Fallback to standard RoPE
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            # Fallback for non-Unsloth models (standard transformers)
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if debug_logging:
            print(f"[SpectralForward] After RoPE:")
            print(f"  Q shape: {query_states.shape}")
            print(f"  K shape: {key_states.shape}")
        
        # 3. Spectral Cache Management - CRITICAL: Store BEFORE GQA expansion!
        # =====================================================================
        
        # Check if past_key_value is already a SpectralCache
        if not isinstance(past_key_value, SpectralCache):
            # First call or tuple-based cache - create SpectralCache
            # CRITICAL: Use num_key_value_heads, NOT num_heads!
            cache = SpectralCache(
                num_heads=num_key_value_heads,  # FIXED: Was num_heads (32), should be num_kv_heads (8)
                head_dim=head_dim,
                block_size=block_size,
                k_rank_keys=k_rank_keys,
                k_rank_values=k_rank_values,
                hot_buffer_size=hot_buffer_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                debug_logging=debug_logging,  # Pass debug flag to cache
            )
            
            # If there was a previous tuple cache, initialize with it
            if past_key_value is not None and isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) == 2:
                    past_K, past_V = past_key_value
                    if debug_logging:
                        print(f"[SpectralForward] Initializing cache with past K/V:")
                        print(f"  Past K shape: {past_K.shape}")
                        print(f"  Past V shape: {past_V.shape}")
                    cache.append(past_K, past_V)
            
            past_key_value = cache
        
        # CRITICAL BUG FIX: Append to cache BEFORE repeat_kv expansion!
        # Unsloth expects cache to store [B, num_kv_heads, T, D], not [B, num_heads, T, D]
        if debug_logging:
            print(f"[SpectralForward] Appending to cache (BEFORE repeat_kv):")
            print(f"  K shape: {key_states.shape} (num_kv_heads={num_key_value_heads})")
            print(f"  V shape: {value_states.shape}")
        
        past_key_value.append(key_states, value_states)
        
        # 4. Attention Computation
        # ==========================
        # NOTE: GQA expansion happens AFTER retrieving full context from cache,
        # not here! We need to expand the COMPLETE history (past + current),
        # not just the current tokens.
        
        if debug_logging:
            print(f"[SpectralForward] Before attention:")
            print(f"  Q shape: {query_states.shape}")
            print(f"  Cache total tokens: {past_key_value.total_tokens}")
            print(f"  Note: K/V will be retrieved and expanded from cache during attention")
        
        if use_spectral_attention and past_key_value.total_tokens > block_size:
            # Use spectral attention (no reconstruction)
            # This is the core optimization!
            
            # Note: spectral_attention currently only supports single-token decode
            # For prefill (q_len > 1), we fall back to standard attention
            if q_len == 1:
                attn_output = spectral_attention_forward(
                    Q=query_states,
                    cache=past_key_value,
                    attention_mask=attention_mask,
                    scale=1.0 / math.sqrt(head_dim),
                )
            else:
                # Prefill: use standard attention with reconstructed K/V
                if debug_logging:
                    print(f"\n[SpectralForward] Prefill path - retrieving full context from cache")
                
                K_full, V_full = past_key_value.get_kv()
                
                if debug_logging:
                    print(f"[SpectralForward] Retrieved from cache:")
                    print(f"  K_full: {K_full.shape if K_full is not None else 'None'}")
                    print(f"  V_full: {V_full.shape if V_full is not None else 'None'}")
                
                # CRITICAL FIX: Expand K/V to match Q heads AFTER getting full context
                if num_key_value_groups > 1:
                    if debug_logging:
                        print(f"[SpectralForward] Expanding K_full/V_full for GQA (groups={num_key_value_groups}):")
                        print(f"  Before expansion: K_full {K_full.shape}, V_full {V_full.shape}")
                    K_full = repeat_kv(K_full, num_key_value_groups)
                    V_full = repeat_kv(V_full, num_key_value_groups)
                    if debug_logging:
                        print(f"  After expansion: K_full {K_full.shape}, V_full {V_full.shape}")
                        print(f"  Q shape for comparison: {query_states.shape}")
                        print(f"  Head dimension match: Q[{query_states.shape[1]}] == K[{K_full.shape[1]}]? {query_states.shape[1] == K_full.shape[1]}")
                
                attn_weights = torch.matmul(query_states, K_full.transpose(2, 3)) / math.sqrt(head_dim)
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, V_full)
        else:
            # Standard attention (for short contexts or validation)
            if debug_logging:
                print(f"\n[SpectralForward] Standard attention path - retrieving full context from cache")
            
            K_full, V_full = past_key_value.get_kv()
            
            if debug_logging:
                print(f"[SpectralForward] Retrieved from cache:")
                print(f"  K_full: {K_full.shape if K_full is not None else 'None'}")
                print(f"  V_full: {V_full.shape if V_full is not None else 'None'}")
            
            # CRITICAL FIX: Expand K/V to match Q heads AFTER getting full context
            if num_key_value_groups > 1:
                if debug_logging:
                    print(f"[SpectralForward] Expanding K_full/V_full for GQA (groups={num_key_value_groups}):")
                    print(f"  Before expansion: K_full {K_full.shape}, V_full {V_full.shape}")
                K_full = repeat_kv(K_full, num_key_value_groups)
                V_full = repeat_kv(V_full, num_key_value_groups)
                if debug_logging:
                    print(f"  After expansion: K_full {K_full.shape}, V_full {V_full.shape}")
                    print(f"  Q shape for comparison: {query_states.shape}")
                    print(f"  Head dimension match: Q[{query_states.shape[1]}] == K[{K_full.shape[1]}]? {query_states.shape[1] == K_full.shape[1]}")
            
            attn_weights = torch.matmul(query_states, K_full.transpose(2, 3)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, V_full)
        
        # 5. Output Projection (Standard)
        # =================================
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if debug_logging:
            print(f"\n[SpectralForward] Forward pass complete:")
            print(f"  Final attn_output: {attn_output.shape}")
            print(f"  Returning cache with {past_key_value.total_tokens} total tokens")
            print(f"  Cache state: {len(past_key_value.cold_blocks)} cold blocks, {past_key_value.hot_K.shape[2] if past_key_value.hot_K is not None else 0} hot tokens")
            print(f"  🔍 Returning cache type: {type(past_key_value)}")
            print(f"  🔍 use_cache flag: {use_cache}")
            print(f"={'='*80}\n")
        
        # Store cache reference for get_cache_stats()
        if use_cache and isinstance(past_key_value, SpectralCache):
            self._spectral_cache = past_key_value
        
        return attn_output, None, past_key_value if use_cache else None
    
    return spectral_forward


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary positional embeddings (RoPE).
    
    Fallback implementation for non-Unsloth models.
    Unsloth models should use fast_rope_embedding instead.
    """
    # Unsqueeze for broadcasting if needed
    # cos, sin shape: [seq_len, head_dim] or [batch, seq_len, head_dim]
    # q, k shape: [batch, num_heads, seq_len, head_dim]
    
    # Handle position_ids if provided
    if position_ids is not None:
        # Select the appropriate cos/sin values for each position
        # Reshape position_ids to index into cos/sin
        # This is a simplified version - actual implementation may vary
        pass
    
    # Ensure cos and sin have the right shape
    if cos.dim() == 2:
        # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    # Rotate half dimensions
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads for Grouped Query Attention (GQA).
    
    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_spectral_forward_inference(
    block_size: int = 512,
    k_rank_keys: int = 16,
    k_rank_values: int = 32,
    hot_buffer_size: int = 64,
    use_spectral_attention: bool = True,
    debug_logging: bool = False,
):
    """
    Creates a spectral-enabled inference forward function for decode steps.
    
    This replaces Unsloth's LlamaAttention_fast_forward_inference to use
    spectral compression during single-token decode.
    
    Args:
        block_size: Tokens per compressed block
        k_rank_keys: Spectral rank for Keys
        k_rank_values: Spectral rank for Values
        hot_buffer_size: Number of recent tokens kept uncompressed
        use_spectral_attention: If True, use direct spectral attention
        debug_logging: Enable detailed logging
        
    Returns:
        spectral_forward_inference: Modified inference function
    """
    
    def spectral_forward_inference(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]],
        position_ids: torch.LongTensor,
        do_prefill: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Fast inference using spectral KV cache for single-token decode.
        
        This is the decode-path equivalent of spectral_forward.
        Instead of reconstructing the full K/V cache, it performs attention
        directly in the spectral space using dual projection.
        """
        if debug_logging:
            print(f"\n{'='*80}\n[SpectralInference] DECODE STEP - Layer {getattr(self, 'layer_idx', 'N/A')}")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  past_key_value type: {type(past_key_value)}")
            print(f"  do_prefill: {do_prefill}")
        
        bsz, q_len, hidden_dim = hidden_states.size()
        assert q_len == 1, f"Inference path expects q_len=1, got {q_len}"
        
        # Get attention parameters
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.head_dim
        num_key_value_groups = num_heads // num_key_value_heads
        
        # QKV projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        if debug_logging:
            print(f"  After QKV projection:")
            print(f"    Q: {query_states.shape}")
            print(f"    K: {key_states.shape}")
            print(f"    V: {value_states.shape}")
        
        # Initialize or get cache
        if not isinstance(past_key_value, SpectralCache):
            if debug_logging:
                print(f"  Creating new SpectralCache (do_prefill={do_prefill})")
            
            cache = SpectralCache(
                num_heads=num_key_value_heads,
                head_dim=head_dim,
                block_size=block_size,
                k_rank_keys=k_rank_keys,
                k_rank_values=k_rank_values,
                hot_buffer_size=hot_buffer_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                debug_logging=debug_logging,
            )
            
            # If we received a tuple cache, initialize from it
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                past_K, past_V = past_key_value
                if debug_logging:
                    print(f"  Initializing from tuple: K {past_K.shape}, V {past_V.shape}")
                cache.append(past_K, past_V)
            
            past_key_value = cache
        
        # Calculate kv_seq_len for RoPE
        kv_seq_len = key_states.shape[-2] + past_key_value.total_tokens
        
        if debug_logging:
            print(f"  RoPE: kv_seq_len={kv_seq_len}")
        
        # Apply RoPE (Unsloth's method)
        if hasattr(self, 'rotary_emb') and hasattr(self.rotary_emb, 'extend_rope_embedding'):
            self.rotary_emb.extend_rope_embedding(value_states, seq_len=kv_seq_len)
            cos, sin = self.rotary_emb.get_cached(kv_seq_len, query_states.device.index)
            
            try:
                from unsloth.models.llama import fast_rope_embedding
                query_states, key_states = fast_rope_embedding(query_states, key_states, cos, sin, position_ids)
            except ImportError:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Append new K/V to cache (BEFORE GQA expansion!)
        if debug_logging:
            print(f"  Appending to cache: K {key_states.shape}, V {value_states.shape}")
        
        past_key_value.append(key_states, value_states)
        
        # Now perform attention
        if use_spectral_attention and past_key_value.total_tokens > block_size:
            # Use spectral attention (direct projection)
            if debug_logging:
                print(f"  Using spectral attention (total_tokens={past_key_value.total_tokens})")
            
            attn_output = spectral_attention_forward(
                Q=query_states,
                cache=past_key_value,
                attention_mask=attention_mask,
                scale=1.0 / math.sqrt(head_dim),
                debug_logging=debug_logging,
            )
        else:
            # Standard attention with reconstructed K/V
            if debug_logging:
                print(f"  Using standard attention (total_tokens={past_key_value.total_tokens})")
            
            K_full, V_full = past_key_value.get_kv()
            
            # GQA expansion (after cache retrieval)
            if num_key_value_groups > 1:
                if debug_logging:
                    print(f"  Expanding for GQA: {K_full.shape} → groups={num_key_value_groups}")
                K_full = repeat_kv(K_full, num_key_value_groups)
                V_full = repeat_kv(V_full, num_key_value_groups)
            
            # Attention computation
            attn_weights = torch.matmul(query_states, K_full.transpose(2, 3)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, V_full)
        
        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output)
        
        if debug_logging:
            print(f"  Output: {attn_output.shape}")
            print(f"  Returning SpectralCache with {past_key_value.total_tokens} tokens")
            print(f"{'='*80}\n")
        
        # Return output and updated cache (as tuple for Unsloth compatibility)
        return attn_output, past_key_value
    
    return spectral_forward_inference


def patch_unsloth_attention(
    model,
    block_size: int = 512,
    k_rank_keys: int = 16,
    k_rank_values: int = 32,
    hot_buffer_size: int = 64,
    use_spectral_attention: bool = True,
    verbose: bool = True,
    debug_logging: bool = False,
):
    """
    Monkey-patch all attention layers in an Unsloth model to use SpectralCache.
    
    This modifies the model in-place, replacing the forward methods of all
    MistralAttention layers with spectral-enabled versions.
    
    Args:
        model: Unsloth FastLanguageModel
        block_size: Tokens per compressed block
        k_rank_keys: Spectral rank for Keys
        k_rank_values: Spectral rank for Values  
        hot_buffer_size: Recent tokens kept uncompressed
        use_spectral_attention: Use direct spectral attention (recommended: True)
        verbose: Print patching confirmation
        
    Returns:
        model: Modified model (in-place)
    """
    
    num_patched = 0
    
    # Patch all transformer layers
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            # Get the original forward method
            original_forward = layer.self_attn.forward
            
            # Create spectral version
            spectral_fwd = create_spectral_forward(
                original_forward=original_forward,
                block_size=block_size,
                k_rank_keys=k_rank_keys,
                k_rank_values=k_rank_values,
                hot_buffer_size=hot_buffer_size,
                use_spectral_attention=use_spectral_attention,
                debug_logging=debug_logging,
            )
            
            # Monkey-patch (bind to instance)
            layer.self_attn.forward = spectral_fwd.__get__(layer.self_attn, type(layer.self_attn))
            num_patched += 1
    
    # CRITICAL: Also patch the INFERENCE function for decode path!
    # Unsloth uses a separate fast_forward_inference function for single-token decode
    # that bypasses the regular forward. We need to patch that too.
    try:
        import unsloth.models.llama as llama_module
        
        # Create the spectral inference function
        spectral_inference_fn = create_spectral_forward_inference(
            block_size=block_size,
            k_rank_keys=k_rank_keys,
            k_rank_values=k_rank_values,
            hot_buffer_size=hot_buffer_size,
            use_spectral_attention=use_spectral_attention,
            debug_logging=debug_logging,
        )
        
        # Monkey-patch the module-level function
        llama_module.LlamaAttention_fast_forward_inference = spectral_inference_fn
        
        if verbose:
            print(f"✅ Patched decode inference path (LlamaAttention_fast_forward_inference)")
    except Exception as e:
        print(f"⚠️  Warning: Could not patch inference function: {e}")
        print(f"   Decode steps may not use spectral cache correctly.")
    
    if verbose:
        print(f"✅ Patched {num_patched} attention layers with SpectralCache")
        print(f"   Config: block_size={block_size}, k_K={k_rank_keys}, k_V={k_rank_values}")
        print(f"   Spectral Attention: {'Enabled' if use_spectral_attention else 'Disabled (validation mode)'}")
    
    return model


def get_cache_stats(model):
    """
    Extract cache statistics from all layers that have been used.
    
    Note: Only collects stats from layers that have processed at least one forward pass
    with spectral cache enabled.
    
    Returns:
        dict: Aggregated cache statistics including:
            - total_tokens: Total tokens across all layers
            - total_blocks: Total compressed blocks
            - compression_ratio: Average compression ratio
            - per_layer_stats: List of per-layer statistics
    """
    total_tokens = 0
    total_blocks = 0
    total_original_bytes = 0
    total_compressed_bytes = 0
    per_layer_stats = []
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_spectral_cache'):
            cache = layer.self_attn._spectral_cache
            if isinstance(cache, SpectralCache):
                stats = cache.get_memory_stats()
                
                total_tokens += stats['total_tokens']
                total_blocks += stats['num_blocks']
                total_original_bytes += stats['original_bytes']
                total_compressed_bytes += stats['compressed_bytes']
                
                per_layer_stats.append({
                    'layer_idx': layer_idx,
                    'tokens': stats['total_tokens'],
                    'blocks': stats['num_blocks'],
                    'compression': stats['compression_ratio'],
                })
    
    return {
        "total_tokens": total_tokens,
        "total_blocks": total_blocks,
        "total_original_bytes": total_original_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "compression_ratio": total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0,
        "layers_with_cache": len(per_layer_stats),
        "per_layer_stats": per_layer_stats,
    }


if __name__ == "__main__":
    print("="*60)
    print("Unsloth Spectral Integration Module")
    print("="*60)
    print("\nThis module provides monkey-patching for Unsloth models.")
    print("Usage:")
    print("  from unsloth_spectral import patch_unsloth_attention")
    print("  model = FastLanguageModel.from_pretrained(...)")
    print("  patch_unsloth_attention(model)")
    print("  # Now model.generate() uses spectral cache!")
    print("="*60)

