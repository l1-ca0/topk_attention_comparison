"""
Implements memory-efficient Top-k attention and multi-head Top-k attention as described in 'Memory-efficient Transformers via Top-k Attention' (Gupta et al., 2021).

Includes:
- Memory-efficient top-k selection
- Query chunking for reduced memory usage
- Custom autograd function with input checkpointing
- PyTorch-only optimizations for benchmarking against standard attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.amp import autocast

from .utils import max_neg_value, create_causal_mask, chunk_tensor, memory_efficient_topk


class TopKAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function for top-k attention with memory-efficient backward pass.
    Handles both 3D (single-head) and 4D (multi-head) input.
    """
    
    @staticmethod
    def forward(ctx, query, key, value, k, mask=None, scale=None, chunk_size=None):
        """
        Forward pass of top-k attention.
        
        Args:
            query: Query tensor [batch_size, n_heads, seq_len_q, d_k]
            key: Key tensor [batch_size, n_heads, seq_len_k, d_k]
            value: Value tensor [batch_size, n_heads, seq_len_v, d_k]
            k: Number of top attention scores to keep
            mask: Optional attention mask
            scale: Scaling factor (default: 1/sqrt(d_k))
            chunk_size: Size of query chunks for memory efficiency
            
        Returns:
            Output tensor [batch_size, n_heads, seq_len_q, d_k]
        """
        # Only supports 4D input (multi-head)
        assert query.dim() == 4, "TopKAttentionFunction only supports 4D input (multi-head)"
        
        batch_size, n_heads, seq_len_q, d_k = query.shape
        seq_len_k = key.shape[2]
        
        if scale is None:
            scale = 1.0 / math.sqrt(d_k)
        
        # Determine chunk size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = seq_len_q
        
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len_q, chunk_size):
            end_idx = min(i + chunk_size, seq_len_q)
            query_chunk = query[:, :, i:end_idx, :]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(query_chunk, key.transpose(-2, -1)) * scale
            
            # Robust mask handling: support 2D, 3D, 4D
            chunk_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    chunk_mask = mask[:, :, i:end_idx, :]
                elif mask.dim() == 3:
                    chunk_mask = mask[:, i:end_idx, :]
                    if chunk_mask.dim() == 3:
                        chunk_mask = chunk_mask.unsqueeze(1)  # Add head dim
                elif mask.dim() == 2:
                    # [seq_len_q, seq_len_k] or [seq_len, seq_len]
                    chunk_mask = mask[i:end_idx, :].unsqueeze(0).unsqueeze(0)  # [1,1,chunk,seq_len_k]
                else:
                    raise ValueError(f"Unsupported mask dim: {mask.dim()}")
                scores = scores.masked_fill(chunk_mask, max_neg_value(scores))
            
            # Get top-k scores and indices
            if k >= seq_len_k:
                topk_scores = scores
                topk_indices = torch.arange(seq_len_k, device=scores.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                    batch_size, n_heads, end_idx - i, -1
                )
            else:
                topk_scores, topk_indices = memory_efficient_topk(scores, k, dim=-1, largest=True, sorted=False)
            
            topk_weights = F.softmax(topk_scores, dim=-1)
            expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k)
            expanded_values = value.unsqueeze(2).expand(-1, -1, end_idx - i, -1, -1)
            topk_values = torch.gather(expanded_values, dim=3, index=expanded_indices)
            chunk_output = torch.sum(topk_weights.unsqueeze(-1) * topk_values, dim=3)
            output[:, :, i:end_idx, :] = chunk_output
        
        ctx.save_for_backward(query, key, value)
        ctx.k = k
        ctx.mask = mask
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.is_4d = True
        ctx.seq_len_q = seq_len_q
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with input checkpointing to save memory.
        """
        # Only supports 4D input (multi-head)
        query, key, value = ctx.saved_tensors
        k = ctx.k
        mask = ctx.mask
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        # Only 4D shape logic
        batch_size, n_heads, _, d_k = query.shape
        seq_len_k = key.shape[2]
        
        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        
        if chunk_size is None or chunk_size <= 0:
            chunk_size = query.shape[-2]
        
        # Only clone/detach the query chunk; use key and value directly with requires_grad_(True)
        key_ = key.detach().requires_grad_(True)
        value_ = value.detach().requires_grad_(True)
        key_.retain_grad()
        value_.retain_grad()
        
        for i in range(0, query.shape[-2], chunk_size):
            end_idx = min(i + chunk_size, query.shape[-2])
            query_chunk = query[:, :, i:end_idx, :].clone().detach().requires_grad_(True)
            query_chunk.retain_grad()
            grad_output_chunk = grad_output[:, :, i:end_idx, :]
            
            with torch.enable_grad():
                scores = torch.matmul(query_chunk, key_.transpose(-2, -1)) * scale
                # Robust mask handling: support 2D, 3D, 4D
                chunk_mask = None
                if mask is not None:
                    if mask.dim() == 4:
                        chunk_mask = mask[:, :, i:end_idx, :]
                    elif mask.dim() == 3:
                        chunk_mask = mask[:, i:end_idx, :]
                        if chunk_mask.dim() == 3:
                            chunk_mask = chunk_mask.unsqueeze(1)
                    elif mask.dim() == 2:
                        chunk_mask = mask[i:end_idx, :].unsqueeze(0).unsqueeze(0)
                    else:
                        raise ValueError(f"Unsupported mask dim: {mask.dim()}")
                    scores = scores.masked_fill(chunk_mask, max_neg_value(scores))
                
                if k >= seq_len_k:
                    topk_scores = scores
                    topk_indices = torch.arange(seq_len_k, device=scores.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                        batch_size, n_heads, end_idx - i, -1
                    )
                else:
                    topk_scores, topk_indices = memory_efficient_topk(scores, k, dim=-1, largest=True, sorted=False)
                
                topk_weights = F.softmax(topk_scores, dim=-1)
                expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_k)
                expanded_values = value_.unsqueeze(2).expand(-1, -1, end_idx - i, -1, -1)
                topk_values = torch.gather(expanded_values, dim=3, index=expanded_indices)
                chunk_output = torch.sum(topk_weights.unsqueeze(-1) * topk_values, dim=3)
                chunk_output.backward(grad_output_chunk)
                
                # Accumulate gradients, check for None
                if query_chunk.grad is not None:
                    grad_query[:, :, i:end_idx, :] = query_chunk.grad
        
        # Accumulate key/value gradients after all chunks
        if key_.grad is not None:
            grad_key += key_.grad
        if value_.grad is not None:
            grad_value += value_.grad
        
        return grad_query, grad_key, grad_value, None, None, None, None


class TopKAttention(nn.Module):
    """
    Top-k Attention mechanism for memory-efficient transformers.
    Now supports explicit dtype (float32, float16, bfloat16), efficient in-place masking, adaptive chunk size, and fully batched top-k.
    """
    
    def __init__(
        self,
        d_model: int,
        k: int,
        dropout: float = 0.1,
        chunk_size: Optional[int] = None,
        use_checkpointing: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize top-k attention.
        Args:
            d_model: Model dimension
            k: Number of top attention scores to keep
            dropout: Dropout probability
            chunk_size: Query chunk size for memory efficiency (None for adaptive; will use min(1024, seq_len_q))
            use_checkpointing: If True, use custom autograd with input checkpointing (default: False)
            dtype: Data type for all computations (default: torch.float32; can be torch.float16 or torch.bfloat16)
        """
        super().__init__()
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.d_model = d_model
        self.k = k
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(d_model)
        self.use_checkpointing = use_checkpointing
        self.dtype = dtype
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
        use_autocast: bool = True,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of top-k attention.
        Args:
            query: Query tensor [batch_size, n_heads, seq_len_q, d_k]
            key: Key tensor [batch_size, n_heads, seq_len_k, d_k]
            value: Value tensor [batch_size, n_heads, seq_len_v, d_k]
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights (not supported)
            use_autocast: If True, use torch.amp.autocast for mixed precision (default: True)
            dtype: Data type for all computations (overrides module default if not None)
        """
        dtype = dtype or self.dtype
        if use_autocast and dtype in (torch.float16, torch.bfloat16):
            with autocast('cuda', dtype=dtype):
                return self._forward_impl(query, key, value, mask, is_causal, return_attention, dtype)
        else:
            return self._forward_impl(query, key, value, mask, is_causal, return_attention, dtype)

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of top-k attention.
        Uses custom autograd with input checkpointing only for 4D input (multi-head),
        otherwise uses standard PyTorch implementation (with chunking if specified).
        """
        # Cast all inputs to the target dtype
        query = query.to(dtype)
        key = key.to(dtype)
        value = value.to(dtype)
        # Input validation for mask shape
        if mask is not None:
            if mask.dim() not in (2, 3, 4):
                raise ValueError("Mask must be 2D, 3D, or 4D tensor")
            mask = mask.to(torch.bool)
        # Handle causal masking
        if is_causal:
            seq_len = query.size(-2)
            causal_mask = create_causal_mask(seq_len, device=query.device, dtype=torch.bool)
            if mask is not None:
                mask = mask | causal_mask
            else:
                mask = causal_mask

        # Adaptive chunk size
        chunk_size = self.chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = min(1024, query.shape[-2])

        # Only use custom autograd for 4D input (multi-head)
        if query.dim() == 4 and self.use_checkpointing:
            return TopKAttentionFunction.apply(query, key, value, self.k, mask, self.scale, chunk_size), None
        # Fallback to standard implementation for 3D input (single-head)
        if chunk_size is not None and chunk_size > 0:
            output = self._chunked_forward(query, key, value, mask, chunk_size, dtype)
        else:
            output = self._standard_forward(query, key, value, mask, dtype)
        return output, None
    
    def _standard_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Standard forward pass without chunking. All ops are fully batched/vectorized.
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # Apply mask (in-place if possible)
        if mask is not None:
            scores = scores.masked_fill(mask, max_neg_value(scores))
        # Get top-k scores and indices (fully batched)
        seq_len_k = scores.size(-1)
        if self.k >= seq_len_k:
            attention_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attention_weights = F.dropout(attention_weights, p=self.dropout)
            output = torch.matmul(attention_weights, value)
        else:
            topk_scores, topk_indices = memory_efficient_topk(scores, self.k, dim=-1, largest=True, sorted=False)
            topk_weights = F.softmax(topk_scores, dim=-1)
            if self.training and self.dropout > 0:
                topk_weights = F.dropout(topk_weights, p=self.dropout)
            # Gather corresponding values (fully batched)
            if value.dim() == 3:
                expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, value.size(-1))
                expanded_values = value.unsqueeze(-3).expand(-1, scores.size(-2), -1, -1)
            else:
                expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, value.size(-1))
                expanded_values = value.unsqueeze(-3).expand(-1, -1, scores.size(-2), -1, -1)
            topk_values = torch.gather(expanded_values, dim=-2, index=expanded_indices)
            output = torch.sum(topk_weights.unsqueeze(-1) * topk_values, dim=-2)
        return output
    
    def _chunked_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1024,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Memory-efficient chunked forward pass. All ops are fully batched/vectorized.
        """
        batch_size = query.size(0)
        seq_len_q = query.size(-2)
        # Determine output shape
        if query.dim() == 3:
            output_shape = (batch_size, seq_len_q, value.size(-1))
        else:
            output_shape = (batch_size, query.size(1), seq_len_q, value.size(-1))
        output = torch.zeros(output_shape, device=query.device, dtype=dtype)
        # Process queries in chunks
        for i in range(0, seq_len_q, chunk_size):
            end_idx = min(i + chunk_size, seq_len_q)
            if query.dim() == 3:
                query_chunk = query[:, i:end_idx, :]
            else:
                query_chunk = query[:, :, i:end_idx, :]
            # Handle mask for this chunk
            chunk_mask = None
            if mask is not None:
                if mask.dim() == 3:
                    chunk_mask = mask[:, i:end_idx, :]
                elif mask.dim() == 4:
                    chunk_mask = mask[:, :, i:end_idx, :]
                else:
                    chunk_mask = mask[i:end_idx, :]
            # Compute attention for this chunk
            chunk_output = self._standard_forward(query_chunk, key, value, chunk_mask, dtype)
            # Store result
            if query.dim() == 3:
                output[:, i:end_idx, :] = chunk_output
            else:
                output[:, :, i:end_idx, :] = chunk_output
        return output


class TopKMultiHeadAttention(nn.Module):
    """
    Top-k Multi-Head Attention mechanism with explicit dtype, adaptive chunking, and fully batched ops.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        k: int,
        dropout: float = 0.1,
        bias: bool = True,
        chunk_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        use_checkpointing: bool = False
    ):
        """
        Initialize top-k multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            k: Number of top attention scores to keep per head
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            chunk_size: Query chunk size for memory efficiency (None for adaptive)
            dtype: Data type for all computations (default: torch.float32; can be torch.float16 or torch.bfloat16)
            use_checkpointing: If True, use custom autograd with input checkpointing (default: False)
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k = k
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.use_checkpointing = use_checkpointing
        
        # Linear layers for Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Top-k attention mechanism
        self.attention = TopKAttention(self.d_k, k, dropout, chunk_size, use_checkpointing=use_checkpointing, dtype=dtype)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
        use_autocast: bool = True,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of top-k multi-head attention.
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights (not supported)
            use_autocast: If True, use torch.amp.autocast for mixed precision (default: True)
            dtype: Data type for all computations (overrides module default if not None)
        """
        dtype = dtype or self.dtype
        if use_autocast and dtype in (torch.float16, torch.bfloat16):
            with autocast('cuda', dtype=dtype):
                return self._forward_impl(query, key, value, mask, is_causal, return_attention, use_autocast, dtype)
        else:
            return self._forward_impl(query, key, value, mask, is_causal, return_attention, use_autocast, dtype)

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
        use_autocast: bool = True,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of top-k multi-head attention.
        Uses custom autograd with input checkpointing only for 4D input (multi-head),
        otherwise uses standard PyTorch implementation (with chunking if specified).
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k, seq_len_v = key.size(1), value.size(1)
        
        # Project to Q, K, V
        Q = self.w_q(query).to(dtype)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key).to(dtype)    # [batch_size, seq_len_k, d_model]
        V = self.w_v(value).to(dtype)  # [batch_size, seq_len_v, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: [batch_size, n_heads, seq_len, d_k]
        
        # Adjust mask for multi-head attention
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        # Apply top-k attention
        attention_output, _ = self.attention(Q, K, V, mask, is_causal, return_attention, use_autocast, dtype)
        # attention_output: [batch_size, n_heads, seq_len_q, d_k]
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        return output, None


class TopKTransformerBlock(nn.Module):
    """
    Transformer block with top-k attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        k: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        chunk_size: Optional[int] = None
    ):
        """
        Initialize top-k transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            k: Number of top attention scores to keep
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            activation: Activation function
            chunk_size: Query chunk size for memory efficiency
        """
        super().__init__()
        self.attention = TopKMultiHeadAttention(d_model, n_heads, k, dropout, chunk_size=chunk_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of top-k transformer block.
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask, is_causal, return_attention)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, None


def get_topk_memory_usage(
    model: nn.Module,
    input_tensor: torch.Tensor,
    k: int
) -> dict:
    """
    Estimate memory usage of a top-k attention model.
    
    Args:
        model: The top-k attention model
        input_tensor: Sample input tensor
        k: Top-k parameter
        
    Returns:
        Dictionary with memory usage statistics
    """
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    batch_size, seq_len, d_model = input_tensor.shape
    if hasattr(model, 'n_heads'):
        n_heads = model.n_heads
        # Top-k attention scores: [batch_size, n_heads, seq_len, k]
        topk_memory = batch_size * n_heads * seq_len * k * 4  # float32
        # Standard attention would be: batch_size * n_heads * seq_len * seq_len * 4
        standard_memory = batch_size * n_heads * seq_len * seq_len * 4
        memory_reduction = standard_memory / topk_memory if topk_memory > 0 else float('inf')
    else:
        topk_memory = batch_size * seq_len * k * 4
        standard_memory = batch_size * seq_len * seq_len * 4
        memory_reduction = standard_memory / topk_memory if topk_memory > 0 else float('inf')
    
    return {
        'parameter_memory': param_memory,
        'estimated_topk_memory': topk_memory,
        'estimated_standard_memory': standard_memory,
        'memory_reduction_factor': memory_reduction,
        'estimated_total_memory': param_memory + topk_memory,
        'memory_complexity': f'O(L·k) where k={k}',
        'k_value': k,
        'seq_len': seq_len
    } 