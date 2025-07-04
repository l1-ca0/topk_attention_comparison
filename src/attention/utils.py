"""
Utility functions for attention mechanisms in this repository.

Includes:
- Mask creation (causal masks)
- Top-k selection (memory-efficient)
- Chunking utilities
- Memory usage estimation
- Other helpers for both standard and top-k attention implementations
"""

import torch
import math
from typing import Optional, Tuple


def max_neg_value(tensor: torch.Tensor) -> float:
    """Get the maximum negative value for the tensor's dtype."""
    return -torch.finfo(tensor.dtype).max


def create_causal_mask(size: int, device: torch.device = None, dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """
    Create a causal mask for self-attention.
    
    Args:
        size: Size of the square mask
        device: Device to place the mask on
        dtype: Data type of the mask
        
    Returns:
        Causal mask where True indicates positions to mask
    """
    mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
    return mask


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard scaled dot-product attention.
    
    Args:
        query: Query tensor [batch_size, seq_len_q, d_model]
        key: Key tensor [batch_size, seq_len_k, d_model]
        value: Value tensor [batch_size, seq_len_v, d_model]
        mask: Optional mask tensor
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Optional scaling factor (default: 1/sqrt(d_model))
        
    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = query.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, max_neg_value(scores))
    
    # Apply causal mask if requested
    if is_causal:
        seq_len = scores.size(-1)
        causal_mask = create_causal_mask(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(causal_mask, max_neg_value(scores))
    
    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0.0:
        attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout_p, training=True)
    
    # Compute output
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def chunk_tensor(tensor: torch.Tensor, chunk_size: int, dim: int = -2) -> list:
    """
    Chunk a tensor along a specified dimension.
    
    Args:
        tensor: Input tensor to chunk
        chunk_size: Size of each chunk
        dim: Dimension to chunk along
        
    Returns:
        List of tensor chunks
    """
    if chunk_size <= 0:
        return [tensor]
    
    tensor_size = tensor.size(dim)
    chunks = []
    
    for i in range(0, tensor_size, chunk_size):
        end_idx = min(i + chunk_size, tensor_size)
        if dim == -2:
            chunk = tensor[:, i:end_idx]
        elif dim == -1:
            chunk = tensor[..., i:end_idx]
        else:
            # Use torch.narrow for general case
            chunk = torch.narrow(tensor, dim, i, end_idx - i)
        chunks.append(chunk)
    
    return chunks


def memory_efficient_topk(
    scores: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient top-k operation.
    
    Args:
        scores: Input tensor
        k: Number of top elements to select
        dim: Dimension along which to select top-k
        largest: If True, return largest elements
        sorted: If True, return sorted results
        
    Returns:
        Tuple of (top_k_values, top_k_indices)
    """
    if k >= scores.size(dim):
        # If k is larger than the dimension size, just sort the entire dimension
        return torch.sort(scores, dim=dim, descending=largest)
    
    return torch.topk(scores, k=k, dim=dim, largest=largest, sorted=sorted)


def estimate_memory_usage(batch_size: int, seq_len: int, d_model: int, n_heads: int, k: Optional[int] = None) -> dict:
    """
    Estimate memory usage for attention mechanisms.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        k: Top-k parameter (None for standard attention)
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    # Assume float32 (4 bytes per element)
    bytes_per_element = 4
    
    # Input tensors (Q, K, V)
    input_memory = 3 * batch_size * seq_len * d_model * bytes_per_element
    
    if k is None:
        # Standard attention
        # Attention scores matrix: [batch_size, n_heads, seq_len, seq_len]
        attention_scores_memory = batch_size * n_heads * seq_len * seq_len * bytes_per_element
        attention_weights_memory = attention_scores_memory  # Same size after softmax
        
        total_memory = input_memory + attention_scores_memory + attention_weights_memory
        
        return {
            'input_memory': input_memory,
            'attention_scores_memory': attention_scores_memory,
            'attention_weights_memory': attention_weights_memory,
            'total_memory': total_memory,
            'type': 'standard_attention'
        }
    else:
        # Top-k attention
        # Top-k scores: [batch_size, n_heads, seq_len, k]
        topk_scores_memory = batch_size * n_heads * seq_len * k * bytes_per_element
        topk_indices_memory = batch_size * n_heads * seq_len * k * 4  # Assuming int32 indices
        
        total_memory = input_memory + topk_scores_memory + topk_indices_memory
        
        return {
            'input_memory': input_memory,
            'topk_scores_memory': topk_scores_memory,
            'topk_indices_memory': topk_indices_memory,
            'total_memory': total_memory,
            'memory_reduction_ratio': (batch_size * n_heads * seq_len * seq_len * bytes_per_element) / topk_scores_memory,
            'type': 'topk_attention'
        } 