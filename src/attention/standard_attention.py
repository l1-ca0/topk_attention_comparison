"""
Implements standard (vanilla) scaled dot-product attention and multi-head attention as described in 'Attention Is All You Need' (Vaswani et al., 2017).

This module provides the baseline attention mechanisms for comparison with Top-k attention in this repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .utils import scaled_dot_product_attention, max_neg_value, create_causal_mask


class StandardAttention(nn.Module):
    """
    Standard scaled dot-product attention mechanism.
    
    This implementation follows the original Transformer paper and serves as
    the baseline for comparison with Top-k attention.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialize standard attention.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of standard attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask [batch_size, seq_len_q, seq_len_k]
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights) where attention_weights is None
            if return_attention is False
        """
        output, attention_weights = scaled_dot_product_attention(
            query, key, value, mask, self.dropout if self.training else 0.0, is_causal, self.scale
        )
        
        if return_attention:
            return output, attention_weights
        return output, None


class StandardMultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention mechanism.
    
    Implements the multi-head attention as described in the original Transformer paper.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = True):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Linear layers for Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention mechanism
        self.attention = StandardAttention(self.d_k, dropout)
        
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
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k, seq_len_v = key.size(1), value.size(1)
        
        # Project to Q, K, V
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: [batch_size, n_heads, seq_len, d_k]
        
        # Adjust mask for multi-head attention
        if mask is not None:
            # Add head dimension: [batch_size, 1, seq_len_q, seq_len_k]
            mask = mask.unsqueeze(1)
        
        # Apply attention
        attention_output, attention_weights = self.attention(
            Q, K, V, mask, is_causal, return_attention
        )
        # attention_output: [batch_size, n_heads, seq_len_q, d_k]
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        if return_attention and attention_weights is not None:
            # Average attention weights across heads for visualization
            attention_weights = attention_weights.mean(dim=1)
            return output, attention_weights
        
        return output, None


class StandardTransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention and feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            activation: Activation function ("relu" or "gelu")
        """
        super().__init__()
        self.attention = StandardMultiHeadAttention(d_model, n_heads, dropout)
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
        Forward pass of transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(
            x, x, x, mask, is_causal, return_attention
        )
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attention_weights


def get_memory_usage(model: nn.Module, input_tensor: torch.Tensor) -> dict:
    """
    Estimate memory usage of a standard attention model.
    
    Args:
        model: The attention model
        input_tensor: Sample input tensor
        
    Returns:
        Dictionary with memory usage statistics
    """
    # This is a simplified estimation
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Estimate activation memory (rough approximation)
    batch_size, seq_len, d_model = input_tensor.shape
    if hasattr(model, 'n_heads'):
        n_heads = model.n_heads
        # Attention scores: [batch_size, n_heads, seq_len, seq_len]
        attention_memory = batch_size * n_heads * seq_len * seq_len * 4  # float32
    else:
        attention_memory = batch_size * seq_len * seq_len * 4
    
    return {
        'parameter_memory': param_memory,
        'estimated_attention_memory': attention_memory,
        'estimated_total_memory': param_memory + attention_memory,
        'memory_complexity': 'O(L²)' if seq_len > 0 else 'O(1)'
    } 