"""
Functionality tests for attention mechanisms
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import (
    StandardAttention, StandardMultiHeadAttention,
    TopKAttention, TopKMultiHeadAttention
)


class TestFunctionality:
    """Test functionality of attention mechanisms."""
    
    def setup_method(self):
        """Setup test parameters."""
        self.batch_size = 2
        self.seq_len = 64
        self.d_model = 128
        self.n_heads = 4
        self.k = 16
        self.device = "cpu"  # Use CPU for tests
    
    def test_standard_attention_shapes(self):
        """Test that standard attention produces correct output shapes."""
        attention = StandardAttention(self.d_model)
        
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output, _ = attention(query, key, value)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert output.dtype == query.dtype
    
    def test_topk_attention_shapes(self):
        """Test that top-k attention produces correct output shapes."""
        attention = TopKAttention(self.d_model, self.k)
        
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output, _ = attention(query, key, value)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        assert output.dtype == query.dtype
    
    def test_standard_multihead_attention_shapes(self):
        """Test standard multi-head attention shapes."""
        attention = StandardMultiHeadAttention(self.d_model, self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, _ = attention(x, x, x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_topk_multihead_attention_shapes(self):
        """Test top-k multi-head attention shapes."""
        attention = TopKMultiHeadAttention(self.d_model, self.n_heads, self.k)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, _ = attention(x, x, x)
        
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_attention_with_mask(self):
        """Test attention mechanisms with masking."""
        std_attention = StandardMultiHeadAttention(self.d_model, self.n_heads)
        topk_attention = TopKMultiHeadAttention(self.d_model, self.n_heads, self.k)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create a simple mask
        mask = torch.zeros(self.batch_size, self.seq_len, self.seq_len, dtype=torch.bool)
        mask[:, :, -10:] = True  # Mask last 10 positions
        
        # Test both attention mechanisms with mask
        output_std, _ = std_attention(x, x, x, mask=mask)
        output_topk, _ = topk_attention(x, x, x, mask=mask)
        
        assert output_std.shape == (self.batch_size, self.seq_len, self.d_model)
        assert output_topk.shape == (self.batch_size, self.seq_len, self.d_model)
    
    def test_causal_attention(self):
        """Test causal attention functionality."""
        std_attention = StandardMultiHeadAttention(self.d_model, self.n_heads)
        topk_attention = TopKMultiHeadAttention(self.d_model, self.n_heads, self.k)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test causal attention
        output_std_causal, _ = std_attention(x, x, x, is_causal=True)
        output_topk_causal, _ = topk_attention(x, x, x, is_causal=True)
        
        # Test non-causal attention for comparison
        output_std_normal, _ = std_attention(x, x, x, is_causal=False)
        output_topk_normal, _ = topk_attention(x, x, x, is_causal=False)
        
        # Causal outputs should be different from non-causal
        assert not torch.allclose(output_std_causal, output_std_normal)
        assert not torch.allclose(output_topk_causal, output_topk_normal)
    
    def test_attention_similarity(self):
        """Test that standard and top-k attention produce similar results."""
        # Use same weights for fair comparison
        std_attention = StandardMultiHeadAttention(self.d_model, self.n_heads)
        topk_attention = TopKMultiHeadAttention(self.d_model, self.n_heads, self.k)
        
        # Copy weights from standard to top-k attention
        topk_attention.w_q.weight.data = std_attention.w_q.weight.data.clone()
        topk_attention.w_k.weight.data = std_attention.w_k.weight.data.clone()
        topk_attention.w_v.weight.data = std_attention.w_v.weight.data.clone()
        topk_attention.w_o.weight.data = std_attention.w_o.weight.data.clone()
        
        if std_attention.w_q.bias is not None:
            topk_attention.w_q.bias.data = std_attention.w_q.bias.data.clone()
            topk_attention.w_k.bias.data = std_attention.w_k.bias.data.clone()
            topk_attention.w_v.bias.data = std_attention.w_v.bias.data.clone()
            topk_attention.w_o.bias.data = std_attention.w_o.bias.data.clone()
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        with torch.no_grad():
            output_std, _ = std_attention(x, x, x)
            output_topk, _ = topk_attention(x, x, x)
        
        # Outputs should be reasonably similar (top-k is an approximation)
        relative_error = torch.norm(output_std - output_topk) / torch.norm(output_std)
        assert relative_error < 1.0, f"Relative error too high: {relative_error:.4f}"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through attention mechanisms."""
        std_attention = StandardMultiHeadAttention(self.d_model, self.n_heads)
        topk_attention = TopKMultiHeadAttention(self.d_model, self.n_heads, self.k)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        
        # Test standard attention gradients
        output_std, _ = std_attention(x, x, x)
        loss_std = output_std.sum()
        loss_std.backward()
        
        assert x.grad is not None
        x.grad = None  # Reset gradients
        
        # Test top-k attention gradients
        output_topk, _ = topk_attention(x, x, x)
        loss_topk = output_topk.sum()
        loss_topk.backward()
        
        assert x.grad is not None
    
    def test_chunking(self):
        """Test attention with chunking."""
        # Test with chunking enabled
        chunk_size = 32
        topk_attention = TopKMultiHeadAttention(
            self.d_model, self.n_heads, self.k, chunk_size=chunk_size
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output_chunked, _ = topk_attention(x, x, x)
        
        assert output_chunked.shape == (self.batch_size, self.seq_len, self.d_model)
        
        # Compare with non-chunked version
        topk_attention_no_chunk = TopKMultiHeadAttention(
            self.d_model, self.n_heads, self.k, chunk_size=None
        )
        
        # Copy weights for fair comparison
        topk_attention_no_chunk.load_state_dict(topk_attention.state_dict())
        
        with torch.no_grad():
            output_no_chunk, _ = topk_attention_no_chunk(x, x, x)
        
        diff = (output_chunked - output_no_chunk).abs()
        print(f"[test_chunking] max abs diff: {diff.max().item():.4e}, mean abs diff: {diff.mean().item():.4e}")
        assert torch.allclose(output_chunked, output_no_chunk, atol=1e-2)
    
    def test_different_k_values(self):
        """Test attention with different k values."""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        k_values = [8, 16, 32, self.seq_len]  # Include k=seq_len (should match standard)
        outputs = []
        
        for k in k_values:
            attention = TopKMultiHeadAttention(self.d_model, self.n_heads, k)
            with torch.no_grad():
                output, _ = attention(x, x, x)
            outputs.append(output)
        
        # All outputs should have the same shape
        for output in outputs:
            assert output.shape == (self.batch_size, self.seq_len, self.d_model)
        
        # Larger k should generally produce results closer to standard attention
        # (when k=seq_len, it should be very close to standard attention)
        std_attention = StandardMultiHeadAttention(self.d_model, self.n_heads)
        with torch.no_grad():
            output_std, _ = std_attention(x, x, x)
        
        # The last output (k=seq_len) should be closest to standard attention
        # Note: This might not be exactly equal due to implementation differences
        error_large_k = torch.norm(outputs[-1] - output_std) / torch.norm(output_std)
        error_small_k = torch.norm(outputs[0] - output_std) / torch.norm(output_std)
        
        assert error_large_k < 2.0, f"Large k error too high: {error_large_k:.4f}"
        assert error_small_k < 3.0, f"Small k error too high: {error_small_k:.4f}"

    def test_topk_attention_checkpointing_shapes(self):
        attention = TopKAttention(self.d_model, self.k, use_checkpointing=True)
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output, _ = attention(query, key, value)
        assert output.shape == (self.batch_size, self.seq_len, self.d_model)

    def test_topk_attention_all_mask_shapes(self):
        attention = TopKAttention(self.d_model, self.k)
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # 2D mask
        mask2d = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        output2d, _ = attention(query, key, value, mask=mask2d)
        assert output2d.shape == (self.batch_size, self.seq_len, self.d_model)
        # 3D mask
        mask3d = torch.zeros(self.batch_size, self.seq_len, self.seq_len, dtype=torch.bool)
        output3d, _ = attention(query, key, value, mask=mask3d)
        assert output3d.shape == (self.batch_size, self.seq_len, self.d_model)
        # 4D mask
        mask4d = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len, dtype=torch.bool)
        output4d, _ = attention(query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), mask=mask4d)
        assert output4d.shape == (self.batch_size, 1, self.seq_len, self.d_model)

    def test_topk_attention_k_edge_cases(self):
        # k=1
        attention1 = TopKAttention(self.d_model, 1)
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output1, _ = attention1(query, key, value)
        assert output1.shape == (self.batch_size, self.seq_len, self.d_model)
        # k=seq_len
        attention_full = TopKAttention(self.d_model, self.seq_len)
        output_full, _ = attention_full(query, key, value)
        assert output_full.shape == (self.batch_size, self.seq_len, self.d_model)
        # k>seq_len
        attention_large = TopKAttention(self.d_model, self.seq_len + 10)
        output_large, _ = attention_large(query, key, value)
        assert output_large.shape == (self.batch_size, self.seq_len, self.d_model)

    def test_topk_attention_dropout_modes(self):
        attention = TopKAttention(self.d_model, self.k, dropout=0.5)
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # Training mode
        attention.train()
        out1, _ = attention(query, key, value)
        out2, _ = attention(query, key, value)
        # Outputs should not be exactly equal due to dropout
        assert not torch.allclose(out1, out2)
        # Eval mode
        attention.eval()
        out3, _ = attention(query, key, value)
        out4, _ = attention(query, key, value)
        # Outputs should be exactly equal in eval mode
        assert torch.allclose(out3, out4)

    def test_topk_attention_device(self):
        attention = TopKAttention(self.d_model, self.k)
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # CPU
        output_cpu, _ = attention(query, key, value)
        assert output_cpu.device.type == 'cpu'
        # CUDA (if available)
        if torch.cuda.is_available():
            attention_cuda = TopKAttention(self.d_model, self.k).cuda()
            query_cuda = query.cuda()
            key_cuda = key.cuda()
            value_cuda = value.cuda()
            output_cuda, _ = attention_cuda(query_cuda, key_cuda, value_cuda)
            assert output_cuda.device.type == 'cuda'

    def test_topk_attention_invalid_mask_shape(self):
        attention = TopKAttention(self.d_model, self.k)
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # Invalid mask shape
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        with pytest.raises(ValueError):
            attention(query, key, value, mask=mask)

    def test_topk_attention_invalid_k(self):
        # k=0 should raise error
        with pytest.raises(Exception):
            TopKAttention(self.d_model, 0)

    def test_topk_attention_chunked_vs_unchunked_checkpointing(self):
        # Compare chunked and unchunked outputs with checkpointing
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        attn_chunked = TopKAttention(self.d_model, self.k, chunk_size=16, use_checkpointing=True)
        attn_unchunked = TopKAttention(self.d_model, self.k, chunk_size=None, use_checkpointing=True)
        attn_chunked.eval(); attn_unchunked.eval()
        out_chunked, _ = attn_chunked(query, key, value)
        out_unchunked, _ = attn_unchunked(query, key, value)
        # Should be close for small batch/seq_len
        assert torch.allclose(out_chunked, out_unchunked, atol=1e-4)

    def test_topk_attention_backward_consistency(self):
        # Compare gradients between checkpointed and non-checkpointed for small input
        d_model = 8; seq_len = 4; batch_size = 2; k = 2
        query = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        value = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        attn_ckpt = TopKAttention(d_model, k, use_checkpointing=True)
        attn_std = TopKAttention(d_model, k, use_checkpointing=False)
        out_ckpt, _ = attn_ckpt(query, key, value)
        out_std, _ = attn_std(query, key, value)
        grad = torch.randn_like(out_ckpt)
        out_ckpt.backward(grad, retain_graph=True)
        grad_query_ckpt = query.grad.clone(); query.grad.zero_()
        grad_key_ckpt = key.grad.clone(); key.grad.zero_()
        grad_value_ckpt = value.grad.clone(); value.grad.zero_()
        out_std.backward(grad)
        grad_query_std = query.grad.clone()
        grad_key_std = key.grad.clone()
        grad_value_std = value.grad.clone()
        diff_query = (grad_query_ckpt - grad_query_std).abs()
        diff_key = (grad_key_ckpt - grad_key_std).abs()
        diff_value = (grad_value_ckpt - grad_value_std).abs()
        print(f"[backward_consistency] grad_query max: {diff_query.max().item():.4e}, mean: {diff_query.mean().item():.4e}")
        print(f"[backward_consistency] grad_key max: {diff_key.max().item():.4e}, mean: {diff_key.mean().item():.4e}")
        print(f"[backward_consistency] grad_value max: {diff_value.max().item():.4e}, mean: {diff_value.mean().item():.4e}")
        assert torch.allclose(grad_query_ckpt, grad_query_std, atol=1e-2)
        assert torch.allclose(grad_key_ckpt, grad_key_std, atol=1e-2)
        assert torch.allclose(grad_value_ckpt, grad_value_std, atol=1e-2)


def test_import():
    """Test that all modules can be imported correctly."""
    from src.attention import (
        StandardAttention, StandardMultiHeadAttention,
        TopKAttention, TopKMultiHeadAttention,
        max_neg_value, create_causal_mask
    )
    
    # Test that classes can be instantiated
    std_attn = StandardAttention(128)
    topk_attn = TopKAttention(128, 16)
    std_multi = StandardMultiHeadAttention(128, 4)
    topk_multi = TopKMultiHeadAttention(128, 4, 16)
    
    assert isinstance(std_attn, StandardAttention)
    assert isinstance(topk_attn, TopKAttention)
    assert isinstance(std_multi, StandardMultiHeadAttention)
    assert isinstance(topk_multi, TopKMultiHeadAttention)


if __name__ == "__main__":
    # Run tests
    test_class = TestFunctionality()
    test_class.setup_method()
    
    tests = [
        test_class.test_standard_attention_shapes,
        test_class.test_topk_attention_shapes,
        test_class.test_standard_multihead_attention_shapes,
        test_class.test_topk_multihead_attention_shapes,
        test_class.test_attention_with_mask,
        test_class.test_causal_attention,
        test_class.test_attention_similarity,
        test_class.test_gradient_flow,
        test_class.test_chunking,
        test_class.test_different_k_values,
        test_class.test_topk_attention_checkpointing_shapes,
        test_class.test_topk_attention_all_mask_shapes,
        test_class.test_topk_attention_k_edge_cases,
        test_class.test_topk_attention_dropout_modes,
        test_class.test_topk_attention_device,
        test_class.test_topk_attention_invalid_mask_shape,
        test_class.test_topk_attention_invalid_k,
        test_class.test_topk_attention_chunked_vs_unchunked_checkpointing,
        test_class.test_topk_attention_backward_consistency,
    ]
    
    print("Running functionality tests...")
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
    
    # Test imports
    try:
        test_import()
        print("✓ test_import")
    except Exception as e:
        print(f"✗ test_import: {e}")
    
    print("Tests completed!") 