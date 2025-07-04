"""
Basic Usage Example: Standard vs Top-k Attention

This example demonstrates the basic usage of both standard and top-k attention
mechanisms and shows how they can be used as drop-in replacements.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import StandardMultiHeadAttention, TopKMultiHeadAttention
from src.benchmarks import MemoryBenchmark


def basic_attention_comparison():
    """Demonstrate basic usage of standard vs top-k attention."""
    
    print("=" * 60)
    print("BASIC ATTENTION COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Model configuration
    batch_size = 2
    seq_len = 512
    d_model = 768
    n_heads = 12
    k = 64  # Number of top attention scores to keep
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Top-k parameter: {k}")
    print()
    
    # Create sample input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Input: [batch_size, seq_len, d_model]
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Initialize attention mechanisms
    print("\nInitializing attention mechanisms...")
    
    # Standard attention
    std_attention = StandardMultiHeadAttention(d_model, n_heads).to(device)
    
    # Top-k attention (drop-in replacement!)
    topk_attention = TopKMultiHeadAttention(d_model, n_heads, k).to(device)
    
    print(f"Standard attention parameters: {sum(p.numel() for p in std_attention.parameters()):,}")
    print(f"Top-k attention parameters: {sum(p.numel() for p in topk_attention.parameters()):,}")
    
    # Forward pass comparison
    print("\nPerforming forward passes...")
    
    with torch.no_grad():
        # Standard attention
        output_std, _ = std_attention(x, x, x)
        
        # Top-k attention
        output_topk, _ = topk_attention(x, x, x)
    
    print(f"Standard attention output shape: {output_std.shape}")
    print(f"Top-k attention output shape: {output_topk.shape}")
    
    # Compare outputs
    output_diff = torch.norm(output_std - output_topk).item()
    relative_diff = output_diff / torch.norm(output_std).item()
    
    print(f"\nOutput comparison:")
    print(f"  Absolute difference: {output_diff:.6f}")
    print(f"  Relative difference: {relative_diff:.6f}")
    print(f"  Max absolute difference: {torch.max(torch.abs(output_std - output_topk)).item():.6f}")
    
    # Memory usage estimation
    print(f"\nMemory usage estimation:")
    
    # Standard attention memory: O(L²)
    std_memory_mb = (batch_size * n_heads * seq_len * seq_len * 4) / 1024**2
    # Top-k attention memory: O(L·k)
    topk_memory_mb = (batch_size * n_heads * seq_len * k * 4) / 1024**2
    
    print(f"  Standard attention (estimated): {std_memory_mb:.2f} MB")
    print(f"  Top-k attention (estimated): {topk_memory_mb:.2f} MB")
    print(f"  Memory reduction factor: {std_memory_mb / topk_memory_mb:.2f}x")
    
    return output_std, output_topk


def attention_with_masking():
    """Demonstrate attention with masking."""
    
    print("\n" + "=" * 60)
    print("ATTENTION WITH MASKING EXAMPLE")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 256
    d_model = 512
    n_heads = 8
    k = 32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create input and attention mask
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Create a simple padding mask (mask out last 50 tokens)
    mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
    mask[:, :, -50:] = True  # Mask out last 50 positions
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Initialize attention mechanisms
    std_attention = StandardMultiHeadAttention(d_model, n_heads).to(device)
    topk_attention = TopKMultiHeadAttention(d_model, n_heads, k).to(device)
    
    with torch.no_grad():
        # Standard attention with mask
        output_std_masked, _ = std_attention(x, x, x, mask=mask)
        
        # Top-k attention with mask
        output_topk_masked, _ = topk_attention(x, x, x, mask=mask)
    
    print(f"Standard attention (masked) output shape: {output_std_masked.shape}")
    print(f"Top-k attention (masked) output shape: {output_topk_masked.shape}")
    
    # Compare with unmasked
    output_std_unmasked, _ = std_attention(x, x, x)
    output_topk_unmasked, _ = topk_attention(x, x, x)
    
    mask_effect_std = torch.norm(output_std_masked - output_std_unmasked).item()
    mask_effect_topk = torch.norm(output_topk_masked - output_topk_unmasked).item()
    
    print(f"\nMask effect comparison:")
    print(f"  Standard attention mask effect: {mask_effect_std:.6f}")
    print(f"  Top-k attention mask effect: {mask_effect_topk:.6f}")


def causal_attention_example():
    """Demonstrate causal (autoregressive) attention."""
    
    print("\n" + "=" * 60)
    print("CAUSAL ATTENTION EXAMPLE")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 128
    d_model = 256
    n_heads = 4
    k = 16
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Causal mask: Lower triangular (each position can only attend to previous positions)")
    
    # Initialize attention mechanisms
    std_attention = StandardMultiHeadAttention(d_model, n_heads).to(device)
    topk_attention = TopKMultiHeadAttention(d_model, n_heads, k).to(device)
    
    with torch.no_grad():
        # Causal attention
        output_std_causal, _ = std_attention(x, x, x, is_causal=True)
        output_topk_causal, _ = topk_attention(x, x, x, is_causal=True)
        
        # Non-causal for comparison
        output_std_normal, _ = std_attention(x, x, x, is_causal=False)
        output_topk_normal, _ = topk_attention(x, x, x, is_causal=False)
    
    causal_diff_std = torch.norm(output_std_causal - output_std_normal).item()
    causal_diff_topk = torch.norm(output_topk_causal - output_topk_normal).item()
    
    print(f"Causal vs non-causal difference:")
    print(f"  Standard attention: {causal_diff_std:.6f}")
    print(f"  Top-k attention: {causal_diff_topk:.6f}")
    
    # Compare causal outputs between attention types
    causal_output_diff = torch.norm(output_std_causal - output_topk_causal).item()
    print(f"  Causal outputs difference (std vs top-k): {causal_output_diff:.6f}")


def memory_benchmark_example():
    """Run a quick memory benchmark."""
    
    print("\n" + "=" * 60)
    print("MEMORY BENCHMARK EXAMPLE")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return
    
    try:
        from src.benchmarks.memory_benchmark import quick_memory_comparison
        
        print("Running quick memory comparison...")
        print("This may take a few minutes...")
        
        # Run with smaller parameters for the example
        benchmark = quick_memory_comparison(
            seq_lengths=[256, 512, 1024],
            k_values=[32, 64],
            d_model=512,
            n_heads=8
        )
        
        print("\nMemory benchmark completed!")
        
    except Exception as e:
        print(f"Memory benchmark failed: {e}")
        print("You can run individual benchmarks using the MemoryBenchmark class")


def performance_tips():
    """Show performance tips and best practices."""
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TIPS")
    print("=" * 60)
    
    tips = [
        "1. Choose k based on your memory constraints and accuracy requirements",
        "   - For seq_len=512: try k=32-64",
        "   - For seq_len=1024: try k=64-128", 
        "   - For seq_len=2048+: try k=128-256",
        "",
        "2. Use chunking for very long sequences:",
        "   topk_attention = TopKMultiHeadAttention(d_model, n_heads, k, chunk_size=1024)",
        "",
        "3. Top-k attention works best for:",
        "   - Long sequence processing (>512 tokens)",
        "   - Memory-constrained environments",
        "   - Fine-tuning existing models",
        "",
        "4. Consider standard attention for:",
        "   - Very short sequences (<128 tokens)",
        "   - When absolute maximum accuracy is critical",
        "",
        "5. Memory reduction scales with sequence length:",
        "   - Reduction factor ≈ seq_len / k",
        "   - For seq_len=2048, k=64: ~32x memory reduction"
    ]
    
    for tip in tips:
        print(tip)


def main():
    """Run all examples."""
    
    print("Top-k Attention vs Standard Attention - Basic Usage Examples")
    print("=" * 80)
    
    try:
        # Basic comparison
        basic_attention_comparison()
        
        # Masking example
        attention_with_masking()
        
        # Causal attention
        causal_attention_example()
        
        # Performance tips
        performance_tips()
        
        # Memory benchmark (optional)
        memory_benchmark_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main() 