#!/usr/bin/env python3
"""
comprehensive_benchmark.py

Main comprehensive benchmark script for Top-k vs Standard Attention.
- Benchmarks memory, speed, accuracy, and scaling for both standard and Top-k attention.
- Robust OOM handling, large batch/seq_len support, and detailed printouts.
- Results are saved to timestamped JSON files for easy analysis and plotting.
- This script supersedes all previous benchmark scripts.

Usage:
    python experiments/comprehensive_benchmark.py --tag full
    # or with --tag memory, --tag speed, --tag accuracy, --tag scaling

"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import argparse
from torch.amp import autocast

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from attention import StandardMultiHeadAttention, TopKMultiHeadAttention
from attention.utils import estimate_memory_usage

# Benchmark settings (match paper and main experiments)
SEQ_LENGTHS = [1024, 2048, 4096, 8192]
K_VALUES = [8, 16, 32, 64, 128, 256]
D_MODEL = 512
N_HEADS = 8
BATCH_SIZE_MEM = 1
BATCH_SIZE_SPEED = 1
NUM_RUNS = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Speed experiment uses smaller batch/seq_len/k to avoid OOM
SEQ_LENGTHS_SPEED = [1024, 2048, 4096]
K_VALUES_SPEED = [8, 16, 32, 64, 128]


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_peak_memory(module, x):
    clear_cuda()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        if torch.cuda.is_available():
            with autocast('cuda', dtype=torch.float16):
                _ = module(x, x, x, use_autocast=True) if hasattr(module, 'forward') and 'use_autocast' in module.forward.__code__.co_varnames else module(x, x, x)
        else:
            _ = module(x, x, x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    else:
        import psutil
        process = psutil.Process()
        peak = process.memory_info().rss / 1024**2
    return peak


def memory_experiment():
    print("Running memory experiments...")
    memory_results = []
    for seq_len in SEQ_LENGTHS:
        print(f"  Testing sequence length: {seq_len}")
        # Standard attention
        std_peaks = []
        for _ in range(NUM_RUNS):
            x = torch.randn(BATCH_SIZE_MEM, seq_len, D_MODEL, device=DEVICE)
            module = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
            try:
                if torch.cuda.is_available():
                    with autocast('cuda', dtype=torch.float16):
                        std_peaks.append(measure_peak_memory(module, x))
                else:
                    std_peaks.append(measure_peak_memory(module, x))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"    OOM at seq_len={seq_len} (standard), skipping...")
                    torch.cuda.empty_cache()
                    std_peaks.append(float('nan'))
                    continue
                else:
                    raise
        std_peak = np.nanmean(std_peaks)
        std_theoretical = estimate_memory_usage(BATCH_SIZE_MEM, seq_len, D_MODEL, N_HEADS)['total_memory'] / 1024**2
        for k in K_VALUES:
            if k >= seq_len:
                continue
            topk_peaks = []
            for _ in range(NUM_RUNS):
                x = torch.randn(BATCH_SIZE_MEM, seq_len, D_MODEL, device=DEVICE)
                module = TopKMultiHeadAttention(D_MODEL, N_HEADS, k).to(DEVICE)
                try:
                    if torch.cuda.is_available():
                        with autocast('cuda', dtype=torch.float16):
                            topk_peaks.append(measure_peak_memory(module, x))
                    else:
                        topk_peaks.append(measure_peak_memory(module, x))
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"    OOM at seq_len={seq_len}, k={k} (topk), skipping...")
                        torch.cuda.empty_cache()
                        topk_peaks.append(float('nan'))
                        continue
                    else:
                        raise
            topk_peak = np.nanmean(topk_peaks)
            topk_theoretical = estimate_memory_usage(BATCH_SIZE_MEM, seq_len, D_MODEL, N_HEADS, k)['total_memory'] / 1024**2
            theoretical_reduction = std_theoretical / topk_theoretical
            actual_reduction = std_peak / topk_peak if topk_peak > 0 else float('nan')
            memory_results.append({
                'seq_len': seq_len,
                'k': k,
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'batch_size': BATCH_SIZE_MEM,
                'std_memory': std_peak,
                'topk_memory': topk_peak,
                'theoretical_reduction': theoretical_reduction,
                'actual_reduction': actual_reduction,
                'memory_ratio': topk_peak / std_peak if std_peak > 0 else 0
            })
            print(f"    k={k}: std_peak={std_peak:.2f}MB, topk_peak={topk_peak:.2f}MB, "
                  f"theoretical={theoretical_reduction:.2f}x, actual={actual_reduction:.2f}x")
    return memory_results


def speed_experiment():
    print("Running speed experiments...")
    speed_results = []
    for seq_len in SEQ_LENGTHS_SPEED:
        print(f"  Testing sequence length: {seq_len}")
        # Standard attention speed
        std_times = []
        for _ in range(NUM_RUNS):
            x = torch.randn(BATCH_SIZE_SPEED, seq_len, D_MODEL, device=DEVICE)
            module = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
            clear_cuda()
            for _ in range(3):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with autocast('cuda', dtype=torch.float16):
                            _ = module(x, x, x)
                    else:
                        _ = module(x, x, x)
            times = []
            for _ in range(NUM_RUNS):
                start_time = time.time()
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with autocast('cuda', dtype=torch.float16):
                            _ = module(x, x, x)
                    else:
                        _ = module(x, x, x)
                times.append(time.time() - start_time)
            std_times.append(np.mean(times))
        std_time = np.mean(std_times)
        std_std = np.std(std_times)
        for k in K_VALUES_SPEED:
            if k >= seq_len:
                continue
            topk_times = []
            for _ in range(NUM_RUNS):
                x = torch.randn(BATCH_SIZE_SPEED, seq_len, D_MODEL, device=DEVICE)
                module = TopKMultiHeadAttention(D_MODEL, N_HEADS, k).to(DEVICE)
                clear_cuda()
                for _ in range(3):
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            with autocast('cuda', dtype=torch.float16):
                                _ = module(x, x, x, use_autocast=True)
                        else:
                            _ = module(x, x, x)
                times = []
                for _ in range(NUM_RUNS):
                    try:
                        start_time = time.time()
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                with autocast('cuda', dtype=torch.float16):
                                    _ = module(x, x, x, use_autocast=True)
                            else:
                                _ = module(x, x, x)
                        times.append(time.time() - start_time)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"    OOM at seq_len={seq_len}, k={k} (topk, speed), skipping...")
                            torch.cuda.empty_cache()
                            times.append(float('nan'))
                            continue
                        else:
                            raise
                topk_times.append(np.nanmean(times))
            topk_time = np.nanmean(topk_times)
            topk_std = np.nanstd(topk_times)
            speedup = std_time / topk_time if topk_time > 0 else float('inf')
            print(f"    k={k}: std_time={std_time:.6f}s, topk_time={topk_time:.6f}s, speedup={speedup:.2f}x")
            speed_results.append({
                'seq_len': seq_len,
                'k': k,
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'batch_size': BATCH_SIZE_SPEED,
                'std_time': std_time,
                'std_time_std': std_std,
                'topk_time': topk_time,
                'topk_time_std': topk_std,
                'speedup': speedup,
                'time_ratio': topk_time / std_time if std_time > 0 else float('inf')
            })
    return speed_results


def accuracy_experiment():
    print("Running accuracy experiments...")
    accuracy_results = []
    for seq_len in SEQ_LENGTHS:
        print(f"  Testing sequence length: {seq_len}")
        std_attention = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
        for k in K_VALUES:
            if k >= seq_len:
                continue
            topk_attention = TopKMultiHeadAttention(D_MODEL, N_HEADS, k).to(DEVICE)
            # Copy weights for fair comparison
            topk_attention.w_q.weight.data = std_attention.w_q.weight.data.clone()
            topk_attention.w_k.weight.data = std_attention.w_k.weight.data.clone()
            topk_attention.w_v.weight.data = std_attention.w_v.weight.data.clone()
            topk_attention.w_o.weight.data = std_attention.w_o.weight.data.clone()
            if std_attention.w_q.bias is not None:
                topk_attention.w_q.bias.data = std_attention.w_q.bias.data.clone()
                topk_attention.w_k.bias.data = std_attention.w_k.bias.data.clone()
                topk_attention.w_v.bias.data = std_attention.w_v.bias.data.clone()
                topk_attention.w_o.bias.data = std_attention.w_o.bias.data.clone()
            relative_errors = []
            cosine_similarities = []
            for _ in range(5):
                x = torch.randn(BATCH_SIZE_MEM, seq_len, D_MODEL, device=DEVICE)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with autocast('cuda', dtype=torch.float16):
                            output_std, _ = std_attention(x, x, x)
                            output_topk, _ = topk_attention(x, x, x, use_autocast=True)
                    else:
                        output_std, _ = std_attention(x, x, x)
                        output_topk, _ = topk_attention(x, x, x)
                relative_error = torch.norm(output_std - output_topk) / torch.norm(output_std)
                relative_errors.append(relative_error.item())
                output_std_flat = output_std.flatten()
                output_topk_flat = output_topk.flatten()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    output_std_flat.unsqueeze(0), output_topk_flat.unsqueeze(0)
                )
                cosine_similarities.append(cosine_sim.item())
            rel_err_mean = np.mean(relative_errors)
            cos_sim_mean = np.mean(cosine_similarities)
            print(f"    k={k}: rel_error={rel_err_mean:.6f}, cos_sim={cos_sim_mean:.6f}")
            accuracy_results.append({
                'seq_len': seq_len,
                'k': k,
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'batch_size': BATCH_SIZE_MEM,
                'relative_error_mean': rel_err_mean,
                'relative_error_std': np.std(relative_errors),
                'cosine_similarity_mean': cos_sim_mean,
                'cosine_similarity_std': np.std(cosine_similarities),
                'k_ratio': k / seq_len
            })
    return accuracy_results


def scaling_experiment():
    print("Running scaling experiments...")
    scaling_results = []
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    for seq_len in seq_lengths:
        print(f"  Testing scaling at sequence length: {seq_len}")
        for model_type in ['standard', 'topk']:
            if model_type == 'standard':
                attention = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
                k_val = seq_len
            else:
                attention = TopKMultiHeadAttention(D_MODEL, N_HEADS, 64).to(DEVICE)
                k_val = 64
            x = torch.randn(BATCH_SIZE_MEM, seq_len, D_MODEL, device=DEVICE)
            clear_cuda()
            start_time = time.time()
            with torch.no_grad():
                output, _ = attention(x, x, x)
            end_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_used = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
            else:
                import psutil
                process = psutil.Process()
                memory_used = process.memory_info().rss / 1024**2
            print(f"    {model_type}: seq_len={seq_len}, k={k_val}, time={end_time-start_time:.6f}s, mem={memory_used:.2f}MB")
            scaling_results.append({
                'seq_len': seq_len,
                'model_type': model_type,
                'k': k_val,
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'time': end_time - start_time,
                'memory': memory_used,
                'theoretical_memory': seq_len * seq_len if model_type == 'standard' else seq_len * k_val,
                'theoretical_time_complexity': f"O(L²)" if model_type == 'standard' else f"O(L·k)"
            })
    return scaling_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file')
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(RESULTS_DIR, f'comp_results_{timestamp}.json')
    results = {
        'memory': memory_experiment(),
        'speed': speed_experiment(),
        'accuracy': accuracy_experiment(),
        'scaling': scaling_experiment()
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 