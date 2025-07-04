#!/usr/bin/env python3
"""
Comprehensive benchmark using the original Top-k repo's AttentionNoCache for Top-k attention.
Benchmarks both standard and original Top-k attention for memory, speed, accuracy, and scaling.
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
import datetime

# Ensure the parent directory is in sys.path so top_k_attention is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/kaggle/working/topk_attention_project')
from top_k_attention.nocache_attention.nocache_attention import AttentionNoCache

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from attention import StandardMultiHeadAttention
from attention.utils import estimate_memory_usage

# Benchmark settings (match paper)
SEQ_LENGTHS = [1024, 2048, 4096, 8192]
K_VALUES = [8, 16, 32, 64, 128, 256]
D_MODEL = 512
N_HEADS = 8
BATCH_SIZE_MEM = 1
BATCH_SIZE_SPEED = 1
NUM_RUNS = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_peak_memory(module, x, is_topk=False, k=None):
    clear_cuda()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        if torch.cuda.is_available():
            if is_topk:
                batch, seq_len, d_model = x.shape
                d_head = d_model // N_HEADS
                Q = x.view(batch, seq_len, N_HEADS, d_head).permute(0,2,1,3).contiguous().view(batch*N_HEADS, seq_len, d_head)
                K = Q.clone(); V = Q.clone()
                activation = lambda x: torch.softmax(x * d_head**-0.5, dim=-1)
                args = {'topk': k, 'Q_chunk_size': 2048}
                _ = module(activation)(Q, K, V, causal_masking=False, args=args)
            else:
                _ = module(x, x, x)
        else:
            if is_topk:
                batch, seq_len, d_model = x.shape
                d_head = d_model // N_HEADS
                Q = x.view(batch, seq_len, N_HEADS, d_head).permute(0,2,1,3).contiguous().view(batch*N_HEADS, seq_len, d_head)
                K = Q.clone(); V = Q.clone()
                activation = lambda x: torch.softmax(x * d_head**-0.5, dim=-1)
                args = {'topk': k, 'Q_chunk_size': 2048}
                _ = module(activation)(Q, K, V, causal_masking=False, args=args)
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


def get_topk_module():
    # Returns the AttentionNoCache class from nocache_attention
    return AttentionNoCache


def memory_experiment():
    print("Running memory experiments with original Top-k implementation...")
    memory_results = []
    for seq_len in SEQ_LENGTHS:
        print(f"  Testing sequence length: {seq_len}")
        # Standard attention
        std_peaks = []
        for _ in range(NUM_RUNS):
            x = torch.randn(BATCH_SIZE_MEM, seq_len, D_MODEL, device=DEVICE)
            module = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
            try:
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
                module = get_topk_module()
                try:
                    topk_peaks.append(measure_peak_memory(module, x, is_topk=True, k=k))
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
    print("Running speed experiments with original Top-k implementation...")
    speed_results = []
    num_runs = 10
    for seq_len in SEQ_LENGTHS:
        print(f"  Testing sequence length: {seq_len}")
        # Standard attention speed
        try:
            std_attention = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
            x = torch.randn(BATCH_SIZE_SPEED, seq_len, D_MODEL, device=DEVICE)
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = std_attention(x, x, x)
            # Measure time
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = std_attention(x, x, x)
                times.append(time.time() - start_time)
            std_time = np.mean(times)
            std_std = np.std(times)
            print(f"    Standard: time={std_time:.6f}s ± {std_std:.6f}s")
            del std_attention, x
        except Exception as e:
            print(f"    Standard attention failed at seq_len={seq_len}: {e}")
            std_time = float('inf')
            std_std = 0
        # Top-k attention speed
        for k in K_VALUES:
            if k > seq_len:
                continue
            try:
                module = get_topk_module()
                batch = BATCH_SIZE_SPEED
                d_head = D_MODEL // N_HEADS
                x = torch.randn(batch, seq_len, D_MODEL, device=DEVICE)
                Q = x.view(batch, seq_len, N_HEADS, d_head).permute(0,2,1,3).contiguous().view(batch*N_HEADS, seq_len, d_head)
                K_ = Q.clone(); V_ = Q.clone()
                activation = lambda x: torch.softmax(x * d_head**-0.5, dim=-1)
                args = {'topk': k, 'Q_chunk_size': 2048}
                # Warmup
                for _ in range(3):
                    _ = module(activation)(Q, K_, V_, causal_masking=False, args=args)
                # Measure time
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = module(activation)(Q, K_, V_, causal_masking=False, args=args)
                    times.append(time.time() - start_time)
                topk_time = np.mean(times)
                topk_std = np.std(times)
                speedup = std_time / topk_time if topk_time > 0 else float('inf')
                print(f"    k={k}: topk_time={topk_time:.6f}s ± {topk_std:.6f}s, speedup={speedup:.2f}x")
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
                del x, Q, K_, V_
            except Exception as e:
                print(f"    Top-k attention failed at seq_len={seq_len}, k={k}: {e}")
    return speed_results


def accuracy_experiment():
    print("Running accuracy experiments with original Top-k implementation...")
    accuracy_results = []
    for seq_len in SEQ_LENGTHS:
        print(f"  Testing sequence length: {seq_len}")
        std_attention = StandardMultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
        for k in K_VALUES:
            if k >= seq_len:
                continue
            topk_attention = get_topk_module()
            # Copy weights for fair comparison (only if possible)
            # Note: AttentionNoCache may not have the same API, so skip weight copy if not possible
            relative_errors = []
            cosine_similarities = []
            for _ in range(5):
                x = torch.randn(BATCH_SIZE_MEM, seq_len, D_MODEL, device=DEVICE)
                with torch.no_grad():
                    output_std, _ = std_attention(x, x, x)
                    # Prepare Q, K, V for original top-k
                    batch, seq_len_, d_model = x.shape
                    d_head = d_model // N_HEADS
                    Q = x.view(batch, seq_len_, N_HEADS, d_head).permute(0,2,1,3).contiguous().view(batch*N_HEADS, seq_len_, d_head)
                    K_ = Q.clone(); V_ = Q.clone()
                    activation = lambda x: torch.softmax(x * d_head**-0.5, dim=-1)
                    args = {'topk': k, 'Q_chunk_size': 2048}
                    output_topk = topk_attention(activation)(Q, K_, V_, causal_masking=False, args=args)
                    # Reshape output_topk to match output_std
                    output_topk = output_topk.view(batch, N_HEADS, seq_len_, d_head).permute(0,2,1,3).contiguous().view(batch, seq_len_, d_model)
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


def save_results(results, tag="original"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"comp_results_{tag}_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("__main__ block reached. Running memory_experiment...")
    memory_results = memory_experiment()
    print("Running speed_experiment...")
    speed_results = speed_experiment()
    print("Running accuracy_experiment...")
    accuracy_results = accuracy_experiment()
    results = {"memory": memory_results, "speed": speed_results, "accuracy": accuracy_results}
    save_results(results, tag="original") 