#!/usr/bin/env python3
"""
Analyze experimental results and create comprehensive visualizations.
This script loads results from comprehensive_benchmark.py and generates analysis reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import glob
import time

# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyze and visualize experimental results."""
    
    def __init__(self, results_file: str = None):
        self.results_file = results_file
        self.results = None
        self.figures = []
        
    def load_results(self, results_file: str = None) -> Dict:
        """Load experimental results from JSON file."""
        if results_file:
            self.results_file = results_file
        elif not self.results_file:
            # Find the most recent results file
            results_files = glob.glob("results/comp_results*.json")
            if not results_files:
                raise FileNotFoundError("No benchmark results found. Run experiments first.")
            self.results_file = max(results_files, key=os.path.getctime)
            print(f"Loading most recent results: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        return self.results
    
    def analyze_memory_results(self) -> Dict:
        """Analyze memory usage results."""
        if not self.results or 'memory' not in self.results:
            return {}
        
        memory_data = pd.DataFrame(self.results['memory'])
        if memory_data.empty:
            return {}
        
        print("MEMORY ANALYSIS")
        print("="*50)
        
        # Summary statistics
        analysis = {
            'summary': {},
            'by_k': {},
            'by_seq_len': {}
        }
        
        # Compute average actual memory reduction as mean(std_memory/topk_memory) for all valid rows
        valid_rows = memory_data[(memory_data['std_memory'] > 0) & (memory_data['topk_memory'] > 0)]
        avg_actual_computed = (valid_rows['std_memory'] / valid_rows['topk_memory']).mean() if not valid_rows.empty else float('nan')

        # Overall summary
        analysis['summary'] = {
            'avg_theoretical_reduction': memory_data['theoretical_reduction'].mean(),
            'avg_actual_reduction': memory_data[memory_data['actual_reduction'] != float('inf')]['actual_reduction'].mean(),
            'avg_actual_computed': avg_actual_computed,
            'max_theoretical_reduction': memory_data['theoretical_reduction'].max(),
            'max_actual_reduction': memory_data[memory_data['actual_reduction'] != float('inf')]['actual_reduction'].max(),
            'configurations_tested': len(memory_data)
        }
        
        print(f"Configurations tested: {analysis['summary']['configurations_tested']}")
        print(f"Average theoretical memory reduction: {analysis['summary']['avg_theoretical_reduction']:.2f}x")
        if not np.isnan(analysis['summary']['avg_actual_reduction']):
            print(f"Average actual memory reduction (from column): {analysis['summary']['avg_actual_reduction']:.2f}x")
        if not np.isnan(analysis['summary']['avg_actual_computed']):
            print(f"Average actual memory reduction (computed): {analysis['summary']['avg_actual_computed']:.2f}x")
        print(f"Maximum theoretical reduction: {analysis['summary']['max_theoretical_reduction']:.2f}x")
        
        # Analysis by k value
        for k in sorted(memory_data['k'].unique()):
            k_data = memory_data[memory_data['k'] == k]
            analysis['by_k'][k] = {
                'avg_reduction': k_data['theoretical_reduction'].mean(),
                'configurations': len(k_data),
                'seq_lengths': sorted(k_data['seq_len'].unique())
            }
            print(f"k={k}: Average reduction {analysis['by_k'][k]['avg_reduction']:.2f}x across {analysis['by_k'][k]['configurations']} configs")
        
        return analysis
    
    def analyze_speed_results(self) -> Dict:
        """Analyze speed performance results."""
        if not self.results or 'speed' not in self.results:
            return {}
        
        speed_data = pd.DataFrame(self.results['speed'])
        if speed_data.empty:
            return {}
        
        print("\nSPEED ANALYSIS")
        print("="*50)
        
        analysis = {
            'summary': {},
            'by_k': {},
            'by_seq_len': {}
        }
        
        # Filter out infinite values
        valid_speed_data = speed_data[
            (speed_data['speedup'] != float('inf')) & 
            (speed_data['speedup'] > 0) &
            (speed_data['std_time'] != float('inf')) &
            (speed_data['topk_time'] > 0)
        ]
        
        if valid_speed_data.empty:
            print("No valid speed data available")
            return analysis
        
        # Overall summary
        analysis['summary'] = {
            'avg_speedup': valid_speed_data['speedup'].mean(),
            'max_speedup': valid_speed_data['speedup'].max(),
            'min_speedup': valid_speed_data['speedup'].min(),
            'configurations_tested': len(valid_speed_data)
        }
        
        print(f"Configurations tested: {analysis['summary']['configurations_tested']}")
        print(f"Average speedup: {analysis['summary']['avg_speedup']:.2f}x")
        print(f"Maximum speedup: {analysis['summary']['max_speedup']:.2f}x")
        print(f"Minimum speedup: {analysis['summary']['min_speedup']:.2f}x")
        
        # Analysis by k value
        for k in sorted(valid_speed_data['k'].unique()):
            k_data = valid_speed_data[valid_speed_data['k'] == k]
            analysis['by_k'][k] = {
                'avg_speedup': k_data['speedup'].mean(),
                'configurations': len(k_data)
            }
            print(f"k={k}: Average speedup {analysis['by_k'][k]['avg_speedup']:.2f}x")
        
        return analysis
    
    def analyze_accuracy_results(self) -> Dict:
        """Analyze accuracy/similarity results."""
        if not self.results or 'accuracy' not in self.results:
            return {}
        
        accuracy_data = pd.DataFrame(self.results['accuracy'])
        if accuracy_data.empty:
            return {}
        
        print("\nACCURACY ANALYSIS")
        print("="*50)
        
        analysis = {
            'summary': {},
            'by_k_ratio': {},
            'by_seq_len': {}
        }
        
        # Overall summary
        analysis['summary'] = {
            'avg_relative_error': accuracy_data['relative_error_mean'].mean(),
            'avg_cosine_similarity': accuracy_data['cosine_similarity_mean'].mean(),
            'max_relative_error': accuracy_data['relative_error_mean'].max(),
            'min_cosine_similarity': accuracy_data['cosine_similarity_mean'].min(),
            'configurations_tested': len(accuracy_data)
        }
        
        print(f"Configurations tested: {analysis['summary']['configurations_tested']}")
        print(f"Average relative error: {analysis['summary']['avg_relative_error']:.4f}")
        print(f"Average cosine similarity: {analysis['summary']['avg_cosine_similarity']:.4f}")
        print(f"Maximum relative error: {analysis['summary']['max_relative_error']:.4f}")
        print(f"Minimum cosine similarity: {analysis['summary']['min_cosine_similarity']:.4f}")
        
        # Analysis by k/seq_len ratio
        for k_ratio in sorted(accuracy_data['k_ratio'].unique()):
            ratio_data = accuracy_data[accuracy_data['k_ratio'] == k_ratio]
            analysis['by_k_ratio'][k_ratio] = {
                'avg_relative_error': ratio_data['relative_error_mean'].mean(),
                'avg_cosine_similarity': ratio_data['cosine_similarity_mean'].mean(),
                'configurations': len(ratio_data)
            }
            print(f"k/L ratio={k_ratio:.3f}: Error={analysis['by_k_ratio'][k_ratio]['avg_relative_error']:.4f}, "
                  f"Similarity={analysis['by_k_ratio'][k_ratio]['avg_cosine_similarity']:.4f}")
        
        return analysis
    
    def create_visualizations(self, save_plots: bool = True) -> List[str]:
        """Create comprehensive visualizations of experimental results."""
        if not self.results:
            self.load_results()
        
        saved_files = []
        
        # Create results directory for plots
        plot_dir = "results/plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Memory Usage Comparison
        if self.results.get('memory'):
            fig = self.plot_memory_comparison()
            if save_plots:
                filename = f"{plot_dir}/memory_comparison.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                saved_files.append(filename)
                plt.close(fig)
        
        # 2. Speed Performance
        if self.results.get('speed'):
            fig = self.plot_speed_comparison()
            if save_plots:
                filename = f"{plot_dir}/speed_comparison.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                saved_files.append(filename)
                plt.close(fig)
        
        # 3. Accuracy Analysis
        if self.results.get('accuracy'):
            fig = self.plot_accuracy_analysis()
            if save_plots:
                filename = f"{plot_dir}/accuracy_analysis.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                saved_files.append(filename)
                plt.close(fig)
        
        # 4. Scaling Analysis
        if self.results.get('scaling'):
            fig = self.plot_scaling_analysis()
            if save_plots:
                filename = f"{plot_dir}/scaling_analysis.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                saved_files.append(filename)
                plt.close(fig)
        
        # 5. Summary Dashboard
        fig = self.create_summary_dashboard()
        if save_plots:
            filename = f"{plot_dir}/summary_dashboard.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)
        
        return saved_files
    
    def plot_memory_comparison(self):
        """Create memory usage comparison plots."""
        memory_data = pd.DataFrame(self.results['memory'])
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Memory Usage Analysis: Standard vs Top-k Attention', fontsize=16, fontweight='bold')

        # 1. Memory Reduction: Actual (GPU) or Theoretical (CPU)
        has_actual = (
            'actual_reduction' in memory_data.columns and
            (memory_data['actual_reduction'].replace([float('inf'), float('nan')], 0) > 0).any()
        )
        for k in sorted(memory_data['k'].unique()):
            k_data = memory_data[memory_data['k'] == k]
            if has_actual:
                valid_actual = k_data[(k_data['std_memory'] > 0) & (k_data['topk_memory'] > 0)]
                if not valid_actual.empty:
                    actual_reduction = valid_actual['std_memory'] / valid_actual['topk_memory']
                    ax1.plot(valid_actual['seq_len'], actual_reduction,
                             marker='s', label=f'k={k} (actual)', linestyle='-')
            else:
                ax1.plot(k_data['seq_len'], k_data['theoretical_reduction'],
                         marker='o', label=f'k={k} (theoretical)', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Memory Reduction Factor')
        ax1.set_title('Memory Reduction (Standard / Top-k) on GPU' if has_actual else 'Theoretical Memory Reduction (L/k)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Memory Ratio by K and Sequence Length
        pivot_data = memory_data.pivot(index='seq_len', columns='k', values='memory_ratio')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('Memory Ratio (Top-k/Standard)')
        ax2.set_xlabel('k value')
        ax2.set_ylabel('Sequence Length')
        
        # 3. Reduction by k/L ratio (actual or theoretical)
        memory_data['k_ratio'] = memory_data['k'] / memory_data['seq_len']
        for seq_len in sorted(memory_data['seq_len'].unique()):
            seq_data = memory_data[memory_data['seq_len'] == seq_len]
            if has_actual:
                valid_actual = seq_data[(seq_data['std_memory'] > 0) & (seq_data['topk_memory'] > 0)]
                if not valid_actual.empty:
                    actual_reduction = valid_actual['std_memory'] / valid_actual['topk_memory']
                    ax3.plot(valid_actual['k_ratio'], actual_reduction, 
                             marker='s', label=f'L={seq_len} (actual)')
            else:
                ax3.plot(seq_data['k_ratio'], seq_data['theoretical_reduction'], 
                         marker='o', label=f'L={seq_len} (theoretical)')
        ax3.set_xlabel('k/L Ratio')
        ax3.set_ylabel('Memory Reduction (Standard / Top-k)' if has_actual else 'Theoretical Memory Reduction (L/k)')
        ax3.set_title('Actual Memory Reduction vs k/L Ratio' if has_actual else 'Theoretical Memory Reduction vs k/L Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Memory savings for different configurations (actual or theoretical)
        if has_actual:
            valid_actual = memory_data[(memory_data['std_memory'] > 0) & (memory_data['topk_memory'] > 0)]
            actual_reduction = valid_actual['std_memory'] / valid_actual['topk_memory']
            ax4.bar(range(len(valid_actual)), actual_reduction, 
                   color='skyblue', alpha=0.7)
            ax4.set_ylabel('Actual Memory Reduction (Standard / Top-k)')
        else:
            ax4.bar(range(len(memory_data)), memory_data['theoretical_reduction'], 
                   color='skyblue', alpha=0.7)
            ax4.set_ylabel('Theoretical Memory Reduction (L/k)')
        ax4.set_xlabel('Configuration Index')
        ax4.set_title('Actual Memory Reduction Across All Configurations' if has_actual else 'Theoretical Memory Reduction Across All Configurations')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_speed_comparison(self):
        """Create speed performance comparison plots."""
        speed_data = pd.DataFrame(self.results['speed'])
        valid_data = speed_data[
            (speed_data['speedup'] != float('inf')) & 
            (speed_data['speedup'] > 0) &
            (speed_data['std_time'] != float('inf'))
        ]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Speed Performance Analysis: Standard vs Top-k Attention', fontsize=16, fontweight='bold')
        
        if valid_data.empty:
            ax1.text(0.5, 0.5, 'No valid speed data available', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=14)
            return fig
        
        # 1. Speedup by sequence length and k
        for k in sorted(valid_data['k'].unique()):
            k_data = valid_data[valid_data['k'] == k]
            ax1.plot(k_data['seq_len'], k_data['speedup'], 
                    marker='o', label=f'k={k}', linewidth=2)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Speed Improvement vs Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Execution time comparison
        for k in sorted(valid_data['k'].unique())[:3]:  # Show top 3 k values
            k_data = valid_data[valid_data['k'] == k]
            ax2.plot(k_data['seq_len'], k_data['std_time'], 
                    marker='s', label=f'Standard (k={k})', linestyle='--', alpha=0.7)
            ax2.plot(k_data['seq_len'], k_data['topk_time'], 
                    marker='o', label=f'Top-k (k={k})', linestyle='-')
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Execution Time Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Speedup distribution
        ax3.hist(valid_data['speedup'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(valid_data['speedup'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {valid_data["speedup"].mean():.2f}x')
        ax3.set_xlabel('Speedup Factor')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Speed Improvements')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. k vs speedup relationship
        k_speedup = valid_data.groupby('k')['speedup'].agg(['mean', 'std']).reset_index()
        ax4.errorbar(k_speedup['k'], k_speedup['mean'], yerr=k_speedup['std'], 
                    marker='o', capsize=5, linewidth=2)
        ax4.set_xlabel('k value')
        ax4.set_ylabel('Average Speedup')
        ax4.set_title('Speedup vs k Value')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_accuracy_analysis(self):
        """Create accuracy analysis plots."""
        accuracy_data = pd.DataFrame(self.results['accuracy'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Accuracy Analysis: Standard vs Top-k Attention', fontsize=16, fontweight='bold')
        
        # 1. Relative error vs k/L ratio
        ax1.scatter(accuracy_data['k_ratio'], accuracy_data['relative_error_mean'], 
                   alpha=0.6, s=50, c=accuracy_data['seq_len'], cmap='viridis')
        ax1.set_xlabel('k/L Ratio')
        ax1.set_ylabel('Relative Error')
        ax1.set_title('Relative Error vs k/L Ratio')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('Sequence Length')
        
        # 2. Cosine similarity vs k/L ratio
        ax2.scatter(accuracy_data['k_ratio'], accuracy_data['cosine_similarity_mean'], 
                   alpha=0.6, s=50, c=accuracy_data['seq_len'], cmap='viridis')
        ax2.set_xlabel('k/L Ratio')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Cosine Similarity vs k/L Ratio')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Sequence Length')
        
        # 3. Error distribution by k value
        k_values = sorted(accuracy_data['k'].unique())
        error_by_k = [accuracy_data[accuracy_data['k'] == k]['relative_error_mean'].values 
                     for k in k_values]
        ax3.boxplot(error_by_k, labels=k_values)
        ax3.set_xlabel('k value')
        ax3.set_ylabel('Relative Error')
        ax3.set_title('Error Distribution by k Value')
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy vs Memory Trade-off
        if 'memory' in self.results:
            memory_data = pd.DataFrame(self.results['memory'])
            # Merge accuracy and memory data
            merged = accuracy_data.merge(
                memory_data[['seq_len', 'k', 'theoretical_reduction']], 
                on=['seq_len', 'k'], how='inner'
            )
            
            if not merged.empty:
                scatter = ax4.scatter(merged['theoretical_reduction'], merged['relative_error_mean'], 
                            alpha=0.6, s=50, c=merged['k_ratio'], cmap='plasma')
                ax4.set_xlabel('Memory Reduction Factor')
                ax4.set_ylabel('Relative Error')
                ax4.set_title('Accuracy vs Memory Trade-off')
                ax4.grid(True, alpha=0.3)
                ax4.set_xscale('log')
                cbar4 = plt.colorbar(scatter, ax=ax4)
                cbar4.set_label('k/L Ratio')
        
        plt.tight_layout()
        return fig
    
    def plot_scaling_analysis(self):
        """Create scaling analysis plots."""
        scaling_data = pd.DataFrame(self.results['scaling'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scaling Analysis: Standard vs Top-k Attention', fontsize=16, fontweight='bold')
        
        # Split data by model type
        std_data = scaling_data[scaling_data['model_type'] == 'standard']
        topk_data = scaling_data[scaling_data['model_type'] == 'topk']
        
        # 1. Time scaling
        if not std_data.empty:
            ax1.plot(std_data['seq_len'], std_data['time'], 
                    marker='s', label='Standard', linewidth=2, markersize=8)
        if not topk_data.empty:
            ax1.plot(topk_data['seq_len'], topk_data['time'], 
                    marker='o', label='Top-k', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Time Complexity Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # 2. Memory scaling
        if not std_data.empty and std_data['memory'].max() > 0:
            ax2.plot(std_data['seq_len'], std_data['memory'], 
                    marker='s', label='Standard', linewidth=2, markersize=8)
        if not topk_data.empty and topk_data['memory'].max() > 0:
            ax2.plot(topk_data['seq_len'], topk_data['memory'], 
                    marker='o', label='Top-k', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Usage (bytes)')
        ax2.set_title('Memory Complexity Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        # 3. Theoretical complexity comparison
        if not std_data.empty:
            ax3.plot(std_data['seq_len'], std_data['theoretical_memory'], 
                    marker='s', label='Standard O(L²)', linewidth=2, linestyle='--')
        if not topk_data.empty:
            ax3.plot(topk_data['seq_len'], topk_data['theoretical_memory'], 
                    marker='o', label='Top-k O(L·k)', linewidth=2, linestyle='--')
        
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Theoretical Memory Complexity')
        ax3.set_title('Theoretical Complexity Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        
        # 4. Speedup vs sequence length
        if not std_data.empty and not topk_data.empty:
            # Merge data to calculate speedup
            merged = std_data.merge(topk_data, on='seq_len', suffixes=('_std', '_topk'))
            merged['speedup'] = merged['time_std'] / merged['time_topk']
            
            ax4.plot(merged['seq_len'], merged['speedup'], 
                    marker='o', linewidth=2, markersize=8, color='green')
            ax4.set_xlabel('Sequence Length')
            ax4.set_ylabel('Speedup Factor')
            ax4.set_title('Speedup vs Sequence Length')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self):
        """Create a summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Top-k Attention Performance Summary', fontsize=16, fontweight='bold')
        
        # Memory reduction summary
        if self.results.get('memory'):
            memory_data = pd.DataFrame(self.results['memory'])
            k_values = sorted(memory_data['k'].unique())
            # Use actual memory reduction if available, else theoretical
            has_actual = (
                'std_memory' in memory_data.columns and 'topk_memory' in memory_data.columns and
                (memory_data['std_memory'] > 0).any() and (memory_data['topk_memory'] > 0).any()
            )
            avg_reductions = []
            for k in k_values:
                k_data = memory_data[memory_data['k'] == k]
                if has_actual:
                    valid = k_data[(k_data['std_memory'] > 0) & (k_data['topk_memory'] > 0)]
                    if not valid.empty:
                        avg_reductions.append((valid['std_memory'] / valid['topk_memory']).mean())
                    else:
                        avg_reductions.append(float('nan'))
                else:
                    avg_reductions.append(k_data['theoretical_reduction'].mean())
            bars = ax1.bar(range(len(k_values)), avg_reductions, color='skyblue', alpha=0.7)
            ax1.set_xticks(range(len(k_values)))
            ax1.set_xticklabels(k_values)
            ax1.set_xlabel('k value')
            ax1.set_ylabel('Avg Memory Reduction (Standard / Top-k)' if has_actual else 'Avg Theoretical Memory Reduction (L/k)')
            ax1.set_title('Memory Reduction by k Value' if has_actual else 'Theoretical Memory Reduction by k Value')
            ax1.grid(True, alpha=0.3)
            # Add value labels
            for bar, reduction in zip(bars, avg_reductions):
                height = bar.get_height()
                if not pd.isna(reduction):
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{reduction:.2f}x', ha='center', va='bottom')
        
        # Speed summary
        if self.results.get('speed'):
            speed_data = pd.DataFrame(self.results['speed'])
            valid_data = speed_data[speed_data['speedup'] != float('inf')]
            if not valid_data.empty:
                k_values = sorted(valid_data['k'].unique())
                avg_speedups = [valid_data[valid_data['k'] == k]['speedup'].mean() for k in k_values]
                
                bars = ax2.bar(range(len(k_values)), avg_speedups, color='lightgreen', alpha=0.7)
                ax2.set_xticks(range(len(k_values)))
                ax2.set_xticklabels(k_values)
                ax2.set_xlabel('k value')
                ax2.set_ylabel('Average Speedup')
                ax2.set_title('Speed Improvement by k Value')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, speedup in zip(bars, avg_speedups):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{speedup:.2f}x', ha='center', va='bottom')
        
        # Accuracy summary
        if self.results.get('accuracy'):
            accuracy_data = pd.DataFrame(self.results['accuracy'])
            k_ratios = sorted(accuracy_data['k_ratio'].unique())
            avg_errors = [accuracy_data[accuracy_data['k_ratio'] == kr]['relative_error_mean'].mean() 
                         for kr in k_ratios]
            ax3.bar(range(len(k_ratios)), avg_errors, color='coral', alpha=0.7)
            ax3.set_xticks(range(len(k_ratios)))
            ax3.set_xticklabels([f'{kr:.2f}' for kr in k_ratios], rotation=45)
            ax3.set_xlabel('k/L Ratio')
            ax3.set_ylabel('Average Relative Error')
            ax3.set_title('Accuracy vs k/L Ratio')
            ax3.grid(True, alpha=0.3)
        
        # Configuration summary
        total_configs = 0
        for category in ['memory', 'speed', 'accuracy', 'scaling']:
            if self.results.get(category):
                total_configs += len(self.results[category])
        
        categories = [cat for cat in ['Memory', 'Speed', 'Accuracy', 'Scaling'] 
                     if self.results.get(cat.lower())]
        counts = [len(self.results[cat.lower()]) for cat in categories]
        
        ax4.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Experimental Configuration Distribution')
        
        plt.tight_layout()
        return fig


def main():
    """Run analysis of experimental results."""
    analyzer = ResultsAnalyzer()
    
    try:
        # Load results
        results = analyzer.load_results()
        
        # Run analysis
        print("\nRunning analysis...")
        memory_analysis = analyzer.analyze_memory_results()
        
        # Create summary visualization
        print("\nCreating summary visualization...")
        os.makedirs("results/plots", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig = analyzer.create_summary_dashboard()
        filename = f"results/plots/comp_summary.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Summary visualization saved to: {filename}")
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 