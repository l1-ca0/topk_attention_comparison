import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Robust project root detection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Find latest comprehensive results file
def get_latest_comp_results(results_dir=RESULTS_DIR):
    files = [f for f in os.listdir(results_dir) if f.startswith('comp_results_') and f.endswith('.json')]
    if not files:
        raise FileNotFoundError('No comprehensive results file found.')
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, files[0])

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='Path to results JSON file')
parser.add_argument('--output', type=str, default=None, help='Path to output image (theoretical)')
parser.add_argument('--output_actual', type=str, default=None, help='Path to output image (actual memory)')
args = parser.parse_args()

if args.input:
    results_file = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
else:
    results_file = get_latest_comp_results()
with open(results_file, 'r') as f:
    results = json.load(f)

df_mem = pd.DataFrame(results['memory'])

# Theoretical memory reduction plot
plt.figure(figsize=(8, 5))
for k in sorted(df_mem['k'].unique()):
    subset = df_mem[df_mem['k'] == k]
    plt.plot(subset['seq_len'], subset['theoretical_reduction'], marker='o', label=f'k={k}')
plt.xlabel('Sequence Length')
plt.ylabel('Theoretical Memory Reduction (L/k)')
plt.title('Memory Reduction by Sequence Length and k')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)

def resolve_output_path(path, default_name):
    if path is None:
        return os.path.join(PLOTS_DIR, default_name)
    if os.path.isabs(path):
        return path
    if os.path.dirname(path):
        return os.path.join(PROJECT_ROOT, path)
    return os.path.join(PLOTS_DIR, path)

output_path = resolve_output_path(args.output, 'memory_comparison_theoretical.png')
plt.savefig(output_path)
print(f'Saved: {output_path}')
plt.close()

# Only plot actual memory reduction if actual data is present
if 'std_memory' in df_mem.columns and 'topk_memory' in df_mem.columns and (df_mem['std_memory'].max() > 0 or df_mem['topk_memory'].max() > 0):
    plt.figure(figsize=(8, 5))
    for k in sorted(df_mem['k'].unique()):
        subset = df_mem[df_mem['k'] == k]
        actual_reduction = subset['std_memory'] / subset['topk_memory']
        plt.plot(subset['seq_len'], actual_reduction, marker='s', linestyle='-', label=f'k={k}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Reduction (Standard / Top-k)')
    plt.title('GPU Memory Reduction by Sequence Length and k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_actual = resolve_output_path(args.output_actual, 'memory_comparison_actual.png')
    plt.savefig(output_actual)
    print(f'Saved: {output_actual}')
    plt.close() 