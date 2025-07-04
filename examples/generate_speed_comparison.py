import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

def get_latest_comp_results(results_dir=RESULTS_DIR):
    files = [f for f in os.listdir(results_dir) if f.startswith('comp_results_') and f.endswith('.json')]
    if not files:
        raise FileNotFoundError('No comprehensive results file found.')
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, files[0])

def resolve_output_dir(path):
    if path is None:
        return PLOTS_DIR
    if os.path.isabs(path):
        return path
    if os.path.dirname(path):
        return os.path.join(PROJECT_ROOT, path)
    return os.path.join(PLOTS_DIR, path)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='Path to results JSON file')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for images')
args = parser.parse_args()

if args.input:
    results_file = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
else:
    results_file = get_latest_comp_results()
with open(results_file, 'r') as f:
    results = json.load(f)

# Speed comparison plot
df_speed = pd.DataFrame(results['speed'])
plt.figure(figsize=(10, 6))
for k in sorted(df_speed['k'].unique()):
    subset = df_speed[df_speed['k'] == k]
    plt.plot(subset['seq_len'], subset['speedup'], marker='o', label=f'k={k}')
plt.xlabel('Sequence Length')
plt.ylabel('Speedup (Standard / Top-k)')
plt.title('Speedup by Sequence Length and k')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = resolve_output_dir(args.output_dir)
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'))
print(f'Saved: {os.path.join(output_dir, 'speedup_comparison.png')}')

# Also plot execution times for standard and top-k
plt.figure(figsize=(10, 6))
for k in sorted(df_speed['k'].unique()):
    subset = df_speed[df_speed['k'] == k]
    plt.plot(subset['seq_len'], subset['std_time'], marker='s', linestyle='--', label=f'Standard (k={k})')
    plt.plot(subset['seq_len'], subset['topk_time'], marker='o', linestyle='-', label=f'Top-k (k={k})')
plt.xlabel('Sequence Length')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time by Sequence Length and k')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'speed_comparison.png'))
print(f'Saved: {os.path.join(output_dir, 'speed_comparison.png')}') 