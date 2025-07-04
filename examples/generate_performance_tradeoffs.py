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

def resolve_output_path(path, default_name):
    if path is None:
        return os.path.join(PLOTS_DIR, default_name)
    if os.path.isabs(path):
        return path
    if os.path.dirname(path):
        return os.path.join(PROJECT_ROOT, path)
    return os.path.join(PLOTS_DIR, path)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='Path to results JSON file')
parser.add_argument('--output', type=str, default=None, help='Path to output image')
args = parser.parse_args()

if args.input:
    results_file = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
else:
    results_file = get_latest_comp_results()
with open(results_file, 'r') as f:
    results = json.load(f)

# Use memory and speed results
df_mem = pd.DataFrame(results['memory'])
df_speed = pd.DataFrame(results['speed'])

# Merge on seq_len and k
merged = pd.merge(df_mem, df_speed, on=['seq_len', 'k'])

# Only keep valid speedup and reduction values
merged = merged[(merged['speedup'] > 0) & (merged['theoretical_reduction'] > 0)]

plt.figure(figsize=(8, 6))
sc = plt.scatter(merged['theoretical_reduction'], merged['speedup'], c=merged['k'], cmap='viridis', s=60)
plt.xlabel('Theoretical Memory Reduction (L/k)')
plt.ylabel('Speedup (Standard / Top-k)')
plt.title('Performance Trade-offs: Memory vs Speedup')
cbar = plt.colorbar(sc)
cbar.set_label('k value')
plt.grid(True, alpha=0.3)
os.makedirs(PLOTS_DIR, exist_ok=True)
output_path = resolve_output_path(args.output, 'performance_tradeoffs.png')
plt.tight_layout()
plt.savefig(output_path)
print(f'Saved: {output_path}') 