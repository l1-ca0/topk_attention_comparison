import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from attention.standard_attention import StandardMultiHeadAttention

# Parameters
seq_len = 32
batch_size = 1
d_model = 64
n_heads = 4

# Create model and input
device = 'cpu'
model = StandardMultiHeadAttention(d_model, n_heads).to(device)
x = torch.randn(batch_size, seq_len, d_model).to(device)

# Get attention weights
with torch.no_grad():
    _, attn_weights = model(x, x, x, return_attention=True)
    # attn_weights shape: (batch, seq_len, seq_len) (averaged over heads)
    attn_weights = attn_weights[0].cpu().numpy()  # (seq_len, seq_len)

# Plot averaged attention pattern
plt.figure(figsize=(5, 4))
plt.imshow(attn_weights, aspect='auto', cmap='viridis')
plt.title('Averaged Attention Pattern (Standard Attention)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
cbar = plt.colorbar()
cbar.set_label('Attention Weight')
os.makedirs('results/plots', exist_ok=True)
plt.tight_layout()
plt.savefig('results/plots/attention_patterns.png')
print('Saved: results/plots/attention_patterns.png') 