import math
import torch as pt
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # x: (B, T, embed_dim) -> required shape: (T, B, embed_dim)
        B, T, _ = x.size()
        x = x.transpose(0, 1)
        # Create a causal mask: disallow attention to future tokens.
        mask = pt.triu(pt.full((T, T), float('-inf'), device=x.device), diagonal=1)
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        return attn_output.transpose(0, 1)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, state_dim, model_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        """
        state_dim: Dimensionality of input and output state. Must be even.
        model_dim: Dimension used inside the Transformer blocks.
        num_layers: Number of TransformerBlocks to stack.
        """
        super().__init__()
        self.state_dim = state_dim
        self.model_dim = model_dim
        # Input linear layer merges two state encodings -> model_dim.
        self.input_linear = nn.Linear(state_dim, model_dim)
        self.pos_linear = nn.Linear(model_dim * 2, model_dim)
        # Stack of TransformerBlocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        # Output projection: model_dim -> state_dim.
        self.output_proj = nn.Linear(model_dim, state_dim)

        # Precompute the frequency tensor for RoPE.
        d = model_dim // 2
        inv_freq = 1.0 / (10000 ** (pt.arange(0, d, dtype=pt.float32) / d))
        self.register_buffer("inv_freq", inv_freq)

    def rope(self, x, pos):
        """
        Apply RoPE to an input tensor x using the given position.
        x: Tensor of shape (B, T, model_dim) where model_dim is even.
        pos: Tensor of shape (B, T) (or (B,) for a single position) representing positions.
        Returns a tensor of shape (B, T, model_dim).
        """
        if pos.dim() == 1:
            pos = pos.unsqueeze(1)  # (B, 1)
        pos = pos.unsqueeze(-1)  # (B, T, 1)
        d = x.size(-1) // 2
        # Expand inv_freq for broadcasting: (1, 1, d)
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0)
        theta = pos * inv_freq  # (B, T, d)
        cos_theta = theta.cos()
        sin_theta = theta.sin()
        x1, x2 = x[..., :d], x[..., d:]
        return pt.cat([x1 * cos_theta - x2 * sin_theta,
                          x1 * sin_theta + x2 * cos_theta], dim=-1)

    def future_encoding(self, pos):
        """
        Create a positional encoding vector for a future position.
        pos: Tensor of shape (B, T) (or (B,) for a single position).
        Returns a tensor of shape (B, T, model_dim) where the first half is cos and the second half is sin.
        """
        if pos.dim() == 1:
            pos = pos.unsqueeze(1)
        pos = pos.unsqueeze(-1)  # (B, T, 1)
        d = self.model_dim // 2
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        theta = pos * inv_freq  # (B, T, d)
        cos_theta = theta.cos()
        sin_theta = theta.sin()
        return pt.cat([cos_theta, sin_theta], dim=-1)

    def forward(self, last_state, current_pos, future_pos):
        """
        last_state: Tensor of shape (B, T, state_dim).
        current_pos: Tensor of shape (B, T) representing current positions.
        future_pos: Tensor of shape (B, T) representing future positions.
        Returns a tensor of shape (B, T, state_dim).
        """
        # Apply RoPE to last_state using current positions.
        last_emb = self.input_linear(last_state)
        current_encoded = self.rope(last_emb, current_pos)  # (B, T, model_dim)
        # Create future position encoding.
        future_encoded = self.future_encoding(future_pos)  # (B, T, model_dim)
        # Merge both encodings.
        merged = pt.cat([current_encoded, future_encoded], dim=-1)  # (B, T, 2 * model_dim)
        # Project to transformer model dimension.
        x = self.pos_linear(merged)  # (B, T, model_dim)
        # Pass through stacked TransformerBlocks.
        for block in self.blocks:
            x = block(x)
        # Project back to state_dim.
        return self.output_proj(x)

# --- Quick Test with Sequence Length 8 ---

if __name__ == "__main__":
    batch_size = 2
    seq_len = 8
    state_dim = 64   # Must be even.
    model_dim = 128
    num_layers = 4
    num_heads = 8
    hidden_dim = 256

    # Create the model.
    model = Model(state_dim, model_dim, num_layers, num_heads, hidden_dim)

    # Example inputs:
    # last_state: (B, T, state_dim)
    last_state = pt.randn(batch_size, seq_len, state_dim)

    # For positions, create increasing sequences per batch.
    # current_pos: e.g., [0,1,2,...,7] for each batch.
    current_pos = pt.arange(seq_len).repeat(batch_size, 1).float()
    # future_pos: e.g., [8,9,10,...,15] for each batch.
    future_pos = (pt.arange(seq_len) + seq_len).repeat(batch_size, 1).float()

    # Forward pass.
    output = model(last_state, current_pos, future_pos)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_len, state_dim)

