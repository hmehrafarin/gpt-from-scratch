import torch
import torch.nn as nn
from config import AttentionConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        # We're doing single linear projection for query, key and
        # value (more efficient!?)
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # Our W(o) projection to model dimension after attention
        # score computation
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # We need a triangular matrix to make sure current tokens
        # Do not attend to future ones
        mask = torch.tril(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=0
        )  # has no gradient, won't move to gpu

        self.register_buffer("mask", mask)  # makes it available in state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get batch, seq_length and model dimension
        B, T, C = x.shape

        # We project and then split
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.cfg.d_model, dim=-1)  # (B, T, C) each

        H = self.cfg.n_heads
        dh = self.cfg.d_head
        # We use view because of cost, view is basically a zero-cost copy
        # it doesn't move anything in the memory
        q = q.view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)
        k = k.view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)
        v = v.view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)

        scale = dh**-0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Now we apply causal mask to avoid tokens to atteend to future ones
        # we cut until the input sequence length and make sure we replace 0
        # by -inf for softmax so we avoid giving weight to 0 value tokens
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        weights = self.dropout(weights)
        """
        Becomes the following:

        tok_0 [  1.0    0.0    0.0  ]
        tok_1 [  0.31   0.69   0.0  ]
        tok_2 [  0.23   0.50   0.27 ]

        each tok_i value now sums up to one
        """
        out = weights @ v  # (B, H, T, dh)
        out = out.transpose(1, 2).contiguous()  # (B, T, H, dh)
        out = out.view(B, T, C)

        return self.out_proj(out)
