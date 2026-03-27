import torch
import torch.nn as nn
from config import GPTConfig


class Embeddings(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # define embedding layer
        self.token_embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size, embedding_dim=cfg.d_model
        )

        self.pos_embedding = nn.Embedding(
            num_embeddings=cfg.max_seq_len, embedding_dim=cfg.d_model
        )

        # Register buffer for the token positions
        pos = torch.arange(self.cfg.max_seq_len)
        self.register_buffer("positions", pos)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # pick the second dimension to get the sequence length
        T = token_ids.shape[-1]  # seq_length

        # get embeddings output by passing the input through the embedding layer
        token_embedding_output = self.token_embedding(token_ids)

        # get position embeddings by passing the positions to the pos_embeddings
        pos_embeddings_output = self.pos_embedding(self.positions[:T])

        # add pos embeddings to embedding output
        out = token_embedding_output + pos_embeddings_output

        return out
