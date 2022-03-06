from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    num_heads: int = 4
    intermediate_size: int = 128
    depth: int = 2
    max_sequence_length: int = 32
    dropout: float = 0.0

    def validate(self) -> None:
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.num_heads,
            self.intermediate_size,
            self.depth,
            self.max_sequence_length,
        )
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


class EncoderBlock(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attention = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        hidden = hidden + self.dropout(attention)
        return hidden + self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))


class BertMLM(nn.Module):
    """Compact pre-norm BERT-style encoder with optional per-block recomputation."""

    def __init__(self, config: BertConfig, *, checkpointing: bool = False) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.checkpointing = checkpointing
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embeddings(token_ids) + self.position_embeddings(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))
