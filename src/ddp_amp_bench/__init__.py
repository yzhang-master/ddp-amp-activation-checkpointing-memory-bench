"""BERT-style training with DDP, AMP and activation checkpointing."""

__version__ = "0.1.0"


def activation_elements(batch_size: int, sequence_length: int, hidden_size: int, depth: int) -> int:
    """Count hidden-state elements stored by one copy of every encoder block."""
    if min(batch_size, sequence_length, hidden_size, depth) <= 0:
        raise ValueError("activation dimensions must be positive")
    return batch_size * sequence_length * hidden_size * depth
