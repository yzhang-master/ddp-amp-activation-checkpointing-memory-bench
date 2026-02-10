from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from ddp_amp_bench.model import BertConfig, BertMLM


def tiny_config() -> BertConfig:
    return BertConfig(
        vocab_size=48,
        hidden_size=24,
        num_heads=4,
        intermediate_size=48,
        depth=2,
        max_sequence_length=16,
    )


def test_checkpointed_forward_and_backward_match_eager() -> None:
    torch.manual_seed(4)
    eager = BertMLM(tiny_config())
    recomputed = BertMLM(tiny_config(), checkpointing=True)
    recomputed.load_state_dict(eager.state_dict())
    tokens = torch.randint(0, 48, (3, 12))
    targets = torch.randint(0, 48, (3, 12))
    eager_loss = torch.nn.functional.cross_entropy(
        eager(tokens).flatten(0, 1), targets.flatten()
    )
    checkpointed_loss = torch.nn.functional.cross_entropy(
        recomputed(tokens).flatten(0, 1), targets.flatten()
    )
    torch.testing.assert_close(checkpointed_loss, eager_loss)
    eager_loss.backward()
    checkpointed_loss.backward()
    for (name, expected), (_, actual) in zip(
        eager.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6, msg=name)


def test_training_invokes_checkpoint_once_per_block() -> None:
    model = BertMLM(tiny_config(), checkpointing=True)
    tokens = torch.randint(0, 48, (2, 8))
    with patch("ddp_amp_bench.model.checkpoint", wraps=checkpoint) as wrapped:
        model(tokens).mean().backward()
    assert wrapped.call_count == model.config.depth


def test_eval_does_not_recompute_and_tied_head_stays_tied() -> None:
    model = BertMLM(tiny_config(), checkpointing=True).eval()
    tokens = torch.randint(0, 48, (2, 8))
    with patch("ddp_amp_bench.model.checkpoint", wraps=checkpoint) as wrapped:
        model(tokens)
    assert wrapped.call_count == 0
    assert model.lm_head.weight is model.token_embeddings.weight
