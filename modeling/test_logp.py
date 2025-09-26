import math

import pytest

torch = pytest.importorskip("torch", reason="torch is required for logp tests")
import torch.nn.functional as F


def cards_logp(cards, card_counts, logits):
    batch = cards.shape[0]

    log_probs = F.log_softmax(logits, dim=-1)
    masked = torch.where(cards.bool(), log_probs, torch.zeros_like(log_probs))
    total_logp = masked.sum(dim=1)

    hand_size = torch.full((batch,), logits.shape[-1], dtype=torch.float32)
    comb_factor = torch.lgamma(hand_size + 1) - (
        torch.lgamma(card_counts + 1) + torch.lgamma(hand_size - card_counts + 1)
    )

    return total_logp + comb_factor


def test_cards_logp_partial_selection():
    cards = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    counts = torch.tensor([4], dtype=torch.float32)
    logits = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32)

    result = cards_logp(cards, counts, logits).item()
    expected = -13.584863263866659
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_cards_logp_all_cards():
    cards = torch.ones((1, 8), dtype=torch.float32)
    counts = torch.tensor([8], dtype=torch.float32)
    logits = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32)

    result = cards_logp(cards, counts, logits).item()
    expected = -31.666717011832027
    assert math.isclose(result, expected, rel_tol=1e-6)
