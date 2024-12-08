import torch
import torch.nn.functional as F


def cards_logp(cards, card_counts, logits):
    BATCH = cards.shape[0]

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    total_logp = torch.sum(
        torch.where(cards.bool(), log_probs, torch.zeros_like(log_probs)), dim=1
    )

    hand_size = torch.zeros((BATCH,), dtype=torch.float32)
    hand_size += 8
    comb_factor = torch.lgamma(hand_size + 1) - (
        torch.lgamma(card_counts + 1) + torch.lgamma(hand_size - card_counts + 1)
    )

    adjusted_logp = total_logp + comb_factor

    return adjusted_logp


print(
    cards_logp(
        torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32),
        torch.tensor([4]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32),
    ).tolist()
)

print(
    cards_logp(
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32),
        torch.tensor([8]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float32),
    ).tolist()
)
