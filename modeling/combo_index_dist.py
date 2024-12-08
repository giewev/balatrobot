import torch
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from gymnasium.spaces import Box
import torch.nn as nn
from itertools import combinations as combos
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ComboIndexDist(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1), where a1 is a  and a2 is a multibinary action with exactly `a1` true entries."""

    combo_map = {}
    reverse_combo_map = {}
    for num_cards in range(0, 6):
        combo_map[num_cards] = {}
        reverse_combo_map[num_cards] = {}
        c = list(combos(range(8), num_cards))
        for i, c in enumerate(c):
            combo_map[num_cards][c] = i
            reverse_combo_map[num_cards][i] = c

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.build_combo_probs()

    def build_combo_probs(self):
        BATCH = self.inputs.shape[0]
        max_combo_count = max(len(self.combo_map[i]) for i in range(6))
        self.combo_probs = torch.zeros(
            (BATCH, 2, 6, max_combo_count),
            dtype=torch.float32,
            device=self.inputs.device,
        )

        self.combo_probs[:, :, 0, 0] = 1.0

        for i in range(0, 5):
            for mode in range(2):
                for j in range(len(self.combo_map[i])):
                    self.add_sub_combo_probs(mode, i, j)

    def add_sub_combo_probs(self, mode, num_cards, combo_index):
        combo_indices = self.reverse_combo_map[num_cards][combo_index]
        card_logits = self.card_logits_immediate(mode)
        masked_logits = card_logits.clone()
        masked_logits[:, combo_indices] = FLOAT_MIN
        new_probs = torch.softmax(masked_logits, dim=1)
        base_prob = self.combo_probs[:, mode, num_cards, combo_index].clone()

        for i in range(0, 8):
            if i in combo_indices:
                continue
            new_indices = tuple(sorted(combo_indices + (i,)))
            new_combo_index = self.combo_map[num_cards + 1][new_indices]
            new_prob = base_prob * new_probs[:, i].clone()
            self.combo_probs[:, mode, num_cards + 1, new_combo_index] += new_prob

    def deterministic_sample(self):
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()
        card_count_dist = self.count_distribution(mode)

        card_counts = card_count_dist.sample()
        indices = self.sample_cards(card_counts + 1, mode, deterministic=True)

        action = torch.stack([mode, card_counts, indices], dim=1)
        self._action_logp = self.logp(action)

        return action

    def sample(self):
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()
        card_count_dist = self.count_distribution(mode)

        card_counts = card_count_dist.sample()
        indices = self.sample_cards(card_counts + 1, mode, deterministic=False)

        action = torch.stack([mode, card_counts, indices], dim=1)
        self._action_logp = self.logp(action)

        return action

    def mode_logits(self):
        return self.inputs[:, 16:18]

    def count_distribution(self, mode):
        play_count_logits = self.play_count_logits()
        discard_count_logits = self.discard_count_logits()
        count_logits = torch.where(
            mode.unsqueeze(1) == 1, play_count_logits, discard_count_logits
        )
        return TorchCategorical(count_logits)

    def play_count_logits(self):
        return self.inputs[:, 18:23]

    def discard_count_logits(self):
        return self.inputs[:, 23:28]

    def card_logits_immediate(self, mode):
        if mode == 0:
            return self.discard_card_logits()
        else:
            return self.play_card_logits()

    def card_logits(self, mode):
        return torch.where(
            mode.unsqueeze(1) == 1, self.play_card_logits(), self.discard_card_logits()
        )

    def play_card_logits(self):
        return self.inputs[:, :8]

    def discard_card_logits(self):
        return self.inputs[:, 8:16]

    def mode_distribution(self):
        return TorchCategorical(self.mode_logits())

    def play_count_distribution(self):
        return TorchCategorical(self.play_count_logits())

    def discard_count_distribution(self):
        return TorchCategorical(self.discard_count_logits())

    def sample_cards(self, card_count, mode, deterministic):
        BATCH = self.inputs.shape[0]
        indices = torch.distributions.Categorical(
            probs=self.combo_probs[torch.arange(BATCH), mode, card_count, :]
        ).sample()

        return indices

        # cards = torch.zeros((BATCH, 8), device=self.inputs.device, dtype=torch.float32)
        # for i in range(BATCH):
        #     combo = self.reverse_combo_map[card_count[i].item()][indices[i].item()]
        #     cards[i, combo] = 1
        # return cards

    def logp(self, actions):
        mode_dist = self.mode_distribution()

        modes = actions[:, 0]
        card_counts = actions[:, 1]
        indices = actions[:, 2]
        # card_count = cards.sum(dim=1) - 1
        # Dummy batch may choose 0 cards resulting in -1, so we clamp to 0
        # card_count = torch.clamp(card_count, 0, 4)

        card_counts_dist = self.count_distribution(modes)

        mode_logp = mode_dist.logp(modes)
        card_counts_logp = card_counts_dist.logp(card_counts)

        cards_logp = self.cards_logp(indices, card_counts + 1, modes)

        return mode_logp + card_counts_logp + cards_logp

    def cards_logp(self, indices, card_counts, modes):
        BATCH = indices.shape[0]

        # card_counts = cards.sum(dim=1)
        # combo_indices = torch.zeros(
        #     (BATCH,), device=self.inputs.device, dtype=torch.int64
        # )
        # for i in range(BATCH):
        #     combo = tuple(sorted([j for j in range(8) if cards[i, j] == 1]))
        #     card_count = len(combo)
        #     combo_index = self.combo_map[card_count][combo]
        #     combo_indices[i] = combo_index
        log_probs = torch.log(
            self.combo_probs[
                torch.arange(BATCH), modes.long(), card_counts.long(), indices
            ]
        )

        return log_probs

    def entropy(self):
        BATCH = self.inputs.shape[0]
        mode_dist = self.mode_distribution()
        mode_entropy = mode_dist.entropy()
        mode_probs = torch.softmax(self.mode_logits(), dim=1)
        count_entropy = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        cards_entropy = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        for mode in range(2):
            if mode == 1:
                count_dist = self.play_count_distribution()
            else:
                count_dist = self.discard_count_distribution()

            count_entropy += mode_probs[:, mode] * count_dist.entropy()
            cards_entropy += mode_probs[:, mode] * self.cards_entropy(count_dist, mode)

        return mode_entropy + count_entropy + cards_entropy

    def cards_entropy(self, count_dist, mode):
        BATCH = self.inputs.shape[0]
        count_probs = torch.softmax(count_dist.inputs, dim=1)
        entropy = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        for n_cards in range(1, 6):
            count_prob = count_probs[:, n_cards - 1]
            card_probs = self.combo_probs[:, mode, n_cards, :]
            dist = torch.distributions.Categorical(probs=card_probs)
            entropy += count_prob * dist.entropy()
        return entropy

    # def kl(self, other):
    #     BATCH = self.inputs.shape[0]
    #     mode_dist = self.mode_distribution()
    #     mode_kl = mode_dist.kl(other.mode_distribution())

    #     mode_probs = torch.softmax(self.mode_logits(), dim=1)
    #     count_kl = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
    #     cards_kl = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
    #     for mode in range(2):
    #         if mode == 1:
    #             count_kl += mode_probs[:, mode] * self.play_count_distribution().kl(
    #                 other.play_count_distribution()
    #             )
    #         else:
    #             count_kl += mode_probs[:, mode] * self.discard_count_distribution().kl(
    #                 other.discard_count_distribution()
    #             )

    #         cards_kl += mode_probs[:, mode] * self.cards_kl(other, mode)
    #     return mode_kl + count_kl + cards_kl

    def full_combo_dist(self):
        # The probability matrix is of shape (BATCH, 2, 6, max_combo_count)
        # where 2 is the number of modes, 6 is the number of card counts (including the 0 count)
        # and max_combo_count is the number of possible combinations of cards for a given count
        # We need to build a matrix of shape (BATCH, 2, 6) which represents the probability of each
        # mode and card count combination
        # Then we can multiply this matrix by the combo_probs matrix and flatten it to get the actual
        # probabilities of each combination
        BATCH = self.inputs.shape[0]
        mode_probs = torch.softmax(self.mode_logits(), dim=1)
        count_probs = torch.stack(
            [
                torch.softmax(self.play_count_logits(), dim=1),
                torch.softmax(self.discard_count_logits(), dim=1),
            ],
            dim=1,
        )
        # Need to pad with a 0 probability for the 0 card count
        count_probs = torch.cat(
            [
                torch.zeros(
                    (BATCH, 2, 1), dtype=torch.float32, device=self.inputs.device
                ),
                count_probs,
            ],
            dim=2,
        )

        mode_count_probs = torch.zeros(
            (BATCH, 2, 6), dtype=torch.float32, device=self.inputs.device
        )
        for mode in range(2):
            mode_count_probs[:, mode, :] = (
                mode_probs[:, mode].unsqueeze(1) * count_probs[:, mode, :]
            )

        adjusted_combo_probs = self.combo_probs * mode_count_probs.unsqueeze(3)
        adjusted_combo_probs = adjusted_combo_probs.view(BATCH, -1)

        adjusted_combo_dist = torch.distributions.Categorical(
            probs=adjusted_combo_probs
        )

        return adjusted_combo_dist

    def kl(self, other):
        adjusted_combo_dist = self.full_combo_dist()
        other_adjusted_combo_dist = other.full_combo_dist()
        return torch.distributions.kl.kl_divergence(
            adjusted_combo_dist, other_adjusted_combo_dist
        )

    def cards_kl(self, other, mode):
        BATCH = self.inputs.shape[0]

        if mode == 1:
            self_count_probs = torch.softmax(self.play_count_logits(), dim=1)
            other_count_probs = torch.softmax(other.play_count_logits(), dim=1)
        else:
            self_count_probs = torch.softmax(self.discard_count_logits(), dim=1)
            other_count_probs = torch.softmax(other.discard_count_logits(), dim=1)

        kl_div = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        for n_cards in range(1, 6):
            self_count_prob = self_count_probs[:, n_cards - 1]
            other_count_prob = other_count_probs[:, n_cards - 1]

            self_dist = torch.distributions.Categorical(
                probs=self.combo_probs[:, mode, n_cards, :]
            )
            other_dist = torch.distributions.Categorical(
                probs=other.combo_probs[:, mode, n_cards, :]
            )

            kl_div += self_count_prob * torch.distributions.kl.kl_divergence(
                self_dist, other_dist
            )

        return kl_div

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size
