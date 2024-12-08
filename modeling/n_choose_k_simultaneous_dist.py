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


class NChooseKSimultaneousDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1), where a1 is a  and a2 is a multibinary action with exactly `a1` true entries."""

    def deterministic_sample(self):
        play_discard_dist = self.play_discard_distribution()
        card_count_dist = self.count_distribution()

        play_discard = play_discard_dist.deterministic_sample()
        card_count = card_count_dist.deterministic_sample()
        cards = self.sample_cards(card_count + 1, deterministic=True)

        action = torch.cat(
            [play_discard.unsqueeze(1), card_count.unsqueeze(1), cards], dim=1
        )
        self._action_logp = self.logp(action)

        return action

    def sample(self):
        play_discard_dist = self.play_discard_distribution()
        card_count_dist = self.count_distribution()

        play_discard = play_discard_dist.sample()
        card_count = card_count_dist.sample()
        cards = self.sample_cards(card_count + 1, deterministic=False)

        action = torch.cat(
            [play_discard.unsqueeze(1), card_count.unsqueeze(1), cards], dim=1
        )
        self._action_logp = self.logp(action)

        return action

    def play_discard_logits(self):
        return self.inputs[:, 8:10]

    def count_logits(self):
        return self.inputs[:, 10:12]

    def card_logits(self):
        return self.inputs[:, :8]

    def play_discard_distribution(self):
        return TorchCategorical(self.play_discard_logits())

    def count_distribution(self):
        return TorchCategorical(self.count_logits())

    def sample_cards(self, card_count, deterministic):
        BATCH = self.inputs.shape[0]
        card_logits = self.card_logits()
        cards = torch.zeros((BATCH, 8), device=self.inputs.device, dtype=torch.float32)

        for i in range(BATCH):
            probs = torch.softmax(card_logits[i], dim=0)
            if deterministic:
                indices = torch.topk(probs, card_count[i].item()).indices
            else:
                indices = torch.multinomial(
                    probs, card_count[i].item(), replacement=False
                )
            cards[i, indices] = 1
        return cards

    def logp(self, actions):
        play_discard_dist = self.play_discard_distribution(self.inputs)
        card_counts_dist = self.count_distribution(self.inputs)

        play_discard = actions[:, 0]
        card_counts = actions[:, 1]
        cards = actions[:, 2:]

        play_discard_logp = play_discard_dist.logp(play_discard)
        card_counts_logp = card_counts_dist.logp(card_counts)
        cards_logp = self.cards_logp(cards, card_counts)

        return play_discard_logp + card_counts_logp + cards_logp

    def cards_logp(self, cards, card_counts):
        BATCH = cards.shape[0]
        card_logits = self.card_logits()

        probs = F.log_softmax(card_logits, dim=-1)
        total_logp = torch.sum(
            torch.where(cards.bool(), probs, torch.zeros_like(probs)), dim=1
        )

        hand_size = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        hand_size += 8
        comb_factor = torch.lgamma(hand_size + 1) - (
            torch.lgamma(card_counts + 1) + torch.lgamma(hand_size - card_counts + 1)
        )

        # Adjust total log-probability by the combinatorial factor
        adjusted_logp = total_logp + comb_factor

        return adjusted_logp

    def entropy(self):
        play_discard_dist = self.play_discard_distribution(self.inputs)
        card_count_dist = self.count_distribution(self.inputs)

        play_discard_entropy = play_discard_dist.entropy()
        card_count_entropy = card_count_dist.entropy()
        cards_entropy = self.cards_entropy(card_count_dist)

        return play_discard_entropy + card_count_entropy + cards_entropy

    def cards_entropy(self, card_count_dist):
        BATCH = self.inputs.shape[0]
        card_logits = self.card_logits()
        count_probs = torch.softmax(card_count_dist.inputs, dim=1)

        entropy = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        for n_cards in range(1, 6):
            count_prob = count_probs[:, n_cards - 1]
            base_probs = torch.softmax(card_logits, dim=1)
            probs = base_probs.detach().clone()
            for x in range(n_cards):
                log_probs = torch.log(probs)
                entropy += count_prob * torch.sum(probs * log_probs, dim=1)
                probs *= 1 - probs
                probs = probs / probs.sum(dim=1, keepdim=True)
        return -entropy

    def kl(self, other):
        play_discard_dist = self.play_discard_distribution(self.inputs)
        card_count_dist = self.count_distribution(self.inputs)

        play_discard_other = other.play_discard_distribution(other.inputs)
        card_count_other = other.count_distribution(other.inputs)

        play_discard_kl = play_discard_dist.kl(play_discard_other)
        card_count_kl = card_count_dist.kl(card_count_other)
        cards_kl = self.cards_kl(other)

        return play_discard_kl + card_count_kl + cards_kl

    def cards_kl(self, other):
        BATCH = self.inputs.shape[0]
        self_card_logits = self.card_logits()
        other_card_logits = other.card_logits()

        self_count_probs = torch.softmax(self.count_logits(), dim=1)
        other_count_probs = torch.softmax(other.count_logits(), dim=1)

        kl_div = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)

        for n_cards in range(1, 6):
            self_count_prob = self_count_probs[:, n_cards - 1]
            other_count_prob = other_count_probs[:, n_cards - 1]

            self_base_probs = torch.softmax(self_card_logits, dim=1)
            other_base_probs = torch.softmax(other_card_logits, dim=1)

            self_probs = self_base_probs.detach().clone()
            other_probs = other_base_probs.detach().clone()

            for _ in range(n_cards):
                self_log_probs = torch.log(self_probs)
                other_log_probs = torch.log(other_probs)

                kl_term = self_probs * (self_log_probs - other_log_probs)
                kl_div += self_count_prob * torch.sum(kl_term, dim=1)

                # Maintain probability normalization
                self_probs = self_probs / self_probs.sum(dim=1, keepdim=True)
                other_probs = other_probs / other_probs.sum(dim=1, keepdim=True)

        return kl_div

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size
