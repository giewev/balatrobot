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
from itertools import combinations


class ModalMultibinaryDist(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1), where a1 is a  and a2 is a multibinary action with exactly `a1` true entries."""

    def deterministic_sample(self):
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()
        # card_count_dist = self.count_distribution(mode)

        # print("mode", mode)
        # card_count = card_count_dist.sample()
        cards = self.sample_cards(mode, deterministic=True)

        action = torch.cat([mode.unsqueeze(1), cards], dim=1)
        # print("action", action)
        self._action_logp = self.logp(action)

        # print("action", action.shape)

        return action

    def sample(self):
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()
        # card_count_dist = self.count_distribution(mode)

        # card_count = card_count_dist.sample()
        cards = self.sample_cards(mode, deterministic=False)

        action = torch.cat([mode.unsqueeze(1), cards], dim=1)
        self._action_logp = self.logp(action)

        # print("action", action.shape)
        return action

    def mode_logits(self):
        return self.inputs[:, 32:34]

    def card_logits(self, mode):
        # in rows where mode is 1, use the play_card_logits
        # in rows where mode is 0, use the discard_card_logits
        # print(self.play_card_logits().shape, self.discard_card_logits().shape)
        mode = mode.unsqueeze(-1).unsqueeze(-1).expand(self.inputs.shape[0], 8, 2)
        return torch.where(
            mode == 1,
            self.play_card_logits(),
            self.discard_card_logits(),
        )

    def play_card_logits(self):
        return torch.stack(
            [
                self.inputs[:, :8],
                self.inputs[:, 8:16],
            ],
            dim=2,
        )

    def discard_card_logits(self):
        return torch.stack(
            [
                self.inputs[:, 16:24],
                self.inputs[:, 24:32],
            ],
            dim=2,
        )

    def positive_logits(self):
        return self.inputs[:, :8]

    def negative_logits(self):
        return self.inputs[:, 8:16]

    def mode_distribution(self):
        return TorchCategorical(self.mode_logits())

    def sample_cards(self, mode, deterministic):
        BATCH = self.inputs.shape[0]
        card_logits = self.card_logits(mode)
        # card_logits = self.card_logits()
        # cards = torch.zeros((BATCH, 8), device=self.inputs.device, dtype=torch.float32)

        # for i in range(BATCH):
        #     probs = torch.softmax(card_logits[i], dim=0)
        #     if deterministic:
        #         indices = torch.topk(probs, card_count[i].item()).indices
        #     else:
        #         indices = torch.multinomial(
        #             probs, card_count[i].item(), replacement=False
        #         )
        #     cards[i, indices] = 1
        # if deterministic:
        #     indices = torch.argmax(card_logits, dim=-1)
        #     cards[torch.arange(BATCH), indices] = 1
        # print("deterministic", cards.shape)
        # print(card_logits.shape)
        # else:
        # print(card_logits.shape)
        # print(card_logits.shape)
        probs = torch.softmax(card_logits, dim=-1)
        # print(probs.shape)
        cards = TorchCategorical(probs).sample()
        # print(cards.shape)
        # For any rows where the sum of the cards is 0, set a random card to 1
        # cards[torch.sum(cards, dim=1) == 0, torch.randint(0, 8, (1,))] = 1

        return cards

    def logp(self, actions):
        mode_dist = self.mode_distribution()

        mode = actions[:, 0]
        # card_count = actions[:, 1]
        cards = actions[:, 1:]
        # print(actions.shape, mode.shape, cards.shape)

        # card_counts_dist = self.count_distribution(mode)

        mode_logp = mode_dist.logp(mode)
        # card_counts_logp = card_counts_dist.logp(card_count)

        cards_logp = torch.sum(self.cards_logp(cards, mode), dim=-1)

        return mode_logp + cards_logp

    def cards_logp(self, cards, mode):
        BATCH = cards.shape[0]
        card_logits = self.card_logits(mode)

        # log_probs = F.log_softmax(card_logits, dim=-1)
        # probs = torch.exp(log_probs)

        # Initialize result_logp with zeros
        # result_logp = torch.zeros(BATCH, device=self.inputs.device, dtype=torch.float32)

        # Convert the binary vectors to indices
        # indices = torch.arange(8, device=self.inputs.device, dtype=torch.int64)
        # indices = indices.unsqueeze(0).expand(BATCH, -1)
        # indices = torch.where(cards.bool(), indices, torch.zeros_like(indices))
        # print(cards.shape, card_logits.shape)

        # print(cards.shape, card_logits.shape)
        # return torch.distributions.Multinomial(
        #     logits=card_logits, total_count=5
        # ).log_prob(cards)

        return TorchCategorical(torch.softmax(card_logits, dim=-1)).logp(cards)

        for i in range(int(card_counts.max().item())):
            mask = card_counts > i

            prob_just_selected = torch.where(
                cards.bool(), probs, torch.zeros_like(probs)
            )
            sum_prob_selected = prob_just_selected.sum(dim=-1, keepdim=True) + 1e-10

            log_prob_selected = torch.log(sum_prob_selected)

            result_logp = result_logp + log_prob_selected.squeeze() * mask.float()

            probs = probs * (1 - prob_just_selected)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return result_logp

    def entropy(self):
        BATCH = self.inputs.shape[0]
        mode_dist = self.mode_distribution()
        mode_entropy = mode_dist.entropy()
        mode_probs = torch.softmax(self.mode_logits(), dim=1)
        # count_entropy = torch.zeros(
        #     (BATCH,), device=self.inputs.device, dtype=torch.float32
        # )
        cards_entropy = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        for mode in range(2):
            # if mode == 1:
            #     count_dist = self.play_count_distribution()
            # else:
            #     count_dist = self.discard_count_distribution()

            # count_entropy += mode_probs[:, mode] * count_dist.entropy()
            cards_entropy += mode_probs[:, mode] * torch.sum(
                self.cards_entropy(mode), dim=-1
            )

        return mode_entropy + cards_entropy

    def cards_entropy(self, mode):
        BATCH = self.inputs.shape[0]
        if mode == 1:
            card_logits = self.play_card_logits()
        else:
            card_logits = self.discard_card_logits()

        return TorchCategorical(torch.softmax(card_logits, dim=-1)).entropy()

        # probs = torch.softmax(card_logits, dim=1)
        # entropy = torch.multinomial(probs, 5, replacement=True).entropy()
        # entropy = torch.distributions.Multinomial(
        #     logits=card_logits, total_count=5
        # ).entropy()

        # entropy = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        # for n_cards in range(1, 6):
        #     count_prob = count_probs[:, n_cards - 1]
        #     entropy += (
        #         count_prob
        #         * torch.distributions.Multinomial(
        #             logits=card_logits, total_count=n_cards
        #         ).entropy()
        #     )
        # base_probs = torch.softmax(card_logits, dim=1)
        # probs = base_probs.clone()
        # for x in range(n_cards):
        #     log_probs = torch.log(probs + 1e-10)
        #     entropy -= count_prob * torch.sum(probs * log_probs, dim=1)
        #     new_probs = probs * (1 - probs)
        #     probs = new_probs / new_probs.sum(dim=1, keepdim=True)
        # return entropy

    def kl(self, other):
        BATCH = self.inputs.shape[0]
        mode_dist = self.mode_distribution()
        mode_kl = mode_dist.kl(other.mode_distribution())

        mode_probs = torch.softmax(self.mode_logits(), dim=1)

        cards_kl = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        for mode in range(2):
            cards_kl += mode_probs[:, mode] * torch.sum(
                self.cards_kl(other, mode), dim=-1
            )
        return mode_kl + cards_kl

    def cards_kl(self, other, mode):
        BATCH = self.inputs.shape[0]

        # if mode == 1:
        #     self_card_logits = self.play_card_logits()
        #     other_card_logits = other.play_card_logits()
        # else:
        #     self_card_logits = self.discard_card_logits()
        #     other_card_logits = other.discard_card_logits()
        self_card_logits = self.card_logits(torch.tensor(mode).to(self.inputs.device))
        other_card_logits = other.card_logits(torch.tensor(mode).to(self.inputs.device))

        self_dist = torch.distributions.Categorical(logits=self_card_logits)
        other_dist = torch.distributions.Categorical(logits=other_card_logits)

        kl_div = torch.distributions.kl_divergence(self_dist, other_dist)

        # kl_div = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        # for n_cards in range(1, 6):
        #     masked_self_card_logits = self_card_logits.clone()
        #     masked_other_card_logits = other_card_logits.clone()
        #     for _ in range(n_cards):
        #         self_dist = torch.distributions.Categorical(
        #             logits=masked_self_card_logits
        #         )
        #         other_dist = torch.distributions.Categorical(
        #             logits=masked_other_card_logits
        #         )

        #         kl_div += self_count_prob * torch.distributions.kl_divergence(
        #             self_dist, other_dist
        #         )

        #     self_dist = torch.distributions.Categorical(logits=self_card_logits)
        #     other_dist = torch.distributions.Categorical(logits=other_card_logits)

        #     kl_div += self_count_prob * torch.distributions.kl_divergence(
        #         self_dist, other_dist
        #     )

        # self_probs = torch.softmax(self_card_logits, dim=1)
        # other_probs = torch.softmax(other_card_logits, dim=1)
        # self_log_probs = torch.log(self_probs + 1e-10)
        # other_log_probs = torch.log(other_probs + 1e-10)

        # for _ in range(n_cards):
        #     kl_term = self_probs * (self_log_probs - other_log_probs)
        #     kl_div += self_count_prob * torch.sum(kl_term, dim=1)

        return kl_div

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size
