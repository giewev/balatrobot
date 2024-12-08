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


class MixedLogitsDist(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1), where a1 is a  and a2 is a multibinary action with exactly `a1` true entries."""

    def deterministic_sample(self):
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()
        card_count_dist = self.count_distribution(mode)

        card_count = card_count_dist.sample()
        cards = self.sample_cards(card_count + 1, mode, deterministic=True)

        action = torch.cat([mode.unsqueeze(1), cards], dim=1)
        self._action_logp = self.logp(action)

        return action

    def sample(self):
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()
        card_count_dist = self.count_distribution(mode)

        card_count = card_count_dist.sample()
        cards = self.sample_cards(card_count + 1, mode, deterministic=False)

        action = torch.cat([mode.unsqueeze(1), cards], dim=1)
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
        card_logits = self.card_logits(mode)
        # card_logits = self.card_logits()
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
        mode_dist = self.mode_distribution()

        mode = actions[:, 0]
        # card_count = actions[:, 1]
        cards = actions[:, 1:]
        card_count = cards.sum(dim=1) - 1
        # Dummy batch may choose 0 cards resulting in -1, so we clamp to 0
        card_count = torch.clamp(card_count, 0, 4)

        card_counts_dist = self.count_distribution(mode)

        mode_logp = mode_dist.logp(mode)
        card_counts_logp = card_counts_dist.logp(card_count)

        cards_logp = self.cards_logp(cards, mode)

        return mode_logp + card_counts_logp + cards_logp

    def cards_logp(self, cards, mode):
        BATCH = cards.shape[0]
        card_logits = self.card_logits(mode)

        # This multinomial calculation assumes "with replacement"
        # This causes some serious math problems
        # Because it doesn't account for the fact that the cards are drawn without replacement
        # And that the probabilities change as cards are drawn
        # So a card with a low probability of being drawn initially will have a higher probability of being drawn later
        # return torch.distributions.Multinomial(
        #     logits=card_logits, total_count=5
        # ).log_prob(cards)

        # Alternatively, we can iteratively "sample" each card
        # Re-normalizing the probabilities after each card is drawn
        # This is a more accurate representation of the actual process
        # But it's also much more computationally expensive
        # And still doesn't fully account for multiple ways to sample the same hand
        log_probs = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        remaining_cards_to_sample = cards.clone()
        iterated_logits = card_logits.clone()
        i = 0
        while remaining_cards_to_sample.sum() > 0:
            i += 1
            # print(i)
            if i > 10:
                raise Exception("Too many iterations in cards_logp")
            has_cards_mask = remaining_cards_to_sample.sum(dim=-1) > 0
            # print(has_cards_mask)
            all_card_probs = torch.softmax(iterated_logits, dim=-1)
            # print(all_card_probs)
            # Probability that we would choose ANY of the remaining cards in the action
            chooseable_card_probs = torch.sum(
                all_card_probs.where(remaining_cards_to_sample == 1, 1e-10), dim=-1
            )
            log_probs += torch.where(
                has_cards_mask, torch.log(chooseable_card_probs), 0
            )

            sampleable_logits = iterated_logits.where(
                remaining_cards_to_sample == 1, FLOAT_MIN
            )
            # sampleable_probs = torch.softmax(sampleable_logits, dim=1)
            next_cards = torch.distributions.Categorical(
                logits=sampleable_logits
            ).sample()
            # print(next_cards)
            remaining_cards_to_sample[torch.arange(BATCH), next_cards] = 0
            iterated_logits[torch.arange(BATCH), next_cards] = FLOAT_MIN
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
        if mode == 1:
            card_logits = self.play_card_logits()
        else:
            card_logits = self.discard_card_logits()

        count_probs = torch.softmax(count_dist.inputs, dim=1)

        raw_entropy = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        count_adjusted_entropy = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        iterated_logits = card_logits.clone()
        for n_cards in range(1, 6):
            count_prob = count_probs[:, n_cards - 1]
            dist = torch.distributions.Categorical(logits=iterated_logits)
            entropy = count_prob * dist.entropy()
            raw_entropy = raw_entropy.clone()
            raw_entropy += entropy
            count_adjusted_entropy = count_adjusted_entropy.clone()
            count_adjusted_entropy += count_prob * raw_entropy

            sampled_choices = dist.sample()
            iterated_logits = iterated_logits.clone()
            iterated_logits[torch.arange(BATCH), sampled_choices] = FLOAT_MIN
        return count_adjusted_entropy

    def kl(self, other):
        BATCH = self.inputs.shape[0]
        mode_dist = self.mode_distribution()
        mode_kl = mode_dist.kl(other.mode_distribution())

        mode_probs = torch.softmax(self.mode_logits(), dim=1)
        count_kl = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        cards_kl = torch.zeros((BATCH,), device=self.inputs.device, dtype=torch.float32)
        for mode in range(2):
            if mode == 1:
                count_kl += mode_probs[:, mode] * self.play_count_distribution().kl(
                    other.play_count_distribution()
                )
            else:
                count_kl += mode_probs[:, mode] * self.discard_count_distribution().kl(
                    other.discard_count_distribution()
                )

            cards_kl += mode_probs[:, mode] * self.cards_kl(other, mode)
        return mode_kl + count_kl + cards_kl

    def cards_kl(self, other, mode):
        BATCH = self.inputs.shape[0]

        if mode == 1:
            self_card_logits = self.play_card_logits()
            other_card_logits = other.play_card_logits()
            self_count_probs = torch.softmax(self.play_count_logits(), dim=1)
            other_count_probs = torch.softmax(other.play_count_logits(), dim=1)
        else:
            self_card_logits = self.discard_card_logits()
            other_card_logits = other.discard_card_logits()
            self_count_probs = torch.softmax(self.discard_count_logits(), dim=1)
            other_count_probs = torch.softmax(other.discard_count_logits(), dim=1)

        raw_kl_div = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        count_adjusted_kl_div = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        self_iterated_logits = self_card_logits.clone()
        other_iterated_logits = other_card_logits.clone()
        for n_cards in range(1, 6):
            self_count_prob = self_count_probs[:, n_cards - 1]
            other_count_prob = other_count_probs[:, n_cards - 1]

            self_dist = torch.distributions.Categorical(logits=self_iterated_logits)
            other_dist = torch.distributions.Categorical(logits=other_iterated_logits)

            raw_kl_div = raw_kl_div.clone()
            raw_kl_div += torch.distributions.kl.kl_divergence(self_dist, other_dist)
            count_adjusted_kl_div = count_adjusted_kl_div.clone()
            count_adjusted_kl_div += self_count_prob * raw_kl_div

            self_sampled_choices = self_dist.sample()
            # other_sampled_choices = other_dist.sample()
            # We use the sample from our distribution only
            # To prevent different -INF masks from causing issues
            self_iterated_logits = self_iterated_logits.clone()
            self_iterated_logits[torch.arange(BATCH), self_sampled_choices] = FLOAT_MIN
            other_iterated_logits = other_iterated_logits.clone()
            other_iterated_logits[torch.arange(BATCH), self_sampled_choices] = FLOAT_MIN

        return count_adjusted_kl_div

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size
