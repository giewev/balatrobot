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


class NChooseKDistribution(TorchDistributionWrapper):
    """Action distribution P(a1, a2) = P(a1) * P(a2 | a1), where a1 is a  and a2 is a multibinary action with exactly `a1` true entries."""

    def deterministic_sample(self):
        a1_play_discard_dist, a1_card_count_dist = self._a1_distributions()
        a1_play_discard = a1_play_discard_dist.deterministic_sample()
        a1_card_count = a1_card_count_dist.deterministic_sample()

        a1 = torch.stack([a1_play_discard, a1_card_count], dim=1)

        a2 = self._sample_a2_deterministically(a1_play_discard, a1_card_count)
        self._action_logp = self.logp(torch.concatenate([a1, a2], dim=1))

        return (a1_play_discard, a1_card_count, a2)

    def sample(self):
        a1_play_discard_dist, a1_card_count_dist = self._a1_distributions()
        a1_play_discard = a1_play_discard_dist.sample()
        a1_card_count = a1_card_count_dist.sample()

        a1 = torch.stack([a1_play_discard, a1_card_count], dim=1)

        a2 = self._sample_a2(a1_play_discard, a1_card_count)
        self._action_logp = self.logp(torch.concatenate([a1, a2], dim=1))

        return (a1_play_discard, a1_card_count, a2)

    def logp(self, actions):
        a1_play_discard, a1_card_count, a2 = (
            actions[:, 0],
            actions[:, 1],
            actions[:, 2:],
        )

        a1_logits = self.model.a1_logits(self.inputs)
        play_discard_logits = a1_logits[:, :2]
        card_count_logits = a1_logits[:, 2:]

        a1_play_discard_logp = TorchCategorical(play_discard_logits).logp(
            a1_play_discard
        )
        a1_card_count_logp = TorchCategorical(card_count_logits).logp(a1_card_count)

        a2_logits = self._a2_logits(a1_play_discard, a1_card_count)
        a2_logp = self._a2_logp(a2, a2_logits, a1_card_count)

        return a1_play_discard_logp + a1_card_count_logp + a2_logp

    def entropy(self):
        a1_play_discard_dist, a1_card_count_dist = self._a1_distributions()
        a2_dist = self._a2_distribution(
            a1_play_discard_dist.sample(), a1_card_count_dist.sample()
        )

        return (
            a1_play_discard_dist.entropy()
            + a1_card_count_dist.entropy()
            + a2_dist.entropy()
        )

    def sampled_action_logp(self):
        return self._action_logp

    def kl(self, other):
        a1_play_discard_dist, a1_card_count_dist = self._a1_distributions()
        other_a1_play_discard_dist, other_a1_card_count_dist = other._a1_distributions()

        a1_play_discard_terms = a1_play_discard_dist.kl(other_a1_play_discard_dist)
        a1_card_count_terms = a1_card_count_dist.kl(other_a1_card_count_dist)

        a1_terms = a1_play_discard_terms + a1_card_count_terms

        play_discard = a1_play_discard_dist.sample()
        card_count = a1_card_count_dist.sample()
        a2_terms = self._a2_distribution(play_discard, card_count).kl(
            other._a2_distribution(play_discard, card_count)
        )

        return a1_terms + a2_terms

    def _a1_distributions(self):
        a1_logits = self.model.a1_logits(self.inputs)
        play_discard_logits = a1_logits[:, :2]
        card_count_logits = a1_logits[:, 2:]

        play_discard_dist = TorchCategorical(play_discard_logits)
        card_count_dist = TorchCategorical(card_count_logits)
        return play_discard_dist, card_count_dist

    def _a2_distribution(self, play_discard, card_count):
        a2_logits = self._a2_logits(play_discard, card_count)
        a2_dist = TorchCategorical(a2_logits)
        return a2_dist

    def _a2_logits(self, play_discard, card_count):
        a1_vec = torch.stack([play_discard, card_count], dim=1)
        a2_logits = self.model.a2_logits(self.inputs, a1_vec)
        return a2_logits

    def _sample_a2(self, play_discard, card_count):
        BATCH = self.inputs.shape[0]
        a2_logits = self._a2_logits(play_discard, card_count)
        a2 = torch.zeros(
            (BATCH, a2_logits.shape[1]), device=self.inputs.device, dtype=torch.float32
        )

        for i in range(BATCH):
            probs = torch.softmax(a2_logits[i], dim=0)
            indices = torch.multinomial(
                probs, card_count[i].item() + 1, replacement=False
            )
            a2[i, indices] = 1

        return a2

    def _sample_a2_deterministically(self, play_discard, card_count):
        BATCH = self.inputs.shape[0]
        a2_logits = self._a2_logits(play_discard, card_count)
        a2 = torch.zeros(
            (BATCH, a2_logits.shape[1]), device=self.inputs.device, dtype=torch.float32
        )

        for i in range(BATCH):
            _, indices = torch.topk(a2_logits[i], card_count[i].item() + 1)
            a2[i, indices] = 1

        return a2

    def _a2_logp(self, a2, a2_logits, card_count):
        logp = torch.zeros(a2.shape[0], device=a2.device)

        for i in range(a2.shape[0]):
            true_indices = torch.nonzero(a2[i], as_tuple=False)
            if len(true_indices) == card_count[i].item() + 1:
                logits_true = a2_logits[i, true_indices]
                logp[i] = torch.sum(F.log_softmax(logits_true, dim=0))
            else:
                logp[i] = -float("inf")

        return logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size
