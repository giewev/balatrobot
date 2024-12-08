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

from ray.rllib.utils.torch_utils import FLOAT_MIN


class SequentialChoiceDistribution(TorchDistributionWrapper):
    """Action distribution where the model selects on card at a time, as one combined action"""

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self._action_logp = None
        self.hand_actions = 8
        self.play_discard_actions = 2
        self._last_sampled_actions = None
        self._last_initial_logits = None

    def deterministic_sample(self):
        return self._sample(deterministic=True)

    def sample(self):
        return self._sample(deterministic=False)

    def _sample(self, deterministic):
        sampled_actions = self.sample_initial_card(deterministic=deterministic)
        while not self.all_finished(sampled_actions):
            sampled_actions = self.sample_next(
                sampled_actions, deterministic=deterministic
            )

        self._action_logp = self.logp(sampled_actions)
        self._last_sampled_actions = sampled_actions

        return sampled_actions

    def sample_initial_card(self, deterministic):
        BATCH = self.inputs.shape[0]
        action_vector_length = self.hand_actions + self.play_discard_actions

        action_vector = torch.zeros(
            (BATCH, action_vector_length),
            device=self.inputs.device,
            dtype=torch.float32,
        )
        logits = self.model.action_module(self.inputs, action_vector)
        self._last_initial_logits = logits

        # Select options from the cards, ignoring play, discard
        # Since at least one card must always be selected
        logits = logits[:, : self.hand_actions]
        action = TorchCategorical(logits).sample()

        # Set the selected card to 1 in the action vector
        action_vector[torch.arange(action_vector.shape[0]), action] = 1

        return action_vector

    def sample_next(self, sampled_actions, deterministic):
        BATCH = self.inputs.shape[0]

        new_logits = self.model.action_module(self.inputs, sampled_actions)
        next_action = TorchCategorical(new_logits).sample()

        for i in range(BATCH):
            selected_row = sampled_actions[i].unsqueeze(0)
            if self.all_finished(selected_row):
                continue

            # new_logits = self.model.action_module(self.inputs, selected_row)
            # next_action = TorchCategorical(new_logits).sample()
            selected_new_action = next_action[i].item()

            # Set the selected card to 1 in the action vector
            sampled_actions[i, selected_new_action] = 1

        return sampled_actions

    def all_finished(self, sampled_actions):
        BATCH = sampled_actions.shape[0]
        return torch.sum(sampled_actions[:, self.hand_actions :]) == BATCH

    def check_safe(self, tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: tensor {name} contains NaN or inf values")

        # I think this is not mathematically correct
        # because it is not determining the probability of each flag
        # conditional on the previous flags, but rather the probability
        # of the entire sequence of flags (less the play/discard flags)
        # If this needs to be fixed, I'll have to either assume an arbitrary
        # sampling order, or store the sampling order as part of the action

    def logp(self, actions):
        self.check_safe(actions, "actions")

        BATCH = actions.shape[0]
        logp = torch.zeros(actions.shape[0], device=actions.device)
        logits = self.model.action_module(self.inputs, torch.zeros_like(actions))
        self.check_safe(logits, "logits")
        probs = F.softmax(logits, dim=1)
        self.check_safe(probs, "probs")

        mask = logits > FLOAT_MIN
        probs = torch.where(mask, torch.clamp(probs, 1e-5, 1 - 1e-5), probs)

        log_probs = torch.log(probs)
        log_probs_neg = torch.log(1 - probs)
        self.check_safe(log_probs, "log_probs")
        self.check_safe(log_probs_neg, "log_probs_neg")

        logp = actions * log_probs + (1 - actions) * log_probs_neg
        logp = torch.sum(logp, dim=1)

        self.check_safe(logp, "logp")

        # Check for extremely large or small values and print a warning
        if torch.any(logp >= 0) or torch.any(logp < -1000):
            print("Warning: logp contains extremely large or small values")
            print(logp)
        return logp

    def entropy(self):
        # Similarly simplified entropy calculation
        BATCH = self.inputs.shape[0]
        action_vector_length = self.hand_actions + self.play_discard_actions

        action_vector = torch.zeros(
            (BATCH, action_vector_length),
            device=self.inputs.device,
            dtype=torch.float32,
        )

        logits = self.model.action_module(self.inputs, action_vector)
        logits = logits[:, : self.hand_actions]

        entropy = TorchCategorical(logits).entropy()

        # Check for any nan or inf values and print a warning
        if torch.isnan(entropy).any() or torch.isinf(entropy).any():
            print("Warning: entropy contains NaN or inf values")

        return entropy

    def sampled_action_logp(self):
        return self._action_logp

    def kl(self, other):
        # Heavily simplified KL calculation which only considers the initial card selection

        BATCH = self.inputs.shape[0]
        action_vector_length = self.hand_actions + self.play_discard_actions
        card_logits = self.model.action_module(
            self.inputs,
            torch.zeros((BATCH, action_vector_length)).to(self.inputs.device),
        )
        card_dist = TorchCategorical(card_logits)
        self.check_safe(card_logits, "card_logits")

        other_card_logits = other.model.action_module(
            other.inputs,
            torch.zeros((BATCH, action_vector_length)).to(self.inputs.device),
        )
        self.check_safe(other_card_logits, "other_card_logits")
        other_card_dist = TorchCategorical(other_card_logits)
        kl = card_dist.kl(other_card_dist)
        # simulate choosing 5 times instead of 1
        kl *= 5

        # Check for any nan or inf values and print a warning
        if torch.isnan(kl).any() or torch.isinf(kl).any():
            print("Warning: kl contains NaN or inf values")
        return kl

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size
