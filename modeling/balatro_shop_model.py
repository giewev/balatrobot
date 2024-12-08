import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.torch_utils import FLOAT_MIN


class BalatroShopModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        context_size = 64
        lstm_hidden_size = 128

        # 0) End shop
        # 1) Purchase left card
        # 2) Purchase right card
        # 3) Re-roll shop
        # 4-8) Sell joker 1-5
        num_actions = 9

        self.joker_layer = nn.LSTM(
            input_size=172, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True
        )

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_layer_1 = nn.Linear(lstm_hidden_size * 2 + 3, 128)
        self.context_layer = nn.Linear(128, context_size)
        self._last_context = None
        self.value_layer = nn.Linear(context_size, 1)
        self.action_layer = nn.Linear(context_size, num_actions)

    def forward(self, input_dict, state, seq_lens):
        non_sequence = ["dollars", "owned_joker_count", "max_jokers"]
        non_sequence = torch.stack(
            [input_dict["obs"][feature] for feature in non_sequence], dim=1
        )  # [?, 3]

        owned_joker_obs = input_dict["obs"]["owned_jokers"]
        owned_joker_count_obs = input_dict["obs"]["owned_joker_count"]
        owned_joker_ht = self.process_joker_seq(owned_joker_obs, owned_joker_count_obs)

        shop_joker_obs = input_dict["obs"]["shop_cards"]
        shop_joker_count_obs = input_dict["obs"]["shop_card_count"]
        shop_joker_ht = self.process_joker_seq(shop_joker_obs, shop_joker_count_obs)

        combined_output = torch.cat(
            (owned_joker_ht, shop_joker_ht, non_sequence), dim=1
        )

        hidden_output = self.relu(self.hidden_layer_1(combined_output))
        self._last_context = self.relu(self.context_layer(hidden_output))
        action_logits = self.action_layer(self._last_context)

        # Mask out invalid actions
        inf_mask = torch.clamp(
            torch.log(input_dict["obs"]["action_mask"]), min=FLOAT_MIN
        )
        action_logits += inf_mask

        return action_logits, []

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)

    def process_joker_seq(self, jokers, joker_counts):
        joker_costs = jokers.values["cost"]
        joker_costs = joker_costs.unsqueeze(2)
        joker_names = jokers.values["name"]
        joker_rarities = jokers.values["rarity"]
        joker_flags = jokers.values["flags"]
        jokers = torch.cat(
            (joker_costs, joker_names, joker_rarities, joker_flags), dim=2
        )

        joker_counts = torch.where(
            joker_counts == 0, torch.ones_like(joker_counts), joker_counts
        )
        packed_jokers = nn.utils.rnn.pack_padded_sequence(
            jokers, joker_counts.to("cpu"), batch_first=True, enforce_sorted=False
        )
        joker_output, (joker_ht, joker_ct) = self.joker_layer(packed_jokers)
        joker_ht = joker_ht.squeeze(0)

        return joker_ht
