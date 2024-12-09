import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet


class BalatroBlindModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.recurrent_hand_layer = nn.LSTM(
            input_size=18,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )
        self.joker_layer = nn.LSTM(
            input_size=172, hidden_size=256, num_layers=1, batch_first=True
        )

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_layer_1 = nn.Linear(512 + 3, 256)
        self.hidden_layer_2 = nn.Linear(256, 256)
        self._last_hidden = None
        self.final_layer = nn.Linear(256, num_outputs)
        self.value_layer = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        # non_sequence = ["chips", "discards_left", "hands_left", "log_chip_goal"]
        non_sequence = ["chips", "discards_left", "hands_left"]
        non_sequence = torch.concat(
            [input_dict["obs"][feature] for feature in non_sequence], dim=1
        )  # [?, 3]
        # Add a temporal dimension to the non-sequence tensor

        # Process the hand sequence
        obs_hand = input_dict["obs"]["hand"]
        ranks = obs_hand.values[0]  # [?, max_hand_size, 13]
        rank_continuous = obs_hand.values[1]  # [?, max_hand_size]
        suits = obs_hand.values[2]  # [?, max_hand_size, 4]
        rank_continuous = rank_continuous.unsqueeze(2)  # [?, max_hand_size, 1]

        hand = torch.cat(
            (ranks, rank_continuous, suits), dim=2
        )  # [?, max_hand_size, 18]
        hand_lengths = input_dict["obs"]["hand_size"]
        # if all hand_lengths == 0, then convert all to 1
        hand_lengths = torch.where(
            hand_lengths == 0, torch.ones_like(hand_lengths), hand_lengths
        )
        packed_hand = nn.utils.rnn.pack_padded_sequence(
            hand, hand_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        hand_output, (hand_ht, hand_ct) = self.recurrent_hand_layer(packed_hand)
        hand_ht = hand_ht.squeeze(0)

        obs_jokers = input_dict["obs"]["owned_jokers"]
        joker_costs = obs_jokers.values["cost"]
        joker_costs = joker_costs.unsqueeze(2)
        joker_names = obs_jokers.values["name"]
        joker_rarities = obs_jokers.values["rarity"]
        joker_flags = obs_jokers.values["flags"]
        jokers = torch.cat(
            (joker_costs, joker_names, joker_rarities, joker_flags), dim=2
        )

        num_jokers = input_dict["obs"]["owned_jokers_count"]
        num_jokers = torch.where(
            num_jokers == 0, torch.ones_like(num_jokers), num_jokers
        )
        packed_jokers = nn.utils.rnn.pack_padded_sequence(
            jokers, num_jokers.to("cpu"), batch_first=True, enforce_sorted=False
        )
        joker_output, (joker_ht, joker_ct) = self.joker_layer(packed_jokers)
        joker_ht = joker_ht.squeeze(0)

        combined_output = torch.cat((joker_ht, hand_ht, non_sequence), dim=1)

        hidden_output = self.relu(self.hidden_layer_1(combined_output))
        self._last_hidden = self.relu(self.hidden_layer_2(hidden_output))
        final_output = self.final_layer(self._last_hidden)

        return final_output, []

    def value_function(self):
        return self.value_layer(self._last_hidden).squeeze(1)
