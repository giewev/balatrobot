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

        context_size = 64
        lstm_hidden_size = 128
        hand_size = 8
        min_card_count = 1
        max_card_count = 5
        count_choices = max_card_count - min_card_count + 1
        play_discard_choices = 2

        self.recurrent_hand_layer = nn.LSTM(
            input_size=18,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.joker_layer = nn.LSTM(
            input_size=172, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True
        )

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_layer_1 = nn.Linear(lstm_hidden_size * 2 + 4, 128)
        self.context_layer = nn.Linear(128, context_size)
        self._last_context = None
        self.value_layer = nn.Linear(context_size, 1)
        self.a1_layer = nn.Linear(context_size, count_choices + play_discard_choices)
        # self.a2_layer = nn.Linear(
        #     context_size + count_choices + play_discard_choices, hand_size
        # )
        self.a2_layer = nn.Linear(context_size + 2, hand_size)

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict)
        non_sequence = ["chips", "discards_left", "hands_left", "log_chip_goal"]
        non_sequence = torch.concat(
            [input_dict["obs"][feature] for feature in non_sequence], dim=1
        )  # [?, 4]

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
        # where hand_lengths == 0, then to 1 because we can't have 0 length
        # print(hand_lengths)
        hand_lengths = torch.where(
            hand_lengths == 0, torch.ones_like(hand_lengths), hand_lengths
        )
        # print(hand_lengths)
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
        # print(num_jokers)
        num_jokers = torch.where(
            num_jokers == 0, torch.ones_like(num_jokers), num_jokers
        )
        # print(num_jokers)
        packed_jokers = nn.utils.rnn.pack_padded_sequence(
            jokers, num_jokers.to("cpu"), batch_first=True, enforce_sorted=False
        )
        joker_output, (joker_ht, joker_ct) = self.joker_layer(packed_jokers)
        joker_ht = joker_ht.squeeze(0)

        combined_output = torch.cat((joker_ht, hand_ht, non_sequence), dim=1)

        hidden_output = self.relu(self.hidden_layer_1(combined_output))
        self._last_context = self.relu(self.context_layer(hidden_output))
        # final_output = self.final_layer(self._last_hidden)

        # print(self._last_context)
        return self._last_context, []

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)

    def a1_logits(self, context):
        return self.a1_layer(context)

    def a2_logits(self, context, a1_values):
        inputs = torch.cat([context, a1_values], dim=1)
        # print(context.shape, a1_values.shape, inputs.shape)
        return self.a2_layer(inputs)
