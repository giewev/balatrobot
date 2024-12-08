import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ParametricBalatroPlayHandModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hand_size = 8
        min_card_count = 1
        max_card_count = 5
        count_choices = max_card_count - min_card_count + 1
        play_discard_choices = 2

        card_embedding_size = 20
        context_size = model_config.get("context_size", 128)
        hidden_size = model_config.get("hidden_size", 128)
        action_module_size = model_config.get("action_module_size", 128)
        # lstm_hidden_size = 128

        self.hand_embedding = nn.Embedding(
            54, card_embedding_size, max_norm=1.0, _freeze=True
        )

        starting_embeddings = (
            # torch.randn(54, card_embedding_size) / 10
            torch.zeros(54, card_embedding_size)
        )  # start with low noise just to break symmetry
        for i in range(4):
            starting_embeddings[i * 13 : (i + 1) * 13, i] = 1.0
            for j in range(13):
                starting_embeddings[i * 13 + j, j + 4] = 1.0
        starting_embeddings[52, 17] = 1.0
        starting_embeddings[53, 18] = 1.0
        self.hand_embedding.weight.data.copy_(starting_embeddings)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        # self.activation = nn.Tanh()
        self.tanh = nn.Tanh()
        self.card_preprocess_layer = nn.Linear(card_embedding_size, card_embedding_size)
        self.hidden_layer_1 = nn.Linear(
            card_embedding_size + card_embedding_size * hand_size + 9,
            hidden_size,
        )
        # self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.context_layer = nn.Linear(hidden_size, context_size)
        self.play_intent_layer = nn.Linear(context_size, card_embedding_size)
        self.discard_intent_layer = nn.Linear(context_size, card_embedding_size)
        self.meta_action_layer = nn.Linear(
            context_size, count_choices + play_discard_choices
        )
        self.value_layer = nn.Linear(context_size, 1)
        self._last_context = None
        self._hand = None

    def forward(self, input_dict, state, seq_lens):

        target_hand_types = input_dict["obs"]["target_hand_types"]

        obs_hand_indices = input_dict["obs"]["hand_indices"].to(torch.int64)
        self._hand = self.hand_embedding(obs_hand_indices)
        # positional encoding
        self._hand[:, :, 19] = (
            torch.arange(-3, 5)
            .unsqueeze(0)
            .expand(self._hand.size(0), -1)
            .to(self._hand.device)
            / 4.0
        )
        self.hand_mean = torch.mean(self._hand, dim=1)

        flattened_hand = self._hand.view(self._hand.size(0), -1)

        obs_hand = input_dict["obs"]["hand"]
        ranks = obs_hand.values[0]  # [?, max_hand_size, 13]
        rank_continuous = obs_hand.values[1]  # [?, max_hand_size]
        suits = obs_hand.values[2]  # [?, max_hand_size, 4]
        rank_continuous = rank_continuous.unsqueeze(2)  # [?, max_hand_size, 1]

        hand = torch.cat(
            (ranks, rank_continuous, suits), dim=2
        )  # [?, max_hand_size, 18]
        hand = hand.view(hand.size(0), -1)

        combined_output = torch.cat(
            (self.hand_mean, flattened_hand, target_hand_types), dim=1
        )

        hidden_output = self.activation(self.hidden_layer_1(combined_output))
        hidden_output = self.activation(self.hidden_layer_2(hidden_output))
        self._last_context = self.context_layer(hidden_output)

        play_intent = self.play_intent_layer(self._last_context)
        play_intent = torch.unsqueeze(play_intent, 1)
        play_intent = self.tanh(play_intent)

        discard_intent = self.discard_intent_layer(self._last_context)
        discard_intent = torch.unsqueeze(discard_intent, 1)
        discard_intent = self.tanh(discard_intent)

        card_logits = torch.sum(self._hand * intent, dim=2)
        meta_action_logits = self.meta_action_layer(self._last_context)
        logits = torch.cat((card_logits, meta_action_logits), dim=1)

        return logits, []

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)
