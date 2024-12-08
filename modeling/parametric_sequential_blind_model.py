import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ParametricSequentialBalatroBlindModel(TorchModelV2, nn.Module):
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
        context_size = 128
        hidden_size = 128
        lstm_hidden_size = 128
        action_module_size = 128

        self.hand_embedding = nn.Embedding(54, card_embedding_size, max_norm=1.0)

        starting_embeddings = (
            torch.randn(54, card_embedding_size) / 10
        )  # start with low noise just to break symmetry
        for i in range(4):
            starting_embeddings[i * 13 : (i + 1) * 13, i] = 1.0
            for j in range(13):
                starting_embeddings[i * 13 + j, j + 4] = 1.0
        starting_embeddings[52:54, :] = torch.randn(2, card_embedding_size)
        self.hand_embedding.weight.data.copy_(starting_embeddings)

        self.joker_layer = nn.LSTM(
            input_size=172, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True
        )

        # self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.Tanh()
        self.tanh = nn.Tanh()
        self.hidden_layer_1 = nn.Linear(
            # lstm_hidden_size
            card_embedding_size + card_embedding_size * hand_size + 4,
            hidden_size,
        )
        self.context_layer = nn.Linear(hidden_size, context_size)
        self.value_layer = nn.Linear(context_size, 1)
        # self.a1_layer = nn.Linear(context_size, count_choices + play_discard_choices)
        # self.a2_layer = nn.Linear(context_size + 2, card_embedding_size)
        self._last_context = None
        self._hand = None

        class _ActionModel(nn.Module):
            def __init__(self):
                super().__init__()
                a_vec_size = hand_size + play_discard_choices
                # self.activation = nn.LeakyReLU(negative_slope=0.01)
                self.activation = nn.Tanh()
                self.h1 = nn.Linear(
                    context_size + a_vec_size + card_embedding_size, action_module_size
                )
                # self.h2 = nn.Linear(action_module_size, a_vec_size)
                self.h2 = nn.Linear(action_module_size, card_embedding_size)
                # self.h_only = nn.Linear(
                #     context_size + a_vec_size, card_embedding_size
                # )
                # self.pd_logit_layer = nn.Linear(
                #     action_module_size, play_discard_choices
                # )
                self._hand_card_embeddings = None
                self.play_and_discard_embeddings = None

            def check_safe(self, tensor, name):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: tensor {name} contains NaN or inf values")

            def forward(self, context, actions):
                already_selected_cards = actions[:, :hand_size]
                already_selected_cards = (
                    self._hand_card_embeddings * already_selected_cards.unsqueeze(2)
                )
                already_selected_cards = torch.sum(already_selected_cards, dim=1)

                inputs = torch.cat([context, actions, already_selected_cards], dim=1)
                h1 = self.h1(inputs)
                h1 = self.activation(h1)
                # logits = self.h2(h1)
                # intent = self.h_only(inputs)
                intent = self.h2(h1)
                intent = torch.unsqueeze(intent, 1)
                # self.check_safe(intent, "intent")
                # self.check_safe(self._hand_card_embeddings, "hand_card_embeddings")

                embeddings = torch.cat(
                    (self._hand_card_embeddings, self.play_and_discard_embeddings),
                    dim=1,
                )
                logits = torch.sum(embeddings * intent, dim=2)

                # self.check_safe(card_logits, "card_logits")
                # pd_logits = self.pd_logit_layer(h1)
                # logits = torch.cat([card_logits, pd_logits], dim=1)
                self.check_safe(logits, "logits")

                # Mask out portions of the logits that were 1 in the input actions
                # This is to prevent the model from sampling the same action twice
                # inf_mask = torch.clamp(torch.log(1 - actions), min=FLOAT_MIN)
                # logits = logits + inf_mask
                # print(logits.shape)
                logits = torch.where(actions == 1, FLOAT_MIN, logits)
                # print(logits.shape)
                # print(self.can_discard.shape)
                # disable discarding if no discards are allowed
                logits[:, 9] = torch.where(
                    self.can_discard.squeeze(1).bool(), logits[:, 9], FLOAT_MIN
                )

                # In each row where 5 cards have already been selected, mask out the card selection actions
                # This is to prevent the model from selecting more than 5 cards
                sum_first_8_A = actions[:, :8].sum(dim=1)
                mask = sum_first_8_A == 5
                mask_expanded = mask.unsqueeze(1).expand(-1, 8)
                logits[:, :8][mask_expanded] = FLOAT_MIN

                return logits

        self.action_module = _ActionModel()

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict)
        non_sequence = ["chips", "discards_left", "hands_left", "log_chip_goal"]
        non_sequence = torch.concat(
            [input_dict["obs"][feature] for feature in non_sequence], dim=1
        )  # [?, 4]

        # if input_dict["obs"]["discard_left"] == 0:
        #     self.action_module.can_discard = False
        # else:
        #     self.action_module.can_discard = True

        self.action_module.can_discard = torch.where(
            input_dict["obs"]["discards_left"] == 0, False, True
        )

        obs_hand_indices = input_dict["obs"]["hand_indices"].to(torch.int64)
        self._hand = self.hand_embedding(obs_hand_indices)
        self.action_module._hand_card_embeddings = self._hand
        self.action_module.play_and_discard_embeddings = self.hand_embedding(
            torch.tensor([52, 53], dtype=torch.int64).to(self._hand.device)
        )
        self.action_module.play_and_discard_embeddings = torch.unsqueeze(
            self.action_module.play_and_discard_embeddings, 0
        )
        self.action_module.play_and_discard_embeddings = (
            self.action_module.play_and_discard_embeddings.expand(
                self._hand.size(0), -1, -1
            )
        )
        # self.hand_sum = torch.sum(self._hand, dim=1)
        # self.hand_bag = self.hand_sum
        self.hand_mean = torch.mean(self._hand, dim=1)
        # self.hand_max = torch.max(self._hand, dim=1)
        # self.hand_bag = torch.cat(
        #     (self.hand_sum, self.hand_mean, self.hand_max.values), dim=1
        # )
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
        # packed_jokers = nn.utils.rnn.pack_padded_sequence(
        #     jokers, num_jokers.to("cpu"), batch_first=True, enforce_sorted=False
        # )
        # joker_output, (joker_ht, joker_ct) = self.joker_layer(packed_jokers)
        # joker_ht = joker_ht.squeeze(0)

        combined_output = torch.cat(
            (self.hand_mean, flattened_hand, non_sequence), dim=1
        )

        hidden_output = self.activation(self.hidden_layer_1(combined_output))
        self._last_context = self.tanh(self.context_layer(hidden_output))
        # final_output = self.final_layer(self._last_hidden)

        # print(self._last_context)
        return self._last_context, []

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)
