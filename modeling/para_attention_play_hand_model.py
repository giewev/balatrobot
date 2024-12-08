import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.torch_utils import FLOAT_MIN
import numpy as np


class ParametricAttentionPlayHandModel(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hand_size = kwargs.get("hand_size", 5)
        min_card_count = 1
        max_card_count = 5
        count_choices = max_card_count - min_card_count + 1
        play_discard_choices = 2
        num_hand_types = 9
        embedding_extension_size = 2
        positional_embedding_size = 2
        card_count_embedding_size = 4

        learn_embeddings = kwargs.get("learn_embeddings", False)
        context_size = kwargs.get("context_size", 16)
        hidden_size = kwargs.get("hidden_size", 64)
        # action_module_size = kwargs.get("action_module_size", 128)
        num_heads = kwargs.get("num_heads", 4)
        num_attention_layers = kwargs.get("num_attention_layers", 1)
        num_hidden_layers = kwargs.get("num_hidden_layers", 1)
        if not learn_embeddings:
            card_embedding_size = 20
        else:
            card_embedding_size = kwargs.get("card_embedding_size", 8)

        self.hand_embedding = nn.Embedding(
            52,
            card_embedding_size
            - embedding_extension_size
            - positional_embedding_size
            - card_count_embedding_size,
            max_norm=1.0,
            _freeze=learn_embeddings,
        )

        if not learn_embeddings:
            starting_embeddings = torch.zeros(52, card_embedding_size)
            for i in range(4):
                starting_embeddings[i * 13 : (i + 1) * 13, i] = 1.0
                for j in range(13):
                    starting_embeddings[i * 13 + j, j + 4] = 1.0
            self.hand_embedding.weight.data.copy_(starting_embeddings)

        self.embedding_extension = nn.Embedding(
            52, embedding_extension_size, max_norm=1.0, _freeze=True
        )
        for i in range(52):
            rank = i % 13
            self.embedding_extension.weight.data[i, 0] = np.sin(rank / 13 * np.pi)
            self.embedding_extension.weight.data[i, 1] = np.cos(rank / 13 * np.pi)

        self.positional_embedding = torch.zeros(
            (1, hand_size, 2), dtype=torch.float32, requires_grad=False
        )
        for i in range(hand_size):
            self.positional_embedding[:, i, 0] = np.sin(i / hand_size * np.pi)
            self.positional_embedding[:, i, 1] = np.cos(i / hand_size * np.pi)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(card_embedding_size)

        # self.card_pre_attention_ff_layer = nn.Linear(
        #     card_embedding_size,
        #     card_embedding_size,
        # )

        self.pre_attention_flat_layer = nn.Linear(
            2 + num_hand_types + (num_hand_types - 1), card_embedding_size
        )

        self.attention_layers = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=card_embedding_size, num_heads=num_heads, batch_first=True
            )
            for _ in range(num_attention_layers)
        )

        self.attention_ff_layers = nn.ModuleList(
            nn.Linear(card_embedding_size, card_embedding_size)
            for _ in range(num_attention_layers)
        )

        hidden_ff_layers = [
            nn.Linear(
                card_embedding_size * hand_size
                + 2
                + num_hand_types
                + (num_hand_types - 1),
                hidden_size,
            )
        ]

        for _ in range(num_hidden_layers - 1):
            hidden_ff_layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_ff_layers = nn.ModuleList(hidden_ff_layers)

        self.context_layer = nn.Linear(hidden_size, context_size)

        self.play_intent_layer = nn.Linear(context_size, card_embedding_size)
        self.discard_intent_layer = nn.Linear(context_size, card_embedding_size)
        # self.play_slot_layer = nn.Linear(context_size, hand_size)
        # self.discard_slot_layer = nn.Linear(context_size, hand_size)
        self.discard_slot_layer = nn.Linear(card_embedding_size, 1)
        self.play_slot_layer = nn.Linear(card_embedding_size, 1)

        self.meta_action_layer = nn.Linear(
            context_size, count_choices * 2 + play_discard_choices
        )
        self.value_layer = nn.Linear(context_size, 1)
        self._last_context = None

    def forward(self, input_dict, state, seq_lens):
        obs_hand_indices = input_dict["obs"]["hand_indices"].to(torch.int64)
        rank_counts = input_dict["obs"]["rank_counts"].unsqueeze(2)
        suit_counts = input_dict["obs"]["suit_counts"].unsqueeze(2)
        run_counts = input_dict["obs"]["run_counts"].unsqueeze(2)
        suited_run_counts = input_dict["obs"]["suited_run_counts"].unsqueeze(2)
        raw_hand = self.hand_embedding(obs_hand_indices)
        extension = self.embedding_extension(obs_hand_indices)
        position = self.positional_embedding.repeat(raw_hand.size(0), 1, 1).to(
            raw_hand.device
        )
        raw_hand = torch.cat(
            (
                raw_hand,
                extension,
                position,
                rank_counts,
                suit_counts,
                run_counts,
                suited_run_counts,
            ),
            dim=2,
        )

        hands_played = input_dict["obs"]["hands_played"] / 5
        hands_played = torch.unsqueeze(hands_played, 1)
        discards_played = input_dict["obs"]["discards_played"] / 10
        discards_played = torch.unsqueeze(discards_played, 1)
        target_hand_types = input_dict["obs"]["target_hand_types"]
        available_hand_types = input_dict["obs"]["available_hand_types"]
        flat_attn_inputs = torch.cat(
            (hands_played, discards_played, target_hand_types, available_hand_types),
            dim=1,
        )
        flat_attn = self.pre_attention_flat_layer(flat_attn_inputs)
        flat_attn = flat_attn.unsqueeze(1)

        q = raw_hand + flat_attn
        kv = raw_hand
        for i in range(len(self.attention_layers)):
            attn_output, _ = self.attention_layers[i](q, kv, kv)
            attn_output = self.norm(q + attn_output)
            ff_output = self.activation(self.attention_ff_layers[i](attn_output))
            ff_output = self.norm(ff_output + attn_output)
            attn = ff_output
            q = attn + flat_attn
            kv = attn

        flat_attn = attn.reshape(attn.size(0), -1)

        hidden = torch.cat(
            (
                flat_attn,
                hands_played,
                discards_played,
                target_hand_types,
                available_hand_types,
            ),
            dim=1,
        )

        for layer in self.hidden_ff_layers:
            hidden = self.activation(layer(hidden))

        self._last_context = self.activation(self.context_layer(hidden))

        play_intent = self.play_intent_layer(self._last_context)
        play_intent = torch.unsqueeze(play_intent, 1)
        play_intent = self.activation(play_intent)
        play_intent_logits = torch.sum(raw_hand * play_intent, dim=2)
        # play_slot_logits = self.play_slot_layer(self._last_context)
        play_slot_logits = self.play_slot_layer(attn).squeeze(2)
        # play_card_logits = play_intent_logits + play_slot_logits
        play_card_logits = play_slot_logits

        discard_intent = self.discard_intent_layer(self._last_context)
        discard_intent = torch.unsqueeze(discard_intent, 1)
        discard_intent = self.activation(discard_intent)
        discard_intent_logits = torch.sum(raw_hand * discard_intent, dim=2)
        # discard_slot_logits = self.discard_slot_layer(self._last_context)
        discard_slot_logits = self.discard_slot_layer(attn).squeeze(2)
        # discard_card_logits = discard_intent_logits + discard_slot_logits
        discard_card_logits = discard_slot_logits

        meta_action_logits = self.meta_action_layer(self._last_context)

        # self.predicted_hand = self.hand_prediction_layer(self._last_context)
        # self.correct_hand = input_dict["obs"]["correct_hand"]

        logits = torch.cat(
            (play_card_logits, discard_card_logits, meta_action_logits), dim=1
        )
        return logits, []

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)

    # @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        l2_lambda = 0.00

        if isinstance(policy_loss, list):
            device = policy_loss[0].device
        else:
            device = policy_loss.device
        l2_reg = torch.tensor(0.0).to(device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        self.l2_loss = l2_lambda * l2_reg

        # self.hand_prediction_loss = nn.CrossEntropyLoss()(
        #     self.predicted_hand, self.correct_hand
        # )

        if isinstance(policy_loss, list):
            return [
                single_loss + self.l2_loss  # + self.predicted_hand
                for single_loss in policy_loss
            ]
        else:
            return policy_loss + self.l2_loss  # + self.predicted_hand
