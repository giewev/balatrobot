import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.torch_utils import FLOAT_MIN
import numpy as np
from modeling.submodules.residual_attention import (
    ResidualAttention,
    CrossSequenceAttention,
)


class AttentionBlindDeckModel(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.hand_size = kwargs.get("hand_size", 5)
        deck_size = kwargs.get("deck_size", 52 - 8)
        max_embedding_norm = kwargs.get("max_embedding_norm", None)
        min_card_count = 1
        max_card_count = 5
        count_choices = max_card_count - min_card_count + 1
        play_discard_choices = 2
        num_hand_types = 9
        rank_embedding_size = 4
        suit_embedding_size = 4
        positional_embedding_size = 2
        card_count_embedding_size = 4

        learn_embeddings = kwargs.get("learn_embeddings", False)
        context_size = kwargs.get("context_size", 16)
        hidden_size = kwargs.get("hidden_size", 64)
        # action_module_size = kwargs.get("action_module_size", 128)
        num_heads = kwargs.get("num_heads", 4)
        self.num_attention_layers = kwargs.get("num_attention_layers", 1)
        num_hidden_layers = kwargs.get("num_hidden_layers", 1)
        if not learn_embeddings:
            card_embedding_size = 20
        else:
            card_embedding_size = kwargs.get("card_embedding_size", 8)

        self.hand_embedding = nn.Embedding(
            53,
            card_embedding_size
            - rank_embedding_size
            - suit_embedding_size
            - positional_embedding_size
            - card_count_embedding_size,
            max_norm=max_embedding_norm,
            _freeze=not learn_embeddings,
            padding_idx=52,
        )

        if not learn_embeddings:
            starting_embeddings = torch.zeros(53, card_embedding_size)
            for i in range(4):
                starting_embeddings[i * 13 : (i + 1) * 13, i] = 1.0
                for j in range(13):
                    starting_embeddings[i * 13 + j, j + 4] = 1.0
            self.hand_embedding.weight.data.copy_(starting_embeddings)

        self.rank_suit_embedding = nn.Embedding(
            53,
            rank_embedding_size + suit_embedding_size,
            max_norm=max_embedding_norm,
            _freeze=True,
        )
        for i in range(52):
            # Rank division by 6 is to simulate the cyclic nature of ranks for straights
            rank = i % 13
            self.rank_suit_embedding.weight.data[i, 0] = np.sin(rank / 6 * np.pi)
            self.rank_suit_embedding.weight.data[i, 1] = np.cos(rank / 6 * np.pi)

            # Rank division by 12 is to provide a more linear representation
            self.rank_suit_embedding.weight.data[i, 2] = np.sin(rank / 12 * np.pi)
            self.rank_suit_embedding.weight.data[i, 3] = np.cos(rank / 12 * np.pi)

            suit = i // 13
            for j in range(4):
                if j == suit:
                    self.rank_suit_embedding.weight.data[i, 4 + suit] = 1.0
                else:
                    self.rank_suit_embedding.weight.data[i, 4 + j] = 0.0
        self.rank_suit_embedding.weight.data[52, :] = 0.0

        # positional_embedding_size = self.hand_size
        # if self.include_deck_info:
        #     positional_embedding_size += deck_size
        self.positional_embedding = torch.zeros(
            (1, self.hand_size, 2), dtype=torch.float32, requires_grad=False
        )
        for i in range(self.hand_size):
            self.positional_embedding[:, i, 0] = np.sin(i / self.hand_size * np.pi)
            self.positional_embedding[:, i, 1] = np.cos(i / self.hand_size * np.pi)
        # if self.include_deck_info:
        #     for i in range(self.hand_size, self.hand_size + deck_size):
        #         self.positional_embedding[:, i, 0] = 0
        #         self.positional_embedding[:, i, 1] = 0

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(card_embedding_size)

        # self.card_pre_attention_ff_layer = nn.Linear(
        #     card_embedding_size,
        #     card_embedding_size,
        # )

        self.pre_attention_flat_layer = nn.Linear(
            2 + (num_hand_types - 1), card_embedding_size
        )

        self.attention_layer = CrossSequenceAttention(
            self.num_attention_layers, num_heads, card_embedding_size
        )

        # self.attention_layers = nn.ModuleList(
        #     nn.MultiheadAttention(
        #         embed_dim=card_embedding_size, num_heads=num_heads, batch_first=True
        #     )
        #     for _ in range(num_attention_layers)
        # )

        # self.attention_ff_layers = nn.ModuleList(
        #     nn.Linear(card_embedding_size, card_embedding_size)
        #     for _ in range(num_attention_layers)
        # )

        # self.deck_attention_layers = nn.ModuleList(
        #     nn.MultiheadAttention(
        #         embed_dim=card_embedding_size, num_heads=num_heads, batch_first=True
        #     )
        #     for _ in range(num_attention_layers)
        # )

        # self.deck_attention_ff_layers = nn.ModuleList(
        #     nn.Linear(card_embedding_size, card_embedding_size)
        #     for _ in range(num_attention_layers)
        # )

        hidden_ff_layers = [
            nn.Linear(
                card_embedding_size * self.hand_size + 2 + (num_hand_types - 1),
                hidden_size,
            )
        ]

        for _ in range(num_hidden_layers - 1):
            hidden_ff_layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_ff_layers = nn.ModuleList(hidden_ff_layers)

        self.context_layer = nn.Linear(hidden_size, context_size)

        # self.play_intent_layer = nn.Linear(context_size, card_embedding_size)
        # self.discard_intent_layer = nn.Linear(context_size, card_embedding_size)
        self.ff_play_slot_layer = nn.Linear(context_size, self.hand_size)
        self.ff_discard_slot_layer = nn.Linear(context_size, self.hand_size)
        self.discard_slot_layer = nn.Linear(card_embedding_size, 1)
        self.play_slot_layer = nn.Linear(card_embedding_size, 1)

        self.meta_action_layer = nn.Linear(
            context_size, play_discard_choices + count_choices * 2
        )
        self.value_layer = nn.Linear(context_size, 1)
        self._last_context = None

    def forward(self, input_dict, state, seq_lens):
        deck_indices = input_dict["obs"]["deck_indices"].to(torch.int64)
        deck_rank_counts = input_dict["obs"]["deck_rank_counts"].unsqueeze(2)
        deck_suit_counts = input_dict["obs"]["deck_suit_counts"].unsqueeze(2)
        deck_run_counts = input_dict["obs"]["deck_run_counts"].unsqueeze(2)
        deck_suited_run_counts = input_dict["obs"]["deck_suited_run_counts"].unsqueeze(
            2
        )
        raw_deck = self.hand_embedding(deck_indices)
        deck_rank_suit_ext = self.rank_suit_embedding(deck_indices)
        null_positional_embedding = torch.zeros(
            (deck_rank_suit_ext.size(0), deck_rank_suit_ext.size(1), 2),
            dtype=torch.float32,
            requires_grad=False,
        ).to(deck_rank_suit_ext.device)
        raw_deck = torch.cat(
            (
                raw_deck,
                deck_rank_suit_ext,
                deck_rank_counts,
                deck_suit_counts,
                deck_run_counts,
                deck_suited_run_counts,
                null_positional_embedding,
            ),
            dim=2,
        )

        obs_hand_indices = input_dict["obs"]["hand_indices"].to(torch.int64)
        rank_counts = input_dict["obs"]["rank_counts"].unsqueeze(2)
        suit_counts = input_dict["obs"]["suit_counts"].unsqueeze(2)
        run_counts = input_dict["obs"]["run_counts"].unsqueeze(2)
        suited_run_counts = input_dict["obs"]["suited_run_counts"].unsqueeze(2)
        raw_hand = self.hand_embedding(obs_hand_indices)
        hand_rank_suit_ext = self.rank_suit_embedding(obs_hand_indices)
        position = self.positional_embedding.repeat(raw_hand.size(0), 1, 1).to(
            raw_hand.device
        )

        raw_hand = torch.cat(
            (
                raw_hand,
                hand_rank_suit_ext,
                rank_counts,
                suit_counts,
                run_counts,
                suited_run_counts,
                position,
            ),
            dim=2,
        )

        # position = self.positional_embedding.repeat(raw_hand.size(0), 1, 1).to(
        #     raw_hand.device
        # )
        # hand = torch.cat((raw_hand, position), dim=2)

        # hand_and_deck = torch.cat((raw_hand, raw_deck), dim=1)
        # hand_and_deck = torch.cat((hand_and_deck, position), dim=2)

        # hands_played = input_dict["obs"]["hands_played"] / 5
        # hands_played = torch.unsqueeze(hands_played, 1)
        # discards_played = input_dict["obs"]["discards_played"] / 10
        # discards_played = torch.unsqueeze(discards_played, 1)
        hands_left = input_dict["obs"]["hands_left"] / 5.0
        # hands_left = torch.unsqueeze(hands_left, 1)
        discards_left = input_dict["obs"]["discards_left"] / 5.0
        # discards_left = torch.unsqueeze(discards_left, 1)
        # target_hand_types = input_dict["obs"]["target_hand_types"]
        available_hand_types = input_dict["obs"]["available_hand_types"]
        flat_attn_inputs = torch.cat(
            (hands_left, hands_left, available_hand_types),
            dim=1,
        )
        flat_attn = self.pre_attention_flat_layer(flat_attn_inputs)
        flat_attn = flat_attn.unsqueeze(1)

        q = raw_hand + flat_attn
        kv = raw_deck + flat_attn
        # for i in range(len(self.attention_layers)):
        #     attn_output, _ = self.attention_layers[i](q, kv, kv)
        #     attn_output = self.norm(q + attn_output)
        #     ff_output = self.activation(self.attention_ff_layers[i](attn_output))
        #     ff_output = self.norm(ff_output + attn_output)
        #     attn = ff_output
        #     q = attn + flat_attn
        #     kv = attn

        attn = self.attention_layer(q, kv)[0]

        # Cut out the deck, keep only the hand
        # attn = attn[:, : self.hand_size, :]

        flat_attn = attn.reshape(attn.size(0), -1)

        hidden = torch.cat(
            (
                flat_attn,
                # hands_played,
                hands_left,
                # discards_played,
                discards_left,
                available_hand_types,
            ),
            dim=1,
        )

        for layer in self.hidden_ff_layers:
            hidden = self.activation(layer(hidden))

        self._last_context = self.activation(self.context_layer(hidden))

        # play_intent = self.play_intent_layer(self._last_context)
        # play_intent = torch.unsqueeze(play_intent, 1)
        # play_intent = self.activation(play_intent)
        # play_intent_logits = torch.sum(raw_hand * play_intent, dim=2)
        # play_slot_logits = self.play_slot_layer(self._last_context)
        play_slot_logits = self.play_slot_layer(attn)
        # Flatten the logits (hand_size, 2) -> (hand_size * 2)
        # play_slot_logits = (
        #     play_slot_logits.permute(0, 2, 1).contiguous().view(-1, self.hand_size * 2)
        # )
        # play_card_logits = play_intent_logits + play_slot_logits
        play_card_logits = play_slot_logits.squeeze(-1)
        play_card_logits = play_card_logits + self.ff_play_slot_layer(
            self._last_context
        )

        # discard_intent = self.discard_intent_layer(self._last_context)
        # discard_intent = torch.unsqueeze(discard_intent, 1)
        # discard_intent = self.activation(discard_intent)
        # discard_intent_logits = torch.sum(raw_hand * discard_intent, dim=2)
        # discard_slot_logits = self.discard_slot_layer(self._last_context)
        discard_slot_logits = self.discard_slot_layer(attn)
        # Flatten the logits (hand_size, 2) -> (hand_size * 2)
        # discard_slot_logits = (
        #     discard_slot_logits.permute(0, 2, 1)
        #     .contiguous()
        #     .view(-1, self.hand_size * 2)
        # )
        # discard_card_logits = discard_intent_logits + discard_slot_logits
        discard_card_logits = discard_slot_logits.squeeze(-1)
        discard_card_logits = discard_card_logits + self.ff_discard_slot_layer(
            self._last_context
        )

        meta_action_logits = self.meta_action_layer(self._last_context)

        # Mask the first meta logit to be negative infinity if discards left is 0
        mask = discards_left.squeeze(1) == 0
        meta_action_logits[:, 0][mask] = FLOAT_MIN

        # self.predicted_hand = self.hand_prediction_layer(self._last_context)
        # self.correct_hand = input_dict["obs"]["correct_hand"]

        # print(
        #     play_card_logits.shape, discard_card_logits.shape, meta_action_logits.shape
        # )
        logits = torch.cat(
            (play_card_logits, discard_card_logits, meta_action_logits), dim=1
        )
        return logits, []

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)

    # @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        l2_lambda = 0.000

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
