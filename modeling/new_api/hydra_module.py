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
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from typing import Mapping, Any
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core import Columns

# from modeling.combo_index_dist import ComboIndexDist
from modeling.new_api.combo_index_dist import ComboIndexDist
from ray.rllib.utils.annotations import override


class HydraModule(TorchRLModule):
    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        self.hand_size = self.config.model_config_dict.get("hand_size", 5)
        self.deck_size = self.config.model_config_dict.get("deck_size", 52 - 8)
        max_embedding_norm = self.config.model_config_dict.get(
            "max_embedding_norm", None
        )
        min_card_count = 1
        max_card_count = 5
        count_choices = max_card_count - min_card_count + 1
        play_discard_choices = 2
        num_hand_types = 9
        rank_embedding_size = 4
        suit_embedding_size = 4
        positional_embedding_size = 2
        card_count_embedding_size = 4
        flat_input_size = 2 + (num_hand_types - 1) + num_hand_types

        learn_embeddings = self.config.model_config_dict.get("learn_embeddings", False)
        context_size = self.config.model_config_dict.get("context_size", 16)
        hidden_size = self.config.model_config_dict.get("hidden_size", 64)
        # action_module_size = self.config.model_config_dict.get("action_module_size", 128)
        num_heads = self.config.model_config_dict.get("num_heads", 4)
        self.num_attention_layers = self.config.model_config_dict.get(
            "num_attention_layers", 1
        )
        num_hidden_layers = self.config.model_config_dict.get("num_hidden_layers", 1)
        if not learn_embeddings:
            card_embedding_size = 20
        else:
            card_embedding_size = self.config.model_config_dict.get(
                "card_embedding_size", 8
            )
        card_w_flat_size = card_embedding_size + flat_input_size
        head_leftovers = card_w_flat_size % num_heads
        if head_leftovers != 0:
            print(
                f"Warning: card_w_flat_size not divisible by num_heads for card_size: {card_embedding_size} num_heads: {num_heads} card_w_flat_size: {card_w_flat_size} head_leftovers: {head_leftovers}"
            )
            card_embedding_size += num_heads - head_leftovers
            card_w_flat_size = card_embedding_size + flat_input_size
            print(
                f"Rounding up to card_embedding_size: {card_embedding_size}, card_w_flat_size: {card_w_flat_size}"
            )
            # print(f"New card_embedding_size: {card_embedding_size}")
            # print(f"New card_w_flat_size: {card_w_flat_size}")

        card_extension_size = (
            rank_embedding_size
            + suit_embedding_size
            + positional_embedding_size
            + card_count_embedding_size
        )

        self.hand_embedding = nn.Embedding(
            53,
            card_embedding_size - card_extension_size,
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

        self.positional_embedding = torch.zeros(
            (1, self.hand_size, 2), dtype=torch.float32, requires_grad=False
        )
        for i in range(self.hand_size):
            self.positional_embedding[:, i, 0] = np.sin(i / self.hand_size * np.pi)
            self.positional_embedding[:, i, 1] = np.cos(i / self.hand_size * np.pi)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(card_w_flat_size)

        # self.pre_attention_flat_layer = nn.Linear(flat_input_size, flat_input_size)
        # self.card_extension_prep_layer = nn.Linear(
        #     card_extension_size, card_extension_size
        # )

        self.pre_attention_net = self.fcnet(
            card_w_flat_size, hidden_size, 1, card_w_flat_size
        )

        self.cross_attention_layer = CrossSequenceAttention(
            self.num_attention_layers, num_heads, card_w_flat_size
        )

        hidden_ff_layers = [
            nn.Linear(
                card_w_flat_size * self.hand_size * 2 + flat_input_size,
                hidden_size,
            )
        ]

        for _ in range(num_hidden_layers - 1):
            hidden_ff_layers.append(nn.Linear(hidden_size, hidden_size))
        self.hidden_ff_layers = nn.ModuleList(hidden_ff_layers)

        self.context_layer = nn.Linear(hidden_size, context_size)

        # self.play_hidden_slots_head = self.fcnet(
        #     context_size, hidden_size, num_hidden_layers, self.hand_size
        # )
        # self.discard_hidden_slots_head = self.fcnet(
        #     context_size, hidden_size, num_hidden_layers, self.hand_size
        # )

        # self.play_hidden_intent_head = nn.Linear(context_size, card_w_flat_size)
        # self.discard_hidden_intent_head = nn.Linear(context_size, card_w_flat_size)

        # discard_attn_fc_layers = []
        # play_attn_fc_layers = []
        # for _ in range(num_hidden_layers):
        #     discard_attn_fc_layers.append(
        #         nn.Linear(card_w_flat_size * 2, card_w_flat_size * 2)
        #     )
        #     discard_attn_fc_layers.append(self.activation)
        #     play_attn_fc_layers.append(
        #         nn.Linear(card_w_flat_size * 2, card_w_flat_size * 2)
        #     )
        #     play_attn_fc_layers.append(self.activation)
        # discard_attn_fc_layers.append(nn.Linear(card_w_flat_size * 2, 1))
        # play_attn_fc_layers.append(nn.Linear(card_w_flat_size * 2, 1))
        # self.discard_attn_fc = nn.Sequential(*discard_attn_fc_layers)
        # self.play_attn_fc = nn.Sequential(*play_attn_fc_layers)

        self.discard_attn_fc = self.fcnet(
            card_w_flat_size * 2, card_w_flat_size * 2, num_hidden_layers, 1
        )
        self.play_attn_fc = self.fcnet(
            card_w_flat_size * 2, card_w_flat_size * 2, num_hidden_layers, 1
        )

        self.meta_action_head = nn.Linear(
            context_size, play_discard_choices + count_choices * 2
        )
        self.value_head = nn.Linear(context_size, 1)
        self.hand_played_head = self.fcnet(
            context_size, hidden_size, num_hidden_layers, 9
        )
        self._last_context = None

    def fcnet(self, in_size, h_size, h_layers, out_size):
        layers = [
            nn.Linear(in_size, h_size),
            nn.LeakyReLU(negative_slope=0.01),
        ]

        for _ in range(h_layers - 1):
            layers.append(nn.Linear(h_size, h_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))

        layers.append(nn.Linear(h_size, out_size))

        return nn.Sequential(*layers)

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return ComboIndexDist

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return ComboIndexDist

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return ComboIndexDist

    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_train(batch)

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_train(batch)

    @override(TorchRLModule)
    def output_specs_exploration(self):
        return ["action_dist_inputs", "vf_preds"]

    @override(TorchRLModule)
    def output_specs_inference(self):
        return ["action_dist_inputs", "vf_preds"]

    @override(TorchRLModule)
    def output_specs_train(self):
        return ["action_dist_inputs", "vf_preds"]

    def value_function(self):
        return self.value_layer(self._last_context).squeeze(1)

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        obs = batch["obs"]
        deck_indices = obs["deck_indices"].to(torch.int64)
        deck_rank_counts = obs["deck_rank_counts"].unsqueeze(2)
        deck_suit_counts = obs["deck_suit_counts"].unsqueeze(2) / 5.0
        deck_run_counts = obs["deck_run_counts"].unsqueeze(2) / 3.0
        deck_suited_run_counts = obs["deck_suited_run_counts"].unsqueeze(2)
        raw_deck = self.hand_embedding(deck_indices)
        deck_rank_suit_ext = self.rank_suit_embedding(deck_indices)
        null_positional_embedding = torch.zeros(
            (deck_rank_suit_ext.size(0), deck_rank_suit_ext.size(1), 2),
            dtype=torch.float32,
            requires_grad=False,
        ).to(deck_rank_suit_ext.device)

        deck_ext = torch.cat(
            (
                deck_rank_suit_ext,
                deck_rank_counts,
                deck_suit_counts,
                deck_run_counts,
                deck_suited_run_counts,
                null_positional_embedding,
            ),
            dim=2,
        )
        # deck_ext = self.card_extension_prep_layer(deck_ext)
        raw_deck = torch.cat(
            (raw_deck, deck_ext),
            dim=2,
        )

        obs_hand_indices = obs["hand_indices"].to(torch.int64)
        rank_counts = obs["rank_counts"].unsqueeze(2)
        suit_counts = obs["suit_counts"].unsqueeze(2)
        run_counts = obs["run_counts"].unsqueeze(2)
        suited_run_counts = obs["suited_run_counts"].unsqueeze(2)
        raw_hand = self.hand_embedding(obs_hand_indices)
        hand_rank_suit_ext = self.rank_suit_embedding(obs_hand_indices)
        position = self.positional_embedding.repeat(raw_hand.size(0), 1, 1).to(
            raw_hand.device
        )

        hand_ext = torch.cat(
            (
                hand_rank_suit_ext,
                rank_counts,
                suit_counts,
                run_counts,
                suited_run_counts,
                position,
            ),
            dim=2,
        )
        # hand_ext = self.card_extension_prep_layer(hand_ext)
        raw_hand = torch.cat(
            (raw_hand, hand_ext),
            dim=2,
        )

        hands_left = obs["hands_left"] / 5.0
        discards_left = obs["discards_left"] / 5.0
        available_hand_types = obs["available_hand_types"]
        target_hand_types = obs["target_hand_types"]
        flat_inputs = torch.cat(
            (hands_left, hands_left, available_hand_types, target_hand_types),
            dim=1,
        )

        # concat the flat inputs to every card in the hand
        hand_w_flat = torch.cat(
            (
                raw_hand,
                flat_inputs.unsqueeze(1).repeat(1, self.hand_size, 1),
            ),
            dim=2,
        )

        deck_w_flat = torch.cat(
            (
                raw_deck,
                flat_inputs.unsqueeze(1).repeat(1, self.deck_size, 1),
            ),
            dim=2,
        )

        q = self.pre_attention_net(hand_w_flat)
        kv = self.pre_attention_net(deck_w_flat)

        cross, q_out, kv_out = self.cross_attention_layer(q, kv)

        flat_cross = cross.reshape(cross.size(0), -1)
        flat_q_out = q_out.reshape(q_out.size(0), -1)

        hidden = torch.cat(
            (
                flat_cross,
                flat_q_out,
                flat_inputs,
            ),
            dim=1,
        )

        for layer in self.hidden_ff_layers:
            hidden = self.activation(layer(hidden))

        self._last_context = self.activation(self.context_layer(hidden))

        cross_and_q = torch.cat((cross, q_out), dim=-1)

        discard_card_logits = self.slot_logits(
            cross_and_q, self._last_context, hand_w_flat, "discard"
        )
        play_card_logits = self.slot_logits(
            cross_and_q, self._last_context, hand_w_flat, "play"
        )

        meta_action_logits = self.meta_action_head(self._last_context)

        # Mask the first meta logit to be negative infinity if discards left is 0
        mask = discards_left.squeeze(1) == 0
        meta_action_logits[:, 0][mask] = FLOAT_MIN

        logits = torch.cat(
            (play_card_logits, discard_card_logits, meta_action_logits), dim=1
        )

        value_preds = self.value_head(self._last_context).squeeze(1)

        hand_preds = self.hand_played_head(self._last_context)

        return {
            "action_dist_inputs": logits,
            # "action_dist": ComboIndexDist(logits),
            "vf_preds": value_preds,
            "hand_prediction": hand_preds,
        }

    def slot_logits(self, attn_w_flat, context, hand_w_flat, mode):
        if mode == "play":
            # hidden_slots_head = self.play_hidden_slots_head
            # hidden_intent_head = self.play_hidden_intent_head
            attn_fc = self.play_attn_fc
        else:
            # hidden_slots_head = self.discard_hidden_slots_head
            # hidden_intent_head = self.discard_hidden_intent_head
            attn_fc = self.discard_attn_fc

        # hidden_intent = hidden_intent_head(context).unsqueeze(1)
        # intent_logits = torch.sum(hidden_intent * hand_w_flat, dim=-1)

        attn_logits = attn_fc(attn_w_flat).squeeze(-1)

        # direct_slot_logits = hidden_slots_head(context)

        # return attn_logits + direct_slot_logits + intent_logits
        return attn_logits
