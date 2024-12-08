from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
from ray.rllib.utils.nested_dict import NestedDict

import torch
import torch.nn as nn


class BasicCommSpeaker(TorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(self) -> None:
        super().__init__(config=None)

        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        self.value_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.comm_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        context = self.encoder(obs["global"])
        action_logits = self.policy_head(context)
        value = self.value_head(context)
        comm_vector = self.comm_head(context)

        return {
            "action_dist": torch.distributions.Categorical(logits=action_logits),
            "vf_preds": value,
            "comm_vector": comm_vector,
        }


class BasicCommListener(TorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(self) -> None:
        super().__init__(config=None)

        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        self.value_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        context = self.encoder(obs["global"])
        action_logits = self.policy_head(context)
        value = self.value_head(context)

        return {
            "action_dist": torch.distributions.Categorical(logits=action_logits),
            "vf_preds": value,
        }


class BasicMultiAgentCommModule(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        # module_specs = self.config.modules
        # module_spec = next(iter(module_specs.values()))
        # global_dim = module_spec.observation_space["global"].shape[0]
        # hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
        # shared_encoder = nn.Sequential(
        #     nn.Linear(global_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        # rl_modules = {}
        # for module_id, module_spec in module_specs.items():
        #     rl_modules[module_id] = BCTorchRLModuleWithSharedGlobalEncoder(
        #         encoder=shared_encoder,
        #         local_dim=module_spec.observation_space["local"].shape[0],
        #         hidden_dim=hidden_dim,
        #         action_dim=module_spec.action_space.n,
        #     )

        # self._rl_modules = rl_modules

        self.speaker = BasicCommSpeaker()
        self.listener = BasicCommListener()

    def _forward_inference(self, batch: NestedDict):
        speaker_out = self.speaker._forward_inference(batch)
        listener_out = self.listener._forward_inference(batch)
