from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    Dict,
    TensorType,
    ModuleID,
    TensorType,
)
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.columns import Columns
import torch


class PPOTorchAuxilliaryLearner(PPOTorchLearner):
    @override(PPOLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config=None,
        batch: NestedDict,
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        # # standard behavior cloning loss
        # action_dist_inputs = fwd_out[Columns.ACTION_DIST_INPUTS]
        # action_dist_class = self._module[module_id].get_train_action_dist_cls()
        # action_dist = action_dist_class.from_logits(action_dist_inputs)
        # loss = -torch.mean(action_dist.logp(batch[Columns.ACTIONS]))
        super_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )

        # print(batch[Columns.OBS, "last_hand_played"])
        # print(batch[Columns.NEXT_OBS, "last_hand_played"])
        # aux_output = fwd_out["hand_prediction"]
        # aux_target = batch["aux_target"]
        # print(batch[Columns.INFOS, "hand_played"])
        # print(batch[Columns.INFOS])
        # print("obs")
        # print(batch[Columns.OBS])
        # empty_target = torch.zeros((9,), dtype=torch.float32, device=aux_output.device)
        # print(batch[Columns.INFOS])
        # print([x.keys() for x in batch[Columns.INFOS]])
        # print(type(batch[Columns.INFOS]))
        # print(batch[Columns.INFOS])
        # print(batch.keys())

        # aux_target = torch.stack(
        #     [
        #         (
        #             torch.from_numpy(x["hand_played"]).to(aux_output.device)
        #             if "hand_played" in x
        #             else empty_target
        #         )
        #         for x in batch[Columns.INFOS]
        #     ],
        #     dim=0,
        # )

        # # Only calculate the auxiliary loss where the target vector is not all zeros
        # mask = torch.sum(aux_target, dim=-1) != 0
        # aux_output = aux_output[mask]
        # aux_target = aux_target[mask]

        # aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #     aux_output, aux_target
        # )

        # self.register_metrics(
        #     module_id,
        #     {
        #         "aux_loss": aux_loss.item(),
        #     },
        # )

        # return super_loss + aux_loss
        return super_loss
