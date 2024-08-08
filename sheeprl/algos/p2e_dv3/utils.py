from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
from lightning import Fabric
import torch

from sheeprl.algos.dreamer_v3.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV3
from sheeprl.algos.dreamer_v3.utils import Moments
from sheeprl.algos.p2e_dv3.agent import build_agent
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/policy_loss_task",
    "Loss/value_loss_task",
    "Loss/policy_loss_exploration",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "Loss/ensemble_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor_task",
    "Grads/critic_task",
    "Grads/actor_exploration",
    "Grads/ensemble",
    # General key name for the exploration critics.
    "Loss/value_loss_exploration",
    "Values_exploration/predicted_values",
    "Values_exploration/lambda_values",
    "Grads/critic_exploration",
    "Rewards/intrinsic",
}.union(AGGREGATOR_KEYS_DV3)
MODELS_TO_REGISTER = {
    "world_model",
    "ensembles",
    "actor_exploration",
    "critic_exploration_intrinsic",
    "target_critic_exploration_intrinsic",
    "moments_exploration_intrinsic",
    "critic_exploration_extrinsic",
    "target_critic_exploration_extrinsic",
    "moments_exploration_extrinsic",
    "actor_task",
    "critic_task",
    "target_critic_task",
    "moments_task",
}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    (
        world_model,
        ensembles,
        actor_task,
        critic_task,
        target_critic_task,
        actor_exploration,
        critics_exploration,
    ) = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["ensembles"] if "exploration" in cfg.algo.name else None,
        state["actor_task"],
        state["critic_task"],
        state["target_critic_task"],
        state["actor_exploration"] if "exploration" in cfg.algo.name else None,
        state["critics_exploration"] if "exploration" in cfg.algo.name else None,
    )
    moments_task = Moments(
        fabric,
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments_task.load_state_dict(state["moments_task"])

    if "exploration" in cfg.algo.name:
        moments_exploration = {
            k: Moments(
                fabric,
                cfg.algo.actor.moments.decay,
                cfg.algo.actor.moments.max,
                cfg.algo.actor.moments.percentile.low,
                cfg.algo.actor.moments.percentile.high,
            )
            for k in critics_exploration.keys()
        }
        for k, m in moments_exploration.items():
            m.load_state_dict(state[f"moments_exploration_{k}"])

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(unwrap_fabric(world_model), artifact_path="world_model")
        model_info["actor_task"] = mlflow.pytorch.log_model(unwrap_fabric(actor_task), artifact_path="actor_task")
        model_info["critic_task"] = mlflow.pytorch.log_model(unwrap_fabric(critic_task), artifact_path="critic_task")
        model_info["target_critic_task"] = mlflow.pytorch.log_model(
            target_critic_task, artifact_path="target_critic_task"
        )
        model_info["moments_task"] = mlflow.pytorch.log_model(moments_task, artifact_path="moments_task")
        if "exploration" in cfg.algo.name:
            model_info["ensembles"] = mlflow.pytorch.log_model(unwrap_fabric(ensembles), artifact_path="ensembles")
            model_info["actor_exploration"] = mlflow.pytorch.log_model(
                unwrap_fabric(actor_exploration), artifact_path="actor_exploration"
            )
            for k in critics_exploration.keys():
                model_info[f"critic_exploration_{k}"] = mlflow.pytorch.log_model(
                    critics_exploration[k]["module"], artifact_path=f"critic_exploration_{k}"
                )
                model_info[f"target_critic_exploration_{k}"] = mlflow.pytorch.log_model(
                    critics_exploration[k]["target_module"], artifact_path=f"target_critic_exploration_{k}"
                )
                model_info[f"moments_exploration_{k}"] = mlflow.pytorch.log_model(
                    moments_exploration[k], artifact_path=f"moments_exploration_{k}"
                )
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info


def compute_intrinsic_reward(cfg, imagined_trajectories, imagined_actions, world_model, ensembles=None):
    if cfg.algo.intrinsic_reward == "curiosity":
        return compute_curiosity_intrinsic_reward(cfg, imagined_trajectories, imagined_actions, world_model)
    elif cfg.algo.intrinsic_reward == "random_network_distillation":
        pass
    elif cfg.algo.intrinsic_reward == "progress":
        pass
    elif cfg.algo.intrinsic_reward == "plan2explore":
        return compute_plan2explore_intrinsic_reward(cfg, imagined_trajectories, imagined_actions, ensembles)
    else:
        raise ValueError(f"Unknown intrinsic reward type: {cfg.algo.intrinsic_reward}")

#curiosity
def compute_curiosity_intrinsic_reward(cfg, imagined_trajectories, imagined_actions, world_model):
    # 1. 현재 상태와 행동으로 다음 상태 예측
    current_states = imagined_trajectories[:-1]  # 마지막 상태 제외
    next_states = imagined_trajectories[1:]      # 첫 상태 제외
    
    predicted_next_states = world_model.transition_model(current_states, imagined_actions[:-1])
    
    # 2. 예측 오차 계산
    prediction_error = F.mse_loss(predicted_next_states, next_states, reduction='none')
    
    # 3. 예측 오차를 내재적 보상으로 사용
    intrinsic_reward = prediction_error.mean(dim=-1, keepdim=True)
    
    # 보상 스케일링 
    if hasattr(cfg.algo, 'curiosity_reward_scale'):
        intrinsic_reward *= cfg.algo.curiosity_reward_scale
    
    return intrinsic_reward

def compute_plan2explore_intrinsic_reward(cfg,imagined_trajectories, imagined_actions, ensembles):
    next_state_embedding = torch.empty(
        len(ensembles),
        cfg.algo.horizon + 1,
        cfg.batch_size * cfg.sequence_length,
        cfg.stochastic_size * cfg.discrete_size,
        device=cfg.device,
    )
    for i, ens in enumerate(ensembles):
        next_state_embedding[i] = ens(
            torch.cat((imagined_trajectories.detach(), imagined_actions.detach()), -1)
        )
    
    # Compute intrinsic reward as the variance of ensemble predictions
    intrinsic_reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
    return intrinsic_reward