import copy
import time
import os
import warnings
from typing import Any, Dict, Sequence, Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, OneHotCategorical
from torchmetrics import SumMetric
from sheeprl.algos.p2e_dv3_ppo.agent import PPOAgent
from sheeprl.utils.utils import gae_worldmodel
from sheeprl.algos.dreamer_v3.agent import WorldModel
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.algos.p2e_dv3_ppo.agent import build_agent
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
from sheeprl.utils.env import make_env
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs, unwrap_fabric

# Decomment the following line if you are using MineDojo on an headless machine
# os.environ["MINEDOJO_HEADLESS"] = "1"

def train(
    fabric: Fabric,
    world_model: WorldModel,
    ppo_agent: PPOAgent,
    old_ppo_agent: Optional[PPOAgent],
    ensembles: _FabricModule,
    world_optimizer: _FabricOptimizer,
    ppo_agent_optimizer: _FabricOptimizer,
    ensemble_optimizer: _FabricOptimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator,
    cfg: DictConfig,
    is_continuous: bool,
    actions_dim: Sequence[int],
) -> None:
    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    data = {k: data[k] for k in data.keys()} #실제로는 batch랑 같은 값. shallow copy의 의미.
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :]) #torch.Size([64, 16, 1]) #16개의 배치에 64 seq_length니깐 앞에 있는 걸 다 1로 바꿔라

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    #첫 번째 타임스텝에는 이전 액션이 없으므로, 0으로 채워진 "더미" 액션을 삽입
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0) #현재 상태, 현재 액션 -> 다음 상태, 보상으로 모델링하는데, rb에는 현재 상태, 이전 액션, 현재 보상으로 되어있으니깐, 각 상태에 대해서 일괄적으로 이전에 취한 액션으로 

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size #32개의 discrete variable 안에 32개의 stochastic variable..?
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device) #torch.Size([1, 1, 4096]) #현재 시점
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device) #torch.Size([1, 1, 32, 32])
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device) #torch.Size([64, 1, 4096]) #전체 시점에 대한 recurrent
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device) #torch.Size([64, 1, 1024]) #32*32
    posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device) #torch.Size([64, 1, 32, 32])
    posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device) #torch.Size([64, 1, 1024])

    # embedded observations from the environment
    embedded_obs = world_model.encoder(batch_obs) #torch.Size([64, 16, 12288]) 64*64*3

    for i in range(0, sequence_length):
        recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
            posterior, recurrent_state, batch_actions[i : i + 1], embedded_obs[i : i + 1], data["is_first"][i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        priors_logits[i] = prior_logits
        posteriors[i] = posterior
        posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
 
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states) #torch.Size([64, 16, 3, 64, 64])

    # compute the distribution over the reconstructed observations
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
        for k in cfg.algo.cnn_keys.decoder
    }
    po.update(
        {
            k: SymlogDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
            for k in cfg.algo.mlp_keys.decoder
        }
    )
    # Compute the distribution over the rewards #reward predictor
    pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states.detach()), dims=1) #torch.Size([64, 16, 1])

    # Compute the distribution over the terminal steps, if required #continue predictor
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states.detach())), 1) #torch.Size([64, 16, 1]
    continues_targets = 1 - data["terminated"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32] #torch.Size([64, 1, 32, 32])
    #prior, posterior의 shape맞춰주기. kl-div 계산해야하니깐.
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # world model optimization step
    world_optimizer.zero_grad(set_to_none=True)
    #결국 world-model loss는 여기서.
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        cfg.algo.world_model.kl_dynamic,
        cfg.algo.world_model.kl_representation,
        cfg.algo.world_model.kl_free_nats,
        cfg.algo.world_model.kl_regularizer,
        pc,
        continues_targets,
        cfg.algo.world_model.continue_scale_factor,
    )
    fabric.backward(rec_loss)
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model,
            optimizer=world_optimizer,
            max_norm=cfg.algo.world_model.clip_gradients,
            error_if_nonfinite=False,
        )
    world_optimizer.step()

    # Free up space
    del posterior
    del prior_logits
    del recurrent_state
    del posterior_logits
    world_optimizer.zero_grad(set_to_none=True)

    # Ensemble Learning
    loss = 0.0
    ensemble_optimizer.zero_grad(set_to_none=True)
    #posterior, action, h_t를 input으로 받아서 다음 상태를 예측하는 것을 훈련, 나중에 이 ensemble model의 variance를 재서 intrinsic reward를 측정하기 위해서
    for ens in ensembles:
        out = ens(
            torch.cat(
                (
                    posteriors.view(*posteriors.shape[:-2], -1).detach(), #torch.Size([64, 16, 1024])
                    recurrent_states.detach(), 
                    data["actions"].detach(),
                ),
                -1,
            )
        )[:-1]
        next_state_embedding_dist = MSEDistribution(out, 1)
        loss -= next_state_embedding_dist.log_prob(posteriors.view(sequence_length, batch_size, -1).detach()[1:]).mean()
    loss.backward()
    ensemble_grad = None
    if cfg.algo.ensembles.clip_gradients is not None and cfg.algo.ensembles.clip_gradients > 0:
        ensemble_grad = fabric.clip_gradients(
            module=ens,
            optimizer=ensemble_optimizer,
            max_norm=cfg.algo.ensembles.clip_gradients,
            error_if_nonfinite=False,
        )
    ensemble_optimizer.step()

    # Behaviour Learning Exploration
    # imagination이니깐 graident에서 분리하기 위해서 detatch 
    #여기까지가 준비과정이네.
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size) 
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        cfg.algo.horizon + 1,
        batch_size * sequence_length,
        stoch_state_size + recurrent_state_size,
        device=device,
    )
    imagined_trajectories[0] = imagined_latent_state #상상 궤적의 첫 번째 타임스텝을 초기화해서 그 trajectories의 첫번째에 latenet_state를 설정,
    imagined_actions = torch.empty(
        cfg.algo.horizon + 1,
        batch_size * sequence_length,
        data["actions"].shape[-1],
        device=device,
    )
    #deterministic하면 True, 아니면 False
    # greedy = cfg.algo.deterministic_actions
    
    actions, _, _, _ = ppo_agent(latent_states.detach()) 
    # print("actions:",actions.shape)
    # print("latent_states:",latent_states.shape)
        
    # imagined_actions[0] = actions

    # imagine trajectories in the latent space
    #여기서 부터 latent space level에서 시뮬레이션.
    # for i in range(1, cfg.algo.horizon + 1):
    #     imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
    #     imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
    #     imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    #     imagined_trajectories[i] = imagined_latent_state
    #     actions, _, _, _ = ppo_agent(imagined_latent_state.detach())
    #     imagined_actions[i] = actions


    next_state_embedding = torch.empty(
        len(ensembles),
        sequence_length,
        batch_size,
        stochastic_size * discrete_size,
        device=device,
    )
    
    for i, ens in enumerate(ensembles):
        next_state_embedding[i] = ens(
            torch.cat((latent_states.detach(), actions.detach()), -1)
        )

    # next_state_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
    # 결과적으로 reward가 여기서 주어지네.     
    reward = next_state_embedding.var(0).mean(-1, keepdim=True) * cfg.algo.intrinsic_reward_multiplier
    values = ppo_agent.critic(latent_states)
        
    # Compute continues
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1).mode
    # print(f"before_continues shape: {continues.shape}")   
    true_continue = (1 - data["terminated"])
    # print("true_continue:",true_continue.shape)
    continues = torch.cat((true_continue, continues[1:]))
    
    # print(f"reward shape: {reward.shape}")
    # print(f"values shape: {values.shape}")
    # print(f"continues shape: {continues.shape}")   
    # print(f"imagined_trajectories: {imagined_trajectories.shape}")
    # reward shape: torch.Size([16, 512, 1])
    # values shape: torch.Size([16, 512, 1])
    # continues shape: torch.Size([16, 512, 1])
    # imagined_trajectories: torch.Size([16, 512, 5120])
    
    # Compute GAE
    num_steps = sequence_length  
    
    returns, advantages = gae_worldmodel(
        rewards=reward,  
        values=values,   
        continues=continues * cfg.algo.gamma, 
        num_steps=num_steps,
        gamma=cfg.algo.gamma,
        gae_lambda=cfg.algo.gae_lambda
    )

    if aggregator and not aggregator.disabled:
        aggregator.update("Rewards/intrinsic", reward.detach().cpu().mean())
        aggregator.update("Values/predicted", values.detach().cpu().mean())
        aggregator.update("Advantages/gae", advantages.detach().cpu().mean())

    ppo_agent_optimizer.zero_grad(set_to_none=True)
    
    # print("latent_states_shape:",latent_states.shape) #각 sequence_length, batch_size만큼의 5120 latent_state가 있다.
    _, current_log_probs, entropy, current_values = ppo_agent(latent_states.detach())

    if old_ppo_agent is None:
        old_ppo_agent = ppo_agent
        
    with torch.no_grad():
        _, old_log_probs, _, _ = old_ppo_agent(latent_states.detach())

    # print(f"current_actions type: {type(current_actions)}")
    # print(f"current_actions shape: {current_actions.shape}")
    # print(f"old_actions shape: {old_actions.shape}")

    # print("current_log_probs:",current_log_probs.shape)
    # print("old_log_probs:",old_log_probs.shape)
    

    # PPO의 policy loss 계산
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # print("ratio:",ratio.shape)
    # print("advantages:",advantages.shape)
    
    surr1 = ratio * advantages
    
    cfg.algo.clip_range = 0.01
    
    surr2 = torch.clamp(ratio, 1.0 - cfg.algo.clip_range, 1.0 + cfg.algo.clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 엔트로피 보너스 계산
    try:
        entropy_bonus = cfg.algo.actor.ent_coef * entropy.mean()
    except NotImplementedError:
        entropy_bonus = torch.zeros_like(policy_loss)

    policy_loss_exploration = policy_loss - entropy_bonus
    fabric.backward(policy_loss_exploration)

    actor_grads_exploration = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads_exploration = fabric.clip_gradients(
            module=ppo_agent,
            optimizer=ppo_agent_optimizer,
            max_norm=cfg.algo.actor.clip_gradients,
            error_if_nonfinite=False,
        )
    ppo_agent_optimizer.step()
    
    #value loss 계산
    #현재 스텝의 보상 + 다음 스텝의 예측된 값 * 에피소드 지속여부
    #true_values = reward[:-1] + cfg.algo.gamma * values[1:] * continues[1:]
    
    value_loss = F.mse_loss(current_values, returns)
    fabric.backward(value_loss)
    
    # reset the world_model gradients, to avoid interferences with task learning
    world_optimizer.zero_grad(set_to_none=True)
    
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        aggregator.update("State/kl", kl.mean().detach())
        aggregator.update(
            "State/post_entropy",
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update("Loss/ensemble_loss", loss.detach().cpu())
        aggregator.update("Loss/policy_loss_exploration", policy_loss_exploration.detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        if ensemble_grad:
            aggregator.update("Grads/ensemble", ensemble_grad.detach())
        if actor_grads_exploration:
            aggregator.update("Grads/actor_exploration", actor_grads_exploration.mean().detach())

    # Reset everything
    ppo_agent_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    ensemble_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    whole_time = time.time()
    whole_interaction_time = 0
    whole_training_time = 0
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # These arguments cannot be changed
    cfg.env.frame_stack = 1
    cfg.algo.player.actor_type = "exploration"

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n]) #(9,)
    )
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    if (
        len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
        and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
    ):
        raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
    if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
        raise RuntimeError(
            "The CNN keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
        )
    if len(set(cfg.algo.mlp_keys.decoder) - set(cfg.algo.mlp_keys.encoder)) > 0:
        raise RuntimeError(
            "The MLP keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.mlp_keys.decoder))}"
        )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
        fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
        fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    (
        world_model,
        ensembles,
        ppo_agent,
        player,
    ) = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["ensembles"] if cfg.checkpoint.resume_from else None,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    ppo_agent_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=ppo_agent.parameters(), _convert_="all"
    )
    ensemble_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=ensembles.parameters(), _convert_="all"
    )
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        ppo_agent_optimizer.load_state_dict(state["actor_task_optimizer"])
        ensemble_optimizer.load_state_dict(state["ensemble_optimizer"])
    (
        world_optimizer,
        ppo_agent_optimizer,
        ensemble_optimizer,
    ) = fabric.setup_optimizers(
        world_optimizer,
        ppo_agent_optimizer,
        ensemble_optimizer,
    )
    
    cfg.metric.aggregator.metrics.pop("Loss/value_loss_exploration", None)
    cfg.metric.aggregator.metrics.pop("Values_exploration/predicted_values", None)
    cfg.metric.aggregator.metrics.pop("Values_exploration/lambda_values", None)
    cfg.metric.aggregator.metrics.pop("Grads/critic_exploration", None)
    cfg.metric.aggregator.metrics.pop("Rewards/intrinsic", None)
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 4
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")

    # Global variables
    train_step = 0
    last_train = 0
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["iter_num"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    prefill_steps = learning_starts - int(learning_starts > 0)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
        learning_starts += start_iter
        prefill_steps += start_iter

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )
    if cfg.checkpoint.every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )

    # Get the first environment observation and start the optimization
    step_data = {}
    obs = envs.reset(seed=cfg.seed)[0] #obs['rgb'].shape -> (3, 64, 64)
    for k in obs_keys: 
        step_data[k] = obs[k][np.newaxis]
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1)) #단일배치에 대해서 envs갯수만큼 0,1로 관리 (1,1,1)
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()
    
    #시뮬레이션
    
    cumulative_per_rank_gradient_steps = 0
    old_ppo_agent = None
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                if (
                    iter_num <= learning_starts #learning_starts가 초반에 랜덤한 액션해보는거네.
                    and cfg.checkpoint.resume_from is None
                    and "minedojo" not in cfg.algo.actor.cls.lower()
                ):
                    real_actions = actions = np.array(envs.action_space.sample())
                    #one_hot encoding.
                    if not is_continuous:
                        actions = np.concatenate(
                            [
                                F.one_hot(torch.as_tensor(act), act_dim).numpy()
                                for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                            ],
                            axis=-1,
                        )
                else:
                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs) #torch.Size([1, num_env, 3, 64, 64])
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    real_actions = actions = player.get_actions(torch_obs, mask=mask)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    if is_continuous:
                        real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )

                step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                rb.add(step_data, validate_args=cfg.buffer.validate_args)

                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
                        rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["is_first"][last_inserted_idx]
                        )
                        step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

            if cfg.metric.log_level > 0 and "final_info" in infos:
                for i, agent_ep_info in enumerate(infos["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

            # Save the real next observation
            real_next_obs = copy.deepcopy(next_obs)
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        for k, v in final_obs.items():
                            real_next_obs[k][idx] = v

            for k in obs_keys:
                step_data[k] = next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            rewards = rewards.reshape((1, cfg.env.num_envs, -1))
            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)

        # Train the agent
        if iter_num >= learning_starts:
            ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            if per_rank_gradient_steps > 0:
                local_data = rb.sample_tensors(
                    cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    dtype=None,
                    device=fabric.device,
                    from_numpy=cfg.buffer.from_numpy,
                )

                # Start training
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for i in range(per_rank_gradient_steps):
                        # 누적 graident_steps에 도달할때마다 타겟 네트워크를 업데이트
                        batch = {k: v[i].float() for k, v in local_data.items()} #torch.Size([64, 16, 3, 64, 64])
                        train(
                            fabric,
                            world_model,
                            ppo_agent,
                            old_ppo_agent,
                            ensembles,
                            world_optimizer,
                            ppo_agent_optimizer,
                            ensemble_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            is_continuous=is_continuous,
                            actions_dim=actions_dim,
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                    whole_training_time += timer_metrics["Time/train_time"]
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                    whole_interaction_time += timer_metrics["Time/env_interaction_time"]
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()
                
                time_string = f'curr_step: {policy_step}, whole_time: {round(time.time() - whole_time, 3)} sec = interaction_time: {round(whole_interaction_time, 3)} + training_time: {round(whole_training_time, 3)}' 
                fabric.print(time_string)
                
            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint Model
        old_ppo_agent = copy.deepcopy(ppo_agent)
        old_ppo_agent.eval()
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            iter_num == total_iters and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor_task": ppo_agent.state_dict(),
                "ensembles": ensembles.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "ensemble_optimizer": ensemble_optimizer.state_dict(),
                "ratio": ratio.state_dict(),
                "iter_num": iter_num * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    # task test zero-shot
    if fabric.is_global_zero and cfg.algo.run_test:
        player.actor_type = "task"
        fabric_player = get_single_device_fabric(fabric)
        player.actor = fabric_player.setup_module(unwrap_fabric(ppo_agent))
        test(player, fabric, cfg, log_dir, "zero-shot", greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "ensembles": ensembles,
            "actor_task": ppo_agent,
        }
        register_model(fabric, log_models, cfg, models_to_log)
