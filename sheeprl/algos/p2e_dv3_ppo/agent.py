
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from lightning.pytorch.utilities.seed import isolate_rng

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits
from sheeprl.utils.distribution import TruncatedNormal
from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.models.models import (
    CNN,
    MLP,
    DeCNN,
    LayerNorm,
    LayerNormChannelLast,
    LayerNormGRUCell,
    MultiDecoder,
    MultiEncoder,
)
from sheeprl.utils.utils import safeatanh, safetanh
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.model import ModuleType, cnn_forward
from sheeprl.utils.utils import symlog
from sheeprl.algos.dreamer_v3.agent import CNNEncoder, CNNDecoder, MLPEncoder, MLPDecoder, RecurrentModel, RSSM, DecoupledRSSM, Actor


class PPOActor(nn.Module):
    def __init__(
        self,
        actor_backbone: torch.nn.Module,
        actor_heads: torch.nn.ModuleList,
        is_continuous: bool,
        distribution: str = "auto",
    ) -> None:
        super().__init__()
        self.actor_backbone = actor_backbone
        self.mlp_heads = actor_heads
        self.is_continuous = is_continuous
        self.distribution = distribution

    def forward(self, x: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None) -> List[Tensor]:
        x = self.actor_backbone(x)
        actions = [head(x) for head in self.mlp_heads]
        
        if greedy:
            actions = [torch.argmax(action, dim=-1) if not self.is_continuous else action for action in actions]
        
        if mask is not None:
            actions = [action * mask.get(f'action_{i}', 1.0) for i, action in enumerate(actions)]
        
        return actions

    def get_actions(self, x: Tensor, greedy: bool = False, mask: Optional[Dict[str, Tensor]] = None) -> List[Tensor]:
        return self.forward(x, greedy, mask)

class PPOAgent(nn.Module):
    def __init__(
        self,
        latent_state_size: int,
        actions_dim: Sequence[int],
        cfg: Dict[str, Any],
        is_continuous: bool = False,
    ):
        super().__init__()
        
        actor_cfg = cfg.algo.actor
        critic_cfg = cfg.algo.critic
        distribution_cfg = cfg.distribution
        
        self.is_continuous = is_continuous
        self.distribution_cfg = distribution_cfg
        self.distribution = distribution_cfg.get("type", "auto").lower() #agent의 action space의 분포.
        if self.distribution not in ("auto", "normal", "tanh_normal", "discrete"):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, `normal` and `tanh_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "discrete" and is_continuous: 
            raise ValueError("You have choose a discrete distribution but `is_continuous` is true")
        elif self.distribution not in {"discrete", "auto"} and not is_continuous:
            raise ValueError("You have choose a continuous distribution but `is_continuous` is false")
        if self.distribution == "auto": #is_continuous가 True면 "normal"을, False면 "discrete"를 선택.
            if is_continuous:
                self.distribution = "normal"
            else:
                self.distribution = "discrete"
        self.actions_dim = actions_dim
        # actor_backbone = (
        #     MLP(
        #         input_dims=latent_state_size,
        #         output_dim=None,
        #         hidden_sizes=[actor_cfg.dense_units] * actor_cfg.mlp_layers,
        #         activation=hydra.utils.get_class(actor_cfg.dense_act),
        #         flatten_dim=None,
        #         norm_layer=[nn.LayerNorm] * actor_cfg.mlp_layers if actor_cfg.layer_norm else None,
        #         norm_args=(
        #             [{"normalized_shape": actor_cfg.dense_units} for _ in range(actor_cfg.mlp_layers)]
        #             if actor_cfg.layer_norm
        #             else None
        #         ),
        #     )
        #     if actor_cfg.mlp_layers > 0
        #     else nn.Identity()
        # )
        
        #각 행동 차원에 대해 두 개의 값을 출력: 평균(μ)과 표준편차(σ)의 로그값이 필요하니깐 곱하기 2를 해주고,  출력된 평균과 표준편차로 정규 분포를 정의한 걸 바탕으로 action의 분포를 구한다음 거기에서 샘플링.
        
        if is_continuous:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, sum(actions_dim) * 2)]) 
        else:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, action_dim) for action_dim in actions_dim]) #dense layer에서 softmax처럼 heads를 구성
        #self.actor = PPOActor(actor_backbone, actor_heads, is_continuous, self.distribution)
        self.actor = Actor(
            latent_state_size=latent_state_size,
            actions_dim=actions_dim,
            is_continuous=is_continuous,
            init_std=actor_cfg.init_std,
            min_std=actor_cfg.min_std,
            dense_units=actor_cfg.dense_units,
            activation=hydra.utils.get_class(actor_cfg.dense_act),
            mlp_layers=actor_cfg.mlp_layers,
            distribution_cfg=cfg.distribution,
            layer_norm_cls=hydra.utils.get_class(actor_cfg.layer_norm.cls),
            layer_norm_kw=actor_cfg.layer_norm.kw,
            unimix=cfg.algo.unimix,
        )


        #입력은 특징(features)이고, 출력은 단일 값(가치 추정치), 
        self.critic = MLP(
            input_dims=latent_state_size,
            output_dim=1,
            hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
            activation=hydra.utils.get_class(critic_cfg.dense_act),
            norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
            norm_args=(
                [{"normalized_shape": critic_cfg.dense_units} for _ in range(critic_cfg.mlp_layers)]
                if critic_cfg.layer_norm
                else None
            ),
        )
        
    def _normal(self, actor_out: Tensor, actions: Optional[List[Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
        std = log_std.exp()
        normal = Independent(Normal(mean, std), 1)
        actions = actions[0]
        log_prob = normal.log_prob(actions)
        return actions, log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1)

    def _tanh_normal(self, actor_out: Tensor, actions: Optional[List[Tensor]] = None) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
        std = log_std.exp()
        normal = Independent(Normal(mean, std), 1)
        tanh_actions = actions[0].float()
        actions = safeatanh(tanh_actions, eps=torch.finfo(tanh_actions.dtype).resolution)
        log_prob = normal.log_prob(actions)
        log_prob -= 2.0 * (
            torch.log(torch.tensor([2.0], dtype=actions.dtype, device=actions.device))
            - tanh_actions
            - torch.nn.functional.softplus(-2.0 * tanh_actions)
        ).sum(-1, keepdim=False)
        return tanh_actions, log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1)
    
    def forward(
        self, latent_state: Tensor, actions: Optional[List[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        print("latent_state:", latent_state.shape)
        actor_out, action_dists = self.actor(latent_state)
        values = self.critic(latent_state)
        print("values:",values)
        
        if self.is_continuous:
            if self.distribution == "normal":
                actions, log_prob, entropy = self._normal(actor_out[0], actions)
            elif self.distribution == "tanh_normal":
                actions, log_prob, entropy = self._tanh_normal(actor_out[0], actions)
            return tuple([actions]), log_prob, entropy, values
        else:
            # should_append가 discrete action space에서 새로운 행동을 샘플링해야하는지에 대한 여부를 확인.
            should_append = False
            actions_logprobs: List[Tensor] = []
            actions_entropies: List[Tensor] = []
            actions_dist: List[Distribution] = action_dists
            if actions is None:
                should_append = True
                actions = []
            
            for i, dist in enumerate(actions_dist):
                actions_entropies.append(dist.entropy()) # 나중에 entropy bonus 등을 계산하기 위해서.
                if should_append:
                    actions.append(dist.sample())
                log_prob = dist.log_prob(actions[i])
                actions_logprobs.append(log_prob)
            
            if isinstance(actions, list):
                actions = torch.cat(actions, dim=-1)
            
            return (
                actions,
                torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
                torch.stack(actions_entropies, dim=-1).sum(dim=-1, keepdim=True),
                values,
            )

    # def forward(
    #     self, latent_state: Tensor, actions: Optional[List[Tensor]] = None
    # ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
    #     print("latent_state:",latent_state.shape)
    #     actor_out: List[Tensor] = self.actor(latent_state)
    #     values = self.critic(latent_state)
    #     if self.is_continuous:
    #         if self.distribution == "normal":
    #             actions, log_prob, entropy = self._normal(actor_out[0], actions)
    #         elif self.distribution == "tanh_normal":
    #             actions, log_prob, entropy = self._tanh_normal(actor_out[0], actions)
    #         return tuple([actions]), log_prob, entropy, values
    #     else:
    #         # should_append가 discrete action space에서 새로운 행동을 샘플링해야하는지에 대한 여부를 확인.
    #         should_append = False
    #         actions_logprobs: List[Tensor] = []
    #         actions_entropies: List[Tensor] = []
    #         actions_dist: List[Distribution] = []
    #         if actions is None:
    #             should_append = True
    #             actions: List[Tensor] = []
    #         for i, logits in enumerate(actor_out):
    #             actions_dist.append(OneHotCategorical(logits=logits))
    #             actions_entropies.append(actions_dist[-1].entropy()) #나중에 이거가지고 탐색(Exploration)과 활용(Exploitation)의 균형을 주기 위해서 entropy bonus같은 거 계산하기 위함인듯.
    #             if should_append:
    #                 actions.append(actions_dist[-1].sample())
    #             actions_logprobs.append(actions_dist[-1].log_prob(actions[i])) 
    #             if isinstance(actions, list):
    #                 actions = torch.cat(actions, dim=-1)
    #             print(f"actions shape in forward: {actions.shape if isinstance(actions, torch.Tensor) else [a.shape for a in actions]}")
    #         return (
    #             actions,
    #             torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
    #             torch.stack(actions_entropies, dim=-1).sum(dim=-1, keepdim=True),
    #             values,
    #         )

class PPOPlayer(nn.Module):
    """
    The model of the PPO player.

    Args:
        encoder (MultiEncoder): the encoder.
        rssm (RSSM | DecoupledRSSM): the RSSM model.
        actor (PPOActor): the PPO actor.
        actions_dim (Sequence[int]): the dimension of the actions.
        num_envs (int): the number of environments.
        stochastic_size (int): the size of the stochastic state.
        recurrent_state_size (int): the size of the recurrent state.
        device (str | torch.device): the device to use.
        discrete_size (int): the dimension of a single Categorical variable in the
            stochastic state (prior or posterior).
    """

    def __init__(
        self,
        encoder: MultiEncoder | _FabricModule,
        rssm: RSSM | DecoupledRSSM,
        actor: PPOActor,
        actions_dim: Sequence[int],
        num_envs: int,
        stochastic_size: int,
        recurrent_state_size: int,
        device: str | torch.device,
        discrete_size: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.stochastic_size = stochastic_size
        self.recurrent_state_size = recurrent_state_size
        self.device = device
        self.discrete_size = discrete_size
        self.decoupled_rssm = isinstance(rssm, DecoupledRSSM)

    @torch.no_grad()
    def init_states(self, reset_envs: Optional[Sequence[int]] = None) -> None:
        """Initialize the states and the actions for the ended environments."""
        if reset_envs is None or len(reset_envs) == 0:
            self.actions = torch.zeros(1, self.num_envs, np.sum(self.actions_dim), device=self.device)
            self.recurrent_state, stochastic_state = self.rssm.get_initial_states((1, self.num_envs))
            self.stochastic_state = stochastic_state.reshape(1, self.num_envs, -1)
        else:
            self.actions[:, reset_envs] = torch.zeros_like(self.actions[:, reset_envs])
            self.recurrent_state[:, reset_envs], stochastic_state = self.rssm.get_initial_states((1, len(reset_envs)))
            self.stochastic_state[:, reset_envs] = stochastic_state.reshape(1, len(reset_envs), -1)

    def get_actions(
        self,
        obs: Dict[str, Tensor],
        greedy: bool = False,
        mask: Optional[Dict[str, Tensor]] = None,
    ) -> Sequence[Tensor]:
        """
        Return the actions based on the current observations.

        Args:
            obs (Dict[str, Tensor]): the current observations.
            greedy (bool): whether or not to use greedy action selection.
            mask (Optional[Dict[str, Tensor]]): action masks, if any.

        Returns:
            The actions the agent has to perform.
        """
        embedded_obs = self.encoder(obs)
        self.recurrent_state = self.rssm.recurrent_model(
            torch.cat((self.stochastic_state, self.actions), -1), self.recurrent_state
        )
        if self.decoupled_rssm:
            _, self.stochastic_state = self.rssm._representation(embedded_obs)
        else:
            _, self.stochastic_state = self.rssm._representation(self.recurrent_state, embedded_obs)
        self.stochastic_state = self.stochastic_state.view(
            *self.stochastic_state.shape[:-2], self.stochastic_size * self.discrete_size
        )
        
        state = torch.cat((self.stochastic_state, self.recurrent_state), -1)
        actions, _ = self.actor(state)
        
        self.actions = torch.cat(actions, -1)
        return actions

    def get_value(self, state: Tensor) -> Tensor:
        """
        Get the value estimate for a given state.

        Args:
            state (Tensor): The state to evaluate.

        Returns:
            Tensor: The estimated value of the state.
        """
        return self.actor.get_value(state)
    
# class PPOPlayer(nn.Module):    
#     def __init__(self, world_model: WorldModel, actor: PPOActor, critic: nn.Module) -> None:
#         super().__init__()
#         self.world_model = world_model
#         self.actor = actor
#         self.critic = critic
        
#     def _normal(self, actor_out: Tensor) -> Tuple[Tensor, Tensor]:
#         mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
#         std = log_std.exp()
#         normal = Independent(Normal(mean, std), 1)
#         actions = normal.sample()
#         log_prob = normal.log_prob(actions)
#         return actions, log_prob.unsqueeze(dim=-1)

#     def _tanh_normal(self, actor_out: Tensor) -> Tuple[Tensor, Tensor]:
#         mean, log_std = torch.chunk(actor_out, chunks=2, dim=-1)
#         std = log_std.exp()
#         normal = Independent(Normal(mean, std), 1)
#         actions = normal.sample().float()
#         tanh_actions = safetanh(actions, eps=torch.finfo(actions.dtype).resolution)
#         log_prob = normal.log_prob(actions)
#         log_prob -= 2.0 * (
#             torch.log(torch.tensor([2.0], dtype=actions.dtype, device=actions.device))
#             - tanh_actions
#             - torch.nn.functional.softplus(-2.0 * tanh_actions)
#         ).sum(-1, keepdim=False)
#         return tanh_actions, log_prob.unsqueeze(dim=-1)
    
#     def forward(self, latent_state: Tensor) -> Tuple[Sequence[Tensor], Tensor, Tensor]:
#         values = self.critic(latent_state)
#         actor_out: List[Tensor] = self.actor(latent_state)
#         if self.actor.is_continuous:
#             if self.actor.distribution == "normal":
#                 actions, log_prob = self._normal(actor_out[0])
#             elif self.actor.distribution == "tanh_normal":
#                 actions, log_prob = self._tanh_normal(actor_out[0])
#             return tuple([actions]), log_prob, values
#         else:
#             actions_dist: List[Distribution] = []
#             actions_logprobs: List[Tensor] = []
#             actions: List[Tensor] = []
#             for i, logits in enumerate(actor_out):
#                 actions_dist.append(OneHotCategorical(logits=logits))
#                 actions.append(actions_dist[-1].sample())
#                 actions_logprobs.append(actions_dist[-1].log_prob(actions[i]))
#             return (
#                 tuple(actions),
#                 torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
#                 values,
#             )
            
#     def get_values(self, latent_state: Tensor) -> Tensor:
#         return self.critic(latent_state)

#     def get_actions(self, latent_state: Tensor, greedy: bool = False) -> Sequence[Tensor]:
#         actor_out: List[Tensor] = self.actor(latent_state)
#         if self.actor.is_continuous:
#             mean, log_std = torch.chunk(actor_out[0], chunks=2, dim=-1)
#             if greedy:
#                 actions = mean
#             else:
#                 std = log_std.exp()
#                 normal = Independent(Normal(mean, std), 1)
#                 actions = normal.sample()
#             if self.actor.distribution == "tanh_normal":
#                 actions = safeatanh(actions, eps=torch.finfo(actions.dtype).resolution)
#             return tuple([actions])
#         else:
#             actions: List[Tensor] = []
#             actions_dist: List[Distribution] = []
#             for logits in actor_out:
#                 actions_dist.append(OneHotCategorical(logits=logits))
#                 if greedy:
#                     actions.append(actions_dist[-1].mode)
#                 else:
#                     actions.append(actions_dist[-1].sample())
#             return tuple(actions)

#     def act(self, obs: Dict[str, Tensor], greedy: bool = False) -> Sequence[Tensor]:
#         if self.world_model is not None:
#             with torch.no_grad():
#                 latent_state = self.world_model.get_latent_state(obs)
#         else:
#             latent_state = obs  # 이미 latent state인 경우
#         return self.get_actions(latent_state, greedy)
        
def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: Dict[str, Any],
    world_model_state: Optional[Dict[str, torch.Tensor]] = None,
    ensembles_state: Optional[Dict[str, torch.Tensor]] = None,
    ppo_agent_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[WorldModel, nn.ModuleList, PPOAgent, PPOPlayer]:
    """Build the models and wrap them with Fabric.

    Args:
        fabric (Fabric): the fabric object.
        actions_dim (Sequence[int]): the dimension of the actions.
        is_continuous (bool): whether or not the actions are continuous.
        cfg (DictConfig): the configs of DreamerV3.
        obs_space (Dict[str, Any]): the observation space.
        world_model_state (Dict[str, Tensor], optional): the state of the world model.
            Default to None.
        actor_state: (Dict[str, Tensor], optional): the state of the actor.
            Default to None.
        critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.
        target_critic_state: (Dict[str, Tensor], optional): the state of the critic.
            Default to None.

    Returns:
        The world model (WorldModel): composed by the encoder, rssm, observation and
        reward models and the continue model.
        The actor (_FabricModule).
        The critic (_FabricModule).
        The target critic (nn.Module).
    """
    world_model_cfg = cfg.algo.world_model
    ppo_agent_cfg = cfg.algo.ppo_agent
    # critic_cfg = cfg.algo.critic

    # Sizes
    recurrent_state_size = world_model_cfg.recurrent_model.recurrent_state_size
    stochastic_size = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size + recurrent_state_size

    # Define models
    cnn_stages = int(np.log2(cfg.env.screen_size) - np.log2(4))
    cnn_encoder = (
        CNNEncoder(
            keys=cfg.algo.cnn_keys.encoder,
            input_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.encoder],
            image_size=obs_space[cfg.algo.cnn_keys.encoder[0]].shape[-2:],
            channels_multiplier=world_model_cfg.encoder.cnn_channels_multiplier,
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.encoder.cnn_layer_norm.cls),
            layer_norm_kw=world_model_cfg.encoder.cnn_layer_norm.kw,
            activation=hydra.utils.get_class(world_model_cfg.encoder.cnn_act),
            stages=cnn_stages,
        )
        if cfg.algo.cnn_keys.encoder is not None and len(cfg.algo.cnn_keys.encoder) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            keys=cfg.algo.mlp_keys.encoder,
            input_dims=[obs_space[k].shape[0] for k in cfg.algo.mlp_keys.encoder],
            mlp_layers=world_model_cfg.encoder.mlp_layers,
            dense_units=world_model_cfg.encoder.dense_units,
            activation=hydra.utils.get_class(world_model_cfg.encoder.dense_act),
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.encoder.mlp_layer_norm.cls),
            layer_norm_kw=world_model_cfg.encoder.mlp_layer_norm.kw,
        )
        if cfg.algo.mlp_keys.encoder is not None and len(cfg.algo.mlp_keys.encoder) > 0
        else None
    )
    encoder = MultiEncoder(cnn_encoder, mlp_encoder)

    recurrent_model = RecurrentModel(
        input_size=int(sum(actions_dim) + stochastic_size),
        recurrent_state_size=world_model_cfg.recurrent_model.recurrent_state_size,
        dense_units=world_model_cfg.recurrent_model.dense_units,
        layer_norm_cls=hydra.utils.get_class(world_model_cfg.recurrent_model.layer_norm.cls),
        layer_norm_kw=world_model_cfg.recurrent_model.layer_norm.kw,
    )
    represention_model_input_size = encoder.output_dim
    if not cfg.algo.world_model.decoupled_rssm:
        represention_model_input_size += recurrent_state_size
    representation_ln_cls = hydra.utils.get_class(world_model_cfg.representation_model.layer_norm.cls)
    representation_model = MLP(
        input_dims=represention_model_input_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.representation_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.representation_model.dense_act),
        layer_args={"bias": representation_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[representation_ln_cls],
        norm_args=[
            {
                **world_model_cfg.representation_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.representation_model.hidden_size,
            }
        ],
    )
    transition_ln_cls = hydra.utils.get_class(world_model_cfg.transition_model.layer_norm.cls)
    transition_model = MLP(
        input_dims=recurrent_state_size,
        output_dim=stochastic_size,
        hidden_sizes=[world_model_cfg.transition_model.hidden_size],
        activation=hydra.utils.get_class(world_model_cfg.transition_model.dense_act),
        layer_args={"bias": transition_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=[transition_ln_cls],
        norm_args=[
            {
                **world_model_cfg.transition_model.layer_norm.kw,
                "normalized_shape": world_model_cfg.transition_model.hidden_size,
            }
        ],
    )

    if cfg.algo.world_model.decoupled_rssm:
        rssm_cls = DecoupledRSSM
    else:
        rssm_cls = RSSM
    rssm = rssm_cls(
        recurrent_model=recurrent_model.apply(init_weights),
        representation_model=representation_model.apply(init_weights),
        transition_model=transition_model.apply(init_weights),
        distribution_cfg=cfg.distribution,
        discrete=world_model_cfg.discrete_size,
        unimix=cfg.algo.unimix,
        learnable_initial_recurrent_state=cfg.algo.world_model.learnable_initial_recurrent_state,
        sampling = world_model_cfg.get('sampling', True)
    ).to(fabric.device)

    cnn_decoder = (
        CNNDecoder(
            keys=cfg.algo.cnn_keys.decoder,
            output_channels=[int(np.prod(obs_space[k].shape[:-2])) for k in cfg.algo.cnn_keys.decoder],
            channels_multiplier=world_model_cfg.observation_model.cnn_channels_multiplier,
            latent_state_size=latent_state_size,
            cnn_encoder_output_dim=cnn_encoder.output_dim,
            image_size=obs_space[cfg.algo.cnn_keys.decoder[0]].shape[-2:],
            activation=hydra.utils.get_class(world_model_cfg.observation_model.cnn_act),
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.observation_model.cnn_layer_norm.cls),
            layer_norm_kw=world_model_cfg.observation_model.mlp_layer_norm.kw,
            stages=cnn_stages,
        )
        if cfg.algo.cnn_keys.decoder is not None and len(cfg.algo.cnn_keys.decoder) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            keys=cfg.algo.mlp_keys.decoder,
            output_dims=[obs_space[k].shape[0] for k in cfg.algo.mlp_keys.decoder],
            latent_state_size=latent_state_size,
            mlp_layers=world_model_cfg.observation_model.mlp_layers,
            dense_units=world_model_cfg.observation_model.dense_units,
            activation=hydra.utils.get_class(world_model_cfg.observation_model.dense_act),
            layer_norm_cls=hydra.utils.get_class(world_model_cfg.observation_model.mlp_layer_norm.cls),
            layer_norm_kw=world_model_cfg.observation_model.mlp_layer_norm.kw,
        )
        if cfg.algo.mlp_keys.decoder is not None and len(cfg.algo.mlp_keys.decoder) > 0
        else None
    )
    
    observation_model = MultiDecoder(cnn_decoder, mlp_decoder)

    reward_ln_cls = hydra.utils.get_class(world_model_cfg.reward_model.layer_norm.cls)
    
    reward_model = MLP(
        input_dims=latent_state_size,
        output_dim=world_model_cfg.reward_model.bins,
        hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.reward_model.dense_act),
        layer_args={"bias": reward_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=reward_ln_cls,
        norm_args={
            **world_model_cfg.reward_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.reward_model.dense_units,
        },
    )

    discount_ln_cls = hydra.utils.get_class(world_model_cfg.discount_model.layer_norm.cls)
    continue_model = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[world_model_cfg.discount_model.dense_units] * world_model_cfg.discount_model.mlp_layers,
        activation=hydra.utils.get_class(world_model_cfg.discount_model.dense_act),
        layer_args={"bias": discount_ln_cls == nn.Identity},
        flatten_dim=None,
        norm_layer=discount_ln_cls,
        norm_args={
            **world_model_cfg.discount_model.layer_norm.kw,
            "normalized_shape": world_model_cfg.discount_model.dense_units,
        },
    )
    world_model = WorldModel(
        encoder.apply(init_weights),
        rssm,
        observation_model.apply(init_weights),
        reward_model.apply(init_weights),
        continue_model.apply(init_weights),
    )

    ppo_agent = PPOAgent(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        cfg = cfg,
        is_continuous=is_continuous,
    )
    

    ppo_agent.apply(init_weights)

    if cfg.algo.hafner_initialization:
        #수정
        ppo_agent.actor.mlp_heads.apply(uniform_init_weights(1.0))
        
        ppo_agent.critic.model[-1].apply(uniform_init_weights(0.0))
        rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
        rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))
        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
        if mlp_decoder is not None:
            mlp_decoder.heads.apply(uniform_init_weights(1.0))
        if cnn_decoder is not None:
            cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))

    # Load models from checkpoint
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if ppo_agent_state:
        ppo_agent.actor.load_state_dict(ppo_agent_state)

    # Create the player agent
    fabric_player = get_single_device_fabric(fabric)
    
    # player = PPOPlayer(
    #     world_model=copy.deepcopy(world_model),
    #     actor=copy.deepcopy(ppo_agent.actor),
    #     critic=copy.deepcopy(ppo_agent.critic)
    # )
    # Create the player agent
    fabric_player = get_single_device_fabric(fabric)
    player = PPOPlayer(
        copy.deepcopy(world_model.encoder),
        copy.deepcopy(world_model.rssm),
        copy.deepcopy(ppo_agent.actor),
        actions_dim,
        cfg.env.num_envs,
        cfg.algo.world_model.stochastic_size,
        cfg.algo.world_model.recurrent_model.recurrent_state_size,
        fabric_player.device,
        discrete_size=cfg.algo.world_model.discrete_size,
    )    
    ens_list = []
    cfg_ensembles = cfg.algo.ensembles
    ensembles_ln_cls = hydra.utils.get_class(cfg_ensembles.layer_norm.cls)
    with isolate_rng(): #앙상블이니깐 서로 다른 시드로 로드
        for i in range(cfg_ensembles.n):
            fabric.seed_everything(cfg.seed + i)
            #서로 다른 예측을 하게해서 그걸 mlp layer에 통과. 이때 입력차원이 action_dim +recurrent_state_size(h_t) + (stochasitc/z_t)
            ens_list.append(
                MLP(
                    input_dims=int(
                        sum(actions_dim)
                        + cfg.algo.world_model.recurrent_model.recurrent_state_size
                        + cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
                    ),
                    output_dim=cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size, #next s_t
                    hidden_sizes=[cfg_ensembles.dense_units] * cfg_ensembles.mlp_layers,
                    activation=hydra.utils.get_class(cfg_ensembles.dense_act),
                    flatten_dim=None,
                    layer_args={"bias": ensembles_ln_cls == nn.Identity},
                    norm_layer=ensembles_ln_cls,
                    norm_args={
                        **cfg_ensembles.layer_norm.kw,
                        "normalized_shape": cfg_ensembles.dense_units,
                    },
                ).apply(init_weights)
            )
    ensembles = nn.ModuleList(ens_list)
    if ensembles_state:
        ensembles.load_state_dict(ensembles_state)
    for i in range(len(ensembles)):
        ensembles[i] = fabric.setup_module(ensembles[i])

    # Setup models with Fabric
    world_model.encoder = fabric.setup_module(world_model.encoder)
    world_model.observation_model = fabric.setup_module(world_model.observation_model)
    world_model.reward_model = fabric.setup_module(world_model.reward_model)
    world_model.rssm.recurrent_model = fabric.setup_module(world_model.rssm.recurrent_model)
    world_model.rssm.representation_model = fabric.setup_module(world_model.rssm.representation_model)
    world_model.rssm.transition_model = fabric.setup_module(world_model.rssm.transition_model)
    if world_model.continue_model:
        world_model.continue_model = fabric.setup_module(world_model.continue_model)
    # Setup PPO agent
    ppo_agent.actor = fabric.setup_module(ppo_agent.actor)
    ppo_agent.critic = fabric.setup_module(ppo_agent.critic)


    # Setup the player agent with a single-device Fabric
    player.encoder = fabric_player.setup_module(player.encoder)
    player.rssm.recurrent_model = fabric_player.setup_module(player.rssm.recurrent_model)
    player.rssm.transition_model = fabric_player.setup_module(player.rssm.transition_model)
    player.rssm.representation_model = fabric_player.setup_module(player.rssm.representation_model)
    player.actor = fabric_player.setup_module(player.actor)

    # Tie weights between the agent and the player
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.parameters(), player.rssm.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(ppo_agent.actor.parameters(), player.actor.parameters()):
        p.data = agent_p.data
    return world_model, ensembles, ppo_agent, player