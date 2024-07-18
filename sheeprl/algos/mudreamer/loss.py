from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence


def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    pv: Distribution,
    lambda_values: Tensor,
    predicted_target_values: Distribution,
    discount: Tensor,
    pa: Distribution,
    actions: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 5 in

    Args:
        po (Dict[str, Distribution]): the distribution returned by the observation_model (decoder).
        observations (Tensor): the observations provided by the environment.
        pr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        pv (Distribution): the value distribution returned by the value_model.
        lambda_values (Tensor): Estimated lambda-values
        predicted_target_values (Distribution): slow moving target critic (실제 critic model에서 사용하는 target critic) TODO: value predictor를 위한 별개의 target critic이 있어야 하는건지?
        discount (Tensor): discount factor
        pa (Distribution): the action distribution returned by the action_model.
        actions (Tensor): The actual actions taken by the agent during the "Environment interaction" phase.
        priors_logits (Tensor): the logits of the prior.
        posteriors_logits (Tensor): the logits of the posterior.
        kl_dynamic (float): the kl-balancing dynamic loss regularizer.
            Defaults to 0.5.
        kl_balancing_alpha (float): the kl-balancing representation loss regularizer.
            Defaults to 0.1.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 1.0.
        kl_regularizer (float): scale factor of the KL divergence.
            Default to 1.0.
        pc (Bernoulli, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        continue_targets (Tensor, optional): the targets for the discount predictor. Those are normally computed
            as `(1 - data["dones"]) * args.gamma`.
            Default to None.
        continue_scale_factor (float): the scale factor for the continue loss.
            Default to 10.

    Returns:
        observation_loss (Tensor): the value of the observation loss.
        KL divergence (Tensor): the KL divergence between the posterior and the prior.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
    """
    rewards.device


    observation_loss = -sum([po[k].log_prob(observations[k]) for k in po.keys()])
    reward_loss = -pr.log_prob(rewards)
    
    # Compute the distribution over the values
    value_loss = -pv.log_prob(lambda_values.detach())
    value_loss = value_loss - pv.log_prob(predicted_target_values.detach())
    value_loss = value_loss * discount.squeeze(-1)

    # TODO
    action_loss = -pa.log_prob(actions)
    
    # KL balancing
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)
    reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + value_loss + action_loss + continue_loss).mean()
    return (
        reconstruction_loss,
        kl.mean(),
        kl_loss.mean(),
        reward_loss.mean(),
        observation_loss.mean(),
        continue_loss.mean(),
    )
