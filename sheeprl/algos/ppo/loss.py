import torch
import torch.nn.functional as F
from torch import Tensor


def policy_loss(
    new_logprobs: Tensor,
    logprobs: Tensor,
    advantages: Tensor,
    clip_coef: float,
    reduction: str = "mean",
) -> Tensor:
    """Compute the policy loss for a batch of data, as described in equation (7) of the paper.

        - Compute the difference between the new and old logprobs.
        - Exponentiate it to find the ratio.
        - Use the ratio and advantages to compute the loss as per equation (7).

    Args:
        new_logprobs (Tensor): the log-probs of the new actions.
        logprobs (Tensor): the log-probs of the sampled actions from the environment.
        advantages (Tensor): the advantages.
        clip_coef (float): the clipping coefficient.

    Returns:
        the policy loss
    """
    # new_logprobs: 새로운 정책에서의 행동 로그 확률
    # logprobs: 이전 정책에서의 행동 로그 확률
    
    logratio = new_logprobs - logprobs
    ratio = logratio.exp() # 새 정책과 이전 정책의 확률 비율을 계산

    pg_loss1 = advantages * ratio #critic이 계산해준 advantage에 ratio를 곱한다...?
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)  #clipping된 값을 가지고
    pg_loss = -torch.min(pg_loss1, pg_loss2) #거기서 clipping된 손실과 클립핑되지 않는 손실중에서 작은 값을 선택하니깐 정책 변화가 작은 경우에는 그냥 pg_loss1을 쓰고, 큰 경우에는 클립핑된 값을 쓴다..!
    reduction = reduction.lower()
    if reduction == "none":
        return pg_loss
    elif reduction == "mean":
        return pg_loss.mean()
    elif reduction == "sum":
        return pg_loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")


def value_loss(
    new_values: Tensor, 
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    reduction: str = "mean",
) -> Tensor:
    #clip_vloss -> value 계산할때 clipping걸꺼냐?
    if not clip_vloss:
        values_pred = new_values #현재 value function이 예측한 새로운 가치
        return F.mse_loss(values_pred, returns, reduction=reduction) #이걸 실제 관측된 리턴값이랑 mse_loss를 가지고 계산
    else:
        v_loss_unclipped = (new_values - returns) ** 2
        v_clipped = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
        return v_loss


def entropy_loss(entropy: Tensor, reduction: str = "mean") -> Tensor:
    ent_loss = -entropy
    reduction = reduction.lower()
    if reduction == "none":
        return ent_loss
    elif reduction == "mean":
        return ent_loss.mean()
    elif reduction == "sum":
        return ent_loss.sum()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")
