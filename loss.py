from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F


def compute_cb_weights(n_sample_per_class: Sequence[int], beta: float = 0):
    """The class balanced weighting factor"""
    if not isinstance(n_sample_per_class, torch.Tensor):
        n_sample_per_class = torch.as_tensor(n_sample_per_class)
    
    return (1 - beta) / (1 - torch.pow(beta, n_sample_per_class))


def cross_entropy(input: torch.Tensor, 
                  target: torch.Tensor,
                  weight: Optional[torch.Tensor] = None,
                  from_logits: bool = True, 
                  reduction: Literal["none", "sum", "mean"] = "mean"):
    N, C = input.shape
    if len(target.shape) == 1:
        target = F.one_hot(target, num_classes=C)  # (N,) -> (N, C)

    if weight is None:
        weight = torch.ones(C, dtype=input.dtype, device=input.device)
    
    log_probs = F.log_softmax(input, dim=-1) if from_logits else torch.log(input + 1e-8)    # (N, d1,...,dk, C)
    ce = -torch.sum(weight * target * log_probs, dim=-1)  # (N, d1,...,dk)
    
    if len(ce.shape) > 1:
        K = len(ce.shape[1:])
        ce = torch.sum(ce, dim=tuple(range(1, K+1))) # (N,)
    
    if reduction == "mean":
        ce = torch.mean(ce)
    elif reduction == "sum":
        ce = torch.sum(ce)

    return ce


def dist_loss(pi: torch.Tensor):
    # pi: (N, K)
    epsilon = 1e-8
    pi_avg = pi.mean(dim=0)
    pi_avg = pi_avg.where(pi_avg < epsilon, epsilon)

    return torch.sum(pi_avg * torch.log(pi_avg))
