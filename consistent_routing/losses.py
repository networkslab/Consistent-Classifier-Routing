"""Hinge surrogate losses for jointly training classifiers and a router.

Equation labels refer to the paper:

* ``eq:stage_wise_surrogate``: cost-sensitive multiclass hinge loss of one
  classifier, built on the hinge loss of Tarigan and van de Geer (2008).
* ``eq:surrogate_hinge_flat``: surrogate for the single-stage (flat) router.
* ``eq:surrogate_hinge_casc``: surrogate for the cascade router.

Classifiers output real-valued scores over ``K`` classes that sum to zero. In
practice a network outputs ``K - 1`` scores and :func:`complete_scores` appends
the last one.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CostLike = Union[float, Sequence[float], Tensor]


def complete_scores(partial: Tensor) -> Tensor:
    """Append the score ``-sum(partial)`` so that the ``K`` scores sum to zero.

    Args:
        partial: tensor of shape ``(..., K - 1)``.

    Returns:
        Tensor of shape ``(..., K)`` whose last dimension sums to zero.
    """
    return torch.cat([partial, -partial.sum(dim=-1, keepdim=True)], dim=-1)


def stage_hinge(scores: Tensor, y: Tensor, cost: CostLike) -> Tensor:
    """Cost-sensitive hinge loss of eq:stage_wise_surrogate, per sample.

    ``sum_{k != y} [s_k + 1/(K-1)]_+ + K/(K-1) * cost``

    Args:
        scores: sum-to-zero scores of shape ``(N, K)`` for one classifier, or
            ``(N, T, K)`` for ``T`` classifiers at once.
        y: labels of shape ``(N,)`` with values in ``{0, ..., K-1}``.
        cost: the cost of the classifier. A scalar for ``(N, K)`` scores, or a
            length-``T`` vector for ``(N, T, K)`` scores.

    Returns:
        Tensor of shape ``(N,)`` or ``(N, T)``.
    """
    num_classes = scores.shape[-1]
    true_class = F.one_hot(y, num_classes).to(torch.bool)
    # Broadcast the label mask over any classifier dimension between N and K.
    true_class = true_class.view(true_class.shape[0], *([1] * (scores.dim() - 2)), num_classes)
    margins = torch.relu(scores + 1.0 / (num_classes - 1)).masked_fill(true_class, 0.0)
    cost = torch.as_tensor(cost, dtype=scores.dtype, device=scores.device)
    return margins.sum(dim=-1) + num_classes / (num_classes - 1) * cost


def cumulative_costs(kappa: CostLike) -> Tensor:
    """Cascade cost vector ``c_t = kappa_1 + ... + kappa_t``.

    A prompt that exits at stage ``t`` of a cascade pays for every classifier it
    passed through. The flat router uses ``c = kappa`` directly.
    """
    return torch.cumsum(torch.as_tensor(kappa, dtype=torch.float32), dim=0)


def exit_probabilities(defer: Tensor) -> Tensor:
    """Probability of exiting at each stage of a cascade.

    ``r_t = (prod_{tau < t} d_tau) * (1 - d_t)`` with ``d_T = 0``, so the last
    stage always exits and the probabilities sum to one.

    Args:
        defer: soft deferral probabilities of shape ``(N, T - 1)``.

    Returns:
        Tensor of shape ``(N, T)``.
    """
    ones = defer.new_ones(defer.shape[0], 1)
    reach = torch.cumprod(torch.cat([ones, defer], dim=1), dim=1)
    stop = torch.cat([1.0 - defer, ones], dim=1)
    return reach * stop


class FlatHingeLoss(nn.Module):
    """Flat-router surrogate of eq:surrogate_hinge_flat.

    ``mean_i sum_t r_t(x_i) * stage_hinge(f_t(x_i, z_{t,i}), y_i; c_t)`` with
    ``c = kappa``.
    """

    def __init__(self, kappa: CostLike):
        super().__init__()
        self.register_buffer("costs", torch.as_tensor(kappa, dtype=torch.float32))

    def forward(self, scores: Tensor, route_probs: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            scores: ``(N, T, K)`` sum-to-zero scores of the ``T`` classifiers.
            route_probs: ``(N, T)`` soft router outputs on the simplex.
            y: ``(N,)`` labels.
        """
        per_stage = stage_hinge(scores, y, self.costs)
        return (route_probs * per_stage).sum(dim=1).mean()


class CascadeHingeLoss(nn.Module):
    """Cascade surrogate of eq:surrogate_hinge_casc.

    ``mean_i sum_t r_t * stage_hinge(f_t(x_i, z_{t,i}), y_i; c_t)``, where
    ``r_t`` comes from :func:`exit_probabilities` and ``c`` is the cumulative
    cost vector of :func:`cumulative_costs`.
    """

    def __init__(self, kappa: CostLike):
        super().__init__()
        self.register_buffer("costs", cumulative_costs(kappa))

    def forward(self, scores: Tensor, defer_probs: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            scores: ``(N, T, K)`` sum-to-zero scores of the ``T`` classifiers.
            defer_probs: ``(N, T - 1)`` soft deferral probabilities.
            y: ``(N,)`` labels.
        """
        per_stage = stage_hinge(scores, y, self.costs)
        return (exit_probabilities(defer_probs) * per_stage).sum(dim=1).mean()
